"""Rights-bounded Floodgate CSA ingestion for deep reanalysis.

The importer deliberately creates *candidate-only* replay records.  An original
move from a public game is useful trajectory evidence, but is neither a policy
target nor an external-teacher verdict.  Every sample therefore has an empty
actor policy and no ``teacher_*`` fields; the adjacent provenance sidecar and
corpus manifest require history-aware MultiPV reanalysis before training.

The game archive is read through bounded, private batch extraction.  Every
listed path and extracted file is checked before CSA bytes are yielded, and
each temporary batch is removed immediately after consumption.  No archive
bytes, CSA text, or derived corpus are redistributable through this module: the
current rights classification is local analysis only until the archive owner
publishes terms that permit more.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Final
from urllib.parse import unquote, urlparse

from rsshogi.core import Board

from .adjudication import terminal_repetition_adjudication
from .artifact_provenance import canonical_json_sha256, identify_file
from .domain import GameRecord, PositionSample, Termination
from .ensemble import normalized_sfen
from .replay import append_games
from .research_context import game_identifier

CORPUS_SCHEMA: Final = "meteo-floodgate-local-corpus-v1"
CANDIDATE_SIDECAR_SCHEMA: Final = "meteo-floodgate-candidate-replay-v1"
RATING_SCHEMA: Final = "floodgate-rating-snapshot-v1"
RIGHTS_CLASSIFICATION: Final = "local-analysis-only"
ORIGINAL_CANDIDATE_SOURCE: Final = "floodgate-original-candidate"
UNKNOWN_IDENTITY_COMPONENT: Final = "unknown"

__all__ = [
    "CANDIDATE_SIDECAR_SCHEMA",
    "CORPUS_SCHEMA",
    "ORIGINAL_CANDIDATE_SOURCE",
    "RIGHTS_CLASSIFICATION",
    "CsaSource",
    "EngineIdentity",
    "EngineIdentityRule",
    "FloodgateCorpusConfig",
    "ParsedCsaGame",
    "RatingPlayer",
    "RatingSnapshot",
    "create_floodgate_corpus",
    "list_7z_csa_members",
    "parse_csa_game",
    "parse_rating_snapshot",
    "read_7z_csa_member",
    "resolve_engine_identities",
    "stream_7z_csa_sources",
]

_RATING_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_COMPACT_RATING_DATE = re.compile(r"(20\d{6})(?=\.html$)")
_MOVE_RECORD = re.compile(r"^[+-]\d{4}[A-Z]{2}$")
_TIME_RECORD = re.compile(r"^T(\d+)$")
_VERSION_RECORD = re.compile(r"^V2(?:\.\d+)?$")
_PLAYER_RECORD = re.compile(r"^N([+-])(.*)$")
_ROW_RECORD = re.compile(r"^P([1-9])(.*)$")
_DECLARED_RATE = re.compile(
    r"^'(black|white)_rate:(.*):([-+]?(?:\d+(?:\.\d*)?|\.\d+))$"
)

_CSA_TO_SFEN: Final[dict[str, str]] = {
    "FU": "P",
    "KY": "L",
    "KE": "N",
    "GI": "S",
    "KI": "G",
    "KA": "B",
    "HI": "R",
    "OU": "K",
    "TO": "+P",
    "NY": "+L",
    "NK": "+N",
    "NG": "+S",
    "UM": "+B",
    "RY": "+R",
}
_HAND_ORDER: Final[tuple[str, ...]] = ("HI", "KA", "KI", "GI", "KE", "KY", "FU")
_DRAW_TERMINALS: Final[frozenset[str]] = frozenset(
    {"%SENNICHITE", "%JISHOGI", "%HIKIWAKE", "%CHUDAN", "%FUZUMI", "%ERROR"}
)
_KNOWN_TERMINALS: Final[frozenset[str]] = frozenset(
    {
        "%TORYO",
        "%CHUDAN",
        "%SENNICHITE",
        "%TIME_UP",
        "%ILLEGAL_MOVE",
        "%+ILLEGAL_ACTION",
        "%-ILLEGAL_ACTION",
        "%JISHOGI",
        "%KACHI",
        "%HIKIWAKE",
        "%MATTA",
        "%TSUMI",
        "%FUZUMI",
        "%ERROR",
        "%MAX_MOVES",
    }
)


@dataclass(frozen=True, slots=True)
class RatingPlayer:
    """One exact player row; no engine-family inference is performed."""

    player_id: str
    display_name: str
    rating: int
    wins: int
    losses: int
    listed_win_rate: float
    last_play: str
    profile_href: str
    zero_loss_uncertain: bool

    @property
    def games(self) -> int:
        return self.wins + self.losses

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["games"] = self.games
        return result


@dataclass(frozen=True, slots=True)
class RatingSnapshot:
    """A byte-identified Floodgate rating table."""

    source_url: str
    source_sha256: str
    source_bytes: int
    snapshot_date: str
    scope: str
    players: tuple[RatingPlayer, ...]
    schema: str = RATING_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_url": self.source_url,
            "source_sha256": self.source_sha256,
            "source_bytes": self.source_bytes,
            "snapshot_date": self.snapshot_date,
            "scope": self.scope,
            "players": [player.to_dict() for player in self.players],
        }


class _RatingTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_rating_table = False
        self.in_row = False
        self.current_class: str | None = None
        self.current_text: list[str] = []
        self.current_href: str | None = None
        self.row: dict[str, str] = {}
        self.rows: list[dict[str, str]] = []
        self.headings: list[str] = []
        self.in_h1 = False
        self.h1_text: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag == "table" and "player-rating" in (attributes.get("class") or "").split():
            if self.in_rating_table:
                raise ValueError("nested Floodgate rating table")
            self.in_rating_table = True
        elif tag == "h1":
            self.in_h1 = True
            self.h1_text = []
        elif self.in_rating_table and tag == "tr":
            if self.in_row:
                raise ValueError("nested Floodgate rating row")
            self.in_row = True
            self.row = {}
        elif self.in_row and tag == "td":
            self.current_class = attributes.get("class")
            self.current_text = []
            self.current_href = None
        elif self.in_row and tag == "a" and self.current_class is not None:
            self.current_href = attributes.get("href")

    def handle_data(self, data: str) -> None:
        if self.in_h1:
            self.h1_text.append(data)
        if self.in_row and self.current_class is not None:
            self.current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1" and self.in_h1:
            self.in_h1 = False
            self.headings.append("".join(self.h1_text).strip())
        elif tag == "td" and self.in_row and self.current_class is not None:
            self._finish_cell()
        elif tag == "tr" and self.in_row:
            # The live page omits the final </td>; close that cell at </tr>.
            if self.current_class is not None:
                self._finish_cell()
            self.in_row = False
            if "name" in self.row:
                self.rows.append(dict(self.row))
        elif tag == "table" and self.in_rating_table:
            self.in_rating_table = False

    def _finish_cell(self) -> None:
        if self.current_class is None:
            return
        value = " ".join("".join(self.current_text).split())
        if self.current_class in self.row:
            raise ValueError(f"duplicate rating column {self.current_class!r}")
        self.row[self.current_class] = value
        if self.current_class == "name":
            if not self.current_href:
                raise ValueError("rating player row has no profile href")
            self.row["profile_href"] = self.current_href
        self.current_class = None
        self.current_text = []
        self.current_href = None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_rating_snapshot(
    raw_html: bytes, *, source_url: str, scope: str
) -> RatingSnapshot:
    """Parse and identify one official rating HTML snapshot, failing closed."""

    if not raw_html:
        raise ValueError("rating HTML must not be empty")
    if not source_url.startswith(("https://", "http://")):
        raise ValueError("rating source URL must be absolute HTTP(S)")
    if not scope.strip() or scope != scope.strip():
        raise ValueError("rating scope must be a non-empty trimmed string")
    try:
        decoded = raw_html.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("rating HTML must be UTF-8") from error
    parser = _RatingTableParser()
    parser.feed(decoded)
    parser.close()
    if parser.in_rating_table or parser.in_row:
        raise ValueError("unterminated rating table")
    if len(parser.headings) != 1:
        raise ValueError("rating page must contain exactly one h1 heading")
    date_match = _RATING_DATE.search(parser.headings[0])
    if date_match is None:
        raise ValueError("rating heading has no snapshot date")
    snapshot_date = date_match.group(1)
    datetime.strptime(snapshot_date, "%Y-%m-%d")
    source_filename = Path(urlparse(source_url).path).name
    url_date = _RATING_DATE.search(source_filename)
    compact_url_date = _COMPACT_RATING_DATE.search(source_filename)
    parsed_url_date = (
        url_date.group(1)
        if url_date is not None
        else (
            datetime.strptime(compact_url_date.group(1), "%Y%m%d").strftime("%Y-%m-%d")
            if compact_url_date is not None
            else None
        )
    )
    if parsed_url_date is not None and parsed_url_date != snapshot_date:
        raise ValueError("rating URL date does not match page heading")
    if not parser.rows:
        raise ValueError("rating table contains no player rows")

    players: list[RatingPlayer] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(parser.rows):
        required = {
            "name",
            "rate",
            "wins",
            "losses",
            "win_rate",
            "last_modified",
            "profile_href",
        }
        if set(row) != required:
            raise ValueError(
                f"rating row {index} has columns {sorted(row)}, expected {sorted(required)}"
            )
        profile_href = row["profile_href"]
        player_file = PurePosixPath(urlparse(profile_href).path).name
        if not player_file.endswith(".html"):
            raise ValueError(f"rating row {index} has invalid profile href")
        player_id = unquote(player_file[:-5])
        if not player_id or player_id in seen_ids:
            raise ValueError(f"duplicate or empty rating player id {player_id!r}")
        seen_ids.add(player_id)
        try:
            rating = int(row["rate"])
            wins = int(row["wins"])
            losses = int(row["losses"])
            listed_win_rate = float(row["win_rate"])
        except ValueError as error:
            raise ValueError(f"rating row {index} has invalid numeric fields") from error
        if wins < 0 or losses < 0 or not 0 <= listed_win_rate <= 1:
            raise ValueError(f"rating row {index} has invalid game counts or win rate")
        games = wins + losses
        if games and not math.isclose(listed_win_rate, wins / games, abs_tol=0.0015):
            raise ValueError(f"rating row {index} has inconsistent win rate")
        try:
            datetime.strptime(row["last_modified"], "%Y-%m-%d %H:%M:%S")
        except ValueError as error:
            raise ValueError(f"rating row {index} has invalid last-play timestamp") from error
        display_name = html.unescape(row["name"])
        if not display_name or display_name != display_name.strip():
            raise ValueError(f"rating row {index} has invalid display name")
        players.append(
            RatingPlayer(
                player_id=player_id,
                display_name=display_name,
                rating=rating,
                wins=wins,
                losses=losses,
                listed_win_rate=listed_win_rate,
                last_play=row["last_modified"],
                profile_href=profile_href,
                zero_loss_uncertain=losses == 0,
            )
        )
    return RatingSnapshot(
        source_url=source_url,
        source_sha256=_sha256_bytes(raw_html),
        source_bytes=len(raw_html),
        snapshot_date=snapshot_date,
        scope=scope,
        players=tuple(players),
    )


@dataclass(frozen=True, slots=True)
class EngineIdentityRule:
    """Explicit identity metadata for exact Floodgate player names."""

    rule_id: str
    player_names: tuple[str, ...]
    family: str
    version: str
    hardware: str

    def __post_init__(self) -> None:
        values = (self.rule_id, self.family, self.version, self.hardware)
        if any(not value.strip() or value != value.strip() for value in values):
            raise ValueError("identity rule fields must be non-empty trimmed strings")
        if not self.player_names or len(set(self.player_names)) != len(self.player_names):
            raise ValueError("identity rule player names must be non-empty and unique")
        if any(not name.strip() or name != name.strip() for name in self.player_names):
            raise ValueError("identity rule player names must be non-empty and trimmed")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EngineIdentity:
    player_name: str
    family: str
    version: str
    hardware: str
    rule_id: str | None
    inferred: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_engine_identities(
    player_names: Iterable[str], rules: Sequence[EngineIdentityRule]
) -> dict[str, EngineIdentity]:
    """Apply exact declared rules; unknown names remain explicitly unknown."""

    by_name: dict[str, EngineIdentityRule] = {}
    rule_ids: set[str] = set()
    for rule in rules:
        if rule.rule_id in rule_ids:
            raise ValueError(f"duplicate engine identity rule id {rule.rule_id!r}")
        rule_ids.add(rule.rule_id)
        for name in rule.player_names:
            if name in by_name:
                raise ValueError(f"player name {name!r} is shadowed by multiple identity rules")
            by_name[name] = rule
    resolved: dict[str, EngineIdentity] = {}
    for name in player_names:
        if name in resolved:
            continue
        matched_rule = by_name.get(name)
        resolved[name] = EngineIdentity(
            player_name=name,
            family=(
                matched_rule.family
                if matched_rule is not None
                else UNKNOWN_IDENTITY_COMPONENT
            ),
            version=(
                matched_rule.version
                if matched_rule is not None
                else UNKNOWN_IDENTITY_COMPONENT
            ),
            hardware=(
                matched_rule.hardware
                if matched_rule is not None
                else UNKNOWN_IDENTITY_COMPONENT
            ),
            rule_id=(matched_rule.rule_id if matched_rule is not None else None),
        )
    return resolved


@dataclass(frozen=True, slots=True)
class CsaSource:
    archive_member: str
    raw_bytes: bytes

    def __post_init__(self) -> None:
        _validate_archive_member(self.archive_member)
        if not self.raw_bytes:
            raise ValueError("CSA source bytes must not be empty")


@dataclass(frozen=True, slots=True)
class ParsedCsaGame:
    record: GameRecord
    archive_member: str
    csa_sha256: str
    csa_bytes: int
    text_encoding: str
    csa_version: str
    black_name: str
    white_name: str
    event: str | None
    start_time: str | None
    end_time: str | None
    move_times_seconds: tuple[int | None, ...]
    terminal_code: str
    terminal_time_seconds: int | None
    terminal_record_count: int
    terminal_validation: str
    initial_position_mode: str
    declared_black_rating: float | None
    declared_white_rating: float | None

    def to_metadata(self) -> dict[str, object]:
        return {
            "game_id": game_identifier(self.record),
            "archive_member": self.archive_member,
            "csa_sha256": self.csa_sha256,
            "csa_bytes": self.csa_bytes,
            "text_encoding": self.text_encoding,
            "csa_version": self.csa_version,
            "black_name": self.black_name,
            "white_name": self.white_name,
            "event": self.event,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "moves": len(self.record.moves),
            "move_times_seconds": list(self.move_times_seconds),
            "terminal_code": self.terminal_code,
            "terminal_time_seconds": self.terminal_time_seconds,
            "terminal_record_count": self.terminal_record_count,
            "terminal_validation": self.terminal_validation,
            "termination": self.record.termination.value,
            "winner": self.record.winner,
            "initial_position_mode": self.initial_position_mode,
            "declared_black_rating": self.declared_black_rating,
            "declared_white_rating": self.declared_white_rating,
        }


def _validate_archive_path(member: str) -> None:
    if (
        not member
        or any(character in member for character in ("\x00", "\r", "\n", "\\"))
        or member.startswith("/")
    ):
        raise ValueError("archive member must be a non-empty POSIX path")
    parts = member.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe archive member path: {member!r}")


def _validate_archive_member(member: str) -> None:
    _validate_archive_path(member)
    pure = PurePosixPath(member)
    if pure.suffix.casefold() != ".csa":
        raise ValueError(f"archive member is not CSA: {member!r}")


@dataclass(frozen=True, slots=True)
class _SevenZipMember:
    path: str
    size: int
    is_directory: bool


def _normalized_archive_parts(member: str) -> tuple[str, ...]:
    return tuple(
        unicodedata.normalize("NFC", part).casefold() for part in member.split("/")
    )


def _seven_zip_executable(seven_zip: str) -> str:
    executable = shutil.which(seven_zip)
    if executable is None:
        raise FileNotFoundError(f"7-Zip executable not found: {seven_zip}")
    return executable


def _is_link_entry(attributes: Mapping[str, str]) -> bool:
    for key, value in attributes.items():
        if "link" in key.casefold() and value.strip() not in {"", "-"}:
            return True
    unix_mode = attributes.get("Attributes", "").rsplit(" ", 1)[-1]
    return bool(re.fullmatch(r"l[rwxSsTt-]{9}", unix_mode))


def _is_directory_entry(attributes: Mapping[str, str]) -> bool:
    raw_attributes = attributes.get("Attributes", "")
    unix_mode = raw_attributes.rsplit(" ", 1)[-1]
    return raw_attributes.startswith("D") or bool(
        re.fullmatch(r"d[rwxSsTt-]{9}", unix_mode)
    )


def _list_7z_members(
    archive_path: str,
    *,
    executable: str,
    timeout_seconds: float,
) -> tuple[_SevenZipMember, ...]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    completed = subprocess.run(
        [executable, "l", "-slt", archive_path],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"7-Zip listing failed: {completed.stderr.strip()}")
    marker = "----------\n"
    if marker not in completed.stdout:
        raise ValueError("7-Zip listing has no member section")
    member_text = completed.stdout.split(marker, 1)[1]
    entries: list[_SevenZipMember] = []
    exact_paths: set[str] = set()
    normalized_paths: dict[tuple[str, ...], str] = {}
    for block in re.split(r"\n\s*\n", member_text):
        attributes: dict[str, str] = {}
        for line in block.splitlines():
            if " = " in line:
                key, value = line.split(" = ", 1)
                attributes[key] = value
        member = attributes.get("Path")
        if member is None:
            continue
        _validate_archive_path(member)
        if attributes.get("Anti", "-").strip() not in {"", "-"}:
            raise ValueError(f"7-Zip archive contains an anti-item: {member!r}")
        if _is_link_entry(attributes):
            raise ValueError(f"7-Zip archive contains a link entry: {member!r}")
        if member in exact_paths:
            raise ValueError("7-Zip archive contains duplicate member paths")
        exact_paths.add(member)
        normalized = _normalized_archive_parts(member)
        collision = normalized_paths.get(normalized)
        if collision is not None:
            raise ValueError(
                "7-Zip archive contains filesystem-colliding member paths: "
                f"{collision!r} and {member!r}"
            )
        normalized_paths[normalized] = member
        is_directory = _is_directory_entry(attributes)
        raw_size = attributes.get("Size")
        if raw_size is None or not raw_size.isdecimal():
            raise ValueError(f"7-Zip member has no valid size: {member!r}")
        size = int(raw_size)
        if is_directory and size != 0:
            raise ValueError(f"7-Zip directory has non-zero size: {member!r}")
        entries.append(
            _SevenZipMember(path=member, size=size, is_directory=is_directory)
        )
    if not entries:
        raise ValueError("7-Zip archive contains no members")
    by_normalized_path = {
        _normalized_archive_parts(entry.path): entry for entry in entries
    }
    for entry in entries:
        parts = _normalized_archive_parts(entry.path)
        for ancestor_length in range(1, len(parts)):
            ancestor = by_normalized_path.get(parts[:ancestor_length])
            if ancestor is not None and not ancestor.is_directory:
                raise ValueError(
                    "7-Zip archive contains a file/directory shadow: "
                    f"{ancestor.path!r} shadows {entry.path!r}"
                )
    return tuple(entries)


def list_7z_csa_members(archive: Path, *, seven_zip: str = "7z") -> tuple[str, ...]:
    """List CSA members without extracting the archive."""

    identity = identify_file(archive)
    entries = _list_7z_members(
        identity.path,
        executable=_seven_zip_executable(seven_zip),
        timeout_seconds=120,
    )
    members = [
        entry.path
        for entry in entries
        if not entry.is_directory and PurePosixPath(entry.path).suffix.casefold() == ".csa"
    ]
    if not members:
        raise ValueError("7-Zip archive contains no CSA members")
    return tuple(sorted(members))


def _batched_members(
    entries: Sequence[_SevenZipMember],
    *,
    maximum_batch_bytes: int,
    maximum_batch_members: int,
) -> Iterator[tuple[_SevenZipMember, ...]]:
    batch: list[_SevenZipMember] = []
    batch_bytes = 0
    for entry in entries:
        if batch and (
            len(batch) >= maximum_batch_members
            or batch_bytes + entry.size > maximum_batch_bytes
        ):
            yield tuple(batch)
            batch = []
            batch_bytes = 0
        batch.append(entry)
        batch_bytes += entry.size
    if batch:
        yield tuple(batch)


def _audit_extracted_batch(
    directory: Path,
    entries: Sequence[_SevenZipMember],
    *,
    maximum_bytes: int,
    maximum_batch_bytes: int,
) -> dict[str, Path]:
    expected = {entry.path: entry for entry in entries}
    expected_directories: set[str] = set()
    for entry in entries:
        parts = entry.path.split("/")
        expected_directories.update(
            "/".join(parts[:length]) for length in range(1, len(parts))
        )
    extracted: dict[str, Path] = {}
    total_bytes = 0
    for root, directories, files in os.walk(directory, topdown=True, followlinks=False):
        root_path = Path(root)
        for name in directories:
            path = root_path / name
            relative = path.relative_to(directory).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ValueError(f"7-Zip extraction created an unsafe directory: {relative!r}")
            if relative not in expected_directories:
                raise ValueError(f"7-Zip extraction created an unexpected directory: {relative!r}")
        for name in files:
            path = root_path / name
            relative = path.relative_to(directory).as_posix()
            extracted_entry = expected.get(relative)
            if extracted_entry is None:
                raise ValueError(f"7-Zip extraction created an unexpected file: {relative!r}")
            file_stat = path.lstat()
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                raise ValueError(f"7-Zip extraction created a non-regular file: {relative!r}")
            if file_stat.st_size != extracted_entry.size:
                raise ValueError(
                    f"7-Zip extracted size mismatch for {relative!r}: "
                    f"listed={extracted_entry.size}, actual={file_stat.st_size}"
                )
            if file_stat.st_size > maximum_bytes:
                raise ValueError(f"CSA member exceeds maximum_bytes={maximum_bytes}")
            total_bytes += file_stat.st_size
            if total_bytes > maximum_batch_bytes:
                raise ValueError(
                    "7-Zip extraction exceeded the configured batch byte limit"
                )
            extracted[relative] = path
    missing = sorted(set(expected) - set(extracted))
    if missing:
        raise ValueError(f"7-Zip extraction omitted requested members: {missing[:3]}")
    return extracted


@contextmanager
def stream_7z_csa_sources(
    archive: Path,
    members: Sequence[str],
    *,
    seven_zip: str = "7z",
    maximum_bytes: int = 2_000_000,
    maximum_batch_bytes: int = 768 * 1024 * 1024,
    maximum_batch_members: int = 50_000,
    timeout_seconds: float = 600,
) -> Iterator[Iterator[CsaSource]]:
    """Yield validated CSA sources from bounded batch extractions.

    Each batch is extracted into a private temporary directory, audited, and
    removed before the next batch.  The iterator holds only the currently
    yielded CSA bytes in memory.  The archive is byte-identified before and
    after a complete iteration so provenance cannot silently drift mid-read.
    """

    if maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive")
    if maximum_batch_bytes < maximum_bytes:
        raise ValueError("maximum_batch_bytes must be at least maximum_bytes")
    if maximum_batch_members < 1:
        raise ValueError("maximum_batch_members must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    requested = tuple(members)
    if not requested:
        raise ValueError("at least one CSA archive member is required")
    for member in requested:
        _validate_archive_member(member)
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate requested CSA archive member")
    normalized_requested = [_normalized_archive_parts(member) for member in requested]
    if len(set(normalized_requested)) != len(normalized_requested):
        raise ValueError("requested CSA members collide on a normalized filesystem")

    executable = _seven_zip_executable(seven_zip)
    archive_before = identify_file(archive)
    entries = _list_7z_members(
        archive_before.path,
        executable=executable,
        timeout_seconds=min(timeout_seconds, 120),
    )
    csa_entries = {
        entry.path: entry
        for entry in entries
        if not entry.is_directory and PurePosixPath(entry.path).suffix.casefold() == ".csa"
    }
    missing = sorted(set(requested) - set(csa_entries))
    if missing:
        raise ValueError(f"requested CSA members are absent from the archive: {missing[:3]}")
    selected = tuple(csa_entries[member] for member in requested)
    oversized = [entry.path for entry in selected if entry.size > maximum_bytes]
    if oversized:
        raise ValueError(
            f"CSA member exceeds maximum_bytes={maximum_bytes}: {oversized[0]!r}"
        )

    temporary_root = Path(tempfile.mkdtemp(prefix="meteo-floodgate-extract-"))

    def source_iterator() -> Generator[CsaSource, None, None]:
        try:
            for batch_index, batch in enumerate(
                _batched_members(
                    selected,
                    maximum_batch_bytes=maximum_batch_bytes,
                    maximum_batch_members=maximum_batch_members,
                )
            ):
                batch_directory = temporary_root / f"batch-{batch_index:06d}"
                batch_directory.mkdir()
                list_file = temporary_root / f"batch-{batch_index:06d}.txt"
                with list_file.open("x", encoding="utf-8", newline="\n") as stream:
                    for entry in batch:
                        stream.write(f"{entry.path}\n")
                completed = subprocess.run(
                    [
                        executable,
                        "x",
                        "-y",
                        "-bd",
                        "-bb0",
                        "-spd",
                        "-ssc",
                        f"-o{batch_directory}",
                        archive_before.path,
                        f"@{list_file}",
                    ],
                    check=False,
                    capture_output=True,
                    timeout=timeout_seconds,
                )
                if completed.returncode != 0:
                    error = completed.stderr.decode("utf-8", errors="replace").strip()
                    if not error:
                        error = completed.stdout.decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"7-Zip batch extraction failed: {error}")
                extracted = _audit_extracted_batch(
                    batch_directory,
                    batch,
                    maximum_bytes=maximum_bytes,
                    maximum_batch_bytes=maximum_batch_bytes,
                )
                for entry in batch:
                    raw_bytes = extracted[entry.path].read_bytes()
                    if len(raw_bytes) != entry.size or len(raw_bytes) > maximum_bytes:
                        raise ValueError(
                            f"CSA member changed during read: {entry.path!r}"
                        )
                    yield CsaSource(archive_member=entry.path, raw_bytes=raw_bytes)
                shutil.rmtree(batch_directory)
                list_file.unlink()
            archive_after = identify_file(Path(archive_before.path))
            if archive_after != archive_before:
                raise ValueError("7-Zip archive changed during CSA streaming")
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

    iterator = source_iterator()
    try:
        yield iterator
    finally:
        iterator.close()
        shutil.rmtree(temporary_root, ignore_errors=True)


def read_7z_csa_member(
    archive: Path,
    member: str,
    *,
    seven_zip: str = "7z",
    maximum_bytes: int = 2_000_000,
) -> CsaSource:
    """Read one CSA member through the safe batch-streaming implementation."""

    with stream_7z_csa_sources(
        archive,
        (member,),
        seven_zip=seven_zip,
        maximum_bytes=maximum_bytes,
        maximum_batch_bytes=maximum_bytes,
        maximum_batch_members=1,
        timeout_seconds=120,
    ) as sources:
        source = next(sources)
        try:
            next(sources)
        except StopIteration:
            return source
        raise AssertionError("single-member CSA stream yielded multiple sources")


def _logical_records(text: str) -> list[tuple[int, str]]:
    records: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip("\r")
        if not line:
            continue
        if line.startswith("'"):
            records.append((line_number, line))
            continue
        comment_index = line.find("'")
        prefix = line if comment_index < 0 else line[:comment_index]
        comment = None if comment_index < 0 else line[comment_index:]
        for component in prefix.split(","):
            if component:
                records.append((line_number, component))
        if comment:
            records.append((line_number, comment))
    return records


def _parse_row_pieces(row_records: Mapping[int, str]) -> list[list[tuple[str, str] | None]]:
    if set(row_records) != set(range(1, 10)):
        missing = sorted(set(range(1, 10)) - set(row_records))
        raise ValueError(f"CSA explicit board is missing rows {missing}")
    board: list[list[tuple[str, str] | None]] = []
    for rank in range(1, 10):
        contents = row_records[rank]
        if len(contents) != 27:
            raise ValueError(f"CSA P{rank} row must contain exactly nine 3-byte cells")
        parsed_row: list[tuple[str, str] | None] = []
        for offset in range(0, 27, 3):
            cell = contents[offset : offset + 3].strip()
            if cell == "*":
                parsed_row.append(None)
                continue
            if len(cell) != 3 or cell[0] not in "+-" or cell[1:] not in _CSA_TO_SFEN:
                raise ValueError(f"invalid CSA board cell {cell!r} on row {rank}")
            parsed_row.append((cell[0], cell[1:]))
        board.append(parsed_row)
    return board


def _standard_board_cells() -> list[list[tuple[str, str] | None]]:
    rows: dict[int, str] = {}
    for line in Board().to_csa().splitlines():
        match = _ROW_RECORD.fullmatch(line)
        if match is not None:
            rows[int(match.group(1))] = match.group(2)
    return _parse_row_pieces(rows)


def _parse_piece_list(
    value: str,
    *,
    sign: str,
    cells: list[list[tuple[str, str] | None]],
    hands: dict[str, dict[str, int]],
) -> None:
    if len(value) % 4:
        raise ValueError(f"CSA P{sign} record must contain 4-byte piece entries")
    for offset in range(0, len(value), 4):
        item = value[offset : offset + 4]
        coordinate, piece = item[:2], item[2:]
        if piece not in _CSA_TO_SFEN:
            raise ValueError(f"invalid CSA piece code {piece!r}")
        if coordinate == "00":
            if piece in {"OU", "TO", "NY", "NK", "NG", "UM", "RY"}:
                raise ValueError(f"piece {piece} cannot be in hand")
            hands[sign][piece] = hands[sign].get(piece, 0) + 1
            continue
        if len(coordinate) != 2 or not all("1" <= digit <= "9" for digit in coordinate):
            raise ValueError(f"invalid CSA placement coordinate {coordinate!r}")
        file_number, rank_number = int(coordinate[0]), int(coordinate[1])
        cell_index = 9 - file_number
        if cells[rank_number - 1][cell_index] is not None:
            raise ValueError(f"duplicate CSA placement at {coordinate}")
        cells[rank_number - 1][cell_index] = (sign, piece)


def _cells_to_sfen(
    cells: list[list[tuple[str, str] | None]],
    hands: Mapping[str, Mapping[str, int]],
    turn_sign: str,
) -> str:
    board_rows: list[str] = []
    for row in cells:
        parts: list[str] = []
        empty = 0
        for cell in row:
            if cell is None:
                empty += 1
                continue
            if empty:
                parts.append(str(empty))
                empty = 0
            sign, code = cell
            symbol = _CSA_TO_SFEN[code]
            if symbol.startswith("+"):
                base = symbol[1:]
                parts.append("+" + (base if sign == "+" else base.lower()))
            else:
                parts.append(symbol if sign == "+" else symbol.lower())
        if empty:
            parts.append(str(empty))
        board_rows.append("".join(parts))
    hand_parts: list[str] = []
    for sign in ("+", "-"):
        for code in _HAND_ORDER:
            count = hands[sign].get(code, 0)
            if count:
                if count > 1:
                    hand_parts.append(str(count))
                symbol = _CSA_TO_SFEN[code]
                hand_parts.append(symbol if sign == "+" else symbol.lower())
    hand_sfen = "".join(hand_parts) or "-"
    turn_sfen = "b" if turn_sign == "+" else "w"
    sfen = f"{'/'.join(board_rows)} {turn_sfen} {hand_sfen} 1"
    try:
        board = Board(sfen)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid CSA initial position: {sfen}") from error
    if not board.is_valid():
        raise ValueError(f"invalid CSA initial position: {sfen}")
    return board.to_sfen()


def _build_initial_sfen(
    *,
    pi_record: str | None,
    row_records: Mapping[int, str],
    plus_pieces: Sequence[str],
    minus_pieces: Sequence[str],
    turn_sign: str,
) -> tuple[str, str]:
    if pi_record is not None and row_records:
        raise ValueError("CSA position cannot combine PI with explicit P1-P9 rows")
    if pi_record is not None:
        cells = _standard_board_cells()
        removals = pi_record[2:]
        if len(removals) % 4:
            raise ValueError("CSA PI handicap suffix must contain 4-byte entries")
        for offset in range(0, len(removals), 4):
            item = removals[offset : offset + 4]
            coordinate, expected_piece = item[:2], item[2:]
            if not all("1" <= digit <= "9" for digit in coordinate):
                raise ValueError(f"invalid CSA PI removal coordinate {coordinate!r}")
            file_number, rank_number = int(coordinate[0]), int(coordinate[1])
            cell_index = 9 - file_number
            current = cells[rank_number - 1][cell_index]
            if current is None or current[1] != expected_piece:
                raise ValueError(f"CSA PI removal does not match piece at {coordinate}")
            cells[rank_number - 1][cell_index] = None
        mode = "PI" if not removals else "PI-handicap"
    elif row_records:
        cells = _parse_row_pieces(row_records)
        mode = "explicit-rows"
    else:
        cells = [[None for _file in range(9)] for _rank in range(9)]
        mode = "piece-list"
    hands: dict[str, dict[str, int]] = {"+": {}, "-": {}}
    for record in plus_pieces:
        _parse_piece_list(record, sign="+", cells=cells, hands=hands)
    for record in minus_pieces:
        _parse_piece_list(record, sign="-", cells=cells, hands=hands)
    return _cells_to_sfen(cells, hands, turn_sign), mode


def _termination_and_winner(
    code: str, board: Board
) -> tuple[Termination, int | None, str]:
    current = board.turn.value
    opponent = board.turn.opponent().value
    if code == "%TORYO":
        return Termination.RESIGNATION, opponent, "protocol_forfeit_no_board_proof_required"
    if code == "%SENNICHITE":
        repetition = terminal_repetition_adjudication(board)
        if repetition is None:
            raise ValueError("CSA %SENNICHITE position is not a legal repetition")
        return Termination.REPETITION, repetition.winner, "locally_verified_repetition"
    if code == "%KACHI":
        return (
            Termination.DECLARATION,
            current,
            "server_reported_declaration_not_locally_proven",
        )
    if code == "%TIME_UP":
        return Termination.TIME_FORFEIT, opponent, "protocol_forfeit_no_board_proof_required"
    if code in {"%ILLEGAL_MOVE", "%MATTA"}:
        return Termination.ILLEGAL_MOVE, opponent, "server_reported_illegal_action"
    if code == "%+ILLEGAL_ACTION":
        return Termination.ILLEGAL_MOVE, 1, "server_reported_illegal_action"
    if code == "%-ILLEGAL_ACTION":
        return Termination.ILLEGAL_MOVE, 0, "server_reported_illegal_action"
    if code == "%TSUMI":
        if not board.is_mated():
            raise ValueError("CSA %TSUMI position is not checkmate")
        return Termination.CHECKMATE, opponent, "locally_verified_checkmate"
    if code in {"%JISHOGI", "%MAX_MOVES"}:
        return Termination.IMPASSE, None, "server_reported_draw_not_locally_proven"
    if code == "%HIKIWAKE":
        return Termination.AGREED_DRAW, None, "server_reported_draw_not_locally_proven"
    if code == "%CHUDAN" or code == "%ERROR":
        return Termination.INTERRUPTION, None, "protocol_interruption"
    if code == "%FUZUMI":
        return Termination.NO_MATE, None, "server_reported_no_mate_not_locally_proven"
    raise ValueError(f"unsupported CSA terminal code {code!r}")


def parse_csa_game(source: CsaSource) -> ParsedCsaGame:
    """Legally replay one CSA V2 record and preserve its exact provenance."""

    try:
        text = source.raw_bytes.decode("utf-8", errors="strict")
        encoding = "utf-8"
    except UnicodeDecodeError:
        try:
            text = source.raw_bytes.decode("cp932", errors="strict")
            encoding = "cp932"
        except UnicodeDecodeError as error:
            raise ValueError("CSA text is neither strict UTF-8 nor strict CP932") from error
    records = _logical_records(text)
    if not records:
        raise ValueError("CSA record is empty")

    version: str | None = None
    players: dict[str, str] = {}
    headers: dict[str, str] = {}
    pi_record: str | None = None
    row_records: dict[int, str] = {}
    plus_pieces: list[str] = []
    minus_pieces: list[str] = []
    turn_sign: str | None = None
    board: Board | None = None
    initial_sfen: str | None = None
    initial_mode: str | None = None
    moves: list[str] = []
    samples: list[PositionSample] = []
    times: list[int | None] = []
    terminal: str | None = None
    terminal_time: int | None = None
    terminal_record_count = 0
    declared_ratings: dict[str, float] = {}

    for line_number, record in records:
        if terminal is not None:
            if record.startswith(("'", "$")):
                if record.startswith("$"):
                    key, separator, value = record[1:].partition(":")
                    if not separator or not key or key in headers:
                        raise ValueError(f"invalid or duplicate CSA header at line {line_number}")
                    headers[key] = value
                continue
            terminal_time_match = _TIME_RECORD.fullmatch(record)
            if terminal_time_match is not None:
                if terminal_time is not None:
                    raise ValueError(
                        f"duplicate CSA terminal time at line {line_number}"
                    )
                terminal_time = int(terminal_time_match.group(1))
                continue
            if record == terminal:
                terminal_record_count += 1
                continue
            raise ValueError(f"CSA data follows terminal record at line {line_number}")
        if _VERSION_RECORD.fullmatch(record):
            if version is not None or board is not None:
                raise ValueError(f"duplicate or late CSA version at line {line_number}")
            version = record
            continue
        player_match = _PLAYER_RECORD.fullmatch(record)
        if player_match is not None:
            side, name = player_match.groups()
            if board is not None or side in players or not name:
                raise ValueError(f"invalid or duplicate CSA player at line {line_number}")
            players[side] = name
            continue
        if record.startswith("$"):
            key, separator, value = record[1:].partition(":")
            if not separator or not key or key in headers:
                raise ValueError(f"invalid or duplicate CSA header at line {line_number}")
            headers[key] = value
            continue
        if record.startswith("'"):
            rate_match = _DECLARED_RATE.fullmatch(record)
            if rate_match is not None:
                color, _identity, rate = rate_match.groups()
                if color in declared_ratings:
                    raise ValueError(f"duplicate declared CSA rating at line {line_number}")
                declared_ratings[color] = float(rate)
            continue
        if record.startswith("PI"):
            if board is not None or pi_record is not None:
                raise ValueError(f"duplicate or late CSA PI at line {line_number}")
            pi_record = record
            continue
        row_match = _ROW_RECORD.fullmatch(record)
        if row_match is not None:
            rank = int(row_match.group(1))
            if board is not None or rank in row_records:
                raise ValueError(f"duplicate or late CSA P{rank} at line {line_number}")
            row_records[rank] = row_match.group(2)
            continue
        if record.startswith("P+") or record.startswith("P-"):
            if board is not None:
                raise ValueError(f"late CSA piece list at line {line_number}")
            target = plus_pieces if record[1] == "+" else minus_pieces
            target.append(record[2:])
            continue
        if record in {"+", "-"}:
            if board is not None or turn_sign is not None:
                raise ValueError(f"duplicate CSA turn record at line {line_number}")
            turn_sign = record
            initial_sfen, initial_mode = _build_initial_sfen(
                pi_record=pi_record,
                row_records=row_records,
                plus_pieces=plus_pieces,
                minus_pieces=minus_pieces,
                turn_sign=turn_sign,
            )
            board = Board(initial_sfen)
            continue
        if _MOVE_RECORD.fullmatch(record):
            if board is None or initial_sfen is None:
                raise ValueError(f"CSA move precedes initial position at line {line_number}")
            expected_sign = "+" if board.turn.value == 0 else "-"
            if record[0] != expected_sign:
                raise ValueError(f"CSA move has wrong side at line {line_number}")
            try:
                move = board.move_from_csa(record)
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid CSA move at line {line_number}: {record}") from error
            if not board.is_legal_move32(move):
                raise ValueError(f"illegal CSA move at line {line_number}: {record}")
            move_usi = move.to_usi()
            samples.append(
                PositionSample(
                    sfen=board.to_sfen(),
                    ply=len(moves),
                    turn=board.turn.value,
                    policy={},
                    root_value=0.0,
                    value_target=0.0,
                    chosen_move=move_usi,
                    actor_source=ORIGINAL_CANDIDATE_SOURCE,
                )
            )
            board.apply_move32(move)
            if not board.is_valid():
                raise ValueError(f"CSA move creates invalid board at line {line_number}")
            moves.append(move_usi)
            times.append(None)
            continue
        time_match = _TIME_RECORD.fullmatch(record)
        if time_match is not None:
            if not moves or times[-1] is not None:
                raise ValueError(f"orphan or duplicate CSA time at line {line_number}")
            times[-1] = int(time_match.group(1))
            continue
        if record.startswith("%"):
            if record not in _KNOWN_TERMINALS:
                raise ValueError(f"unknown CSA terminal code at line {line_number}: {record}")
            if board is None:
                raise ValueError(f"CSA terminal precedes initial position at line {line_number}")
            terminal = record
            terminal_record_count = 1
            continue
        raise ValueError(f"unsupported CSA record at line {line_number}: {record!r}")

    if version is None or not _VERSION_RECORD.fullmatch(version):
        raise ValueError("CSA V2 version is required")
    if set(players) != {"+", "-"}:
        raise ValueError("CSA N+ and N- player names are required")
    if board is None or initial_sfen is None or initial_mode is None:
        raise ValueError("CSA initial position and turn are required")
    if terminal is None:
        raise ValueError("CSA terminal record is required")
    for header_name in ("START_TIME", "END_TIME"):
        header_value = headers.get(header_name)
        if header_value is not None:
            try:
                datetime.strptime(header_value, "%Y/%m/%d %H:%M:%S")
            except ValueError as error:
                raise ValueError(f"invalid CSA ${header_name}") from error
    termination, winner, terminal_validation = _termination_and_winner(terminal, board)
    outcome_samples = tuple(
        replace(
            sample,
            value_target=(
                0.0
                if winner is None
                else (1.0 if sample.turn == winner else -1.0)
            ),
        )
        for sample in samples
    )
    game = GameRecord(
        initial_sfen=initial_sfen,
        moves=tuple(moves),
        samples=outcome_samples,
        winner=winner,
        termination=termination,
    )
    return ParsedCsaGame(
        record=game,
        archive_member=source.archive_member,
        csa_sha256=_sha256_bytes(source.raw_bytes),
        csa_bytes=len(source.raw_bytes),
        text_encoding=encoding,
        csa_version=version,
        black_name=players["+"],
        white_name=players["-"],
        event=headers.get("EVENT"),
        start_time=headers.get("START_TIME"),
        end_time=headers.get("END_TIME"),
        move_times_seconds=tuple(times),
        terminal_code=terminal,
        terminal_time_seconds=terminal_time,
        terminal_record_count=terminal_record_count,
        terminal_validation=terminal_validation,
        initial_position_mode=initial_mode,
        declared_black_rating=declared_ratings.get("black"),
        declared_white_rating=declared_ratings.get("white"),
    )


@dataclass(frozen=True, slots=True)
class FloodgateCorpusConfig:
    min_rating: int = 3_900
    min_games: int = 20
    accepted_terminal_codes: tuple[str, ...] = (
        "%TORYO",
        "%SENNICHITE",
        "%TIME_UP",
        "%KACHI",
    )
    include_zero_loss_players: bool = True
    split_seed: str = "meteo-floodgate-corpus-v1"
    split_weights: tuple[tuple[str, int], ...] = (
        ("train", 80),
        ("validation", 10),
        ("sealed_final_test", 10),
    )
    overlap_protection_order: tuple[str, ...] = (
        "sealed_final_test",
        "validation",
        "train",
    )

    def __post_init__(self) -> None:
        if self.min_games < 1:
            raise ValueError("min_games must be positive")
        if not self.accepted_terminal_codes:
            raise ValueError("accepted_terminal_codes must not be empty")
        if len(set(self.accepted_terminal_codes)) != len(self.accepted_terminal_codes):
            raise ValueError("accepted_terminal_codes must be unique")
        unknown = set(self.accepted_terminal_codes) - _KNOWN_TERMINALS
        if unknown:
            raise ValueError(f"accepted_terminal_codes contains unknown codes: {sorted(unknown)}")
        if not self.split_seed.strip() or self.split_seed != self.split_seed.strip():
            raise ValueError("split_seed must be a non-empty trimmed string")
        names = tuple(name for name, _weight in self.split_weights)
        if (
            not names
            or len(set(names)) != len(names)
            or any(not name.strip() or name != name.strip() for name in names)
        ):
            raise ValueError("split names must be non-empty, trimmed, and unique")
        if any(weight < 1 for _name, weight in self.split_weights):
            raise ValueError("split weights must be positive integers")
        if set(self.overlap_protection_order) != set(names) or len(
            self.overlap_protection_order
        ) != len(names):
            raise ValueError("overlap protection order must contain every split exactly once")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _SelectedGame:
    parsed: ParsedCsaGame
    black_rating: RatingPlayer
    white_rating: RatingPlayer
    black_identity: EngineIdentity
    white_identity: EngineIdentity
    split: str


def _rating_candidates(snapshot: RatingSnapshot) -> dict[str, tuple[RatingPlayer, ...]]:
    grouped: dict[str, list[RatingPlayer]] = defaultdict(list)
    for player in snapshot.players:
        grouped[player.display_name].append(player)
    return {name: tuple(players) for name, players in grouped.items()}


def _assigned_split(raw_sha256: str, config: FloodgateCorpusConfig) -> str:
    total = sum(weight for _name, weight in config.split_weights)
    digest = hashlib.sha256(f"{config.split_seed}\0{raw_sha256}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % total
    cumulative = 0
    for name, weight in config.split_weights:
        cumulative += weight
        if bucket < cumulative:
            return name
    raise AssertionError("split bucket was not assigned")


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("xb") as stream:
        stream.write(_canonical_json_bytes(payload))


def _pairwise_overlap(
    positions: Mapping[str, set[str]],
) -> list[dict[str, object]]:
    overlaps: list[dict[str, object]] = []
    names = sorted(positions)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            intersection = positions[left] & positions[right]
            overlaps.append(
                {
                    "left": left,
                    "right": right,
                    "normalized_sample_positions": len(intersection),
                    "examples": sorted(intersection)[:3],
                }
            )
    return overlaps


def create_floodgate_corpus(
    output_directory: Path,
    *,
    archive: Path,
    csa_sources: Iterable[CsaSource],
    rating_snapshot: RatingSnapshot,
    config: FloodgateCorpusConfig | None = None,
    identity_rules: Sequence[EngineIdentityRule] = (),
) -> dict[str, object]:
    """Create one immutable local-only corpus and its candidate replays.

    Games are assigned to one split by raw-CSA hash.  Since nearly all games
    share the start position, sample positions that occur in more than one
    split are retained only in the highest-protection split.  Complete moves
    remain in the GameRecord so a retained sample still has its exact history.
    """

    resolved_config = config or FloodgateCorpusConfig()
    output_path = output_directory.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    archive_identity = identify_file(archive)
    source_rows: list[dict[str, object]] = []
    parsed_games: list[ParsedCsaGame] = []
    parse_failure_rows: list[dict[str, object]] = []
    seen_members: set[str] = set()
    seen_hashes: set[str] = set()
    source_count = 0
    for source in csa_sources:
        source_count += 1
        if source.archive_member in seen_members:
            raise ValueError("duplicate CSA archive member")
        seen_members.add(source.archive_member)
        source_sha256 = _sha256_bytes(source.raw_bytes)
        if source_sha256 in seen_hashes:
            raise ValueError("duplicate CSA byte content")
        seen_hashes.add(source_sha256)
        source_row: dict[str, object] = {
            "archive_member": source.archive_member,
            "sha256": source_sha256,
            "bytes": len(source.raw_bytes),
        }
        source_rows.append(source_row)
        try:
            parsed_games.append(parse_csa_game(source))
        except ValueError as error:
            parse_failure_rows.append(
                {
                    "archive_member": source.archive_member,
                    "csa_sha256": source_row["sha256"],
                    "black_name": None,
                    "white_name": None,
                    "split_assignment": None,
                    "included": False,
                    "exclusion_reasons": ["csa_parse_failed"],
                    "parse_error": str(error),
                }
            )
    if source_count == 0:
        raise ValueError("at least one CSA source is required")
    if not parsed_games:
        raise ValueError("no CSA source passed strict parsing")
    identities = resolve_engine_identities(
        (
            name
            for game in parsed_games
            for name in (game.black_name, game.white_name)
        ),
        identity_rules,
    )
    ratings_by_name = _rating_candidates(rating_snapshot)

    selected: list[_SelectedGame] = []
    inspected_rows: list[dict[str, object]] = list(parse_failure_rows)
    for game in parsed_games:
        reasons: list[str] = []
        black_candidates = ratings_by_name.get(game.black_name, ())
        white_candidates = ratings_by_name.get(game.white_name, ())
        if len(black_candidates) != 1:
            reasons.append(
                "black_rating_missing" if not black_candidates else "black_rating_ambiguous"
            )
        if len(white_candidates) != 1:
            reasons.append(
                "white_rating_missing" if not white_candidates else "white_rating_ambiguous"
            )
        black_rating = black_candidates[0] if len(black_candidates) == 1 else None
        white_rating = white_candidates[0] if len(white_candidates) == 1 else None
        for color, player in (("black", black_rating), ("white", white_rating)):
            if player is None:
                continue
            if player.rating < resolved_config.min_rating:
                reasons.append(f"{color}_rating_below_minimum")
            if player.games < resolved_config.min_games:
                reasons.append(f"{color}_games_below_minimum")
            if player.zero_loss_uncertain and not resolved_config.include_zero_loss_players:
                reasons.append(f"{color}_zero_loss_disallowed")
        if game.terminal_code not in resolved_config.accepted_terminal_codes:
            reasons.append("terminal_code_not_accepted")
        split = _assigned_split(game.csa_sha256, resolved_config)
        inspected_rows.append(
            {
                "archive_member": game.archive_member,
                "csa_sha256": game.csa_sha256,
                "black_name": game.black_name,
                "white_name": game.white_name,
                "split_assignment": split,
                "included": not reasons,
                "exclusion_reasons": reasons,
            }
        )
        if reasons:
            continue
        if black_rating is None or white_rating is None:
            raise AssertionError("rating exclusion failed closed")
        selected.append(
            _SelectedGame(
                parsed=game,
                black_rating=black_rating,
                white_rating=white_rating,
                black_identity=identities[game.black_name],
                white_identity=identities[game.white_name],
                split=split,
            )
        )
    if not selected:
        raise ValueError("strong-vs-strong filter selected no games")

    protection_rank = {
        name: index for index, name in enumerate(resolved_config.overlap_protection_order)
    }
    owners: dict[str, str] = {}
    for selected_game in selected:
        for sample in selected_game.parsed.record.samples:
            key = normalized_sfen(sample.sfen)
            current = owners.get(key)
            if current is None or protection_rank[selected_game.split] < protection_rank[current]:
                owners[key] = selected_game.split

    split_games: dict[str, list[GameRecord]] = {
        name: [] for name, _weight in resolved_config.split_weights
    }
    split_positions: dict[str, set[str]] = {name: set() for name in split_games}
    game_rows: list[dict[str, object]] = []
    for selected_game in sorted(
        selected, key=lambda item: (item.split, item.parsed.csa_sha256)
    ):
        retained = tuple(
            sample
            for sample in selected_game.parsed.record.samples
            if owners[normalized_sfen(sample.sfen)] == selected_game.split
        )
        withheld = len(selected_game.parsed.record.samples) - len(retained)
        record = replace(selected_game.parsed.record, samples=retained)
        split_games[selected_game.split].append(record)
        split_positions[selected_game.split].update(
            normalized_sfen(sample.sfen) for sample in retained
        )
        row = selected_game.parsed.to_metadata()
        row.update(
            {
                "split": selected_game.split,
                "retained_candidate_samples": len(retained),
                "withheld_cross_split_samples": withheld,
                "black_rating": selected_game.black_rating.to_dict(),
                "white_rating": selected_game.white_rating.to_dict(),
                "black_identity": selected_game.black_identity.to_dict(),
                "white_identity": selected_game.white_identity.to_dict(),
                "zero_loss_rating_uncertainty": (
                    selected_game.black_rating.zero_loss_uncertain
                    or selected_game.white_rating.zero_loss_uncertain
                ),
            }
        )
        game_rows.append(row)

    overlaps = _pairwise_overlap(split_positions)
    if any(row["normalized_sample_positions"] != 0 for row in overlaps):
        raise AssertionError("cross-split normalized-position de-duplication failed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    try:
        candidate_directory = temporary / "candidates"
        candidate_directory.mkdir()
        replay_rows: dict[str, dict[str, object]] = {}
        for split_name, _weight in resolved_config.split_weights:
            replay = candidate_directory / f"{split_name}.jsonl"
            append_games(replay, split_games[split_name])
            replay_sha = identify_file(replay)
            sidecar_payload: dict[str, object] = {
                "schema": CANDIDATE_SIDECAR_SCHEMA,
                "split": split_name,
                "replay_sha256": replay_sha.sha256,
                "original_move_role": "candidate_only",
                "teacher_truth": False,
                "reanalysis_required": True,
                "required_history_mode": "game_prefix",
                "required_teacher_output": "typed_MultiPV_with_full_PV",
                "direct_training_allowed": False,
                "rights": RIGHTS_CLASSIFICATION,
                "corpus_manifest": "../manifest.json",
            }
            sidecar = replay.with_suffix(replay.suffix + ".provenance.json")
            _write_json(sidecar, sidecar_payload)
            sidecar_identity = identify_file(sidecar)
            replay_rows[split_name] = {
                "path": f"candidates/{replay.name}",
                "sha256": replay_sha.sha256,
                "bytes": replay_sha.bytes,
                "games": len(split_games[split_name]),
                "candidate_samples": sum(
                    len(game.samples) for game in split_games[split_name]
                ),
                "unique_normalized_sample_positions": len(split_positions[split_name]),
                "normalized_position_fingerprint": canonical_json_sha256(
                    {"positions": sorted(split_positions[split_name])}
                ),
                "provenance_sidecar": {
                    "path": f"candidates/{sidecar.name}",
                    "sha256": sidecar_identity.sha256,
                    "bytes": sidecar_identity.bytes,
                },
            }

        manifest: dict[str, object] = {
            "schema": CORPUS_SCHEMA,
            "rights": {
                "classification": RIGHTS_CLASSIFICATION,
                "archive_redistribution_allowed": False,
                "csa_redistribution_allowed": False,
                "derived_dataset_redistribution_allowed": False,
                "derived_checkpoint_redistribution_allowed": False,
                "reason": (
                    "No explicit Floodgate game-archive license was confirmed; keep the archive, "
                    "candidate replay, reanalysis output, and checkpoints local until clarified."
                ),
            },
            "label_contract": {
                "original_move_role": "candidate_only",
                "original_move_is_teacher_truth": False,
                "terminal_result_role": "observed_game_outcome_only",
                "terminal_result_is_teacher_value": False,
                "candidate_actor_policy": "empty",
                "teacher_fields": "unset",
                "reanalysis_required": True,
                "required_history_mode": "game_prefix",
                "required_teacher_output": "typed_MultiPV_with_full_PV",
                "direct_training_allowed": False,
            },
            "inputs": {
                "archive": archive_identity.to_dict(),
                "rating_snapshot": rating_snapshot.to_dict(),
                "csa_sources": [
                    row
                    for row in sorted(
                        source_rows, key=lambda item: str(item["archive_member"])
                    )
                ],
            },
            "filter": resolved_config.to_dict(),
            "identity_contract": {
                "inference_performed": False,
                "unknown_component": UNKNOWN_IDENTITY_COMPONENT,
                "rules": [rule.to_dict() for rule in identity_rules],
            },
            "inspection": {
                "sources": source_count,
                "parsed_games": len(parsed_games),
                "parse_failures": len(parse_failure_rows),
                "selected_games": len(selected),
                "excluded_games": source_count - len(selected),
                "rows": sorted(inspected_rows, key=lambda row: str(row["archive_member"])),
            },
            "splits": replay_rows,
            "leakage_guard": {
                "assignment_unit": "game",
                "assignment_key": "sha256(split_seed + NUL + raw_csa_sha256)",
                "normalized_identity_omits_sfen_move_counter": True,
                "cross_split_duplicate_policy": (
                    "retain samples only in the earliest overlap_protection_order split; preserve "
                    "full game moves so every retained sample keeps exact history"
                ),
                "pairwise_intersections": overlaps,
                "verified_zero_cross_split_overlap": True,
            },
            "games": game_rows,
        }
        _write_json(temporary / "manifest.json", manifest)
        if output_path.exists():
            raise FileExistsError(output_path)
        os.rename(temporary, output_path)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
