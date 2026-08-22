"""Create-only value-only PSV shards for post-bootstrap NNUE updates.

The post-bootstrap loop deliberately does not encode a policy in ``Move16``.
Only scalar labels that are exact on one pinned scorer's scale, or exact rule
terminals, can enter the compact 40-byte PSV stream.  Bounds and unresolved
positions remain in the score-matrix/reanalysis queue because PackedSfenValue
has nowhere to preserve their interval semantics.

Every emitted record is accompanied by a receipt row.  The sidecar keeps the
identity that PSV itself cannot represent: scorer, search budget, label kind,
whether WDL is known, and the pre-labelled dataset split.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import numpy as np
from rsshogi.core import Board
from rsshogi.numpy import PackedSfenValue

from .artifact_provenance import sha256_file
from .distillation_targets import CANONICAL_SCORER_IDS
from .ensemble import normalized_sfen
from .teacher_data import PSV_RECORD_BYTES

INCREMENTAL_VALUE_LABEL_SCHEMA = "meteo-incremental-value-label-v1"
INCREMENTAL_VALUE_REPLAY_SCHEMA = "meteo-incremental-value-replay-v1"
INCREMENTAL_VALUE_SPLIT_SCHEMA = "meteo-incremental-value-split-v1"
PROVEN_MATE_SCORE = 32_000

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_LABEL_INPUT_FIELDS = {
    "schema",
    "sfen",
    "score_cp",
    "game_ply",
    "game_result",
    "game_result_known",
    "kind",
    "scorer_id",
    "requested_nodes",
    "source_receipt_sha256",
    "qsearch_leaf_rescored",
    "history_dependent",
}


class ScalarLabelKind(StrEnum):
    """Exact sources representable by a single scalar PSV score."""

    ANCHOR_SEARCH_EXACT = "anchor_search_exact"
    PROVEN_TERMINAL = "proven_terminal"
    PROVEN_MATE = "proven_mate"


def _require_sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in _SHA256_CHARACTERS for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _position_identity(canonical_position: str) -> str:
    return hashlib.sha256(canonical_position.encode("utf-8")).hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    normalized_keys: dict[str, str] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        normalized = key.casefold()
        if normalized in normalized_keys:
            raise ValueError(
                "case-insensitive JSON object key collision: "
                f"{normalized_keys[normalized]!r} and {key!r}"
            )
        normalized_keys[normalized] = key
        result[key] = value
    return result


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value: Any = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _install_create_only(path: Path, payload: bytes) -> str:
    """Install bytes atomically, allowing only byte-identical idempotent resume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FileExistsError(f"refusing symlinked create-only artifact: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace different artifact: {path}")
        return hashlib.sha256(payload).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise
            if path.is_symlink() or path.read_bytes() != payload:
                raise FileExistsError(f"refusing concurrently replaced artifact: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class IncrementalValueSplit:
    """A split decision made before search labels are generated."""

    split_id: str
    source_games_sha256: str
    position_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.split_id, str) or not isinstance(self.source_games_sha256, str):
            raise TypeError("split identifiers must be strings")
        if self.split_id not in {"train", "calibration", "held_out_test"}:
            raise ValueError("split_id must be train, calibration, or held_out_test")
        _require_sha256(self.source_games_sha256, label="source games")
        if not self.position_ids:
            raise ValueError("split must contain at least one position identity")
        for position_id in self.position_ids:
            _require_sha256(position_id, label="position identity")
        if len(set(self.position_ids)) != len(self.position_ids):
            raise ValueError("split contains duplicate position identities")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": INCREMENTAL_VALUE_SPLIT_SCHEMA,
            "split_id": self.split_id,
            "source_games_sha256": self.source_games_sha256,
            "position_ids": list(self.position_ids),
            "label_generation_happens_after_split": True,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()


@dataclass(frozen=True, slots=True)
class ScalarValueLabel:
    """One exact, board-only scalar label on a single scorer's scale."""

    sfen: str
    score_cp: int
    game_ply: int
    game_result: int
    game_result_known: bool
    kind: ScalarLabelKind
    scorer_id: str | None
    requested_nodes: int | None
    source_receipt_sha256: str
    qsearch_leaf_rescored: bool
    history_dependent: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.sfen, str):
            raise TypeError("scalar-label SFEN must be a string")
        try:
            board = Board(self.sfen)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid scalar-label SFEN: {self.sfen}") from error
        if not board.is_valid():
            raise ValueError(f"invalid scalar-label SFEN: {self.sfen}")
        object.__setattr__(self, "sfen", board.to_sfen())
        if (
            isinstance(self.score_cp, bool)
            or not isinstance(self.score_cp, int)
            or not -32_768 <= self.score_cp <= 32_767
        ):
            raise ValueError("score_cp must fit signed int16")
        if (
            isinstance(self.game_ply, bool)
            or not isinstance(self.game_ply, int)
            or not 0 <= self.game_ply <= 65_535
        ):
            raise ValueError("game_ply must fit unsigned int16")
        if isinstance(self.game_result, bool) or not isinstance(self.game_result, int):
            raise ValueError("game_result must be an integer")
        if self.game_result not in {-1, 0, 1}:
            raise ValueError("game_result must be -1, 0, or 1 in side-to-move perspective")
        if type(self.game_result_known) is not bool:
            raise TypeError("game_result_known must be bool")
        if type(self.qsearch_leaf_rescored) is not bool or type(self.history_dependent) is not bool:
            raise TypeError("qsearch_leaf_rescored and history_dependent must be bool")
        if self.scorer_id is not None and not isinstance(self.scorer_id, str):
            raise TypeError("scorer_id must be a string or null")
        if self.requested_nodes is not None and (
            isinstance(self.requested_nodes, bool) or not isinstance(self.requested_nodes, int)
        ):
            raise TypeError("requested_nodes must be an integer or null")
        if not isinstance(self.source_receipt_sha256, str):
            raise TypeError("source_receipt_sha256 must be a string")
        _require_sha256(self.source_receipt_sha256, label="source receipt")
        if self.history_dependent:
            raise ValueError(
                "history-dependent repetition/perpetual-check labels cannot enter board-only NNUE"
            )
        if self.kind is ScalarLabelKind.ANCHOR_SEARCH_EXACT:
            if self.scorer_id not in CANONICAL_SCORER_IDS:
                raise ValueError("anchor search label requires one canonical scorer")
            if self.requested_nodes is None or self.requested_nodes < 1:
                raise ValueError("anchor search label requires a positive requested_nodes")
            if not self.qsearch_leaf_rescored:
                raise ValueError("anchor search label requires the qsearch leaf to be rescored")
            if abs(self.score_cp) >= PROVEN_MATE_SCORE:
                raise ValueError("ordinary anchor score cannot use the proven-mate score domain")
        elif self.kind is ScalarLabelKind.PROVEN_MATE:
            if abs(self.score_cp) != PROVEN_MATE_SCORE:
                raise ValueError("proven mate label must use exactly +/-32000")
            if not self.game_result_known or self.game_result != (1 if self.score_cp > 0 else -1):
                raise ValueError("proven mate result must be known and match the score sign")
            if self.scorer_id is not None or self.requested_nodes is not None:
                raise ValueError("proven mate is a rule/proof label, not a scorer cp label")
        elif self.kind is ScalarLabelKind.PROVEN_TERMINAL:
            if not self.game_result_known:
                raise ValueError("proven terminal requires a known rule result")
            if self.score_cp != self.game_result * PROVEN_MATE_SCORE:
                raise ValueError("proven terminal score must be game_result * 32000")
            if self.scorer_id is not None or self.requested_nodes is not None:
                raise ValueError("proven terminal is not a scorer label")
        if not self.game_result_known and self.game_result != 0:
            raise ValueError("unknown WDL must use neutral PSV game_result=0")

    @property
    def normalized_position(self) -> str:
        return normalized_sfen(self.sfen)

    @property
    def position_id(self) -> str:
        return _position_identity(self.normalized_position)

    def to_receipt_row(self) -> dict[str, object]:
        row = self.to_input_row()
        row["normalized_sfen"] = self.normalized_position
        row["position_id"] = self.position_id
        row["move16"] = 0
        return row

    def to_input_row(self) -> dict[str, object]:
        row = asdict(self)
        row["schema"] = INCREMENTAL_VALUE_LABEL_SCHEMA
        row["kind"] = self.kind.value
        return row


def load_incremental_value_split(path: Path) -> IncrementalValueSplit:
    """Read and verify one pre-label split receipt without accepting extra fields."""

    requested = Path(os.path.abspath(os.fspath(path.expanduser())))
    if requested.is_symlink():
        raise ValueError("incremental split receipt must be a regular non-symlink file")
    source = requested.resolve(strict=True)
    if not source.is_file():
        raise ValueError("incremental split receipt must be a regular non-symlink file")
    value = _strict_json_object(source.read_bytes(), label="incremental split receipt")
    expected = {
        "schema",
        "split_id",
        "source_games_sha256",
        "position_ids",
        "label_generation_happens_after_split",
    }
    if set(value) != expected:
        raise ValueError("incremental split receipt fields do not match the schema")
    if value["schema"] != INCREMENTAL_VALUE_SPLIT_SCHEMA:
        raise ValueError("incremental split receipt schema mismatch")
    if value["label_generation_happens_after_split"] is not True:
        raise ValueError("split receipt does not prove split-before-label generation")
    raw_position_ids = value["position_ids"]
    if not isinstance(raw_position_ids, list) or not all(
        isinstance(item, str) for item in raw_position_ids
    ):
        raise ValueError("split position_ids must be a JSON string array")
    if not isinstance(value["split_id"], str) or not isinstance(value["source_games_sha256"], str):
        raise ValueError("split identifiers must be strings")
    return IncrementalValueSplit(
        split_id=value["split_id"],
        source_games_sha256=value["source_games_sha256"],
        position_ids=tuple(raw_position_ids),
    )


def load_scalar_value_labels(path: Path) -> tuple[ScalarValueLabel, ...]:
    """Read strict JSONL labels produced after the immutable split decision."""

    requested = Path(os.path.abspath(os.fspath(path.expanduser())))
    if requested.is_symlink():
        raise ValueError("scalar labels must be a regular non-symlink file")
    source = requested.resolve(strict=True)
    if not source.is_file():
        raise ValueError("scalar labels must be a regular non-symlink file")
    labels: list[ScalarValueLabel] = []
    with source.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank scalar label row at line {line_number}")
            value = _strict_json_object(line, label=f"scalar label line {line_number}")
            if set(value) != _LABEL_INPUT_FIELDS:
                raise ValueError(f"scalar label fields mismatch at line {line_number}")
            if value["schema"] != INCREMENTAL_VALUE_LABEL_SCHEMA:
                raise ValueError(f"scalar label schema mismatch at line {line_number}")
            try:
                label = ScalarValueLabel(
                    sfen=value["sfen"],
                    score_cp=value["score_cp"],
                    game_ply=value["game_ply"],
                    game_result=value["game_result"],
                    game_result_known=value["game_result_known"],
                    kind=ScalarLabelKind(value["kind"]),
                    scorer_id=value["scorer_id"],
                    requested_nodes=value["requested_nodes"],
                    source_receipt_sha256=value["source_receipt_sha256"],
                    qsearch_leaf_rescored=value["qsearch_leaf_rescored"],
                    history_dependent=value["history_dependent"],
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid scalar label at line {line_number}") from error
            labels.append(label)
    if not labels:
        raise ValueError("scalar labels file is empty")
    if len({label.position_id for label in labels}) != len(labels):
        raise ValueError("scalar labels contain duplicate normalized board positions")
    return tuple(labels)


def validate_incremental_split_set(
    splits: Sequence[IncrementalValueSplit],
) -> dict[str, object]:
    """Prove the required train/calibration/held-out identities are disjoint."""

    expected = {"train", "calibration", "held_out_test"}
    by_id = {split.split_id: split for split in splits}
    if len(by_id) != len(splits):
        raise ValueError("incremental split set contains duplicate split IDs")
    if set(by_id) != expected:
        raise ValueError("incremental split set must contain train, calibration, and held_out_test")
    source_hashes = {split.source_games_sha256 for split in splits}
    if len(source_hashes) != 1:
        raise ValueError("incremental splits do not share one immutable source-games artifact")
    overlaps: dict[str, list[str]] = {}
    ordered_ids = ("train", "calibration", "held_out_test")
    for index, left_id in enumerate(ordered_ids):
        left = set(by_id[left_id].position_ids)
        for right_id in ordered_ids[index + 1 :]:
            overlap = sorted(left & set(by_id[right_id].position_ids))
            overlaps[f"{left_id}__{right_id}"] = overlap
    if any(overlaps.values()):
        raise ValueError("incremental normalized positions overlap across dataset splits")
    return {
        "schema": "meteo-incremental-value-split-set-v1",
        "source_games_sha256": next(iter(source_hashes)),
        "splits": {
            split_id: {
                "positions": len(by_id[split_id].position_ids),
                "receipt_sha256": by_id[split_id].sha256,
            }
            for split_id in ordered_ids
        },
        "normalized_position_overlap": overlaps,
        "verified_zero_cross_split_overlap": True,
    }


def build_incremental_value_replay(
    labels: Sequence[ScalarValueLabel],
    *,
    split: IncrementalValueSplit,
    output_directory: Path,
) -> dict[str, object]:
    """Materialize a deduplicated value-only PSV plus a complete create-only receipt.

    The function accepts only the training split.  Calibration and sealed test
    identities use the same type so callers can prove disjointness, but they
    must never become optimizer input.
    """

    if split.split_id != "train":
        raise ValueError("only the train split can be converted into optimizer PSV")
    if not labels:
        raise ValueError("at least one scalar label is required")
    output = Path(os.path.abspath(os.fspath(output_directory.expanduser())))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite incremental value replay: {output}")

    expected_ids = set(split.position_ids)
    observed_ids = [label.position_id for label in labels]
    if any(position_id not in expected_ids for position_id in observed_ids):
        raise ValueError("scalar label is absent from the pre-labelled train split")
    if len(set(observed_ids)) != len(observed_ids):
        raise ValueError("duplicate normalized board position in incremental labels")

    ordered = tuple(sorted(labels, key=lambda item: item.position_id))
    records = np.zeros(len(ordered), dtype=PackedSfenValue)
    receipt_rows: list[dict[str, object]] = []
    for index, label in enumerate(ordered):
        board = Board(label.sfen)
        records[index]["sfen"] = np.frombuffer(board.to_packed_sfen(), dtype=np.uint8)
        records[index]["score"] = label.score_cp
        records[index]["move"] = 0
        records[index]["game_ply"] = label.game_ply
        records[index]["game_result"] = label.game_result
        receipt_rows.append(label.to_receipt_row())
    psv_bytes = records.tobytes()
    if len(psv_bytes) != len(ordered) * PSV_RECORD_BYTES:
        raise AssertionError("PackedSfenValue byte size changed unexpectedly")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        psv_path = stage / "train.psv"
        split_path = stage / "split-receipt.json"
        labels_path = stage / "labels.jsonl"
        _install_create_only(psv_path, psv_bytes)
        _install_create_only(split_path, _canonical_json_bytes(split.to_dict()))
        _install_create_only(
            labels_path,
            b"".join(_canonical_json_bytes(row) for row in receipt_rows),
        )
        receipt: dict[str, object] = {
            "schema": INCREMENTAL_VALUE_REPLAY_SCHEMA,
            "mode": "value_only_nnue_incremental_exact_labels",
            "records": len(ordered),
            "record_bytes": PSV_RECORD_BYTES,
            "policy_targets": 0,
            "move16": "zero_for_every_record",
            "cross_teacher_value_average": False,
            "bounds_written_as_point_targets": False,
            "unresolved_positions_written": False,
            "history_dependent_positions_written": False,
            "split": {
                "id": split.split_id,
                "receipt_sha256": sha256_file(split_path),
                "source_games_sha256": split.source_games_sha256,
            },
            "psv": {
                "file": "train.psv",
                "bytes": psv_path.stat().st_size,
                "sha256": sha256_file(psv_path),
            },
            "labels": {
                "file": "labels.jsonl",
                "bytes": labels_path.stat().st_size,
                "sha256": sha256_file(labels_path),
                "kinds": {
                    kind.value: sum(label.kind is kind for label in ordered)
                    for kind in ScalarLabelKind
                },
                "known_wdl": sum(label.game_result_known for label in ordered),
            },
            "publication_allowed": False,
            "complete": True,
        }
        _install_create_only(stage / "receipt.json", _canonical_json_bytes(receipt))
        directory_descriptor = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.rename(stage, output)
        parent_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        if stage.exists():
            for child in stage.iterdir():
                child.unlink(missing_ok=True)
            stage.rmdir()
        raise
    return receipt
