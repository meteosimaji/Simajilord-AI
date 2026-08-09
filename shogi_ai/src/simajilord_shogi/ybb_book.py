"""Read YaneuraOu binary books as *positions requiring reanalysis*.

The binary book is never treated as a policy or value target.  Its moves,
evaluations, and depths are available only to integrity and anti-book audits;
training inputs must be produced by a fresh no-book teacher search.  This
separation lets Meteo learn the positions without copying the opening book.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType

from rsshogi.core import Board, Move

from .artifact_provenance import FileIdentity, identify_file

YBB_SCHEMA = "meteo-ybb-no-book-reanalysis-plan-v1"
YBB_RECEIPT_SCHEMA = "meteo-ybb-no-book-reanalysis-receipt-v1"
YBB_MAGIC = b"YANE-BINBOOK-V1\x00"
YBB_MOVE_DEPTH_FLAG = 1
YBB_KNOWN_FLAGS = YBB_MOVE_DEPTH_FLAG
YBB_HEADER = struct.Struct("<16sQQ")
YBB_INDEX = struct.Struct("<32sQHH")
YBB_MOVE = struct.Struct("<HhH")
YBB_MOVE_WITHOUT_DEPTH = struct.Struct("<Hh")
YBB_SPECIAL_MOVE16 = frozenset({0, 129, 258, 387})


def _require_sha256(value: str, *, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _stable_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class YbbHeader:
    record_count: int
    flags: int
    index_bytes: int
    moves_base: int
    moves_bytes: int
    move_record_bytes: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class YbbIndexEntry:
    index: int
    packed_sfen: bytes
    moves_offset: int
    ply: int
    move_count: int

    def __post_init__(self) -> None:
        if self.index < 0 or len(self.packed_sfen) != 32:
            raise ValueError("invalid YBB index entry identity")
        if self.moves_offset < 0 or self.ply < 0 or self.move_count < 1:
            raise ValueError("invalid YBB index entry counters")


@dataclass(frozen=True, slots=True)
class YbbBookMove:
    """One book record for local integrity/anti-book audits, never a label."""

    move16: int
    move_usi: str
    evaluation: int
    depth: int | None


@dataclass(frozen=True, slots=True)
class YbbPositionCandidate:
    """An observable position that must receive a new no-book analysis."""

    index: int
    sfen: str
    normalized_sfen: str
    ply: int
    turn: int
    book_candidate_count: int
    packed_sfen_sha256: str
    reanalysis_required: bool = True
    book_moves_are_training_targets: bool = False

    def __post_init__(self) -> None:
        _require_sha256(self.packed_sfen_sha256, label="packed SFEN")
        if not self.reanalysis_required or self.book_moves_are_training_targets:
            raise ValueError("YBB positions must remain no-book reanalysis candidates")


@dataclass(frozen=True, slots=True)
class YbbStructuralAudit:
    source: FileIdentity
    header: YbbHeader
    entries_scanned: int
    move_records_referenced: int
    first_ply: int
    last_ply: int
    minimum_ply: int
    maximum_ply: int
    minimum_move_count: int
    maximum_move_count: int
    duplicate_packed_sfens: int
    descending_packed_sfens: int
    noncontiguous_move_ranges: int

    def __post_init__(self) -> None:
        if self.entries_scanned != self.header.record_count:
            raise ValueError("YBB audit did not cover every index record")
        if any(
            count != 0
            for count in (
                self.duplicate_packed_sfens,
                self.descending_packed_sfens,
                self.noncontiguous_move_ranges,
            )
        ):
            raise ValueError("YBB index is not a unique, ordered, contiguous corpus")

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source.to_dict(),
            "header": self.header.to_dict(),
            "entries_scanned": self.entries_scanned,
            "move_records_referenced": self.move_records_referenced,
            "first_ply": self.first_ply,
            "last_ply": self.last_ply,
            "minimum_ply": self.minimum_ply,
            "maximum_ply": self.maximum_ply,
            "minimum_move_count": self.minimum_move_count,
            "maximum_move_count": self.maximum_move_count,
            "duplicate_packed_sfens": self.duplicate_packed_sfens,
            "descending_packed_sfens": self.descending_packed_sfens,
            "noncontiguous_move_ranges": self.noncontiguous_move_ranges,
        }


class YaneuraOuBinaryBook:
    """Random-access, bounded-memory reader for ``YANE-BINBOOK-V1`` files."""

    def __init__(self, path: Path) -> None:
        resolved = path.expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"YBB input is not a regular file: {resolved}")
        descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            descriptor_stat = os.fstat(descriptor)
            if not stat.S_ISREG(descriptor_stat.st_mode):
                raise ValueError("YBB descriptor is not a regular file")
            header_bytes = os.pread(descriptor, YBB_HEADER.size, 0)
            if len(header_bytes) != YBB_HEADER.size:
                raise ValueError("YBB file is shorter than its header")
            magic, record_count, flags = YBB_HEADER.unpack(header_bytes)
            if magic != YBB_MAGIC:
                raise ValueError("unsupported YBB magic")
            if record_count < 1:
                raise ValueError("YBB record count must be positive")
            if flags & ~YBB_KNOWN_FLAGS:
                raise ValueError(f"YBB contains unknown flags: {flags:#x}")
            move_record_bytes = (
                YBB_MOVE.size if flags & YBB_MOVE_DEPTH_FLAG else YBB_MOVE_WITHOUT_DEPTH.size
            )
            index_bytes = record_count * YBB_INDEX.size
            moves_base = YBB_HEADER.size + index_bytes
            if moves_base > descriptor_stat.st_size:
                raise ValueError("YBB index extends beyond the file")
            moves_bytes = descriptor_stat.st_size - moves_base
            if moves_bytes % move_record_bytes:
                raise ValueError("YBB moves region is not record aligned")
        except BaseException:
            os.close(descriptor)
            raise
        self.path = resolved
        self._descriptor = descriptor
        self._device = descriptor_stat.st_dev
        self._inode = descriptor_stat.st_ino
        self._size = descriptor_stat.st_size
        self.header = YbbHeader(
            record_count=record_count,
            flags=flags,
            index_bytes=index_bytes,
            moves_base=moves_base,
            moves_bytes=moves_bytes,
            move_record_bytes=move_record_bytes,
        )

    def __enter__(self) -> YaneuraOuBinaryBook:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def __len__(self) -> int:
        return self.header.record_count

    def _require_open(self) -> int:
        if self._descriptor < 0:
            raise ValueError("YBB reader is closed")
        current = os.fstat(self._descriptor)
        if (
            current.st_dev != self._device
            or current.st_ino != self._inode
            or current.st_size != self._size
        ):
            raise RuntimeError("YBB source identity changed while open")
        return self._descriptor

    def _read_exact(self, offset: int, size: int) -> bytes:
        descriptor = self._require_open()
        if offset < 0 or size < 0 or offset + size > self._size:
            raise ValueError("YBB read is outside the source file")
        value = os.pread(descriptor, size, offset)
        if len(value) != size:
            raise RuntimeError("short YBB read")
        return value

    def index_entry(self, index: int) -> YbbIndexEntry:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        offset = YBB_HEADER.size + index * YBB_INDEX.size
        packed_sfen, moves_offset, ply, move_count = YBB_INDEX.unpack(
            self._read_exact(offset, YBB_INDEX.size)
        )
        end = moves_offset + move_count * self.header.move_record_bytes
        if end > self.header.moves_bytes:
            raise ValueError(f"YBB move range is outside the file at record {index}")
        return YbbIndexEntry(
            index=index,
            packed_sfen=packed_sfen,
            moves_offset=moves_offset,
            ply=ply,
            move_count=move_count,
        )

    def iter_index_entries(
        self, *, start: int = 0, stop: int | None = None, chunk_records: int = 65_536
    ) -> Iterator[YbbIndexEntry]:
        end = len(self) if stop is None else stop
        if start < 0 or end < start or end > len(self):
            raise ValueError("invalid YBB index range")
        if chunk_records < 1:
            raise ValueError("chunk_records must be positive")
        cursor = start
        while cursor < end:
            count = min(chunk_records, end - cursor)
            raw = self._read_exact(
                YBB_HEADER.size + cursor * YBB_INDEX.size,
                count * YBB_INDEX.size,
            )
            for relative in range(count):
                packed_sfen, moves_offset, ply, move_count = YBB_INDEX.unpack_from(
                    raw, relative * YBB_INDEX.size
                )
                index = cursor + relative
                move_end = moves_offset + move_count * self.header.move_record_bytes
                if move_count < 1 or move_end > self.header.moves_bytes:
                    raise ValueError(f"invalid YBB move range at record {index}")
                yield YbbIndexEntry(
                    index=index,
                    packed_sfen=packed_sfen,
                    moves_offset=moves_offset,
                    ply=ply,
                    move_count=move_count,
                )
            cursor += count

    @staticmethod
    def _board_for_entry(entry: YbbIndexEntry) -> Board:
        board = Board()
        try:
            board.set_packed_sfen(entry.packed_sfen)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid packed SFEN at YBB record {entry.index}") from error
        if not board.is_valid():
            raise ValueError(f"invalid board at YBB record {entry.index}")
        if bytes(board.to_packed_sfen()) != entry.packed_sfen:
            raise ValueError(f"packed SFEN round trip failed at YBB record {entry.index}")
        return board

    def position_candidate(self, index: int) -> YbbPositionCandidate:
        entry = self.index_entry(index)
        board = self._board_for_entry(entry)
        normalized = board.to_sfen().rsplit(" ", 1)[0]
        return YbbPositionCandidate(
            index=index,
            sfen=f"{normalized} {entry.ply}",
            normalized_sfen=normalized,
            ply=entry.ply,
            turn=int(board.turn),
            book_candidate_count=entry.move_count,
            packed_sfen_sha256=hashlib.sha256(entry.packed_sfen).hexdigest(),
        )

    def iter_position_candidates(
        self, *, start: int = 0, stop: int | None = None
    ) -> Iterator[YbbPositionCandidate]:
        for entry in self.iter_index_entries(start=start, stop=stop):
            board = self._board_for_entry(entry)
            normalized = board.to_sfen().rsplit(" ", 1)[0]
            yield YbbPositionCandidate(
                index=entry.index,
                sfen=f"{normalized} {entry.ply}",
                normalized_sfen=normalized,
                ply=entry.ply,
                turn=int(board.turn),
                book_candidate_count=entry.move_count,
                packed_sfen_sha256=hashlib.sha256(entry.packed_sfen).hexdigest(),
            )

    def book_moves_for_audit(self, index: int) -> tuple[YbbBookMove, ...]:
        """Decode local book data only to validate or challenge it.

        Callers must not copy these evaluations, depths, or moves into a
        ``teacher_policy``/``teacher_value`` field.
        """

        entry = self.index_entry(index)
        board = self._board_for_entry(entry)
        offset = self.header.moves_base + entry.moves_offset
        raw = self._read_exact(offset, entry.move_count * self.header.move_record_bytes)
        candidates: list[YbbBookMove] = []
        seen_moves: set[str] = set()
        for candidate_index in range(entry.move_count):
            record_offset = candidate_index * self.header.move_record_bytes
            if self.header.flags & YBB_MOVE_DEPTH_FLAG:
                move16, evaluation, depth = YBB_MOVE.unpack_from(raw, record_offset)
            else:
                move16, evaluation = YBB_MOVE_WITHOUT_DEPTH.unpack_from(raw, record_offset)
                depth = None
            if move16 in YBB_SPECIAL_MOVE16:
                raise ValueError(
                    f"special Move16 {move16} cannot be a YBB candidate at record {index}"
                )
            move = Move(move16)
            if not board.is_legal_move(move):
                raise ValueError(
                    f"illegal Move16 {move16} at YBB record {index}: {move.to_usi()}"
                )
            move_usi = move.to_usi()
            if move_usi in seen_moves:
                raise ValueError(f"duplicate root move {move_usi} at YBB record {index}")
            seen_moves.add(move_usi)
            candidates.append(
                YbbBookMove(
                    move16=move16,
                    move_usi=move_usi,
                    evaluation=evaluation,
                    depth=depth,
                )
            )
        return tuple(candidates)

    def structural_audit(self) -> YbbStructuralAudit:
        previous_packed: bytes | None = None
        expected_moves_offset = 0
        move_records = 0
        duplicate_packed_sfens = 0
        descending_packed_sfens = 0
        noncontiguous_move_ranges = 0
        minimum_ply = 65_535
        maximum_ply = 0
        minimum_move_count = 65_535
        maximum_move_count = 0
        first_ply = -1
        last_ply = -1
        entries_scanned = 0
        for entry in self.iter_index_entries():
            if first_ply < 0:
                first_ply = entry.ply
            last_ply = entry.ply
            minimum_ply = min(minimum_ply, entry.ply)
            maximum_ply = max(maximum_ply, entry.ply)
            minimum_move_count = min(minimum_move_count, entry.move_count)
            maximum_move_count = max(maximum_move_count, entry.move_count)
            if previous_packed is not None:
                if entry.packed_sfen == previous_packed:
                    duplicate_packed_sfens += 1
                elif entry.packed_sfen < previous_packed:
                    descending_packed_sfens += 1
            previous_packed = entry.packed_sfen
            if entry.moves_offset != expected_moves_offset:
                noncontiguous_move_ranges += 1
            expected_moves_offset = (
                entry.moves_offset + entry.move_count * self.header.move_record_bytes
            )
            move_records += entry.move_count
            entries_scanned += 1
        if expected_moves_offset != self.header.moves_bytes:
            noncontiguous_move_ranges += 1
        return YbbStructuralAudit(
            source=identify_file(self.path),
            header=self.header,
            entries_scanned=entries_scanned,
            move_records_referenced=move_records,
            first_ply=first_ply,
            last_ply=last_ply,
            minimum_ply=minimum_ply,
            maximum_ply=maximum_ply,
            minimum_move_count=minimum_move_count,
            maximum_move_count=maximum_move_count,
            duplicate_packed_sfens=duplicate_packed_sfens,
            descending_packed_sfens=descending_packed_sfens,
            noncontiguous_move_ranges=noncontiguous_move_ranges,
        )


@dataclass(frozen=True, slots=True)
class YbbReanalysisShard:
    shard_id: str
    start_index: int
    stop_index: int
    positions: int
    range_sha256: str

    def __post_init__(self) -> None:
        if not self.shard_id or self.start_index < 0 or self.stop_index <= self.start_index:
            raise ValueError("invalid YBB reanalysis shard")
        if self.positions != self.stop_index - self.start_index:
            raise ValueError("YBB shard position count does not match its range")
        _require_sha256(self.range_sha256, label="YBB shard range")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class YbbReanalysisPlan:
    source: FileIdentity
    source_url: str
    source_archive_sha256: str
    rights_scope: str
    header: YbbHeader
    structural_audit: YbbStructuralAudit
    required_teacher_ids: tuple[str, ...]
    shards: tuple[YbbReanalysisShard, ...]
    screen_nodes: int
    deep_nodes: int
    priority_nodes: int
    multipv: int

    def __post_init__(self) -> None:
        _require_sha256(self.source_archive_sha256, label="source archive")
        if not self.source_url.startswith(("https://", "http://")):
            raise ValueError("YBB source URL must be absolute HTTP(S)")
        if self.rights_scope != "user_authorized_local_analysis_only":
            raise ValueError("YBB plan must remain local-only until separate terms are reviewed")
        if not self.required_teacher_ids or len(set(self.required_teacher_ids)) != len(
            self.required_teacher_ids
        ):
            raise ValueError("YBB plan requires unique no-book teacher ids")
        if self.multipv < 2 or not (
            0 < self.screen_nodes < self.deep_nodes < self.priority_nodes
        ):
            raise ValueError("YBB plan requires increasing positive search budgets and MultiPV")
        cursor = 0
        for shard in self.shards:
            if shard.start_index != cursor:
                raise ValueError("YBB plan shards do not cover a contiguous prefix")
            cursor = shard.stop_index
        if cursor != self.header.record_count:
            raise ValueError("YBB plan does not cover every source record exactly once")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": YBB_SCHEMA,
            "source": self.source.to_dict(),
            "source_url": self.source_url,
            "source_archive_sha256": self.source_archive_sha256,
            "rights_scope": self.rights_scope,
            "header": self.header.to_dict(),
            "structural_audit": self.structural_audit.to_dict(),
            "required_teacher_ids": list(self.required_teacher_ids),
            "shards": [shard.to_dict() for shard in self.shards],
            "search_ladder": {
                "screen_nodes": self.screen_nodes,
                "deep_nodes": self.deep_nodes,
                "priority_nodes": self.priority_nodes,
                "multipv": self.multipv,
                "book_enabled": False,
                "history_mode": "position_only_unless_a_verified_prefix_is_available",
            },
            "training_contract": {
                "book_moves_are_labels": False,
                "book_evaluations_are_labels": False,
                "book_depths_are_labels": False,
                "fresh_no_book_multipv_required": True,
                "all_source_records_covered_exactly_once": True,
                "deep_escalation": (
                    "depth flips, teacher disagreement, mate, defense, early deviation, "
                    "or held-out weakness"
                ),
                "publication": "source book and its derived move/eval data are excluded",
            },
        }

    @property
    def sha256(self) -> str:
        return _stable_sha256(self.to_dict())


def build_ybb_reanalysis_plan(
    book_path: Path,
    *,
    source_url: str,
    source_archive_sha256: str,
    required_teacher_ids: Sequence[str],
    shard_size: int = 100_000,
    screen_nodes: int = 20_000,
    deep_nodes: int = 200_000,
    priority_nodes: int = 2_000_000,
    multipv: int = 8,
    local_only_user_authorized: bool = False,
) -> YbbReanalysisPlan:
    """Audit a complete YBB index and make an exactly-covering no-book plan."""

    if not local_only_user_authorized:
        raise PermissionError(
            "YBB terms do not approve publication; explicit local-only user "
            "authorization is required"
        )
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    teacher_ids = tuple(required_teacher_ids)
    with YaneuraOuBinaryBook(book_path) as book:
        audit = book.structural_audit()
        source = audit.source
        header = book.header
    shards: list[YbbReanalysisShard] = []
    for shard_number, start in enumerate(range(0, header.record_count, shard_size)):
        stop = min(start + shard_size, header.record_count)
        identity = {
            "schema": YBB_SCHEMA,
            "source_sha256": source.sha256,
            "start_index": start,
            "stop_index": stop,
        }
        shards.append(
            YbbReanalysisShard(
                shard_id=f"ybb-{shard_number:06d}",
                start_index=start,
                stop_index=stop,
                positions=stop - start,
                range_sha256=_stable_sha256(identity),
            )
        )
    return YbbReanalysisPlan(
        source=source,
        source_url=source_url,
        source_archive_sha256=source_archive_sha256,
        rights_scope="user_authorized_local_analysis_only",
        header=header,
        structural_audit=audit,
        required_teacher_ids=teacher_ids,
        shards=tuple(shards),
        screen_nodes=screen_nodes,
        deep_nodes=deep_nodes,
        priority_nodes=priority_nodes,
        multipv=multipv,
    )


def write_ybb_reanalysis_plan(plan: YbbReanalysisPlan, output: Path) -> str:
    """Create an immutable plan without an overwrite race."""

    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite YBB plan: {destination}")
    payload = json.dumps(
        plan.to_dict(), indent=2, sort_keys=True, allow_nan=False, ensure_ascii=True
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite YBB plan: {destination}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class YbbReanalysisReceipt:
    teacher_id: str
    shard_id: str
    source_sha256: str
    range_sha256: str
    output_replay_sha256: str
    positions_attempted: int
    positions_succeeded: int
    positions_failed: int
    nodes: int
    multipv: int
    book_enabled: bool
    schema: str = YBB_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != YBB_RECEIPT_SCHEMA or not self.teacher_id or not self.shard_id:
            raise ValueError("invalid YBB receipt identity")
        for label, value in (
            ("receipt source", self.source_sha256),
            ("receipt range", self.range_sha256),
            ("receipt output", self.output_replay_sha256),
        ):
            _require_sha256(value, label=label)
        if min(
            self.positions_attempted,
            self.positions_succeeded,
            self.positions_failed,
            self.nodes,
            self.multipv,
        ) < 0:
            raise ValueError("YBB receipt counters must be non-negative")
        if self.positions_succeeded + self.positions_failed != self.positions_attempted:
            raise ValueError("YBB receipt attempted count does not reconcile")
        if self.book_enabled:
            raise ValueError("YBB teacher reanalysis must have the opening book disabled")


def verify_ybb_reanalysis_coverage(
    plan: YbbReanalysisPlan, receipts: Sequence[YbbReanalysisReceipt]
) -> dict[str, object]:
    """Require every (teacher, shard) cell to succeed exactly once."""

    shards = {shard.shard_id: shard for shard in plan.shards}
    required = {
        (teacher_id, shard.shard_id)
        for teacher_id in plan.required_teacher_ids
        for shard in plan.shards
    }
    observed: dict[tuple[str, str], YbbReanalysisReceipt] = {}
    for receipt in receipts:
        key = (receipt.teacher_id, receipt.shard_id)
        if key in observed:
            raise ValueError(f"duplicate YBB receipt for {key}")
        if key not in required:
            raise ValueError(f"unexpected YBB receipt for {key}")
        shard = shards[receipt.shard_id]
        if receipt.source_sha256 != plan.source.sha256:
            raise ValueError("YBB receipt source digest does not match the plan")
        if receipt.range_sha256 != shard.range_sha256:
            raise ValueError("YBB receipt range digest does not match the plan")
        if (
            receipt.positions_attempted != shard.positions
            or receipt.positions_succeeded != shard.positions
            or receipt.positions_failed != 0
        ):
            raise ValueError("YBB receipt does not prove complete successful shard coverage")
        if receipt.nodes < plan.screen_nodes or receipt.multipv < plan.multipv:
            raise ValueError("YBB receipt search budget is below the plan")
        observed[key] = receipt
    missing = sorted(required - set(observed))
    if missing:
        raise ValueError(f"YBB no-book reanalysis is incomplete: {len(missing)} cells missing")
    return {
        "schema": "meteo-ybb-no-book-reanalysis-coverage-v1",
        "plan_sha256": plan.sha256,
        "source_records": plan.header.record_count,
        "teachers": list(plan.required_teacher_ids),
        "shards": len(plan.shards),
        "required_receipts": len(required),
        "complete": True,
        "book_moves_used_as_labels": False,
    }
