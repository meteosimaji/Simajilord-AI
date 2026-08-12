"""Fetch small, immutable samples from reviewed public PSV position sources.

The source Move16, score, and game result are deliberately discarded as
training labels.  These depth-9 corpora save position-generation time, but the
positions still require current-teacher reanalysis before an optimizer may see
them.  Every HTTP byte range and output position is bound by a create-only
receipt so a later relabel run can reproduce the exact source sample.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import numpy as np
from rsshogi.core import Board, Move
from rsshogi.numpy import PackedSfenValue

from .ensemble import normalized_sfen
from .teacher_data import PSV_RECORD_BYTES
from .teacher_lineage import (
    PUBLIC_CORPORA,
    CorpusSeedSamplingScope,
    PsvMoveFieldContract,
    corpus_lineage,
)

PUBLIC_PSV_SEED_SCHEMA = "meteo-public-psv-position-seeds-v1"
PUBLIC_POSITION_SEED_SCHEMA = "meteo-public-position-seed-v1"
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_USER_AGENT = "Simajilord-Meteo-public-position-seed/1"
_ALLOWED_CORPORA = tuple(
    corpus.corpus_id
    for corpus in PUBLIC_CORPORA
    if corpus.seed_sampling_scope is not None
)


@dataclass(frozen=True, slots=True)
class HttpRangeResult:
    payload: bytes
    start: int
    end: int
    total_bytes: int
    etag: str | None
    final_url: str


def available_public_psv_corpora() -> tuple[str, ...]:
    return _ALLOWED_CORPORA


def _reject_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key from public corpus API: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant from public corpus API: {value}")


def _fetch_url(url: str, *, timeout_seconds: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return cast(bytes, response.read())


def _fetch_range(
    url: str,
    *,
    start: int,
    end: int,
    timeout_seconds: float,
) -> HttpRangeResult:
    if start < 0 or end < start:
        raise ValueError("HTTP byte range must be non-negative and ordered")
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": _USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = getattr(response, "status", response.getcode())
        if status != 206:
            raise ValueError(f"public corpus server ignored byte range with HTTP {status}")
        content_range = response.headers.get("Content-Range")
        match = _CONTENT_RANGE.fullmatch(content_range or "")
        if match is None:
            raise ValueError("public corpus response lacks a strict Content-Range header")
        observed_start, observed_end, total_bytes = map(int, match.groups())
        if (observed_start, observed_end) != (start, end):
            raise ValueError(
                "public corpus returned a different byte range: "
                f"requested={start}-{end} observed={observed_start}-{observed_end}"
            )
        payload = response.read()
        if len(payload) != end - start + 1:
            raise ValueError("public corpus byte-range payload length is inconsistent")
        return HttpRangeResult(
            payload=payload,
            start=start,
            end=end,
            total_bytes=total_bytes,
            etag=response.headers.get("ETag"),
            final_url=response.geturl(),
        )


def _strict_api_payload(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("public corpus API did not return strict UTF-8 JSON") from error
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError("public corpus API root must be an object")
    return cast(dict[str, Any], payload)


def _stable_hash(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _file_names(
    payload: Mapping[str, object],
    *,
    revision: str,
    filename_pattern: str | None,
) -> tuple[str, ...]:
    reported_revision = payload.get("sha")
    if reported_revision is not None and reported_revision != revision:
        raise ValueError(
            "public corpus API revision mismatch: "
            f"expected={revision} observed={reported_revision}"
        )
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        raise ValueError("public corpus API siblings must be an array")
    names: list[str] = []
    for index, sibling in enumerate(siblings):
        if not isinstance(sibling, dict) or not isinstance(sibling.get("rfilename"), str):
            raise ValueError(f"public corpus sibling {index} has no string filename")
        name = cast(str, sibling["rfilename"])
        if name.endswith(".bin") and (
            filename_pattern is None or re.fullmatch(filename_pattern, name) is not None
        ):
            names.append(name)
    if not names or len(names) != len(set(names)):
        raise ValueError("public corpus has no PSV files or has duplicate filenames")
    return tuple(names)


def _decode_seed_rows(
    payload: bytes,
    *,
    corpus_id: str,
    revision: str,
    filename: str,
    first_record: int,
    move_field_contract: PsvMoveFieldContract,
) -> tuple[list[dict[str, object]], int]:
    if not payload or len(payload) % PSV_RECORD_BYTES:
        raise ValueError("downloaded PSV range is not a positive whole-record payload")
    records: Any = np.frombuffer(payload, dtype=PackedSfenValue)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    duplicates = 0
    for local_index, record in enumerate(records):
        board = Board()
        packed_sfen = record["sfen"].tobytes()
        board.set_packed_sfen(packed_sfen)
        source_index = first_record + local_index
        if not board.is_valid():
            raise ValueError(f"invalid board at {filename} record {source_index}")
        move16 = int(record["move"])
        move_usi: str | None
        if move_field_contract is PsvMoveFieldContract.ZERO_VALUE_ONLY_REQUIRED:
            if move16 != 0:
                raise ValueError(
                    f"value-only PSV source unexpectedly has Move16={move16} at "
                    f"{filename} record {source_index}"
                )
            move_usi = None
        else:
            if move16 == 0:
                raise ValueError(
                    f"legal-move PSV source unexpectedly has Move16=0 at {filename} "
                    f"record {source_index}"
                )
            move = Move(move16)
            if not board.is_legal_move(move):
                raise ValueError(
                    f"PSV source has illegal move {move.to_usi()} at "
                    f"{filename} record {source_index}"
                )
            move_usi = move.to_usi()
        game_result = int(record["game_result"])
        if game_result not in {-1, 0, 1}:
            raise ValueError(
                f"invalid game result {game_result} at {filename} record {source_index}"
            )
        sfen = board.to_sfen()
        board_key = normalized_sfen(sfen)
        if board_key in seen:
            duplicates += 1
            continue
        seen.add(board_key)
        rows.append(
            {
                "schema": PUBLIC_POSITION_SEED_SCHEMA,
                "sfen": sfen,
                "normalized_sfen": board_key,
                "source": {
                    "corpus_id": corpus_id,
                    "revision": revision,
                    "filename": filename,
                    "record_index": source_index,
                    "packed_sfen_sha256": hashlib.sha256(packed_sfen).hexdigest(),
                },
                "discarded_legacy_annotations": {
                    "move": move_usi,
                    "move16_contract": move_field_contract.value,
                    "score": int(record["score"]),
                    "game_result": game_result,
                    "game_ply": int(record["game_ply"]),
                    "allowed_as_meteo_training_label": False,
                },
                "required_next_stage": "current_three_teacher_reanalysis",
            }
        )
    return rows, duplicates


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode()
        for row in rows
    )


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fetch_public_psv_position_seeds(
    corpus_id: str,
    output: Path,
    *,
    file_count: int = 8,
    records_per_file: int = 4096,
    seed: str = "meteo-public-position-seeds-v1",
    timeout_seconds: float = 60.0,
    allow_user_attested_local_only: bool = False,
) -> dict[str, object]:
    """Download deterministic PSV ranges and emit unlabeled position seeds."""

    if corpus_id not in _ALLOWED_CORPORA:
        raise ValueError(f"public PSV corpus must be one of {_ALLOWED_CORPORA!r}")
    if isinstance(file_count, bool) or file_count < 1:
        raise ValueError("file_count must be a positive integer")
    if isinstance(records_per_file, bool) or records_per_file < 1:
        raise ValueError("records_per_file must be a positive integer")
    if not seed:
        raise ValueError("sampling seed must not be empty")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")

    lineage = corpus_lineage(corpus_id)
    scope = lineage.seed_sampling_scope
    move_field_contract = lineage.move_field_contract
    if scope is None or move_field_contract is None or lineage.revision is None:
        raise PermissionError("corpus is not approved for deterministic PSV seed sampling")
    if scope is CorpusSeedSamplingScope.PUBLIC_LICENSED and lineage.license_id is None:
        raise PermissionError("public PSV seed ingestion requires a reviewed source license")
    if (
        scope is CorpusSeedSamplingScope.USER_ATTESTED_LOCAL_ONLY
        and not allow_user_attested_local_only
    ):
        raise PermissionError(
            "local-only corpus requires --allow-user-attested-local-only acknowledgement"
        )
    repository_path = lineage.repository_url.removeprefix("https://huggingface.co/datasets/")
    if repository_path == lineage.repository_url or not repository_path:
        raise ValueError("public PSV source is not a canonical Hugging Face dataset URL")
    revision = lineage.revision
    api_url = f"https://huggingface.co/api/datasets/{repository_path}/revision/{revision}"
    api_raw = _fetch_url(api_url, timeout_seconds=timeout_seconds)
    names = _file_names(
        _strict_api_payload(api_raw),
        revision=revision,
        filename_pattern=lineage.sample_filename_pattern,
    )
    if file_count > len(names):
        raise ValueError(
            f"requested {file_count} files but public corpus only exposes {len(names)}"
        )
    selected_names = tuple(
        sorted(names, key=lambda name: (_stable_hash(seed, name), name))[:file_count]
    )

    destination = output.expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite public position seed output: {output}")
    parent = destination.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
    try:
        all_rows: list[dict[str, object]] = []
        selected_ranges: list[dict[str, object]] = []
        within_range_duplicates = 0
        for filename in selected_names:
            file_url = (
                f"https://huggingface.co/datasets/{repository_path}/resolve/{revision}/"
                f"{quote(filename, safe='/')}"
            )
            probe = _fetch_range(
                file_url,
                start=0,
                end=PSV_RECORD_BYTES - 1,
                timeout_seconds=timeout_seconds,
            )
            if probe.total_bytes % PSV_RECORD_BYTES:
                raise ValueError(f"remote PSV file is not record aligned: {filename}")
            total_records = probe.total_bytes // PSV_RECORD_BYTES
            if records_per_file > total_records:
                raise ValueError(
                    f"requested {records_per_file} records from {filename} with "
                    f"only {total_records}"
                )
            maximum_start = total_records - records_per_file
            offset_digest = _stable_hash(seed, f"{filename}\0record-offset")
            first_record = int(offset_digest, 16) % (maximum_start + 1)
            start = first_record * PSV_RECORD_BYTES
            end = start + records_per_file * PSV_RECORD_BYTES - 1
            downloaded = _fetch_range(
                file_url,
                start=start,
                end=end,
                timeout_seconds=timeout_seconds,
            )
            if downloaded.total_bytes != probe.total_bytes:
                raise ValueError(f"remote PSV size changed during sampling: {filename}")
            rows, duplicates = _decode_seed_rows(
                downloaded.payload,
                corpus_id=corpus_id,
                revision=revision,
                filename=filename,
                first_record=first_record,
                move_field_contract=move_field_contract,
            )
            within_range_duplicates += duplicates
            all_rows.extend(rows)
            selected_ranges.append(
                {
                    "filename": filename,
                    "remote_total_bytes": downloaded.total_bytes,
                    "remote_total_records": total_records,
                    "first_record": first_record,
                    "records_requested": records_per_file,
                    "byte_range": [start, end],
                    "payload_bytes": len(downloaded.payload),
                    "payload_sha256": hashlib.sha256(downloaded.payload).hexdigest(),
                    "etag": downloaded.etag,
                    "resolve_url": file_url,
                }
            )

        deduplicated: dict[str, dict[str, object]] = {}
        cross_range_duplicates = 0
        for row in all_rows:
            key = cast(str, row["normalized_sfen"])
            if key in deduplicated:
                cross_range_duplicates += 1
                continue
            deduplicated[key] = row
        final_rows = tuple(
            sorted(
                deduplicated.values(),
                key=lambda row: (_stable_hash(seed, cast(str, row["normalized_sfen"])),),
            )
        )
        positions_path = stage / "positions.jsonl"
        _write_new(positions_path, _jsonl_bytes(final_rows))
        receipt: dict[str, object] = {
            "schema": PUBLIC_PSV_SEED_SCHEMA,
            "status": "complete",
            "corpus": lineage.to_dict(),
            "api": {
                "url": api_url,
                "payload_sha256": hashlib.sha256(api_raw).hexdigest(),
                "payload_bytes": len(api_raw),
            },
            "sampling": {
                "seed": seed,
                "file_count": file_count,
                "records_per_file": records_per_file,
                "selected_ranges": selected_ranges,
            },
            "output": {
                "file": positions_path.name,
                "sha256": _sha256_file(positions_path),
                "bytes": positions_path.stat().st_size,
                "position_seeds": len(final_rows),
                "source_records_downloaded": file_count * records_per_file,
                "within_range_duplicates_removed": within_range_duplicates,
                "cross_range_duplicates_removed": cross_range_duplicates,
            },
            "contract": {
                "source_rights_scope": scope.value,
                "license_verified": lineage.license_id,
                "operator_acknowledged_user_attested_local_reuse": (
                    scope is CorpusSeedSamplingScope.USER_ATTESTED_LOCAL_ONLY
                    and allow_user_attested_local_only
                ),
                "source_and_derived_artifacts_local_only": (
                    scope is CorpusSeedSamplingScope.USER_ATTESTED_LOCAL_ONLY
                ),
                "source_move_score_and_result_are_training_labels": False,
                "position_seed_only": True,
                "qsearch_or_root_reanalysis_required": True,
                "current_three_teacher_reanalysis_required": True,
                "optimizer_input_allowed": False,
                "heldout_or_promotion_evidence_allowed": False,
                "source_game_lineage_available": False,
            },
            "promotion_blockers": [
                "legacy_depth9_labels_discarded",
                "current_three_teacher_score_matrix_missing",
                "source_game_lineage_unavailable",
                "history_transcript_unavailable",
            ]
            + (
                ["source_rights_are_user_attested_local_only"]
                if scope is CorpusSeedSamplingScope.USER_ATTESTED_LOCAL_ONLY
                else []
            ),
        }
        _write_new(stage / "receipt.json", _json_bytes(receipt))
        _fsync_directory(stage)
        os.rename(stage, destination)
        _fsync_directory(parent)
        return receipt
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
