"""Create exact scalar labels only after rescoring converted qsearch leaves.

``qsearch_psv`` deliberately moves a PackedSfenValue record without recomputing
its score.  This module verifies that ineligible conversion receipt, decodes
every resulting board, discards every inherited score, and invokes one pinned
canonical USI scorer with a fixed node budget and ``MultiPV=1``.  Only exact
centipawn results become :class:`ScalarValueLabel` rows.  Mate, bounds, short
budgets, and malformed/absent scalar results remain in a separate unresolved
queue with lossless search evidence.

PackedSfenValue has no repetition history, so every emitted label explicitly
uses board-only semantics.  This is not a replacement for history-aware rule
adjudication.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
from rsshogi.core import Board
from rsshogi.numpy import PackedSfenValue

from .artifact_provenance import sha256_file
from .domain import TeacherScoreBound, TeacherScoreKind, TeacherVariation
from .ensemble import normalized_sfen
from .incremental_value_replay import ScalarLabelKind, ScalarValueLabel
from .model_rights import model_rights
from .nnue_training import probe_value_only_psv
from .production_score_matrix import ExternalUsiProductionScorer
from .qsearch_leaf import QSEARCH_LEAF_RECEIPT_SCHEMA

QSEARCH_LEAF_RESCORE_RECEIPT_SCHEMA = "meteo-qsearch-leaf-rescore-receipt-v1"
QSEARCH_LEAF_SEARCH_RECEIPT_SCHEMA = "meteo-qsearch-leaf-search-receipt-v1"
QSEARCH_LEAF_UNRESOLVED_SCHEMA = "meteo-qsearch-leaf-unresolved-v1"
BOARD_ONLY_CONTEXT = "board_only_sfen_no_repetition_or_perpetual_check_history"


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    normalized: dict[str, str] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        folded = key.casefold()
        if folded in normalized:
            raise ValueError(
                f"case-insensitive JSON object key collision: {normalized[folded]!r} and {key!r}"
            )
        normalized[folded] = key
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value: object = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value), raw


def _regular_file(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    _reject_symlink_components(absolute, label=label)
    if not absolute.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {absolute}")
    return absolute


def _regular_directory(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    _reject_symlink_components(absolute, label=label)
    if not absolute.is_dir():
        raise ValueError(f"{label} must be a regular non-symlink directory: {absolute}")
    return absolute


def _reject_symlink_components(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink: {current}")


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_create_only_directory(stage: Path, destination: Path) -> None:
    """Reserve a new directory and hard-link complete staged files into it."""

    try:
        destination.mkdir(mode=0o700)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite qsearch rescore artifact: {destination}"
        ) from error
    try:
        for name in ("labels.jsonl", "searches.jsonl", "unresolved.jsonl", "receipt.json"):
            os.link(stage / name, destination / name)
        descriptor = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        shutil.rmtree(destination)
        raise
    shutil.rmtree(stage)


def _validate_qsearch_conversion(directory: Path) -> tuple[Path, dict[str, Any], str, int]:
    root = _regular_directory(directory, label="qsearch conversion directory")
    receipt_path = _regular_file(root / "receipt.json", label="qsearch conversion receipt")
    receipt, receipt_bytes = _strict_json(receipt_path, label="qsearch conversion receipt")
    expected_fields = {
        "schema",
        "source",
        "engine",
        "output",
        "transcript",
        "score_was_recomputed_at_leaf",
        "eligible_for_value_training",
        "required_next_stage",
        "complete",
    }
    if set(receipt) != expected_fields:
        raise ValueError("qsearch conversion receipt fields do not match schema v1")
    if receipt["schema"] != QSEARCH_LEAF_RECEIPT_SCHEMA or receipt["complete"] is not True:
        raise ValueError("qsearch conversion receipt is incomplete or has the wrong schema")
    if receipt["score_was_recomputed_at_leaf"] is not False:
        raise ValueError("qsearch conversion receipt unexpectedly claims score recomputation")
    if receipt["eligible_for_value_training"] is not False:
        raise ValueError("qsearch conversion must still be ineligible before anchor rescoring")
    if receipt["required_next_stage"] != "single_anchor_rescore_every_leaf":
        raise ValueError("qsearch conversion does not require the expected rescore stage")
    output = receipt["output"]
    if not isinstance(output, dict) or set(output) != {"file", "bytes", "sha256", "records"}:
        raise ValueError("qsearch conversion output identity is malformed")
    if output["file"] != "leaves.psv":
        raise ValueError("qsearch conversion output must be the local leaves.psv artifact")
    leaf = _regular_file(root / "leaves.psv", label="qsearch leaf PSV")
    expected_sha = output["sha256"]
    if not isinstance(expected_sha, str) or sha256_file(leaf) != expected_sha:
        raise ValueError("qsearch leaf PSV SHA-256 does not match its conversion receipt")
    expected_bytes = output["bytes"]
    expected_records = output["records"]
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes != leaf.stat().st_size
        or isinstance(expected_records, bool)
        or not isinstance(expected_records, int)
        or expected_records < 1
    ):
        raise ValueError("qsearch leaf PSV byte/record counts do not match its receipt")
    probe = probe_value_only_psv(leaf)
    if probe["records"] != expected_records:
        raise ValueError("qsearch leaf PSV probe record count does not match its receipt")
    return leaf, receipt, hashlib.sha256(receipt_bytes).hexdigest(), expected_records


def _variation_payload(variation: TeacherVariation | None) -> dict[str, object] | None:
    return None if variation is None else variation.to_dict()


def _transcript_sha256(sent: tuple[str, ...], lines: tuple[str, ...]) -> str:
    return _json_sha256({"sent_commands": sent, "transcript_lines": lines})


def _reason_codes(
    *,
    variation: TeacherVariation | None,
    bestmove: str,
    reported_nodes: int | None,
    requested_nodes: int,
    sent_commands: tuple[str, ...],
    transcript_lines: tuple[str, ...],
) -> tuple[str, ...]:
    reasons: set[str] = set()
    if reported_nodes is None or reported_nodes < requested_nodes:
        reasons.add("requested_node_budget_not_fulfilled")
    if not sent_commands or not transcript_lines:
        reasons.add("lossless_search_transcript_missing")
    if variation is None:
        reasons.add("one_rank1_variation_not_available")
        return tuple(sorted(reasons))
    if bestmove != variation.move:
        reasons.add("bestmove_does_not_match_rank1_variation")
    if variation.score_kind is TeacherScoreKind.MATE:
        reasons.add("mate_requires_separate_proof")
    elif variation.bound is not TeacherScoreBound.EXACT:
        reasons.add("score_is_bound_not_exact")
    elif variation.score_cp is None or not -31_999 <= variation.score_cp <= 31_999:
        reasons.add("centipawn_score_outside_scalar_label_domain")
    return tuple(sorted(reasons))


def rescore_qsearch_leaves(
    qsearch_directory: Path,
    *,
    anchor: ExternalUsiProductionScorer,
    requested_nodes: int,
    output_directory: Path,
) -> dict[str, object]:
    """Rescore every unique leaf and create exact labels plus unresolved evidence."""

    if isinstance(requested_nodes, bool) or requested_nodes < 1:
        raise ValueError("requested_nodes must be a positive integer")
    if anchor.identity.scorer_id != anchor.engine.policy.policy_id:
        raise ValueError("anchor identity and rights policy do not match")
    if anchor.engine.nodes != requested_nodes:
        raise ValueError("anchor engine and rescore request must use the same fixed node budget")
    if anchor.engine.multipv != 1 or anchor.identity.multipv != 1:
        raise ValueError("qsearch leaf rescoring requires MultiPV=1")
    anchor.engine.policy.require_training_permission()
    model_rights(anchor.identity.scorer_id).teacher_policy(
        allow_limited_local=anchor.engine.policy.training_outputs_local_only
    ).require_training_permission()

    leaf_path, conversion_receipt, conversion_receipt_sha, source_records = (
        _validate_qsearch_conversion(qsearch_directory)
    )
    destination = Path(os.path.abspath(os.fspath(output_directory.expanduser())))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite qsearch rescore artifact: {destination}")
    _reject_symlink_components(destination.parent, label="qsearch rescore output parent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _regular_directory(destination.parent, label="qsearch rescore output parent")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))

    try:
        records: Any = np.memmap(leaf_path, dtype=PackedSfenValue, mode="r")
        positions: dict[str, tuple[Board, list[tuple[int, int, int]]]] = {}
        for index in range(len(records)):
            board = Board()
            board.set_packed_sfen(records[index]["sfen"].tobytes())
            if not board.is_valid():
                raise ValueError(f"qsearch leaf record {index} does not decode to a valid board")
            key = normalized_sfen(board.to_sfen())
            source = (
                index,
                int(records[index]["score"]),
                int(records[index]["game_ply"]),
            )
            if key in positions:
                positions[key][1].append(source)
            else:
                positions[key] = (board, [source])
    except BaseException:
        shutil.rmtree(stage)
        raise

    labels: list[ScalarValueLabel] = []
    search_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []
    try:
        anchor.engine.start()
        startup_hash = _json_sha256(anchor.engine.startup_provenance.to_dict())
        if startup_hash != anchor.identity.startup_transcript_sha256:
            raise ValueError("anchor startup transcript does not match its pinned identity")
        executable = Path(anchor.engine.startup_provenance.resolved_executable)
        if sha256_file(executable) != anchor.identity.engine_sha256:
            raise ValueError("anchor executable bytes do not match its pinned identity")
        for key in sorted(positions):
            board, source_rows = positions[key]
            anchor.engine.new_game()
            analysis = anchor.engine.analyse(board)
            rank_one = tuple(candidate for candidate in analysis.candidates if candidate.rank == 1)
            variation = rank_one[0] if len(rank_one) == 1 else None
            reasons = _reason_codes(
                variation=variation,
                bestmove=analysis.bestmove,
                reported_nodes=analysis.nodes,
                requested_nodes=requested_nodes,
                sent_commands=analysis.sent_commands,
                transcript_lines=analysis.transcript_lines,
            )
            sent_commands = analysis.sent_commands
            transcript_lines = analysis.transcript_lines
            transcript_sha256 = _transcript_sha256(sent_commands, transcript_lines)
            if (
                f"position sfen {board.to_sfen()}" not in sent_commands
                or f"go nodes {requested_nodes}" not in sent_commands
            ):
                reasons = tuple(sorted({*reasons, "exact_fixed_node_commands_missing"}))
            unsigned_search: dict[str, object] = {
                "schema": QSEARCH_LEAF_SEARCH_RECEIPT_SCHEMA,
                "position_context": BOARD_ONLY_CONTEXT,
                "normalized_sfen": key,
                "sfen": board.to_sfen(),
                "source_record_indices": [row[0] for row in source_rows],
                "old_scores_discarded": [row[1] for row in source_rows],
                "old_score_used_as_target": False,
                "anchor_identity_sha256": anchor.identity.identity_sha256,
                "scorer_id": anchor.identity.scorer_id,
                "requested_nodes": requested_nodes,
                "reported_nodes": analysis.nodes,
                "bestmove": analysis.bestmove,
                "variation": _variation_payload(variation),
                "depth": analysis.depth,
                "seldepth": analysis.seldepth,
                "time_ms": analysis.time_ms,
                "nps": analysis.nps,
                "sent_commands": sent_commands,
                "transcript_lines": transcript_lines,
                "transcript_sha256": transcript_sha256,
                "resolution": "exact_cp_label" if not reasons else "unresolved",
                "unresolved_reasons": reasons,
            }
            search_sha = _json_sha256(unsigned_search)
            search_row = dict(unsigned_search)
            search_row["search_receipt_sha256"] = search_sha
            search_rows.append(search_row)
            if reasons:
                unresolved = dict(search_row)
                unresolved["schema"] = QSEARCH_LEAF_UNRESOLVED_SCHEMA
                unresolved_rows.append(unresolved)
                continue
            assert variation is not None and variation.score_cp is not None
            labels.append(
                ScalarValueLabel(
                    sfen=board.to_sfen(),
                    score_cp=variation.score_cp,
                    game_ply=source_rows[0][2],
                    game_result=0,
                    game_result_known=False,
                    kind=ScalarLabelKind.ANCHOR_SEARCH_EXACT,
                    scorer_id=anchor.identity.scorer_id,
                    requested_nodes=requested_nodes,
                    source_receipt_sha256=search_sha,
                    qsearch_leaf_rescored=True,
                    history_dependent=False,
                )
            )

        labels_bytes = b"".join(_json_bytes(label.to_input_row()) for label in labels)
        searches_bytes = b"".join(_json_bytes(row) for row in search_rows)
        unresolved_bytes = b"".join(_json_bytes(row) for row in unresolved_rows)
        _write_new(stage / "labels.jsonl", labels_bytes)
        _write_new(stage / "searches.jsonl", searches_bytes)
        _write_new(stage / "unresolved.jsonl", unresolved_bytes)
        reason_counts: dict[str, int] = {}
        for row in unresolved_rows:
            raw_reasons = cast(tuple[str, ...], row["unresolved_reasons"])
            for reason in raw_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        receipt: dict[str, object] = {
            "schema": QSEARCH_LEAF_RESCORE_RECEIPT_SCHEMA,
            "qsearch_conversion": {
                "receipt_sha256": conversion_receipt_sha,
                "leaf_psv_sha256": sha256_file(leaf_path),
                "records": source_records,
                "score_was_recomputed_at_leaf": conversion_receipt["score_was_recomputed_at_leaf"],
                "eligible_for_value_training_before_rescore": conversion_receipt[
                    "eligible_for_value_training"
                ],
            },
            "anchor": asdict(anchor.identity),
            "requested_nodes": requested_nodes,
            "multipv": 1,
            "position_context": BOARD_ONLY_CONTEXT,
            "source_record_count": source_records,
            "unique_board_count": len(positions),
            "duplicate_source_record_count": source_records - len(positions),
            "labels": {
                "file": "labels.jsonl",
                "sha256": hashlib.sha256(labels_bytes).hexdigest(),
                "rows": len(labels),
                "qsearch_leaf_rescored": True,
                "exact_centipawn_only": True,
            },
            "search_receipts": {
                "file": "searches.jsonl",
                "sha256": hashlib.sha256(searches_bytes).hexdigest(),
                "rows": len(search_rows),
            },
            "unresolved_queue": {
                "file": "unresolved.jsonl",
                "sha256": hashlib.sha256(unresolved_bytes).hexdigest(),
                "rows": len(unresolved_rows),
                "reason_counts": dict(sorted(reason_counts.items())),
                "mate_bound_or_short_budget_never_written_as_point_label": True,
            },
            "old_score_used_as_target": False,
            "history_dependent_labels_emitted": False,
            "local_only": True,
            "publication_allowed": False,
            "complete": True,
        }
        _write_new(stage / "receipt.json", _json_bytes(receipt))
        descriptor = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _publish_create_only_directory(stage, destination)
        return receipt
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
