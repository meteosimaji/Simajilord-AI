"""Build a receipt-bound, non-promotable unanimous-teacher smoke corpus.

The online committee benchmark was designed as a strength probe, not as a
training-data producer.  This module therefore ignores every move actually
played by the committee.  It admits a position only when the three canonical
teachers independently have the same unique best move at the deepest root
budget, that choice is stable across root budgets, and fresh opponent-reply
backups are exact and preserve the same unique best move for every teacher.

The resulting corpus intentionally remains a tiny pipeline smoke test.  It is
not a production label source: only the committee candidate union is covered,
history is not serialized in the source evidence, and the current benchmark
uses one opening pair.  Receipts preserve those blockers rather than silently
turning the corpus into held-out or promotion evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from rsshogi.core import Board

from .anchor_labeling import AnchorCandidateScore, resolve_anchor_target
from .distillation_targets import CANONICAL_SCORER_IDS
from .domain import GameRecord, PositionSample, TeacherScoreBound, Termination
from .ensemble import normalized_sfen
from .model_rights import model_rights

UNANIMOUS_SMOKE_SCHEMA = "meteo-strict-unanimous-smoke-v1"
UNANIMOUS_LINEAGE_SCHEMA = "meteo-strict-unanimous-lineage-v1"
UNANIMOUS_SPLIT_RECEIPT_SCHEMA = "meteo-strict-unanimous-split-receipt-v1"
UNANIMOUS_SCORE_RECEIPT_SCHEMA = "meteo-strict-unanimous-score-matrix-receipt-v1"
UNANIMOUS_CALIBRATION_RECEIPT_SCHEMA = (
    "meteo-strict-unanimous-calibration-receipt-v1"
)
COMMITTEE_CONFIG_SCHEMA = "meteo-three-teacher-committee-config-v1"
COMMITTEE_BENCHMARK_SCHEMA = "meteo-three-teacher-committee-benchmark-v1"
COMMITTEE_PAIR_SCHEMA = "meteo-three-teacher-committee-pair-v1"
COMMITTEE_ID = "meteo-three-teacher-reply-reanalysed-interval-committee-v3"

_TIE_TOLERANCE = 1e-12
_SPLITS = ("train", "calibration", "heldout")


def _reject_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_bytes(raw: bytes, *, label: str) -> object:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    return cast(dict[str, Any], value)


def _sequence(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _regular_file(path: Path, *, label: str) -> Path:
    supplied = path.expanduser()
    if supplied.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    resolved = supplied.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved


def _read_strict_json(path: Path, *, label: str) -> tuple[object, bytes]:
    resolved = _regular_file(path, label=label)
    raw = resolved.read_bytes()
    return _strict_json_bytes(raw, label=label), raw


def _finite_q(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a numeric Q value")
    number = float(value)
    if not math.isfinite(number) or not -1.0 <= number <= 1.0:
        raise ValueError(f"{label} must be finite in [-1, 1]")
    return number


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: object, *, label: str) -> int:
    result = _nonnegative_int(value, label=label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _unique_tokens(value: object, *, label: str) -> tuple[str, ...]:
    rows = _sequence(value, label=label)
    tokens = tuple(str(item) for item in rows)
    if any(not token or any(character.isspace() for character in token) for token in tokens):
        raise ValueError(f"{label} contains an invalid token")
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"{label} contains duplicate tokens")
    return tokens


def _teacher_ids(value: object, *, label: str) -> tuple[str, ...]:
    identifiers = _unique_tokens(value, label=label)
    if identifiers != CANONICAL_SCORER_IDS:
        raise ValueError(
            f"{label} must equal the canonical teacher order {CANONICAL_SCORER_IDS!r}"
        )
    return identifiers


def _unique_best(values: Mapping[str, float]) -> str | None:
    maximum = max(values.values())
    best = tuple(
        move
        for move, value in values.items()
        if math.isclose(value, maximum, rel_tol=0.0, abs_tol=_TIE_TOLERANCE)
    )
    return best[0] if len(best) == 1 else None


def _parse_exact_root_matrix(
    value: object,
    *,
    candidates: tuple[str, ...],
) -> tuple[dict[str, dict[str, float]] | None, str | None]:
    rows = _sequence(value, label="deep root teacher_intervals")
    if len(rows) != len(CANONICAL_SCORER_IDS):
        raise ValueError("deep root matrix must contain every canonical teacher exactly once")
    result: dict[str, dict[str, float]] = {}
    for index, raw_teacher in enumerate(rows):
        teacher_row = _sequence(raw_teacher, label=f"root teacher row {index}")
        if len(teacher_row) != 2:
            raise ValueError("root teacher row must contain teacher ID and move rows")
        teacher_id = str(teacher_row[0])
        if teacher_id != CANONICAL_SCORER_IDS[index] or teacher_id in result:
            raise ValueError("root matrix teacher IDs are missing, duplicated, or reordered")
        move_rows = _sequence(teacher_row[1], label=f"root matrix {teacher_id}")
        values: dict[str, float] = {}
        all_exact = True
        for raw_move_row in move_rows:
            move_row = _sequence(raw_move_row, label=f"root score row {teacher_id}")
            if len(move_row) != 4:
                raise ValueError("root score row must be [move, lower, upper, bound]")
            move = str(move_row[0])
            if move in values:
                raise ValueError(f"duplicate root score for {teacher_id}:{move}")
            lower = _finite_q(move_row[1], label=f"root lower {teacher_id}:{move}")
            upper = _finite_q(move_row[2], label=f"root upper {teacher_id}:{move}")
            if lower > upper:
                raise ValueError(f"inverted root interval for {teacher_id}:{move}")
            bound = str(move_row[3])
            if bound not in {item.value for item in TeacherScoreBound}:
                raise ValueError(f"unknown root bound {bound!r}")
            all_exact = all_exact and bound == TeacherScoreBound.EXACT.value and lower == upper
            values[move] = lower
        if set(values) != set(candidates):
            raise ValueError(f"root matrix for {teacher_id} does not cover the candidate union")
        result[teacher_id] = values
        if not all_exact:
            return None, "root_non_exact"
    return result, None


def _parse_exact_reply_matrix(
    reply: Mapping[str, object],
    *,
    root_candidates: tuple[str, ...],
) -> tuple[dict[str, dict[str, float]] | None, str | None]:
    reanalysed = _unique_tokens(
        reply.get("reanalysed_candidates"), label="reply reanalysed_candidates"
    )
    if not reanalysed or not set(reanalysed).issubset(root_candidates):
        raise ValueError("reply reanalysis candidates must be a non-empty root subset")
    raw_candidates = _sequence(reply.get("candidates"), label="reply candidates")
    if len(raw_candidates) != len(reanalysed):
        raise ValueError("reply candidate evidence count does not match its candidate list")
    backed: dict[str, dict[str, float]] = {
        teacher_id: {} for teacher_id in CANONICAL_SCORER_IDS
    }
    observed_moves: list[str] = []
    for candidate_index, raw_candidate in enumerate(raw_candidates):
        candidate = _mapping(raw_candidate, label=f"reply candidate {candidate_index}")
        move = str(candidate.get("move", ""))
        if move not in reanalysed or move in observed_moves:
            raise ValueError("reply candidate move is outside, or duplicated in, its declared set")
        observed_moves.append(move)
        terminal = candidate.get("terminal_root_value")
        if terminal is not None:
            _finite_q(terminal, label=f"terminal root value {move}")
            return None, "terminal_or_mate_special_case"
        if _nonnegative_int(
            candidate.get("persistent_bounds"), label=f"reply persistent bounds {move}"
        ) != 0:
            return None, "reply_raw_non_exact"
        raw_backed = _sequence(
            candidate.get("teacher_backed_intervals"),
            label=f"reply backed intervals {move}",
        )
        if len(raw_backed) != len(CANONICAL_SCORER_IDS):
            raise ValueError("reply backed matrix must contain every canonical teacher")
        for teacher_index, raw_row in enumerate(raw_backed):
            row = _sequence(raw_row, label=f"reply backed row {move}:{teacher_index}")
            if len(row) != 3:
                raise ValueError("reply backed row must be [teacher, lower, upper]")
            teacher_id = str(row[0])
            if teacher_id != CANONICAL_SCORER_IDS[teacher_index]:
                raise ValueError("reply backed teacher IDs are missing, duplicated, or reordered")
            lower = _finite_q(row[1], label=f"reply lower {teacher_id}:{move}")
            upper = _finite_q(row[2], label=f"reply upper {teacher_id}:{move}")
            if lower > upper:
                raise ValueError(f"inverted reply interval for {teacher_id}:{move}")
            if lower != upper:
                return None, "reply_backed_non_exact"
            if move in backed[teacher_id]:
                raise ValueError(f"duplicate reply-backed score for {teacher_id}:{move}")
            backed[teacher_id][move] = lower
    if tuple(observed_moves) != reanalysed:
        raise ValueError("reply candidate evidence is not in the declared deterministic order")
    if any(set(values) != set(reanalysed) for values in backed.values()):
        raise ValueError("reply-backed matrix is incomplete")
    return backed, None


@dataclass(frozen=True, slots=True)
class _StrictPosition:
    opponent_id: str
    evidence_sha256: str
    evidence_bytes: int
    evidence_row_index: int
    game_index: int
    opening_index: int
    committee_decision_index: int
    sfen: str
    normalized_sfen: str
    best_move: str
    reply_requested_nodes: int
    root_values: dict[str, dict[str, float]]
    reply_backed_values: dict[str, dict[str, float]]

    @property
    def game_group(self) -> str:
        return f"{self.opponent_id}:game-{self.game_index}"

    @property
    def opening_group(self) -> str:
        return f"opening-{self.opening_index}"


def _strict_position(
    raw: object,
    *,
    opponent_id: str,
    evidence_sha256: str,
    evidence_bytes: int,
    evidence_row_index: int,
) -> tuple[_StrictPosition | None, str | None]:
    row = _mapping(raw, label=f"evidence row {evidence_row_index}")
    sfen = str(row.get("target_sfen", ""))
    board = Board(sfen)
    if not board.is_valid():
        raise ValueError(f"evidence row {evidence_row_index} has an invalid SFEN")
    candidates = _unique_tokens(row.get("candidate_moves"), label="candidate_moves")
    if not candidates:
        raise ValueError("committee candidate union must not be empty")
    legal_moves = {move.to_usi() for move in board.legal_moves()}
    if not set(candidates).issubset(legal_moves):
        raise ValueError("committee candidate union contains an illegal root move")

    proposal_rows = _sequence(row.get("teacher_proposals"), label="teacher_proposals")
    if len(proposal_rows) != len(CANONICAL_SCORER_IDS):
        raise ValueError("teacher proposals must contain all canonical teachers")
    proposed_union: set[str] = set()
    for index, raw_proposal in enumerate(proposal_rows):
        proposal = _sequence(raw_proposal, label=f"teacher proposal {index}")
        if len(proposal) != 2 or str(proposal[0]) != CANONICAL_SCORER_IDS[index]:
            raise ValueError("teacher proposal IDs are missing, duplicated, or reordered")
        moves = _unique_tokens(proposal[1], label=f"teacher proposal moves {index}")
        if not set(moves).issubset(legal_moves):
            raise ValueError("teacher proposal contains an illegal move")
        proposed_union.update(moves)
    if proposed_union != set(candidates):
        raise ValueError("candidate_moves is not exactly the teacher proposal union")

    budgets = _sequence(row.get("budgets"), label="committee budgets")
    if len(budgets) < 2:
        raise ValueError("unanimous smoke evidence requires at least two root budgets")
    requested_budgets: list[int] = []
    for index, raw_budget in enumerate(budgets):
        budget = _mapping(raw_budget, label=f"committee budget {index}")
        requested_budgets.append(
            _positive_int(
                budget.get("requested_nodes_per_candidate"),
                label=f"requested nodes for root budget {index}",
            )
        )
    if requested_budgets != sorted(set(requested_budgets)):
        raise ValueError("root budgets must be strictly increasing")
    if row.get("budget_choice_stable") is not True:
        return None, "root_budget_choice_unstable"

    deepest = _mapping(budgets[-1], label="deepest root budget")
    root_values, reason = _parse_exact_root_matrix(
        deepest.get("teacher_intervals"), candidates=candidates
    )
    if root_values is None:
        return None, reason
    # The committee receipt stores normalized Q only.  ExternalUsiTeacher maps
    # an engine-reported mate to +/-1, so an endpoint cannot be distinguished
    # from a saturated finite score here.  Keep both out of the ordinary smoke
    # policy path until score kind and mate proof are explicitly receipted.
    if any(
        abs(value) == 1.0
        for teacher_values in root_values.values()
        for value in teacher_values.values()
    ):
        return None, "endpoint_value_or_untyped_mate"
    root_best = tuple(_unique_best(root_values[teacher]) for teacher in CANONICAL_SCORER_IDS)
    if any(move is None for move in root_best):
        return None, "root_top_tie"
    if len(set(root_best)) != 1:
        return None, "root_teacher_disagreement"
    common_root_best = cast(str, root_best[0])
    if any(
        str(_mapping(raw_budget, label="root budget").get("chosen_move", ""))
        != common_root_best
        for raw_budget in budgets
    ):
        return None, "root_budget_choice_unstable"

    reply = _mapping(row.get("reply_reanalysis"), label="reply_reanalysis")
    reply_requested_nodes = _positive_int(
        reply.get("reply_score_nodes_per_teacher_per_reply"),
        label="reply score nodes per teacher per reply",
    )
    backed_values, reason = _parse_exact_reply_matrix(reply, root_candidates=candidates)
    if backed_values is None:
        return None, reason
    if any(
        abs(value) == 1.0
        for teacher_values in backed_values.values()
        for value in teacher_values.values()
    ):
        return None, "endpoint_value_or_untyped_mate"
    reply_best = tuple(
        _unique_best(backed_values[teacher]) for teacher in CANONICAL_SCORER_IDS
    )
    if any(move is None for move in reply_best):
        return None, "reply_top_tie"
    if len(set(reply_best)) != 1:
        return None, "reply_teacher_disagreement"
    common_reply_best = cast(str, reply_best[0])
    if common_reply_best != common_root_best:
        return None, "root_reply_disagreement"
    if (
        str(reply.get("root_choice", "")) != common_root_best
        or str(reply.get("chosen_move", "")) != common_root_best
        or str(row.get("chosen_move", "")) != common_root_best
    ):
        return None, "derived_choice_mismatch"

    return (
        _StrictPosition(
            opponent_id=opponent_id,
            evidence_sha256=evidence_sha256,
            evidence_bytes=evidence_bytes,
            evidence_row_index=evidence_row_index,
            game_index=_nonnegative_int(row.get("game_index"), label="game_index"),
            opening_index=_nonnegative_int(row.get("opening_index"), label="opening_index"),
            committee_decision_index=_nonnegative_int(
                row.get("committee_decision_index"), label="committee_decision_index"
            ),
            sfen=board.to_sfen(),
            normalized_sfen=normalized_sfen(sfen),
            best_move=common_root_best,
            reply_requested_nodes=reply_requested_nodes,
            root_values=root_values,
            reply_backed_values=backed_values,
        ),
        None,
    )


def _stable_hash(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _deduplicate_positions(
    positions: Sequence[_StrictPosition],
    *,
    seed: str,
) -> tuple[list[_StrictPosition], int, int]:
    by_key: dict[str, list[_StrictPosition]] = defaultdict(list)
    for position in positions:
        by_key[position.normalized_sfen].append(position)
    selected: list[_StrictPosition] = []
    duplicate_rows = 0
    cross_game_keys = 0
    for _key, rows in by_key.items():
        if len({row.game_group for row in rows}) > 1:
            cross_game_keys += 1
        ordered = sorted(
            rows,
            key=lambda row: (
                _stable_hash(
                    seed,
                    f"{row.opponent_id}:{row.game_index}:{row.committee_decision_index}",
                ),
                row.opponent_id,
                row.game_index,
                row.committee_decision_index,
            ),
        )
        selected.append(ordered[0])
        duplicate_rows += len(ordered) - 1
    return selected, duplicate_rows, cross_game_keys


def _assign_splits(
    positions: Sequence[_StrictPosition],
    *,
    requested: Mapping[str, int],
    seed: str,
) -> dict[str, list[_StrictPosition]]:
    for split in _SPLITS:
        value = requested[split]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{split} count must be a non-negative integer")
    groups: dict[str, list[_StrictPosition]] = defaultdict(list)
    for position in positions:
        groups[position.game_group].append(position)
    ordered_groups = sorted(groups, key=lambda group: (_stable_hash(seed, group), group))
    for group in groups:
        groups[group].sort(
            key=lambda row: (
                _stable_hash(seed, row.normalized_sfen),
                row.normalized_sfen,
            )
        )

    assigned_groups: set[str] = set()
    result: dict[str, list[_StrictPosition]] = {split: [] for split in _SPLITS}
    # Reserve independent full-game source groups for the two diagnostic splits
    # first.  Unused rows from a reserved group are intentionally discarded.
    for split in ("heldout", "calibration"):
        count = requested[split]
        if count == 0:
            continue
        candidate_group = next(
            (
                candidate
                for candidate in ordered_groups
                if candidate not in assigned_groups and len(groups[candidate]) >= count
            ),
            None,
        )
        if candidate_group is None:
            raise ValueError(
                f"not enough strict positions in one unused game group for {split}={count}"
            )
        assigned_groups.add(candidate_group)
        result[split] = groups[candidate_group][:count]

    remaining = requested["train"]
    for group in ordered_groups:
        if group in assigned_groups or remaining == 0:
            continue
        take = min(remaining, len(groups[group]))
        result["train"].extend(groups[group][:take])
        assigned_groups.add(group)
        remaining -= take
    if remaining:
        raise ValueError(
            "not enough strict positions outside calibration/heldout game groups: "
            f"missing {remaining} train positions"
        )

    observed_keys: set[str] = set()
    observed_groups: set[str] = set()
    for split in _SPLITS:
        rows = result[split]
        if len(rows) != requested[split]:
            raise AssertionError("split assignment produced the wrong number of positions")
        keys = {row.normalized_sfen for row in rows}
        split_groups = {row.game_group for row in rows}
        if len(keys) != len(rows) or observed_keys & keys:
            raise AssertionError("normalized board position leaked across smoke splits")
        if observed_groups & split_groups:
            raise AssertionError("source game group leaked across smoke splits")
        observed_keys.update(keys)
        observed_groups.update(split_groups)
    return result


def _sample_from_position(
    position: _StrictPosition,
    *,
    scorer_id: str,
    policy_temperature: float,
) -> PositionSample:
    values = position.reply_backed_values[scorer_id]
    target = resolve_anchor_target(
        scorer_id,
        tuple(
            AnchorCandidateScore(
                move=move,
                q_value=value,
                bound=TeacherScoreBound.EXACT,
                requested_nodes=position.reply_requested_nodes,
            )
            for move, value in sorted(values.items())
        ),
        policy_temperature=policy_temperature,
    )
    if target.chosen_move != position.best_move or target.value_interval is None:
        raise AssertionError("unanimous scorer target changed the verified unique best move")
    value = target.value_interval[0]
    board = Board(position.sfen)
    move_number = int(position.sfen.split()[3])
    return PositionSample(
        sfen=position.sfen,
        ply=max(0, move_number - 1),
        turn=board.turn.value,
        policy=dict(target.policy),
        root_value=value,
        value_target=value,
        teacher_policy=dict(target.policy),
        teacher_value=value,
        chosen_move=position.best_move,
        teacher_best_move=position.best_move,
        teacher_source=scorer_id,
        teacher_context=(
            "strict three-teacher unanimous root and reply-backed exact smoke target; "
            "partial candidate union; actual benchmark move not used"
        ),
        teacher_move_values=dict(values),
        teacher_policy_temperature=policy_temperature,
    )


def _game_from_sample(sample: PositionSample) -> GameRecord:
    # The wrapper is only the existing JSONL transport.  AGREED_DRAW prevents
    # the replay loader from dropping it as an incomplete MAX_PLIES game; no
    # game result contributes to training because teacher mixes are set to one.
    return GameRecord(
        initial_sfen=sample.sfen,
        moves=(),
        samples=(sample,),
        winner=None,
        termination=Termination.AGREED_DRAW,
    )


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
    ).encode("utf-8")


def _jsonl_bytes(games: Sequence[GameRecord]) -> bytes:
    return b"".join(
        (
            json.dumps(
                {"schema": 1, "game": game.to_dict()},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for game in games
    )


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_record(path: Path) -> dict[str, object]:
    return {
        "name": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _lineage_sidecar(
    *,
    split: str,
    replay: Path,
    scorer_id: str,
    source_receipts: Sequence[dict[str, object]],
    policy_temperature: float,
) -> dict[str, object]:
    return {
        "schema": UNANIMOUS_LINEAGE_SCHEMA,
        "rights_mode": "limited_local",
        "local_only_user_authorized": True,
        "publication_allowed": False,
        "promotable_checkpoint": False,
        "split": split,
        "replay": _file_record(replay),
        "teachers": [
            {
                "role": "single_scorer" if teacher_id == scorer_id else "proposer_challenger",
                "rights": model_rights(teacher_id).to_dict(),
            }
            for teacher_id in CANONICAL_SCORER_IDS
        ],
        "derivation": {
            "mode": "strict_unanimous_multi_proposer_single_scorer_smoke",
            "scorer_id": scorer_id,
            "candidate_sources": list(CANONICAL_SCORER_IDS),
            "cross_teacher_value_average": False,
            "majority_vote": False,
            "actual_benchmark_moves_used_as_labels": False,
            "root_requirement": "all teachers exact, unique top-1, unanimous, budget-stable",
            "reply_requirement": (
                "all child searches exact, all backed values exact, unique top-1 unanimous, "
                "same as root"
            ),
            "policy": "softmax of scorer reply-backed exact Q over reanalysed candidates",
            "policy_temperature_q_units": policy_temperature,
            "terminal_or_mate_special_cases_included": False,
        },
        "source_receipts": list(source_receipts),
    }


def _selected_receipt_row(
    split: str,
    position: _StrictPosition,
    *,
    scorer_id: str,
) -> dict[str, object]:
    return {
        "split": split,
        "source": {
            "opponent_id": position.opponent_id,
            "evidence_sha256": position.evidence_sha256,
            "evidence_bytes": position.evidence_bytes,
            "evidence_row_index": position.evidence_row_index,
            "game_index": position.game_index,
            "opening_index": position.opening_index,
            "committee_decision_index": position.committee_decision_index,
        },
        "sfen_sha256": hashlib.sha256(position.sfen.encode("utf-8")).hexdigest(),
        "normalized_sfen": position.normalized_sfen,
        "best_move": position.best_move,
        "scorer_id": scorer_id,
        "reply_requested_nodes_per_teacher_per_reply": position.reply_requested_nodes,
        "root_teacher_values": position.root_values,
        "reply_backed_teacher_values": position.reply_backed_values,
        "scorer_reply_backed_values": position.reply_backed_values[scorer_id],
    }


def build_unanimous_smoke_corpus(
    benchmark_root: Path,
    output: Path,
    *,
    scorer_id: str = "soujou-tsec7-paid",
    train_count: int = 20,
    calibration_count: int = 10,
    heldout_count: int = 10,
    policy_temperature: float = 0.05,
    split_seed: str = "meteo-strict-unanimous-smoke-v1",
) -> dict[str, object]:
    """Materialize a create-only smoke corpus from a complete committee run."""

    if scorer_id not in CANONICAL_SCORER_IDS:
        raise ValueError(f"scorer_id must be one of {CANONICAL_SCORER_IDS!r}")
    if not math.isfinite(policy_temperature) or policy_temperature <= 0.0:
        raise ValueError("policy temperature must be finite and positive")
    if not split_seed:
        raise ValueError("split seed must not be empty")
    requested = {
        "train": train_count,
        "calibration": calibration_count,
        "heldout": heldout_count,
    }
    for split, count in requested.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{split} count must be a non-negative integer")
    if sum(requested.values()) == 0:
        raise ValueError("at least one smoke position must be requested")

    supplied_root = benchmark_root.expanduser()
    if supplied_root.is_symlink():
        raise ValueError("benchmark root must not be a symlink")
    root = supplied_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    target = output.expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite unanimous smoke corpus: {target}")
    if root == target or root in target.parents:
        raise ValueError("unanimous smoke output must not be inside its immutable benchmark input")

    config_value, config_raw = _read_strict_json(root / "config.json", label="committee config")
    config = _mapping(config_value, label="committee config")
    if config.get("schema") != COMMITTEE_CONFIG_SCHEMA:
        raise ValueError("unsupported committee config schema")
    contract = _mapping(config.get("contract"), label="committee config contract")
    if contract.get("committee_id") != COMMITTEE_ID:
        raise ValueError("committee config has the wrong decision rule")
    _teacher_ids(contract.get("canonical_teacher_ids"), label="config canonical teachers")
    if contract.get("actual_moves_are_training_labels") is not False:
        raise ValueError("committee benchmark must explicitly forbid actual moves as labels")
    source_budgets = _sequence(
        contract.get("score_budgets_per_teacher_per_candidate"),
        label="config root budgets",
    )
    source_budget_values = tuple(
        _positive_int(value, label="config root budget") for value in source_budgets
    )
    if len(source_budget_values) < 2 or source_budget_values != tuple(
        sorted(set(source_budget_values))
    ):
        raise ValueError("config root budgets must be strictly increasing")
    _positive_int(
        contract.get("reply_score_nodes_per_teacher_per_reply"),
        label="config reply score nodes",
    )

    aggregate_value, aggregate_raw = _read_strict_json(
        root / "report.json", label="committee aggregate report"
    )
    aggregate = _mapping(aggregate_value, label="committee aggregate report")
    if aggregate.get("schema") != COMMITTEE_BENCHMARK_SCHEMA:
        raise ValueError("committee aggregate report is missing or incomplete")
    if aggregate.get("config") != config:
        raise ValueError("committee aggregate report is not bound to the exact config")
    if aggregate.get("all_games_complete") is not True:
        raise ValueError("committee benchmark did not finish every game")
    if aggregate.get("completed_opponents") != len(CANONICAL_SCORER_IDS):
        raise ValueError("committee benchmark did not finish all canonical opponents")
    result_rows = _sequence(aggregate.get("results"), label="aggregate results")
    if {str(_mapping(row, label="aggregate result").get("opponent")) for row in result_rows} != set(
        CANONICAL_SCORER_IDS
    ):
        raise ValueError("aggregate report opponent set differs from the canonical teachers")

    source_receipts: list[dict[str, object]] = [
        {
            "kind": "config",
            "path": str(root / "config.json"),
            "sha256": _sha256_bytes(config_raw),
            "bytes": len(config_raw),
        },
        {
            "kind": "aggregate_report",
            "path": str(root / "report.json"),
            "sha256": _sha256_bytes(aggregate_raw),
            "bytes": len(aggregate_raw),
        },
    ]
    strict_positions: list[_StrictPosition] = []
    exclusions: Counter[str] = Counter()
    total_evidence_rows = 0
    for opponent_id in CANONICAL_SCORER_IDS:
        pair_dir = root / "opponents" / opponent_id
        report_value, report_raw = _read_strict_json(
            pair_dir / "report.json", label=f"committee pair report {opponent_id}"
        )
        report = _mapping(report_value, label=f"committee pair report {opponent_id}")
        if report.get("schema") != COMMITTEE_PAIR_SCHEMA or report.get("opponent") != opponent_id:
            raise ValueError(f"invalid committee pair report for {opponent_id}")
        pair_contract = _mapping(report.get("contract"), label="committee pair contract")
        _teacher_ids(
            pair_contract.get("canonical_teacher_ids"),
            label=f"pair canonical teachers {opponent_id}",
        )
        if pair_contract.get("actual_moves_are_training_labels") is not False:
            raise ValueError("pair report must forbid actual benchmark moves as labels")
        evidence_identity = _mapping(
            report.get("committee_evidence"), label="pair evidence identity"
        )
        evidence_path = _regular_file(
            pair_dir / "committee-evidence.json", label=f"committee evidence {opponent_id}"
        )
        evidence_raw = evidence_path.read_bytes()
        evidence_sha256 = _sha256_bytes(evidence_raw)
        if evidence_identity.get("sha256") != evidence_sha256 or evidence_identity.get(
            "bytes"
        ) != len(evidence_raw):
            raise ValueError(f"pair report does not bind the exact evidence for {opponent_id}")
        evidence_value = _strict_json_bytes(
            evidence_raw, label=f"committee evidence {opponent_id}"
        )
        evidence_rows = _sequence(evidence_value, label=f"committee evidence {opponent_id}")
        if report.get("committee_decisions") != len(evidence_rows):
            raise ValueError("pair report decision count differs from its evidence")
        total_evidence_rows += len(evidence_rows)
        source_receipts.extend(
            (
                {
                    "kind": "pair_report",
                    "opponent_id": opponent_id,
                    "path": str(pair_dir / "report.json"),
                    "sha256": _sha256_bytes(report_raw),
                    "bytes": len(report_raw),
                },
                {
                    "kind": "score_matrix_evidence",
                    "opponent_id": opponent_id,
                    "path": str(evidence_path),
                    "sha256": evidence_sha256,
                    "bytes": len(evidence_raw),
                    "rows": len(evidence_rows),
                },
            )
        )
        for evidence_row_index, raw_row in enumerate(evidence_rows):
            position, reason = _strict_position(
                raw_row,
                opponent_id=opponent_id,
                evidence_sha256=evidence_sha256,
                evidence_bytes=len(evidence_raw),
                evidence_row_index=evidence_row_index,
            )
            if position is None:
                if reason is None:
                    raise AssertionError("excluded evidence row lost its reason")
                exclusions[reason] += 1
            else:
                strict_positions.append(position)

    deduplicated, duplicate_rows, cross_game_duplicate_keys = _deduplicate_positions(
        strict_positions, seed=split_seed
    )
    splits = _assign_splits(deduplicated, requested=requested, seed=split_seed)
    selected = [(split, position) for split in _SPLITS for position in splits[split]]
    selected_keys = {
        split: {position.normalized_sfen for position in splits[split]} for split in _SPLITS
    }
    normalized_overlap = {
        f"{left}_vs_{right}": sorted(selected_keys[left] & selected_keys[right])
        for left_index, left in enumerate(_SPLITS)
        for right in _SPLITS[left_index + 1 :]
    }
    game_groups = {
        split: sorted({position.game_group for position in splits[split]})
        for split in _SPLITS
    }
    opening_groups = {
        split: sorted({position.opening_group for position in splits[split]})
        for split in _SPLITS
    }
    opening_overlap = {
        f"{left}_vs_{right}": sorted(
            set(opening_groups[left]) & set(opening_groups[right])
        )
        for left_index, left in enumerate(_SPLITS)
        for right in _SPLITS[left_index + 1 :]
    }
    blockers = [
        "pipeline_smoke_only",
        "partial_candidate_union_not_all_legal_moves",
        "source_evidence_lacks_history_transcript",
        "source_value_scale_not_serialized_explicitly_in_benchmark_report",
        "mate_and_terminal_special_cases_excluded",
        "insufficient_positions_for_formal_model_selection",
    ]
    if any(opening_overlap.values()):
        blockers.append("opening_root_shared_across_diagnostic_splits")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        for split in _SPLITS:
            samples = [
                _sample_from_position(
                    position,
                    scorer_id=scorer_id,
                    policy_temperature=policy_temperature,
                )
                for position in splits[split]
            ]
            replay_path = temporary / f"{split}.jsonl"
            _write_new(replay_path, _jsonl_bytes([_game_from_sample(sample) for sample in samples]))
            sidecar = _lineage_sidecar(
                split=split,
                replay=replay_path,
                scorer_id=scorer_id,
                source_receipts=source_receipts,
                policy_temperature=policy_temperature,
            )
            _write_new(
                replay_path.with_suffix(replay_path.suffix + ".provenance.json"),
                _json_bytes(sidecar),
            )

        split_receipt: dict[str, object] = {
            "schema": UNANIMOUS_SPLIT_RECEIPT_SCHEMA,
            "seed": split_seed,
            "requested_counts": requested,
            "actual_counts": {split: len(splits[split]) for split in _SPLITS},
            "game_groups": game_groups,
            "game_group_overlap": {
                f"{left}_vs_{right}": sorted(
                    set(game_groups[left]) & set(game_groups[right])
                )
                for left_index, left in enumerate(_SPLITS)
                for right in _SPLITS[left_index + 1 :]
            },
            "opening_groups": opening_groups,
            "opening_group_overlap": opening_overlap,
            "normalized_position_overlap": normalized_overlap,
            "board_only_duplicate_rows_dropped": duplicate_rows,
            "board_keys_seen_in_multiple_source_games": cross_game_duplicate_keys,
            "history_aware_split_verified": False,
            "promotion_and_elo_eligible": False,
            "promotion_blockers": blockers,
        }
        _write_new(temporary / "split-receipt.json", _json_bytes(split_receipt))

        score_receipt: dict[str, object] = {
            "schema": UNANIMOUS_SCORE_RECEIPT_SCHEMA,
            "committee_id": COMMITTEE_ID,
            "source_receipts": source_receipts,
            "total_evidence_rows": total_evidence_rows,
            "strict_rows_before_board_dedup": len(strict_positions),
            "strict_rows_after_board_dedup": len(deduplicated),
            "selected_rows": len(selected),
            "exclusion_counts": dict(sorted(exclusions.items())),
            "selection_contract": {
                "actual_benchmark_moves_used_as_labels": False,
                "all_three_root_values_exact": True,
                "root_unique_best_unanimous": True,
                "root_choice_stable_across_budgets": True,
                "all_reply_child_searches_exact": True,
                "all_reply_backed_values_exact": True,
                "reply_unique_best_unanimous": True,
                "root_and_reply_best_identical": True,
                "terminal_and_mate_special_cases_excluded": True,
                "cross_teacher_value_average": False,
                "single_scorer": scorer_id,
            },
            "positions": [
                _selected_receipt_row(split, position, scorer_id=scorer_id)
                for split, position in selected
            ],
        }
        _write_new(temporary / "score-matrix-receipt.json", _json_bytes(score_receipt))

        calibration_receipt: dict[str, object] = {
            "schema": UNANIMOUS_CALIBRATION_RECEIPT_SCHEMA,
            "mode": "smoke_only_no_cross_teacher_calibration",
            "scorer_id": scorer_id,
            "input_value_units": "normalized Q values exactly as serialized by the benchmark",
            "cross_teacher_values_averaged": False,
            "score_to_q_conversion_reapplied": False,
            "benchmark_builder_omitted_explicit_value_scale_from_its_receipt": True,
            "source_code_default_observed_for_benchmark": {
                "formula": "q=tanh(cp/D)",
                "D": 1200.0,
                "Ponanza_C": 600.0,
                "status": "source-code-derived, not independently serialized per analysis",
            },
            "policy": {
                "formula": "softmax(anchor reply-backed exact Q / temperature)",
                "temperature_q_units": policy_temperature,
                "alpha_beta_reported_nodes_used_as_policy_mass": False,
            },
            "fit_dataset": None,
            "heldout_calibration_claimed": False,
            "production_eligible": False,
            "blockers": blockers,
        }
        _write_new(
            temporary / "calibration-receipt.json", _json_bytes(calibration_receipt)
        )

        artifacts = [
            _file_record(path)
            for path in sorted(temporary.iterdir(), key=lambda item: item.name)
            if path.is_file()
        ]
        if len({str(row["name"]).casefold() for row in artifacts}) != len(artifacts):
            raise ValueError("smoke artifact filenames collide case-insensitively")
        manifest: dict[str, object] = {
            "schema": UNANIMOUS_SMOKE_SCHEMA,
            "status": "complete",
            "purpose": "pipeline_validation_only",
            "promotable_checkpoint": False,
            "production_training_eligible": False,
            "scorer_id": scorer_id,
            "candidate_sources": list(CANONICAL_SCORER_IDS),
            "counts": {split: len(splits[split]) for split in _SPLITS},
            "source_evidence_rows": total_evidence_rows,
            "strict_positions": len(strict_positions),
            "artifacts": artifacts,
            "artifact_count": len(artifacts),
            "total_artifact_bytes": sum(cast(int, row["bytes"]) for row in artifacts),
            "blockers": blockers,
        }
        _write_new(temporary / "manifest.json", _json_bytes(manifest))
        _fsync_directory(temporary)
        temporary.rename(target)
        _fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest
