"""Create-only, provenance-complete deduplication of one teacher replay.

Meteo's current network observes only the board, side to move, and hands.  A
history-aware USI teacher can nevertheless return different targets when the
same observable position is reached through different game prefixes.  This
module preserves those raw searches in the immutable input replay and emits
one equally weighted arithmetic-mean target per observable position.

The representative output sample stays inside exactly one original
``GameRecord``.  Games and histories are never concatenated.  A merged sample
has no ``teacher_variations`` because independent ranked PV lists cannot be
truthfully presented as one ranking; every source occurrence remains
addressable by replay/game/sample index and cryptographic hashes in the
manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

from rsshogi.core import Board, Move

from .artifact_provenance import FileIdentity, InputArtifact, identify_file, identify_input_artifact
from .domain import GameRecord, PositionSample, Termination
from .ensemble import normalized_sfen
from .external_usi import UsiPositionHistory
from .game import replay_and_validate
from .replay import SCHEMA_VERSION, load_games

SINGLE_TEACHER_DEDUP_SCHEMA = "meteo-single-teacher-dedup-v2"
SINGLE_TEACHER_DEDUP_METHOD = (
    "within one declared split, normalize each complete teacher policy and take an "
    "equal-weight arithmetic mean of policy probabilities and signed teacher values "
    "for every canonical SFEN board/turn/hands identity"
)

__all__ = (
    "SINGLE_TEACHER_DEDUP_METHOD",
    "SINGLE_TEACHER_DEDUP_SCHEMA",
    "SingleTeacherDedupBuild",
    "SingleTeacherDedupReport",
    "TeacherDedupOccurrence",
    "TeacherDedupPositionReport",
    "build_single_teacher_dedup",
    "write_single_teacher_dedup",
)


@dataclass(frozen=True, slots=True)
class TeacherDedupOccurrence:
    """A lossless pointer and integrity record for one raw teacher search."""

    input_game_index: int
    input_sample_index: int
    game_id: str
    game_record_sha256: str
    sample_sha256: str
    history_context_sha256: str
    history_move_count: int
    sample_sfen: str
    ply: int
    turn: int
    teacher_source: str
    teacher_context: str | None
    teacher_policy_sha256: str
    teacher_target_sha256: str
    teacher_value: float
    teacher_best_move: str | None
    typed_variation_count: int
    typed_variations_sha256: str | None
    teacher_policy_temperature: float | None
    teacher_value_scale: float | None


@dataclass(frozen=True, slots=True)
class TeacherDedupPositionReport:
    """Conflict diagnostics and occurrence lineage for one emitted position."""

    normalized_sfen: str
    output_sfen: str
    output_sample_sha256: str
    representative_game_index: int
    representative_sample_index: int
    occurrence_count: int
    distinct_input_games: int
    distinct_history_contexts: int
    distinct_teacher_contexts: int
    policy_support_union_size: int
    policy_support_intersection_size: int
    policy_conflict: bool
    value_conflict: bool
    target_conflict: bool
    distinct_policy_targets: int
    distinct_value_targets: int
    generalized_policy_js_divergence_nats: float
    mean_policy_l1_to_average: float
    maximum_policy_l1_to_average: float
    teacher_value: float
    teacher_value_minimum: float
    teacher_value_maximum: float
    teacher_value_width: float
    teacher_value_standard_deviation: float
    averaged_teacher_best_move: str
    occurrences: tuple[TeacherDedupOccurrence, ...]


@dataclass(frozen=True, slots=True)
class TeacherDedupOverlapGuard:
    replay: FileIdentity
    games: int
    samples: int
    normalized_positions: int


@dataclass(frozen=True, slots=True)
class SingleTeacherDedupInputReport:
    artifact: InputArtifact
    games: int
    eligible_games: int
    incomplete_games_skipped: int
    raw_samples: int
    complete_teacher_observations: int
    unlabeled_samples_skipped: int
    unique_normalized_positions: int
    duplicate_observations_collapsed: int
    teacher_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SingleTeacherDedupReport:
    schema: str
    split: str
    method: str
    position_identity: str
    input: SingleTeacherDedupInputReport
    overlap_guards: tuple[TeacherDedupOverlapGuard, ...]
    output_games: int
    output_samples: int
    duplicate_positions: int
    duplicate_observations_collapsed: int
    positions_with_multiple_histories: int
    positions_with_policy_conflicts: int
    positions_with_value_conflicts: int
    positions_with_any_target_conflict: int
    group_size_histogram: dict[str, int]
    maximum_policy_l1_to_average: float
    maximum_teacher_value_width: float
    positions: tuple[TeacherDedupPositionReport, ...]
    invariants: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class SingleTeacherDedupBuild:
    """In-memory child replay plus the exact input identity used to build it."""

    input_replay: Path
    games: tuple[GameRecord, ...]
    report: SingleTeacherDedupReport


@dataclass(frozen=True, slots=True)
class _TeacherObservation:
    game_index: int
    sample_index: int
    game: GameRecord
    sample: PositionSample
    normalized_policy: dict[str, float]
    teacher_value: float
    occurrence: TeacherDedupOccurrence


def _json_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _move(move_usi: str, *, field: str, sfen: str) -> Move:
    try:
        return Move.from_usi(move_usi)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid USI move {move_usi!r} in {field} at {sfen}") from error


def _validate_policy(
    policy: Mapping[str, float],
    *,
    board: Board,
    field: str,
    require_positive_mass: bool,
) -> dict[str, float]:
    accumulated: defaultdict[str, float] = defaultdict(float)
    for move_usi in sorted(policy):
        probability = float(policy[move_usi])
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError(f"{field} contains a negative or non-finite probability")
        move = _move(move_usi, field=field, sfen=board.to_sfen())
        if not board.is_legal_move(move):
            raise ValueError(f"illegal {field} move {move_usi!r} at {board.to_sfen()}")
        if probability > 0.0:
            accumulated[move_usi] += probability
    total = math.fsum(accumulated.values())
    if require_positive_mass and total <= 0.0:
        raise ValueError(f"{field} must contain positive probability mass")
    if total <= 0.0:
        return {}
    return {move: accumulated[move] / total for move in sorted(accumulated)}


def _validate_root_move(move_usi: str | None, *, board: Board, field: str) -> None:
    if move_usi is None:
        return
    move = _move(move_usi, field=field, sfen=board.to_sfen())
    if not board.is_legal_move(move):
        raise ValueError(f"illegal {field} move {move_usi!r} at {board.to_sfen()}")


def _validate_typed_variations(sample: PositionSample, *, board: Board) -> None:
    if sample.teacher_variations is None:
        return
    for variation in sample.teacher_variations:
        variation_board = Board(board.to_sfen())
        for pv_index, move_usi in enumerate(variation.pv):
            move = _move(
                move_usi,
                field=f"teacher_variations rank {variation.rank} PV[{pv_index}]",
                sfen=variation_board.to_sfen(),
            )
            if not variation_board.is_legal_move(move):
                raise ValueError(
                    "illegal typed teacher PV move "
                    f"{move_usi!r} at {variation_board.to_sfen()}"
                )
            variation_board.apply_move(move)


def _normalized_teacher_target(
    game: GameRecord,
    sample: PositionSample,
) -> tuple[dict[str, float], float, UsiPositionHistory]:
    if sample.teacher_policy is None or sample.teacher_value is None:
        raise ValueError("a complete teacher target requires both policy and value")
    if sample.teacher_source is None or not sample.teacher_source.strip():
        raise ValueError("a complete teacher target requires a non-empty teacher_source")
    try:
        board = Board(sample.sfen)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid sample SFEN: {sample.sfen!r}") from error
    if not board.is_valid():
        raise ValueError(f"invalid sample SFEN: {sample.sfen!r}")
    if sample.turn not in {0, 1} or sample.turn != board.turn.value:
        raise ValueError(
            f"sample turn {sample.turn} does not match board turn {board.turn.value}"
        )
    _validate_policy(
        sample.policy,
        board=board,
        field="actor policy",
        require_positive_mass=False,
    )
    teacher_policy = _validate_policy(
        sample.teacher_policy,
        board=board,
        field="teacher policy",
        require_positive_mass=True,
    )
    _validate_root_move(sample.actor_best_move, board=board, field="actor_best_move")
    _validate_root_move(sample.chosen_move, board=board, field="chosen_move")
    _validate_root_move(sample.teacher_best_move, board=board, field="teacher_best_move")
    _validate_typed_variations(sample, board=board)
    teacher_value = float(sample.teacher_value)
    if not math.isfinite(teacher_value) or not -1.0 <= teacher_value <= 1.0:
        raise ValueError("teacher_value must be finite and in [-1, 1]")
    history = UsiPositionHistory.from_game(game, sample)
    return teacher_policy, (0.0 if teacher_value == 0.0 else teacher_value), history


def _policy_average(policies: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not policies:
        raise ValueError("at least one policy is required")
    support = sorted(set().union(*(policy.keys() for policy in policies)))
    averaged = {
        move: math.fsum(policy.get(move, 0.0) for policy in policies) / len(policies)
        for move in support
    }
    total = math.fsum(averaged.values())
    if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"averaged teacher policy is not normalized: {total}")
    return {move: averaged[move] / total for move in support if averaged[move] > 0.0}


def _entropy(policy: Mapping[str, float]) -> float:
    return -math.fsum(
        probability * math.log(probability)
        for probability in policy.values()
        if probability > 0.0
    )


def _l1(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return math.fsum(
        abs(left.get(move, 0.0) - right.get(move, 0.0))
        for move in left.keys() | right.keys()
    )


def _unanimous_optional_float(
    observations: Sequence[_TeacherObservation], attribute: str
) -> float | None:
    values = {
        cast(float | None, getattr(observation.sample, attribute))
        for observation in observations
    }
    return values.pop() if len(values) == 1 else None


def _teacher_occurrence(
    *,
    input_game_index: int,
    input_sample_index: int,
    game: GameRecord,
    sample: PositionSample,
    normalized_policy: dict[str, float],
    teacher_value: float,
    history: UsiPositionHistory,
    game_record_sha256: str,
) -> TeacherDedupOccurrence:
    sample_payload = sample.to_dict()
    history_payload = {
        "initial_sfen": history.initial_sfen,
        "moves": history.moves,
        "target_sfen": history.target_sfen,
    }
    variation_payload = (
        None
        if sample.teacher_variations is None
        else [variation.to_dict() for variation in sample.teacher_variations]
    )
    policy_payload = {"teacher_policy": normalized_policy}
    target_payload = {
        "teacher_policy": normalized_policy,
        "teacher_value": teacher_value,
    }
    return TeacherDedupOccurrence(
        input_game_index=input_game_index,
        input_sample_index=input_sample_index,
        game_id=_json_sha256(
            {
                "initial_sfen": game.initial_sfen,
                "moves": game.moves,
                "winner": game.winner,
                "termination": game.termination.value,
            }
        ),
        game_record_sha256=game_record_sha256,
        sample_sha256=_json_sha256(sample_payload),
        history_context_sha256=_json_sha256(history_payload),
        history_move_count=len(history.moves),
        sample_sfen=sample.sfen,
        ply=sample.ply,
        turn=sample.turn,
        teacher_source=cast(str, sample.teacher_source),
        teacher_context=sample.teacher_context,
        teacher_policy_sha256=_json_sha256(policy_payload),
        teacher_target_sha256=_json_sha256(target_payload),
        teacher_value=teacher_value,
        teacher_best_move=sample.teacher_best_move,
        typed_variation_count=len(sample.teacher_variations or ()),
        typed_variations_sha256=(
            None
            if variation_payload is None
            else _json_sha256({"teacher_variations": variation_payload})
        ),
        teacher_policy_temperature=sample.teacher_policy_temperature,
        teacher_value_scale=sample.teacher_value_scale,
    )


def _average_position(
    normalized: str,
    observations: Sequence[_TeacherObservation],
    *,
    split: str,
) -> tuple[PositionSample, TeacherDedupPositionReport, _TeacherObservation]:
    if not observations:
        raise ValueError("cannot average an empty position group")
    ordered = tuple(sorted(observations, key=lambda row: (row.game_index, row.sample_index)))
    representative = ordered[0]
    policies = [observation.normalized_policy for observation in ordered]
    averaged_policy = _policy_average(policies)
    values = [observation.teacher_value for observation in ordered]
    averaged_value = math.fsum(values) / len(values)
    averaged_best_move = min(
        averaged_policy,
        key=lambda move: (-averaged_policy[move], move),
    )
    representative_sample = representative.sample
    actor_best_move = representative_sample.actor_best_move
    if actor_best_move is None and representative_sample.policy:
        actor_best_move = min(
            representative_sample.policy,
            key=lambda move: (-representative_sample.policy[move], move),
        )
    output_sample = replace(
        representative_sample,
        teacher_policy=averaged_policy,
        teacher_value=averaged_value,
        policy_reversal=(
            actor_best_move is not None and actor_best_move != averaged_best_move
        ),
        teacher_best_move=averaged_best_move,
        teacher_regret=None,
        teacher_regret_is_lower_bound=False,
        teacher_nodes=None,
        teacher_depth_ratio=None,
        teacher_time_ms=None,
        teacher_nps=None,
        teacher_depth=None,
        teacher_context=f"single-teacher-dedup:{split}:equal-mean:{len(ordered)}",
        # Independent searches can have incompatible rankings, mate routes,
        # depths, and bounds.  Their exact typed PVs remain in the hashed raw
        # occurrences; inventing one combined list would corrupt provenance.
        teacher_variations=None,
        teacher_policy_temperature=_unanimous_optional_float(
            ordered, "teacher_policy_temperature"
        ),
        teacher_value_scale=_unanimous_optional_float(ordered, "teacher_value_scale"),
    )
    policy_signatures = {tuple(policy.items()) for policy in policies}
    value_signatures = set(values)
    mean_l1_values = [_l1(policy, averaged_policy) for policy in policies]
    generalized_js = max(
        0.0,
        _entropy(averaged_policy)
        - math.fsum(_entropy(policy) for policy in policies) / len(policies),
    )
    mean_value = averaged_value
    value_variance = math.fsum((value - mean_value) ** 2 for value in values) / len(values)
    supports = [set(policy) for policy in policies]
    report = TeacherDedupPositionReport(
        normalized_sfen=normalized,
        output_sfen=output_sample.sfen,
        output_sample_sha256=_json_sha256(output_sample.to_dict()),
        representative_game_index=representative.game_index,
        representative_sample_index=representative.sample_index,
        occurrence_count=len(ordered),
        distinct_input_games=len({observation.game_index for observation in ordered}),
        distinct_history_contexts=len(
            {observation.occurrence.history_context_sha256 for observation in ordered}
        ),
        distinct_teacher_contexts=len(
            {observation.occurrence.teacher_context for observation in ordered}
        ),
        policy_support_union_size=len(set.union(*supports)),
        policy_support_intersection_size=len(set.intersection(*supports)),
        policy_conflict=len(policy_signatures) > 1,
        value_conflict=len(value_signatures) > 1,
        target_conflict=len(policy_signatures) > 1 or len(value_signatures) > 1,
        distinct_policy_targets=len(policy_signatures),
        distinct_value_targets=len(value_signatures),
        generalized_policy_js_divergence_nats=generalized_js,
        mean_policy_l1_to_average=math.fsum(mean_l1_values) / len(mean_l1_values),
        maximum_policy_l1_to_average=max(mean_l1_values),
        teacher_value=averaged_value,
        teacher_value_minimum=min(values),
        teacher_value_maximum=max(values),
        teacher_value_width=max(values) - min(values),
        teacher_value_standard_deviation=math.sqrt(max(0.0, value_variance)),
        averaged_teacher_best_move=averaged_best_move,
        occurrences=tuple(observation.occurrence for observation in ordered),
    )
    return output_sample, report, representative


def _overlap_guards(
    output_keys: set[str], forbidden_replays: Sequence[Path]
) -> tuple[TeacherDedupOverlapGuard, ...]:
    reports: list[TeacherDedupOverlapGuard] = []
    seen_paths: set[str] = set()
    for supplied in forbidden_replays:
        replay = supplied.expanduser().resolve()
        normalized_path = str(replay).casefold()
        if normalized_path in seen_paths:
            raise ValueError(f"duplicate overlap-guard replay: {replay}")
        seen_paths.add(normalized_path)
        identity = identify_file(replay)
        games = load_games(replay)
        if identify_file(replay) != identity:
            raise RuntimeError(f"overlap-guard replay changed while being read: {replay}")
        guarded_keys = {
            normalized_sfen(sample.sfen)
            for game in games
            for sample in game.samples
        }
        overlap = output_keys & guarded_keys
        if overlap:
            examples = "; ".join(sorted(overlap)[:3])
            raise ValueError(
                f"split overlap guard rejected {len(overlap)} normalized positions from "
                f"{replay}; examples: {examples}"
            )
        reports.append(
            TeacherDedupOverlapGuard(
                replay=identity,
                games=len(games),
                samples=sum(len(game.samples) for game in games),
                normalized_positions=len(guarded_keys),
            )
        )
    return tuple(reports)


def build_single_teacher_dedup(
    input_replay: Path,
    *,
    split: str,
    forbid_overlap_with: Sequence[Path] = (),
) -> SingleTeacherDedupBuild:
    """Build one board-observation training split from one immutable teacher replay.

    Only complete games and samples containing both teacher policy and value are
    eligible.  Samples with neither target are counted and skipped; a half-filled
    target fails closed.  Every eligible occurrence is history-validated against
    its own ``GameRecord`` before any averaging occurs.
    """

    split_name = split.strip()
    if not split_name:
        raise ValueError("split must not be empty")
    source = input_replay.expanduser().resolve()
    artifact = identify_input_artifact(
        source,
        adjacent_suffixes=(
            ".ensemble.json",
            ".provenance.json",
            ".disagreement.json",
            ".arbitration.json",
            ".dedup.json",
        ),
    )
    games = tuple(load_games(source))
    # Reject a file that changed between hashing and parsing rather than
    # publishing lineage for bytes other than those actually consumed.
    if (
        identify_input_artifact(
            source,
            adjacent_suffixes=(
                ".ensemble.json",
                ".provenance.json",
                ".disagreement.json",
                ".arbitration.json",
                ".dedup.json",
            ),
        )
        != artifact
    ):
        raise RuntimeError(
            f"input replay or adjacent lineage changed while it was being read: {source}"
        )

    grouped: defaultdict[str, list[_TeacherObservation]] = defaultdict(list)
    unlabeled_samples = 0
    incomplete_games = 0
    teacher_sources: set[str] = set()
    for game_index, game in enumerate(games):
        replay_and_validate(game)
        if game.termination is Termination.MAX_PLIES:
            incomplete_games += 1
            continue
        game_record_sha256 = _json_sha256(game.to_dict())
        for sample_index, sample in enumerate(game.samples):
            has_policy = sample.teacher_policy is not None
            has_value = sample.teacher_value is not None
            if not has_policy and not has_value:
                unlabeled_samples += 1
                continue
            if has_policy != has_value:
                raise ValueError(
                    "teacher sample must provide both policy and value at "
                    f"game {game_index}, sample {sample_index}"
                )
            normalized_policy, teacher_value, history = _normalized_teacher_target(
                game, sample
            )
            teacher_sources.add(cast(str, sample.teacher_source))
            key = normalized_sfen(sample.sfen)
            occurrence = _teacher_occurrence(
                input_game_index=game_index,
                input_sample_index=sample_index,
                game=game,
                sample=sample,
                normalized_policy=normalized_policy,
                teacher_value=teacher_value,
                history=history,
                game_record_sha256=game_record_sha256,
            )
            grouped[key].append(
                _TeacherObservation(
                    game_index=game_index,
                    sample_index=sample_index,
                    game=game,
                    sample=sample,
                    normalized_policy=normalized_policy,
                    teacher_value=teacher_value,
                    occurrence=occurrence,
                )
            )

    if not grouped:
        raise ValueError("at least one complete teacher-labelled sample is required")
    if len(teacher_sources) != 1:
        raise ValueError(
            "single-teacher dedup requires exactly one teacher_source, got "
            f"{sorted(teacher_sources)!r}"
        )

    output_rows: defaultdict[int, list[tuple[int, str, PositionSample]]] = defaultdict(list)
    position_reports: list[TeacherDedupPositionReport] = []
    group_histogram: Counter[int] = Counter()
    output_keys = set(grouped)
    guard_reports = _overlap_guards(output_keys, forbid_overlap_with)
    for key in sorted(grouped):
        observations = grouped[key]
        group_histogram[len(observations)] += 1
        output_sample, position_report, representative = _average_position(
            key,
            observations,
            split=split_name,
        )
        output_rows[representative.game_index].append(
            (representative.sample_index, key, output_sample)
        )
        position_reports.append(position_report)

    output_games: list[GameRecord] = []
    for game_index in sorted(output_rows):
        samples = tuple(
            sample
            for _sample_index, _key, sample in sorted(
                output_rows[game_index], key=lambda row: (row[0], row[1])
            )
        )
        output_games.append(replace(games[game_index], samples=samples))

    complete_observations = sum(len(rows) for rows in grouped.values())
    duplicate_collapsed = complete_observations - len(grouped)
    report = SingleTeacherDedupReport(
        schema=SINGLE_TEACHER_DEDUP_SCHEMA,
        split=split_name,
        method=SINGLE_TEACHER_DEDUP_METHOD,
        position_identity=(
            "rsshogi-canonical SFEN board, side-to-move, and hands; move counter ignored"
        ),
        input=SingleTeacherDedupInputReport(
            artifact=artifact,
            games=len(games),
            eligible_games=len(games) - incomplete_games,
            incomplete_games_skipped=incomplete_games,
            raw_samples=sum(len(game.samples) for game in games),
            complete_teacher_observations=complete_observations,
            unlabeled_samples_skipped=unlabeled_samples,
            unique_normalized_positions=len(grouped),
            duplicate_observations_collapsed=duplicate_collapsed,
            teacher_sources=tuple(sorted(teacher_sources)),
        ),
        overlap_guards=guard_reports,
        output_games=len(output_games),
        output_samples=len(grouped),
        duplicate_positions=sum(len(rows) > 1 for rows in grouped.values()),
        duplicate_observations_collapsed=duplicate_collapsed,
        positions_with_multiple_histories=sum(
            report.distinct_history_contexts > 1 for report in position_reports
        ),
        positions_with_policy_conflicts=sum(
            report.policy_conflict for report in position_reports
        ),
        positions_with_value_conflicts=sum(
            report.value_conflict for report in position_reports
        ),
        positions_with_any_target_conflict=sum(
            report.target_conflict for report in position_reports
        ),
        group_size_histogram={
            str(size): count for size, count in sorted(group_histogram.items())
        },
        maximum_policy_l1_to_average=max(
            report.maximum_policy_l1_to_average for report in position_reports
        ),
        maximum_teacher_value_width=max(
            report.teacher_value_width for report in position_reports
        ),
        positions=tuple(position_reports),
        invariants=(
            "The input replay is read-only and identified before and after parsing.",
            "Every emitted normalized position occurs exactly once in the declared split.",
            "Every input and averaged teacher policy is legal and normalized to unit mass.",
            "Every eligible sample is reconstructed from its own initial_sfen and move prefix.",
            "Policy and signed value use an equal-weight arithmetic mean over all occurrences.",
            "Each output GameRecord is one unchanged source game with only its sample "
            "subset replaced.",
            "The replay and manifest are create-only; the replay is installed last as "
            "commit marker.",
        ),
        limitations=(
            "The current Meteo network has no history planes, so averaging aliases "
            "history-dependent teacher knowledge into the observable board state instead of "
            "representing it explicitly.",
            "The merged PositionSample deliberately clears typed PV rankings and non-additive "
            "search telemetry. Raw typed variations remain in the immutable input at the "
            "hashed game/sample "
            "indices recorded by positions[].occurrences[].",
            "A representative actor sample and containing game preserve one genuine story; they do "
            "not assert that other occurrence histories followed that representative trajectory.",
        ),
    )
    return SingleTeacherDedupBuild(
        input_replay=source,
        games=tuple(output_games),
        report=report,
    )


def _temporary_sibling(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        dir=target.parent,
    )
    os.close(descriptor)
    path = Path(name)
    path.chmod(0o644)
    return path


def _write_replay(path: Path, games: Sequence[GameRecord]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for game in games:
            stream.write(
                json.dumps(
                    {"schema": SCHEMA_VERSION, "game": game.to_dict()},
                    allow_nan=False,
                )
            )
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except FileNotFoundError:
        return False


def _install_create_only_pair(
    temporary_replay: Path,
    output: Path,
    temporary_manifest: Path,
    manifest: Path,
) -> None:
    """Install two complete files without overwriting; replay visibility commits the pair."""

    installed_manifest = False
    try:
        try:
            os.link(temporary_manifest, manifest)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite dedup manifest: {manifest}") from error
        installed_manifest = True
        try:
            os.link(temporary_replay, output)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite dedup replay: {output}") from error
    except BaseException:
        if installed_manifest and _same_file(temporary_manifest, manifest):
            manifest.unlink()
        raise
    finally:
        temporary_replay.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)


def write_single_teacher_dedup(
    build: SingleTeacherDedupBuild,
    output: Path,
    *,
    manifest: Path | None = None,
) -> dict[str, object]:
    """Atomically publish a new deduplicated replay and its complete manifest.

    Both targets are staged and fsynced first.  Hard-link installation provides
    create-only atomicity for each file, and the replay is made visible last so
    its presence is the pair's commit marker.  A detected race rolls back the
    manifest created by this call without touching another process's file.
    """

    output_path = output.expanduser().resolve()
    manifest_path = (
        manifest.expanduser().resolve()
        if manifest is not None
        else output_path.with_suffix(output_path.suffix + ".dedup.json")
    )
    path_labels = {
        str(build.input_replay).casefold(): "input replay",
        str(output_path).casefold(): "output replay",
        str(manifest_path).casefold(): "dedup manifest",
    }
    if len(path_labels) != 3:
        raise ValueError("input replay, output replay, and dedup manifest paths must differ")
    lineage_paths = {
        identity.path.casefold()
        for identity in build.report.input.artifact.adjacent_lineage
    }
    if (
        str(output_path).casefold() in lineage_paths
        or str(manifest_path).casefold() in lineage_paths
    ):
        raise ValueError("dedup output paths must not shadow input lineage files")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite dedup replay: {output_path}")
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite dedup manifest: {manifest_path}")
    current_artifact = identify_input_artifact(
        build.input_replay,
        adjacent_suffixes=(
            ".ensemble.json",
            ".provenance.json",
            ".disagreement.json",
            ".arbitration.json",
            ".dedup.json",
        ),
    )
    if current_artifact != build.report.input.artifact:
        raise RuntimeError("input replay or adjacent lineage changed after deduplication")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_replay = _temporary_sibling(output_path)
    temporary_manifest = _temporary_sibling(manifest_path)
    try:
        _write_replay(temporary_replay, build.games)
        if tuple(load_games(temporary_replay)) != build.games:
            raise RuntimeError("staged dedup replay does not round-trip to the build")
        output_identity = FileIdentity(
            path=str(output_path),
            sha256=identify_file(temporary_replay).sha256,
            bytes=temporary_replay.stat().st_size,
        )
        payload = build.report.to_dict()
        payload["output"] = {
            "replay": output_identity.to_dict(),
            "manifest": str(manifest_path),
            "games": len(build.games),
            "samples": sum(len(game.samples) for game in build.games),
        }
        _write_json(temporary_manifest, payload)
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite dedup replay: {output_path}")
        if manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite dedup manifest: {manifest_path}")
        _install_create_only_pair(
            temporary_replay,
            output_path,
            temporary_manifest,
            manifest_path,
        )
    except BaseException:
        temporary_replay.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise

    return payload
