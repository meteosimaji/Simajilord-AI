"""Held-out metrics for full-distribution policy/value distillation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from rsshogi.core import Board, Move

from .domain import PositionSample
from .encoding import HistoryInput, board_from_history_input, history_input_sha256
from .evaluator import BatchEvaluator, Evaluation


@dataclass(frozen=True, slots=True)
class AlignmentMetrics:
    samples: int
    unique_positions: int
    policy_cross_entropy: float
    policy_js_divergence: float
    teacher_mass_at_1: float
    teacher_mass_at_3: float
    teacher_mass_at_5: float
    teacher_best_top_1: float
    teacher_best_top_3: float
    teacher_best_top_5: float
    value_samples: int
    value_mse: float | None
    value_brier: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    overall: AlignmentMetrics
    by_teacher: dict[str, AlignmentMetrics]
    by_phase: dict[str, AlignmentMetrics]

    def to_dict(self) -> dict[str, object]:
        return {
            "overall": self.overall.to_dict(),
            "by_teacher": {
                name: metrics.to_dict() for name, metrics in sorted(self.by_teacher.items())
            },
            "by_phase": {
                name: metrics.to_dict() for name, metrics in sorted(self.by_phase.items())
            },
        }


@dataclass(frozen=True, slots=True)
class _ScoredSample:
    position_key: str
    teacher: str
    phase: str
    policy_cross_entropy: float
    policy_js_divergence: float
    teacher_mass_at_1: float
    teacher_mass_at_3: float
    teacher_mass_at_5: float
    teacher_best_top_1: float
    teacher_best_top_3: float
    teacher_best_top_5: float
    value_squared_error: float | None
    value_brier: float | None


def normalized_position_key(sfen: str) -> str:
    """Ignore only the SFEN move counter when detecting position leakage."""

    fields = sfen.split()
    if len(fields) != 4:
        raise ValueError(f"invalid SFEN field count: {sfen!r}")
    return " ".join(fields[:3])


def _phase(sample: PositionSample) -> str:
    move_number = int(sample.sfen.split()[3])
    absolute_ply = 2 * (move_number - 1) + sample.turn
    if absolute_ply < 24:
        return "opening"
    if absolute_ply < 80:
        return "middlegame"
    return "endgame"


def _normalized_teacher_policy(sample: PositionSample, board: Board) -> dict[str, float]:
    if sample.teacher_policy is None:
        raise ValueError("held-out alignment requires teacher_policy for every sample")
    legal = {move.to_usi() for move in board.legal_moves()}
    policy: dict[str, float] = {}
    for move, probability in sample.teacher_policy.items():
        if move not in legal or not board.is_legal_move(Move.from_usi(move)):
            raise ValueError(f"illegal teacher move {move!r} for {sample.sfen!r}")
        weight = float(probability)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid teacher probability for {move!r}: {probability!r}")
        policy[move] = policy.get(move, 0.0) + weight
    total = sum(policy.values())
    if total <= 0:
        raise ValueError(f"empty teacher policy for {sample.sfen!r}")
    return {move: probability / total for move, probability in policy.items()}


def _normalized_prediction(evaluation: Evaluation, board: Board) -> dict[str, float]:
    legal = {move.to_usi() for move in board.legal_moves()}
    if set(evaluation.policy) != legal:
        raise ValueError("evaluator policy must contain every legal move and no illegal move")
    policy: dict[str, float] = {}
    for move, probability in evaluation.policy.items():
        weight = float(probability)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid evaluator probability for {move!r}: {probability!r}")
        policy[move] = weight
    total = sum(policy.values())
    if total <= 0:
        raise ValueError("evaluator returned an empty policy")
    return {move: probability / total for move, probability in policy.items()}


def _jensen_shannon(
    teacher: dict[str, float], prediction: dict[str, float], *, epsilon: float
) -> float:
    moves = set(teacher) | set(prediction)
    divergence = 0.0
    for move in moves:
        teacher_probability = teacher.get(move, 0.0)
        predicted_probability = prediction.get(move, 0.0)
        midpoint = 0.5 * (teacher_probability + predicted_probability)
        if teacher_probability > 0:
            divergence += 0.5 * teacher_probability * math.log(
                teacher_probability / max(midpoint, epsilon)
            )
        if predicted_probability > 0:
            divergence += 0.5 * predicted_probability * math.log(
                predicted_probability / max(midpoint, epsilon)
            )
    return divergence


def _score_sample(
    sample: PositionSample,
    evaluation: Evaluation,
    *,
    position_key: str,
    epsilon: float,
) -> _ScoredSample:
    board = Board(sample.sfen)
    teacher = _normalized_teacher_policy(sample, board)
    prediction = _normalized_prediction(evaluation, board)
    cross_entropy = -sum(
        probability * math.log(max(prediction.get(move, 0.0), epsilon))
        for move, probability in teacher.items()
    )
    predicted_moves = sorted(prediction, key=prediction.__getitem__, reverse=True)
    teacher_best = max(teacher, key=teacher.__getitem__)

    def mass_at(limit: int) -> float:
        return sum(teacher.get(move, 0.0) for move in predicted_moves[:limit])

    teacher_value = sample.teacher_value
    squared_error: float | None = None
    brier: float | None = None
    if teacher_value is not None:
        if not -1.0 <= teacher_value <= 1.0 or not -1.0 <= evaluation.value <= 1.0:
            raise ValueError("teacher and evaluator values must both be in [-1, 1]")
        squared_error = (evaluation.value - teacher_value) ** 2
        brier = (((evaluation.value + 1.0) / 2.0) - ((teacher_value + 1.0) / 2.0)) ** 2
    return _ScoredSample(
        position_key=position_key,
        teacher=sample.teacher_source or "unknown",
        phase=_phase(sample),
        policy_cross_entropy=cross_entropy,
        policy_js_divergence=_jensen_shannon(teacher, prediction, epsilon=epsilon),
        teacher_mass_at_1=mass_at(1),
        teacher_mass_at_3=mass_at(3),
        teacher_mass_at_5=mass_at(5),
        teacher_best_top_1=float(teacher_best in predicted_moves[:1]),
        teacher_best_top_3=float(teacher_best in predicted_moves[:3]),
        teacher_best_top_5=float(teacher_best in predicted_moves[:5]),
        value_squared_error=squared_error,
        value_brier=brier,
    )


def _aggregate(scored: Sequence[_ScoredSample]) -> AlignmentMetrics:
    if not scored:
        raise ValueError("cannot aggregate an empty held-out group")
    squared_errors = [
        row.value_squared_error for row in scored if row.value_squared_error is not None
    ]
    brier_scores = [row.value_brier for row in scored if row.value_brier is not None]

    def mean(attribute: str) -> float:
        return sum(float(getattr(row, attribute)) for row in scored) / len(scored)

    return AlignmentMetrics(
        samples=len(scored),
        unique_positions=len({row.position_key for row in scored}),
        policy_cross_entropy=mean("policy_cross_entropy"),
        policy_js_divergence=mean("policy_js_divergence"),
        teacher_mass_at_1=mean("teacher_mass_at_1"),
        teacher_mass_at_3=mean("teacher_mass_at_3"),
        teacher_mass_at_5=mean("teacher_mass_at_5"),
        teacher_best_top_1=mean("teacher_best_top_1"),
        teacher_best_top_3=mean("teacher_best_top_3"),
        teacher_best_top_5=mean("teacher_best_top_5"),
        value_samples=len(squared_errors),
        value_mse=(
            sum(squared_errors) / len(squared_errors)
            if squared_errors
            else None
        ),
        value_brier=(
            sum(brier_scores) / len(brier_scores)
            if brier_scores
            else None
        ),
    )


def evaluate_teacher_alignment(
    evaluator: BatchEvaluator,
    samples: Sequence[PositionSample],
    *,
    histories: Sequence[HistoryInput] | None = None,
    batch_size: int = 32,
    epsilon: float = 1e-12,
) -> AlignmentReport:
    """Evaluate a model against complete MultiPV targets without train-set leakage.

    With history-input-v2, deduplication and inference use the exact replay
    prefix rather than SFEN alone.  Different teachers may intentionally score
    the same exact position history independently.
    """

    if batch_size < 1 or not 0 < epsilon < 1:
        raise ValueError("batch_size and epsilon must be valid")
    if histories is not None and len(histories) != len(samples):
        raise ValueError("exact histories must align one-to-one with evaluation samples")
    paired_histories: Sequence[HistoryInput | None] = (
        [None] * len(samples) if histories is None else histories
    )
    deduplicated: dict[
        tuple[str, str], tuple[PositionSample, HistoryInput | None, str]
    ] = {}
    signatures: dict[tuple[str, str], tuple[tuple[tuple[str, float], ...], float | None]] = {}
    for sample, history in zip(samples, paired_histories, strict=True):
        if history is not None:
            sample_sfen = Board(sample.sfen).to_sfen()
            history_sfen = board_from_history_input(history).to_sfen()
            if history_sfen != sample_sfen:
                raise ValueError(
                    "exact history target SFEN does not match its evaluation sample"
                )
        if sample.teacher_policy is None:
            continue
        position_key = (
            normalized_position_key(sample.sfen)
            if history is None
            else history_input_sha256(history)
        )
        key = (sample.teacher_source or "unknown", position_key)
        signature = (tuple(sorted(sample.teacher_policy.items())), sample.teacher_value)
        previous = signatures.get(key)
        if previous is not None and previous != signature:
            raise ValueError(f"conflicting duplicate teacher target for {key[0]} at {key[1]}")
        signatures[key] = signature
        deduplicated.setdefault(key, (sample, history, position_key))
    selected = list(deduplicated.values())
    if not selected:
        raise ValueError("at least one teacher-labelled sample is required")

    scored: list[_ScoredSample] = []
    for offset in range(0, len(selected), batch_size):
        batch = selected[offset : offset + batch_size]
        batch_samples = [sample for sample, _history, _key in batch]
        batch_histories = [history for _sample, history, _key in batch]
        if histories is None:
            evaluations = evaluator.evaluate_batch(
                [Board(sample.sfen) for sample in batch_samples]
            )
        else:
            if any(history is None for history in batch_histories):
                raise AssertionError("history-aware evaluation lost an exact prefix")
            exact_batch_histories = [
                history for history in batch_histories if history is not None
            ]
            evaluate_histories = getattr(evaluator, "evaluate_history_batch", None)
            if not callable(evaluate_histories):
                raise TypeError(
                    "history-input-v2 alignment requires an evaluator with "
                    "evaluate_history_batch"
                )
            evaluations = list(evaluate_histories(exact_batch_histories))
        if len(evaluations) != len(batch):
            raise ValueError("evaluator returned the wrong batch length")
        scored.extend(
            _score_sample(
                sample,
                evaluation,
                position_key=position_key,
                epsilon=epsilon,
            )
            for (sample, _history, position_key), evaluation in zip(
                batch,
                evaluations,
                strict=True,
            )
        )

    by_teacher_rows: defaultdict[str, list[_ScoredSample]] = defaultdict(list)
    by_phase_rows: defaultdict[str, list[_ScoredSample]] = defaultdict(list)
    for row in scored:
        by_teacher_rows[row.teacher].append(row)
        by_phase_rows[row.phase].append(row)
    return AlignmentReport(
        overall=_aggregate(scored),
        by_teacher={name: _aggregate(rows) for name, rows in by_teacher_rows.items()},
        by_phase={name: _aggregate(rows) for name, rows in by_phase_rows.items()},
    )
