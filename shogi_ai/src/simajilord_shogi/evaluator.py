"""Policy-value evaluator contracts and deterministic test evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rsshogi.core import Board


@dataclass(frozen=True, slots=True)
class Evaluation:
    policy: dict[str, float]
    value: float


class Evaluator(Protocol):
    def evaluate(self, board: Board) -> Evaluation: ...


class BatchEvaluator(Evaluator, Protocol):
    def evaluate_batch(self, boards: list[Board]) -> list[Evaluation]: ...


class UniformEvaluator:
    """Uniform legal policy used to test rules/search without neural hardware."""

    def evaluate(self, board: Board) -> Evaluation:
        moves = board.legal_moves()
        probability = 1.0 / len(moves) if moves else 0.0
        return Evaluation({move.to_usi(): probability for move in moves}, 0.0)


def normalize_legal_policy(
    board: Board, raw_policy: dict[str, float], prior_floor: float
) -> dict[str, float]:
    """Mask illegal moves and preserve a nonzero route to every legal move."""

    legal = [move.to_usi() for move in board.legal_moves()]
    if not legal:
        return {}
    scores = np.asarray(
        [max(0.0, float(raw_policy.get(move, 0.0))) + prior_floor for move in legal],
        dtype=np.float64,
    )
    total = float(scores.sum())
    if not np.isfinite(total) or total <= 0:
        scores.fill(1.0)
        total = float(len(scores))
    return {move: float(score / total) for move, score in zip(legal, scores, strict=True)}
