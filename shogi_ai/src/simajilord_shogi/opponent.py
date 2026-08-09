"""Persistent opponent tendencies, weaknesses, and robust exploit priors."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rsshogi.core import Board, Move

from .domain import GameRecord
from .evaluator import Evaluation, Evaluator


@dataclass(frozen=True, slots=True)
class Weakness:
    sfen: str
    played_move: str
    teacher_move: str
    regret: float
    ply: int


@dataclass(slots=True)
class OpponentProfile:
    name: str
    games: int = 0
    prefix_window: int = 16
    position_moves: dict[str, dict[str, int]] = field(default_factory=dict)
    prefix_moves: dict[str, dict[str, int]] = field(default_factory=dict)
    weaknesses: list[Weakness] = field(default_factory=list)

    def observe_game(self, game: GameRecord, opponent_color: int) -> None:
        """Update only after a game, preventing future-move leakage midgame."""

        board = Board(game.initial_sfen)
        history: list[str] = []
        samples = {sample.ply: sample for sample in game.samples}
        for ply, move_usi in enumerate(game.moves):
            if board.turn.value == opponent_color:
                position_key = str(board.zobrist_hash())
                _increment(self.position_moves, position_key, move_usi)
                prefix_key = self._prefix_key(history)
                _increment(self.prefix_moves, prefix_key, move_usi)
                sample = samples.get(ply)
                if (
                    sample is not None
                    and sample.teacher_best_move is not None
                    and sample.teacher_regret is not None
                    and sample.teacher_regret > 0
                ):
                    self.weaknesses.append(
                        Weakness(
                            sfen=sample.sfen,
                            played_move=move_usi,
                            teacher_move=sample.teacher_best_move,
                            regret=sample.teacher_regret,
                            ply=ply,
                        )
                    )
            move = Move.from_usi(move_usi)
            if not board.is_legal_move(move):
                raise ValueError(f"illegal observed move at ply {ply}: {move_usi}")
            board.apply_move(move)
            history.append(move_usi)
        self.games += 1
        self.weaknesses.sort(key=lambda weakness: weakness.regret, reverse=True)
        del self.weaknesses[1000:]

    def predict(self, board: Board, history: list[str] | None = None) -> dict[str, float]:
        """Predict a likely move from exact positions and opening/plan prefixes."""

        counts: dict[str, float] = {}
        for move, count in self.position_moves.get(str(board.zobrist_hash()), {}).items():
            counts[move] = counts.get(move, 0.0) + 2.0 * count
        if history is not None:
            for move, count in self.prefix_moves.get(self._prefix_key(history), {}).items():
                counts[move] = counts.get(move, 0.0) + count
        legal = {move.to_usi() for move in board.legal_moves()}
        counts = {move: count for move, count in counts.items() if move in legal}
        total = sum(counts.values())
        return {move: count / total for move, count in counts.items()} if total else {}

    def likely_continuations(
        self, history: list[str], *, limit: int = 5
    ) -> list[tuple[str, float]]:
        counts = self.prefix_moves.get(self._prefix_key(history), {})
        total = sum(counts.values())
        if total == 0:
            return []
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return [(move, count / total) for move, count in ranked[:limit]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "games": self.games,
            "prefix_window": self.prefix_window,
            "position_moves": self.position_moves,
            "prefix_moves": self.prefix_moves,
            "weaknesses": [asdict(weakness) for weakness in self.weaknesses],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> OpponentProfile:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["weaknesses"] = [Weakness(**weakness) for weakness in payload.get("weaknesses", [])]
        return cls(**payload)

    def _prefix_key(self, history: list[str]) -> str:
        return " ".join(history[-self.prefix_window :])


class OpponentConditionedEvaluator:
    """Blend likely opponent moves into priors without pruning robust replies."""

    def __init__(
        self,
        base: Evaluator,
        profile: OpponentProfile,
        opponent_color: int,
        *,
        exploit_weight: float = 0.3,
    ) -> None:
        if not 0 <= exploit_weight <= 0.5:
            raise ValueError("exploit_weight must be between zero and 0.5")
        self.base = base
        self.profile = profile
        self.opponent_color = opponent_color
        self.exploit_weight = exploit_weight

    def evaluate(self, board: Board) -> Evaluation:
        return self._blend(board, self.base.evaluate(board))

    def evaluate_batch(self, boards: list[Board]) -> list[Evaluation]:
        method = getattr(self.base, "evaluate_batch", None)
        base_results = (
            list(method(boards))
            if callable(method)
            else [self.base.evaluate(board) for board in boards]
        )
        return [
            self._blend(board, result) for board, result in zip(boards, base_results, strict=True)
        ]

    def _blend(self, board: Board, evaluation: Evaluation) -> Evaluation:
        if board.turn.value != self.opponent_color:
            return evaluation
        predicted = self.profile.predict(board)
        if not predicted:
            return evaluation
        weight = self.exploit_weight
        policy = {
            move: (1 - weight) * probability + weight * predicted.get(move, 0.0)
            for move, probability in evaluation.policy.items()
        }
        total = sum(policy.values())
        return Evaluation(
            policy={move: probability / total for move, probability in policy.items()},
            value=evaluation.value,
        )


def _increment(table: dict[str, dict[str, int]], key: str, move: str) -> None:
    row = table.setdefault(key, {})
    row[move] = row.get(move, 0) + 1
