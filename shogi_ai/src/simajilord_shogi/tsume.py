"""Legal forced-mate validation and automatic puzzle mining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rsshogi.core import Board, Move
from rsshogi.types import RepetitionState

from .domain import GameRecord


@dataclass(frozen=True, slots=True)
class TsumeVariation:
    move: str
    replies: tuple[TsumeVariation, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TsumePuzzle:
    sfen: str
    max_plies: int
    first_move: str
    solution: TsumeVariation
    nodes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TsumeSolutionSet:
    """Every independently proved root move that mates within the bound."""

    sfen: str
    max_plies: int
    solutions: tuple[TsumeVariation, ...]
    nodes: int

    @property
    def first_moves(self) -> tuple[str, ...]:
        return tuple(solution.move for solution in self.solutions)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TsumeSolver:
    """Small-depth exact tsume solver.

    The attacker must check on every move. Every legal defensive reply must
    remain forced mate. Standard legality, including double-pawn and illegal
    pawn-drop mate, is delegated to rsshogi.
    """

    def __init__(
        self,
        *,
        node_limit: int = 1_000_000,
        history_exact: bool = False,
    ) -> None:
        if node_limit < 1:
            raise ValueError("node_limit must be positive")
        self.node_limit = node_limit
        self.history_exact = history_exact
        self.nodes = 0
        self._cache: dict[tuple[int, int], TsumeVariation | None] = {}

    def solve_all(self, board: Board, max_plies: int) -> TsumeSolutionSet:
        """Prove all root checking moves that force mate within ``max_plies``.

        This deliberately does not collapse a position to one canonical move.
        Different teachers may choose different, equally correct mating lines.
        """

        if max_plies < 1 or max_plies % 2 == 0:
            raise ValueError("tsume depth must be a positive odd number")
        self.nodes = 0
        self._cache.clear()
        winning: list[TsumeVariation] = []
        for move in board.legal_moves():
            result = self._try_attack_move(board, move, max_plies)
            if result is not None:
                winning.append(result)
        return TsumeSolutionSet(
            sfen=board.to_sfen(),
            max_plies=max_plies,
            solutions=tuple(winning),
            nodes=self.nodes,
        )

    def solve_unique(self, board: Board, max_plies: int) -> TsumePuzzle | None:
        solution_set = self.solve_all(board, max_plies)
        if len(solution_set.solutions) != 1:
            return None
        solution = solution_set.solutions[0]
        return TsumePuzzle(
            sfen=solution_set.sfen,
            max_plies=max_plies,
            first_move=solution.move,
            solution=solution,
            nodes=solution_set.nodes,
        )

    def _try_attack_move(self, board: Board, move: Move, remaining: int) -> TsumeVariation | None:
        self._count_node()
        after_attack = board.copy()
        after_attack.apply_move(move)
        if not after_attack.is_in_check():
            return None
        if after_attack.is_mated() or not after_attack.legal_moves():
            return TsumeVariation(move.to_usi())
        if remaining < 3 or after_attack.repetition_state() != RepetitionState.NONE:
            return None

        defenses: list[TsumeVariation] = []
        for defense in after_attack.legal_moves():
            after_defense = after_attack.copy()
            after_defense.apply_move(defense)
            continuation = self._solve_attacker(after_defense, remaining - 2)
            if continuation is None:
                return None
            defenses.append(TsumeVariation(defense.to_usi(), (continuation,)))
        return TsumeVariation(move.to_usi(), tuple(defenses))

    def _solve_attacker(self, board: Board, remaining: int) -> TsumeVariation | None:
        key = (board.zobrist_hash(), remaining)
        if not self.history_exact and key in self._cache:
            return self._cache[key]
        for move in board.legal_moves():
            result = self._try_attack_move(board, move, remaining)
            if result is not None:
                if not self.history_exact:
                    self._cache[key] = result
                return result
        if not self.history_exact:
            self._cache[key] = None
        return None

    def _count_node(self) -> None:
        self.nodes += 1
        if self.nodes > self.node_limit:
            raise RuntimeError("tsume node limit exceeded")


def mine_unique_tsume(
    games: list[GameRecord],
    *,
    max_plies: int,
    node_limit: int = 1_000_000,
    maximum_puzzles: int = 100,
) -> list[TsumePuzzle]:
    """Mine validated, unique-first-move puzzles from generated game positions."""

    puzzles: list[TsumePuzzle] = []
    seen: set[int] = set()
    for game in games:
        for sample in reversed(game.samples):
            board = Board(sample.sfen)
            position_key = board.zobrist_hash()
            if position_key in seen:
                continue
            seen.add(position_key)
            try:
                puzzle = TsumeSolver(node_limit=node_limit).solve_unique(board, max_plies)
            except RuntimeError:
                continue
            if puzzle is not None:
                puzzles.append(puzzle)
                if len(puzzles) >= maximum_puzzles:
                    return puzzles
    return puzzles
