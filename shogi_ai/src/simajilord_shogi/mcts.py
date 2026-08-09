"""Neural PUCT search with explicit low-prior root coverage."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from time import perf_counter

import numpy as np
from rsshogi.core import Board, Move
from rsshogi.types import RepetitionState

from .config import SearchConfig
from .evaluator import Evaluation, Evaluator, normalize_legal_policy
from .memory import SearchMemoryGuard


@dataclass(slots=True)
class Edge:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    child: Node | None = None
    virtual_visits: int = 0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(slots=True)
class Node:
    visits: int = 0
    edges: dict[str, Edge] = field(default_factory=dict)
    expanded: bool = False
    virtual_visits: int = 0


@dataclass(frozen=True, slots=True)
class _LeafReservation:
    """One simulation path held until its shared neural batch completes."""

    root_index: int
    path: tuple[tuple[Node, Edge], ...]
    leaf_node: Node | None
    leaf_board: Board | None
    terminal_leaf_value: float | None


@dataclass(frozen=True, slots=True)
class SearchResult:
    policy: dict[str, float]
    q_values: dict[str, float]
    root_value: float
    best_move: str
    simulations: int
    discovery_simulation: int
    elapsed_seconds: float
    nodes_per_second: float
    peak_tree_nodes: int
    tree_recycles: int


def terminal_value(board: Board) -> float | None:
    """Value from the current side-to-move perspective, or None if playable."""

    repetition = board.repetition_state()
    if repetition == RepetitionState.WIN or repetition == RepetitionState.SUPERIOR:
        return 1.0
    if repetition == RepetitionState.LOSE or repetition == RepetitionState.INFERIOR:
        return -1.0
    if repetition == RepetitionState.DRAW:
        return 0.0
    if board.can_declare_win():
        return 1.0
    if board.is_mated() or not board.legal_moves():
        return -1.0
    return None


class MCTS:
    def __init__(
        self,
        evaluator: Evaluator,
        config: SearchConfig,
        *,
        seed: int | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.memory_guard = SearchMemoryGuard(explicit_node_limit=config.max_tree_nodes)

    def search(self, board: Board, *, add_root_noise: bool = False) -> SearchResult:
        return self.search_many([board], add_root_noise=add_root_noise)[0]

    def search_many(
        self, boards: list[Board], *, add_root_noise: bool = False
    ) -> list[SearchResult]:
        """Search roots with one neural batch spanning roots and reserved leaves.

        A batch size of one follows the original sequential-per-root algorithm.
        Larger batches reserve several distinct leaves from each root with virtual
        visits, evaluate them together, roll every reservation back, and only then
        commit the real backups.  Simulations are never fabricated when a forced
        line reaches an already-reserved leaf; that root simply starts a smaller
        batch and resumes after the pending evaluation has completed.
        """

        if not boards:
            return []
        if any(terminal_value(board) is not None for board in boards):
            raise ValueError("cannot search a terminal position")
        started = perf_counter()
        roots = [Node() for _ in boards]
        logical_tree_nodes = len(roots)
        peak_tree_nodes = logical_tree_nodes
        tree_recycles = 0
        for root, board, evaluation in zip(roots, boards, self._evaluate_many(boards), strict=True):
            self._expand_from_evaluation(root, board, evaluation)
            if add_root_noise:
                self._add_dirichlet_noise(root)

        simulation_targets = [
            max(self.config.simulations, self.config.root_min_visits * len(root.edges))
            for root in roots
        ]
        best_first_seen: list[dict[str, int]] = [{} for _ in roots]
        completed_simulations = [0] * len(roots)
        next_memory_check = self.config.memory_check_interval
        evaluator_supports_batch = callable(getattr(self.evaluator, "evaluate_batch", None))
        intra_root_batch_size = (
            self.config.intra_root_batch_size if evaluator_supports_batch else 1
        )
        while any(
            completed < target
            for completed, target in zip(completed_simulations, simulation_targets, strict=True)
        ):
            reservations: list[_LeafReservation] = []
            reservations_per_root = [0] * len(roots)
            reserved_leaf_ids: set[int] = set()
            pending_evaluation_count = 0
            immediate_progress = 0
            blocked_roots: set[int] = set()
            stop_batch = False
            force_memory_check = False

            # Slot-major order retains the old multi-root scheduling at batch=1
            # and gives every active root one reservation before a second one.
            for _slot in range(intra_root_batch_size):
                for index, (root, board) in enumerate(zip(roots, boards, strict=True)):
                    if (
                        index in blocked_roots
                        or completed_simulations[index]
                        + reservations_per_root[index]
                        >= simulation_targets[index]
                    ):
                        continue
                    if pending_evaluation_count >= self.config.max_evaluation_batch_size:
                        stop_batch = True
                        break
                    selected, created_nodes = self._reserve_leaf(
                        index,
                        root,
                        board,
                        reserved_leaf_ids,
                    )
                    if selected is None:
                        # A forced/very narrow line collided with an unexpanded
                        # leaf already in this batch. Evaluating now is the only
                        # way to advance without inventing a simulation.
                        blocked_roots.add(index)
                        continue
                    logical_tree_nodes += created_nodes
                    peak_tree_nodes = max(peak_tree_nodes, logical_tree_nodes)
                    if selected.terminal_leaf_value is not None:
                        # A known rules result needs no neural batch and can
                        # improve later selections in this same wave immediately.
                        self._backpropagate(list(selected.path), selected.terminal_leaf_value)
                        completed_simulations[index] += 1
                        current_best = max(
                            root.edges,
                            key=lambda move: root.edges[move].visits,
                        )
                        best_first_seen[index].setdefault(
                            current_best,
                            completed_simulations[index],
                        )
                        immediate_progress += 1
                    else:
                        if selected.leaf_node is None or selected.leaf_board is None:
                            raise AssertionError("neural leaf reservation is incomplete")
                        reservations.append(selected)
                        reservations_per_root[index] += 1
                        reserved_leaf_ids.add(id(selected.leaf_node))
                        pending_evaluation_count += 1
                    if logical_tree_nodes >= self.memory_guard.node_limit:
                        stop_batch = True
                        force_memory_check = True
                        break
                if stop_batch:
                    break

            if not reservations and immediate_progress == 0:
                raise AssertionError("MCTS batch scheduler made no simulation progress")

            pending_boards = [
                reservation.leaf_board
                for reservation in reservations
                if reservation.leaf_board is not None
            ]
            if any(board is None for board in pending_boards):
                raise AssertionError("pending neural evaluation is missing its board")
            evaluations = iter(self._evaluate_many(pending_boards))

            # No real statistic may retain virtual visits. Roll back the whole
            # wave before expansion/backpropagation so detached or terminal paths
            # cannot leak pessimistic virtual values into the published result.
            for reservation in reversed(reservations):
                self._rollback_virtual_visits(reservation.path)

            for reservation in reservations:
                if reservation.leaf_node is None or reservation.leaf_board is None:
                    raise AssertionError("neural leaf reservation is incomplete")
                value = self._expand_from_evaluation(
                    reservation.leaf_node,
                    reservation.leaf_board,
                    next(evaluations),
                )
                self._backpropagate(list(reservation.path), value)
                index = reservation.root_index
                completed_simulations[index] += 1
                root = roots[index]
                current_best = max(root.edges, key=lambda move: root.edges[move].visits)
                best_first_seen[index].setdefault(
                    current_best,
                    completed_simulations[index],
                )
            try:
                next(evaluations)
            except StopIteration:
                pass
            else:
                raise AssertionError("unused neural evaluation in MCTS batch")

            completed_rounds = max(completed_simulations)
            scheduled_memory_check = completed_rounds >= next_memory_check
            if force_memory_check or scheduled_memory_check:
                if self.memory_guard.should_recycle(logical_tree_nodes):
                    for root in roots:
                        for edge in root.edges.values():
                            edge.child = None
                    logical_tree_nodes = len(roots)
                    tree_recycles += 1
                    self.memory_guard.reclaim()
                if scheduled_memory_check:
                    next_memory_check = (
                        completed_rounds // self.config.memory_check_interval + 1
                    ) * self.config.memory_check_interval

        elapsed = max(perf_counter() - started, 1e-9)
        total_simulations = sum(simulation_targets)
        aggregate_nps = total_simulations / elapsed
        return [
            self._result(
                root,
                target,
                first_seen,
                elapsed_seconds=elapsed * target / total_simulations,
                nodes_per_second=aggregate_nps,
                peak_tree_nodes=peak_tree_nodes,
                tree_recycles=tree_recycles,
            )
            for root, target, first_seen in zip(
                roots, simulation_targets, best_first_seen, strict=True
            )
        ]

    def _reserve_leaf(
        self,
        root_index: int,
        root: Node,
        board: Board,
        reserved_leaf_ids: set[int],
    ) -> tuple[_LeafReservation | None, int]:
        """Select and virtually reserve one leaf without committing a visit."""

        leaf_board = board.copy()
        leaf = root
        path: list[tuple[Node, Edge]] = []
        created_nodes = 0
        while True:
            terminal = terminal_value(leaf_board)
            if terminal is not None:
                reservation = _LeafReservation(
                    root_index=root_index,
                    path=tuple(path),
                    leaf_node=None,
                    leaf_board=None,
                    terminal_leaf_value=terminal,
                )
                return reservation, created_nodes
            if not leaf.expanded:
                if id(leaf) in reserved_leaf_ids:
                    return None, created_nodes
                reservation = _LeafReservation(
                    root_index=root_index,
                    path=tuple(path),
                    leaf_node=leaf,
                    leaf_board=leaf_board,
                    terminal_leaf_value=None,
                )
                self._apply_virtual_visits(reservation.path)
                return reservation, created_nodes
            move_usi, edge = self._select_edge(leaf, is_root=leaf is root)
            leaf_board.apply_move(Move.from_usi(move_usi))
            if edge.child is None:
                edge.child = Node()
                created_nodes += 1
            path.append((leaf, edge))
            leaf = edge.child

    @staticmethod
    def _apply_virtual_visits(path: tuple[tuple[Node, Edge], ...]) -> None:
        for parent, edge in path:
            parent.virtual_visits += 1
            edge.virtual_visits += 1

    @staticmethod
    def _rollback_virtual_visits(path: tuple[tuple[Node, Edge], ...]) -> None:
        for parent, edge in path:
            if parent.virtual_visits < 1 or edge.virtual_visits < 1:
                raise AssertionError("unbalanced MCTS virtual-visit rollback")
            parent.virtual_visits -= 1
            edge.virtual_visits -= 1

    def _result(
        self,
        root: Node,
        simulations: int,
        best_first_seen: dict[str, int],
        *,
        elapsed_seconds: float,
        nodes_per_second: float,
        peak_tree_nodes: int,
        tree_recycles: int,
    ) -> SearchResult:
        if root.virtual_visits or any(edge.virtual_visits for edge in root.edges.values()):
            raise AssertionError("MCTS result retained a virtual visit")
        visits = {move: edge.visits for move, edge in root.edges.items()}
        total_visits = sum(visits.values())
        if total_visits != simulations:
            raise AssertionError(
                f"MCTS committed {total_visits} root visits for {simulations} simulations"
            )
        policy = {move: count / total_visits for move, count in visits.items()}
        q_values = {move: edge.q for move, edge in root.edges.items()}
        best_move = max(root.edges, key=lambda move: (root.edges[move].visits, root.edges[move].q))
        root_value = sum(edge.q * edge.visits for edge in root.edges.values()) / total_visits
        return SearchResult(
            policy=policy,
            q_values=q_values,
            root_value=float(root_value),
            best_move=best_move,
            simulations=simulations,
            discovery_simulation=best_first_seen.get(best_move, simulations),
            elapsed_seconds=elapsed_seconds,
            nodes_per_second=nodes_per_second,
            peak_tree_nodes=peak_tree_nodes,
            tree_recycles=tree_recycles,
        )

    def _expand(self, node: Node, board: Board) -> float:
        evaluation = self.evaluator.evaluate(board)
        return self._expand_from_evaluation(node, board, evaluation)

    def _expand_from_evaluation(self, node: Node, board: Board, evaluation: Evaluation) -> float:
        policy = normalize_legal_policy(board, evaluation.policy, self.config.policy_prior_floor)
        node.edges = {move: Edge(prior=prior) for move, prior in policy.items()}
        node.expanded = True
        return max(-1.0, min(1.0, float(evaluation.value)))

    def _evaluate_many(self, boards: list[Board]) -> list[Evaluation]:
        batch_method = getattr(self.evaluator, "evaluate_batch", None)
        if callable(batch_method):
            evaluations: list[Evaluation] = []
            batch_limit = self.config.max_evaluation_batch_size
            for start in range(0, len(boards), batch_limit):
                batch = boards[start : start + batch_limit]
                result = batch_method(batch)
                if len(result) != len(batch):
                    raise ValueError("batch evaluator returned the wrong number of results")
                evaluations.extend(result)
            return evaluations
        return [self.evaluator.evaluate(board) for board in boards]

    @staticmethod
    def _backpropagate(path: list[tuple[Node, Edge]], leaf_value: float) -> None:
        value = leaf_value
        for parent, edge in reversed(path):
            value = -value
            edge.visits += 1
            edge.value_sum += value
            parent.visits += 1

    def _simulate(self, node: Node, board: Board, *, is_root: bool = False) -> float:
        terminal = terminal_value(board)
        if terminal is not None:
            return terminal
        if not node.expanded:
            return self._expand(node, board)

        move_usi, edge = self._select_edge(node, is_root=is_root)
        board.apply_move(Move.from_usi(move_usi))
        if edge.child is None:
            edge.child = Node()
        child_value = self._simulate(edge.child, board)
        value = -child_value
        edge.visits += 1
        edge.value_sum += value
        node.visits += 1
        return value

    def _select_edge(self, node: Node, *, is_root: bool) -> tuple[str, Edge]:
        if is_root and self.config.root_min_visits:
            underexplored = [
                (move, edge)
                for move, edge in node.edges.items()
                if edge.visits + edge.virtual_visits < self.config.root_min_visits
            ]
            if underexplored:
                # Search the least promising unvisited move first. This prevents a
                # low network prior from permanently hiding a deep tactical move.
                return min(underexplored, key=lambda item: (item[1].prior, item[0]))

        parent_scale = sqrt(max(1, node.visits + node.virtual_visits))
        return max(
            node.edges.items(),
            key=lambda item: (
                self._selection_q(item[1])
                + self.config.c_puct
                * item[1].prior
                * parent_scale
                / (1 + item[1].visits + item[1].virtual_visits),
                item[1].prior,
            ),
        )

    def _selection_q(self, edge: Edge) -> float:
        effective_visits = edge.visits + edge.virtual_visits
        if effective_visits == 0:
            return 0.0
        # Pending leaves have no network value yet, so zero is the conservative
        # placeholder at the default loss. Positive virtual loss can be tested
        # explicitly, but is never enabled silently.
        virtual_loss = self.config.intra_root_virtual_loss * edge.virtual_visits
        return (edge.value_sum - virtual_loss) / effective_visits

    def _add_dirichlet_noise(self, node: Node) -> None:
        if not node.edges or self.config.dirichlet_fraction == 0:
            return
        noise = self.rng.dirichlet([self.config.dirichlet_alpha] * len(node.edges))
        fraction = self.config.dirichlet_fraction
        for edge, sample in zip(node.edges.values(), noise, strict=True):
            edge.prior = (1 - fraction) * edge.prior + fraction * float(sample)


def choose_move(
    result: SearchResult,
    *,
    temperature: float,
    rng: np.random.Generator,
) -> str:
    moves = list(result.policy)
    if temperature <= 1e-8:
        return result.best_move
    probabilities = np.asarray([result.policy[move] for move in moves], dtype=np.float64)
    probabilities = probabilities ** (1.0 / temperature)
    probabilities /= probabilities.sum()
    return str(rng.choice(moves, p=probabilities))
