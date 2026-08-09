from __future__ import annotations

from rsshogi.core import Board, Move
from rsshogi.types import RepetitionState

from simajilord_shogi.config import SearchConfig
from simajilord_shogi.evaluator import Evaluation, UniformEvaluator
from simajilord_shogi.game import play_game, replay_and_validate
from simajilord_shogi.mcts import MCTS

MATE_IN_ONE_SFEN = "4k4/9/3B5/9/9/9/9/9/4K4 b G 1"
FORCED_REPLY_SFEN = (
    "2p1g1G1l/1r2p2+P1/2g1ng3/1pl1k1p2/p1R1+Bp1+PS/"
    "PP2P1PL1/L1nP1PNPN/1S1p1+s1SP/2K3b2 w p 156"
)


class MateMoveHasTinyPrior:
    def evaluate(self, board: Board) -> Evaluation:
        moves = [move.to_usi() for move in board.legal_moves()]
        policy = {move: 1.0 for move in moves}
        if "G*5b" in policy:
            policy["G*5b"] = 1e-30
        return Evaluation(policy, 0.0)


class RecordingBatchEvaluator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def evaluate(self, board: Board) -> Evaluation:
        return self._evaluation(board)

    def evaluate_batch(self, boards: list[Board]) -> list[Evaluation]:
        self.batch_sizes.append(len(boards))
        return [self._evaluation(board) for board in boards]

    @staticmethod
    def _evaluation(board: Board) -> Evaluation:
        return Evaluation({move.to_usi(): 1.0 for move in board.legal_moves()}, 0.0)


class FixedLeafValueBatchEvaluator(RecordingBatchEvaluator):
    @staticmethod
    def _evaluation(board: Board) -> Evaluation:
        return Evaluation({move.to_usi(): 1.0 for move in board.legal_moves()}, 0.75)


class PreferMoveBatchEvaluator(RecordingBatchEvaluator):
    def __init__(self, preferred_move: str) -> None:
        super().__init__()
        self.preferred_move = preferred_move

    def _evaluation(self, board: Board) -> Evaluation:
        return Evaluation(
            {
                move.to_usi(): 1.0 if move.to_usi() == self.preferred_move else 0.0
                for move in board.legal_moves()
            },
            0.75,
        )


def test_root_coverage_discovers_tiny_prior_mate() -> None:
    board = Board(MATE_IN_ONE_SFEN)
    result = MCTS(
        MateMoveHasTinyPrior(),
        SearchConfig(simulations=1, root_min_visits=1, dirichlet_fraction=0),
        seed=1,
    ).search(board)

    assert result.simulations == len(board.legal_moves())
    assert all(probability > 0 for probability in result.policy.values())
    assert result.best_move == "G*5b"
    assert result.q_values["G*5b"] == 1.0
    assert result.elapsed_seconds > 0
    assert result.nodes_per_second > 0


def test_intra_root_batch_evaluates_multiple_distinct_leaves() -> None:
    evaluator = RecordingBatchEvaluator()
    result = MCTS(
        evaluator,
        SearchConfig(
            simulations=64,
            root_min_visits=0,
            dirichlet_fraction=0,
            intra_root_batch_size=8,
        ),
        seed=7,
    ).search(Board())

    assert evaluator.batch_sizes == [1] + [8] * 8
    assert result.simulations == 64
    assert sum(result.policy.values()) == 1.0
    assert all(-1 <= value <= 1 for value in result.q_values.values())


def test_intra_root_collision_does_not_fabricate_a_forced_line_simulation() -> None:
    board = Board(FORCED_REPLY_SFEN)
    assert len(board.legal_moves()) == 1
    evaluator = RecordingBatchEvaluator()

    result = MCTS(
        evaluator,
        SearchConfig(
            simulations=1,
            root_min_visits=0,
            dirichlet_fraction=0,
            intra_root_batch_size=16,
        ),
    ).search(board)

    # Root expansion and the one real simulation each evaluate one board. The
    # other 15 slots collide with the reserved leaf and are not counted.
    assert evaluator.batch_sizes == [1, 1]
    assert result.simulations == 1
    assert result.policy == {board.legal_moves()[0].to_usi(): 1.0}


def test_intra_root_batch_preserves_root_coverage_and_tactical_mate() -> None:
    board = Board(MATE_IN_ONE_SFEN)
    result = MCTS(
        MateMoveHasTinyPrior(),
        SearchConfig(
            simulations=1,
            root_min_visits=1,
            dirichlet_fraction=0,
            intra_root_batch_size=16,
        ),
        seed=1,
    ).search(board)

    assert result.simulations == len(board.legal_moves())
    assert result.best_move == "G*5b"
    assert result.q_values["G*5b"] == 1.0


def test_batched_leaf_value_is_negated_back_to_the_root_perspective() -> None:
    result = MCTS(
        FixedLeafValueBatchEvaluator(),
        SearchConfig(
            simulations=1,
            root_min_visits=0,
            dirichlet_fraction=0,
            intra_root_batch_size=8,
        ),
    ).search(Board())

    assert result.root_value == -0.75
    assert result.q_values[result.best_move] == -0.75


def test_batched_terminal_repetition_is_backed_up_as_a_draw() -> None:
    board = Board("4k4/9/9/9/9/9/9/9/4K4 b - 1")
    cycle = ("5i5h", "5a5b", "5h5i", "5b5a")
    for move_usi in cycle * 2 + cycle[:3]:
        board.apply_move(Move.from_usi(move_usi))
    assert board.repetition_state() == RepetitionState.NONE

    result = MCTS(
        PreferMoveBatchEvaluator("5b5a"),
        SearchConfig(
            simulations=1,
            root_min_visits=0,
            dirichlet_fraction=0,
            intra_root_batch_size=8,
        ),
    ).search(board)

    assert result.best_move == "5b5a"
    assert result.q_values["5b5a"] == 0.0
    assert result.root_value == 0.0


def test_batch_size_one_retains_the_sequential_search_snapshot() -> None:
    result = MCTS(
        UniformEvaluator(),
        SearchConfig(
            simulations=40,
            root_min_visits=0,
            dirichlet_fraction=0,
            intra_root_batch_size=1,
        ),
        seed=7,
    ).search(Board())

    assert result.best_move == "1g1f"
    assert result.discovery_simulation == 1
    assert result.root_value == 0.0
    assert result.policy["1g1f"] == 0.05
    assert result.policy["2h1h"] == 0.025


def test_non_batch_evaluator_keeps_sequential_semantics() -> None:
    sequential = MCTS(
        UniformEvaluator(),
        SearchConfig(
            simulations=80,
            root_min_visits=0,
            dirichlet_fraction=0,
            intra_root_batch_size=1,
        ),
    ).search(Board())
    requested_batch = MCTS(
        UniformEvaluator(),
        SearchConfig(
            simulations=80,
            root_min_visits=0,
            dirichlet_fraction=0,
            intra_root_batch_size=16,
        ),
    ).search(Board())

    assert requested_batch.policy == sequential.policy
    assert requested_batch.q_values == sequential.q_values
    assert requested_batch.best_move == sequential.best_move


def test_intra_root_batch_is_seed_deterministic_with_root_noise() -> None:
    config = SearchConfig(
        simulations=64,
        root_min_visits=0,
        intra_root_batch_size=8,
    )

    first = MCTS(UniformEvaluator(), config, seed=91).search(Board(), add_root_noise=True)
    second = MCTS(UniformEvaluator(), config, seed=91).search(Board(), add_root_noise=True)

    assert first.policy == second.policy
    assert first.q_values == second.q_values
    assert first.best_move == second.best_move
    assert first.discovery_simulation == second.discovery_simulation


def test_neural_batch_cap_bounds_existing_multi_root_batching() -> None:
    evaluator = RecordingBatchEvaluator()
    results = MCTS(
        evaluator,
        SearchConfig(
            simulations=4,
            root_min_visits=0,
            dirichlet_fraction=0,
            intra_root_batch_size=8,
            max_evaluation_batch_size=3,
        ),
    ).search_many([Board(), Board(MATE_IN_ONE_SFEN)])

    assert len(results) == 2
    assert all(result.simulations == 4 for result in results)
    assert max(evaluator.batch_sizes) <= 3
    assert any(batch_size > 1 for batch_size in evaluator.batch_sizes)


def test_search_config_validates_intra_root_batch_controls() -> None:
    for field, value in (
        ("intra_root_batch_size", 0),
        ("intra_root_virtual_loss", -0.1),
        ("intra_root_virtual_loss", 1.1),
        ("max_evaluation_batch_size", 0),
    ):
        try:
            SearchConfig(**{field: value})
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid {field}={value} was accepted")


def test_game_reaches_checkmate_and_replays_legally() -> None:
    record = play_game(
        UniformEvaluator(),
        SearchConfig(
            simulations=1,
            root_min_visits=1,
            max_plies=4,
            temperature_moves=0,
            resign_threshold=None,
            dirichlet_fraction=0,
        ),
        initial_sfen=MATE_IN_ONE_SFEN,
        self_play_noise=False,
    )

    final_board = replay_and_validate(record)
    assert record.termination.value == "checkmate"
    assert record.winner == 0
    assert record.moves == ("G*5b",)
    assert final_board.is_mated()
    assert record.samples[0].value_target == 1.0
