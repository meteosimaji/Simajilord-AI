from __future__ import annotations

import pytest
from rsshogi.core import Board

from simajilord_shogi.config import SearchConfig
from simajilord_shogi.evaluator import Evaluation, UniformEvaluator
from simajilord_shogi.mcts import MCTS
from simajilord_shogi.memory import automatic_memory_budget


class BatchUniformEvaluator(UniformEvaluator):
    def evaluate_batch(self, boards: list[Board]) -> list[Evaluation]:
        return [self.evaluate(board) for board in boards]


def test_automatic_memory_budget_retains_reserve_and_bounds_mlx_and_tree() -> None:
    budget = automatic_memory_budget()

    assert 0 < budget.available_bytes <= budget.total_bytes
    assert budget.reserve_bytes >= 8 * 1024**3
    assert budget.growth_budget_bytes <= max(0, budget.available_bytes - budget.reserve_bytes)
    assert budget.mlx_memory_limit_bytes <= int(budget.total_bytes * budget.target_fraction)
    assert budget.tree_node_limit >= 1024


def test_search_recycles_tree_at_explicit_high_water_mark_without_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = MCTS(
        BatchUniformEvaluator(),
        SearchConfig(
            simulations=1100,
            root_min_visits=0,
            dirichlet_fraction=0,
            max_tree_nodes=1024,
            memory_check_interval=1,
            intra_root_batch_size=16,
        ),
    )
    # This test isolates the explicit logical-node high-water mark. CI runners
    # can start below the independent system-memory reserve, which legitimately
    # causes the production guard to recycle earlier and made this assertion
    # depend on host pressure rather than the configured node limit.
    monkeypatch.setattr(
        search.memory_guard,
        "should_recycle",
        lambda logical_nodes: logical_nodes >= search.memory_guard.node_limit,
    )
    result = search.search(Board())

    assert result.simulations == 1100
    assert result.peak_tree_nodes >= 1024
    assert result.tree_recycles >= 1
