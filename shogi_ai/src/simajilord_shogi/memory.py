"""Unified-memory budgeting for MLX inference, training, and Python search trees."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass

import mlx.core as mx
import psutil

GIB = 1024**3
ESTIMATED_PYTHON_TREE_NODE_BYTES = 4096


@dataclass(frozen=True, slots=True)
class MemoryBudget:
    """A conservative high-water mark derived from current, not nominal, memory."""

    total_bytes: int
    available_bytes: int
    process_rss_bytes: int
    reserve_bytes: int
    growth_budget_bytes: int
    mlx_active_bytes: int
    mlx_memory_limit_bytes: int
    mlx_cache_limit_bytes: int
    tree_budget_bytes: int
    tree_node_limit: int
    target_fraction: float
    tree_fraction: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def automatic_memory_budget(
    *,
    target_fraction: float = 0.82,
    reserve_fraction: float = 0.15,
    minimum_reserve_bytes: int = 8 * GIB,
    tree_fraction: float = 0.35,
    estimated_tree_node_bytes: int = ESTIMATED_PYTHON_TREE_NODE_BYTES,
) -> MemoryBudget:
    """Use most available unified memory while retaining an OS/app safety reserve."""

    if not 0.5 <= target_fraction < 0.95:
        raise ValueError("target_fraction must be in [0.5, 0.95)")
    if not 0.05 <= reserve_fraction < 0.5:
        raise ValueError("reserve_fraction must be in [0.05, 0.5)")
    if not 0.05 <= tree_fraction <= 0.7:
        raise ValueError("tree_fraction must be in [0.05, 0.7]")
    if minimum_reserve_bytes < GIB or estimated_tree_node_bytes < 256:
        raise ValueError("memory reserve or tree-node estimate is implausibly small")

    virtual = psutil.virtual_memory()
    total = int(virtual.total)
    available = int(virtual.available)
    process_rss = int(psutil.Process().memory_info().rss)
    reserve = max(minimum_reserve_bytes, int(total * reserve_fraction))
    process_ceiling = int(total * target_fraction)
    growth_from_system = max(0, available - reserve)
    growth_from_process = max(0, process_ceiling - process_rss)
    growth_budget = min(growth_from_system, growth_from_process)
    mlx_active = int(mx.get_active_memory())

    # Keep independent room for Python/Rust board state, temporary tensors, and
    # the search tree. MLX receives the remaining high-water allowance.
    tree_budget = int(growth_budget * tree_fraction)
    mlx_growth = int(growth_budget * (1.0 - tree_fraction) * 0.85)
    mlx_limit = max(mlx_active + 256 * 1024**2, mlx_active + mlx_growth)
    mlx_limit = min(mlx_limit, process_ceiling)
    cache_limit = min(2 * GIB, max(256 * 1024**2, mlx_limit // 10))
    tree_node_limit = max(1024, tree_budget // estimated_tree_node_bytes)
    return MemoryBudget(
        total_bytes=total,
        available_bytes=available,
        process_rss_bytes=process_rss,
        reserve_bytes=reserve,
        growth_budget_bytes=growth_budget,
        mlx_active_bytes=mlx_active,
        mlx_memory_limit_bytes=mlx_limit,
        mlx_cache_limit_bytes=cache_limit,
        tree_budget_bytes=tree_budget,
        tree_node_limit=tree_node_limit,
        target_fraction=target_fraction,
        tree_fraction=tree_fraction,
    )


def configure_mlx_memory(budget: MemoryBudget | None = None) -> MemoryBudget:
    """Apply MLX active/cache limits before a large allocation is attempted."""

    selected = budget or automatic_memory_budget()
    mx.set_memory_limit(selected.mlx_memory_limit_bytes)
    mx.set_cache_limit(selected.mlx_cache_limit_bytes)
    return selected


class SearchMemoryGuard:
    """Poll current pressure and request bounded tree recycling before OOM."""

    def __init__(self, *, explicit_node_limit: int | None = None) -> None:
        self.budget = automatic_memory_budget()
        self.node_limit = explicit_node_limit or self.budget.tree_node_limit
        if self.node_limit < 1024:
            raise ValueError("search tree node limit must be at least 1024")

    def should_recycle(self, logical_nodes: int) -> bool:
        if logical_nodes >= self.node_limit:
            return True
        return int(psutil.virtual_memory().available) <= self.budget.reserve_bytes

    @staticmethod
    def reclaim() -> None:
        mx.clear_cache()
        gc.collect()
