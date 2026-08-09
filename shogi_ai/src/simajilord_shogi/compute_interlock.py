"""Cycle-free trainer entry point for the human-play compute interlock.

The concrete interlock lives with the ShogiHome bridge because it owns both
human and training leases.  ``checkpoint`` already imports ``trainer``, so a
top-level trainer import of ``human_gui`` would create a runtime cycle.  This
small adapter delays that import until an optimizer step is about to run, after
module initialization has completed, while still returning the same concrete
lease implementation and using the same on-disk lock.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Protocol, cast

from rsshogi.core import Board

from .evaluator import Evaluation, Evaluator


class ComputeLeaseSettings(Protocol):
    """Runtime settings shared by training and background neural inference."""

    @property
    def state_root(self) -> Path: ...

    @property
    def generation(self) -> int | None: ...

    @property
    def ttl_seconds(self) -> float: ...

    @property
    def heartbeat_interval_seconds(self) -> float: ...

    @property
    def wait_timeout_seconds(self) -> float: ...

    @property
    def poll_interval_seconds(self) -> float: ...


class TrainingStepLease(Protocol):
    """The narrow context-manager contract needed by the trainer."""

    def assert_healthy(self) -> None: ...

    def release(self) -> None: ...

    def __enter__(self) -> TrainingStepLease: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


def training_step(
    state_root: Path,
    *,
    generation: int | None = None,
    ttl_seconds: float = 60.0,
    heartbeat_interval_seconds: float = 5.0,
    wait_timeout_seconds: float = 0.0,
    poll_interval_seconds: float = 0.05,
) -> TrainingStepLease:
    """Acquire one TOCTOU-safe lease around exactly one optimizer step.

    Importing here, rather than at module import time, avoids the existing
    ``checkpoint -> trainer`` dependency cycle.  Corrupt or symlinked lease
    state raises instead of allowing the optimizer to continue.
    """

    from .human_gui import ComputeInterlock

    return ComputeInterlock(state_root).training_step(
        generation=generation,
        ttl_seconds=ttl_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        wait_timeout_seconds=wait_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )


class InterlockedEvaluator:
    """Serialize background neural inference and yield promptly to human play.

    MCTS may call either ``evaluate`` or ``evaluate_batch``.  One lease covers
    exactly one neural batch, so a waiting ShogiHome game gets priority between
    batches instead of waiting for an entire self-play game or arena suite.
    """

    def __init__(self, evaluator: Evaluator, settings: ComputeLeaseSettings) -> None:
        self.evaluator = evaluator
        self.settings = settings

    def evaluate(self, board: Board) -> Evaluation:
        return self.evaluate_batch([board])[0]

    def evaluate_batch(self, boards: list[Board]) -> list[Evaluation]:
        if not boards:
            return []
        with training_step(
            self.settings.state_root,
            generation=self.settings.generation,
            ttl_seconds=self.settings.ttl_seconds,
            heartbeat_interval_seconds=self.settings.heartbeat_interval_seconds,
            wait_timeout_seconds=self.settings.wait_timeout_seconds,
            poll_interval_seconds=self.settings.poll_interval_seconds,
        ) as lease:
            batch_method = getattr(self.evaluator, "evaluate_batch", None)
            if callable(batch_method):
                evaluations = cast(
                    Callable[[list[Board]], list[Evaluation]], batch_method
                )(boards)
            else:
                evaluations = [self.evaluator.evaluate(board) for board in boards]
            lease.assert_healthy()
        if len(evaluations) != len(boards):
            raise RuntimeError("interlocked evaluator returned the wrong batch size")
        return evaluations
