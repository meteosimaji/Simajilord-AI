"""Lazy readers for public teacher formats that do not fit in memory."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, overload

import numpy as np
from rsshogi.core import Board, Move
from rsshogi.numpy import PackedSfenValue

from .domain import PositionSample

PSV_RECORD_BYTES = 40


class PackedSfenValueDataset(Sequence[PositionSample]):
    """Memory-mapped YaneuraOu PSV records exposed as teacher samples.

    PSV stores a Move16 and a result from the current side-to-move's perspective:
    +1 win, -1 loss, and 0 draw. The score is also side-to-move relative.
    ``score_scale`` is the raw denominator D in ``tanh(cp / D)``; the default
    1200 is Ponanza coefficient C=600 expressed as the signed-value denominator
    D=2C.
    """

    teacher_only = True

    def __init__(
        self,
        path: Path,
        *,
        source_name: str,
        offset: int = 0,
        stride: int = 1,
        limit: int | None = None,
        score_scale: float = 1_200.0,
        evaluation_weight: float = 0.5,
    ) -> None:
        size = path.stat().st_size
        if size == 0 or size % PSV_RECORD_BYTES:
            raise ValueError(f"PSV size must be a positive multiple of {PSV_RECORD_BYTES}: {path}")
        if not source_name.strip():
            raise ValueError("source_name must not be empty")
        if offset < 0 or stride < 1 or (limit is not None and limit < 1):
            raise ValueError("offset, stride, and limit must select a non-empty range")
        if score_scale <= 0 or not 0 <= evaluation_weight <= 1:
            raise ValueError("score_scale must be positive and evaluation_weight must be in [0, 1]")
        record_count = size // PSV_RECORD_BYTES
        if offset >= record_count:
            raise ValueError("offset is outside the PSV file")
        available = (record_count - offset + stride - 1) // stride
        self.path = path
        self.source_name = source_name
        self.offset = offset
        self.stride = stride
        self.score_scale = score_scale
        self.evaluation_weight = evaluation_weight
        self._length = min(available, limit) if limit is not None else available
        self._records: Any = np.memmap(path, dtype=PackedSfenValue, mode="r")

    def __len__(self) -> int:
        return self._length

    @overload
    def __getitem__(self, index: int) -> PositionSample: ...

    @overload
    def __getitem__(self, index: slice) -> list[PositionSample]: ...

    def __getitem__(self, index: int | slice) -> PositionSample | list[PositionSample]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        record = self._records[self.offset + index * self.stride]
        board = Board()
        board.set_packed_sfen(record["sfen"].tobytes())
        move16 = int(record["move"])
        if not board.is_valid():
            raise ValueError(f"invalid packed position at PSV sample {index}")
        if move16 == 0:
            raise ValueError(
                "PSV sample has Move16=0 and is value-only; the policy-supervised "
                "PackedSfenValueDataset must not invent a move target"
            )
        move = Move(move16)
        if not board.is_legal_move(move):
            raise ValueError(
                f"illegal Move16 {move16} at PSV sample {index}: {move.to_usi()}"
            )
        game_result = int(record["game_result"])
        if game_result not in {-1, 0, 1}:
            raise ValueError(
                f"invalid side-to-move game_result {game_result} at PSV sample {index}"
            )
        score_value = math.tanh(int(record["score"]) / self.score_scale)
        teacher_value = (
            self.evaluation_weight * score_value + (1.0 - self.evaluation_weight) * game_result
        )
        move_usi = move.to_usi()
        return PositionSample(
            sfen=board.to_sfen(),
            ply=int(record["game_ply"]),
            turn=int(board.turn),
            policy={move_usi: 1.0},
            root_value=score_value,
            value_target=float(game_result),
            teacher_policy={move_usi: 1.0},
            teacher_value=teacher_value,
            chosen_move=move_usi,
            actor_best_move=move_usi,
            teacher_best_move=move_usi,
            teacher_source=self.source_name,
            teacher_value_scale=self.score_scale,
        )

    def __iter__(self) -> Iterator[PositionSample]:
        for index in range(len(self)):
            yield self[index]
