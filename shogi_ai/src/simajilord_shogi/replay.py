"""Versioned JSONL replay storage."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

from .domain import GameRecord, PositionSample

SCHEMA_VERSION = 1


def append_games(path: Path, games: Iterable[GameRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as stream:
        for game in games:
            stream.write(json.dumps({"schema": SCHEMA_VERSION, "game": game.to_dict()}))
            stream.write("\n")
            written += 1
    return written


def load_games(path: Path) -> list[GameRecord]:
    games: list[GameRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            payload = json.loads(line)
            if payload.get("schema") != SCHEMA_VERSION:
                raise ValueError(f"unsupported schema at {path}:{line_number}")
            games.append(GameRecord.from_dict(payload["game"]))
    return games


def position_samples(
    games: Iterable[GameRecord], *, include_incomplete: bool = False
) -> list[PositionSample]:
    return [
        sample
        for game in games
        if include_incomplete or game.termination.value != "max_plies"
        for sample in game.samples
    ]


def largest_blunders(games: Iterable[GameRecord], *, limit: int = 20) -> list[PositionSample]:
    candidates = [
        sample for game in games for sample in game.samples if sample.teacher_regret is not None
    ]
    return sorted(
        candidates,
        key=lambda sample: sample.teacher_regret or 0.0,
        reverse=True,
    )[:limit]


def search_telemetry(games: Iterable[GameRecord]) -> dict[str, dict[str, float | int]]:
    """Aggregate comparable node/time/NPS counters by the actor that produced them."""

    totals: dict[str, dict[str, float | int]] = {}
    for game in games:
        for sample in game.samples:
            if sample.actor_simulations is None or sample.actor_search_seconds is None:
                continue
            source = sample.actor_source or "unknown"
            row = totals.setdefault(source, {"moves": 0, "nodes": 0, "seconds": 0.0})
            row["moves"] = int(row["moves"]) + 1
            row["nodes"] = int(row["nodes"]) + sample.actor_simulations
            row["seconds"] = float(row["seconds"]) + sample.actor_search_seconds
            row["peak_tree_nodes"] = max(
                int(row.get("peak_tree_nodes", 0)), sample.actor_peak_tree_nodes or 0
            )
            row["tree_recycles"] = int(row.get("tree_recycles", 0)) + sample.actor_tree_recycles
    for row in totals.values():
        seconds = float(row["seconds"])
        row["nps"] = int(row["nodes"]) / seconds if seconds > 0 else 0.0
    return totals


def evaluation_value_telemetry(
    games: Iterable[GameRecord],
) -> dict[str, dict[str, float | int]]:
    """Summarize side-to-move root values separately for every acting engine."""

    values_by_source: dict[str, list[float]] = {}
    for game in games:
        for sample in game.samples:
            value = float(sample.root_value)
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(
                    f"actor root value must be finite and within [-1, 1], got {value}"
                )
            values_by_source.setdefault(sample.actor_source or "unknown", []).append(value)
    result: dict[str, dict[str, float | int]] = {}
    for source, values in sorted(values_by_source.items()):
        count = len(values)
        mean = sum(values) / count
        result[source] = {
            "positions": count,
            "mean": mean,
            "mean_absolute": sum(abs(value) for value in values) / count,
            "root_mean_square": math.sqrt(sum(value * value for value in values) / count),
            "minimum": min(values),
            "maximum": max(values),
            "negative": sum(value < 0.0 for value in values),
            "zero": sum(value == 0.0 for value in values),
            "positive": sum(value > 0.0 for value in values),
        }
    return result
