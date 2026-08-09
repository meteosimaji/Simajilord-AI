"""Immutable, reproducible opening splits for self-improvement and arena play."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rsshogi.core import Board

from .ensemble import normalized_sfen

OPENING_SUITE_SCHEMA = "meteo-opening-suite-v1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _sha256_bytes(serialized.encode("utf-8"))


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"opening suite contains duplicate JSON object key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class OpeningPosition:
    """One canonical position whose identity deliberately excludes the move counter."""

    sfen: str
    normalized_key: str
    sha256: str

    @classmethod
    def from_sfen(cls, sfen: str) -> OpeningPosition:
        key = normalized_sfen(sfen)
        return cls(
            sfen=Board(sfen).to_sfen(),
            normalized_key=key,
            sha256=_sha256_bytes(key.encode("utf-8")),
        )

    def to_manifest(self) -> dict[str, str]:
        return {
            "sfen": self.sfen,
            "normalized_key": self.normalized_key,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class OpeningSplit:
    name: str
    positions: tuple[OpeningPosition, ...]
    sha256: str

    def to_manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "count": len(self.positions),
            "sha256": self.sha256,
            "positions": [position.to_manifest() for position in self.positions],
        }


@dataclass(frozen=True, slots=True)
class OpeningSuite:
    """A read-only JSON opening suite pinned by both source and normalized hashes."""

    source: Path
    source_sha256: str
    normalized_sha256: str
    splits: tuple[OpeningSplit, ...]

    def split(self, name: str) -> OpeningSplit:
        matches = tuple(split for split in self.splits if split.name == name)
        if not matches:
            available = ", ".join(split.name for split in self.splits)
            raise ValueError(f"opening split {name!r} is missing; available splits: {available}")
        if len(matches) != 1:
            raise AssertionError(f"opening suite contains duplicate split name {name!r}")
        return matches[0]

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema": OPENING_SUITE_SCHEMA,
            "source": str(self.source),
            "source_sha256": self.source_sha256,
            "normalized_sha256": self.normalized_sha256,
            "splits": [split.to_manifest() for split in self.splits],
        }


def load_opening_suite(path: Path) -> OpeningSuite:
    """Load a strict JSON suite without creating or changing it.

    Format::

        {"schema": "meteo-opening-suite-v1",
         "splits": {"actor": ["... SFEN ..."], "arena": ["... SFEN ..."]}}

    A normalized position may occur only once in the entire document, including
    across splits. The move counter is not part of a position's identity.
    """

    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError(f"opening suite does not exist or is not readable: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"opening suite is not a regular file: {resolved}")
    source_bytes = resolved.read_bytes()
    try:
        payload: Any = json.loads(
            source_bytes,
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"opening suite must be valid UTF-8 JSON: {resolved}") from error
    if not isinstance(payload, dict):
        raise ValueError("opening suite root must be a JSON object")
    unknown_fields = set(payload) - {"schema", "splits"}
    if unknown_fields:
        raise ValueError(f"opening suite has unknown fields: {sorted(unknown_fields)}")
    if payload.get("schema") != OPENING_SUITE_SCHEMA:
        raise ValueError(f"opening suite schema must be {OPENING_SUITE_SCHEMA!r}")
    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict) or not raw_splits:
        raise ValueError("opening suite splits must be a non-empty JSON object")

    positions_by_key: dict[str, tuple[str, int]] = {}
    splits: list[OpeningSplit] = []
    normalized_document: dict[str, list[str]] = {}
    for raw_name, raw_positions in raw_splits.items():
        if not isinstance(raw_name, str) or not raw_name.strip() or raw_name != raw_name.strip():
            raise ValueError("opening split names must be non-empty trimmed strings")
        if not isinstance(raw_positions, list) or not raw_positions:
            raise ValueError(f"opening split {raw_name!r} must be a non-empty JSON array")
        positions: list[OpeningPosition] = []
        for index, raw_sfen in enumerate(raw_positions):
            if not isinstance(raw_sfen, str):
                raise ValueError(f"opening split {raw_name!r} entry {index} must be an SFEN string")
            position = OpeningPosition.from_sfen(raw_sfen)
            previous = positions_by_key.get(position.normalized_key)
            if previous is not None:
                previous_split, previous_index = previous
                raise ValueError(
                    "duplicate normalized opening SFEN "
                    f"in {previous_split!r}[{previous_index}] and {raw_name!r}[{index}]: "
                    f"{position.normalized_key}"
                )
            positions_by_key[position.normalized_key] = (raw_name, index)
            positions.append(position)
        keys = [position.normalized_key for position in positions]
        normalized_document[raw_name] = keys
        splits.append(
            OpeningSplit(
                name=raw_name,
                positions=tuple(positions),
                sha256=_stable_sha256(keys),
            )
        )

    return OpeningSuite(
        source=resolved,
        source_sha256=_sha256_bytes(source_bytes),
        normalized_sha256=_stable_sha256(
            {
                "schema": OPENING_SUITE_SCHEMA,
                "splits": normalized_document,
            }
        ),
        splits=tuple(splits),
    )
