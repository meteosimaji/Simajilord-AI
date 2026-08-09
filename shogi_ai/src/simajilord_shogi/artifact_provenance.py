"""Deterministic file identities for create-only research artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """A resolved input file and the bytes that were actually consumed."""

    path: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InputArtifact:
    """A primary input plus adjacent lineage files that constrain its meaning."""

    file: FileIdentity
    adjacent_lineage: tuple[FileIdentity, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    """Hash a regular file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identify_file(path: Path) -> FileIdentity:
    """Resolve and identify one existing regular file."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return FileIdentity(
        path=str(resolved),
        sha256=sha256_file(resolved),
        bytes=resolved.stat().st_size,
    )


def identify_input_artifact(
    path: Path,
    *,
    adjacent_suffixes: tuple[str, ...] = (
        ".ensemble.json",
        ".provenance.json",
        ".disagreement.json",
        ".arbitration.json",
    ),
) -> InputArtifact:
    """Identify a replay and every known adjacent lineage sidecar that exists."""

    primary = identify_file(path)
    primary_path = Path(primary.path)
    lineage: list[FileIdentity] = []
    seen: set[str] = set()
    for suffix in adjacent_suffixes:
        sidecar = primary_path.with_suffix(primary_path.suffix + suffix)
        normalized = str(sidecar).casefold()
        if normalized in seen:
            raise ValueError(f"duplicate adjacent lineage path: {sidecar}")
        seen.add(normalized)
        if sidecar.exists():
            if not sidecar.is_file():
                raise ValueError(f"adjacent lineage path is not a file: {sidecar}")
            lineage.append(identify_file(sidecar))
    return InputArtifact(file=primary, adjacent_lineage=tuple(lineage))


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    """Hash one JSON value using stable keys, separators, and UTF-8 encoding."""

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
