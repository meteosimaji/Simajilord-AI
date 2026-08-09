"""ShogiHome bridge and a local debug GUI for checkpoint-pinned human play.

The production UI is an unmodified upstream ShogiHome process talking to
:class:`ActiveCheckpointUsiEngine`.  The small HTTP board in this module is a
diagnostic fallback, not a replacement ShogiHome fork.  Neither path imports
the trainer or mutates its model.  A complete checkpoint is published through
:class:`CheckpointRegistry`; each game pins one verified identity and keeps it
until the game ends.  A later promotion is only visible to a newly-created game.
"""

# The module contains one self-contained HTML/CSS/JS asset.  Long lines and the
# Japanese multiplication glyph below belong to that browser asset, not Python.
# ruff: noqa: E501, RUF001

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import secrets
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

from rsshogi.core import Board, Move
from rsshogi.types import Color, RepetitionState

from .arena import DirectPlayer, MctsPlayer
from .checkpoint import load_checkpoint
from .config import SearchConfig
from .evaluator import Evaluator
from .model import MLXEvaluator
from .usi import UsiEngine

CHECKPOINT_POINTER_SCHEMA = "meteo-human-gui-checkpoint-pointer-v1"
CHECKPOINT_PUBLICATION_SCHEMA = "meteo-human-gui-checkpoint-publication-v1"
TRAINING_STATUS_SCHEMA = "meteo-human-gui-training-status-v1"
HUMAN_PLAY_LEASE_SCHEMA = "meteo-human-play-lease-v1"
TRAINING_STEP_LEASE_SCHEMA = "meteo-training-step-lease-v1"
SESSION_SCHEMA = "meteo-human-gui-session-v1"
HUMAN_GAME_SCHEMA = "meteo-human-game-candidate-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SESSION_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
_MAX_JSON_BODY_BYTES = 32 * 1024
_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _strict_json_loads(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value!r} in {label}")

    try:
        value: Any = json.loads(
            raw,
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid UTF-8 JSON in {label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return cast(dict[str, Any], value)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON object durably without exposing a partial destination."""

    if path.is_symlink():
        raise ValueError(f"refusing to replace symlinked artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing symlinked lock: {path}")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_json_file(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"refusing symlinked {label}: {path}")
    try:
        return _strict_json_loads(path.read_bytes(), label=label)
    except OSError as error:
        raise ValueError(f"cannot read {label}: {path}") from error


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_finite_number(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _reject_symlink_components(root: Path, child: Path) -> None:
    """Reject traversal and every symlink below a trusted, real root."""

    root = root.resolve(strict=True)
    absolute = child.expanduser().absolute()
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes configured root: {child}") from error
    current = root
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise ValueError(f"invalid path component in {child}")
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlinked path component is forbidden: {current}")


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    generation: int
    relative_path: str
    checkpoint_sha256: str
    metadata_sha256: str
    weights_sha256: str
    step: int
    published_unix_seconds: float

    def __post_init__(self) -> None:
        if self.generation < 0 or self.step < 0:
            raise ValueError("checkpoint generation and step must be non-negative")
        relative = Path(self.relative_path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("checkpoint identity path must be a safe relative path")
        for label, digest in (
            ("checkpoint", self.checkpoint_sha256),
            ("metadata", self.metadata_sha256),
            ("weights", self.weights_sha256),
        ):
            _require_sha256(digest, label=f"{label} identity")
        _require_finite_number(self.published_unix_seconds, label="publication time")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointIdentity:
        return cls(
            generation=int(value["generation"]),
            relative_path=str(value["relative_path"]),
            checkpoint_sha256=str(value["checkpoint_sha256"]),
            metadata_sha256=str(value["metadata_sha256"]),
            weights_sha256=str(value["weights_sha256"]),
            step=int(value["step"]),
            published_unix_seconds=float(value["published_unix_seconds"]),
        )


class CheckpointRegistry:
    """Atomic active-checkpoint pointer shared by trainer and GUI processes."""

    def __init__(self, checkpoint_root: Path, state_root: Path) -> None:
        configured_checkpoint_root = checkpoint_root.expanduser().absolute()
        configured_state_root = state_root.expanduser().absolute()
        if configured_checkpoint_root.is_symlink():
            raise ValueError("checkpoint_root must not be a symlink")
        if configured_state_root.is_symlink():
            raise ValueError("state_root must not be a symlink")
        self.checkpoint_root = configured_checkpoint_root.resolve(strict=True)
        self.state_root = configured_state_root.resolve()
        if not self.checkpoint_root.is_dir():
            raise ValueError("checkpoint_root must be a real directory")
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.publications = self.state_root / "checkpoint-publications"
        self.active_path = self.state_root / "active-checkpoint.json"
        self.lock_path = self.state_root / ".checkpoint-registry.lock"

    def _checkpoint_path(self, relative_path: str) -> Path:
        candidate = self.checkpoint_root / relative_path
        _reject_symlink_components(self.checkpoint_root, candidate)
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(self.checkpoint_root)
        except ValueError as error:
            raise ValueError("checkpoint path escapes configured root") from error
        if not resolved.is_dir():
            raise ValueError(f"checkpoint is not a directory: {resolved}")
        return resolved

    def identify(self, checkpoint: Path, *, generation: int) -> CheckpointIdentity:
        if generation < 0:
            raise ValueError("generation must be non-negative")
        absolute = checkpoint.expanduser().absolute()
        _reject_symlink_components(self.checkpoint_root, absolute)
        resolved = absolute.resolve(strict=True)
        try:
            relative = resolved.relative_to(self.checkpoint_root)
        except ValueError as error:
            raise ValueError("checkpoint is outside checkpoint_root") from error
        if any(part.startswith(".") and ".tmp" in part for part in relative.parts):
            raise ValueError("temporary checkpoint directories cannot be published")
        metadata_path = resolved / "metadata.json"
        weights_path = resolved / "weights.safetensors"
        if metadata_path.is_symlink() or weights_path.is_symlink():
            raise ValueError("checkpoint child symlinks are forbidden")
        if not metadata_path.is_file() or not weights_path.is_file():
            raise ValueError("checkpoint is incomplete: metadata and weights are required")
        if weights_path.stat().st_size <= 0:
            raise ValueError("checkpoint weights file is empty")
        metadata = _read_json_file(metadata_path, label="checkpoint metadata")
        step = int(metadata.get("step", -1))
        if step < 0 or not isinstance(metadata.get("model"), dict):
            raise ValueError("checkpoint metadata is incomplete")
        metadata_sha256 = _sha256_file(metadata_path)
        weights_sha256 = _sha256_file(weights_path)
        checkpoint_sha256 = hashlib.sha256(
            f"{metadata_sha256}:{weights_sha256}".encode()
        ).hexdigest()
        return CheckpointIdentity(
            generation=generation,
            relative_path=relative.as_posix(),
            checkpoint_sha256=checkpoint_sha256,
            metadata_sha256=metadata_sha256,
            weights_sha256=weights_sha256,
            step=step,
            published_unix_seconds=time.time(),
        )

    def verify(self, identity: CheckpointIdentity) -> Path:
        checkpoint = self._checkpoint_path(identity.relative_path)
        metadata_path = checkpoint / "metadata.json"
        weights_path = checkpoint / "weights.safetensors"
        if (
            metadata_path.is_symlink()
            or weights_path.is_symlink()
            or not metadata_path.is_file()
            or not weights_path.is_file()
        ):
            raise ValueError("pinned checkpoint is incomplete or symlinked")
        observed_metadata = _sha256_file(metadata_path)
        observed_weights = _sha256_file(weights_path)
        observed_checkpoint = hashlib.sha256(
            f"{observed_metadata}:{observed_weights}".encode()
        ).hexdigest()
        if not (
            hmac.compare_digest(observed_metadata, identity.metadata_sha256)
            and hmac.compare_digest(observed_weights, identity.weights_sha256)
            and hmac.compare_digest(observed_checkpoint, identity.checkpoint_sha256)
        ):
            raise ValueError("pinned checkpoint bytes changed after publication")
        metadata = _read_json_file(metadata_path, label="pinned checkpoint metadata")
        if int(metadata.get("step", -1)) != identity.step:
            raise ValueError("pinned checkpoint step changed after publication")
        return checkpoint

    def publish(self, checkpoint: Path, *, generation: int) -> CheckpointIdentity:
        """Publish a verified checkpoint; active pointer replacement is the last write."""

        identity = self.identify(checkpoint, generation=generation)
        with _exclusive_file_lock(self.lock_path):
            if self.active_path.exists():
                current = self.read_active()
                if (
                    identity.generation == current.generation
                    and identity.relative_path == current.relative_path
                    and identity.checkpoint_sha256 == current.checkpoint_sha256
                    and identity.step == current.step
                ):
                    return current
                if generation <= current.generation:
                    raise ValueError("checkpoint generation must increase monotonically")
            self.publications.mkdir(parents=True, exist_ok=True)
            publication_name = (
                f"generation-{generation:08d}-{identity.checkpoint_sha256[:16]}.json"
            )
            publication_path = self.publications / publication_name
            publication: dict[str, Any] = {
                "schema": CHECKPOINT_PUBLICATION_SCHEMA,
                "identity": identity.to_dict(),
            }
            publication["content_sha256"] = _payload_sha256(publication)
            if publication_path.exists():
                existing = _read_json_file(
                    publication_path, label="checkpoint publication"
                )
                if existing != publication:
                    raise FileExistsError("different checkpoint publication already exists")
            else:
                _atomic_replace_json(publication_path, publication)
            pointer: dict[str, Any] = {
                "schema": CHECKPOINT_POINTER_SCHEMA,
                "publication": publication_name,
                "generation": generation,
                "checkpoint_sha256": identity.checkpoint_sha256,
            }
            pointer["content_sha256"] = _payload_sha256(pointer)
            _atomic_replace_json(self.active_path, pointer)
        return identity

    def read_active(self) -> CheckpointIdentity:
        pointer = _read_json_file(self.active_path, label="active checkpoint pointer")
        if pointer.get("schema") != CHECKPOINT_POINTER_SCHEMA:
            raise ValueError("unsupported active checkpoint pointer")
        pointer_hash = _require_sha256(
            pointer.pop("content_sha256", None), label="active pointer content"
        )
        if not hmac.compare_digest(pointer_hash, _payload_sha256(pointer)):
            raise ValueError("active checkpoint pointer hash mismatch")
        publication_name = pointer.get("publication")
        if (
            not isinstance(publication_name, str)
            or Path(publication_name).name != publication_name
        ):
            raise ValueError("active checkpoint publication filename is invalid")
        publication_path = self.publications / publication_name
        publication = _read_json_file(publication_path, label="checkpoint publication")
        if publication.get("schema") != CHECKPOINT_PUBLICATION_SCHEMA:
            raise ValueError("unsupported checkpoint publication")
        publication_hash = _require_sha256(
            publication.pop("content_sha256", None), label="publication content"
        )
        if not hmac.compare_digest(publication_hash, _payload_sha256(publication)):
            raise ValueError("checkpoint publication hash mismatch")
        identity_value = publication.get("identity")
        if not isinstance(identity_value, dict):
            raise ValueError("checkpoint publication identity is missing")
        identity = CheckpointIdentity.from_dict(identity_value)
        if (
            int(pointer.get("generation", -1)) != identity.generation
            or pointer.get("checkpoint_sha256") != identity.checkpoint_sha256
        ):
            raise ValueError("active pointer and checkpoint publication disagree")
        self.verify(identity)
        return identity


@dataclass(frozen=True, slots=True)
class HumanPlayLeaseRecord:
    session_id: str
    process_id: int
    nonce: str
    state: str
    started_unix_seconds: float
    heartbeat_unix_seconds: float
    expires_unix_seconds: float
    checkpoint_generation: int
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if _SESSION_ID_RE.fullmatch(self.session_id) is None:
            raise ValueError("invalid human-play lease session id")
        if self.process_id < 1:
            raise ValueError("human-play lease process id must be positive")
        _require_sha256(self.nonce, label="human-play lease nonce")
        if self.state not in {"active", "inactive"}:
            raise ValueError("unsupported human-play lease state")
        for label, value in (
            ("start", self.started_unix_seconds),
            ("heartbeat", self.heartbeat_unix_seconds),
            ("expiry", self.expires_unix_seconds),
        ):
            _require_finite_number(value, label=f"human-play lease {label}")
        if self.heartbeat_unix_seconds < self.started_unix_seconds:
            raise ValueError("human-play lease heartbeat precedes its start")
        if self.state == "active" and self.expires_unix_seconds <= self.heartbeat_unix_seconds:
            raise ValueError("active human-play lease must expire after its heartbeat")
        if self.checkpoint_generation < 0:
            raise ValueError("human-play lease checkpoint generation must be non-negative")
        _require_sha256(self.checkpoint_sha256, label="human-play lease checkpoint")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HumanPlayLeaseRecord:
        return cls(
            session_id=str(value["session_id"]),
            process_id=int(value["process_id"]),
            nonce=str(value["nonce"]),
            state=str(value["state"]),
            started_unix_seconds=float(value["started_unix_seconds"]),
            heartbeat_unix_seconds=float(value["heartbeat_unix_seconds"]),
            expires_unix_seconds=float(value["expires_unix_seconds"]),
            checkpoint_generation=int(value["checkpoint_generation"]),
            checkpoint_sha256=str(value["checkpoint_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class TrainingStepLeaseRecord:
    lease_id: str
    process_id: int
    nonce: str
    started_unix_seconds: float
    heartbeat_unix_seconds: float
    expires_unix_seconds: float
    generation: int | None

    def __post_init__(self) -> None:
        if self.lease_id != f"{self.process_id}-{self.nonce[:16]}":
            raise ValueError("training-step lease id does not match its owner")
        if self.process_id < 1:
            raise ValueError("training-step lease process id must be positive")
        _require_sha256(self.nonce, label="training-step lease nonce")
        for label, value in (
            ("start", self.started_unix_seconds),
            ("heartbeat", self.heartbeat_unix_seconds),
            ("expiry", self.expires_unix_seconds),
        ):
            _require_finite_number(value, label=f"training-step lease {label}")
        if self.heartbeat_unix_seconds < self.started_unix_seconds:
            raise ValueError("training-step heartbeat precedes its start")
        if self.expires_unix_seconds <= self.heartbeat_unix_seconds:
            raise ValueError("training-step lease must expire after its heartbeat")
        if self.generation is not None and self.generation < 0:
            raise ValueError("training-step generation must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrainingStepLeaseRecord:
        generation = value.get("generation")
        return cls(
            lease_id=str(value["lease_id"]),
            process_id=int(value["process_id"]),
            nonce=str(value["nonce"]),
            started_unix_seconds=float(value["started_unix_seconds"]),
            heartbeat_unix_seconds=float(value["heartbeat_unix_seconds"]),
            expires_unix_seconds=float(value["expires_unix_seconds"]),
            generation=None if generation is None else int(generation),
        )


def _read_training_step_lease(path: Path) -> TrainingStepLeaseRecord:
    payload = _read_json_file(path, label="training-step lease")
    if payload.get("schema") != TRAINING_STEP_LEASE_SCHEMA:
        raise ValueError("unsupported training-step lease")
    expected = _require_sha256(
        payload.pop("content_sha256", None), label="training-step lease content"
    )
    if not hmac.compare_digest(expected, _payload_sha256(payload)):
        raise ValueError("training-step lease hash mismatch")
    lease = payload.get("lease")
    if not isinstance(lease, dict):
        raise ValueError("training-step lease record is missing")
    record = TrainingStepLeaseRecord.from_dict(lease)
    if path.name != f"{record.lease_id}.json":
        raise ValueError("training-step lease filename and id disagree")
    return record


def _write_training_step_lease(path: Path, record: TrainingStepLeaseRecord) -> None:
    payload: dict[str, Any] = {
        "schema": TRAINING_STEP_LEASE_SCHEMA,
        "lease": record.to_dict(),
    }
    payload["content_sha256"] = _payload_sha256(payload)
    _atomic_replace_json(path, payload)


def _training_step_records(root: Path) -> tuple[TrainingStepLeaseRecord, ...]:
    records: list[TrainingStepLeaseRecord] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_symlink():
            raise ValueError(f"refusing symlinked training-step lease: {path}")
        if not path.is_file() or path.suffix != ".json":
            raise ValueError(f"unexpected training-step lease artifact: {path}")
        records.append(_read_training_step_lease(path))
    return tuple(records)


@dataclass(frozen=True, slots=True)
class HumanPlayLeaseSnapshot:
    observed_unix_seconds: float
    active_session_ids: tuple[str, ...]
    active_count: int
    expired_count: int
    inactive_count: int

    def __post_init__(self) -> None:
        _require_finite_number(self.observed_unix_seconds, label="lease snapshot time")
        if self.active_count != len(self.active_session_ids):
            raise ValueError("lease snapshot active count disagrees with session ids")
        if min(self.active_count, self.expired_count, self.inactive_count) < 0:
            raise ValueError("lease snapshot counts must be non-negative")

    @property
    def training_must_yield(self) -> bool:
        return self.active_count > 0


class HumanPlayLeaseStore:
    """Multi-process lease set used by the trainer's resource scheduler.

    Every concurrent ShogiHome USI process owns a different file.  A reader must
    treat any integrity exception as ``training_must_yield=True``.  Correctly
    formed active records stop counting after their short expiry so a crashed
    USI process cannot pause training forever.
    """

    def __init__(self, state_root: Path) -> None:
        configured_root = state_root.expanduser().absolute()
        if configured_root.is_symlink():
            raise ValueError("state_root must not be a symlink")
        self.state_root = configured_root.resolve()
        self.root = self.state_root / "human-play-leases"
        self.training_root = self.state_root / "training-step-leases"
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.training_root.mkdir(parents=True, exist_ok=True)
        if (
            self.state_root.is_symlink()
            or self.root.is_symlink()
            or self.training_root.is_symlink()
        ):
            raise ValueError("compute interlock roots must not be symlinks")
        self.lock_path = self.state_root / ".compute-interlock.lock"
        # A human requester holds this gate while waiting for an in-flight
        # optimizer step to finish.  New training steps must pass the same gate,
        # so a tight training loop cannot release and immediately reacquire the
        # main lock forever while starving ShogiHome.
        self.human_priority_lock_path = self.state_root / ".human-priority.lock"

    def _path(self, session_id: str) -> Path:
        if _SESSION_ID_RE.fullmatch(session_id) is None:
            raise ValueError("invalid human-play lease session id")
        path = self.root / f"{session_id}.json"
        _reject_symlink_components(self.root, path)
        return path

    def _read(self, path: Path) -> HumanPlayLeaseRecord:
        payload = _read_json_file(path, label="human-play lease")
        if payload.get("schema") != HUMAN_PLAY_LEASE_SCHEMA:
            raise ValueError("unsupported human-play lease")
        expected = _require_sha256(
            payload.pop("content_sha256", None), label="human-play lease content"
        )
        if not hmac.compare_digest(expected, _payload_sha256(payload)):
            raise ValueError("human-play lease hash mismatch")
        lease = payload.get("lease")
        if not isinstance(lease, dict):
            raise ValueError("human-play lease record is missing")
        record = HumanPlayLeaseRecord.from_dict(lease)
        if path.name != f"{record.session_id}.json":
            raise ValueError("human-play lease filename and session id disagree")
        return record

    def _write(self, path: Path, record: HumanPlayLeaseRecord) -> None:
        payload: dict[str, Any] = {
            "schema": HUMAN_PLAY_LEASE_SCHEMA,
            "lease": record.to_dict(),
        }
        payload["content_sha256"] = _payload_sha256(payload)
        _atomic_replace_json(path, payload)

    def _records_locked(self) -> tuple[HumanPlayLeaseRecord, ...]:
        records: list[HumanPlayLeaseRecord] = []
        for path in sorted(self.root.iterdir(), key=lambda item: item.name):
            if path.is_symlink():
                raise ValueError(f"refusing symlinked human-play lease: {path}")
            if not path.is_file() or path.suffix != ".json":
                raise ValueError(f"unexpected human-play lease artifact: {path}")
            records.append(self._read(path))
        return tuple(records)

    def acquire(
        self,
        session_id: str,
        checkpoint: CheckpointIdentity,
        *,
        ttl_seconds: float = 15.0,
        heartbeat_interval_seconds: float = 3.0,
        wait_timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 0.05,
    ) -> HumanPlayLeaseHandle:
        if not 3 <= ttl_seconds <= 300 or not 0.1 <= heartbeat_interval_seconds < ttl_seconds:
            raise ValueError("human-play lease TTL/heartbeat interval is invalid")
        if wait_timeout_seconds < 0 or not 0.01 <= poll_interval_seconds <= 1:
            raise ValueError("human-play lease wait configuration is invalid")
        path = self._path(session_id)
        nonce = secrets.token_hex(32)
        record: HumanPlayLeaseRecord | None = None
        deadline = time.monotonic() + wait_timeout_seconds
        with _exclusive_file_lock(self.human_priority_lock_path):
            while True:
                with _exclusive_file_lock(self.lock_path):
                    self._records_locked()
                    training_records = _training_step_records(self.training_root)
                    active_training = tuple(
                        item
                        for item in training_records
                        if item.expires_unix_seconds > time.time()
                    )
                    if not active_training:
                        now = time.time()
                        if path.exists():
                            existing = self._read(path)
                            if (
                                existing.state == "active"
                                and existing.expires_unix_seconds > now
                            ):
                                raise FileExistsError(
                                    "active human-play lease session id already exists"
                                )
                        record = HumanPlayLeaseRecord(
                            session_id=session_id,
                            process_id=os.getpid(),
                            nonce=nonce,
                            state="active",
                            started_unix_seconds=now,
                            heartbeat_unix_seconds=now,
                            expires_unix_seconds=now + ttl_seconds,
                            checkpoint_generation=checkpoint.generation,
                            checkpoint_sha256=checkpoint.checkpoint_sha256,
                        )
                        self._write(path, record)
                        break
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for active training steps to finish")
                time.sleep(poll_interval_seconds)
        if record is None:
            raise AssertionError("human-play lease acquisition produced no record")
        return HumanPlayLeaseHandle(
            self,
            record,
            ttl_seconds=ttl_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )

    def heartbeat(
        self,
        session_id: str,
        nonce: str,
        *,
        ttl_seconds: float,
    ) -> HumanPlayLeaseRecord:
        path = self._path(session_id)
        with _exclusive_file_lock(self.lock_path):
            training_records = _training_step_records(self.training_root)
            current = self._read(path)
            if not hmac.compare_digest(current.nonce, nonce):
                raise PermissionError("human-play lease nonce mismatch")
            if current.process_id != os.getpid() or current.state != "active":
                raise ValueError("human-play lease is not owned and active")
            now = time.time()
            if current.expires_unix_seconds <= now:
                raise TimeoutError("expired human-play lease cannot be revived")
            if any(record.expires_unix_seconds > now for record in training_records):
                raise RuntimeError("training step became active before human heartbeat")
            updated = replace(
                current,
                heartbeat_unix_seconds=now,
                expires_unix_seconds=now + ttl_seconds,
            )
            self._write(path, updated)
            return updated

    def release(self, session_id: str, nonce: str) -> HumanPlayLeaseRecord:
        path = self._path(session_id)
        with _exclusive_file_lock(self.lock_path):
            _training_step_records(self.training_root)
            current = self._read(path)
            if not hmac.compare_digest(current.nonce, nonce):
                raise PermissionError("human-play lease nonce mismatch")
            if current.process_id != os.getpid():
                raise PermissionError("human-play lease process owner mismatch")
            if current.state == "inactive":
                return current
            now = time.time()
            updated = replace(
                current,
                state="inactive",
                heartbeat_unix_seconds=now,
                expires_unix_seconds=now,
            )
            self._write(path, updated)
            return updated

    def snapshot(self, *, now: float | None = None) -> HumanPlayLeaseSnapshot:
        observed = time.time() if now is None else _require_finite_number(now, label="lease time")
        active: list[str] = []
        expired = 0
        inactive = 0
        with _exclusive_file_lock(self.lock_path):
            _training_step_records(self.training_root)
            for record in self._records_locked():
                if record.state == "inactive":
                    inactive += 1
                elif record.expires_unix_seconds > observed:
                    active.append(record.session_id)
                else:
                    expired += 1
        return HumanPlayLeaseSnapshot(
            observed_unix_seconds=observed,
            active_session_ids=tuple(active),
            active_count=len(active),
            expired_count=expired,
            inactive_count=inactive,
        )


class HumanPlayLeaseHandle:
    """Background heartbeat owner for one USI game lease."""

    def __init__(
        self,
        store: HumanPlayLeaseStore,
        record: HumanPlayLeaseRecord,
        *,
        ttl_seconds: float,
        heartbeat_interval_seconds: float,
    ) -> None:
        self.store = store
        self.record = record
        self.ttl_seconds = ttl_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"meteo-human-play-lease-{record.session_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval_seconds):
            try:
                self.record = self.store.heartbeat(
                    self.record.session_id,
                    self.record.nonce,
                    ttl_seconds=self.ttl_seconds,
                )
            except BaseException as error:
                self._error = error
                return

    def assert_healthy(self) -> None:
        if self._error is not None:
            raise RuntimeError("human-play lease heartbeat failed") from self._error
        if not self._thread.is_alive() and not self._stop.is_set():
            raise RuntimeError("human-play lease heartbeat stopped unexpectedly")

    def heartbeat(self) -> HumanPlayLeaseRecord:
        self.assert_healthy()
        self.record = self.store.heartbeat(
            self.record.session_id,
            self.record.nonce,
            ttl_seconds=self.ttl_seconds,
        )
        return self.record

    def release(self) -> HumanPlayLeaseRecord:
        self._stop.set()
        self._thread.join(timeout=self.heartbeat_interval_seconds + 1.0)
        if self._thread.is_alive():
            raise RuntimeError("human-play lease heartbeat thread did not stop")
        heartbeat_error = self._error
        self.record = self.store.release(self.record.session_id, self.record.nonce)
        if heartbeat_error is not None:
            raise RuntimeError("human-play lease heartbeat failed before release") from heartbeat_error
        return self.record


@dataclass(frozen=True, slots=True)
class ComputeInterlockSnapshot:
    observed_unix_seconds: float
    active_human_sessions: tuple[str, ...]
    active_training_steps: tuple[str, ...]
    expired_human_leases: int
    expired_training_leases: int

    @property
    def human_active_count(self) -> int:
        return len(self.active_human_sessions)

    @property
    def training_step_active_count(self) -> int:
        return len(self.active_training_steps)


class ComputeInterlock:
    """Cross-process, TOCTOU-safe exclusion between human play and MLX steps.

    Human sessions may coexist with other human sessions.  MLX optimizer work is
    serialized to one training-step lease on this state root, allowing multiple
    experiments to interleave safely without simultaneous Metal/unified-memory
    spikes.  A human lease and a training-step lease can never be created
    concurrently because both decisions and writes occur under the same file
    lock.  A separate human-priority gate prevents a tight training loop from
    starving a waiting ShogiHome game between optimizer steps.
    """

    def __init__(self, state_root: Path) -> None:
        self.human = HumanPlayLeaseStore(state_root)
        self.state_root = self.human.state_root
        self.human_root = self.human.root
        self.training_root = self.human.training_root
        self.lock_path = self.human.lock_path
        self.human_priority_lock_path = self.human.human_priority_lock_path

    def snapshot(self, *, now: float | None = None) -> ComputeInterlockSnapshot:
        observed = time.time() if now is None else _require_finite_number(now, label="lease time")
        with _exclusive_file_lock(self.lock_path):
            human_records = self.human._records_locked()
            training_records = _training_step_records(self.training_root)
        human_active = tuple(
            record.session_id
            for record in human_records
            if record.state == "active" and record.expires_unix_seconds > observed
        )
        training_active = tuple(
            record.lease_id
            for record in training_records
            if record.expires_unix_seconds > observed
        )
        return ComputeInterlockSnapshot(
            observed_unix_seconds=observed,
            active_human_sessions=human_active,
            active_training_steps=training_active,
            expired_human_leases=sum(
                record.state == "active" and record.expires_unix_seconds <= observed
                for record in human_records
            ),
            expired_training_leases=sum(
                record.expires_unix_seconds <= observed for record in training_records
            ),
        )

    def acquire_human(
        self,
        session_id: str,
        checkpoint: CheckpointIdentity,
        *,
        ttl_seconds: float = 15.0,
        heartbeat_interval_seconds: float = 3.0,
        wait_timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 0.05,
    ) -> HumanPlayLeaseHandle:
        return self.human.acquire(
            session_id,
            checkpoint,
            ttl_seconds=ttl_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            wait_timeout_seconds=wait_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    def _training_path(self, lease_id: str) -> Path:
        if not re.fullmatch(r"[1-9][0-9]*-[0-9a-f]{16}", lease_id):
            raise ValueError("invalid training-step lease id")
        path = self.training_root / f"{lease_id}.json"
        _reject_symlink_components(self.training_root, path)
        return path

    def acquire_training_step(
        self,
        *,
        generation: int | None = None,
        ttl_seconds: float = 60.0,
        heartbeat_interval_seconds: float = 5.0,
        wait_timeout_seconds: float = 0.0,
        poll_interval_seconds: float = 0.05,
    ) -> TrainingStepLeaseHandle:
        if generation is not None and generation < 0:
            raise ValueError("training-step generation must be non-negative")
        if not 3 <= ttl_seconds <= 3600 or not 0.1 <= heartbeat_interval_seconds < ttl_seconds:
            raise ValueError("training-step lease TTL/heartbeat interval is invalid")
        if wait_timeout_seconds < 0 or not 0.01 <= poll_interval_seconds <= 1:
            raise ValueError("training-step wait configuration is invalid")
        deadline = time.monotonic() + wait_timeout_seconds
        while True:
            with (
                _exclusive_file_lock(self.human_priority_lock_path),
                _exclusive_file_lock(self.lock_path),
            ):
                now = time.time()
                human_records = self.human._records_locked()
                training_records = _training_step_records(self.training_root)
                active_humans = tuple(
                    item
                    for item in human_records
                    if item.state == "active" and item.expires_unix_seconds > now
                )
                active_training = tuple(
                    item for item in training_records if item.expires_unix_seconds > now
                )
                if not active_humans and not active_training:
                    nonce = secrets.token_hex(32)
                    lease_id = f"{os.getpid()}-{nonce[:16]}"
                    path = self._training_path(lease_id)
                    if path.exists():
                        raise FileExistsError("training-step lease id collision")
                    record = TrainingStepLeaseRecord(
                        lease_id=lease_id,
                        process_id=os.getpid(),
                        nonce=nonce,
                        started_unix_seconds=now,
                        heartbeat_unix_seconds=now,
                        expires_unix_seconds=now + ttl_seconds,
                        generation=generation,
                    )
                    _write_training_step_lease(path, record)
                    return TrainingStepLeaseHandle(
                        self,
                        record,
                        ttl_seconds=ttl_seconds,
                        heartbeat_interval_seconds=heartbeat_interval_seconds,
                    )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "human play is active or another training step is active; "
                    "training step lease not acquired"
                )
            time.sleep(poll_interval_seconds)

    def training_step(
        self,
        *,
        generation: int | None = None,
        ttl_seconds: float = 60.0,
        heartbeat_interval_seconds: float = 5.0,
        wait_timeout_seconds: float = 0.0,
        poll_interval_seconds: float = 0.05,
    ) -> TrainingStepLeaseHandle:
        """Acquire the context manager that must enclose one optimizer step."""

        return self.acquire_training_step(
            generation=generation,
            ttl_seconds=ttl_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            wait_timeout_seconds=wait_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    def heartbeat_training_step(
        self,
        lease_id: str,
        nonce: str,
        *,
        ttl_seconds: float,
    ) -> TrainingStepLeaseRecord:
        path = self._training_path(lease_id)
        with _exclusive_file_lock(self.lock_path):
            human_records = self.human._records_locked()
            current = _read_training_step_lease(path)
            if not hmac.compare_digest(current.nonce, nonce):
                raise PermissionError("training-step lease nonce mismatch")
            if current.process_id != os.getpid():
                raise PermissionError("training-step lease process owner mismatch")
            now = time.time()
            if current.expires_unix_seconds <= now:
                raise TimeoutError("expired training-step lease cannot be revived")
            if any(
                record.state == "active" and record.expires_unix_seconds > now
                for record in human_records
            ):
                raise RuntimeError("human play became active before training heartbeat")
            updated = replace(
                current,
                heartbeat_unix_seconds=now,
                expires_unix_seconds=now + ttl_seconds,
            )
            _write_training_step_lease(path, updated)
            return updated

    def release_training_step(self, lease_id: str, nonce: str) -> None:
        path = self._training_path(lease_id)
        with _exclusive_file_lock(self.lock_path):
            self.human._records_locked()
            current = _read_training_step_lease(path)
            if not hmac.compare_digest(current.nonce, nonce):
                raise PermissionError("training-step lease nonce mismatch")
            if current.process_id != os.getpid():
                raise PermissionError("training-step lease process owner mismatch")
            path.unlink()
            _fsync_directory(self.training_root)


class TrainingStepLeaseHandle:
    """Context-managed, heartbeating lease around exactly one optimizer step."""

    def __init__(
        self,
        interlock: ComputeInterlock,
        record: TrainingStepLeaseRecord,
        *,
        ttl_seconds: float,
        heartbeat_interval_seconds: float,
    ) -> None:
        self.interlock = interlock
        self.record = record
        self.ttl_seconds = ttl_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._released = False
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"meteo-training-step-lease-{record.lease_id}",
            daemon=True,
        )
        self._thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval_seconds):
            try:
                self.record = self.interlock.heartbeat_training_step(
                    self.record.lease_id,
                    self.record.nonce,
                    ttl_seconds=self.ttl_seconds,
                )
            except BaseException as error:
                self._error = error
                return

    def assert_healthy(self) -> None:
        if self._error is not None:
            raise RuntimeError("training-step lease heartbeat failed") from self._error
        if not self._thread.is_alive() and not self._stop.is_set():
            raise RuntimeError("training-step heartbeat stopped unexpectedly")

    def release(self) -> None:
        if self._released:
            return
        self._stop.set()
        self._thread.join(timeout=self.heartbeat_interval_seconds + 1.0)
        if self._thread.is_alive():
            raise RuntimeError("training-step heartbeat thread did not stop")
        heartbeat_error = self._error
        self.interlock.release_training_step(self.record.lease_id, self.record.nonce)
        self._released = True
        if heartbeat_error is not None:
            raise RuntimeError(
                "training-step lease heartbeat failed before release"
            ) from heartbeat_error

    def __enter__(self) -> TrainingStepLeaseHandle:
        self.assert_healthy()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        self.release()


@dataclass(frozen=True, slots=True)
class TrainingStatus:
    state: str
    generation: int | None
    progress: float | None
    message: str
    updated_unix_seconds: float
    process_id: int | None = None

    def __post_init__(self) -> None:
        if self.state not in {"idle", "running", "paused", "error"}:
            raise ValueError("unsupported training state")
        if self.generation is not None and self.generation < 0:
            raise ValueError("training generation must be non-negative")
        if self.progress is not None and not 0 <= self.progress <= 1:
            raise ValueError("training progress must be in [0, 1]")
        if not self.message.strip() or len(self.message) > 500:
            raise ValueError("training status message must be non-empty and bounded")
        _require_finite_number(self.updated_unix_seconds, label="training update time")
        if self.process_id is not None and self.process_id < 1:
            raise ValueError("training process id must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrainingStatus:
        generation = value.get("generation")
        progress = value.get("progress")
        process_id = value.get("process_id")
        return cls(
            state=str(value["state"]),
            generation=None if generation is None else int(generation),
            progress=None if progress is None else float(progress),
            message=str(value["message"]),
            updated_unix_seconds=float(value["updated_unix_seconds"]),
            process_id=None if process_id is None else int(process_id),
        )


class TrainingStatusStore:
    """Hash-checked status only; the GUI never receives trainer/model objects."""

    def __init__(self, path: Path) -> None:
        configured_path = path.expanduser().absolute()
        if configured_path.is_symlink():
            raise ValueError("training status path must not be a symlink")
        self.path = configured_path.resolve()
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")

    def publish(self, status: TrainingStatus) -> None:
        payload: dict[str, Any] = {
            "schema": TRAINING_STATUS_SCHEMA,
            "status": status.to_dict(),
        }
        payload["content_sha256"] = _payload_sha256(payload)
        with _exclusive_file_lock(self.lock_path):
            _atomic_replace_json(self.path, payload)

    def read(self) -> TrainingStatus:
        if not self.path.exists():
            return TrainingStatus("idle", None, None, "学習プロセスは未接続です", time.time())
        payload = _read_json_file(self.path, label="training status")
        if payload.get("schema") != TRAINING_STATUS_SCHEMA:
            raise ValueError("unsupported training status")
        expected = _require_sha256(
            payload.pop("content_sha256", None), label="training status content"
        )
        if not hmac.compare_digest(expected, _payload_sha256(payload)):
            raise ValueError("training status hash mismatch")
        value = payload.get("status")
        if not isinstance(value, dict):
            raise ValueError("training status record is missing")
        return TrainingStatus.from_dict(value)


@dataclass(frozen=True, slots=True)
class HumanSession:
    session_id: str
    version: int
    human_color: int
    initial_sfen: str
    current_sfen: str
    moves: tuple[str, ...]
    status: str
    winner: int | None
    termination: str | None
    pinned_checkpoint: CheckpointIdentity
    created_unix_seconds: float
    updated_unix_seconds: float
    raw_recorded: bool = False

    def __post_init__(self) -> None:
        if _SESSION_ID_RE.fullmatch(self.session_id) is None:
            raise ValueError("invalid session id")
        if self.version < 0 or self.human_color not in {0, 1}:
            raise ValueError("invalid session version or human color")
        if self.status not in {"playing", "complete"}:
            raise ValueError("invalid session status")
        if self.winner not in {None, 0, 1}:
            raise ValueError("invalid session winner")
        if self.status == "complete" and self.termination is None:
            raise ValueError("complete session requires termination")
        if self.status == "playing" and (self.winner is not None or self.termination is not None):
            raise ValueError("playing session cannot have a result")
        _require_finite_number(self.created_unix_seconds, label="session creation time")
        _require_finite_number(self.updated_unix_seconds, label="session update time")
        board = Board(self.initial_sfen)
        if not board.is_valid():
            raise ValueError("invalid initial session SFEN")
        for ply, move_usi in enumerate(self.moves):
            try:
                move = Move.from_usi(move_usi)
            except ValueError as error:
                raise ValueError(f"invalid stored move at ply {ply}") from error
            if not board.is_legal_move(move):
                raise ValueError(f"illegal stored move at ply {ply}: {move_usi}")
            board.apply_move(move)
        if board.to_sfen() != self.current_sfen:
            raise ValueError("session SFEN does not match its move history")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        return cast(dict[str, object], value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HumanSession:
        pinned = value.get("pinned_checkpoint")
        if not isinstance(pinned, dict):
            raise ValueError("session checkpoint identity is missing")
        return cls(
            session_id=str(value["session_id"]),
            version=int(value["version"]),
            human_color=int(value["human_color"]),
            initial_sfen=str(value["initial_sfen"]),
            current_sfen=str(value["current_sfen"]),
            moves=tuple(str(move) for move in value["moves"]),
            status=str(value["status"]),
            winner=None if value.get("winner") is None else int(value["winner"]),
            termination=(
                None if value.get("termination") is None else str(value["termination"])
            ),
            pinned_checkpoint=CheckpointIdentity.from_dict(pinned),
            created_unix_seconds=float(value["created_unix_seconds"]),
            updated_unix_seconds=float(value["updated_unix_seconds"]),
            raw_recorded=bool(value.get("raw_recorded", False)),
        )


class HumanGameLog:
    """Append-only, idempotent, hash-chained candidate-only human games."""

    def __init__(self, path: Path) -> None:
        configured_path = path.expanduser().absolute()
        if configured_path.is_symlink():
            raise ValueError("human-game log path must not be a symlink")
        self.path = configured_path.resolve()
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")

    def _verified_rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        if self.path.is_symlink():
            raise ValueError("refusing symlinked human-game log")
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("human-game log ends in a partial row")
        rows: list[dict[str, Any]] = []
        previous_hash: str | None = None
        seen_sessions: set[str] = set()
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line:
                raise ValueError(f"blank human-game log row at line {line_number}")
            row = _strict_json_loads(line, label=f"human-game row {line_number}")
            observed = _require_sha256(
                row.pop("record_sha256", None), label=f"human-game row {line_number}"
            )
            if not hmac.compare_digest(observed, _payload_sha256(row)):
                raise ValueError(f"human-game row hash mismatch at line {line_number}")
            if row.get("previous_record_sha256") != previous_hash:
                raise ValueError(f"human-game hash chain mismatch at line {line_number}")
            if row.get("schema") != HUMAN_GAME_SCHEMA:
                raise ValueError(f"unsupported human-game row at line {line_number}")
            session_id = row.get("session_id")
            if not isinstance(session_id, str) or session_id in seen_sessions:
                raise ValueError(f"duplicate/invalid human session at line {line_number}")
            seen_sessions.add(session_id)
            row["record_sha256"] = observed
            rows.append(row)
            previous_hash = observed
        return rows

    def _append_candidate(self, candidate: Mapping[str, Any]) -> str:
        session_id = candidate.get("session_id")
        if not isinstance(session_id, str) or _SESSION_ID_RE.fullmatch(session_id) is None:
            raise ValueError("human-game candidate requires a valid session id")
        with _exclusive_file_lock(self.lock_path):
            rows = self._verified_rows()
            for existing_row in rows:
                if existing_row["session_id"] == session_id:
                    return str(existing_row["record_sha256"])
            previous_hash = None if not rows else str(rows[-1]["record_sha256"])
            row: dict[str, Any] = {
                "schema": HUMAN_GAME_SCHEMA,
                "session_id": session_id,
                "candidate_only": True,
                "allowed_destination": "human_raw_candidate_pool",
                "forbidden_destinations": ["arena", "validation", "sealed_test"],
                "teacher_reanalysis_required": True,
                "deduplication_required": True,
                "local_only": True,
                **{
                    key: value
                    for key, value in candidate.items()
                    if key
                    not in {
                        "schema",
                        "candidate_only",
                        "allowed_destination",
                        "forbidden_destinations",
                        "teacher_reanalysis_required",
                        "deduplication_required",
                        "local_only",
                        "previous_record_sha256",
                        "record_sha256",
                    }
                },
                "previous_record_sha256": previous_hash,
            }
            record_sha256 = _payload_sha256(row)
            row["record_sha256"] = record_sha256
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.is_symlink():
                raise ValueError("refusing symlinked human-game log")
            flags = (
                os.O_APPEND
                | os.O_CREAT
                | os.O_WRONLY
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(self.path, flags, 0o600)
            try:
                with os.fdopen(descriptor, "ab", closefd=True) as stream:
                    stream.write(_json_bytes(row))
                    stream.flush()
                    os.fsync(stream.fileno())
                _fsync_directory(self.path.parent)
            except BaseException:
                # Never continue on this log until its complete-row invariant is verified.
                raise
            return record_sha256

    def append_session(self, session: HumanSession) -> str:
        if session.status != "complete" or session.termination is None:
            raise ValueError("only complete human games may be logged")
        return self._append_candidate(
            {
                "session_id": session.session_id,
                "source": "meteo_debug_gui",
                "initial_sfen": session.initial_sfen,
                "moves": list(session.moves),
                "human_color": session.human_color,
                "winner": session.winner,
                "termination": session.termination,
                "pinned_checkpoint": session.pinned_checkpoint.to_dict(),
                "created_unix_seconds": session.created_unix_seconds,
                "completed_unix_seconds": session.updated_unix_seconds,
                "shogihome_record_reconciliation_required": False,
            }
        )

    def append_usi_observation(
        self,
        *,
        session_id: str,
        pinned_checkpoint: CheckpointIdentity,
        initial_sfen: str,
        observed_moves: Sequence[str],
        last_position_command: str,
        last_position_sfen: str,
        pending_bestmove: str | None,
        engine_color: int | None,
        winner: int | None,
        gameover_result: str,
        created_unix_seconds: float,
    ) -> str:
        """Record the USI-visible prefix; ShogiHome's auto-save remains authoritative.

        USI does not guarantee that an engine sees a human's terminal move before
        ``gameover``.  The record therefore remains candidate-only and explicitly
        requires reconciliation with ShogiHome's auto-saved KIF/CSA/JKF file.
        """

        if engine_color not in {None, 0, 1}:
            raise ValueError("USI observation engine color is invalid")
        if winner not in {None, 0, 1}:
            raise ValueError("USI observation winner is invalid")
        if gameover_result not in {"win", "lose", "draw"}:
            raise ValueError("unsupported USI gameover result")
        return self._append_candidate(
            {
                "session_id": session_id,
                "source": "shogihome_usi_bridge",
                "initial_sfen": initial_sfen,
                "moves": list(observed_moves),
                "engine_color": engine_color,
                "human_color": None if engine_color is None else 1 - engine_color,
                "winner": winner,
                "winner_requires_shogihome_reconciliation": engine_color is None,
                "termination": "usi_gameover_unverified",
                "gameover_result_from_engine_perspective": gameover_result,
                "last_position_command": last_position_command,
                "last_position_sfen": last_position_sfen,
                "pending_bestmove": pending_bestmove,
                "usi_observation_may_omit_terminal_human_move": True,
                "shogihome_record_reconciliation_required": True,
                "recommended_shogihome_auto_save_formats": [".csa", ".jkf", ".kif"],
                "pinned_checkpoint": pinned_checkpoint.to_dict(),
                "created_unix_seconds": created_unix_seconds,
                "completed_unix_seconds": time.time(),
            }
        )


class PlayerFactory(Protocol):
    def __call__(self, identity: CheckpointIdentity) -> DirectPlayer: ...


class CheckpointPlayerFactory:
    """Loads only a pinned published checkpoint; no trainer state is shared."""

    def __init__(
        self,
        registry: CheckpointRegistry,
        search: SearchConfig,
    ) -> None:
        self.registry = registry
        self.search = search

    def __call__(self, identity: CheckpointIdentity) -> DirectPlayer:
        checkpoint = self.registry.verify(identity)
        model, step = load_checkpoint(checkpoint)
        if step != identity.step:
            raise ValueError("loaded checkpoint step differs from pinned publication")
        return MctsPlayer(MLXEvaluator(model), self.search, seed=identity.generation)


class VersionConflictError(RuntimeError):
    pass


class SessionNotFoundError(KeyError):
    pass


class SessionStore:
    def __init__(self, root: Path) -> None:
        configured_root = root.expanduser().absolute()
        if configured_root.is_symlink():
            raise ValueError("session root must not be a symlink")
        self.root = configured_root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        if _SESSION_ID_RE.fullmatch(session_id) is None:
            raise ValueError("invalid session id")
        path = self.root / f"{session_id}.json"
        _reject_symlink_components(self.root, path)
        return path

    def save(self, session: HumanSession) -> None:
        payload: dict[str, Any] = {"schema": SESSION_SCHEMA, "session": session.to_dict()}
        payload["content_sha256"] = _payload_sha256(payload)
        _atomic_replace_json(self._path(session.session_id), payload)

    def load(self, session_id: str) -> HumanSession:
        path = self._path(session_id)
        if not path.exists():
            raise SessionNotFoundError(session_id)
        payload = _read_json_file(path, label="human session")
        if payload.get("schema") != SESSION_SCHEMA:
            raise ValueError("unsupported human session")
        expected = _require_sha256(
            payload.pop("content_sha256", None), label="human session content"
        )
        if not hmac.compare_digest(expected, _payload_sha256(payload)):
            raise ValueError("human session hash mismatch")
        value = payload.get("session")
        if not isinstance(value, dict):
            raise ValueError("human session payload is missing")
        return HumanSession.from_dict(value)


def _adjudicate(board: Board) -> tuple[int | None, str] | None:
    repetition = board.repetition_state()
    if repetition != RepetitionState.NONE:
        if repetition in (RepetitionState.WIN, RepetitionState.SUPERIOR):
            return board.turn.value, "repetition"
        if repetition in (RepetitionState.LOSE, RepetitionState.INFERIOR):
            return board.turn.opponent().value, "repetition"
        return None, "repetition"
    if board.can_declare_win():
        return board.turn.value, "entering_king_declaration"
    if board.is_mated() or not board.legal_moves():
        return board.turn.opponent().value, "checkmate"
    return None


def _session_board(session: HumanSession) -> Board:
    """Rebuild the exact recorded prefix, retaining history for v2 inference."""

    board = Board(session.initial_sfen)
    for move_usi in session.moves:
        board.apply_move(Move.from_usi(move_usi))
    if board.to_sfen() != session.current_sfen:
        raise ValueError("session history no longer matches its current SFEN")
    return board


class HumanGuiService:
    """Thread-safe game service with optimistic session versions."""

    def __init__(
        self,
        registry: CheckpointRegistry,
        training_status: TrainingStatusStore,
        sessions: SessionStore,
        human_games: HumanGameLog,
        player_factory: PlayerFactory,
        *,
        max_concurrent_searches: int = 1,
    ) -> None:
        if max_concurrent_searches < 1:
            raise ValueError("max_concurrent_searches must be positive")
        self.registry = registry
        self.training_status = training_status
        self.sessions = sessions
        self.human_games = human_games
        self.player_factory = player_factory
        self._search_slots = threading.BoundedSemaphore(max_concurrent_searches)
        self._lock = threading.RLock()
        self._session_locks: dict[str, threading.RLock] = {}
        self._players: dict[str, DirectPlayer] = {}

    def _session_lock(self, session_id: str) -> threading.RLock:
        with self._lock:
            return self._session_locks.setdefault(session_id, threading.RLock())

    def _player(self, session: HumanSession) -> DirectPlayer:
        with self._lock:
            player = self._players.get(session.session_id)
            if player is None:
                self.registry.verify(session.pinned_checkpoint)
                player = self.player_factory(session.pinned_checkpoint)
                self._players[session.session_id] = player
            return player

    def _advance_ai(self, session: HumanSession) -> HumanSession:
        board = _session_board(session)
        adjudication = _adjudicate(board)
        if adjudication is not None:
            winner, termination = adjudication
            return replace(
                session,
                version=session.version + 1,
                status="complete",
                winner=winner,
                termination=termination,
                updated_unix_seconds=time.time(),
            )
        if board.turn.value == session.human_color:
            return session
        player = self._player(session)
        with self._search_slots:
            decision = player.choose_move(board)
        _require_finite_number(decision.value, label="AI decision value")
        if decision.move in {"resign", "win"}:
            if decision.move == "resign":
                return replace(
                    session,
                    version=session.version + 1,
                    status="complete",
                    winner=session.human_color,
                    termination="resignation",
                    updated_unix_seconds=time.time(),
                )
            if not board.can_declare_win():
                raise ValueError("AI declared win without a legal entering-king declaration")
            return replace(
                session,
                version=session.version + 1,
                status="complete",
                winner=board.turn.value,
                termination="entering_king_declaration",
                updated_unix_seconds=time.time(),
            )
        try:
            move = Move.from_usi(decision.move)
        except ValueError as error:
            raise ValueError("AI returned malformed USI move") from error
        if not board.is_legal_move(move):
            raise ValueError("AI returned an illegal move")
        board.apply_move(move)
        result = _adjudicate(board)
        return replace(
            session,
            version=session.version + 1,
            current_sfen=board.to_sfen(),
            moves=(*session.moves, decision.move),
            status="complete" if result is not None else "playing",
            winner=None if result is None else result[0],
            termination=None if result is None else result[1],
            updated_unix_seconds=time.time(),
        )

    def _persist_and_log(self, session: HumanSession) -> HumanSession:
        self.sessions.save(session)
        if session.status == "complete" and not session.raw_recorded:
            self.human_games.append_session(session)
            session = replace(session, raw_recorded=True)
            self.sessions.save(session)
        return session

    def new_session(self, human_color: int) -> HumanSession:
        if human_color not in {0, 1}:
            raise ValueError("human_color must be 0 (black) or 1 (white)")
        identity = self.registry.read_active()
        now = time.time()
        board = Board()
        session = HumanSession(
            session_id=uuid.uuid4().hex,
            version=0,
            human_color=human_color,
            initial_sfen=board.to_sfen(),
            current_sfen=board.to_sfen(),
            moves=(),
            status="playing",
            winner=None,
            termination=None,
            pinned_checkpoint=identity,
            created_unix_seconds=now,
            updated_unix_seconds=now,
        )
        with self._session_lock(session.session_id):
            if human_color == Color.WHITE.value:
                session = self._advance_ai(session)
            return self._persist_and_log(session)

    def get_session(self, session_id: str) -> HumanSession:
        with self._session_lock(session_id):
            session = self.sessions.load(session_id)
            self.registry.verify(session.pinned_checkpoint)
            if session.status == "complete" and not session.raw_recorded:
                session = self._persist_and_log(session)
            return session

    def move(self, session_id: str, move_usi: str, *, expected_version: int) -> HumanSession:
        with self._session_lock(session_id):
            session = self.sessions.load(session_id)
            if session.version != expected_version:
                raise VersionConflictError(
                    f"session version is {session.version}, not {expected_version}"
                )
            if session.status != "playing":
                raise ValueError("game is already complete")
            board = _session_board(session)
            if board.turn.value != session.human_color:
                raise ValueError("it is not the human player's turn")
            try:
                move = Move.from_usi(move_usi)
            except ValueError as error:
                raise ValueError("malformed USI move") from error
            if not board.is_legal_move(move):
                raise ValueError("illegal move")
            board.apply_move(move)
            result = _adjudicate(board)
            session = replace(
                session,
                version=session.version + 1,
                current_sfen=board.to_sfen(),
                moves=(*session.moves, move_usi),
                status="complete" if result is not None else "playing",
                winner=None if result is None else result[0],
                termination=None if result is None else result[1],
                updated_unix_seconds=time.time(),
            )
            if session.status == "playing":
                session = self._advance_ai(session)
            return self._persist_and_log(session)

    def resign(self, session_id: str, *, expected_version: int) -> HumanSession:
        with self._session_lock(session_id):
            session = self.sessions.load(session_id)
            if session.version != expected_version:
                raise VersionConflictError(
                    f"session version is {session.version}, not {expected_version}"
                )
            if session.status != "playing":
                raise ValueError("game is already complete")
            session = replace(
                session,
                version=session.version + 1,
                status="complete",
                winner=1 - session.human_color,
                termination="resignation",
                updated_unix_seconds=time.time(),
            )
            return self._persist_and_log(session)

    def status_payload(self) -> dict[str, object]:
        active = self.registry.read_active()
        training = self.training_status.read()
        return {
            "active_checkpoint": active.to_dict(),
            "training": training.to_dict(),
            "parallelism": {
                "training_process_separate": True,
                "mutable_model_state_shared": False,
            },
        }

    @staticmethod
    def session_payload(session: HumanSession) -> dict[str, object]:
        board = _session_board(session)
        legal_moves = (
            sorted(move.to_usi() for move in board.legal_moves())
            if session.status == "playing" and board.turn.value == session.human_color
            else []
        )
        return {
            **session.to_dict(),
            "turn": board.turn.value,
            "legal_moves": legal_moves,
            "human_turn": bool(
                session.status == "playing" and board.turn.value == session.human_color
            ),
        }


def _parse_usi_position_command(line: str) -> tuple[str, tuple[str, ...], str]:
    """Return initial SFEN, verified move history, and resulting SFEN."""

    parts = line.split()
    if len(parts) < 2 or parts[0] != "position":
        raise ValueError("expected one USI position command")
    if parts[1] == "startpos":
        initial_sfen = Board().to_sfen()
        move_index = 2
    elif parts[1] == "sfen" and len(parts) >= 6:
        initial_sfen = " ".join(parts[2:6])
        move_index = 6
    else:
        raise ValueError("unsupported or malformed USI position command")
    if move_index == len(parts):
        moves: tuple[str, ...] = ()
    else:
        if parts[move_index] != "moves":
            raise ValueError("unexpected tokens after USI initial position")
        moves = tuple(parts[move_index + 1 :])
    board = Board(initial_sfen)
    if not board.is_valid():
        raise ValueError("USI position contains an invalid initial SFEN")
    for ply, move_usi in enumerate(moves):
        try:
            move = Move.from_usi(move_usi)
        except ValueError as error:
            raise ValueError(f"malformed USI history move at ply {ply}") from error
        if not board.is_legal_move(move):
            raise ValueError(f"illegal USI history move at ply {ply}: {move_usi}")
        board.apply_move(move)
    return initial_sfen, moves, board.to_sfen()


class ActiveCheckpointUsiEngine(UsiEngine):
    """ShogiHome-facing USI process with a per-game atomic checkpoint pin.

    The active pointer is read only during construction and ``usinewgame``.
    Neither ``position`` nor ``go`` can switch the evaluator, even if training
    publishes a new champion mid-game.
    """

    def __init__(
        self,
        registry: CheckpointRegistry,
        training_status: TrainingStatusStore,
        human_games: HumanGameLog,
        search: SearchConfig,
        *,
        evaluator_loader: Callable[[CheckpointIdentity], Evaluator] | None = None,
        compute_interlock: ComputeInterlock | None = None,
    ) -> None:
        self.registry = registry
        self.training_status = training_status
        self.human_games = human_games
        self.compute_interlock = compute_interlock or ComputeInterlock(
            self.registry.state_root
        )
        self._evaluator_loader = evaluator_loader or self._load_evaluator
        self.pinned_checkpoint = self.registry.read_active()
        evaluator = self._evaluator_loader(self.pinned_checkpoint)
        super().__init__(evaluator, search)
        self._game_session_id: str | None = None
        self._game_created_unix_seconds = 0.0
        self._initial_sfen = Board().to_sfen()
        self._observed_moves: tuple[str, ...] = ()
        self._last_position_command = "position startpos"
        self._last_position_sfen = Board().to_sfen()
        self._pending_bestmove: str | None = None
        self._engine_color: int | None = None
        self._gameover_logged = False
        self._human_lease: HumanPlayLeaseHandle | None = None

    def _load_evaluator(self, identity: CheckpointIdentity) -> Evaluator:
        checkpoint = self.registry.verify(identity)
        model, step = load_checkpoint(checkpoint)
        if step != identity.step:
            raise ValueError("loaded USI checkpoint step differs from its publication")
        return MLXEvaluator(model)

    def _release_human_lease(self) -> None:
        lease = self._human_lease
        if lease is None:
            return
        lease.release()
        self._human_lease = None

    def _assert_human_lease(self) -> None:
        if self._human_lease is None:
            raise RuntimeError("human-play compute lease is not active")
        self._human_lease.assert_healthy()

    def _acquire_search_lease(self) -> None:
        if self._game_session_id is None:
            raise RuntimeError("USI game session is not initialized")
        self._release_human_lease()
        self._human_lease = self.compute_interlock.acquire_human(
            self._game_session_id,
            self.pinned_checkpoint,
        )

    def _begin_game(self) -> None:
        self._release_human_lease()
        session_id = uuid.uuid4().hex
        self._game_session_id = session_id
        self._game_created_unix_seconds = time.time()
        self._initial_sfen = Board().to_sfen()
        self._observed_moves = ()
        self._last_position_command = "position startpos"
        self._last_position_sfen = Board().to_sfen()
        self._pending_bestmove = None
        self._engine_color = None
        self._gameover_logged = False

    def _pin_for_new_game(self, identity: CheckpointIdentity) -> None:
        if (
            identity.generation == self.pinned_checkpoint.generation
            and identity.checkpoint_sha256 == self.pinned_checkpoint.checkpoint_sha256
        ):
            return
        evaluator = self._evaluator_loader(identity)
        # Publish to this object only after checkpoint verification/loading succeeds.
        self.evaluator = evaluator
        self.pinned_checkpoint = identity

    def _status_info(self) -> str:
        try:
            training = self.training_status.read()
            generation = "-" if training.generation is None else str(training.generation)
            progress = (
                "-"
                if training.progress is None
                else f"{round(training.progress * 100, 1)}%"
            )
            message = " ".join(training.message.split())[:160]
            training_text = (
                f"training={training.state} generation={generation} "
                f"progress={progress} message={message}"
            )
        except (OSError, ValueError):
            training_text = "training=unavailable status_integrity_error=true"
        pin = self.pinned_checkpoint
        return (
            "info string meteo_pin "
            f"generation={pin.generation} checkpoint={pin.checkpoint_sha256[:16]} "
            f"step={pin.step} {training_text}"
        )

    def _observe_position(self, line: str) -> None:
        initial_sfen, moves, resulting_sfen = _parse_usi_position_command(line)
        if self._game_session_id is None:
            self._begin_game()
        if self._observed_moves and initial_sfen != self._initial_sfen:
            raise ValueError("USI initial position changed within one game")
        expected_prefix = self._moves_visible_to_engine()
        if expected_prefix and (
            len(moves) < len(expected_prefix)
            or moves[: len(expected_prefix)] != expected_prefix
        ):
            raise ValueError("USI position history branched within one game")
        self._initial_sfen = initial_sfen
        self._observed_moves = moves
        self._last_position_command = line
        self._last_position_sfen = resulting_sfen
        self._pending_bestmove = None

    def _moves_visible_to_engine(self) -> tuple[str, ...]:
        pending = self._pending_bestmove
        if pending is None or pending in {"resign", "win"}:
            return self._observed_moves
        board = Board(self._last_position_sfen)
        try:
            move = Move.from_usi(pending)
        except ValueError as error:
            raise ValueError("pending USI bestmove is malformed") from error
        if not board.is_legal_move(move):
            raise ValueError("pending USI bestmove is illegal")
        return (*self._observed_moves, pending)

    def _record_gameover(self, result: str) -> None:
        if self._gameover_logged:
            return
        if self._game_session_id is None:
            self._begin_game()
        if self._game_session_id is None:
            raise AssertionError("USI game session initialization failed")
        winner = (
            self._engine_color
            if result == "win"
            else (
                1 - self._engine_color
                if result == "lose" and self._engine_color is not None
                else None
            )
        )
        self.human_games.append_usi_observation(
            session_id=self._game_session_id,
            pinned_checkpoint=self.pinned_checkpoint,
            initial_sfen=self._initial_sfen,
            observed_moves=self._moves_visible_to_engine(),
            last_position_command=self._last_position_command,
            last_position_sfen=self._last_position_sfen,
            pending_bestmove=self._pending_bestmove,
            engine_color=self._engine_color,
            winner=winner,
            gameover_result=result,
            created_unix_seconds=self._game_created_unix_seconds,
        )
        self._gameover_logged = True

    def handle(self, line: str) -> list[str]:
        parts = line.split()
        command = parts[0] if parts else ""
        if command == "usinewgame":
            identity = self.registry.read_active()
            self._begin_game()
            try:
                self._pin_for_new_game(identity)
            except BaseException:
                self._release_human_lease()
                self._game_session_id = None
                raise
            return super().handle(line)
        if command == "position":
            self._observe_position(line)
            responses = super().handle(line)
            if self.board.to_sfen() != self._last_position_sfen:
                raise ValueError("USI engine board disagrees with verified position history")
            return responses
        if command == "go":
            if self._game_session_id is None:
                self._begin_game()
            if self._engine_color is None:
                self._engine_color = self.board.turn.value
            # Evolution remains the default workload.  The game keeps its
            # checkpoint pin for its whole story, but compute is reserved only
            # for this one USI search.  Training/self-play continue while the
            # human is thinking between ``go`` commands.
            self._acquire_search_lease()
            try:
                self._assert_human_lease()
                responses = super().handle(line)
                self._assert_human_lease()
            finally:
                self._release_human_lease()
            bestmoves = [response.split(maxsplit=1)[1] for response in responses if response.startswith("bestmove ")]
            if len(bestmoves) != 1:
                raise ValueError("Meteo USI search must return exactly one bestmove")
            self._pending_bestmove = bestmoves[0].split()[0]
            return [self._status_info(), *responses]
        if command == "gameover":
            if len(parts) != 2 or parts[1] not in {"win", "lose", "draw"}:
                raise ValueError("USI gameover must be win, lose, or draw")
            self._record_gameover(parts[1])
            return super().handle(line)
        if command == "quit":
            try:
                return super().handle(line)
            finally:
                self._release_human_lease()
        responses = super().handle(line)
        if command == "usi":
            name = (
                f"id name Meteo g{self.pinned_checkpoint.generation} "
                f"{self.pinned_checkpoint.checkpoint_sha256[:12]}"
            )
            return [name if response.startswith("id name ") else response for response in responses]
        return responses


_HTML = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Meteo Debug 対局室</title>
<style>
:root{--ink:#172033;--muted:#667085;--line:#d6dae2;--accent:#2457c5;--ok:#247447;--paper:#f8f8f6;--wood:#e9c277}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Yu Gothic UI",sans-serif}
header{height:68px;padding:0 24px;display:flex;align-items:center;border-bottom:1px solid var(--line);background:#fff}h1{font-size:28px;margin:0;letter-spacing:.02em}
.shell{max-width:1380px;margin:auto;padding:16px 20px 20px}.status{display:grid;grid-template-columns:1.4fr 1fr;gap:12px;margin-bottom:12px}.status>div{border:1px solid var(--line);padding:12px 16px;background:#fff;font-size:14px}.pinned{border-color:#9bb7f3!important;color:#123d94}.training{border-color:#a8cfb5!important;color:#185d36}
.layout{display:grid;grid-template-columns:minmax(560px,850px) minmax(300px,1fr);gap:16px}.board-wrap{min-width:0}.board{aspect-ratio:1;display:grid;grid-template-columns:repeat(9,1fr);border:2px solid #3d3325;background:var(--wood);box-shadow:0 8px 25px #0001}.sq{border-right:1px solid #584a33;border-bottom:1px solid #584a33;display:flex;align-items:center;justify-content:center;font-family:"Yu Mincho","Hiragino Mincho ProN",serif;font-size:clamp(20px,3vw,40px);position:relative}.sq:nth-child(9n){border-right:0}.piece.white{transform:rotate(180deg)}
.hand{min-height:48px;display:flex;align-items:center;gap:8px;padding:7px 10px;background:#fff;border:1px solid var(--line);font-size:14px}.hand:first-child{margin-bottom:8px}.hand:last-child{margin-top:8px}.hand strong{min-width:70px}.hand-piece{font-family:"Yu Mincho",serif;font-size:22px}
.side{background:#fff;border:1px solid var(--line);padding:18px;display:flex;flex-direction:column;gap:18px}.side h2{font-size:16px;margin:0 0 8px}.side label{font-size:14px}.choices{display:flex;gap:22px}.actions{display:grid;grid-template-columns:1fr 1fr;gap:10px}button,select{font:600 15px/1.2 inherit;min-height:44px;border-radius:4px;border:1px solid var(--line);background:#fff;padding:8px 12px}button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}button.danger{color:#b42318;border-color:#d92d20}button:disabled,select:disabled{opacity:.45}.turn{font-size:22px;font-weight:700}.moves{height:260px;overflow:auto;border-top:1px solid var(--line);padding-top:8px;line-height:1.8;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.note{margin-top:auto;padding:12px;border:1px solid #bfd0f5;color:#234a9b;background:#f6f8ff;font-size:13px}.error{color:#b42318;min-height:20px;font-size:13px}
footer{margin-top:14px;border:1px solid var(--line);background:#fff;padding:12px 16px;font-size:13px;color:var(--muted);display:flex;justify-content:space-between;gap:20px}
@media(max-width:900px){.status,.layout{grid-template-columns:1fr}.board{min-width:0}.side{min-height:420px}footer{flex-direction:column}.shell{padding:10px}.status>div{font-size:12px}}
</style></head><body><header><h1>Meteo Debug 対局室</h1></header><main class="shell">
<section class="status"><div class="pinned" id="pinned">この対局: 未開始</div><div class="training" id="training">学習状態を確認中…</div></section>
<div class="layout"><section class="board-wrap"><div class="hand" id="top-hand"></div><div class="board" id="board" aria-label="将棋盤"></div><div class="hand" id="bottom-hand"></div></section>
<aside class="side"><div><h2>先後</h2><div class="choices"><label><input type="radio" name="side" value="0" checked> 先手</label><label><input type="radio" name="side" value="1"> 後手</label></div></div>
<div class="actions"><button class="primary" id="new">新規対局</button><button class="danger" id="resign" disabled>投了</button></div>
<div><h2>現在の手番</h2><div class="turn" id="turn">対局を開始してください</div></div>
<div><h2>合法手</h2><select id="move" disabled><option>対局を開始してください</option></select><button class="primary" id="play" disabled style="width:100%;margin-top:8px">指す</button></div>
<div><h2>指し手</h2><div class="moves" id="moves">—</div></div><div class="error" id="error"></div>
<div class="note">昇格モデルは次の対局から反映されます。局中のcheckpointは固定です。</div></aside></div>
<footer><span>localhost + secret header で保護</span><span>人間棋譜は候補専用RAWとして保存し、教師再解析前は学習・arena・validation・sealed testに入りません</span></footer>
</main><script>
const TOKEN=__TOKEN__;let session=null;const E=id=>document.getElementById(id);
const pieceName={P:'歩',L:'香',N:'桂',S:'銀',G:'金',B:'角',R:'飛',K:'玉','+P':'と','+L':'杏','+N':'圭','+S':'全','+B':'馬','+R':'龍'};
async function api(path,method='GET',body=null){const r=await fetch(path,{method,headers:{'X-Meteo-Token':TOKEN,'Content-Type':'application/json'},body:body?JSON.stringify(body):null});const v=await r.json();if(!r.ok)throw new Error(v.error||`HTTP ${r.status}`);return v}
function parseSfen(sfen){const [board,turn,hands]=sfen.split(' ');const rows=[];for(const row of board.split('/')){const out=[];for(let i=0;i<row.length;i++){let c=row[i];if(/\d/.test(c)){for(let n=0;n<Number(c);n++)out.push(null)}else{let p=c;if(c==='+')p=c+row[++i];out.push({name:pieceName[p.toUpperCase()]||p,white:p[p.length-1]===p[p.length-1].toLowerCase()})}}rows.push(out)}return{rows,turn,hands}}
function handPieces(h){if(h==='-')return[[],[]];const black=[],white=[];let n='';for(const c of h){if(/\d/.test(c)){n+=c;continue}const count=Number(n||1);n='';const p={name:pieceName[c.toUpperCase()]||c,count};(c===c.toLowerCase()?white:black).push(p)}return[black,white]}
function renderBoard(s){const parsed=parseSfen(s.current_sfen);const rotate=s.human_color===1;let cells=[];for(let r=0;r<9;r++)for(let c=0;c<9;c++){const rr=rotate?8-r:r,cc=rotate?8-c:c;cells.push(parsed.rows[rr][cc])}E('board').innerHTML=cells.map(p=>`<div class="sq">${p?`<span class="piece ${p.white?'white':''}">${p.name}</span>`:''}</div>`).join('');const hands=handPieces(parsed.hands),fmt=(items,label)=>`<strong>${label}</strong>`+items.map(p=>`<span class="hand-piece">${p.name}${p.count>1?'×'+p.count:''}</span>`).join('');const human=s.human_color,top=1-human;E('top-hand').innerHTML=fmt(hands[top],top===0?'先手 持ち駒':'後手 持ち駒');E('bottom-hand').innerHTML=fmt(hands[human],human===0?'先手 持ち駒':'後手 持ち駒')}
function render(s){session=s;renderBoard(s);const p=s.pinned_checkpoint;E('pinned').textContent=`この対局: generation ${p.generation} / checkpoint ${p.checkpoint_sha256.slice(0,12)}… (固定)`;E('turn').textContent=s.status==='complete'?`終局: ${s.termination} / ${s.winner===null?'引分':s.winner===s.human_color?'あなたの勝ち':'Meteoの勝ち'}`:s.human_turn?'あなたの手番':'Meteo 思考中';E('moves').textContent=s.moves.length?s.moves.map((m,i)=>`${i+1}. ${m}`).join('\n'):'—';const sel=E('move');sel.innerHTML=s.legal_moves.map(m=>`<option value="${m}">${m}</option>`).join('')||'<option>指せる手はありません</option>';sel.disabled=!s.human_turn;E('play').disabled=!s.human_turn;E('resign').disabled=s.status!=='playing'}
async function refreshStatus(){try{const s=await api('/api/status');const t=s.training;E('training').textContent=t.state==='running'?`学習: generation ${t.generation??'—'} / ${t.progress===null?'進行中':Math.round(t.progress*100)+'%'} — ${t.message}`:`学習: ${t.message}`}catch(e){E('training').textContent='学習状態の検証に失敗しました';E('error').textContent=e.message}}
E('new').onclick=async()=>{try{E('error').textContent='';const side=Number(document.querySelector('input[name=side]:checked').value);render(await api('/api/sessions','POST',{human_color:side}))}catch(e){E('error').textContent=e.message}};
E('play').onclick=async()=>{if(!session)return;try{E('error').textContent='Meteoが応手を探索しています…';render(await api(`/api/sessions/${session.session_id}/move`,'POST',{move:E('move').value,version:session.version}));E('error').textContent=''}catch(e){E('error').textContent=e.message}};
E('resign').onclick=async()=>{if(!session)return;try{render(await api(`/api/sessions/${session.session_id}/resign`,'POST',{version:session.version}))}catch(e){E('error').textContent=e.message}};
refreshStatus();setInterval(refreshStatus,5000);
</script></body></html>"""


def _session_id_from_path(path: str, suffix: str = "") -> str | None:
    pattern = rf"/api/sessions/([0-9a-f]{{32}}){re.escape(suffix)}\Z"
    match = re.fullmatch(pattern, path)
    return None if match is None else match.group(1)


class _GuiRequestHandler(BaseHTTPRequestHandler):
    server: MeteoGuiHttpServer

    def log_message(self, _format: str, *args: object) -> None:
        return

    def _security_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
        )

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        encoded = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(encoded)

    def _reject_nonlocal(self) -> bool:
        try:
            remote = ipaddress.ip_address(self.client_address[0])
        except ValueError:
            self._json(HTTPStatus.FORBIDDEN, {"error": "invalid remote address"})
            return True
        host_header = self.headers.get("Host", "")
        host = host_header.rsplit(":", 1)[0].strip("[]").casefold()
        if not remote.is_loopback or host not in _ALLOWED_HOSTS:
            self._json(HTTPStatus.FORBIDDEN, {"error": "localhost access only"})
            return True
        return False

    def _authorized(self) -> bool:
        provided = self.headers.get("X-Meteo-Token", "")
        if not hmac.compare_digest(provided, self.server.access_token):
            self._json(HTTPStatus.FORBIDDEN, {"error": "invalid access token"})
            return False
        return True

    def _body(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if content_type != "application/json":
            raise ValueError("Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if not 0 < length <= _MAX_JSON_BODY_BYTES:
            raise ValueError("JSON request body size is invalid")
        return _strict_json_loads(self.rfile.read(length), label="HTTP request")

    def do_GET(self) -> None:
        if self._reject_nonlocal():
            return
        path = urlsplit(self.path).path
        try:
            if path == "/":
                html = _HTML.replace("__TOKEN__", json.dumps(self.server.access_token)).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self._security_headers()
                self.end_headers()
                self.wfile.write(html)
                return
            if not self._authorized():
                return
            if path == "/api/status":
                self._json(HTTPStatus.OK, self.server.service.status_payload())
                return
            session_id = _session_id_from_path(path)
            if session_id is not None:
                session = self.server.service.get_session(session_id)
                self._json(
                    HTTPStatus.OK, self.server.service.session_payload(session)
                )
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except SessionNotFoundError:
            self._json(HTTPStatus.NOT_FOUND, {"error": "session not found"})
        except (OSError, ValueError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def do_POST(self) -> None:
        if self._reject_nonlocal() or not self._authorized():
            return
        path = urlsplit(self.path).path
        try:
            body = self._body()
            if path == "/api/sessions":
                session = self.server.service.new_session(int(body["human_color"]))
                self._json(
                    HTTPStatus.CREATED, self.server.service.session_payload(session)
                )
                return
            move_session = _session_id_from_path(path, "/move")
            if move_session is not None:
                session = self.server.service.move(
                    move_session,
                    str(body["move"]),
                    expected_version=int(body["version"]),
                )
                self._json(HTTPStatus.OK, self.server.service.session_payload(session))
                return
            resign_session = _session_id_from_path(path, "/resign")
            if resign_session is not None:
                session = self.server.service.resign(
                    resign_session, expected_version=int(body["version"])
                )
                self._json(HTTPStatus.OK, self.server.service.session_payload(session))
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except SessionNotFoundError:
            self._json(HTTPStatus.NOT_FOUND, {"error": "session not found"})
        except VersionConflictError as error:
            self._json(HTTPStatus.CONFLICT, {"error": str(error)})
        except (KeyError, OSError, TypeError, ValueError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})


class MeteoGuiHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        service: HumanGuiService,
        *,
        access_token: str,
    ) -> None:
        if not access_token or len(access_token) < 32:
            raise ValueError("access token must contain at least 32 characters")
        try:
            address_ip = ipaddress.ip_address(address[0])
        except ValueError as error:
            raise ValueError("GUI server bind address must be a loopback IP") from error
        if not address_ip.is_loopback:
            raise ValueError("GUI server may only bind to loopback")
        self.service = service
        self.access_token = access_token
        super().__init__(address, _GuiRequestHandler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Meteo local human-play GUI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish", help="atomically publish a promoted checkpoint")
    publish.add_argument("checkpoint_root", type=Path)
    publish.add_argument("state_root", type=Path)
    publish.add_argument("checkpoint", type=Path)
    publish.add_argument("--generation", type=int, required=True)

    status = subparsers.add_parser("training-status", help="publish out-of-process status")
    status.add_argument("status_file", type=Path)
    status.add_argument("--state", choices=("idle", "running", "paused", "error"), required=True)
    status.add_argument("--generation", type=int)
    status.add_argument("--progress", type=float)
    status.add_argument("--message", required=True)
    status.add_argument("--process-id", type=int)

    usi = subparsers.add_parser(
        "usi", help="run the active-checkpoint USI bridge for an unmodified ShogiHome"
    )
    usi.add_argument("checkpoint_root", type=Path)
    usi.add_argument("state_root", type=Path)
    usi.add_argument("human_game_log", type=Path)
    usi.add_argument("--simulations", type=int, default=800)
    usi.add_argument("--max-tree-nodes", type=int)

    serve = subparsers.add_parser(
        "serve", help="serve the standalone localhost debug/fallback GUI"
    )
    serve.add_argument("checkpoint_root", type=Path)
    serve.add_argument("state_root", type=Path)
    serve.add_argument("human_game_log", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--simulations", type=int, default=800)
    serve.add_argument("--max-concurrent-searches", type=int, default=1)
    serve.add_argument("--max-tree-nodes", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "publish":
        registry = CheckpointRegistry(args.checkpoint_root, args.state_root)
        identity = registry.publish(args.checkpoint, generation=args.generation)
        print(json.dumps(identity.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "training-status":
        status = TrainingStatus(
            state=args.state,
            generation=args.generation,
            progress=args.progress,
            message=args.message,
            updated_unix_seconds=time.time(),
            process_id=args.process_id,
        )
        TrainingStatusStore(args.status_file).publish(status)
        print(json.dumps(status.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "usi":
        registry = CheckpointRegistry(args.checkpoint_root, args.state_root)
        status_store = TrainingStatusStore(args.state_root / "training-status.json")
        human_games = HumanGameLog(args.human_game_log)
        search = SearchConfig(
            simulations=args.simulations,
            temperature_moves=0,
            temperature=0.0,
            dirichlet_fraction=0.0,
            max_tree_nodes=args.max_tree_nodes,
        )
        ActiveCheckpointUsiEngine(
            registry,
            status_store,
            human_games,
            search,
        ).run()
        return 0
    if args.command != "serve":
        raise AssertionError("unreachable human GUI command")
    registry = CheckpointRegistry(args.checkpoint_root, args.state_root)
    registry.read_active()
    status_store = TrainingStatusStore(args.state_root / "training-status.json")
    sessions = SessionStore(args.state_root / "human-sessions")
    human_games = HumanGameLog(args.human_game_log)
    search = SearchConfig(
        simulations=args.simulations,
        temperature_moves=0,
        temperature=0.0,
        dirichlet_fraction=0.0,
        max_tree_nodes=args.max_tree_nodes,
    )
    service = HumanGuiService(
        registry,
        status_store,
        sessions,
        human_games,
        CheckpointPlayerFactory(registry, search),
        max_concurrent_searches=args.max_concurrent_searches,
    )
    token = secrets.token_urlsafe(32)
    server = MeteoGuiHttpServer((args.host, args.port), service, access_token=token)
    host, port = server.server_address[:2]
    host_text = host.decode() if isinstance(host, bytes) else host
    print(f"Meteo GUI: http://{host_text}:{port}/", flush=True)
    print(f"Access token is embedded in that localhost page: {token}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
