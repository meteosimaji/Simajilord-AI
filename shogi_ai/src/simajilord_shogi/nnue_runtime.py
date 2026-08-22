"""Create-only, rights-safe runtime bundles for Meteo NNUE exports.

The value-only learner exports files rather than the legacy Policy+Value
checkpoint format.  This module turns one such export into an immutable
YaneuraOu runtime without copying path-bearing private provenance into the
runtime contract.  A bundle is published only after two independent USI load
smokes, exact ``getoption`` verification, content hashing, and contract reload.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, cast

from rsshogi.core import Board, Move

from .artifact_provenance import canonical_json_sha256
from .external_usi import (
    ExternalTeacherPolicy,
    ExternalUsiTeacher,
    UsiOptionValueVerification,
)

NNUE_RUNTIME_CONTRACT_SCHEMA = "meteo-nnue-yaneuraou-runtime-v1"
NNUE_RUNTIME_REGISTRY_SCHEMA = "meteo-nnue-runtime-registry-v1"
SANITIZED_EXPORT_RECEIPT_SCHEMA = "meteo-nnue-sanitized-export-receipt-v1"

_EXPORT_RECEIPT_SCHEMA = "meteo-nagisa-mlx-yaneuraou-export-v1"
_CONTRACT_FILE = "runtime-contract.json"
_SANITIZED_RECEIPT_FILE = "source-export-receipt.json"
_ENGINE_RELATIVE_PATH = "engine/YaneuraOu"
_NN_RELATIVE_PATH = "eval/nn.bin"
_PROGRESS_RELATIVE_PATH = "eval/progress.bin"
_EVAL_OPTIONS_RELATIVE_PATH = "eval/eval_options.txt"
_RUNTIME_FILE_PATHS = (
    _ENGINE_RELATIVE_PATH,
    _NN_RELATIVE_PATH,
    _PROGRESS_RELATIVE_PATH,
    _EVAL_OPTIONS_RELATIVE_PATH,
    _SANITIZED_RECEIPT_FILE,
)
_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_PATH_MARKERS = ("/users/", "/private/", "file://", "\\users\\")


def _json_sha256(payload: Mapping[str, object]) -> str:
    return canonical_json_sha256(cast(dict[str, Any], dict(payload)))


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a non-empty trimmed string")
    if "\n" in value or "\r" in value or "\0" in value:
        raise ValueError(f"{label} must be a single line without NUL")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_signed_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _require_bool(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _require_object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _require_list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return cast(list[object], value)


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    parsed: dict[str, object] = {}
    canonical_names: dict[str, str] = {}
    for key, value in pairs:
        normalized = key.casefold()
        previous = canonical_names.get(normalized)
        if previous is not None:
            raise ValueError(
                f"JSON object contains a case-insensitive key collision: {previous!r} and {key!r}"
            )
        canonical_names[normalized] = key
        parsed[key] = value
    return parsed


def _reject_nonfinite_json_constant(value: str) -> object:
    raise ValueError(f"JSON document contains a non-finite value: {value}")


def _load_json_object(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_json_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    return _require_object(value, label=label)


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_symlink_components(path: Path, *, label: str) -> Path:
    """Reject both leaf symlinks and symlinked parent traversal."""

    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink: {current}")
    return absolute


def _require_directory(path: Path, *, label: str) -> Path:
    absolute = _reject_symlink_components(path, label=label)
    if not absolute.is_dir():
        raise NotADirectoryError(absolute)
    return absolute


def _require_regular_file(path: Path, *, label: str) -> Path:
    absolute = _reject_symlink_components(path, label=label)
    try:
        metadata = absolute.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise FileNotFoundError(absolute) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {absolute}")
    return absolute


def _casefold_unique_children(directory: Path, *, label: str) -> None:
    names: dict[str, str] = {}
    for child in directory.iterdir():
        if child.is_symlink():
            raise ValueError(f"{label} contains a symlink: {child.name}")
        normalized = child.name.casefold()
        previous = names.get(normalized)
        if previous is not None:
            raise ValueError(
                f"{label} contains a case-insensitive name collision: "
                f"{previous!r} and {child.name!r}"
            )
        names[normalized] = child.name


def _read_regular_bytes(path: Path, *, label: str, maximum_bytes: int) -> bytes:
    source = _require_regular_file(path, label=label)
    size = source.stat(follow_symlinks=False).st_size
    if size > maximum_bytes:
        raise ValueError(f"{label} exceeds its {maximum_bytes}-byte safety limit")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} changed away from a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(maximum_bytes + 1)
        if len(payload) > maximum_bytes:
            raise ValueError(f"{label} exceeds its {maximum_bytes}-byte safety limit")
        after = os.fstat(descriptor)
        if (metadata.st_size, metadata.st_mtime_ns, metadata.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise RuntimeError(f"{label} changed while it was being read")
        return payload
    finally:
        os.close(descriptor)


def _hash_regular_file(path: Path, *, label: str) -> tuple[str, int]:
    source = _require_regular_file(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} changed away from a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"{label} changed while it was being hashed")
        return digest.hexdigest(), after.st_size
    finally:
        os.close(descriptor)


def _copy_regular_create_only(source: Path, destination: Path, *, executable: bool) -> str:
    source = _require_regular_file(source, label="runtime source artifact")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite staged runtime file: {destination}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = os.open(source, flags)
    digest = hashlib.sha256()
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"runtime source artifact is not a regular file: {source}")
        with (
            os.fdopen(source_descriptor, "rb", closefd=False) as input_stream,
            destination.open("xb") as output_stream,
        ):
            while chunk := input_stream.read(1024 * 1024):
                digest.update(chunk)
                output_stream.write(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        after = os.fstat(source_descriptor)
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise RuntimeError(f"runtime source artifact changed while copied: {source}")
    finally:
        os.close(source_descriptor)
    destination.chmod(0o500 if executable else 0o400)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_create_only(path: Path, payload: Mapping[str, object]) -> None:
    serialized = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with path.open("xb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite JSON artifact: {path}") from error
    path.chmod(0o400)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    destination = _reject_symlink_components(path, label="runtime registry")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent, label="runtime registry parent")
    if destination.exists() and not destination.is_file():
        raise ValueError(f"runtime registry is not a regular file: {destination}")
    if destination.is_symlink():
        raise ValueError(f"runtime registry must not be a symlink: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        serialized = (
            json.dumps(
                dict(payload),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o400)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        raise


def _safe_relative_path(value: str, *, label: str) -> str:
    normalized = _require_string(value, label=label)
    path = PurePosixPath(normalized)
    if (
        path == PurePosixPath(".")
        or path.is_absolute()
        or ".." in path.parts
        or normalized.startswith(("~", "file:"))
    ):
        raise ValueError(f"{label} must be a safe runtime-relative path")
    return normalized


def _safe_option_value(value: str | int, *, label: str) -> str | int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must use an explicit USI string rather than bool")
    if isinstance(value, int):
        return value
    normalized = _require_string(value, label=label)
    lowered = normalized.casefold()
    if os.path.isabs(normalized) or any(marker in lowered for marker in _PRIVATE_PATH_MARKERS):
        raise ValueError(f"{label} must not contain an absolute or private path")
    if normalized.startswith("~") or "://" in normalized:
        raise ValueError(f"{label} must not contain a home-relative path or URI")
    if ".." in PurePosixPath(normalized).parts:
        raise ValueError(f"{label} must not traverse a parent directory")
    return normalized


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """One no-book, reproducible YaneuraOu execution profile."""

    profile_id: str
    threads: int = 1
    hash_megabytes: int = 64
    hash_option_name: str = "USI_Hash"
    multipv: int = 1
    smoke_nodes: int = 2_000
    timeout_seconds: float = 300.0
    extra_options: tuple[tuple[str, str | int], ...] = ()

    def __post_init__(self) -> None:
        if _PROFILE_ID.fullmatch(self.profile_id) is None:
            raise ValueError("runtime profile_id must match [a-z0-9][a-z0-9._-]{0,63}")
        for label, value in (
            ("threads", self.threads),
            ("hash_megabytes", self.hash_megabytes),
            ("multipv", self.multipv),
            ("smoke_nodes", self.smoke_nodes),
        ):
            _require_int(value, label=label, minimum=1)
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("runtime timeout_seconds must be finite and positive")
        _require_string(self.hash_option_name, label="hash option name")
        if any(character.isspace() for character in self.hash_option_name):
            raise ValueError("hash option name must be compatible with YaneuraOu getoption")
        self.requested_options(fv_scale=16, routing="progress8kpabs")

    def requested_options(
        self, *, fv_scale: int, routing: str
    ) -> tuple[tuple[str, str | int], ...]:
        options: tuple[tuple[str, str | int], ...] = (
            ("BookFile", "no_book"),
            ("EnteringKingRule", "CSARule27"),
            ("EvalDir", "eval"),
            ("FV_SCALE", fv_scale),
            (self.hash_option_name, self.hash_megabytes),
            ("LS_BUCKET_MODE", routing),
            ("LS_PROGRESS_COEFF", "progress.bin"),
            ("PvInterval", 0),
            ("Threads", self.threads),
            ("USI_OwnBook", "false"),
            *self.extra_options,
        )
        seen: dict[str, str] = {}
        for name, value in options:
            validated_name = _require_string(name, label="runtime option name")
            if any(character.isspace() for character in validated_name):
                raise ValueError(
                    f"runtime option name is incompatible with getoption: {validated_name!r}"
                )
            normalized = validated_name.casefold()
            previous = seen.get(normalized)
            if previous is not None:
                raise ValueError(
                    "case-insensitive runtime option collision: "
                    f"{previous!r} and {validated_name!r}"
                )
            seen[normalized] = validated_name
            _safe_option_value(value, label=f"runtime option {validated_name}")
        if "multipv" in seen:
            raise ValueError("MultiPV is reserved by the runtime smoke adapter")
        return options


@dataclass(frozen=True, slots=True)
class RuntimeFileIdentity:
    relative_path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        _safe_relative_path(self.relative_path, label="runtime artifact path")
        _require_sha256(self.sha256, label="runtime artifact SHA-256")
        _require_int(self.bytes, label="runtime artifact bytes")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_object(cls, value: object) -> RuntimeFileIdentity:
        row = _require_object(value, label="runtime file identity")
        if set(row) != {"relative_path", "sha256", "bytes"}:
            raise ValueError("runtime file identity has unexpected or missing fields")
        return cls(
            relative_path=_safe_relative_path(
                _require_string(row["relative_path"], label="runtime artifact path"),
                label="runtime artifact path",
            ),
            sha256=_require_sha256(row["sha256"], label="runtime artifact SHA-256"),
            bytes=_require_int(row["bytes"], label="runtime artifact bytes"),
        )


@dataclass(frozen=True, slots=True)
class ExportReceiptIdentity:
    schema: str
    sha256: str
    optimizer_step: int
    architecture: str
    routing: str
    yaneuraou_fv_scale: int
    local_only: bool
    nn_bin_sha256: str
    progress_bin_sha256: str

    def __post_init__(self) -> None:
        if self.schema != _EXPORT_RECEIPT_SCHEMA:
            raise ValueError("unsupported source export receipt schema")
        _require_sha256(self.sha256, label="source export receipt SHA-256")
        _require_int(self.optimizer_step, label="optimizer step")
        _require_string(self.architecture, label="NNUE architecture")
        _require_string(self.routing, label="NNUE routing")
        _require_int(self.yaneuraou_fv_scale, label="YaneuraOu FV_SCALE", minimum=1)
        _require_bool(self.local_only, label="source export local_only")
        if not self.local_only:
            raise PermissionError("source export identity must remain local-only")
        _require_sha256(self.nn_bin_sha256, label="nn.bin SHA-256")
        _require_sha256(self.progress_bin_sha256, label="progress.bin SHA-256")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_object(cls, value: object) -> ExportReceiptIdentity:
        row = _require_object(value, label="source export identity")
        expected = {
            "schema",
            "sha256",
            "optimizer_step",
            "architecture",
            "routing",
            "yaneuraou_fv_scale",
            "local_only",
            "nn_bin_sha256",
            "progress_bin_sha256",
        }
        if set(row) != expected:
            raise ValueError("source export identity has unexpected or missing fields")
        return cls(
            schema=_require_string(row["schema"], label="source export schema"),
            sha256=_require_sha256(row["sha256"], label="source export receipt SHA-256"),
            optimizer_step=_require_int(row["optimizer_step"], label="optimizer step"),
            architecture=_require_string(row["architecture"], label="NNUE architecture"),
            routing=_require_string(row["routing"], label="NNUE routing"),
            yaneuraou_fv_scale=_require_int(
                row["yaneuraou_fv_scale"], label="YaneuraOu FV_SCALE", minimum=1
            ),
            local_only=_require_bool(row["local_only"], label="source export local_only"),
            nn_bin_sha256=_require_sha256(row["nn_bin_sha256"], label="nn.bin SHA-256"),
            progress_bin_sha256=_require_sha256(
                row["progress_bin_sha256"], label="progress.bin SHA-256"
            ),
        )


@dataclass(frozen=True, slots=True)
class AppliedRuntimeOption:
    name: str
    requested_value: str
    applied_value: str
    verified: bool

    def __post_init__(self) -> None:
        _require_string(self.name, label="applied runtime option name")
        if any(character.isspace() for character in self.name):
            raise ValueError("applied runtime option name is incompatible with getoption")
        _safe_option_value(self.requested_value, label=f"requested USI option {self.name}")
        _safe_option_value(self.applied_value, label=f"applied USI option {self.name}")
        if not _require_bool(self.verified, label="runtime option verification"):
            raise ValueError("every staged runtime option must be verified")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_object(cls, value: object) -> AppliedRuntimeOption:
        row = _require_object(value, label="applied runtime option")
        if set(row) != {"name", "requested_value", "applied_value", "verified"}:
            raise ValueError("applied runtime option has unexpected or missing fields")
        return cls(
            name=_require_string(row["name"], label="applied runtime option name"),
            requested_value=_require_string(
                row["requested_value"], label="requested runtime option value"
            ),
            applied_value=_require_string(
                row["applied_value"], label="applied runtime option value"
            ),
            verified=_require_bool(row["verified"], label="runtime option verification"),
        )


@dataclass(frozen=True, slots=True)
class LoadSmokeEvidence:
    identity_sha256: str
    usi_name: str | None
    usi_author: str | None
    identity_redacted: bool
    applied_options: tuple[AppliedRuntimeOption, ...]
    requested_nodes: int
    observed_nodes: int
    depth: int | None
    bestmove: str
    score_kind: str
    score_bound: str
    score_cp: int | None
    mate_plies: int | None
    mate_unknown_sign: int | None
    startup_stdout_sha256: str
    startup_stdout_bytes: int
    startup_stderr_sha256: str
    startup_stderr_bytes: int
    warnings_sha256: str
    warning_count: int
    fatal_diagnostic_count: int

    def __post_init__(self) -> None:
        _require_sha256(self.identity_sha256, label="USI identity SHA-256")
        if self.usi_name is not None:
            _require_string(self.usi_name, label="USI name")
        elif not self.identity_redacted:
            raise ValueError("missing USI name must be explicitly redacted")
        if self.usi_author is not None:
            _require_string(self.usi_author, label="USI author")
        _require_bool(self.identity_redacted, label="identity redacted")
        _require_unique_option_names(self.applied_options)
        if not self.applied_options:
            raise ValueError("load smoke evidence must contain verified runtime options")
        _require_int(self.requested_nodes, label="requested smoke nodes", minimum=1)
        _require_int(self.observed_nodes, label="observed smoke nodes", minimum=1)
        if self.depth is not None:
            _require_int(self.depth, label="smoke depth")
        move = Move.from_usi(_require_string(self.bestmove, label="smoke bestmove"))
        if not Board().is_legal_move(move):
            raise ValueError("smoke bestmove must be legal in the start position")
        if self.score_bound not in {"exact", "lowerbound", "upperbound"}:
            raise ValueError("smoke score bound is invalid")
        if self.score_kind == "cp":
            if (
                self.score_cp is None
                or self.mate_plies is not None
                or self.mate_unknown_sign is not None
            ):
                raise ValueError("centipawn smoke score must contain only score_cp")
        elif self.score_kind == "mate":
            if self.score_cp is not None or (
                (self.mate_plies is None) == (self.mate_unknown_sign is None)
            ):
                raise ValueError(
                    "mate smoke score must contain mate plies or an unknown-distance sign"
                )
            if self.mate_unknown_sign not in {None, -1, 1}:
                raise ValueError("unknown-distance mate sign must be -1 or 1")
        else:
            raise ValueError("smoke score kind must be cp or mate")
        for label, digest in (
            ("startup stdout SHA-256", self.startup_stdout_sha256),
            ("startup stderr SHA-256", self.startup_stderr_sha256),
            ("warnings SHA-256", self.warnings_sha256),
        ):
            _require_sha256(digest, label=label)
        _require_int(self.startup_stdout_bytes, label="startup stdout bytes")
        _require_int(self.startup_stderr_bytes, label="startup stderr bytes")
        if self.warning_count != 0 or self.fatal_diagnostic_count != 0:
            raise ValueError("staged NNUE runtime load smoke must be diagnostic-free")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["applied_options"] = [item.to_dict() for item in self.applied_options]
        return payload

    def semantic_identity(self) -> dict[str, object]:
        """Stable fields that must match on an independent reload.

        Raw startup byte counts and digests remain in each evidence record, but
        are not semantic equality fields.  Real YaneuraOu builds may print a
        process-specific allocation address or timing line while still exposing
        the same identity, options, diagnostics, and deterministic fixed-node
        search result.  Requiring byte-identical informational stdout would
        reject a valid runtime without adding a model-load guarantee.
        """

        return {
            "identity_sha256": self.identity_sha256,
            "usi_name": self.usi_name,
            "usi_author": self.usi_author,
            "identity_redacted": self.identity_redacted,
            "applied_options": [item.to_dict() for item in self.applied_options],
            "requested_nodes": self.requested_nodes,
            "observed_nodes": self.observed_nodes,
            "depth": self.depth,
            "bestmove": self.bestmove,
            "score_kind": self.score_kind,
            "score_bound": self.score_bound,
            "score_cp": self.score_cp,
            "mate_plies": self.mate_plies,
            "mate_unknown_sign": self.mate_unknown_sign,
            "warnings_sha256": self.warnings_sha256,
            "warning_count": self.warning_count,
            "fatal_diagnostic_count": self.fatal_diagnostic_count,
        }

    @classmethod
    def from_object(cls, value: object) -> LoadSmokeEvidence:
        row = _require_object(value, label="load smoke evidence")
        expected = {
            "identity_sha256",
            "usi_name",
            "usi_author",
            "identity_redacted",
            "applied_options",
            "requested_nodes",
            "observed_nodes",
            "depth",
            "bestmove",
            "score_kind",
            "score_bound",
            "score_cp",
            "mate_plies",
            "mate_unknown_sign",
            "startup_stdout_sha256",
            "startup_stdout_bytes",
            "startup_stderr_sha256",
            "startup_stderr_bytes",
            "warnings_sha256",
            "warning_count",
            "fatal_diagnostic_count",
        }
        if set(row) != expected:
            raise ValueError("load smoke evidence has unexpected or missing fields")

        def optional_string(item: object, *, label: str) -> str | None:
            return None if item is None else _require_string(item, label=label)

        def optional_nonnegative_int(item: object, *, label: str) -> int | None:
            return None if item is None else _require_int(item, label=label)

        def optional_signed_int(item: object, *, label: str) -> int | None:
            return None if item is None else _require_signed_int(item, label=label)

        options = tuple(
            AppliedRuntimeOption.from_object(item)
            for item in _require_list(row["applied_options"], label="applied runtime options")
        )
        _require_unique_option_names(options)
        return cls(
            identity_sha256=_require_sha256(row["identity_sha256"], label="USI identity SHA-256"),
            usi_name=optional_string(row["usi_name"], label="USI name"),
            usi_author=optional_string(row["usi_author"], label="USI author"),
            identity_redacted=_require_bool(row["identity_redacted"], label="identity redacted"),
            applied_options=options,
            requested_nodes=_require_int(
                row["requested_nodes"], label="requested smoke nodes", minimum=1
            ),
            observed_nodes=_require_int(
                row["observed_nodes"], label="observed smoke nodes", minimum=1
            ),
            depth=optional_nonnegative_int(row["depth"], label="smoke depth"),
            bestmove=_require_string(row["bestmove"], label="smoke bestmove"),
            score_kind=_require_string(row["score_kind"], label="smoke score kind"),
            score_bound=_require_string(row["score_bound"], label="smoke score bound"),
            score_cp=optional_signed_int(row["score_cp"], label="smoke centipawn score"),
            mate_plies=optional_signed_int(row["mate_plies"], label="smoke mate plies"),
            mate_unknown_sign=optional_signed_int(
                row["mate_unknown_sign"], label="smoke unknown mate sign"
            ),
            startup_stdout_sha256=_require_sha256(
                row["startup_stdout_sha256"], label="startup stdout SHA-256"
            ),
            startup_stdout_bytes=_require_int(
                row["startup_stdout_bytes"], label="startup stdout bytes"
            ),
            startup_stderr_sha256=_require_sha256(
                row["startup_stderr_sha256"], label="startup stderr SHA-256"
            ),
            startup_stderr_bytes=_require_int(
                row["startup_stderr_bytes"], label="startup stderr bytes"
            ),
            warnings_sha256=_require_sha256(row["warnings_sha256"], label="warnings SHA-256"),
            warning_count=_require_int(row["warning_count"], label="warning count"),
            fatal_diagnostic_count=_require_int(
                row["fatal_diagnostic_count"], label="fatal diagnostic count"
            ),
        )


def _require_unique_option_names(options: Sequence[AppliedRuntimeOption]) -> None:
    seen: dict[str, str] = {}
    for option in options:
        normalized = option.name.casefold()
        previous = seen.get(normalized)
        if previous is not None:
            raise ValueError(
                f"case-insensitive applied option collision: {previous!r} and {option.name!r}"
            )
        seen[normalized] = option.name


@dataclass(frozen=True, slots=True)
class NnueRuntimeContract:
    schema: str
    profile_id: str
    scope: str
    publication_allowed: bool
    runtime_content_sha256: str
    source_export: ExportReceiptIdentity
    engine: RuntimeFileIdentity
    artifacts: tuple[RuntimeFileIdentity, ...]
    initial_load_smoke: LoadSmokeEvidence
    reload_load_smoke: LoadSmokeEvidence
    reload_consistent: bool
    contract_sha256: str

    @property
    def nn_bin_sha256(self) -> str:
        matches = [
            item.sha256 for item in self.artifacts if item.relative_path == _NN_RELATIVE_PATH
        ]
        if len(matches) != 1:
            raise ValueError("runtime contract must identify exactly one nn.bin")
        return matches[0]

    def content_payload(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "source_export": self.source_export.to_dict(),
            "engine": self.engine.to_dict(),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "applied_options": [item.to_dict() for item in self.initial_load_smoke.applied_options],
        }

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "scope": self.scope,
            "publication_allowed": self.publication_allowed,
            "runtime_content_sha256": self.runtime_content_sha256,
            "source_export": self.source_export.to_dict(),
            "engine": self.engine.to_dict(),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "initial_load_smoke": self.initial_load_smoke.to_dict(),
            "reload_load_smoke": self.reload_load_smoke.to_dict(),
            "reload_consistent": self.reload_consistent,
            "retention_contract": {
                "playable_weight_window_limit": 2,
                "roles": ["latest", "previous", "champion"],
                "champion_must_reference_playable_window": True,
                "third_independent_weight_allowed": False,
            },
            "privacy_contract": {
                "absolute_paths_recorded": False,
                "raw_source_receipt_copied": False,
                "raw_usi_transcript_recorded": False,
                "local_only": True,
            },
            "installation_contract": {
                "create_only_destination": True,
                "same_filesystem_atomic_rename": True,
                "independent_load_smoke_processes": 2,
                "contract_and_content_reloaded_before_and_after_rename": True,
            },
        }
        if include_hash:
            payload["contract_sha256"] = self.contract_sha256
        return payload


@dataclass(frozen=True, slots=True)
class RuntimeReference:
    relative_directory: str
    profile_id: str
    runtime_content_sha256: str
    contract_sha256: str
    nn_bin_sha256: str

    def __post_init__(self) -> None:
        _safe_relative_path(self.relative_directory, label="runtime relative directory")
        if _PROFILE_ID.fullmatch(self.profile_id) is None:
            raise ValueError("runtime registry profile_id is invalid")
        _require_sha256(self.runtime_content_sha256, label="runtime content SHA-256")
        _require_sha256(self.contract_sha256, label="runtime contract SHA-256")
        _require_sha256(self.nn_bin_sha256, label="nn.bin SHA-256")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_object(cls, value: object) -> RuntimeReference:
        row = _require_object(value, label="runtime registry reference")
        expected = {
            "relative_directory",
            "profile_id",
            "runtime_content_sha256",
            "contract_sha256",
            "nn_bin_sha256",
        }
        if set(row) != expected:
            raise ValueError("runtime registry reference has unexpected or missing fields")
        profile_id = _require_string(row["profile_id"], label="runtime profile_id")
        if _PROFILE_ID.fullmatch(profile_id) is None:
            raise ValueError("runtime registry profile_id is invalid")
        return cls(
            relative_directory=_safe_relative_path(
                _require_string(row["relative_directory"], label="runtime relative directory"),
                label="runtime relative directory",
            ),
            profile_id=profile_id,
            runtime_content_sha256=_require_sha256(
                row["runtime_content_sha256"], label="runtime content SHA-256"
            ),
            contract_sha256=_require_sha256(
                row["contract_sha256"], label="runtime contract SHA-256"
            ),
            nn_bin_sha256=_require_sha256(row["nn_bin_sha256"], label="nn.bin SHA-256"),
        )


@dataclass(frozen=True, slots=True)
class NnueRuntimeRegistry:
    latest: RuntimeReference
    previous: RuntimeReference | None
    champion: RuntimeReference
    registry_sha256: str

    def __post_init__(self) -> None:
        playable = (self.latest,) if self.previous is None else (self.latest, self.previous)
        if len({item.runtime_content_sha256 for item in playable}) != len(playable):
            raise ValueError("latest and previous runtimes must have distinct content identities")
        if len({item.nn_bin_sha256 for item in playable}) != len(playable):
            raise ValueError("latest and previous must identify distinct playable weights")
        if self.champion not in playable:
            raise ValueError("champion must reference latest or previous runtime exactly")
        paths = [item.relative_directory.casefold() for item in playable]
        if len(set(paths)) != len(paths):
            raise ValueError("playable runtime paths collide ignoring case")
        _require_sha256(self.registry_sha256, label="runtime registry SHA-256")

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": NNUE_RUNTIME_REGISTRY_SCHEMA,
            "latest": self.latest.to_dict(),
            "previous": None if self.previous is None else self.previous.to_dict(),
            "champion": self.champion.to_dict(),
            "playable_weight_window_limit": 2,
            "playable_weight_count": 1 if self.previous is None else 2,
            "champion_must_reference_playable_window": True,
        }
        if include_hash:
            payload["registry_sha256"] = self.registry_sha256
        return payload


@dataclass(frozen=True, slots=True)
class _ValidatedExport:
    directory: Path
    nn_bin: Path
    progress_bin: Path
    eval_options: Path
    receipt: Path
    identity: ExportReceiptIdentity
    nn_bytes: int
    progress_bytes: int
    eval_options_sha256: str
    eval_options_bytes: int


def _parse_eval_options(raw: bytes, *, fv_scale: int, routing: str) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("eval_options.txt must be UTF-8") from error
    parsed: dict[str, str] = {}
    canonical_names: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"malformed eval_options.txt line: {line!r}")
        name, value = parts
        normalized = name.casefold()
        if normalized in canonical_names:
            raise ValueError(
                "case-insensitive eval option collision: "
                f"{canonical_names[normalized]!r} and {name!r}"
            )
        canonical_names[normalized] = name
        parsed[name] = value
    expected = {
        "LS_BUCKET_MODE": routing,
        "LS_PROGRESS_COEFF": "progress.bin",
        "FV_SCALE": str(fv_scale),
    }
    if parsed != expected:
        raise ValueError(
            "eval_options.txt does not match the pinned Meteo runtime contract: "
            f"observed={parsed!r} expected={expected!r}"
        )


def _validate_export(export_directory: Path) -> _ValidatedExport:
    directory = _require_directory(export_directory, label="NNUE export directory")
    _casefold_unique_children(directory, label="NNUE export directory")
    nn_bin = _require_regular_file(directory / "nn.bin", label="exported nn.bin")
    progress_bin = _require_regular_file(directory / "progress.bin", label="exported progress.bin")
    eval_options = _require_regular_file(
        directory / "eval_options.txt", label="exported eval_options.txt"
    )
    receipt = _require_regular_file(directory / "receipt.json", label="NNUE export receipt")
    receipt_raw = _read_regular_bytes(
        receipt, label="NNUE export receipt", maximum_bytes=1024 * 1024
    )
    receipt_row = _load_json_object(receipt_raw, label="NNUE export receipt")
    if receipt_row.get("schema") != _EXPORT_RECEIPT_SCHEMA:
        raise ValueError("unsupported NNUE export receipt schema")
    local_only = _require_bool(receipt_row.get("local_only"), label="export local_only")
    if not local_only:
        raise PermissionError(
            "NNUE runtime publication is fail-closed; this adapter currently accepts only "
            "private local-only Meteo exports"
        )
    nn_row = _require_object(receipt_row.get("nn_bin"), label="receipt nn_bin")
    receipt_nn_sha256 = _require_sha256(nn_row.get("sha256"), label="receipt nn.bin SHA-256")
    receipt_nn_bytes = _require_int(nn_row.get("bytes"), label="receipt nn.bin bytes")
    actual_nn_sha256, actual_nn_bytes = _hash_regular_file(nn_bin, label="exported nn.bin")
    if (receipt_nn_sha256, receipt_nn_bytes) != (actual_nn_sha256, actual_nn_bytes):
        raise ValueError("nn.bin does not match its export receipt")
    receipt_progress_sha256 = _require_sha256(
        receipt_row.get("progress_bin_sha256"), label="receipt progress.bin SHA-256"
    )
    actual_progress_sha256, actual_progress_bytes = _hash_regular_file(
        progress_bin, label="exported progress.bin"
    )
    if receipt_progress_sha256 != actual_progress_sha256:
        raise ValueError("progress.bin does not match its export receipt")
    fv_scale = _require_int(
        receipt_row.get("yaneuraou_fv_scale"), label="receipt FV_SCALE", minimum=1
    )
    routing = _require_string(receipt_row.get("routing"), label="receipt routing")
    eval_options_raw = _read_regular_bytes(
        eval_options, label="exported eval_options.txt", maximum_bytes=4096
    )
    _parse_eval_options(eval_options_raw, fv_scale=fv_scale, routing=routing)
    identity = ExportReceiptIdentity(
        schema=_EXPORT_RECEIPT_SCHEMA,
        sha256=hashlib.sha256(receipt_raw).hexdigest(),
        optimizer_step=_require_int(
            receipt_row.get("optimizer_step"), label="receipt optimizer step"
        ),
        architecture=_require_string(receipt_row.get("architecture"), label="NNUE architecture"),
        routing=routing,
        yaneuraou_fv_scale=fv_scale,
        local_only=True,
        nn_bin_sha256=actual_nn_sha256,
        progress_bin_sha256=actual_progress_sha256,
    )
    return _ValidatedExport(
        directory=directory,
        nn_bin=nn_bin,
        progress_bin=progress_bin,
        eval_options=eval_options,
        receipt=receipt,
        identity=identity,
        nn_bytes=actual_nn_bytes,
        progress_bytes=actual_progress_bytes,
        eval_options_sha256=hashlib.sha256(eval_options_raw).hexdigest(),
        eval_options_bytes=len(eval_options_raw),
    )


def _runtime_identity(root: Path, relative_path: str) -> RuntimeFileIdentity:
    path = _require_regular_file(root / relative_path, label=f"runtime file {relative_path}")
    digest, size = _hash_regular_file(path, label=f"runtime file {relative_path}")
    return RuntimeFileIdentity(
        relative_path=relative_path,
        sha256=digest,
        bytes=size,
    )


def _sanitized_identity_value(value: str) -> tuple[str | None, bool]:
    lowered = value.casefold()
    if (
        len(value) > 256
        or any(character in value for character in ("/", "\\", "\0", "\n", "\r"))
        or any(marker in lowered for marker in _PRIVATE_PATH_MARKERS)
    ):
        return None, True
    return value, False


def _extract_usi_identity(identity_lines: Sequence[str]) -> tuple[str | None, str | None, bool]:
    names = [
        line.removeprefix("id name ") for line in identity_lines if line.startswith("id name ")
    ]
    authors = [
        line.removeprefix("id author ") for line in identity_lines if line.startswith("id author ")
    ]
    if len(names) != 1 or len(authors) > 1:
        raise ValueError("USI runtime must report exactly one name and at most one author")
    name, name_redacted = _sanitized_identity_value(names[0])
    author: str | None = None
    author_redacted = False
    if authors:
        author, author_redacted = _sanitized_identity_value(authors[0])
    return name, author, name_redacted or author_redacted


def _run_load_smoke(
    root: Path,
    *,
    options: tuple[tuple[str, str | int], ...],
    multipv: int,
    nodes: int,
    timeout_seconds: float,
) -> LoadSmokeEvidence:
    policy = ExternalTeacherPolicy(
        policy_id="meteo-nnue-runtime-smoke",
        name="Meteo NNUE runtime smoke",
        source="local content-addressed runtime",
        analysis_allowed=True,
        training_outputs_allowed=False,
        redistribution_allowed=False,
    )
    with ExternalUsiTeacher(
        [str(root / _ENGINE_RELATIVE_PATH)],
        policy,
        nodes=nodes,
        multipv=multipv,
        options=dict(options),
        timeout_seconds=timeout_seconds,
        training_use=False,
        working_directory=root,
        option_value_verification=UsiOptionValueVerification.YANEURAOU_GETOPTION,
    ) as engine:
        startup = engine.startup_provenance
        analysis = engine.analyse(Board())
    if startup.warnings or startup.fatal_diagnostics:
        raise RuntimeError(
            "NNUE runtime load smoke emitted startup diagnostics; raw text is deliberately "
            "not copied into the runtime contract"
        )
    if not startup.applied_options or not all(item.verified for item in startup.applied_options):
        raise RuntimeError("NNUE runtime load smoke did not verify every requested option")
    applied = tuple(
        AppliedRuntimeOption(
            name=item.name,
            requested_value=_require_string(
                item.requested_value, label=f"requested USI option {item.name}"
            ),
            applied_value=_require_string(
                item.applied_value, label=f"applied USI option {item.name}"
            ),
            verified=item.verified,
        )
        for item in startup.applied_options
    )
    _require_unique_option_names(applied)
    for item in applied:
        _safe_option_value(item.requested_value, label=f"requested USI option {item.name}")
        _safe_option_value(item.applied_value, label=f"applied USI option {item.name}")
    try:
        bestmove = Move.from_usi(analysis.bestmove)
    except (RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"NNUE runtime smoke did not return a legal move token: {analysis.bestmove!r}"
        ) from error
    board = Board()
    if not board.is_legal_move(bestmove):
        raise RuntimeError(
            f"NNUE runtime smoke returned an illegal start-position move: {analysis.bestmove}"
        )
    best_variations = [item for item in analysis.candidates if item.move == analysis.bestmove]
    if len(best_variations) != 1:
        raise RuntimeError(
            "NNUE runtime smoke did not return exactly one scored bestmove variation"
        )
    variation = best_variations[0]
    if analysis.nodes is None or analysis.nodes < 1:
        raise RuntimeError("NNUE runtime smoke did not report a positive node count")
    identity_raw = "\n".join(startup.identity_lines).encode("utf-8")
    usi_name, usi_author, identity_redacted = _extract_usi_identity(startup.identity_lines)
    return LoadSmokeEvidence(
        identity_sha256=hashlib.sha256(identity_raw).hexdigest(),
        usi_name=usi_name,
        usi_author=usi_author,
        identity_redacted=identity_redacted,
        applied_options=applied,
        requested_nodes=nodes,
        observed_nodes=analysis.nodes,
        depth=analysis.depth,
        bestmove=analysis.bestmove,
        score_kind=variation.score_kind.value,
        score_bound=variation.bound.value,
        score_cp=variation.score_cp,
        mate_plies=variation.mate_plies,
        mate_unknown_sign=variation.mate_unknown_sign,
        startup_stdout_sha256=startup.stdout_sha256,
        startup_stdout_bytes=startup.stdout_bytes,
        startup_stderr_sha256=startup.stderr_sha256,
        startup_stderr_bytes=startup.stderr_bytes,
        warnings_sha256=startup.warnings_sha256,
        warning_count=len(startup.warnings),
        fatal_diagnostic_count=len(startup.fatal_diagnostics),
    )


def _assert_contract_has_no_private_paths(
    value: object, *, label: str = "runtime contract"
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_contract_has_no_private_paths(key, label=label)
            _assert_contract_has_no_private_paths(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _assert_contract_has_no_private_paths(item, label=label)
    elif isinstance(value, str):
        lowered = value.casefold()
        if os.path.isabs(value) or any(marker in lowered for marker in _PRIVATE_PATH_MARKERS):
            raise ValueError(f"{label} contains an absolute or private path")


def _expected_runtime_tree(root: Path) -> None:
    expected_root = {"engine", "eval", _SANITIZED_RECEIPT_FILE, _CONTRACT_FILE}
    expected_engine = {"YaneuraOu"}
    expected_eval = {"nn.bin", "progress.bin", "eval_options.txt"}
    for directory, expected, label in (
        (root, expected_root, "runtime root"),
        (root / "engine", expected_engine, "runtime engine directory"),
        (root / "eval", expected_eval, "runtime eval directory"),
    ):
        _casefold_unique_children(directory, label=label)
        observed = {item.name for item in directory.iterdir()}
        if observed != expected:
            raise ValueError(f"{label} has unexpected or missing entries: {observed!r}")


def _contract_from_payload(payload: Mapping[str, object]) -> NnueRuntimeContract:
    expected_root = {
        "schema",
        "profile_id",
        "scope",
        "publication_allowed",
        "runtime_content_sha256",
        "source_export",
        "engine",
        "artifacts",
        "initial_load_smoke",
        "reload_load_smoke",
        "reload_consistent",
        "retention_contract",
        "privacy_contract",
        "installation_contract",
        "contract_sha256",
    }
    if set(payload) != expected_root:
        raise ValueError("runtime contract has unexpected or missing fields")
    if payload["schema"] != NNUE_RUNTIME_CONTRACT_SCHEMA:
        raise ValueError("unsupported NNUE runtime contract schema")
    profile_id = _require_string(payload["profile_id"], label="runtime profile_id")
    if _PROFILE_ID.fullmatch(profile_id) is None:
        raise ValueError("runtime profile_id is invalid")
    source_export = ExportReceiptIdentity.from_object(payload["source_export"])
    engine = RuntimeFileIdentity.from_object(payload["engine"])
    artifacts = tuple(
        RuntimeFileIdentity.from_object(item)
        for item in _require_list(payload["artifacts"], label="runtime artifacts")
    )
    paths = [engine.relative_path, *(item.relative_path for item in artifacts)]
    if len({path.casefold() for path in paths}) != len(paths):
        raise ValueError("runtime artifact paths collide ignoring case")
    if set(paths) != set(_RUNTIME_FILE_PATHS):
        raise ValueError("runtime contract does not identify the exact required file set")
    initial = LoadSmokeEvidence.from_object(payload["initial_load_smoke"])
    reload = LoadSmokeEvidence.from_object(payload["reload_load_smoke"])
    contract = NnueRuntimeContract(
        schema=NNUE_RUNTIME_CONTRACT_SCHEMA,
        profile_id=profile_id,
        scope=_require_string(payload["scope"], label="runtime scope"),
        publication_allowed=_require_bool(
            payload["publication_allowed"], label="runtime publication_allowed"
        ),
        runtime_content_sha256=_require_sha256(
            payload["runtime_content_sha256"], label="runtime content SHA-256"
        ),
        source_export=source_export,
        engine=engine,
        artifacts=artifacts,
        initial_load_smoke=initial,
        reload_load_smoke=reload,
        reload_consistent=_require_bool(
            payload["reload_consistent"], label="runtime reload consistency"
        ),
        contract_sha256=_require_sha256(
            payload["contract_sha256"], label="runtime contract SHA-256"
        ),
    )
    if contract.scope != "private_local_only" or contract.publication_allowed:
        raise PermissionError("NNUE runtime contract must remain private and local-only")
    if not contract.source_export.local_only:
        raise PermissionError("NNUE runtime source export is not marked local-only")
    if not contract.reload_consistent or (
        initial.semantic_identity() != reload.semantic_identity()
    ):
        raise ValueError("NNUE runtime independent load smokes are inconsistent")
    artifacts_by_path = {item.relative_path: item for item in contract.artifacts}
    if artifacts_by_path[_NN_RELATIVE_PATH].sha256 != source_export.nn_bin_sha256:
        raise ValueError("runtime nn.bin identity does not match its source export")
    if artifacts_by_path[_PROGRESS_RELATIVE_PATH].sha256 != source_export.progress_bin_sha256:
        raise ValueError("runtime progress.bin identity does not match its source export")
    requested_options = {
        item.name.casefold(): item.requested_value for item in initial.applied_options
    }
    required_options = {
        "bookfile": "no_book",
        "enteringkingrule": "CSARule27",
        "evaldir": "eval",
        "fv_scale": str(source_export.yaneuraou_fv_scale),
        "ls_bucket_mode": source_export.routing,
        "ls_progress_coeff": "progress.bin",
        "pvinterval": "0",
        "usi_ownbook": "false",
    }
    for name, required_value in required_options.items():
        observed_value = requested_options.get(name)
        if observed_value is None or observed_value.casefold() != required_value.casefold():
            raise ValueError(f"runtime option {name!r} does not match the pinned execution profile")
    for positive_option in ("multipv", "threads"):
        observed_value = requested_options.get(positive_option)
        try:
            numeric = int(observed_value) if observed_value is not None else 0
        except ValueError as error:
            raise ValueError(
                f"runtime option {positive_option!r} must be a positive integer"
            ) from error
        if numeric < 1:
            raise ValueError(f"runtime option {positive_option!r} must be a positive integer")
    if contract.runtime_content_sha256 != _json_sha256(contract.content_payload()):
        raise ValueError("NNUE runtime content identity does not match the contract")
    retention = _require_object(payload["retention_contract"], label="retention contract")
    if retention != {
        "playable_weight_window_limit": 2,
        "roles": ["latest", "previous", "champion"],
        "champion_must_reference_playable_window": True,
        "third_independent_weight_allowed": False,
    }:
        raise ValueError("NNUE runtime retention contract was changed")
    privacy = _require_object(payload["privacy_contract"], label="privacy contract")
    if privacy != {
        "absolute_paths_recorded": False,
        "raw_source_receipt_copied": False,
        "raw_usi_transcript_recorded": False,
        "local_only": True,
    }:
        raise ValueError("NNUE runtime privacy contract was changed")
    installation = _require_object(payload["installation_contract"], label="installation contract")
    if installation != {
        "create_only_destination": True,
        "same_filesystem_atomic_rename": True,
        "independent_load_smoke_processes": 2,
        "contract_and_content_reloaded_before_and_after_rename": True,
    }:
        raise ValueError("NNUE runtime installation contract was changed")
    without_hash = dict(payload)
    without_hash.pop("contract_sha256")
    if contract.contract_sha256 != _json_sha256(without_hash):
        raise ValueError("NNUE runtime contract SHA-256 is invalid")
    _assert_contract_has_no_private_paths(dict(payload))
    return contract


def _validate_sanitized_export_receipt(root: Path, *, contract: NnueRuntimeContract) -> None:
    receipt_path = _require_regular_file(
        root / _SANITIZED_RECEIPT_FILE, label="sanitized source export receipt"
    )
    raw = _read_regular_bytes(
        receipt_path,
        label="sanitized source export receipt",
        maximum_bytes=1024 * 1024,
    )
    payload = _load_json_object(raw, label="sanitized source export receipt")
    expected = {
        "schema",
        "source_export",
        "files",
        "local_only",
        "publication_allowed",
        "raw_source_metadata_copied",
        "receipt_sha256",
    }
    if set(payload) != expected or payload["schema"] != SANITIZED_EXPORT_RECEIPT_SCHEMA:
        raise ValueError("unsupported or malformed sanitized source export receipt")
    digest = _require_sha256(payload["receipt_sha256"], label="sanitized receipt SHA-256")
    without_hash = dict(payload)
    without_hash.pop("receipt_sha256")
    if digest != _json_sha256(without_hash):
        raise ValueError("sanitized source export receipt SHA-256 is invalid")
    if ExportReceiptIdentity.from_object(payload["source_export"]) != contract.source_export:
        raise ValueError("sanitized receipt source identity differs from runtime contract")
    if (
        payload["local_only"] is not True
        or payload["publication_allowed"] is not False
        or payload["raw_source_metadata_copied"] is not False
    ):
        raise PermissionError("sanitized source export receipt changed its privacy scope")
    receipt_files = tuple(
        RuntimeFileIdentity.from_object(item)
        for item in _require_list(payload["files"], label="sanitized receipt files")
    )
    if len({item.relative_path.casefold() for item in receipt_files}) != len(receipt_files):
        raise ValueError("sanitized receipt file paths collide ignoring case")
    expected_files = tuple(
        item for item in contract.artifacts if item.relative_path != _SANITIZED_RECEIPT_FILE
    )
    if receipt_files != expected_files:
        raise ValueError("sanitized receipt file identities differ from runtime contract")
    _assert_contract_has_no_private_paths(payload, label="sanitized source export receipt")


def load_nnue_runtime_contract(runtime_directory: Path) -> NnueRuntimeContract:
    """Reload and verify an installed or staged runtime without executing it."""

    root = _require_directory(runtime_directory, label="NNUE runtime directory")
    contract_path = _require_regular_file(root / _CONTRACT_FILE, label="NNUE runtime contract")
    raw = _read_regular_bytes(
        contract_path, label="NNUE runtime contract", maximum_bytes=4 * 1024 * 1024
    )
    payload = _load_json_object(raw, label="NNUE runtime contract")
    contract = _contract_from_payload(payload)
    _expected_runtime_tree(root)
    _validate_sanitized_export_receipt(root, contract=contract)
    for identity in (contract.engine, *contract.artifacts):
        observed = _runtime_identity(root, identity.relative_path)
        if observed != identity:
            raise ValueError(f"runtime artifact identity mismatch: {identity.relative_path}")
    engine_mode = (root / contract.engine.relative_path).stat(follow_symlinks=False).st_mode
    if not engine_mode & stat.S_IXUSR:
        raise PermissionError("staged YaneuraOu engine is not owner-executable")
    return contract


def verify_nnue_runtime(
    runtime_directory: Path,
    *,
    repeat_load_smoke: bool = False,
    timeout_seconds: float = 300.0,
) -> NnueRuntimeContract:
    """Verify hashes and optionally start a third independent engine process."""

    contract = load_nnue_runtime_contract(runtime_directory)
    if not repeat_load_smoke:
        return contract
    applied = contract.reload_load_smoke.applied_options
    multipv_rows = [item for item in applied if item.name.casefold() == "multipv"]
    if len(multipv_rows) != 1:
        raise ValueError("runtime contract must contain exactly one MultiPV option")
    multipv = _require_int(int(multipv_rows[0].requested_value), label="runtime MultiPV", minimum=1)
    options = tuple(
        (item.name, item.requested_value) for item in applied if item.name.casefold() != "multipv"
    )
    observed = _run_load_smoke(
        _require_directory(runtime_directory, label="NNUE runtime directory"),
        options=options,
        multipv=multipv,
        nodes=contract.reload_load_smoke.requested_nodes,
        timeout_seconds=timeout_seconds,
    )
    if observed.semantic_identity() != contract.reload_load_smoke.semantic_identity():
        raise RuntimeError("installed NNUE runtime load smoke differs from staged reload evidence")
    return contract


def stage_nnue_runtime(
    export_directory: Path,
    engine: Path,
    destination: Path,
    *,
    profile: RuntimeProfile,
) -> NnueRuntimeContract:
    """Atomically install one private Meteo export as a verified YaneuraOu runtime."""

    source = _validate_export(export_directory)
    engine_source = _require_regular_file(engine, label="YaneuraOu engine")
    engine_mode = engine_source.stat(follow_symlinks=False).st_mode
    if not engine_mode & stat.S_IXUSR:
        raise PermissionError("YaneuraOu source engine must be owner-executable")
    target = _reject_symlink_components(destination, label="NNUE runtime destination")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite NNUE runtime destination: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(target.parent, label="NNUE runtime destination parent")
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    published_directory_identity: tuple[int, int] | None = None
    try:
        (stage / "engine").mkdir(mode=0o700)
        (stage / "eval").mkdir(mode=0o700)
        copied_engine_sha256 = _copy_regular_create_only(
            engine_source, stage / _ENGINE_RELATIVE_PATH, executable=True
        )
        copied_nn_sha256 = _copy_regular_create_only(
            source.nn_bin, stage / _NN_RELATIVE_PATH, executable=False
        )
        copied_progress_sha256 = _copy_regular_create_only(
            source.progress_bin, stage / _PROGRESS_RELATIVE_PATH, executable=False
        )
        copied_options_sha256 = _copy_regular_create_only(
            source.eval_options, stage / _EVAL_OPTIONS_RELATIVE_PATH, executable=False
        )
        if copied_nn_sha256 != source.identity.nn_bin_sha256:
            raise RuntimeError("staged nn.bin changed during copy")
        if copied_progress_sha256 != source.identity.progress_bin_sha256:
            raise RuntimeError("staged progress.bin changed during copy")
        if copied_options_sha256 != source.eval_options_sha256:
            raise RuntimeError("staged eval_options.txt changed during copy")
        sanitized_receipt: dict[str, object] = {
            "schema": SANITIZED_EXPORT_RECEIPT_SCHEMA,
            "source_export": source.identity.to_dict(),
            "files": [
                {
                    "relative_path": _NN_RELATIVE_PATH,
                    "sha256": copied_nn_sha256,
                    "bytes": source.nn_bytes,
                },
                {
                    "relative_path": _PROGRESS_RELATIVE_PATH,
                    "sha256": copied_progress_sha256,
                    "bytes": source.progress_bytes,
                },
                {
                    "relative_path": _EVAL_OPTIONS_RELATIVE_PATH,
                    "sha256": copied_options_sha256,
                    "bytes": source.eval_options_bytes,
                },
            ],
            "local_only": True,
            "publication_allowed": False,
            "raw_source_metadata_copied": False,
        }
        sanitized_receipt["receipt_sha256"] = _json_sha256(sanitized_receipt)
        _assert_contract_has_no_private_paths(sanitized_receipt, label="sanitized export receipt")
        _write_json_create_only(stage / _SANITIZED_RECEIPT_FILE, sanitized_receipt)
        _fsync_directory(stage / "engine")
        _fsync_directory(stage / "eval")

        options = profile.requested_options(
            fv_scale=source.identity.yaneuraou_fv_scale,
            routing=source.identity.routing,
        )
        initial_smoke = _run_load_smoke(
            stage,
            options=options,
            multipv=profile.multipv,
            nodes=profile.smoke_nodes,
            timeout_seconds=profile.timeout_seconds,
        )
        reload_smoke = _run_load_smoke(
            stage,
            options=options,
            multipv=profile.multipv,
            nodes=profile.smoke_nodes,
            timeout_seconds=profile.timeout_seconds,
        )
        if initial_smoke.semantic_identity() != reload_smoke.semantic_identity():
            raise RuntimeError("independent NNUE runtime load smokes were inconsistent")

        engine_identity = _runtime_identity(stage, _ENGINE_RELATIVE_PATH)
        if engine_identity.sha256 != copied_engine_sha256:
            raise RuntimeError("staged engine identity changed before contract publication")
        artifacts = tuple(
            _runtime_identity(stage, relative_path)
            for relative_path in _RUNTIME_FILE_PATHS
            if relative_path != _ENGINE_RELATIVE_PATH
        )
        provisional = NnueRuntimeContract(
            schema=NNUE_RUNTIME_CONTRACT_SCHEMA,
            profile_id=profile.profile_id,
            scope="private_local_only",
            publication_allowed=False,
            runtime_content_sha256="0" * 64,
            source_export=source.identity,
            engine=engine_identity,
            artifacts=artifacts,
            initial_load_smoke=initial_smoke,
            reload_load_smoke=reload_smoke,
            reload_consistent=True,
            contract_sha256="0" * 64,
        )
        content_sha256 = _json_sha256(provisional.content_payload())
        with_content = replace(provisional, runtime_content_sha256=content_sha256)
        contract_without_hash = with_content.to_dict(include_hash=False)
        contract = replace(
            with_content,
            contract_sha256=_json_sha256(contract_without_hash),
        )
        contract_payload = contract.to_dict()
        _assert_contract_has_no_private_paths(contract_payload)
        _write_json_create_only(stage / _CONTRACT_FILE, contract_payload)
        _fsync_directory(stage)
        loaded_stage = load_nnue_runtime_contract(stage)
        if loaded_stage != contract:
            raise RuntimeError("staged NNUE runtime changed when its contract was reloaded")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"NNUE runtime destination appeared during staging: {target}")
        os.rename(stage, target)
        published_metadata = target.stat(follow_symlinks=False)
        published_directory_identity = (
            published_metadata.st_dev,
            published_metadata.st_ino,
        )
        _fsync_directory(target.parent)
        installed = load_nnue_runtime_contract(target)
        if installed != contract:
            raise RuntimeError("installed NNUE runtime differs after atomic rename")
        return installed
    except BaseException:
        if (
            stage.exists()
            and stage.parent == target.parent
            and stage.name.startswith(f".{target.name}.stage-")
        ):
            shutil.rmtree(stage)
        if published_directory_identity is not None and os.path.lexists(target):
            installed_metadata = target.stat(follow_symlinks=False)
            observed_identity = (installed_metadata.st_dev, installed_metadata.st_ino)
            if (
                stat.S_ISDIR(installed_metadata.st_mode)
                and observed_identity == published_directory_identity
            ):
                shutil.rmtree(target)
                _fsync_directory(target.parent)
        raise


def _runtime_reference(runtime: Path, *, registry_parent: Path) -> RuntimeReference:
    root = _require_directory(runtime, label="registered NNUE runtime")
    parent = _require_directory(registry_parent, label="runtime registry parent")
    try:
        relative = root.relative_to(parent)
    except ValueError as error:
        raise ValueError("registered NNUE runtimes must be inside the registry parent") from error
    relative_text = _safe_relative_path(relative.as_posix(), label="registered runtime path")
    contract = load_nnue_runtime_contract(root)
    return RuntimeReference(
        relative_directory=relative_text,
        profile_id=contract.profile_id,
        runtime_content_sha256=contract.runtime_content_sha256,
        contract_sha256=contract.contract_sha256,
        nn_bin_sha256=contract.nn_bin_sha256,
    )


def publish_nnue_runtime_registry(
    registry_path: Path,
    *,
    latest_runtime: Path,
    previous_runtime: Path | None,
    champion_runtime: Path,
) -> NnueRuntimeRegistry:
    """Atomically publish latest/previous/champion pointers with at most two weights."""

    destination = _absolute_lexical(registry_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = _require_directory(destination.parent, label="runtime registry parent")
    latest = _runtime_reference(latest_runtime, registry_parent=parent)
    previous = (
        None
        if previous_runtime is None
        else _runtime_reference(previous_runtime, registry_parent=parent)
    )
    champion = _runtime_reference(champion_runtime, registry_parent=parent)
    provisional = NnueRuntimeRegistry(
        latest=latest,
        previous=previous,
        champion=champion,
        registry_sha256="0" * 64,
    )
    digest = _json_sha256(provisional.to_dict(include_hash=False))
    registry = NnueRuntimeRegistry(
        latest=latest,
        previous=previous,
        champion=champion,
        registry_sha256=digest,
    )
    payload = registry.to_dict()
    _assert_contract_has_no_private_paths(payload, label="runtime registry")
    _write_json_atomic(destination, payload)
    loaded = load_nnue_runtime_registry(destination)
    if loaded != registry:
        raise RuntimeError("runtime registry changed when reloaded after atomic publication")
    return loaded


def load_nnue_runtime_registry(registry_path: Path) -> NnueRuntimeRegistry:
    """Reload a registry and verify every referenced runtime contract and weight."""

    path = _require_regular_file(registry_path, label="NNUE runtime registry")
    raw = _read_regular_bytes(path, label="NNUE runtime registry", maximum_bytes=1024 * 1024)
    payload = _load_json_object(raw, label="NNUE runtime registry")
    expected = {
        "schema",
        "latest",
        "previous",
        "champion",
        "playable_weight_window_limit",
        "playable_weight_count",
        "champion_must_reference_playable_window",
        "registry_sha256",
    }
    if set(payload) != expected or payload["schema"] != NNUE_RUNTIME_REGISTRY_SCHEMA:
        raise ValueError("unsupported or malformed NNUE runtime registry")
    latest = RuntimeReference.from_object(payload["latest"])
    previous = (
        None if payload["previous"] is None else RuntimeReference.from_object(payload["previous"])
    )
    champion = RuntimeReference.from_object(payload["champion"])
    count = 1 if previous is None else 2
    if payload["playable_weight_window_limit"] != 2 or payload["playable_weight_count"] != count:
        raise ValueError("runtime registry playable weight count is invalid")
    if payload["champion_must_reference_playable_window"] is not True:
        raise ValueError("runtime registry champion reference rule is disabled")
    registry = NnueRuntimeRegistry(
        latest=latest,
        previous=previous,
        champion=champion,
        registry_sha256=_require_sha256(
            payload["registry_sha256"], label="runtime registry SHA-256"
        ),
    )
    without_hash = dict(payload)
    without_hash.pop("registry_sha256")
    if registry.registry_sha256 != _json_sha256(without_hash):
        raise ValueError("runtime registry SHA-256 is invalid")
    parent = path.parent
    references = {latest, champion}
    if previous is not None:
        references.add(previous)
    for reference in references:
        runtime = _require_directory(
            parent / reference.relative_directory, label="registered NNUE runtime"
        )
        contract = load_nnue_runtime_contract(runtime)
        observed = RuntimeReference(
            relative_directory=reference.relative_directory,
            profile_id=contract.profile_id,
            runtime_content_sha256=contract.runtime_content_sha256,
            contract_sha256=contract.contract_sha256,
            nn_bin_sha256=contract.nn_bin_sha256,
        )
        if observed != reference:
            raise ValueError(f"runtime registry reference is stale: {reference.relative_directory}")
    _assert_contract_has_no_private_paths(dict(payload), label="runtime registry")
    return registry
