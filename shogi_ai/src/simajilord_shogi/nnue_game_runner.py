"""Strict standalone runner for external-USI NNUE trajectory generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .external_usi import ExternalTeacherPolicy, UsiOptionValueVerification
from .model_rights import DistillationScope, model_rights
from .nnue_game_generation import (
    GeneratedNnueGame,
    NnueGameGenerationConfig,
    NnueGameGenerationResult,
    NnueGameGenerator,
    NnueGenerationOpening,
    UsiEngineSpec,
)
from .nnue_runtime import NnueRuntimeContract, load_nnue_runtime_contract

NNUE_GAME_RUNNER_CONFIG_SCHEMA = "meteo-nnue-game-runner-config-v1"
NNUE_GAME_BUNDLE_RECEIPT_SCHEMA = "meteo-nnue-game-bundle-receipt-v1"
NNUE_GAME_JSONL_SCHEMA = "meteo-nnue-generated-game-v1"


@dataclass(frozen=True, slots=True)
class PinnedEngineArtifact:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ValueError(f"pinned engine artifact path must be absolute: {self.path}")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError(
                "pinned engine artifact sha256 must be 64 lowercase hexadecimal characters"
            )

    def revalidate(self, *, actor_source: str) -> None:
        if self.path.is_symlink():
            raise RuntimeError(
                f"pinned engine artifact became a symlink for {actor_source!r}: {self.path}"
            )
        try:
            resolved = self.path.resolve(strict=True)
        except OSError as error:
            raise RuntimeError(
                f"pinned engine artifact is missing for {actor_source!r}: {self.path}"
            ) from error
        if resolved != self.path or not resolved.is_file():
            raise RuntimeError(
                f"pinned engine artifact changed type or path for {actor_source!r}: {self.path}"
            )
        observed = _sha256_file(resolved)
        if observed != self.sha256:
            raise RuntimeError(
                f"pinned engine artifact hash changed for {actor_source!r}: {self.path}"
            )

    def to_receipt(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "path_scope": "private_local_provenance_only",
            "path_publication_allowed": False,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class LoadedEngineInput:
    actor_source: str
    kind: str
    rights_id: str | None = None
    runtime_directory: Path | None = None
    runtime_contract: NnueRuntimeContract | None = None
    pinned_artifacts: tuple[PinnedEngineArtifact, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "rights_profile":
            if self.rights_id is None or self.runtime_directory is not None:
                raise ValueError("rights-profile engine input has inconsistent provenance")
            if self.runtime_contract is not None:
                raise ValueError("rights-profile engine input cannot carry a runtime contract")
            if not self.pinned_artifacts:
                raise ValueError("rights-profile engine input requires pinned artifacts")
            return
        if self.kind != "meteo_runtime_contract":
            raise ValueError(f"unknown engine input kind: {self.kind!r}")
        if (
            self.rights_id is not None
            or self.runtime_directory is None
            or self.runtime_contract is None
        ):
            raise ValueError("Meteo runtime engine input has inconsistent provenance")

    def revalidate(self) -> None:
        for artifact in self.pinned_artifacts:
            artifact.revalidate(actor_source=self.actor_source)
        if self.runtime_directory is None:
            return
        observed = load_nnue_runtime_contract(self.runtime_directory)
        if observed != self.runtime_contract:
            raise RuntimeError(
                f"Meteo runtime contract changed after config load: {self.actor_source!r}"
            )

    def to_receipt(self) -> dict[str, object]:
        if self.kind == "rights_profile":
            return {
                "actor_source": self.actor_source,
                "kind": self.kind,
                "rights_id": self.rights_id,
                "meteo_runtime_contract": None,
                "pinned_artifacts": [artifact.to_receipt() for artifact in self.pinned_artifacts],
            }
        assert self.runtime_directory is not None
        assert self.runtime_contract is not None
        return {
            "actor_source": self.actor_source,
            "kind": self.kind,
            "rights_id": None,
            "pinned_artifacts": [],
            "meteo_runtime_contract": {
                "runtime_directory": str(self.runtime_directory),
                "path_scope": "private_local_provenance_only",
                "path_publication_allowed": False,
                "contract": self.runtime_contract.to_dict(),
            },
        }


@dataclass(frozen=True, slots=True)
class LoadedNnueGameRunnerConfig:
    source: Path
    source_sha256: str
    source_bytes: int
    first: UsiEngineSpec
    second: UsiEngineSpec
    engine_inputs: tuple[LoadedEngineInput, LoadedEngineInput]
    generation: NnueGameGenerationConfig

    def revalidate_engine_inputs(self) -> None:
        for engine_input in self.engine_inputs:
            engine_input.revalidate()

    def revalidate_meteo_runtime_contracts(self) -> None:
        """Compatibility alias; all pinned engine inputs are now revalidated."""

        self.revalidate_engine_inputs()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    normalized_keys: dict[str, str] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"runner config contains duplicate JSON object key {key!r}")
        normalized = key.casefold()
        previous = normalized_keys.get(normalized)
        if previous is not None:
            raise ValueError(
                "runner config contains duplicate JSON object keys ignoring case: "
                f"{previous!r} and {key!r}"
            )
        normalized_keys[normalized] = key
        result[key] = value
    return result


def _object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _array(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return cast(list[object], value)


def _exact_fields(
    value: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    label: str,
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ValueError(f"{label} is missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {sorted(unknown)}")


def _string(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError(f"{label} must be a non-empty trimmed single-line string")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a JSON boolean")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive JSON integer")
    return value


def _positive_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive JSON number")
    result = float(value)
    if not (result > 0.0 and result < float("inf")):
        raise ValueError(f"{label} must be finite and positive")
    return result


def _resolved_config_path(value: object, *, source: Path, label: str) -> Path:
    rendered = _string(value, label=label)
    path = Path(rendered).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve(strict=False)


def _lexical_config_path(value: object, *, source: Path, label: str) -> Path:
    """Resolve config-relative syntax without hiding a symlink from the runtime loader."""

    rendered = _string(value, label=label)
    path = Path(rendered).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    return Path(os.path.abspath(os.fspath(path)))


def _opening(value: object, *, index: int) -> NnueGenerationOpening:
    row = _object(value, label=f"opening {index}")
    _exact_fields(
        row,
        required=frozenset({"initial_sfen", "moves"}),
        label=f"opening {index}",
    )
    moves = _array(row["moves"], label=f"opening {index} moves")
    canonical_moves = tuple(
        _string(move, label=f"opening {index} move {move_index}")
        for move_index, move in enumerate(moves)
    )
    return NnueGenerationOpening(
        initial_sfen=_string(row["initial_sfen"], label=f"opening {index} initial_sfen"),
        moves=canonical_moves,
    )


def _engine_options(value: object, *, engine_index: int) -> tuple[tuple[str, str | int], ...]:
    rows = _array(value, label=f"engine {engine_index} options")
    options: list[tuple[str, str | int]] = []
    normalized_names: set[str] = set()
    for option_index, raw_option in enumerate(rows):
        row = _object(raw_option, label=f"engine {engine_index} option {option_index}")
        _exact_fields(
            row,
            required=frozenset({"name", "value"}),
            label=f"engine {engine_index} option {option_index}",
        )
        name = _string(row["name"], label=f"engine {engine_index} option {option_index} name")
        normalized = name.casefold()
        if normalized in normalized_names:
            raise ValueError(
                f"engine {engine_index} contains duplicate USI option ignoring case: {name}"
            )
        if normalized == "multipv":
            raise ValueError("game runner fixes MultiPV=1; omit it from engine options")
        raw_value = row["value"]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (str, int)):
            raise ValueError(
                f"engine {engine_index} option {name!r} value must be a string or integer"
            )
        if isinstance(raw_value, str) and ("\n" in raw_value or "\r" in raw_value):
            raise ValueError(f"engine {engine_index} option {name!r} contains a line break")
        normalized_names.add(normalized)
        options.append((name, raw_value))
    return tuple(options)


def _fatal_diagnostics(value: object, *, engine_index: int) -> tuple[tuple[str, str], ...]:
    rows = _array(value, label=f"engine {engine_index} expected diagnostics")
    diagnostics: list[tuple[str, str]] = []
    for diagnostic_index, raw_diagnostic in enumerate(rows):
        row = _object(
            raw_diagnostic,
            label=f"engine {engine_index} expected diagnostic {diagnostic_index}",
        )
        _exact_fields(
            row,
            required=frozenset({"channel", "line"}),
            label=f"engine {engine_index} expected diagnostic {diagnostic_index}",
        )
        channel = _string(
            row["channel"],
            label=f"engine {engine_index} expected diagnostic {diagnostic_index} channel",
        )
        if channel not in {"stdout", "stderr"}:
            raise ValueError("expected USI diagnostic channel must be stdout or stderr")
        line = _string(
            row["line"],
            label=f"engine {engine_index} expected diagnostic {diagnostic_index} line",
        )
        diagnostics.append((channel, line))
    return tuple(diagnostics)


_COMMON_ENGINE_FIELDS = frozenset(
    {
        "actor_source",
        "nodes",
        "timeout_seconds",
        "policy_temperature",
        "value_scale",
    }
)
_RIGHTS_ENGINE_FIELDS = _COMMON_ENGINE_FIELDS | {
    "rights_id",
    "allow_limited_local",
    "command",
    "options",
    "working_directory",
    "option_value_verification",
    "expected_fatal_startup_diagnostics",
    "artifact_identities",
}
_METEO_RUNTIME_ENGINE_FIELDS = _COMMON_ENGINE_FIELDS | {"meteo_runtime_contract"}


def _common_engine_values(
    row: Mapping[str, object], *, engine_index: int
) -> tuple[str, int, float, float, float]:
    return (
        _string(row["actor_source"], label=f"engine {engine_index} actor_source"),
        _positive_integer(row["nodes"], label=f"engine {engine_index} nodes"),
        _positive_number(row["timeout_seconds"], label=f"engine {engine_index} timeout_seconds"),
        _positive_number(
            row["policy_temperature"],
            label=f"engine {engine_index} policy_temperature",
        ),
        _positive_number(row["value_scale"], label=f"engine {engine_index} value_scale"),
    )


def _rights_engine_spec(
    row: Mapping[str, object],
    *,
    source: Path,
    engine_index: int,
) -> tuple[UsiEngineSpec, LoadedEngineInput]:
    _exact_fields(
        row,
        required=frozenset(_RIGHTS_ENGINE_FIELDS),
        label=f"engine {engine_index}",
    )
    actor_source, nodes, timeout_seconds, policy_temperature, value_scale = _common_engine_values(
        row, engine_index=engine_index
    )
    rights_id = _string(row["rights_id"], label=f"engine {engine_index} rights_id")
    rights = model_rights(rights_id)
    allow_limited = _boolean(
        row["allow_limited_local"],
        label=f"engine {engine_index} allow_limited_local",
    )
    if rights.distillation_scope is DistillationScope.LOCAL_AUTHORIZED_ONLY:
        if not allow_limited:
            raise PermissionError(
                f"engine {engine_index} rights profile {rights_id!r} requires an explicit "
                "allow_limited_local=true acknowledgement"
            )
    elif allow_limited:
        raise ValueError(
            f"engine {engine_index} allow_limited_local is only valid for a "
            "LOCAL_AUTHORIZED_ONLY rights profile"
        )
    if rights.distillation_scope is DistillationScope.NOT_AUTHORIZED:
        raise PermissionError(
            f"engine {engine_index} rights profile {rights_id!r} is not authorized"
        )

    raw_command = _array(row["command"], label=f"engine {engine_index} command")
    if not raw_command:
        raise ValueError(f"engine {engine_index} command must not be empty")
    command = tuple(
        _string(argument, label=f"engine {engine_index} command argument {argument_index}")
        for argument_index, argument in enumerate(raw_command)
    )
    raw_artifacts = _array(
        row["artifact_identities"], label=f"engine {engine_index} artifact_identities"
    )
    if not raw_artifacts:
        raise ValueError(f"engine {engine_index} artifact_identities must not be empty")
    pinned_artifacts: list[PinnedEngineArtifact] = []
    for artifact_index, raw_artifact in enumerate(raw_artifacts):
        artifact = _object(
            raw_artifact,
            label=f"engine {engine_index} artifact identity {artifact_index}",
        )
        _exact_fields(
            artifact,
            required=frozenset({"path", "sha256"}),
            label=f"engine {engine_index} artifact identity {artifact_index}",
        )
        lexical = _lexical_config_path(
            artifact["path"],
            source=source,
            label=f"engine {engine_index} artifact path {artifact_index}",
        )
        if lexical.is_symlink():
            raise ValueError(f"engine {engine_index} artifact must not be a symlink: {lexical}")
        try:
            resolved_artifact = lexical.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"engine {engine_index} artifact does not exist: {lexical}") from error
        if resolved_artifact != lexical or not resolved_artifact.is_file():
            raise ValueError(
                f"engine {engine_index} artifact must be a regular non-symlink file: {lexical}"
            )
        expected_sha256 = _string(
            artifact["sha256"],
            label=f"engine {engine_index} artifact sha256 {artifact_index}",
        )
        pinned = PinnedEngineArtifact(path=resolved_artifact, sha256=expected_sha256)
        pinned.revalidate(actor_source=actor_source)
        pinned_artifacts.append(pinned)
    artifact_paths = tuple(artifact.path for artifact in pinned_artifacts)
    if tuple(sorted(artifact_paths, key=os.fspath)) != artifact_paths:
        raise ValueError(f"engine {engine_index} artifact_identities must be sorted by path")
    if len(set(artifact_paths)) != len(artifact_paths):
        raise ValueError(f"engine {engine_index} artifact_identities contains a duplicate path")
    executable = Path(command[0]).expanduser()
    if not executable.is_absolute():
        raise ValueError(f"engine {engine_index} command executable must be an absolute path")
    executable = executable.resolve(strict=False)
    if executable not in artifact_paths:
        raise ValueError(
            f"engine {engine_index} command executable must be included in artifact_identities"
        )
    working_directory_value = row["working_directory"]
    working_directory = (
        None
        if working_directory_value is None
        else _resolved_config_path(
            working_directory_value,
            source=source,
            label=f"engine {engine_index} working_directory",
        )
    )
    try:
        verification = UsiOptionValueVerification(
            _string(
                row["option_value_verification"],
                label=f"engine {engine_index} option_value_verification",
            )
        )
    except ValueError as error:
        choices = ", ".join(mode.value for mode in UsiOptionValueVerification)
        raise ValueError(
            f"engine {engine_index} option_value_verification must be one of: {choices}"
        ) from error

    spec = UsiEngineSpec.from_rights_profile(
        actor_source=actor_source,
        command=command,
        rights_id=rights_id,
        nodes=nodes,
        allow_limited_local=allow_limited,
        options=_engine_options(row["options"], engine_index=engine_index),
        timeout_seconds=timeout_seconds,
        working_directory=working_directory,
        policy_temperature=policy_temperature,
        value_scale=value_scale,
        option_value_verification=verification,
        expected_fatal_startup_diagnostics=_fatal_diagnostics(
            row["expected_fatal_startup_diagnostics"],
            engine_index=engine_index,
        ),
        artifact_paths=artifact_paths,
    )
    return spec, LoadedEngineInput(
        actor_source=actor_source,
        kind="rights_profile",
        rights_id=rights_id,
        pinned_artifacts=tuple(pinned_artifacts),
    )


def _meteo_runtime_engine_spec(
    row: Mapping[str, object],
    *,
    source: Path,
    engine_index: int,
) -> tuple[UsiEngineSpec, LoadedEngineInput]:
    _exact_fields(
        row,
        required=frozenset(_METEO_RUNTIME_ENGINE_FIELDS),
        label=f"engine {engine_index}",
    )
    actor_source, nodes, timeout_seconds, policy_temperature, value_scale = _common_engine_values(
        row, engine_index=engine_index
    )
    configured_path = _lexical_config_path(
        row["meteo_runtime_contract"],
        source=source,
        label=f"engine {engine_index} meteo_runtime_contract",
    )
    runtime_directory = (
        configured_path.parent
        if configured_path.name == "runtime-contract.json"
        else configured_path
    )
    contract = load_nnue_runtime_contract(runtime_directory)
    applied_options = contract.initial_load_smoke.applied_options
    multipv_options = tuple(
        option for option in applied_options if option.name.casefold() == "multipv"
    )
    if len(multipv_options) != 1 or multipv_options[0].requested_value != "1":
        raise ValueError("Meteo runtime contract must pin MultiPV=1 for game generation")
    options = tuple(
        (option.name, option.requested_value)
        for option in applied_options
        if option.name.casefold() != "multipv"
    )
    artifacts = (
        runtime_directory / "runtime-contract.json",
        runtime_directory / contract.engine.relative_path,
        *(runtime_directory / artifact.relative_path for artifact in contract.artifacts),
    )
    policy = ExternalTeacherPolicy(
        policy_id=f"meteo-nnue-runtime-{contract.contract_sha256}",
        name=f"Meteo NNUE runtime {contract.profile_id}",
        source=(f"verified local content-addressed NNUE runtime {contract.runtime_content_sha256}"),
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=False,
        training_outputs_local_only=True,
    )
    spec = UsiEngineSpec(
        actor_source=actor_source,
        command=(str(runtime_directory / contract.engine.relative_path),),
        policy=policy,
        nodes=nodes,
        options=options,
        timeout_seconds=timeout_seconds,
        working_directory=runtime_directory,
        policy_temperature=policy_temperature,
        value_scale=value_scale,
        option_value_verification=UsiOptionValueVerification.YANEURAOU_GETOPTION,
        expected_fatal_startup_diagnostics=(),
        artifact_paths=artifacts,
    )
    return spec, LoadedEngineInput(
        actor_source=actor_source,
        kind="meteo_runtime_contract",
        runtime_directory=runtime_directory,
        runtime_contract=contract,
    )


def _engine_spec(
    value: object,
    *,
    source: Path,
    engine_index: int,
) -> tuple[UsiEngineSpec, LoadedEngineInput]:
    row = _object(value, label=f"engine {engine_index}")
    has_rights_id = "rights_id" in row
    has_meteo_runtime = "meteo_runtime_contract" in row
    if has_rights_id == has_meteo_runtime:
        raise ValueError(
            f"engine {engine_index} must provide exactly one of rights_id or meteo_runtime_contract"
        )
    if has_rights_id:
        return _rights_engine_spec(row, source=source, engine_index=engine_index)
    return _meteo_runtime_engine_spec(row, source=source, engine_index=engine_index)


def load_nnue_game_runner_config(path: Path) -> LoadedNnueGameRunnerConfig:
    """Load an immutable strict JSON config before any engine is started."""

    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"runner config must not be a symlink: {expanded}")
    try:
        resolved = expanded.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"runner config does not exist or is unreadable: {expanded}") from error
    if not resolved.is_file():
        raise ValueError(f"runner config must be a regular file: {resolved}")
    source_bytes = resolved.read_bytes()
    try:
        payload = json.loads(
            source_bytes,
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("runner config must be valid UTF-8 JSON") from error
    root = _object(payload, label="runner config root")
    _exact_fields(
        root,
        required=frozenset({"schema", "engines", "generation"}),
        label="runner config root",
    )
    if root["schema"] != NNUE_GAME_RUNNER_CONFIG_SCHEMA:
        raise ValueError(f"runner config schema must be {NNUE_GAME_RUNNER_CONFIG_SCHEMA!r}")
    raw_engines = _array(root["engines"], label="runner config engines")
    if len(raw_engines) != 2:
        raise ValueError("runner config must contain exactly two engines")
    first, first_input = _engine_spec(raw_engines[0], source=resolved, engine_index=0)
    second, second_input = _engine_spec(raw_engines[1], source=resolved, engine_index=1)

    raw_generation = _object(root["generation"], label="runner config generation")
    _exact_fields(
        raw_generation,
        required=frozenset({"generation_id", "max_plies", "parallelism", "openings"}),
        label="runner config generation",
    )
    raw_openings = _array(raw_generation["openings"], label="generation openings")
    if not raw_openings:
        raise ValueError("generation openings must not be empty")
    generation = NnueGameGenerationConfig(
        generation_id=_string(raw_generation["generation_id"], label="generation_id"),
        max_plies=_positive_integer(raw_generation["max_plies"], label="max_plies"),
        parallelism=_positive_integer(raw_generation["parallelism"], label="parallelism"),
        openings=tuple(
            _opening(raw_opening, index=index) for index, raw_opening in enumerate(raw_openings)
        ),
    )
    # Force duplicate-history validation now, before any process starts.
    generation.normalized_openings()
    # Force actor-identity ambiguity validation now as well.
    NnueGameGenerator(first, second, generation)
    return LoadedNnueGameRunnerConfig(
        source=resolved,
        source_sha256=_sha256_bytes(source_bytes),
        source_bytes=len(source_bytes),
        first=first,
        second=second,
        engine_inputs=(first_input, second_input),
        generation=generation,
    )


def _game_json_line(game: GeneratedNnueGame) -> bytes:
    payload = {
        "schema": NNUE_GAME_JSONL_SCHEMA,
        "game": game.record.to_dict(),
        "trajectory_receipt": game.receipt.to_dict(),
    }
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_games_create_only(path: Path, games: tuple[GeneratedNnueGame, ...]) -> None:
    if not games:
        raise ValueError("game bundle cannot contain zero games")
    try:
        with path.open("xb") as stream:
            for game in games:
                stream.write(_game_json_line(game))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite games JSONL: {path}") from error


def _write_json_create_only(path: Path, payload: dict[str, object]) -> None:
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    complete = {
        **payload,
        "payload_sha256": _sha256_bytes(canonical),
        "payload_sha256_scope": "receipt_without_payload_sha256_fields",
    }
    encoded = (
        json.dumps(
            complete,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite bundle receipt: {path}") from error


def _bundle_receipt(
    loaded: LoadedNnueGameRunnerConfig,
    result: NnueGameGenerationResult,
    *,
    games_path: Path,
) -> dict[str, object]:
    generation = result.receipt.to_dict()
    game_receipts = generation.pop("games")
    if not isinstance(game_receipts, list) or len(game_receipts) != len(result.games):
        raise AssertionError("generation receipt contains a mismatched game count")
    return {
        "schema": NNUE_GAME_BUNDLE_RECEIPT_SCHEMA,
        # This raw bundle contains config, command, cwd, artifact, and process
        # provenance.  Model-output rights do not make those private execution
        # paths publishable; publication requires a separate sanitized export.
        "local_only": True,
        "publication_allowed": False,
        "strong_wdl_eligibility": {
            "allowed_only_for_rules_terminal_trajectory": True,
            "allowed_terminations": [
                "checkmate",
                "repetition",
                "entering_king_declaration",
            ],
            "resignation_is_eligible": False,
            "evaluation_adjudication_is_eligible": False,
            "max_plies_is_eligible": False,
            "per_game_gate": "trajectory_receipt.strong_wdl_label_allowed",
            "per_ply_gate": "trajectory_receipt.plies[].strong_value_target",
        },
        "config": {
            "path": str(loaded.source),
            "path_scope": "private_local_provenance_only",
            "path_publication_allowed": False,
            "sha256": loaded.source_sha256,
            "bytes": loaded.source_bytes,
            "schema": NNUE_GAME_RUNNER_CONFIG_SCHEMA,
        },
        "engine_inputs": [engine_input.to_receipt() for engine_input in loaded.engine_inputs],
        "games_jsonl": {
            "path": games_path.name,
            "schema": NNUE_GAME_JSONL_SCHEMA,
            "games": len(result.games),
            "bytes": games_path.stat().st_size,
            "sha256": _sha256_file(games_path),
            "contains_complete_trajectory_receipts": True,
        },
        "generation": generation,
    }


def _publish_bundle_create_only(
    loaded: LoadedNnueGameRunnerConfig,
    result: NnueGameGenerationResult,
    output_directory: Path,
) -> tuple[Path, Path]:
    destination = Path(os.path.abspath(os.fspath(output_directory.expanduser())))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite game-generation bundle: {destination}")
    lock_path = destination.parent / f".{destination.name}.nnue-games.lock"
    lock_descriptor: int | None = None
    temporary: Path | None = None
    try:
        try:
            lock_descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"another publisher already reserved this bundle: {destination}"
            ) from error
        os.write(lock_descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(lock_descriptor)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.tmp-",
                dir=str(destination.parent),
            )
        )
        games_path = temporary / "games.jsonl"
        receipt_path = temporary / "receipt.json"
        _write_games_create_only(games_path, result.games)
        _write_json_create_only(
            receipt_path,
            _bundle_receipt(loaded, result, games_path=games_path),
        )
        directory_descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite game-generation bundle: {destination}")
        os.rename(temporary, destination)
        temporary = None
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return destination / "games.jsonl", destination / "receipt.json"
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        lock_path.unlink(missing_ok=True)


def run_nnue_game_bundle(config_path: Path, output_directory: Path) -> dict[str, object]:
    """Generate, atomically publish, and summarize one strict paired-game bundle."""

    output = Path(os.path.abspath(os.fspath(output_directory.expanduser())))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite game-generation bundle: {output}")
    loaded = load_nnue_game_runner_config(config_path)
    loaded.revalidate_engine_inputs()
    result = NnueGameGenerator(
        loaded.first,
        loaded.second,
        loaded.generation,
    ).generate()
    loaded.revalidate_engine_inputs()
    games_path, receipt_path = _publish_bundle_create_only(loaded, result, output)
    return {
        "schema": NNUE_GAME_BUNDLE_RECEIPT_SCHEMA,
        "output": str(output),
        "games_jsonl": str(games_path),
        "receipt": str(receipt_path),
        "metrics": result.receipt.metrics.to_dict(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simajilord-nnue-games",
        description="Generate complete external-USI NNUE game trajectories",
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_nnue_game_bundle(args.config, args.output)
    print(
        json.dumps(
            summary,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
