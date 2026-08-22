"""Strict production runner for canonical external-USI score matrices.

The JSON contract deliberately fails closed.  It binds every canonical scorer
to the reviewed rights row, executable bytes, evaluation/runtime artifacts,
verified USI options, parser/scorer/calibration artifacts, and the exact game
history.  A-N/A-W/A-S run one canonical proposer, B runs all three, and B9
adds independently receipted proposals from the other six registered models.
Every proposal process is separate from ``MultiPV=1`` scoring.  One configured
anchor alone scores the candidate/reply union; proposal membership, an actual
played move, majority agreement, and cross-teacher averaging never become a
correctness label.

This runner operates on the exact history root.  It does not perform qsearch
leaf conversion or claim that a qsearch leaf was re-evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn, cast

from .artifact_provenance import sha256_file
from .distillation_targets import CANONICAL_SCORER_IDS
from .external_usi import (
    ExternalUsiTeacher,
    UsiOptionValueVerification,
    UsiPositionHistory,
)
from .model_rights import DistillationScope, model_rights
from .production_score_matrix import (
    ACTUAL_PLAYED_CANDIDATE_SOURCE_ID,
    AUXILIARY_PROPOSER_IDS,
    EXPERIMENT_ARM_AUXILIARY_SOURCES,
    EXPERIMENT_ARM_PROPOSAL_SOURCES,
    METEO_CANDIDATE_SOURCE_ID,
    SEARCH_STATE_RESET_COMMAND,
    AuxiliaryProposalAuthorization,
    CandidateProposalEvidence,
    CandidateSourceKind,
    CanonicalProposalIdentity,
    CanonicalScorerIdentity,
    ExternalUsiProductionScorer,
    ProductionScoreMatrixBuilder,
    ProductionScoreMatrixReceipt,
    history_context_sha256,
)

SCORE_MATRIX_RUNNER_CONFIG_SCHEMA = "meteo-score-matrix-runner-config-v2"
SCORE_MATRIX_RUNNER_SUMMARY_SCHEMA = "meteo-score-matrix-runner-summary-v2"
PRIVATE_OUTPUT_SCOPE = "private_local_only"
LIMITED_LOCAL_SCOPE = "private_local_distillation_only"
REQUIRED_OPTION_VERIFICATION = UsiOptionValueVerification.YANEURAOU_GETOPTION

_SHA256_HEXDIGITS = frozenset("0123456789abcdef")


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    normalized_keys: dict[str, str] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        normalized = key.casefold()
        if normalized in normalized_keys:
            raise ValueError(
                "case-insensitive JSON object key collision: "
                f"{normalized_keys[normalized]!r} and {key!r}"
            )
        normalized_keys[normalized] = key
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _lexical_absolute(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {path}")
    return Path(os.path.abspath(os.fspath(expanded)))


def _reject_symlink_components(path: Path, *, label: str) -> Path:
    absolute = _lexical_absolute(path, label=label)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink: {current}")
    return absolute


def _regular_file(path: Path, *, label: str) -> Path:
    absolute = _reject_symlink_components(path, label=label)
    try:
        metadata = absolute.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise FileNotFoundError(absolute) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {absolute}")
    return absolute


def _executable_file(path: Path, *, label: str) -> Path:
    executable = _regular_file(path, label=label)
    if not os.access(executable, os.X_OK):
        raise PermissionError(f"{label} is not executable: {executable}")
    return executable


def _regular_directory(path: Path, *, label: str) -> Path:
    absolute = _reject_symlink_components(path, label=label)
    try:
        metadata = absolute.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise FileNotFoundError(absolute) from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink directory: {absolute}")
    return absolute


def _create_only_destination(path: Path) -> Path:
    destination = _lexical_absolute(path, label="score-matrix output")
    parent = _regular_directory(destination.parent, label="score-matrix output parent")
    destination = parent / destination.name
    if os.path.lexists(destination):
        raise FileExistsError(
            f"refusing to overwrite production score-matrix receipt: {destination}"
        )
    return destination


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _sequence(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _expect_keys(value: Mapping[str, object], keys: set[str], *, label: str) -> None:
    observed = set(value)
    if observed != keys:
        missing = sorted(keys - observed)
        extra = sorted(observed - keys)
        raise ValueError(f"{label} fields mismatch: missing={missing!r} extra={extra!r}")


def _string(value: object, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if not allow_empty and (not value or value.strip() != value):
        raise ValueError(f"{label} must be a non-empty trimmed string")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be a single-line string")
    return value


def _sha256(value: object, *, label: str) -> str:
    digest = _string(value, label=label)
    if len(digest) != 64 or any(character not in _SHA256_HEXDIGITS for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_float(value: object, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    result = tuple(
        _string(item, label=f"{label}[{index}]")
        for index, item in enumerate(_sequence(value, label=label))
    )
    return result


def _read_strict_config(path: Path) -> tuple[Path, dict[str, Any], str]:
    source = _regular_file(path, label="score-matrix config")
    raw = source.read_bytes()
    try:
        payload: object = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"score-matrix config is not UTF-8: {source}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"score-matrix config is not valid JSON: {source}") from error
    return source, _mapping(payload, label="score-matrix config"), hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    path: Path
    sha256: str

    def identity(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


def _verified_artifact(value: object, *, label: str) -> VerifiedArtifact:
    raw = _mapping(value, label=label)
    _expect_keys(raw, {"path", "sha256"}, label=label)
    path = _regular_file(Path(_string(raw["path"], label=f"{label}.path")), label=label)
    expected = _sha256(raw["sha256"], label=f"{label}.sha256")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected} observed={observed} path={path}"
        )
    return VerifiedArtifact(path=path, sha256=observed)


def _artifact_set(
    value: object,
    expected_digest: object,
    *,
    label: str,
) -> tuple[tuple[VerifiedArtifact, ...], str]:
    artifacts = tuple(
        _verified_artifact(item, label=f"{label}[{index}]")
        for index, item in enumerate(_sequence(value, label=label))
    )
    if not artifacts:
        raise ValueError(f"{label} must contain at least one artifact")
    paths = tuple(str(artifact.path) for artifact in artifacts)
    if paths != tuple(sorted(set(paths))):
        raise ValueError(f"{label} paths must be unique and sorted")
    digest = _canonical_json_sha256({"artifacts": [artifact.identity() for artifact in artifacts]})
    expected = _sha256(expected_digest, label=f"{label}_sha256")
    if digest != expected:
        raise ValueError(
            f"{label} aggregate SHA-256 mismatch: expected={expected} observed={digest}"
        )
    return artifacts, digest


@dataclass(frozen=True, slots=True)
class CanonicalScorerConfig:
    scorer_id: str
    command: tuple[str, ...]
    working_directory: Path
    engine_sha256: str
    evaluation_artifacts_sha256: str
    options: tuple[tuple[str, str | int], ...]
    hash_option_name: str
    proposal_multipv: int
    book_option_name: str
    book_option_value: str
    parser_sha256: str
    scorer_code_sha256: str
    calibration_sha256: str
    timeout_seconds: float
    value_scale: float
    expected_fatal_startup_diagnostics: tuple[tuple[str, str], ...]

    @property
    def threads(self) -> int:
        return self._integer_option("Threads")

    @property
    def hash_mb(self) -> int:
        return self._integer_option(self.hash_option_name)

    def _integer_option(self, requested_name: str) -> int:
        values = {name.casefold(): value for name, value in self.options}
        raw = values[requested_name.casefold()]
        if isinstance(raw, int):
            return raw
        try:
            parsed = int(raw)
        except ValueError as error:
            raise ValueError(f"{requested_name} option must be an integer") from error
        return parsed


def _option_rows(raw_options: object, *, label: str) -> tuple[tuple[str, str | int], ...]:
    options: list[tuple[str, str | int]] = []
    names: set[str] = set()
    for index, item in enumerate(_sequence(raw_options, label=label)):
        row_label = f"{label}[{index}]"
        row = _mapping(item, label=row_label)
        _expect_keys(row, {"name", "value"}, label=row_label)
        name = _string(row["name"], label=f"{row_label}.name")
        raw_value = row["value"]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (str, int)):
            raise ValueError(f"{row_label}.value must be a string or integer")
        if isinstance(raw_value, str):
            option_value: str | int = _string(
                raw_value,
                label=f"{row_label}.value",
                allow_empty=True,
            )
        else:
            option_value = raw_value
        normalized = name.casefold()
        if normalized == "multipv":
            raise ValueError("MultiPV is fixed to 1 and must not appear in scorer options")
        if normalized in names:
            raise ValueError(f"duplicate scorer option ignoring case: {name}")
        names.add(normalized)
        options.append((name, option_value))
    if options != sorted(options, key=lambda item: item[0].casefold()):
        raise ValueError(f"{label} must be sorted by case-insensitive option name")
    if "threads" not in names:
        raise ValueError(f"{label} must explicitly include Threads")
    return tuple(options)


def _fatal_diagnostics(value: object, *, label: str) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for index, item in enumerate(_sequence(value, label=label)):
        row_label = f"{label}[{index}]"
        row = _mapping(item, label=row_label)
        _expect_keys(row, {"channel", "line"}, label=row_label)
        channel = _string(row["channel"], label=f"{row_label}.channel")
        line = _string(row["line"], label=f"{row_label}.line")
        result.append((channel, line))
    return tuple(result)


def _canonical_scorer_config(value: object, *, index: int) -> CanonicalScorerConfig:
    label = f"canonical_scorers[{index}]"
    raw = _mapping(value, label=label)
    _expect_keys(
        raw,
        {
            "scorer_id",
            "engine",
            "working_directory",
            "evaluation_artifacts",
            "evaluation_artifacts_sha256",
            "options",
            "hash_option_name",
            "proposal_multipv",
            "book_disable",
            "option_value_verification",
            "parser_artifact",
            "scorer_code_artifact",
            "calibration_artifact",
            "timeout_seconds",
            "value_scale",
            "expected_fatal_startup_diagnostics",
        },
        label=label,
    )
    scorer_id = _string(raw["scorer_id"], label=f"{label}.scorer_id")

    engine_raw = _mapping(raw["engine"], label=f"{label}.engine")
    _expect_keys(engine_raw, {"path", "sha256", "arguments"}, label=f"{label}.engine")
    engine_path = _executable_file(
        Path(_string(engine_raw["path"], label=f"{label}.engine.path")),
        label=f"{label}.engine",
    )
    expected_engine_sha256 = _sha256(engine_raw["sha256"], label=f"{label}.engine.sha256")
    observed_engine_sha256 = sha256_file(engine_path)
    if observed_engine_sha256 != expected_engine_sha256:
        raise ValueError(
            f"{label}.engine SHA-256 mismatch: expected={expected_engine_sha256} "
            f"observed={observed_engine_sha256}"
        )
    arguments = _string_tuple(engine_raw["arguments"], label=f"{label}.engine.arguments")
    command = (str(engine_path), *arguments)

    working_directory = _regular_directory(
        Path(_string(raw["working_directory"], label=f"{label}.working_directory")),
        label=f"{label}.working_directory",
    )
    _artifacts, evaluation_artifacts_sha256 = _artifact_set(
        raw["evaluation_artifacts"],
        raw["evaluation_artifacts_sha256"],
        label=f"{label}.evaluation_artifacts",
    )
    options = _option_rows(raw["options"], label=f"{label}.options")
    hash_option_name = _string(raw["hash_option_name"], label=f"{label}.hash_option_name")
    if hash_option_name not in {"Hash", "USI_Hash"}:
        raise ValueError(f"{label}.hash_option_name must be exactly Hash or USI_Hash")
    option_names = {name.casefold(): name for name, _value in options}
    selected_hash_name = option_names.get(hash_option_name.casefold())
    if selected_hash_name != hash_option_name:
        raise ValueError(f"{label}.options must contain the exact declared hash option spelling")
    present_hash_options = set(option_names) & {"hash", "usi_hash"}
    if present_hash_options != {hash_option_name.casefold()}:
        raise ValueError(f"{label}.options must contain exactly one of Hash or USI_Hash")
    proposal_multipv = _integer(
        raw["proposal_multipv"], label=f"{label}.proposal_multipv", minimum=2
    )

    book_raw = _mapping(raw["book_disable"], label=f"{label}.book_disable")
    _expect_keys(
        book_raw,
        {"enabled", "option_name", "option_value"},
        label=f"{label}.book_disable",
    )
    if _boolean(book_raw["enabled"], label=f"{label}.book_disable.enabled"):
        raise ValueError("canonical production scorers must explicitly disable the opening book")
    book_option_name = _string(book_raw["option_name"], label=f"{label}.book_disable.option_name")
    book_option_value = _string(
        book_raw["option_value"],
        label=f"{label}.book_disable.option_value",
        allow_empty=True,
    )
    option_map = {name.casefold(): str(value) for name, value in options}
    if option_map.get(book_option_name.casefold()) != book_option_value:
        raise ValueError("book-disable option must exactly match one configured USI option")

    verification = _string(
        raw["option_value_verification"],
        label=f"{label}.option_value_verification",
    )
    if verification != REQUIRED_OPTION_VERIFICATION.value:
        raise ValueError(
            "production scoring requires YaneuraOu getoption verification for every option"
        )
    parser = _verified_artifact(raw["parser_artifact"], label=f"{label}.parser_artifact")
    scorer_code = _verified_artifact(
        raw["scorer_code_artifact"], label=f"{label}.scorer_code_artifact"
    )
    calibration = _verified_artifact(
        raw["calibration_artifact"], label=f"{label}.calibration_artifact"
    )
    config = CanonicalScorerConfig(
        scorer_id=scorer_id,
        command=command,
        working_directory=working_directory,
        engine_sha256=observed_engine_sha256,
        evaluation_artifacts_sha256=evaluation_artifacts_sha256,
        options=options,
        hash_option_name=hash_option_name,
        proposal_multipv=proposal_multipv,
        book_option_name=book_option_name,
        book_option_value=book_option_value,
        parser_sha256=parser.sha256,
        scorer_code_sha256=scorer_code.sha256,
        calibration_sha256=calibration.sha256,
        timeout_seconds=_finite_float(
            raw["timeout_seconds"], label=f"{label}.timeout_seconds", positive=True
        ),
        value_scale=_finite_float(raw["value_scale"], label=f"{label}.value_scale", positive=True),
        expected_fatal_startup_diagnostics=_fatal_diagnostics(
            raw["expected_fatal_startup_diagnostics"],
            label=f"{label}.expected_fatal_startup_diagnostics",
        ),
    )
    if config.threads != 1:
        raise ValueError("production canonical scoring requires Threads=1")
    if config.hash_mb < 1:
        raise ValueError("production canonical scoring requires a positive hash size")
    return config


def _validate_rights_authorization(value: object, scorer_ids: tuple[str, ...]) -> None:
    raw = _mapping(value, label="limited_local_authorization")
    _expect_keys(raw, {"authorized", "scope", "rights_ids"}, label="limited_local_authorization")
    if not _boolean(raw["authorized"], label="limited_local_authorization.authorized"):
        raise PermissionError("limited-local canonical scorers require explicit authorization")
    if _string(raw["scope"], label="limited_local_authorization.scope") != LIMITED_LOCAL_SCOPE:
        raise PermissionError("limited-local authorization has the wrong scope")
    authorized_ids = _string_tuple(
        raw["rights_ids"], label="limited_local_authorization.rights_ids"
    )
    if authorized_ids != tuple(sorted(set(authorized_ids))):
        raise ValueError("limited-local authorized rights IDs must be unique and sorted")
    required_limited = tuple(
        sorted(
            scorer_id
            for scorer_id in scorer_ids
            if model_rights(scorer_id).distillation_scope is DistillationScope.LOCAL_AUTHORIZED_ONLY
        )
    )
    if authorized_ids != required_limited:
        raise PermissionError(
            "limited-local authorization must list the exact required rights IDs: "
            f"expected={required_limited!r} observed={authorized_ids!r}"
        )
    for scorer_id in scorer_ids:
        rights = model_rights(scorer_id)
        allow_limited = scorer_id in authorized_ids
        policy = rights.teacher_policy(allow_limited_local=allow_limited)
        policy.require_training_permission()
        if allow_limited and (
            not policy.training_outputs_local_only or policy.redistribution_allowed
        ):
            raise PermissionError("limited-local rights policy unexpectedly permits publication")


@dataclass(frozen=True, slots=True)
class ProposalConfig:
    producer: str
    moves: tuple[str, ...]
    requested_nodes: int
    reported_nodes: int
    multipv: int
    sent_commands: tuple[str, ...]
    transcript_lines: tuple[str, ...]
    transcript_sha256: str
    producer_identity_sha256: str
    producer_code_sha256: str


@dataclass(frozen=True, slots=True)
class AuxiliaryProposalConfig:
    source_id: str
    rights_id: str
    proposal: ProposalConfig
    authorization: AuxiliaryProposalAuthorization


def _proposal_config(value: object, *, label: str, single_move: bool) -> ProposalConfig:
    raw = _mapping(value, label=label)
    _expect_keys(
        raw,
        {
            "producer",
            "moves",
            "requested_nodes",
            "reported_nodes",
            "multipv",
            "sent_commands",
            "transcript_lines",
            "transcript_sha256",
            "producer_artifacts",
            "producer_identity_sha256",
            "producer_code_artifact",
        },
        label=label,
    )
    producer = _string(raw["producer"], label=f"{label}.producer")
    moves = _string_tuple(raw["moves"], label=f"{label}.moves")
    if not moves or moves != tuple(sorted(set(moves))):
        raise ValueError(f"{label}.moves must be non-empty, unique, and sorted")
    if single_move and len(moves) != 1:
        raise ValueError(f"{label}.moves must contain exactly one actual played move")
    sent_commands = _string_tuple(raw["sent_commands"], label=f"{label}.sent_commands")
    transcript_lines = _string_tuple(raw["transcript_lines"], label=f"{label}.transcript_lines")
    if not sent_commands or not transcript_lines:
        raise ValueError(f"{label} must preserve non-empty sent and received transcript lines")
    transcript_sha256 = _sha256(raw["transcript_sha256"], label=f"{label}.transcript_sha256")
    expected_transcript = _canonical_json_sha256(
        {"sent_commands": sent_commands, "transcript_lines": transcript_lines}
    )
    if transcript_sha256 != expected_transcript:
        raise ValueError(f"{label} transcript SHA-256 does not match its raw lines")
    _artifacts, producer_identity_sha256 = _artifact_set(
        raw["producer_artifacts"],
        raw["producer_identity_sha256"],
        label=f"{label}.producer_artifacts",
    )
    producer_code = _verified_artifact(
        raw["producer_code_artifact"], label=f"{label}.producer_code_artifact"
    )
    return ProposalConfig(
        producer=producer,
        moves=moves,
        requested_nodes=_integer(raw["requested_nodes"], label=f"{label}.requested_nodes"),
        reported_nodes=_integer(raw["reported_nodes"], label=f"{label}.reported_nodes"),
        multipv=_integer(raw["multipv"], label=f"{label}.multipv"),
        sent_commands=sent_commands,
        transcript_lines=transcript_lines,
        transcript_sha256=transcript_sha256,
        producer_identity_sha256=producer_identity_sha256,
        producer_code_sha256=producer_code.sha256,
    )


def _proposal_evidence(
    config: ProposalConfig,
    *,
    source_id: str,
    source_kind: CandidateSourceKind,
    history: UsiPositionHistory,
) -> CandidateProposalEvidence:
    return CandidateProposalEvidence(
        source_id=source_id,
        producer=config.producer,
        source_kind=source_kind,
        target_sfen=history.target_sfen,
        history_context_sha256=history_context_sha256(history),
        moves=config.moves,
        requested_nodes=config.requested_nodes,
        reported_nodes=config.reported_nodes,
        multipv=config.multipv,
        sent_commands=config.sent_commands,
        transcript_lines=config.transcript_lines,
        transcript_sha256=config.transcript_sha256,
        producer_identity_sha256=config.producer_identity_sha256,
        producer_code_sha256=config.producer_code_sha256,
    )


def _auxiliary_proposal_config(value: object, *, index: int) -> AuxiliaryProposalConfig:
    label = f"auxiliary_proposals[{index}]"
    raw = _mapping(value, label=label)
    _expect_keys(raw, {"source_id", "rights_id", "proposal"}, label=label)
    source_id = _string(raw["source_id"], label=f"{label}.source_id")
    rights_id = _string(raw["rights_id"], label=f"{label}.rights_id")
    if source_id not in AUXILIARY_PROPOSER_IDS or rights_id != source_id:
        raise ValueError(f"{label} source_id and rights_id must name one auxiliary registry row")
    rights = model_rights(rights_id)
    policy = rights.teacher_policy()
    policy.require_training_permission()
    proposal = _proposal_config(raw["proposal"], label=f"{label}.proposal", single_move=False)
    if proposal.requested_nodes < 1 or proposal.reported_nodes < proposal.requested_nodes:
        raise ValueError(f"{label}.proposal must fulfill a positive requested node budget")
    if proposal.multipv < 2:
        raise ValueError(f"{label}.proposal requires MultiPV >= 2")
    if (
        not proposal.sent_commands
        or proposal.sent_commands[0] != SEARCH_STATE_RESET_COMMAND
        or proposal.sent_commands.count(SEARCH_STATE_RESET_COMMAND) != 1
    ):
        raise ValueError(f"{label}.proposal must record exactly one preceding usinewgame reset")
    return AuxiliaryProposalConfig(
        source_id=source_id,
        rights_id=rights_id,
        proposal=proposal,
        authorization=AuxiliaryProposalAuthorization(
            source_id=source_id,
            rights_id=rights_id,
            rights_reviewed_at=rights.reviewed_at,
            distillation_scope=rights.distillation_scope.value,
            rights_registry_identity_sha256=_canonical_json_sha256(rights.to_dict()),
            proposal_identity_sha256=proposal.producer_identity_sha256,
        ),
    )


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    config_sha256: str
    history: UsiPositionHistory
    requested_nodes: int
    root_proposal_nodes: int
    reply_proposal_nodes: int
    interval_dominance_margin: float
    experiment_arm_id: str
    scoring_anchor_id: str
    proposal_source_ids: tuple[str, ...]
    auxiliary_proposal_source_ids: tuple[str, ...]
    canonical_scorers: tuple[CanonicalScorerConfig, ...]
    meteo_proposal: ProposalConfig
    actual_played_move: ProposalConfig | None
    auxiliary_proposals: tuple[AuxiliaryProposalConfig, ...]


def load_runner_config(path: Path) -> RunnerConfig:
    """Read and fully validate one immutable score-matrix run contract."""

    _source, payload, config_sha256 = _read_strict_config(path)
    _expect_keys(
        payload,
        {
            "schema",
            "output_scope",
            "limited_local_authorization",
            "history",
            "search",
            "experiment",
            "canonical_scorers",
            "meteo_proposal",
            "actual_played_move",
            "auxiliary_proposals",
        },
        label="score-matrix config",
    )
    if _string(payload["schema"], label="schema") != SCORE_MATRIX_RUNNER_CONFIG_SCHEMA:
        raise ValueError("unsupported score-matrix runner config schema")
    if _string(payload["output_scope"], label="output_scope") != PRIVATE_OUTPUT_SCOPE:
        raise PermissionError("canonical score matrices containing limited teachers stay private")

    history_raw = _mapping(payload["history"], label="history")
    _expect_keys(history_raw, {"initial_sfen", "moves", "target_sfen"}, label="history")
    history = UsiPositionHistory(
        initial_sfen=_string(history_raw["initial_sfen"], label="history.initial_sfen"),
        moves=_string_tuple(history_raw["moves"], label="history.moves"),
        target_sfen=_string(history_raw["target_sfen"], label="history.target_sfen"),
    )

    search_raw = _mapping(payload["search"], label="search")
    _expect_keys(
        search_raw,
        {
            "requested_nodes",
            "root_proposal_nodes",
            "reply_proposal_nodes",
            "interval_dominance_margin",
        },
        label="search",
    )
    requested_nodes = _integer(
        search_raw["requested_nodes"], label="search.requested_nodes", minimum=1
    )
    root_proposal_nodes = _integer(
        search_raw["root_proposal_nodes"], label="search.root_proposal_nodes", minimum=1
    )
    reply_proposal_nodes = _integer(
        search_raw["reply_proposal_nodes"], label="search.reply_proposal_nodes", minimum=1
    )
    interval_dominance_margin = _finite_float(
        search_raw["interval_dominance_margin"],
        label="search.interval_dominance_margin",
    )
    if interval_dominance_margin < 0.0:
        raise ValueError("search.interval_dominance_margin must be non-negative")

    experiment_raw = _mapping(payload["experiment"], label="experiment")
    _expect_keys(
        experiment_raw,
        {
            "arm_id",
            "scoring_anchor_id",
            "proposal_source_ids",
            "auxiliary_proposal_source_ids",
        },
        label="experiment",
    )
    experiment_arm_id = _string(experiment_raw["arm_id"], label="experiment.arm_id")
    expected_proposal_sources = EXPERIMENT_ARM_PROPOSAL_SOURCES.get(experiment_arm_id)
    if expected_proposal_sources is None:
        raise ValueError(f"unsupported score-matrix experiment arm: {experiment_arm_id!r}")
    scoring_anchor_id = _string(
        experiment_raw["scoring_anchor_id"], label="experiment.scoring_anchor_id"
    )
    if scoring_anchor_id not in CANONICAL_SCORER_IDS:
        raise ValueError("experiment.scoring_anchor_id must be one canonical teacher")
    proposal_source_ids = _string_tuple(
        experiment_raw["proposal_source_ids"], label="experiment.proposal_source_ids"
    )
    if proposal_source_ids != expected_proposal_sources:
        raise ValueError(
            "experiment proposal sources do not exactly match the declared baseline arm"
        )
    auxiliary_proposal_source_ids = _string_tuple(
        experiment_raw["auxiliary_proposal_source_ids"],
        label="experiment.auxiliary_proposal_source_ids",
    )
    expected_auxiliary_sources = EXPERIMENT_ARM_AUXILIARY_SOURCES[experiment_arm_id]
    if auxiliary_proposal_source_ids != expected_auxiliary_sources:
        raise ValueError(
            "experiment auxiliary proposal sources do not exactly match the declared arm"
        )

    scorer_values = _sequence(payload["canonical_scorers"], label="canonical_scorers")
    scorers = tuple(
        _canonical_scorer_config(value, index=index) for index, value in enumerate(scorer_values)
    )
    scorer_ids = tuple(scorer.scorer_id for scorer in scorers)
    if scorer_ids != CANONICAL_SCORER_IDS:
        raise ValueError(
            "canonical_scorers must contain NAGISA, Suisho11Plus, and Soujou in canonical order"
        )
    if len({scorer.hash_mb for scorer in scorers}) != 1:
        raise ValueError("canonical scorers must configure one common numeric hash size")
    if len({scorer.proposal_multipv for scorer in scorers}) != 1:
        raise ValueError("canonical proposal processes must configure one common MultiPV K")
    _validate_rights_authorization(payload["limited_local_authorization"], scorer_ids)

    meteo = _proposal_config(payload["meteo_proposal"], label="meteo_proposal", single_move=False)
    actual_raw = payload["actual_played_move"]
    actual = (
        None
        if actual_raw is None
        else _proposal_config(actual_raw, label="actual_played_move", single_move=True)
    )
    auxiliary_proposals = tuple(
        _auxiliary_proposal_config(value, index=index)
        for index, value in enumerate(
            _sequence(payload["auxiliary_proposals"], label="auxiliary_proposals")
        )
    )
    if tuple(proposal.source_id for proposal in auxiliary_proposals) != (
        auxiliary_proposal_source_ids
    ):
        raise ValueError(
            "auxiliary_proposals must exactly match the experiment source IDs in order"
        )
    auxiliary_identities = tuple(
        proposal.proposal.producer_identity_sha256 for proposal in auxiliary_proposals
    )
    if len(set(auxiliary_identities)) != len(auxiliary_identities):
        raise ValueError("auxiliary proposal artifact identities must be unique")
    legal_moves = {move.to_usi() for move in history.target_board().legal_moves()}
    for proposal_label, proposal in (
        ("meteo_proposal", meteo),
        ("actual_played_move", actual),
        *(
            (f"auxiliary_proposals[{index}]", row.proposal)
            for index, row in enumerate(auxiliary_proposals)
        ),
    ):
        if proposal is None:
            continue
        outside = sorted(set(proposal.moves) - legal_moves)
        if outside:
            raise ValueError(f"{proposal_label} contains illegal moves: {outside!r}")
        if history.position_command() not in proposal.sent_commands:
            raise ValueError(
                f"{proposal_label} transcript omits the exact history position command"
            )
    return RunnerConfig(
        config_sha256=config_sha256,
        history=history,
        requested_nodes=requested_nodes,
        root_proposal_nodes=root_proposal_nodes,
        reply_proposal_nodes=reply_proposal_nodes,
        interval_dominance_margin=interval_dominance_margin,
        experiment_arm_id=experiment_arm_id,
        scoring_anchor_id=scoring_anchor_id,
        proposal_source_ids=proposal_source_ids,
        auxiliary_proposal_source_ids=auxiliary_proposal_source_ids,
        canonical_scorers=scorers,
        meteo_proposal=meteo,
        actual_played_move=actual,
        auxiliary_proposals=auxiliary_proposals,
    )


def _startup_sha256(engine: ExternalUsiTeacher) -> str:
    return _canonical_json_sha256(engine.startup_provenance.to_dict())


def _open_scorer(
    config: CanonicalScorerConfig,
    *,
    requested_nodes: int,
    locally_authorized_ids: frozenset[str],
) -> ExternalUsiProductionScorer:
    rights = model_rights(config.scorer_id)
    policy = rights.teacher_policy(allow_limited_local=config.scorer_id in locally_authorized_ids)
    scoring_teacher = ExternalUsiTeacher(
        list(config.command),
        policy,
        nodes=requested_nodes,
        multipv=1,
        options=dict(config.options),
        timeout_seconds=config.timeout_seconds,
        training_use=True,
        working_directory=config.working_directory,
        value_scale=config.value_scale,
        option_value_verification=REQUIRED_OPTION_VERIFICATION,
        expected_fatal_startup_diagnostics=config.expected_fatal_startup_diagnostics,
    )
    proposal_teacher = ExternalUsiTeacher(
        list(config.command),
        policy,
        nodes=requested_nodes,
        multipv=config.proposal_multipv,
        options=dict(config.options),
        timeout_seconds=config.timeout_seconds,
        training_use=True,
        working_directory=config.working_directory,
        value_scale=config.value_scale,
        option_value_verification=REQUIRED_OPTION_VERIFICATION,
        expected_fatal_startup_diagnostics=config.expected_fatal_startup_diagnostics,
    )
    try:
        scoring_teacher.start()
        proposal_teacher.start()
        for role, teacher in (
            ("scoring", scoring_teacher),
            ("proposal", proposal_teacher),
        ):
            provenance = teacher.startup_provenance
            if provenance.resolved_executable != config.command[0]:
                raise RuntimeError(
                    f"{role} USI startup resolved a different executable than the pinned config"
                )
            if provenance.arguments != config.command[1:]:
                raise RuntimeError(f"{role} USI startup arguments differ from the pinned config")
            if provenance.working_directory != str(config.working_directory):
                raise RuntimeError(
                    f"{role} USI startup working directory differs from the pinned config"
                )
    except BaseException:
        proposal_teacher.close()
        scoring_teacher.close()
        raise
    scoring_identity_options = tuple(
        sorted(
            ((*config.options, ("MultiPV", 1))),
            key=lambda item: item[0].casefold(),
        )
    )
    proposal_identity_options = tuple(
        sorted(
            ((*config.options, ("MultiPV", config.proposal_multipv))),
            key=lambda item: item[0].casefold(),
        )
    )
    scoring_identity = CanonicalScorerIdentity(
        scorer_id=config.scorer_id,
        engine_sha256=config.engine_sha256,
        evaluation_artifacts_sha256=config.evaluation_artifacts_sha256,
        options=tuple((name, str(value)) for name, value in scoring_identity_options),
        startup_transcript_sha256=_startup_sha256(scoring_teacher),
        parser_sha256=config.parser_sha256,
        scorer_code_sha256=config.scorer_code_sha256,
        calibration_sha256=config.calibration_sha256,
        threads=config.threads,
        hash_mb=config.hash_mb,
        multipv=1,
        book_enabled=False,
        hash_option_name=config.hash_option_name,
    )
    proposal_identity = CanonicalProposalIdentity(
        scorer_id=config.scorer_id,
        engine_sha256=config.engine_sha256,
        evaluation_artifacts_sha256=config.evaluation_artifacts_sha256,
        options=tuple((name, str(value)) for name, value in proposal_identity_options),
        startup_transcript_sha256=_startup_sha256(proposal_teacher),
        parser_sha256=config.parser_sha256,
        proposer_code_sha256=config.scorer_code_sha256,
        threads=config.threads,
        hash_mb=config.hash_mb,
        multipv=config.proposal_multipv,
        book_enabled=False,
        hash_option_name=config.hash_option_name,
    )
    try:
        return ExternalUsiProductionScorer(
            scoring_teacher,
            scoring_identity,
            proposal_engine=proposal_teacher,
            proposal_identity=proposal_identity,
        )
    except BaseException:
        proposal_teacher.close()
        scoring_teacher.close()
        raise


def _run_loaded_score_matrix(
    config: RunnerConfig, destination: Path
) -> ProductionScoreMatrixReceipt:
    locally_authorized_ids = frozenset(
        scorer.scorer_id
        for scorer in config.canonical_scorers
        if model_rights(scorer.scorer_id).distillation_scope
        is DistillationScope.LOCAL_AUTHORIZED_ONLY
    )
    proposals: list[CandidateProposalEvidence] = [
        _proposal_evidence(
            config.meteo_proposal,
            source_id=METEO_CANDIDATE_SOURCE_ID,
            source_kind=CandidateSourceKind.PROPOSAL_ONLY,
            history=config.history,
        )
    ]
    if config.actual_played_move is not None:
        proposals.append(
            _proposal_evidence(
                config.actual_played_move,
                source_id=ACTUAL_PLAYED_CANDIDATE_SOURCE_ID,
                source_kind=CandidateSourceKind.ACTUAL_PLAYED_MOVE,
                history=config.history,
            )
        )
    proposals.extend(
        _proposal_evidence(
            auxiliary.proposal,
            source_id=auxiliary.source_id,
            source_kind=CandidateSourceKind.PROPOSAL_ONLY,
            history=config.history,
        )
        for auxiliary in config.auxiliary_proposals
    )

    with ExitStack() as stack:
        scorers: list[ExternalUsiProductionScorer] = []
        required_adapter_ids = set(config.proposal_source_ids) | {config.scoring_anchor_id}
        for scorer_config in config.canonical_scorers:
            if scorer_config.scorer_id not in required_adapter_ids:
                continue
            scorer = _open_scorer(
                scorer_config,
                requested_nodes=config.requested_nodes,
                locally_authorized_ids=locally_authorized_ids,
            )
            stack.callback(scorer.close)
            scorers.append(scorer)
        legal_moves = tuple(
            sorted(move.to_usi() for move in config.history.target_board().legal_moves())
        )
        scorers_by_id = {scorer.identity.scorer_id: scorer for scorer in scorers}
        for source_id in config.proposal_source_ids:
            scorer = scorers_by_id[source_id]
            proposal = scorer.propose_replies(
                config.history,
                legal_moves=legal_moves,
                requested_nodes=config.root_proposal_nodes,
            )
            proposals.append(replace(proposal, source_kind=CandidateSourceKind.PROPOSAL_ONLY))
        builder = ProductionScoreMatrixBuilder(
            scorers,
            requested_nodes=config.requested_nodes,
            reply_proposal_nodes=config.reply_proposal_nodes,
            interval_dominance_margin=config.interval_dominance_margin,
            experiment_arm_id=config.experiment_arm_id,
            scoring_anchor_id=config.scoring_anchor_id,
            proposal_source_ids=config.proposal_source_ids,
            auxiliary_proposal_source_ids=config.auxiliary_proposal_source_ids,
            auxiliary_proposal_authorizations=tuple(
                auxiliary.authorization for auxiliary in config.auxiliary_proposals
            ),
        )
        receipt = builder.build(config.history, proposals)
        if receipt.qsearch_leaf_re_evaluated:
            raise AssertionError("runner v1 cannot claim qsearch-leaf re-evaluation")
        if receipt.cross_teacher_value_average or receipt.majority_vote:
            raise AssertionError("runner must never produce averaged or majority-vote truth")
        receipt.write_create_only(destination)
    return receipt


def run_score_matrix(config_path: Path, output_path: Path) -> ProductionScoreMatrixReceipt:
    """Run the strict external-USI matrix and atomically create its receipt."""

    destination = _create_only_destination(output_path)
    config = load_runner_config(config_path)
    return _run_loaded_score_matrix(config, destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simajilord-score-matrix",
        description="build one create-only, rights-gated canonical USI score matrix",
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_runner_config(args.config)
    destination = _create_only_destination(args.output)
    receipt = _run_loaded_score_matrix(config, destination)
    summary = {
        "schema": SCORE_MATRIX_RUNNER_SUMMARY_SCHEMA,
        "config_sha256": config.config_sha256,
        "output": str(destination),
        "receipt_sha256": receipt.receipt_sha256,
        "candidate_count": len(receipt.candidates),
        "canonical_scorer_ids": list(CANONICAL_SCORER_IDS),
        "experiment_arm_id": config.experiment_arm_id,
        "scoring_anchor_id": config.scoring_anchor_id,
        "proposal_source_ids": list(config.proposal_source_ids),
        "auxiliary_proposal_source_ids": list(config.auxiliary_proposal_source_ids),
        "output_scope": PRIVATE_OUTPUT_SCOPE,
        "local_only": True,
        "publication_allowed": False,
        "qsearch_leaf_re_evaluated": False,
        "cross_teacher_value_average": False,
        "majority_vote": False,
    }
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
