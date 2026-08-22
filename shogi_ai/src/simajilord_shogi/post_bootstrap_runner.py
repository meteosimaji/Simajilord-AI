"""Strict production glue for the post-100B Meteo NNUE pipeline.

The module intentionally joins only stages whose input receipts can be proved
locally.  A generated move is retained as a proposal source, never promoted to
a policy or value label.  Qsearch output becomes optimizer input only after one
pinned canonical scorer has re-evaluated every accepted leaf and the resulting
search receipts have been checked again by the replay builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from rsshogi.core import Board, Move

from .adjudication import adjudicate_board
from .artifact_provenance import sha256_file
from .distillation_targets import CANONICAL_SCORER_IDS
from .external_usi import ExternalUsiTeacher, UsiOptionValueVerification
from .incremental_mlx import (
    incremental_mlx_status,
    prepare_incremental_mlx_run,
    run_incremental_mlx,
)
from .incremental_value_replay import (
    ScalarLabelKind,
    build_incremental_value_replay,
    load_incremental_value_split,
    load_scalar_value_labels,
)
from .model_rights import model_rights
from .nnue_game_generation import NNUE_GAME_RECEIPT_SCHEMA
from .nnue_game_runner import (
    NNUE_GAME_BUNDLE_RECEIPT_SCHEMA,
    NNUE_GAME_JSONL_SCHEMA,
)
from .post_bootstrap import validate_registered_model_usage
from .production_score_matrix import (
    CanonicalScorerIdentity,
    ExternalUsiProductionScorer,
)
from .qsearch_leaf_rescore import (
    QSEARCH_LEAF_RESCORE_RECEIPT_SCHEMA,
    QSEARCH_LEAF_SEARCH_RECEIPT_SCHEMA,
    rescore_qsearch_leaves,
)
from .score_matrix_runner import (
    PRIVATE_OUTPUT_SCOPE,
    CanonicalScorerConfig,
    VerifiedArtifact,
    _artifact_set,
    _canonical_scorer_config,
    _validate_rights_authorization,
)

QSEARCH_RESCORE_RUNNER_CONFIG_SCHEMA = "meteo-qsearch-rescore-runner-config-v1"
QSEARCH_RESCORE_RUNNER_SUMMARY_SCHEMA = "meteo-qsearch-rescore-runner-summary-v1"
GAME_POSITION_SOURCE_SCHEMA = "meteo-game-position-source-v1"
GAME_POSITION_SPLIT_SCHEMA = "meteo-game-position-split-v1"
GAME_POSITION_BUNDLE_SCHEMA = "meteo-game-position-source-bundle-v1"

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_RESCORE_RECEIPT_FIELDS = {
    "schema",
    "qsearch_conversion",
    "anchor",
    "requested_nodes",
    "multipv",
    "position_context",
    "source_record_count",
    "unique_board_count",
    "duplicate_source_record_count",
    "labels",
    "search_receipts",
    "unresolved_queue",
    "old_score_used_as_target",
    "history_dependent_labels_emitted",
    "local_only",
    "publication_allowed",
    "complete",
}
_GAME_BUNDLE_RECEIPT_FIELDS = {
    "schema",
    "local_only",
    "publication_allowed",
    "strong_wdl_eligibility",
    "config",
    "engine_inputs",
    "games_jsonl",
    "generation",
    "payload_sha256",
    "payload_sha256_scope",
}
_TRAJECTORY_FIELDS = {
    "schema",
    "game_id",
    "opening_index",
    "color_swap_leg",
    "initial_sfen",
    "opening_prefix_moves",
    "starting_sfen",
    "opening_history_context_sha256",
    "black_actor_source",
    "white_actor_source",
    "generated_moves",
    "full_game_moves",
    "final_sfen",
    "plies",
    "winner",
    "termination",
    "rules_terminal_trajectory",
    "safety_truncated",
    "strong_wdl_label_allowed",
    "strong_wdl_label_black",
    "game_seconds",
}
_PLY_FIELDS = {
    "generated_ply",
    "absolute_ply",
    "turn",
    "actor_source",
    "rights_policy_id",
    "sfen",
    "move",
    "root_value",
    "nodes",
    "nps",
    "search_seconds",
    "history_prefix_length",
    "history_context_sha256",
    "resignation_overridden",
    "termination",
    "rules_terminal_trajectory",
    "strong_value_target",
    "history_prefix",
}


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value).rstrip(b"\n")).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    normalized: dict[str, str] = {}
    for key, value in pairs:
        folded = key.casefold()
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        if folded in normalized:
            raise ValueError(
                f"case-insensitive JSON object key collision: {normalized[folded]!r} and {key!r}"
            )
        normalized[folded] = key
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value: object = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_symlink_components(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink: {current}")


def _regular_file(path: Path, *, label: str) -> Path:
    source = _absolute(path)
    _reject_symlink_components(source, label=label)
    try:
        metadata = source.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise FileNotFoundError(source) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {source}")
    return source


def _regular_directory(path: Path, *, label: str) -> Path:
    source = _absolute(path)
    _reject_symlink_components(source, label=label)
    try:
        metadata = source.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise FileNotFoundError(source) from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink directory: {source}")
    return source


def _strict_json_file(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    source = _regular_file(path, label=label)
    payload = source.read_bytes()
    return _strict_json_bytes(payload, label=label), payload


def _expect_fields(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} fields mismatch: missing={sorted(expected - set(value))!r} "
            f"extra={sorted(set(value) - expected)!r}"
        )


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _sequence(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _string(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError(f"{label} must be a non-empty trimmed single-line string")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _sha256(value: object, *, label: str) -> str:
    digest = _string(value, label=label)
    if len(digest) != 64 or any(character not in _SHA256_CHARACTERS for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _local_basename(value: object, *, label: str) -> str:
    name = _string(value, label=label)
    if Path(name).is_absolute() or Path(name).name != name:
        raise ValueError(f"{label} must be one local basename")
    return name


def _line_count(path: Path, *, label: str) -> int:
    count = 0
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{label} contains blank row {line_number}")
            count += 1
    return count


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_directory(stage: Path, destination: Path, filenames: Sequence[str]) -> None:
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite output: {destination}") from error
    try:
        for filename in filenames:
            os.link(stage / filename, destination / filename)
        descriptor = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    shutil.rmtree(stage)


@dataclass(frozen=True, slots=True)
class QsearchAnchorRunnerConfig:
    source: Path
    source_sha256: str
    requested_nodes: int
    scorer: CanonicalScorerConfig
    locally_authorized_ids: frozenset[str]
    pinned_artifacts: tuple[VerifiedArtifact, ...]

    def revalidate(self) -> None:
        for artifact in self.pinned_artifacts:
            if sha256_file(_regular_file(artifact.path, label="pinned scorer artifact")) != (
                artifact.sha256
            ):
                raise RuntimeError(f"pinned scorer artifact changed: {artifact.path}")


def load_qsearch_anchor_config(path: Path) -> QsearchAnchorRunnerConfig:
    """Load one immutable, rights-gated, single-scorer rescore contract."""

    source = _regular_file(path, label="qsearch rescore config")
    payload_bytes = source.read_bytes()
    payload = _strict_json_bytes(payload_bytes, label="qsearch rescore config")
    _expect_fields(
        payload,
        {
            "schema",
            "output_scope",
            "limited_local_authorization",
            "requested_nodes",
            "anchor",
        },
        label="qsearch rescore config",
    )
    if payload["schema"] != QSEARCH_RESCORE_RUNNER_CONFIG_SCHEMA:
        raise ValueError("unsupported qsearch rescore runner config schema")
    if payload["output_scope"] != PRIVATE_OUTPUT_SCOPE:
        raise PermissionError("canonical qsearch rescore output must remain private/local")
    requested_nodes = _positive_integer(payload["requested_nodes"], label="requested_nodes")
    anchor_raw = _mapping(payload["anchor"], label="anchor")
    scorer = _canonical_scorer_config(anchor_raw, index=0)
    if scorer.scorer_id not in CANONICAL_SCORER_IDS:
        raise ValueError("qsearch anchor must be one canonical scorer")
    _validate_rights_authorization(payload["limited_local_authorization"], (scorer.scorer_id,))
    authorization = _mapping(
        payload["limited_local_authorization"], label="limited_local_authorization"
    )
    authorized_ids = frozenset(
        _string(item, label="limited_local_authorization.rights_ids[]")
        for item in _sequence(
            authorization["rights_ids"], label="limited_local_authorization.rights_ids"
        )
    )
    engine = _mapping(anchor_raw["engine"], label="anchor.engine")
    engine_artifact = VerifiedArtifact(
        path=_regular_file(
            Path(_string(engine["path"], label="anchor.engine.path")),
            label="anchor engine",
        ),
        sha256=_sha256(engine["sha256"], label="anchor.engine.sha256"),
    )
    evaluation, _aggregate = _artifact_set(
        anchor_raw["evaluation_artifacts"],
        anchor_raw["evaluation_artifacts_sha256"],
        label="anchor.evaluation_artifacts",
    )
    singleton_artifacts = tuple(
        VerifiedArtifact(
            path=_regular_file(
                Path(
                    _string(
                        _mapping(anchor_raw[field], label=field)["path"],
                        label=f"{field}.path",
                    )
                ),
                label=field,
            ),
            sha256=_sha256(
                _mapping(anchor_raw[field], label=field)["sha256"], label=f"{field}.sha256"
            ),
        )
        for field in ("parser_artifact", "scorer_code_artifact", "calibration_artifact")
    )
    pins_by_path: dict[str, VerifiedArtifact] = {}
    for artifact in (engine_artifact, *evaluation, *singleton_artifacts):
        existing = pins_by_path.get(str(artifact.path))
        if existing is not None and existing.sha256 != artifact.sha256:
            raise ValueError("qsearch scorer pin set gives one path conflicting hashes")
        pins_by_path[str(artifact.path)] = artifact
    pinned = tuple(pins_by_path[path] for path in sorted(pins_by_path))
    config = QsearchAnchorRunnerConfig(
        source=source,
        source_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        requested_nodes=requested_nodes,
        scorer=scorer,
        locally_authorized_ids=authorized_ids,
        pinned_artifacts=pinned,
    )
    config.revalidate()
    return config


def run_qsearch_rescore(
    config_path: Path,
    qsearch_directory: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Run one pinned anchor scorer and create a local-only rescore bundle."""

    config = load_qsearch_anchor_config(config_path)
    config.revalidate()
    scorer_config = config.scorer
    rights = model_rights(scorer_config.scorer_id)
    policy = rights.teacher_policy(
        allow_limited_local=scorer_config.scorer_id in config.locally_authorized_ids
    )
    engine = ExternalUsiTeacher(
        list(scorer_config.command),
        policy,
        nodes=config.requested_nodes,
        multipv=1,
        options=dict(scorer_config.options),
        timeout_seconds=scorer_config.timeout_seconds,
        training_use=True,
        working_directory=scorer_config.working_directory,
        value_scale=scorer_config.value_scale,
        option_value_verification=UsiOptionValueVerification.YANEURAOU_GETOPTION,
        expected_fatal_startup_diagnostics=scorer_config.expected_fatal_startup_diagnostics,
    )
    try:
        engine.start()
        provenance = engine.startup_provenance
        if (
            provenance.resolved_executable != scorer_config.command[0]
            or provenance.arguments != scorer_config.command[1:]
            or provenance.working_directory != str(scorer_config.working_directory)
        ):
            raise RuntimeError("qsearch anchor startup differs from the pinned config")
        identity_options = tuple(
            sorted(
                ((*scorer_config.options, ("MultiPV", 1))),
                key=lambda item: item[0].casefold(),
            )
        )
        identity = CanonicalScorerIdentity(
            scorer_id=scorer_config.scorer_id,
            engine_sha256=scorer_config.engine_sha256,
            evaluation_artifacts_sha256=scorer_config.evaluation_artifacts_sha256,
            options=tuple((name, str(value)) for name, value in identity_options),
            startup_transcript_sha256=_canonical_json_sha256(provenance.to_dict()),
            parser_sha256=scorer_config.parser_sha256,
            scorer_code_sha256=scorer_config.scorer_code_sha256,
            calibration_sha256=scorer_config.calibration_sha256,
            threads=scorer_config.threads,
            hash_mb=scorer_config.hash_mb,
            multipv=1,
            book_enabled=False,
            hash_option_name=scorer_config.hash_option_name,
        )
        scorer = ExternalUsiProductionScorer(engine, identity)
    except BaseException:
        engine.close()
        raise
    try:
        receipt = rescore_qsearch_leaves(
            qsearch_directory,
            anchor=scorer,
            requested_nodes=config.requested_nodes,
            output_directory=output_directory,
        )
    finally:
        engine.close()
    config.revalidate()
    return {
        "schema": QSEARCH_RESCORE_RUNNER_SUMMARY_SCHEMA,
        "config_sha256": config.source_sha256,
        "scorer_id": config.scorer.scorer_id,
        "requested_nodes": config.requested_nodes,
        "output": str(_absolute(output_directory)),
        "receipt_sha256": sha256_file(_absolute(output_directory) / "receipt.json"),
        "labels": cast(dict[str, object], receipt["labels"])["rows"],
        "unresolved": cast(dict[str, object], receipt["unresolved_queue"])["rows"],
        "local_only": True,
        "publication_allowed": False,
    }


def _verified_receipt_artifact(
    root: Path,
    value: object,
    *,
    label: str,
    expected_keys: set[str],
) -> tuple[Path, dict[str, Any]]:
    identity = _mapping(value, label=label)
    _expect_fields(identity, expected_keys, label=label)
    filename = _local_basename(identity["file"], label=f"{label}.file")
    artifact = _regular_file(root / filename, label=label)
    if sha256_file(artifact) != _sha256(identity["sha256"], label=f"{label}.sha256"):
        raise ValueError(f"{label} SHA-256 mismatch")
    return artifact, identity


def _verify_rescore_bundle(
    directory: Path,
) -> tuple[Path, tuple[Any, ...], dict[str, str]]:
    root = _regular_directory(directory, label="qsearch rescore bundle")
    receipt_path = _regular_file(root / "receipt.json", label="qsearch rescore receipt")
    receipt, _payload = _strict_json_file(receipt_path, label="qsearch rescore receipt")
    _expect_fields(receipt, _RESCORE_RECEIPT_FIELDS, label="qsearch rescore receipt")
    if (
        receipt["schema"] != QSEARCH_LEAF_RESCORE_RECEIPT_SCHEMA
        or receipt["complete"] is not True
        or receipt["local_only"] is not True
        or receipt["publication_allowed"] is not False
        or receipt["multipv"] != 1
        or receipt["old_score_used_as_target"] is not False
        or receipt["history_dependent_labels_emitted"] is not False
    ):
        raise ValueError("qsearch rescore receipt is not a complete local-only exact contract")
    labels_path, labels_identity = _verified_receipt_artifact(
        root,
        receipt["labels"],
        label="rescore labels",
        expected_keys={"file", "sha256", "rows", "qsearch_leaf_rescored", "exact_centipawn_only"},
    )
    searches_path, searches_identity = _verified_receipt_artifact(
        root,
        receipt["search_receipts"],
        label="rescore searches",
        expected_keys={"file", "sha256", "rows"},
    )
    unresolved_path, unresolved_identity = _verified_receipt_artifact(
        root,
        receipt["unresolved_queue"],
        label="rescore unresolved",
        expected_keys={
            "file",
            "sha256",
            "rows",
            "reason_counts",
            "mate_bound_or_short_budget_never_written_as_point_label",
        },
    )
    if (
        labels_identity["qsearch_leaf_rescored"] is not True
        or labels_identity["exact_centipawn_only"] is not True
        or unresolved_identity["mate_bound_or_short_budget_never_written_as_point_label"]
        is not True
    ):
        raise ValueError("qsearch rescore bundle does not prove exact-label filtering")
    for path, identity, artifact_label in (
        (labels_path, labels_identity, "rescore labels"),
        (searches_path, searches_identity, "rescore searches"),
        (unresolved_path, unresolved_identity, "rescore unresolved"),
    ):
        expected_rows = identity["rows"]
        if isinstance(expected_rows, bool) or not isinstance(expected_rows, int):
            raise ValueError(f"{artifact_label}.rows must be an integer")
        if _line_count(path, label=artifact_label) != expected_rows:
            raise ValueError(f"{artifact_label} row count mismatch")
    labels = load_scalar_value_labels(labels_path)
    if len(labels) != labels_identity["rows"]:
        raise ValueError("rescore label parser count mismatch")

    search_by_sha: dict[str, dict[str, Any]] = {}
    with searches_path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = _strict_json_bytes(line, label=f"rescore search row {line_number}")
            if row.get("schema") != QSEARCH_LEAF_SEARCH_RECEIPT_SCHEMA:
                raise ValueError("rescore search row schema mismatch")
            digest = _sha256(row.get("search_receipt_sha256"), label="search_receipt_sha256")
            unsigned = dict(row)
            unsigned.pop("search_receipt_sha256")
            if _canonical_json_sha256(unsigned) != digest:
                raise ValueError("rescore search row self-hash mismatch")
            if digest in search_by_sha:
                raise ValueError("duplicate rescore search receipt SHA-256")
            search_by_sha[digest] = row
    exact_searches = {
        digest: row
        for digest, row in search_by_sha.items()
        if row.get("resolution") == "exact_cp_label"
    }
    label_source_hashes: set[str] = set()
    for label in labels:
        if label.kind is not ScalarLabelKind.ANCHOR_SEARCH_EXACT:
            raise ValueError("qsearch rescore bundle contains a non-anchor scalar label")
        search = exact_searches.get(label.source_receipt_sha256)
        if search is None:
            raise ValueError("scalar label source_receipt_sha256 has no exact search receipt")
        variation = _mapping(search.get("variation"), label="exact search variation")
        if (
            search.get("sfen") != label.sfen
            or search.get("scorer_id") != label.scorer_id
            or search.get("requested_nodes") != label.requested_nodes
            or variation.get("score_kind") != "cp"
            or variation.get("bound") != "exact"
            or variation.get("score_cp") != label.score_cp
            or search.get("unresolved_reasons") not in ([], ())
        ):
            raise ValueError("scalar label does not exactly match its search receipt")
        label_source_hashes.add(label.source_receipt_sha256)
    if label_source_hashes != set(exact_searches):
        raise ValueError("exact search receipts and scalar labels are not one-to-one")
    if len(search_by_sha) != receipt["unique_board_count"]:
        raise ValueError("search receipt count does not equal unique-board count")
    if len(exact_searches) != labels_identity["rows"]:
        raise ValueError("exact search count does not equal label count")
    if len(search_by_sha) - len(exact_searches) != unresolved_identity["rows"]:
        raise ValueError("unresolved search count does not match receipt")
    identities = {
        "receipt": sha256_file(receipt_path),
        "labels": sha256_file(labels_path),
        "searches": sha256_file(searches_path),
        "unresolved": sha256_file(unresolved_path),
    }
    return labels_path, labels, identities


def build_incremental_replay_from_rescore(
    rescore_directory: Path,
    split_receipt: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Build train PSV only from a fully verified qsearch-rescore bundle."""

    labels_path, labels, identities = _verify_rescore_bundle(rescore_directory)
    split_path = _regular_file(split_receipt, label="incremental train split receipt")
    split_sha = sha256_file(split_path)
    split = load_incremental_value_split(split_path)
    if split.split_id != "train":
        raise ValueError("qsearch rescore replay requires the pre-labelled train split")
    if any(label.position_id not in set(split.position_ids) for label in labels):
        raise ValueError("rescore label is absent from the pre-labelled train split")
    # Close the load/build TOCTOU window for every external byte stream.
    refreshed_labels, refreshed, refreshed_identities = _verify_rescore_bundle(rescore_directory)
    if refreshed_labels != labels_path or refreshed_identities != identities or refreshed != labels:
        raise RuntimeError("qsearch rescore bundle changed during replay validation")
    if sha256_file(split_path) != split_sha or load_incremental_value_split(split_path) != split:
        raise RuntimeError("incremental split receipt changed during replay validation")
    receipt = build_incremental_value_replay(
        labels,
        split=split,
        output_directory=output_directory,
    )
    return {
        "schema": "meteo-rescore-to-incremental-replay-summary-v1",
        "source_rescore_receipt_sha256": identities["receipt"],
        "source_split_receipt_sha256": split_sha,
        "output": str(_absolute(output_directory)),
        "records": receipt["records"],
        "policy_targets": 0,
        "local_only": True,
        "publication_allowed": False,
    }


class _DisjointSet:
    def __init__(self, values: Sequence[str]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self._parent[high] = low


def _history_sha256(initial_sfen: str, moves: Sequence[str]) -> str:
    return hashlib.sha256(
        initial_sfen.encode("utf-8") + b"\0" + " ".join(moves).encode("ascii")
    ).hexdigest()


def _position_id(sfen: str) -> str:
    board = Board(sfen)
    fields = board.to_sfen().split()
    return hashlib.sha256(" ".join(fields[:3]).encode("utf-8")).hexdigest()


def _verify_bundle_receipt(root: Path) -> tuple[Path, str, dict[str, Any]]:
    receipt_path = _regular_file(root / "receipt.json", label="game bundle receipt")
    receipt, receipt_bytes = _strict_json_file(receipt_path, label="game bundle receipt")
    _expect_fields(receipt, _GAME_BUNDLE_RECEIPT_FIELDS, label="game bundle receipt")
    if (
        receipt.get("schema") != NNUE_GAME_BUNDLE_RECEIPT_SCHEMA
        or receipt.get("local_only") is not True
        or receipt.get("publication_allowed") is not False
        or receipt.get("payload_sha256_scope") != "receipt_without_payload_sha256_fields"
    ):
        raise ValueError("game bundle receipt is not a complete private/local receipt")
    expected_payload_sha = _sha256(receipt.get("payload_sha256"), label="payload_sha256")
    unsigned = dict(receipt)
    unsigned.pop("payload_sha256")
    unsigned.pop("payload_sha256_scope")
    if _canonical_json_sha256(unsigned) != expected_payload_sha:
        raise ValueError("game bundle receipt self-hash mismatch")
    games_identity = _mapping(receipt.get("games_jsonl"), label="games_jsonl")
    if games_identity.get("schema") != NNUE_GAME_JSONL_SCHEMA:
        raise ValueError("game bundle games schema mismatch")
    filename = _local_basename(games_identity.get("path"), label="games_jsonl.path")
    games_path = _regular_file(root / filename, label="game bundle games JSONL")
    if (
        games_identity.get("bytes") != games_path.stat().st_size
        or games_identity.get("sha256") != sha256_file(games_path)
        or games_identity.get("contains_complete_trajectory_receipts") is not True
    ):
        raise ValueError("game bundle games JSONL identity mismatch")
    return games_path, hashlib.sha256(receipt_bytes).hexdigest(), receipt


def _verified_game_positions(
    games_path: Path,
    *,
    expected_games: int,
    source_receipt_sha256: str,
    source_games_sha256: str,
) -> tuple[list[dict[str, object]], dict[str, set[str]]]:
    positions: list[dict[str, object]] = []
    game_positions: dict[str, set[str]] = {}
    game_ids: set[str] = set()
    with games_path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = _strict_json_bytes(line, label=f"game JSONL row {line_number}")
            _expect_fields(row, {"schema", "game", "trajectory_receipt"}, label="game row")
            if row["schema"] != NNUE_GAME_JSONL_SCHEMA:
                raise ValueError("game JSONL row schema mismatch")
            game = _mapping(row["game"], label="game")
            trajectory = _mapping(row["trajectory_receipt"], label="trajectory_receipt")
            _expect_fields(
                game,
                {"initial_sfen", "moves", "samples", "winner", "termination"},
                label="game",
            )
            _expect_fields(trajectory, _TRAJECTORY_FIELDS, label="trajectory_receipt")
            if trajectory.get("schema") != NNUE_GAME_RECEIPT_SCHEMA:
                raise ValueError("trajectory receipt schema mismatch")
            game_id = _sha256(trajectory.get("game_id"), label="trajectory game_id")
            if game_id in game_ids:
                raise ValueError("duplicate game_id in generated-game bundle")
            game_ids.add(game_id)
            initial_sfen = _string(game.get("initial_sfen"), label="game.initial_sfen")
            full_moves = tuple(
                _string(move, label="game.moves[]")
                for move in _sequence(game.get("moves"), label="game.moves")
            )
            trajectory_moves = tuple(
                _string(move, label="trajectory.full_game_moves[]")
                for move in _sequence(
                    trajectory.get("full_game_moves"), label="trajectory.full_game_moves"
                )
            )
            if (
                trajectory.get("initial_sfen") != initial_sfen
                or trajectory_moves != full_moves
                or trajectory.get("winner") != game.get("winner")
                or trajectory.get("termination") != game.get("termination")
            ):
                raise ValueError("game record and trajectory receipt disagree")
            board = Board(initial_sfen)
            canonical_initial = board.to_sfen()
            plies = _sequence(trajectory.get("plies"), label="trajectory.plies")
            samples = _sequence(game.get("samples"), label="game.samples")
            opening_moves = _sequence(
                trajectory.get("opening_prefix_moves"), label="trajectory.opening_prefix_moves"
            )
            generated_moves = _sequence(
                trajectory.get("generated_moves"), label="trajectory.generated_moves"
            )
            if tuple(opening_moves) + tuple(generated_moves) != full_moves:
                raise ValueError("opening/generated trajectory does not reconstruct full game")
            if len(plies) != len(generated_moves):
                raise ValueError("trajectory ply count does not match generated move count")
            if len(samples) != len(plies):
                raise ValueError("game sample count does not match trajectory ply count")
            generated_start = len(opening_moves)
            opening_board = Board(canonical_initial)
            for opening_ply, opening_move in enumerate(opening_moves):
                parsed_opening_move = Move.from_usi(_string(opening_move, label="opening move"))
                if not opening_board.is_legal_move(parsed_opening_move):
                    raise ValueError(f"opening prefix contains illegal move at ply {opening_ply}")
                opening_board.apply_move(parsed_opening_move)
            if trajectory.get("starting_sfen") != opening_board.to_sfen() or trajectory.get(
                "opening_history_context_sha256"
            ) != _history_sha256(canonical_initial, cast(list[str], opening_moves)):
                raise ValueError("trajectory opening prefix evidence mismatch")
            per_game_ids: set[str] = set()
            for absolute_ply, move_usi in enumerate(full_moves):
                move = Move.from_usi(move_usi)
                if not board.is_legal_move(move):
                    raise ValueError(f"game contains illegal move at ply {absolute_ply}")
                if absolute_ply >= generated_start:
                    ply = _mapping(plies[absolute_ply - generated_start], label="trajectory ply")
                    _expect_fields(ply, _PLY_FIELDS, label="trajectory ply")
                    sample = _mapping(samples[absolute_ply - generated_start], label="game sample")
                    history = _mapping(ply.get("history_prefix"), label="history_prefix")
                    prefix = full_moves[:absolute_ply]
                    if (
                        ply.get("absolute_ply") != absolute_ply
                        or ply.get("sfen") != board.to_sfen()
                        or ply.get("move") != move_usi
                        or history.get("source") != "game.full_game_moves"
                        or history.get("length") != absolute_ply
                        or history.get("sha256") != _history_sha256(canonical_initial, prefix)
                        or ply.get("generated_ply") != absolute_ply - generated_start
                        or ply.get("history_prefix_length") != absolute_ply
                        or ply.get("history_context_sha256")
                        != _history_sha256(canonical_initial, prefix)
                        or sample.get("sfen") != board.to_sfen()
                        or sample.get("ply") != absolute_ply
                        or sample.get("turn") != ply.get("turn")
                        or sample.get("chosen_move") != move_usi
                        or sample.get("actor_source") != ply.get("actor_source")
                    ):
                        raise ValueError("trajectory ply history or move evidence mismatch")
                    position_id = _position_id(board.to_sfen())
                    per_game_ids.add(position_id)
                    positions.append(
                        {
                            "schema": GAME_POSITION_SOURCE_SCHEMA,
                            "source_bundle_receipt_sha256": source_receipt_sha256,
                            "source_games_sha256": source_games_sha256,
                            "game_id": game_id,
                            "position_id": position_id,
                            "initial_sfen": canonical_initial,
                            "history_moves": list(prefix),
                            "full_game_moves": list(full_moves),
                            "history_context_sha256": _history_sha256(canonical_initial, prefix),
                            "sfen": board.to_sfen(),
                            "absolute_ply": absolute_ply,
                            "actor_source": ply.get("actor_source"),
                            "played_move": move_usi,
                            "played_move_role": "proposal_only_not_label",
                            "label_generated": False,
                            "termination": trajectory.get("termination"),
                            "rules_terminal_trajectory": trajectory.get(
                                "rules_terminal_trajectory"
                            ),
                            "strong_wdl_label_allowed": trajectory.get("strong_wdl_label_allowed"),
                        }
                    )
                board.apply_move(move)
            if trajectory.get("final_sfen") != board.to_sfen():
                raise ValueError("trajectory final SFEN does not match replay")
            adjudication = adjudicate_board(board)
            rules_terminal = trajectory.get("rules_terminal_trajectory")
            if rules_terminal is True:
                if adjudication is None:
                    raise ValueError("rules-terminal trajectory is not terminal on replay")
                if (
                    adjudication.termination.value != trajectory.get("termination")
                    or adjudication.winner != trajectory.get("winner")
                    or trajectory.get("strong_wdl_label_allowed") is not True
                ):
                    raise ValueError("rules-terminal adjudication disagrees with receipt")
            elif (
                rules_terminal is not False
                or adjudication is not None
                or trajectory.get("termination") != "max_plies"
                or trajectory.get("strong_wdl_label_allowed") is not False
            ):
                raise ValueError("non-terminal trajectory has an invalid WDL gate")
            game_positions[game_id] = per_game_ids
    if len(game_ids) != expected_games:
        raise ValueError("game JSONL count does not match its bundle receipt")
    if not positions:
        raise ValueError("game bundle contains no generated position source")
    return positions, game_positions


def _component_splits(game_positions: Mapping[str, set[str]]) -> dict[str, str]:
    game_ids = sorted(game_positions)
    disjoint = _DisjointSet(game_ids)
    position_owner: dict[str, str] = {}
    for game_id in game_ids:
        for position_id in sorted(game_positions[game_id]):
            owner = position_owner.setdefault(position_id, game_id)
            disjoint.union(owner, game_id)
    components: dict[str, list[str]] = defaultdict(list)
    for game_id in game_ids:
        components[disjoint.find(game_id)].append(game_id)
    assignments: dict[str, str] = {}
    for members in components.values():
        component_identity = hashlib.sha256("\0".join(sorted(members)).encode()).hexdigest()
        bucket = int(component_identity[:16], 16) % 100
        split = "train" if bucket < 90 else "calibration" if bucket < 95 else "held_out_test"
        for game_id in members:
            assignments[game_id] = split
    return assignments


def extract_game_position_source(
    game_bundle_directory: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Extract a deterministic, no-label position source from full game trajectories."""

    root = _regular_directory(game_bundle_directory, label="game bundle")
    games_path, receipt_sha, receipt = _verify_bundle_receipt(root)
    games_identity = _mapping(receipt["games_jsonl"], label="games_jsonl")
    games_sha = _sha256(games_identity["sha256"], label="games_jsonl.sha256")
    expected_games = _positive_integer(games_identity["games"], label="games_jsonl.games")
    positions, game_positions = _verified_game_positions(
        games_path,
        expected_games=expected_games,
        source_receipt_sha256=receipt_sha,
        source_games_sha256=games_sha,
    )
    assignments = _component_splits(game_positions)
    for row in positions:
        row["split_id"] = assignments[cast(str, row["game_id"])]
    positions.sort(
        key=lambda row: (
            cast(str, row["split_id"]),
            cast(str, row["game_id"]),
            cast(int, row["absolute_ply"]),
        )
    )
    split_payload: dict[str, object] = {
        "schema": GAME_POSITION_SPLIT_SCHEMA,
        "source_games_sha256": games_sha,
        "assignment_unit": "connected_game_component_by_normalized_board_position",
        "assignment_algorithm": "sha256(component_game_ids)%100;train<90;calibration<95;else_test",
        "label_generation_happens_after_split": True,
        "played_moves_are_proposal_only": True,
        "splits": {},
    }
    split_rows = cast(dict[str, object], split_payload["splits"])
    for split_id in ("train", "calibration", "held_out_test"):
        games = sorted(game for game, assigned in assignments.items() if assigned == split_id)
        position_ids = sorted(
            {cast(str, row["position_id"]) for row in positions if row["split_id"] == split_id}
        )
        split_rows[split_id] = {
            "games": games,
            "positions": position_ids,
            "game_count": len(games),
            "position_count": len(position_ids),
        }
    position_sets = [
        set(cast(dict[str, Any], split_rows[split_id])["positions"])
        for split_id in ("train", "calibration", "held_out_test")
    ]
    if any(
        position_sets[left] & position_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise AssertionError("component split leaked a normalized board position")

    destination = _absolute(output_directory)
    _reject_symlink_components(destination.parent, label="position-source output parent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = _regular_directory(destination.parent, label="position-source output parent")
    destination = parent / destination.name
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite position-source bundle: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
    try:
        positions_bytes = b"".join(_canonical_json_bytes(row) for row in positions)
        split_bytes = _canonical_json_bytes(split_payload)
        _write_new(stage / "positions.jsonl", positions_bytes)
        _write_new(stage / "split-receipt.json", split_bytes)
        output_receipt = {
            "schema": GAME_POSITION_BUNDLE_SCHEMA,
            "source": {
                "bundle_receipt_sha256": receipt_sha,
                "games_sha256": games_sha,
                "games": expected_games,
            },
            "positions": {
                "file": "positions.jsonl",
                "sha256": hashlib.sha256(positions_bytes).hexdigest(),
                "rows": len(positions),
            },
            "split": {
                "file": "split-receipt.json",
                "sha256": hashlib.sha256(split_bytes).hexdigest(),
            },
            "initial_sfen_and_full_history_preserved": True,
            "played_moves_are_proposal_only": True,
            "labels_emitted": 0,
            "local_only": True,
            "publication_allowed": False,
            "complete": True,
        }
        _write_new(stage / "receipt.json", _canonical_json_bytes(output_receipt))
        # Recheck immutable source identities immediately before publication.
        if (
            sha256_file(games_path) != games_sha
            or sha256_file(root / "receipt.json") != receipt_sha
        ):
            raise RuntimeError("game bundle changed during position extraction")
        _publish_directory(
            stage,
            destination,
            ("positions.jsonl", "split-receipt.json", "receipt.json"),
        )
        return output_receipt
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


def validate_registered_model_usage_receipts(
    bundle_receipt_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, object]:
    """Validate all nine registered-model game minima and publish one gate JSON."""

    if not bundle_receipt_paths:
        raise ValueError("registered-model usage gate requires game bundle receipts")
    sources = tuple(
        _regular_file(path, label=f"game bundle receipt {index}")
        for index, path in enumerate(bundle_receipt_paths)
    )
    rendered = tuple(str(path) for path in sources)
    if rendered != tuple(sorted(set(rendered))):
        raise ValueError("game bundle receipt paths must be unique and sorted")
    receipts: list[dict[str, Any]] = []
    file_hashes: list[str] = []
    for index, source in enumerate(sources):
        receipt, payload = _strict_json_file(source, label=f"game bundle receipt {index}")
        receipts.append(receipt)
        file_hashes.append(hashlib.sha256(payload).hexdigest())
    gate = validate_registered_model_usage(receipts)
    if len(set(gate.source_bundle_sha256)) != len(receipts):
        raise ValueError("model-usage gate source bundle payload hashes must be unique")
    for source, expected in zip(sources, file_hashes, strict=True):
        if sha256_file(source) != expected:
            raise RuntimeError("game bundle receipt changed during usage-gate validation")
    destination = _absolute(output_path)
    _reject_symlink_components(destination.parent, label="model usage gate output parent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = _regular_directory(destination.parent, label="model usage gate output parent")
    destination = parent / destination.name
    try:
        _write_new(destination, _canonical_json_bytes(gate.to_dict()))
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite model usage gate: {destination}") from error
    parent_descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return gate.to_dict()


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simajilord-post-bootstrap",
        description="strict create-only post-100B NNUE stage runner",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rescore = subparsers.add_parser("rescore-qsearch")
    rescore.add_argument("config", type=Path)
    rescore.add_argument("qsearch_directory", type=Path)
    rescore.add_argument("output_directory", type=Path)

    replay = subparsers.add_parser("build-incremental-replay")
    replay.add_argument("rescore_directory", type=Path)
    replay.add_argument("split_receipt", type=Path)
    replay.add_argument("output_directory", type=Path)

    extract = subparsers.add_parser("extract-game-positions")
    extract.add_argument("game_bundle_directory", type=Path)
    extract.add_argument("output_directory", type=Path)

    usage = subparsers.add_parser("validate-model-usage")
    usage.add_argument("output", type=Path)
    usage.add_argument("bundle_receipts", type=Path, nargs="+")

    prepare = subparsers.add_parser("prepare-incremental-mlx")
    prepare.add_argument("output_directory", type=Path)
    prepare.add_argument("--base-run-directory", type=Path, required=True)
    prepare.add_argument("--base-checkpoint", type=Path, required=True)
    prepare.add_argument("--broad-anchor-psv", type=Path, required=True)
    prepare.add_argument("--broad-anchor-receipt", type=Path, required=True)
    prepare.add_argument("--broad-anchor-split-receipt", type=Path, required=True)
    prepare.add_argument("--hard-exact-psv", type=Path, required=True)
    prepare.add_argument("--hard-exact-receipt", type=Path, required=True)
    prepare.add_argument("--hard-exact-split-receipt", type=Path, required=True)
    prepare.add_argument("--calibration-psv", type=Path, required=True)
    prepare.add_argument("--calibration-receipt", type=Path, required=True)
    prepare.add_argument("--calibration-split-receipt", type=Path, required=True)
    prepare.add_argument("--additional-optimizer-steps", type=int, required=True)
    prepare.add_argument("--learning-rate", type=float, required=True)
    prepare.add_argument("--broad-weight", type=int, default=3)
    prepare.add_argument("--hard-weight", type=int, default=1)
    prepare.add_argument("--checkpoint-batches", type=int, default=128)
    prepare.add_argument("--log-batches", type=int, default=16)
    prepare.add_argument("--calibration-validation-positions", type=int, default=262_144)
    prepare.add_argument("--quantisation-audit-positions", type=int, default=8_192)

    run = subparsers.add_parser("run-incremental-mlx")
    run.add_argument("run_directory", type=Path)
    status = subparsers.add_parser("incremental-status")
    status.add_argument("run_directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "rescore-qsearch":
        result = run_qsearch_rescore(args.config, args.qsearch_directory, args.output_directory)
    elif args.command == "build-incremental-replay":
        result = build_incremental_replay_from_rescore(
            args.rescore_directory, args.split_receipt, args.output_directory
        )
    elif args.command == "extract-game-positions":
        result = extract_game_position_source(args.game_bundle_directory, args.output_directory)
    elif args.command == "validate-model-usage":
        result = validate_registered_model_usage_receipts(args.bundle_receipts, args.output)
    elif args.command == "prepare-incremental-mlx":
        result = prepare_incremental_mlx_run(
            args.output_directory,
            base_run_directory=args.base_run_directory,
            base_checkpoint=args.base_checkpoint,
            broad_anchor_psv=args.broad_anchor_psv,
            broad_anchor_receipt=args.broad_anchor_receipt,
            broad_anchor_split_receipt=args.broad_anchor_split_receipt,
            hard_exact_psv=args.hard_exact_psv,
            hard_exact_receipt=args.hard_exact_receipt,
            hard_exact_split_receipt=args.hard_exact_split_receipt,
            calibration_psv=args.calibration_psv,
            calibration_receipt=args.calibration_receipt,
            calibration_split_receipt=args.calibration_split_receipt,
            additional_optimizer_steps=args.additional_optimizer_steps,
            learning_rate=args.learning_rate,
            broad_weight=args.broad_weight,
            hard_weight=args.hard_weight,
            checkpoint_batches=args.checkpoint_batches,
            log_batches=args.log_batches,
            calibration_validation_positions=args.calibration_validation_positions,
            quantisation_audit_positions=args.quantisation_audit_positions,
        )
    elif args.command == "run-incremental-mlx":
        result = run_incremental_mlx(args.run_directory)
    elif args.command == "incremental-status":
        result = incremental_mlx_status(args.run_directory)
    else:  # pragma: no cover - argparse enforces a known command.
        raise AssertionError(f"unknown command: {args.command}")
    _print_json(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
