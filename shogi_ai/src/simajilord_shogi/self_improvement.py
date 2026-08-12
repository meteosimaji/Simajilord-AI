"""Versioned candidate generation with deep teaching and gated promotion."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from rsshogi.core import Board

from .arena import PairedArenaSummary, evaluate_checkpoint_pair
from .checkpoint import bounded_checkpoint_provenance, load_checkpoint, save_checkpoint
from .compute_interlock import InterlockedEvaluator
from .config import ReanalysisConfig, SearchConfig
from .distillation_metrics import evaluate_teacher_alignment
from .domain import GameRecord, PositionSample, Termination
from .ensemble import normalized_sfen
from .model import MLXEvaluator
from .opening_suite import OpeningPosition, OpeningSuite, load_opening_suite
from .reanalysis import reanalyse_games
from .replay import append_games, load_games, position_samples
from .rights_lineage import expected_lineage_rights_summary, summarize_teacher_sidecar
from .selfplay import batched_self_play
from .trainer import TrainingInterlockConfig, train


@dataclass(frozen=True, slots=True)
class SelfImprovementConfig:
    actor_games: int = 32
    actor_simulations: int = 800
    actor_temperature_moves: int = 24
    actor_resign_threshold: float | None = None
    teacher_simulations: int = 6400
    reanalyse_fraction: float = 0.5
    reanalysis_parallel_positions: int = 64
    replay_window_generations: int = 8
    checkmate_sample_priority: float = 2.0
    checkmate_horizon_plies: int = 16
    training_steps: int = 1000
    batch_size: int = 64
    learning_rate: float = 1e-4
    arena_games: int | None = None
    arena_simulations: int = 1600
    promotion_min_games: int | None = None
    promotion_min_pairs: int = 32
    promotion_lower_bound: float = 0.5
    arena_bootstrap_iterations: int = 20_000
    max_plies: int | None = None
    initial_sfen: str | None = None
    opening_suite: str | None = None
    actor_opening_split: str = "actor"
    arena_opening_split: str = "arena"
    seed: int = 0
    anchor_replays: tuple[str, ...] = ()
    validation_replays: tuple[str, ...] = ()
    validation_batch_size: int = 64
    maximum_validation_policy_cross_entropy_increase: float = 0.0
    maximum_validation_value_mse_increase: float = 0.0
    teacher_policy_mix: float = 0.75
    teacher_value_mix: float = 0.0
    implicit_policy_temperature: float = 0.10
    implicit_policy_mix: float = 0.5
    proven_mate_policy_loss_weight: float = 0.25
    legal_label_smoothing: float = 0.01
    curriculum_depth_ratio: float = 64.0
    minimum_teacher_policy_mix: float = 0.1
    maximum_gradient_norm: float = 1.0
    maximum_probe_loss_ratio: float = 1.25
    max_tree_nodes: int | None = None
    memory_check_interval: int = 1024
    intra_root_batch_size: int = 1
    intra_root_virtual_loss: float = 0.0
    max_evaluation_batch_size: int = 256

    def __post_init__(self) -> None:
        positive = (
            self.actor_games,
            self.actor_simulations,
            self.teacher_simulations,
            self.reanalysis_parallel_positions,
            self.replay_window_generations,
            self.checkmate_horizon_plies,
            self.training_steps,
            self.batch_size,
            self.arena_simulations,
            self.promotion_min_pairs,
            self.arena_bootstrap_iterations,
            self.validation_batch_size,
        )
        if any(value < 1 for value in positive):
            raise ValueError("self-improvement counts must be positive")
        if self.teacher_simulations < self.actor_simulations * 2:
            raise ValueError("teacher_simulations must be at least twice actor_simulations")
        if self.actor_temperature_moves < 0:
            raise ValueError("actor_temperature_moves must be non-negative")
        if self.actor_resign_threshold is not None and not (
            -1 <= self.actor_resign_threshold <= 0
        ):
            raise ValueError("actor_resign_threshold must be between -1 and 0")
        if self.arena_games is not None and self.arena_games < 2:
            raise ValueError("arena_games must be at least two when configured")
        if self.arena_games is not None and self.arena_games % 2:
            raise ValueError("arena_games must be even for paired colors")
        if self.promotion_min_games is not None and self.promotion_min_games < 1:
            raise ValueError("promotion_min_games must be positive when configured")
        if not 0 < self.reanalyse_fraction <= 1:
            raise ValueError("reanalyse_fraction must be in (0, 1]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not math.isfinite(self.checkmate_sample_priority) or self.checkmate_sample_priority < 0:
            raise ValueError("checkmate_sample_priority must be finite and non-negative")
        if not 0 <= self.promotion_lower_bound <= 1:
            raise ValueError("promotion_lower_bound must be in [0, 1]")
        if self.max_plies is not None and self.max_plies < 1:
            raise ValueError("max_plies must be positive when configured")
        if self.initial_sfen is not None and not Board(self.initial_sfen).is_valid():
            raise ValueError("initial_sfen is not a valid shogi position")
        if self.opening_suite is not None and not self.opening_suite.strip():
            raise ValueError("opening_suite path must not be empty")
        if self.opening_suite is not None and self.initial_sfen is not None:
            raise ValueError("opening_suite and legacy initial_sfen are mutually exclusive")
        if self.opening_suite is not None and self.arena_games is not None:
            raise ValueError("arena_games is legacy-only when opening_suite is configured")
        if self.opening_suite is not None and self.promotion_min_games is not None:
            raise ValueError("promotion_min_games is legacy-only; use promotion_min_pairs")
        if not self.actor_opening_split.strip() or not self.arena_opening_split.strip():
            raise ValueError("opening split names must not be empty")
        if (
            self.actor_opening_split != self.actor_opening_split.strip()
            or self.arena_opening_split != self.arena_opening_split.strip()
        ):
            raise ValueError("opening split names must be trimmed")
        if self.actor_opening_split == self.arena_opening_split:
            raise ValueError("actor and arena opening splits must be distinct")
        if self.seed < 0:
            raise ValueError("self-improvement seed must be non-negative")
        if any(not path.strip() or path != path.strip() for path in self.anchor_replays):
            raise ValueError("anchor replay paths must be non-empty and trimmed")
        if any(not path.strip() or path != path.strip() for path in self.validation_replays):
            raise ValueError("validation replay paths must be non-empty and trimmed")
        if len(set(self.anchor_replays)) != len(self.anchor_replays):
            raise ValueError("anchor replay paths must be unique")
        if len(set(self.validation_replays)) != len(self.validation_replays):
            raise ValueError("validation replay paths must be unique")
        validation_tolerances = (
            self.maximum_validation_policy_cross_entropy_increase,
            self.maximum_validation_value_mse_increase,
        )
        if any(not math.isfinite(value) or value < 0 for value in validation_tolerances):
            raise ValueError("validation regression tolerances must be finite and non-negative")
        if not 0 < self.teacher_policy_mix <= 1 or not 0 <= self.teacher_value_mix <= 1:
            raise ValueError("teacher policy/value mixes are out of range")
        if (
            not math.isfinite(self.implicit_policy_temperature)
            or self.implicit_policy_temperature <= 0
        ):
            raise ValueError("implicit policy temperature must be finite and positive")
        if not math.isfinite(self.implicit_policy_mix) or not 0 <= self.implicit_policy_mix <= 1:
            raise ValueError("implicit policy mix must be finite in [0, 1]")
        if (
            not math.isfinite(self.proven_mate_policy_loss_weight)
            or not 0 <= self.proven_mate_policy_loss_weight <= 1
        ):
            raise ValueError("proven-mate policy loss weight must be finite in [0, 1]")
        if not 0 <= self.legal_label_smoothing < 1:
            raise ValueError("legal_label_smoothing must be in [0, 1)")
        if self.curriculum_depth_ratio < 1:
            raise ValueError("curriculum_depth_ratio must be at least one")
        if not 0 < self.minimum_teacher_policy_mix <= self.teacher_policy_mix:
            raise ValueError("minimum teacher policy mix is out of range")
        if self.maximum_gradient_norm <= 0 or self.maximum_probe_loss_ratio < 1:
            raise ValueError("training stability limits are invalid")
        if self.max_tree_nodes is not None and self.max_tree_nodes < 1024:
            raise ValueError("max_tree_nodes must be at least 1024 when configured")
        if self.memory_check_interval < 1:
            raise ValueError("memory_check_interval must be positive")
        if self.intra_root_batch_size < 1 or self.max_evaluation_batch_size < 1:
            raise ValueError("MCTS neural batch sizes must be positive")
        if not 0 <= self.intra_root_virtual_loss <= 1:
            raise ValueError("MCTS intra-root virtual loss must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    generation: int
    status: str
    promoted: bool
    champion_before: str
    candidate: str
    champion_after: str
    manifest: str
    arena: PairedArenaSummary


GENERATION_MANIFEST_SCHEMA = 5
SELF_PLAY_RL_CONTRACT_SCHEMA = "meteo-self-play-rl-contract-v1"
CHECKPOINT_HISTORY_LIMIT = 2
GENERATION_SEED_STRIDE = 10_000_019
_GENERATION_DIRECTORY_RE = re.compile(r"generation-[0-9]{6}\Z")
_REPLAY_RECEIPT_KEYS = {
    "actor.jsonl": "actor_replay",
    "deep.jsonl": "deep_replay",
    "arena.jsonl": "arena_replay",
}
_STAGES = (
    "initialized",
    "actor_saved",
    "reanalysed",
    "trained",
    "validated",
    "arena_complete",
    "complete",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_games(path: Path, games: list[GameRecord]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    append_games(temporary, games)
    temporary.replace(path)


def _checkpoint_digest(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    for name in ("metadata.json", "weights.safetensors"):
        file_path = checkpoint / name
        with file_path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _sha256_replay(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replay_receipt(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"replay artifact must be a regular file: {path}")
    return {
        "name": path.name,
        "sha256": _sha256_replay(path),
        "bytes": path.stat().st_size,
    }


def _resolved_replay(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _checkpoint_state_record(checkpoint: Path) -> dict[str, object]:
    resolved = checkpoint.expanduser().resolve(strict=True)
    identity = bounded_checkpoint_provenance(resolved)
    return {
        "path": str(resolved),
        "step": identity["step"],
        "load_checkpoint_sha256": identity["load_checkpoint_sha256"],
        "all_files_sha256": identity["all_files_sha256"],
    }


def _retained_checkpoint_records(latest: Path, previous: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for checkpoint in (latest, previous):
        record = _checkpoint_state_record(checkpoint)
        normalized = str(record["path"]).casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        records.append(record)
    return records[:CHECKPOINT_HISTORY_LIMIT]


def _prune_managed_candidates(workdir: Path, retained: set[Path]) -> list[str]:
    """Remove only completed generation candidates owned by this workdir."""

    root = workdir.expanduser().resolve(strict=True)
    retained_resolved = {path.expanduser().resolve(strict=True) for path in retained}
    removed: list[str] = []
    for generation_dir in sorted(root.iterdir()):
        if (
            not generation_dir.is_dir()
            or _GENERATION_DIRECTORY_RE.fullmatch(generation_dir.name) is None
        ):
            continue
        candidate = generation_dir / "candidate"
        if not candidate.exists() and not candidate.is_symlink():
            continue
        if candidate.is_symlink():
            raise ValueError(f"managed candidate must not be a symlink: {candidate}")
        candidate_resolved = candidate.resolve(strict=True)
        if not candidate_resolved.is_dir() or not candidate_resolved.is_relative_to(root):
            raise ValueError(f"managed candidate escaped its workdir: {candidate}")
        if candidate_resolved in retained_resolved:
            continue
        manifest_path = generation_dir / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError(f"managed candidate lacks a regular manifest: {candidate}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("stage") != "complete":
            continue
        if manifest.get("candidate") != str(candidate_resolved):
            raise ValueError(f"managed candidate does not match its manifest: {candidate}")
        shutil.rmtree(candidate_resolved)
        removed.append(str(candidate_resolved))
    return removed


def _prune_managed_replays(
    workdir: Path,
    *,
    current_generation: int,
    replay_window_generations: int,
) -> dict[str, object]:
    """Bound replay storage while retaining manifests and the active training window.

    The immediately preceding generation keeps its actor and arena records for
    diagnosis.  Deep reanalysis records remain available for exactly the
    configured replay window.  A file is removed only after its completed
    generation manifest proves the file name, byte count, and SHA-256 digest.
    """

    root = workdir.expanduser().resolve(strict=True)
    first_deep_generation = max(1, current_generation - replay_window_generations + 1)
    removed: list[dict[str, object]] = []
    reclaimed_bytes = 0
    for generation_dir in sorted(root.iterdir()):
        if (
            not generation_dir.is_dir()
            or _GENERATION_DIRECTORY_RE.fullmatch(generation_dir.name) is None
        ):
            continue
        generation = int(generation_dir.name.removeprefix("generation-"))
        if generation >= current_generation:
            continue
        manifest_path = generation_dir / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError(f"managed replay lacks a regular manifest: {generation_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != GENERATION_MANIFEST_SCHEMA
            or manifest.get("stage") != "complete"
        ):
            continue
        removable_names = ["actor.jsonl", "arena.jsonl"]
        if generation < first_deep_generation:
            removable_names.append("deep.jsonl")
        for name in removable_names:
            replay_path = generation_dir / name
            if not replay_path.exists() and not replay_path.is_symlink():
                continue
            if replay_path.is_symlink() or not replay_path.is_file():
                raise ValueError(f"managed replay must be a regular file: {replay_path}")
            replay_resolved = replay_path.resolve(strict=True)
            if not replay_resolved.is_relative_to(root):
                raise ValueError(f"managed replay escaped its workdir: {replay_path}")
            receipt_key = _REPLAY_RECEIPT_KEYS[name]
            observed = _replay_receipt(replay_path)
            if manifest.get(receipt_key) != observed:
                raise ValueError(
                    f"managed replay does not match {receipt_key} receipt: {replay_path}"
                )
            observed_bytes = replay_path.stat().st_size
            replay_path.unlink()
            reclaimed_bytes += observed_bytes
            removed.append(
                {
                    "generation": generation,
                    "name": name,
                    "sha256": observed["sha256"],
                    "bytes": observed_bytes,
                }
            )
    return {
        "replay_window_generations": replay_window_generations,
        "first_retained_deep_generation": first_deep_generation,
        "reclaimed_bytes": reclaimed_bytes,
        "removed": removed,
    }


def _replay_identity(path: Path, *, kind: str) -> dict[str, object]:
    resolved = _resolved_replay(path, label=f"{kind} replay")
    sidecars: list[dict[str, object]] = []
    for suffix in (".ensemble.json", ".provenance.json"):
        sidecar = resolved.with_suffix(resolved.suffix + suffix)
        if not sidecar.exists() and not sidecar.is_symlink():
            continue
        resolved_sidecar = _resolved_replay(sidecar, label="training lineage sidecar")
        sidecar_bytes = resolved_sidecar.read_bytes()
        sidecar_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()
        try:
            sidecar_payload: object = json.loads(sidecar_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"training lineage sidecar is not valid UTF-8 JSON: {resolved_sidecar}"
            ) from error
        if not isinstance(sidecar_payload, dict):
            raise ValueError(
                f"training lineage sidecar must be a JSON object: {resolved_sidecar}"
            )
        sidecars.append(
            {
                "kind": "teacher_lineage_sidecar",
                "sha256": sidecar_sha256,
                "bytes": len(sidecar_bytes),
                "rights_restriction_summary": summarize_teacher_sidecar(
                    sidecar_payload,
                    sidecar_sha256=sidecar_sha256,
                ),
            }
        )
    return {
        "kind": kind,
        "name": resolved.name,
        "sha256": _sha256_replay(resolved),
        "bytes": resolved.stat().st_size,
        "lineage_sidecars": sidecars,
    }


def _self_improvement_lineage(
    champion: Path,
    replay_paths: list[Path],
    *,
    self_play_paths: tuple[Path, ...],
    current_deep_path: Path,
    generation: int,
    config: SelfImprovementConfig,
) -> dict[str, object]:
    parent = bounded_checkpoint_provenance(champion)
    self_play_path_set = set(self_play_paths)
    training_inputs = [
        _replay_identity(
            replay_path,
            kind=(
                "self_play_deep_reanalysis_current"
                if replay_path == current_deep_path
                else (
                    "self_play_deep_reanalysis_replay_window"
                    if replay_path in self_play_path_set
                    else "anchor"
                )
            ),
        )
        for replay_path in replay_paths
    ]
    dataset_payload = json.dumps(
        training_inputs,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    lineage: dict[str, object] = {
        "schema": "meteo-training-lineage-v2",
        "method": "single-network AlphaZero-style self-play policy iteration",
        "generation": generation,
        "parent_checkpoint": parent,
        "training_inputs": training_inputs,
        "dataset_fingerprint": hashlib.sha256(dataset_payload).hexdigest(),
        # The deeper search is generated by the Meteo parent itself, not an
        # independently licensed external teacher. Parent restrictions are
        # still inherited through the bounded parent identity.
        "teacher_sources": [],
        "hyperparameters": {
            "training_mode": SELF_PLAY_RL_CONTRACT_SCHEMA,
            "require_teacher": False,
            "all_complete_self_play_positions_use_actor_mcts_policy": True,
            "all_complete_self_play_positions_use_game_outcome_value": True,
            "external_model_position_routing": False,
            "replay_window_generations": config.replay_window_generations,
            "reanalysis_parallel_positions": config.reanalysis_parallel_positions,
            "actor_resign_threshold": config.actor_resign_threshold,
            "arena_resign_threshold": None,
            "checkmate_sample_priority": config.checkmate_sample_priority,
            "checkmate_horizon_plies": config.checkmate_horizon_plies,
            "training_steps": config.training_steps,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "teacher_policy_mix": config.teacher_policy_mix,
            "teacher_value_mix": config.teacher_value_mix,
            "implicit_policy_temperature": config.implicit_policy_temperature,
            "implicit_policy_mix": config.implicit_policy_mix,
            "proven_mate_policy_loss_weight": config.proven_mate_policy_loss_weight,
            "legal_label_smoothing": config.legal_label_smoothing,
            "curriculum_depth_ratio": config.curriculum_depth_ratio,
            "minimum_teacher_policy_mix": config.minimum_teacher_policy_mix,
            "maximum_gradient_norm": config.maximum_gradient_norm,
            "maximum_probe_loss_ratio": config.maximum_probe_loss_ratio,
        },
        "optimizer": {
            "name": "AdamW",
            "state_restored": False,
            "reset_reason": "new self-play generation dataset",
        },
        "rng_state_restored": False,
        "exact_resume_from_parent": False,
        "output_contains_exact_resume_state": False,
    }
    lineage["rights_restriction_summary"] = expected_lineage_rights_summary(lineage)
    return lineage


def _recent_self_play_replays(
    workdir: Path,
    *,
    generation: int,
    current_deep_path: Path,
    window_generations: int,
) -> tuple[Path, ...]:
    """Return a bounded, chronological replay window from this RL lineage only."""

    first_generation = max(1, generation - window_generations + 1)
    paths: list[Path] = []
    for replay_generation in range(first_generation, generation + 1):
        generation_dir = workdir / f"generation-{replay_generation:06d}"
        replay_path = (
            current_deep_path
            if replay_generation == generation
            else generation_dir / "deep.jsonl"
        )
        if replay_generation != generation:
            manifest_path = generation_dir / "manifest.json"
            if manifest_path.is_symlink() or not manifest_path.is_file():
                raise ValueError(
                    f"replay-window generation lacks a regular manifest: {manifest_path}"
                )
            prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                not isinstance(prior_manifest, dict)
                or prior_manifest.get("schema") != GENERATION_MANIFEST_SCHEMA
                or prior_manifest.get("stage") != "complete"
            ):
                raise ValueError(
                    "replay window contains a pre-RL or incomplete generation; "
                    "start the corrected learner in a new workdir"
                )
        paths.append(_resolved_replay(replay_path, label="self-play replay-window input"))
    return tuple(paths)


def _self_play_training_dataset(
    self_play_paths: tuple[Path, ...],
    anchor_paths: tuple[Path, ...],
    *,
    actor_resign_threshold: float | None,
    teacher_policy_mix: float,
    teacher_value_mix: float,
    checkmate_horizon_plies: int,
    implicit_policy_mix: float = 0.5,
    proven_mate_policy_loss_weight: float = 0.25,
) -> tuple[list[PositionSample], dict[str, object]]:
    """Build and describe the complete-position RL dataset without teacher filtering."""

    samples: list[PositionSample] = []
    self_play_samples = 0
    deep_policy_samples = 0
    deep_value_samples = 0
    checkmate_samples = 0
    complete_games = 0
    incomplete_games = 0
    termination_counts: dict[str, int] = {}
    replay_records: list[dict[str, object]] = []
    for replay_path in self_play_paths:
        games = load_games(replay_path)
        replay_complete_games = 0
        replay_incomplete_games = 0
        replay_samples = 0
        replay_deep_policy_samples = 0
        for game_index, game in enumerate(games):
            if game.termination is Termination.MAX_PLIES:
                replay_incomplete_games += 1
                continue
            if actor_resign_threshold is None and game.termination is Termination.RESIGNATION:
                raise ValueError(
                    "no-resignation self-play replay contains a resignation: "
                    f"{replay_path}:{game_index}"
                )
            replay_complete_games += 1
            termination_counts[game.termination.value] = (
                termination_counts.get(game.termination.value, 0) + 1
            )
            for sample_index, sample in enumerate(game.samples):
                expected_value = (
                    0.0
                    if game.winner is None
                    else (1.0 if sample.turn == game.winner else -1.0)
                )
                if sample.value_target != expected_value:
                    raise ValueError(
                        "self-play outcome target has the wrong side-to-move perspective: "
                        f"{replay_path}:{game_index}:{sample_index}"
                    )
                if game.termination is Termination.CHECKMATE:
                    checkmate_distance = len(game.samples) - sample_index
                    sample = replace(
                        sample,
                        terminal_checkmate_distance=checkmate_distance,
                    )
                    if checkmate_distance <= checkmate_horizon_plies:
                        checkmate_samples += 1
                samples.append(sample)
                replay_samples += 1
                if sample.teacher_policy is not None:
                    replay_deep_policy_samples += 1
                if sample.teacher_value is not None:
                    deep_value_samples += 1
        complete_games += replay_complete_games
        incomplete_games += replay_incomplete_games
        self_play_samples += replay_samples
        deep_policy_samples += replay_deep_policy_samples
        replay_records.append(
            {
                "name": replay_path.name,
                "sha256": _sha256_replay(replay_path),
                "complete_games": replay_complete_games,
                "incomplete_games_excluded": replay_incomplete_games,
                "outcome_samples": replay_samples,
                "deep_policy_overlay_samples": replay_deep_policy_samples,
            }
        )

    anchor_samples = 0
    for anchor_path in anchor_paths:
        current_anchor_samples = position_samples(load_games(anchor_path))
        samples.extend(current_anchor_samples)
        anchor_samples += len(current_anchor_samples)
    if not samples:
        raise ValueError("self-play RL dataset contains no complete training samples")

    return samples, {
        "schema": SELF_PLAY_RL_CONTRACT_SCHEMA,
        "single_meteo_network": True,
        "external_model_position_routing": False,
        "require_teacher": False,
        "complete_self_play_games": complete_games,
        "incomplete_self_play_games_excluded": incomplete_games,
        "termination_counts": dict(sorted(termination_counts.items())),
        "self_play_outcome_samples": self_play_samples,
        "actor_mcts_policy_samples": self_play_samples,
        "actor_policy_has_nonzero_mix_for_deep_samples": teacher_policy_mix < 1.0,
        "deep_policy_overlay_samples": deep_policy_samples,
        "deep_value_available_samples": deep_value_samples,
        "checkmate_horizon_plies": checkmate_horizon_plies,
        "checkmate_horizon_samples": checkmate_samples,
        "teacher_policy_mix": teacher_policy_mix,
        "teacher_value_mix": teacher_value_mix,
        "implicit_policy_mix": implicit_policy_mix,
        "proven_mate_policy_loss_weight": proven_mate_policy_loss_weight,
        "implicit_policy_source": "all-legal-root-Q-softmax",
        "proven_mate_policy_loss": "internally-verified-set-mass",
        "game_outcome_value_mix": 1.0 - teacher_value_mix,
        "anchor_samples": anchor_samples,
        "training_samples": len(samples),
        "actor_resign_threshold": actor_resign_threshold,
        "arena_resign_threshold": None,
        "replay_window": replay_records,
    }


def _bounded_key_receipt(keys: set[str]) -> dict[str, object]:
    ordered = sorted(keys)
    digest = hashlib.sha256()
    for key in ordered:
        encoded = key.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return {
        "count": len(ordered),
        "sha256": digest.hexdigest(),
        "examples": ordered[:32],
        "examples_truncated": len(ordered) > 32,
    }


def _validation_gate(
    candidate: Path,
    champion: Path,
    *,
    training_replay_paths: list[Path],
    config: SelfImprovementConfig,
    compute_interlock: TrainingInterlockConfig | None,
) -> dict[str, object]:
    if not config.validation_replays:
        return {
            "configured": False,
            "passed": False,
            "blockers": ["independent_validation_not_configured"],
            "replays": [],
            "training_overlap": _bounded_key_receipt(set()),
        }

    validation_paths = [
        _resolved_replay(Path(path), label="validation replay")
        for path in config.validation_replays
    ]
    if len({str(path).casefold() for path in validation_paths}) != len(validation_paths):
        raise ValueError("validation replay paths resolve to duplicate files")
    training_paths = [
        _resolved_replay(path, label="training replay") for path in training_replay_paths
    ]
    if {str(path).casefold() for path in validation_paths} & {
        str(path).casefold() for path in training_paths
    }:
        raise ValueError("validation replay must be distinct from every training replay")

    training_keys = _training_position_keys(training_paths)
    overlap_keys: set[str] = set()
    validation_games: list[list[GameRecord]] = []
    for validation_path in validation_paths:
        games = load_games(validation_path)
        validation_games.append(games)
        overlap_keys.update(
            training_keys
            & {normalized_sfen(sample.sfen) for game in games for sample in game.samples}
        )
    if overlap_keys:
        return {
            "configured": True,
            "passed": False,
            "blockers": ["training_validation_position_overlap"],
            "replays": [],
            "training_overlap": _bounded_key_receipt(overlap_keys),
        }

    candidate_model, _candidate_step = load_checkpoint(candidate)
    champion_model, _champion_step = load_checkpoint(champion)
    candidate_base = MLXEvaluator(candidate_model)
    champion_base = MLXEvaluator(champion_model)
    candidate_evaluator = (
        candidate_base
        if compute_interlock is None
        else InterlockedEvaluator(candidate_base, compute_interlock)
    )
    champion_evaluator = (
        champion_base
        if compute_interlock is None
        else InterlockedEvaluator(champion_base, compute_interlock)
    )
    blockers: list[str] = []
    replay_reports: list[dict[str, object]] = []
    for index, (validation_path, games) in enumerate(
        zip(validation_paths, validation_games, strict=True)
    ):
        samples = position_samples(games)
        champion_report = evaluate_teacher_alignment(
            champion_evaluator,
            samples,
            batch_size=config.validation_batch_size,
        )
        candidate_report = evaluate_teacher_alignment(
            candidate_evaluator,
            samples,
            batch_size=config.validation_batch_size,
        )
        policy_delta = (
            candidate_report.overall.policy_cross_entropy
            - champion_report.overall.policy_cross_entropy
        )
        policy_passed = policy_delta <= config.maximum_validation_policy_cross_entropy_increase
        champion_value_mse = champion_report.overall.value_mse
        candidate_value_mse = candidate_report.overall.value_mse
        value_delta: float | None
        if champion_value_mse is None and candidate_value_mse is None:
            value_delta = None
            value_passed = True
        elif champion_value_mse is None or candidate_value_mse is None:
            value_delta = None
            value_passed = False
        else:
            value_delta = candidate_value_mse - champion_value_mse
            value_passed = value_delta <= config.maximum_validation_value_mse_increase
        if not policy_passed:
            blockers.append(f"validation_policy_regression:{index}")
        if not value_passed:
            blockers.append(f"validation_value_regression:{index}")
        replay_reports.append(
            {
                "index": index,
                "name": validation_path.name,
                "sha256": _sha256_replay(validation_path),
                "bytes": validation_path.stat().st_size,
                "champion": champion_report.to_dict(),
                "candidate": candidate_report.to_dict(),
                "policy_cross_entropy_delta": policy_delta,
                "value_mse_delta": value_delta,
                "policy_passed": policy_passed,
                "value_passed": value_passed,
                "passed": policy_passed and value_passed,
            }
        )
    return {
        "configured": True,
        "passed": not blockers,
        "blockers": blockers,
        "policy_cross_entropy_increase_tolerance": (
            config.maximum_validation_policy_cross_entropy_increase
        ),
        "value_mse_increase_tolerance": config.maximum_validation_value_mse_increase,
        "replays": replay_reports,
        "training_overlap": _bounded_key_receipt(set()),
    }


def _stage_at_least(manifest: dict[str, Any], stage: str) -> bool:
    return _STAGES.index(str(manifest["stage"])) >= _STAGES.index(stage)


def _selected_split_manifest(suite: OpeningSuite, name: str) -> dict[str, object]:
    return suite.split(name).to_manifest()


def _opening_plan(
    config: SelfImprovementConfig,
) -> tuple[
    tuple[OpeningPosition, ...],
    tuple[OpeningPosition, ...],
    dict[str, object],
    bool,
]:
    """Resolve immutable production splits or the explicitly non-promotable legacy mode."""

    if config.opening_suite is not None:
        suite = load_opening_suite(Path(config.opening_suite))
        actor_split = suite.split(config.actor_opening_split)
        arena_split = suite.split(config.arena_opening_split)
        metadata = suite.to_manifest()
        metadata.update(
            mode="immutable_split_suite",
            actor_split=_selected_split_manifest(suite, actor_split.name),
            arena_split=_selected_split_manifest(suite, arena_split.name),
            promotion_eligible=True,
        )
        return actor_split.positions, arena_split.positions, metadata, False

    legacy = OpeningPosition.from_sfen(config.initial_sfen or Board().to_sfen())
    pair_count = (config.arena_games or 2) // 2
    metadata = {
        "mode": "legacy_single_opening_debug_only",
        "promotion_eligible": False,
        "repetitions": pair_count,
        "arena_split": {
            "name": "legacy-single-opening",
            "count": 1,
            "sha256": legacy.sha256,
            "positions": [legacy.to_manifest()],
        },
    }
    return (legacy,), tuple(legacy for _ in range(pair_count)), metadata, True


def _training_position_keys(replay_paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for replay_path in replay_paths:
        for game in load_games(replay_path):
            keys.add(normalized_sfen(game.initial_sfen))
            keys.update(normalized_sfen(sample.sfen) for sample in game.samples)
    return keys


def _result_from_manifest(manifest_path: Path, manifest: dict[str, Any]) -> GenerationResult:
    arena = PairedArenaSummary.from_dict(manifest["arena"])
    return GenerationResult(
        generation=int(manifest["generation"]),
        status=str(manifest["status"]),
        promoted=bool(manifest["promoted"]),
        champion_before=str(manifest["champion_before"]),
        candidate=str(manifest["candidate"]),
        champion_after=str(manifest["champion_after"]),
        manifest=str(manifest_path),
        arena=arena,
    )


def run_generation(
    bootstrap_champion: Path,
    workdir: Path,
    config: SelfImprovementConfig,
    *,
    training_interlock: TrainingInterlockConfig | None = None,
) -> GenerationResult:
    """Create or resume one generation and update the champion pointer only on promotion."""

    workdir = workdir.expanduser().resolve()
    actor_openings, arena_openings, opening_manifest, legacy_single_opening = _opening_plan(config)
    state_path = workdir / "state.json"
    if state_path.exists():
        saved_state = json.loads(state_path.read_text(encoding="utf-8"))
        if saved_state.get("rl_contract_schema") != SELF_PLAY_RL_CONTRACT_SCHEMA:
            raise ValueError(
                "pre-RL self-improvement state cannot enter the corrected learner; "
                "start it in a new workdir"
            )
        saved_opening_manifest = saved_state.get("opening_suite")
        if saved_opening_manifest is not None and saved_opening_manifest != opening_manifest:
            raise ValueError("opening suite changed within an existing improvement lineage")
        champion = Path(saved_state["champion"])
        generation = int(saved_state["generation"]) + 1
        cumulative_positions_seen_before = int(
            saved_state.get("cumulative_positions_seen", 0)
        )
    else:
        champion = bootstrap_champion.resolve()
        generation = 1
        cumulative_positions_seen_before = 0
    generation_seed = config.seed + (generation - 1) * GENERATION_SEED_STRIDE
    champion_model, _champion_step = load_checkpoint(champion)
    if (
        champion_model.config.history_input_version != 1
        or champion_model.config.canonical_head_version != 1
    ):
        raise ValueError(
            "the current self-improvement pipeline is legacy-only and cannot consume a "
            "history-input-v2/canonical-head-v2 champion; no generation was started"
        )
    generation_dir = workdir / f"generation-{generation:06d}"
    manifest_path = generation_dir / "manifest.json"
    actor_path = generation_dir / "actor.jsonl"
    deep_path = generation_dir / "deep.jsonl"
    candidate_path = generation_dir / "candidate"
    arena_replay_path = generation_dir / "arena.jsonl"
    expected_config = json.loads(json.dumps(asdict(config)))
    generation_interlock = (
        None if training_interlock is None else replace(training_interlock, generation=generation)
    )

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_schema = manifest.get("schema")
        if manifest_schema != GENERATION_MANIFEST_SCHEMA:
            raise ValueError(
                "pre-RL generation manifests cannot resume under the corrected training contract"
            )
        if manifest.get("config") != expected_config:
            raise ValueError("an incomplete generation exists with a different configuration")
        if manifest.get("opening_suite") != opening_manifest:
            raise ValueError("opening suite changed after generation creation")
        if manifest["champion_before"] != str(champion):
            raise ValueError("generation champion does not match current state")
        if manifest.get("generation_seed") != generation_seed:
            raise ValueError("generation seed does not match the deterministic seed schedule")
        if (
            manifest.get("cumulative_positions_seen_before")
            != cumulative_positions_seen_before
        ):
            raise ValueError("cumulative position counter changed during generation resume")
        if manifest["stage"] == "complete":
            latest = Path(str(manifest["candidate"]))
            previous = Path(str(manifest["champion_before"]))
            checkpoint_history = _retained_checkpoint_records(latest, previous)
            _atomic_json(
                state_path,
                {
                    "generation": generation,
                    "champion": manifest["champion_after"],
                    "last_manifest": str(manifest_path),
                    "opening_suite": opening_manifest,
                    "rl_contract_schema": SELF_PLAY_RL_CONTRACT_SCHEMA,
                    "checkpoint_history_limit": CHECKPOINT_HISTORY_LIMIT,
                    "checkpoint_history": checkpoint_history,
                    "cumulative_positions_seen": manifest["cumulative_positions_seen"],
                },
            )
            _prune_managed_candidates(workdir, {latest, previous})
            _prune_managed_replays(
                workdir,
                current_generation=generation,
                replay_window_generations=config.replay_window_generations,
            )
            return _result_from_manifest(manifest_path, manifest)
    else:
        manifest = {
            "schema": GENERATION_MANIFEST_SCHEMA,
            "generation": generation,
            "status": "running",
            "stage": "initialized",
            "config": expected_config,
            "rl_contract_schema": SELF_PLAY_RL_CONTRACT_SCHEMA,
            "opening_suite": opening_manifest,
            "generation_seed": generation_seed,
            "generation_seed_stride": GENERATION_SEED_STRIDE,
            "cumulative_positions_seen_before": cumulative_positions_seen_before,
            "champion_before": str(champion),
            "champion_before_sha256": _checkpoint_digest(champion),
        }
        _atomic_json(manifest_path, manifest)

    def record_stage_progress(
        phase: str,
        payload: Mapping[str, int | float],
    ) -> None:
        manifest["stage_progress"] = {"phase": phase, **payload}
        _atomic_json(manifest_path, manifest)

    actor_search = SearchConfig(
        simulations=config.actor_simulations,
        implicit_policy_temperature=config.implicit_policy_temperature,
        max_plies=config.max_plies,
        temperature_moves=config.actor_temperature_moves,
        resign_threshold=config.actor_resign_threshold,
        max_tree_nodes=config.max_tree_nodes,
        memory_check_interval=config.memory_check_interval,
        intra_root_batch_size=config.intra_root_batch_size,
        intra_root_virtual_loss=config.intra_root_virtual_loss,
        max_evaluation_batch_size=config.max_evaluation_batch_size,
    )
    actor_offset = generation_seed % len(actor_openings)
    initial_sfens = [
        actor_openings[(actor_offset + index) % len(actor_openings)].sfen
        for index in range(config.actor_games)
    ]
    if not _stage_at_least(manifest, "actor_saved"):
        record_stage_progress(
            "actor",
            {"games_total": config.actor_games, "games_completed": 0},
        )
        actor_records = batched_self_play(
            champion,
            actor_search,
            games=config.actor_games,
            seed=generation_seed,
            initial_sfens=initial_sfens,
            compute_interlock=generation_interlock,
            progress_callback=lambda progress: record_stage_progress(
                "actor", {"games_total": config.actor_games, **progress}
            ),
        )
        _atomic_games(actor_path, actor_records)
        manifest.pop("stage_progress", None)
        manifest.update(
            stage="actor_saved",
            actor_games=len(actor_records),
            actor_complete=sum(game.termination.value != "max_plies" for game in actor_records),
            actor_opening_sha256=[
                actor_openings[(actor_offset + index) % len(actor_openings)].sha256
                for index in range(config.actor_games)
            ],
            actor_replay=_replay_receipt(actor_path),
            actor_compute_interlock=(
                {"enabled": False}
                if generation_interlock is None
                else generation_interlock.public_metadata()
            ),
        )
        _atomic_json(manifest_path, manifest)

    if not _stage_at_least(manifest, "reanalysed"):
        champion_model, _ = load_checkpoint(champion)
        base_evaluator = MLXEvaluator(champion_model)
        evaluator = (
            base_evaluator
            if generation_interlock is None
            else InterlockedEvaluator(base_evaluator, generation_interlock)
        )
        teacher_config = ReanalysisConfig(
            teacher_simulation_multiplier=max(
                2, config.teacher_simulations // config.actor_simulations
            ),
            minimum_teacher_simulations=config.teacher_simulations,
            reanalyse_fraction=config.reanalyse_fraction,
        )
        deep_games = reanalyse_games(
            load_games(actor_path),
            evaluator,
            actor_search,
            teacher_config,
            seed=generation_seed,
            max_parallel_positions=config.reanalysis_parallel_positions,
            progress_callback=lambda completed, total: record_stage_progress(
                "reanalysis",
                {
                    "positions_completed": completed,
                    "positions_total": total,
                    "parallel_positions": config.reanalysis_parallel_positions,
                },
            ),
        )
        _atomic_games(deep_path, deep_games)
        manifest.pop("stage_progress", None)
        manifest.update(
            stage="reanalysed",
            deep_teacher_samples=sum(
                sample.teacher_policy is not None for game in deep_games for sample in game.samples
            ),
            deep_replay=_replay_receipt(deep_path),
            teacher_compute_interlock=(
                {"enabled": False}
                if generation_interlock is None
                else generation_interlock.public_metadata()
            ),
        )
        _atomic_json(manifest_path, manifest)

    self_play_replay_paths = _recent_self_play_replays(
        workdir,
        generation=generation,
        current_deep_path=deep_path,
        window_generations=config.replay_window_generations,
    )
    anchor_replay_paths = tuple(
        _resolved_replay(Path(path), label="anchor replay") for path in config.anchor_replays
    )
    if set(self_play_replay_paths) & set(anchor_replay_paths):
        raise ValueError("self-play replay window and anchor replay paths must be disjoint")
    training_replay_paths = [*self_play_replay_paths, *anchor_replay_paths]

    if not _stage_at_least(manifest, "trained"):
        champion_model, champion_step = load_checkpoint(champion)
        samples, rl_training_contract = _self_play_training_dataset(
            self_play_replay_paths,
            anchor_replay_paths,
            actor_resign_threshold=config.actor_resign_threshold,
            teacher_policy_mix=config.teacher_policy_mix,
            teacher_value_mix=config.teacher_value_mix,
            implicit_policy_mix=config.implicit_policy_mix,
            proven_mate_policy_loss_weight=config.proven_mate_policy_loss_weight,
            checkmate_horizon_plies=config.checkmate_horizon_plies,
        )
        metrics = train(
            champion_model,
            samples,
            steps=config.training_steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            seed=generation_seed,
            require_teacher=False,
            teacher_policy_mix=config.teacher_policy_mix,
            teacher_value_mix=config.teacher_value_mix,
            implicit_policy_mix=config.implicit_policy_mix,
            proven_mate_policy_loss_weight=config.proven_mate_policy_loss_weight,
            legal_label_smoothing=config.legal_label_smoothing,
            curriculum_depth_ratio=config.curriculum_depth_ratio,
            minimum_teacher_policy_mix=config.minimum_teacher_policy_mix,
            maximum_gradient_norm=config.maximum_gradient_norm,
            maximum_probe_loss_ratio=config.maximum_probe_loss_ratio,
            checkmate_sample_priority=config.checkmate_sample_priority,
            checkmate_horizon_plies=config.checkmate_horizon_plies,
            training_interlock=generation_interlock,
            progress_callback=lambda progress: record_stage_progress(
                "training", progress
            ),
        )
        save_checkpoint(
            champion_model,
            candidate_path,
            step=champion_step + config.training_steps,
            lineage=_self_improvement_lineage(
                champion,
                training_replay_paths,
                self_play_paths=self_play_replay_paths,
                current_deep_path=deep_path.resolve(),
                generation=generation,
                config=config,
            ),
        )
        retention_pruned_before_arena = _prune_managed_candidates(
            workdir, {candidate_path.resolve(), champion.resolve()}
        )
        manifest.pop("stage_progress", None)
        manifest.update(
            stage="trained",
            candidate=str(candidate_path.resolve()),
            candidate_sha256=_checkpoint_digest(candidate_path),
            training=asdict(metrics),
            position_presentations=metrics.steps * config.batch_size,
            rl_training_contract=rl_training_contract,
            training_compute_interlock=(
                {"enabled": False}
                if generation_interlock is None
                else generation_interlock.public_metadata()
            ),
            retention_pruned_before_arena=retention_pruned_before_arena,
        )
        _atomic_json(manifest_path, manifest)

    if not _stage_at_least(manifest, "validated"):
        validation = _validation_gate(
            candidate_path,
            champion,
            training_replay_paths=training_replay_paths,
            config=config,
            compute_interlock=generation_interlock,
        )
        manifest.update(stage="validated", validation=validation)
        _atomic_json(manifest_path, manifest)

    if not _stage_at_least(manifest, "arena_complete"):
        arena_search = SearchConfig(
            simulations=config.arena_simulations,
            # Production promotion games continue to a rules result.  The
            # legacy single-opening smoke mode may retain its explicit cutoff.
            max_plies=config.max_plies if legacy_single_opening else None,
            temperature_moves=0,
            temperature=0.0,
            dirichlet_fraction=0.0,
            resign_threshold=None,
            max_tree_nodes=config.max_tree_nodes,
            memory_check_interval=config.memory_check_interval,
            intra_root_batch_size=config.intra_root_batch_size,
            intra_root_virtual_loss=config.intra_root_virtual_loss,
            max_evaluation_batch_size=config.max_evaluation_batch_size,
        )
        arena_keys = {opening.normalized_key for opening in arena_openings}
        training_arena_overlap = tuple(
            sorted(arena_keys & _training_position_keys(training_replay_paths))
        )
        promotion_min_pairs = config.promotion_min_pairs
        if legacy_single_opening and config.promotion_min_games is not None:
            promotion_min_pairs = max(1, (config.promotion_min_games + 1) // 2)
        arena_seed = generation_seed + 1_000_000
        summary, arena_records = evaluate_checkpoint_pair(
            candidate_path,
            champion,
            arena_search,
            openings=[opening.sfen for opening in arena_openings],
            seed=arena_seed,
            promotion_min_pairs=promotion_min_pairs,
            promotion_lower_bound=config.promotion_lower_bound,
            bootstrap_iterations=config.arena_bootstrap_iterations,
            allow_repeated_openings=legacy_single_opening,
            legacy_single_opening=legacy_single_opening,
            promotion_eligible=(
                not legacy_single_opening
                and isinstance(manifest.get("validation"), dict)
                and manifest["validation"].get("passed") is True
            ),
            training_arena_overlap_keys=training_arena_overlap,
            compute_interlock=generation_interlock,
            progress_callback=lambda progress: record_stage_progress(
                "arena", progress
            ),
        )
        _atomic_games(arena_replay_path, arena_records)
        manifest.pop("stage_progress", None)
        manifest.update(
            stage="arena_complete",
            arena=summary.to_dict(),
            arena_opening_normalized_keys=[opening.normalized_key for opening in arena_openings],
            arena_opening_sha256=[opening.sha256 for opening in arena_openings],
            training_arena_overlap_keys=list(training_arena_overlap),
            arena_ci={
                "method": summary.ci_method,
                "confidence_level": summary.confidence_level,
                "seed": arena_seed,
                "bootstrap_iterations": summary.bootstrap_iterations,
            },
            arena_replay=_replay_receipt(arena_replay_path),
            arena_rules_contract={
                "network_resignation": False,
                "max_plies": arena_search.max_plies,
                "production_continues_to_rules_terminal": not legacy_single_opening,
                "incomplete_games_block_promotion": True,
            },
            arena_compute_interlock=(
                {"enabled": False}
                if generation_interlock is None
                else generation_interlock.public_metadata()
            ),
        )
        _atomic_json(manifest_path, manifest)

    summary = PairedArenaSummary.from_dict(manifest["arena"])
    champion_after = candidate_path.resolve() if summary.promoted else champion
    checkpoint_history = _retained_checkpoint_records(candidate_path, champion)
    cumulative_positions_seen = cumulative_positions_seen_before + int(
        manifest["position_presentations"]
    )
    manifest.update(
        stage="complete",
        status="promoted" if summary.promoted else "rejected",
        promoted=summary.promoted,
        champion_after=str(champion_after),
        checkpoint_history_limit=CHECKPOINT_HISTORY_LIMIT,
        checkpoint_history=checkpoint_history,
        cumulative_positions_seen=cumulative_positions_seen,
    )
    _atomic_json(manifest_path, manifest)
    _atomic_json(
        state_path,
        {
            "generation": generation,
            "champion": str(champion_after),
            "last_manifest": str(manifest_path),
            "opening_suite": opening_manifest,
            "rl_contract_schema": SELF_PLAY_RL_CONTRACT_SCHEMA,
            "checkpoint_history_limit": CHECKPOINT_HISTORY_LIMIT,
            "checkpoint_history": checkpoint_history,
            "cumulative_positions_seen": cumulative_positions_seen,
        },
    )
    retention_pruned_after_completion = _prune_managed_candidates(
        workdir, {candidate_path.resolve(), champion.resolve()}
    )
    manifest["retention_pruned_after_completion"] = retention_pruned_after_completion
    manifest["replay_retention_after_completion"] = _prune_managed_replays(
        workdir,
        current_generation=generation,
        replay_window_generations=config.replay_window_generations,
    )
    _atomic_json(manifest_path, manifest)
    return _result_from_manifest(manifest_path, manifest)


def run_self_improvement(
    bootstrap_champion: Path,
    workdir: Path,
    config: SelfImprovementConfig,
    *,
    generations: int,
    training_interlock: TrainingInterlockConfig | None = None,
) -> list[GenerationResult]:
    if generations < 1:
        raise ValueError("generations must be positive")
    return [
        run_generation(
            bootstrap_champion,
            workdir,
            config,
            training_interlock=training_interlock,
        )
        for _ in range(generations)
    ]
