"""Versioned candidate generation with deep teaching and gated promotion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from rsshogi.core import Board

from .arena import PairedArenaSummary, evaluate_checkpoint_pair
from .checkpoint import load_checkpoint, save_checkpoint
from .compute_interlock import InterlockedEvaluator
from .config import ReanalysisConfig, SearchConfig
from .domain import GameRecord
from .ensemble import normalized_sfen
from .model import MLXEvaluator
from .opening_suite import OpeningPosition, OpeningSuite, load_opening_suite
from .reanalysis import reanalyse_game
from .replay import append_games, load_games, position_samples
from .selfplay import batched_self_play
from .trainer import TrainingInterlockConfig, train


@dataclass(frozen=True, slots=True)
class SelfImprovementConfig:
    actor_games: int = 32
    actor_simulations: int = 800
    actor_temperature_moves: int = 24
    teacher_simulations: int = 6400
    reanalyse_fraction: float = 0.5
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
    teacher_policy_mix: float = 0.75
    teacher_value_mix: float = 0.5
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
            self.training_steps,
            self.batch_size,
            self.arena_simulations,
            self.promotion_min_pairs,
            self.arena_bootstrap_iterations,
        )
        if any(value < 1 for value in positive):
            raise ValueError("self-improvement counts must be positive")
        if self.teacher_simulations < self.actor_simulations * 2:
            raise ValueError("teacher_simulations must be at least twice actor_simulations")
        if self.actor_temperature_moves < 0:
            raise ValueError("actor_temperature_moves must be non-negative")
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
        if not 0 < self.teacher_policy_mix <= 1 or not 0 <= self.teacher_value_mix <= 1:
            raise ValueError("teacher policy/value mixes are out of range")
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


_STAGES = ("initialized", "actor_saved", "reanalysed", "trained", "arena_complete", "complete")


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

    actor_openings, arena_openings, opening_manifest, legacy_single_opening = _opening_plan(config)
    state_path = workdir / "state.json"
    if state_path.exists():
        saved_state = json.loads(state_path.read_text(encoding="utf-8"))
        saved_opening_manifest = saved_state.get("opening_suite")
        if saved_opening_manifest is not None and saved_opening_manifest != opening_manifest:
            raise ValueError("opening suite changed within an existing improvement lineage")
        champion = Path(saved_state["champion"])
        generation = int(saved_state["generation"]) + 1
    else:
        champion = bootstrap_champion.resolve()
        generation = 1
    load_checkpoint(champion)
    generation_dir = workdir / f"generation-{generation:06d}"
    manifest_path = generation_dir / "manifest.json"
    actor_path = generation_dir / "actor.jsonl"
    deep_path = generation_dir / "deep.jsonl"
    candidate_path = generation_dir / "candidate"
    arena_replay_path = generation_dir / "arena.jsonl"
    expected_config = json.loads(json.dumps(asdict(config)))
    generation_interlock = (
        None
        if training_interlock is None
        else replace(training_interlock, generation=generation)
    )

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != 2:
            raise ValueError(
                "legacy generation manifests cannot resume under paired-opening promotion rules"
            )
        if manifest["config"] != expected_config:
            raise ValueError("an incomplete generation exists with a different configuration")
        if manifest.get("opening_suite") != opening_manifest:
            raise ValueError("opening suite changed after generation creation")
        if manifest["champion_before"] != str(champion):
            raise ValueError("generation champion does not match current state")
        if manifest["stage"] == "complete":
            _atomic_json(
                state_path,
                {
                    "generation": generation,
                    "champion": manifest["champion_after"],
                    "last_manifest": str(manifest_path),
                    "opening_suite": opening_manifest,
                },
            )
            return _result_from_manifest(manifest_path, manifest)
    else:
        manifest = {
            "schema": 2,
            "generation": generation,
            "status": "running",
            "stage": "initialized",
            "config": expected_config,
            "opening_suite": opening_manifest,
            "champion_before": str(champion),
            "champion_before_sha256": _checkpoint_digest(champion),
        }
        _atomic_json(manifest_path, manifest)

    actor_search = SearchConfig(
        simulations=config.actor_simulations,
        max_plies=config.max_plies,
        temperature_moves=config.actor_temperature_moves,
        max_tree_nodes=config.max_tree_nodes,
        memory_check_interval=config.memory_check_interval,
        intra_root_batch_size=config.intra_root_batch_size,
        intra_root_virtual_loss=config.intra_root_virtual_loss,
        max_evaluation_batch_size=config.max_evaluation_batch_size,
    )
    actor_offset = config.seed % len(actor_openings)
    initial_sfens = [
        actor_openings[(actor_offset + index) % len(actor_openings)].sfen
        for index in range(config.actor_games)
    ]
    if not _stage_at_least(manifest, "actor_saved"):
        actor_records = batched_self_play(
            champion,
            actor_search,
            games=config.actor_games,
            seed=config.seed,
            initial_sfens=initial_sfens,
            compute_interlock=generation_interlock,
        )
        _atomic_games(actor_path, actor_records)
        manifest.update(
            stage="actor_saved",
            actor_games=len(actor_records),
            actor_complete=sum(game.termination.value != "max_plies" for game in actor_records),
            actor_opening_sha256=[
                actor_openings[(actor_offset + index) % len(actor_openings)].sha256
                for index in range(config.actor_games)
            ],
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
        deep_games = [
            reanalyse_game(game, evaluator, actor_search, teacher_config, seed=config.seed + index)
            for index, game in enumerate(load_games(actor_path))
        ]
        _atomic_games(deep_path, deep_games)
        manifest.update(
            stage="reanalysed",
            deep_teacher_samples=sum(
                sample.teacher_policy is not None for game in deep_games for sample in game.samples
            ),
            teacher_compute_interlock=(
                {"enabled": False}
                if generation_interlock is None
                else generation_interlock.public_metadata()
            ),
        )
        _atomic_json(manifest_path, manifest)

    if not _stage_at_least(manifest, "trained"):
        champion_model, champion_step = load_checkpoint(champion)
        replay_paths = [deep_path, *(Path(path) for path in config.anchor_replays)]
        samples = position_samples(
            game for replay_path in replay_paths for game in load_games(replay_path)
        )
        metrics = train(
            champion_model,
            samples,
            steps=config.training_steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            seed=config.seed,
            require_teacher=True,
            teacher_policy_mix=config.teacher_policy_mix,
            teacher_value_mix=config.teacher_value_mix,
            legal_label_smoothing=config.legal_label_smoothing,
            curriculum_depth_ratio=config.curriculum_depth_ratio,
            minimum_teacher_policy_mix=config.minimum_teacher_policy_mix,
            maximum_gradient_norm=config.maximum_gradient_norm,
            maximum_probe_loss_ratio=config.maximum_probe_loss_ratio,
            training_interlock=generation_interlock,
        )
        save_checkpoint(
            champion_model,
            candidate_path,
            step=champion_step + config.training_steps,
        )
        manifest.update(
            stage="trained",
            candidate=str(candidate_path.resolve()),
            candidate_sha256=_checkpoint_digest(candidate_path),
            training=asdict(metrics),
            training_compute_interlock=(
                {"enabled": False}
                if generation_interlock is None
                else generation_interlock.public_metadata()
            ),
        )
        _atomic_json(manifest_path, manifest)

    if not _stage_at_least(manifest, "arena_complete"):
        arena_search = SearchConfig(
            simulations=config.arena_simulations,
            max_plies=config.max_plies,
            temperature_moves=0,
            temperature=0.0,
            dirichlet_fraction=0.0,
            max_tree_nodes=config.max_tree_nodes,
            memory_check_interval=config.memory_check_interval,
            intra_root_batch_size=config.intra_root_batch_size,
            intra_root_virtual_loss=config.intra_root_virtual_loss,
            max_evaluation_batch_size=config.max_evaluation_batch_size,
        )
        training_replay_paths = [
            deep_path,
            *(Path(path) for path in config.anchor_replays),
        ]
        arena_keys = {opening.normalized_key for opening in arena_openings}
        training_arena_overlap = tuple(
            sorted(arena_keys & _training_position_keys(training_replay_paths))
        )
        promotion_min_pairs = config.promotion_min_pairs
        if legacy_single_opening and config.promotion_min_games is not None:
            promotion_min_pairs = max(1, (config.promotion_min_games + 1) // 2)
        arena_seed = config.seed + 1_000_000
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
            promotion_eligible=not legacy_single_opening,
            training_arena_overlap_keys=training_arena_overlap,
            compute_interlock=generation_interlock,
        )
        _atomic_games(arena_replay_path, arena_records)
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
            arena_compute_interlock=(
                {"enabled": False}
                if generation_interlock is None
                else generation_interlock.public_metadata()
            ),
        )
        _atomic_json(manifest_path, manifest)

    summary = PairedArenaSummary.from_dict(manifest["arena"])
    champion_after = candidate_path.resolve() if summary.promoted else champion
    manifest.update(
        stage="complete",
        status="promoted" if summary.promoted else "rejected",
        promoted=summary.promoted,
        champion_after=str(champion_after),
    )
    _atomic_json(manifest_path, manifest)
    _atomic_json(
        state_path,
        {
            "generation": generation,
            "champion": str(champion_after),
            "last_manifest": str(manifest_path),
            "opening_suite": opening_manifest,
        },
    )
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
