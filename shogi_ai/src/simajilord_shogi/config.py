"""Validated configuration shared by inference, search, and training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelProfile = Literal["smoke", "development", "competition_v1", "competition_v2"]


def optional_max_plies(value: int) -> int | None:
    """Interpret zero or a negative CLI cutoff as an unbounded rules game."""

    return value if value > 0 else None


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Policy-value network shape.

    `competition_v1` uses the public dlshogi ResNet stem and head topology.
    Transformer-related fields are reserved for the verified competition-v2 implementation.
    """

    name: str
    residual_blocks: int
    channels: int
    policy_channels: int = 27
    value_channels: int = 32
    value_hidden: int = 256
    transformer_interval: int | None = None
    attention_heads: int = 8
    se_interval: int | None = None
    dlshogi_legacy: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("model name must not be empty")
        if self.residual_blocks < 1:
            raise ValueError("residual_blocks must be positive")
        if self.channels < 8:
            raise ValueError("channels must be at least 8")
        if self.policy_channels != 27:
            raise ValueError("dlshogi policy head must have exactly 27 channels")
        if self.dlshogi_legacy and self.value_channels != 27:
            raise ValueError("dlshogi legacy ResNet value head must have 27 channels")
        if self.value_channels < 1 or self.value_hidden < 1:
            raise ValueError("value head dimensions must be positive")
        if self.channels % self.attention_heads != 0:
            raise ValueError("channels must be divisible by attention_heads")
        for label, interval in (
            ("transformer_interval", self.transformer_interval),
            ("se_interval", self.se_interval),
        ):
            if interval is not None and interval < 1:
                raise ValueError(f"{label} must be positive when enabled")


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Neural MCTS and game-adjudication settings."""

    simulations: int = 800
    c_puct: float = 1.5
    policy_prior_floor: float = 1e-4
    root_min_visits: int = 1
    dirichlet_alpha: float = 0.15
    dirichlet_fraction: float = 0.25
    temperature_moves: int = 24
    temperature: float = 1.0
    max_plies: int | None = None
    resign_threshold: float | None = -0.98
    resign_min_ply: int = 60
    max_tree_nodes: int | None = None
    memory_check_interval: int = 1024
    # Batch=1 is the exact historical search. Raising it is opt-in because
    # simultaneous leaf reservations change traversal order at a fixed budget
    # and therefore require a paired playing-strength gate. A zero virtual loss
    # keeps the batched visit distribution substantially closer to sequential
    # PUCT; positive loss remains available for measured experiments.
    intra_root_batch_size: int = 1
    intra_root_virtual_loss: float = 0.0
    max_evaluation_batch_size: int = 256

    def __post_init__(self) -> None:
        if self.simulations < 1:
            raise ValueError("simulations must be positive")
        if self.c_puct <= 0:
            raise ValueError("c_puct must be positive")
        if not 0 <= self.policy_prior_floor < 1:
            raise ValueError("policy_prior_floor must be between 0 and 1")
        if self.root_min_visits < 0:
            raise ValueError("root_min_visits must be non-negative")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        if not 0 <= self.dirichlet_fraction <= 1:
            raise ValueError("dirichlet_fraction must be between 0 and 1")
        if self.temperature_moves < 0 or self.temperature < 0:
            raise ValueError("temperature settings must be non-negative")
        if self.max_plies is not None and self.max_plies < 1:
            raise ValueError("max_plies must be positive when configured")
        if self.resign_threshold is not None and not -1 <= self.resign_threshold <= 0:
            raise ValueError("resign_threshold must be between -1 and 0")
        if self.resign_min_ply < 0:
            raise ValueError("resign_min_ply must be non-negative")
        if self.max_tree_nodes is not None and self.max_tree_nodes < 1024:
            raise ValueError("max_tree_nodes must be at least 1024 when configured")
        if self.memory_check_interval < 1:
            raise ValueError("memory_check_interval must be positive")
        if self.intra_root_batch_size < 1:
            raise ValueError("intra_root_batch_size must be positive")
        if not 0 <= self.intra_root_virtual_loss <= 1:
            raise ValueError("intra_root_virtual_loss must be between zero and one")
        if self.max_evaluation_batch_size < 1:
            raise ValueError("max_evaluation_batch_size must be positive")


@dataclass(frozen=True, slots=True)
class ReanalysisConfig:
    """Asymmetric deep-teacher settings used to discover policy reversals."""

    teacher_simulation_multiplier: int = 8
    minimum_teacher_simulations: int = 6400
    uncertainty_threshold: float = 0.15
    reanalyse_fraction: float = 0.5
    reversal_priority: float = 4.0

    def __post_init__(self) -> None:
        if self.teacher_simulation_multiplier < 2:
            raise ValueError("teacher_simulation_multiplier must be at least two")
        if self.minimum_teacher_simulations < 1:
            raise ValueError("minimum_teacher_simulations must be positive")
        if not 0 <= self.uncertainty_threshold <= 1:
            raise ValueError("uncertainty_threshold must be between zero and one")
        if not 0 <= self.reanalyse_fraction <= 1:
            raise ValueError("reanalyse_fraction must be between zero and one")
        if self.reversal_priority < 1:
            raise ValueError("reversal_priority must be at least one")


def model_profile(profile: ModelProfile) -> ModelConfig:
    """Return a named, immutable model profile."""

    profiles: dict[ModelProfile, ModelConfig] = {
        "smoke": ModelConfig(
            name="smoke_2x32",
            residual_blocks=2,
            channels=32,
            value_channels=8,
            value_hidden=64,
            attention_heads=4,
        ),
        "development": ModelConfig(
            name="development_10x128",
            residual_blocks=10,
            channels=128,
            value_channels=16,
            value_hidden=128,
            attention_heads=8,
        ),
        "competition_v1": ModelConfig(
            name="dlshogi_resnet20x256",
            residual_blocks=20,
            channels=256,
            value_channels=27,
            value_hidden=256,
            attention_heads=8,
            dlshogi_legacy=True,
        ),
        "competition_v2": ModelConfig(
            name="hybrid_40x512",
            residual_blocks=40,
            channels=512,
            value_channels=32,
            value_hidden=512,
            transformer_interval=10,
            attention_heads=8,
            se_interval=5,
        ),
    }
    return profiles[profile]
