"""MLX policy-value ResNet with dlshogi-compatible input and output shapes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from rsshogi.core import Board
from rsshogi.policy import MOVE_LABEL_COUNT, move_label

from .config import ModelConfig
from .encoding import (
    BOARD_SIZE,
    FEATURES1_NUM,
    FEATURES2_NUM,
    HISTORY_INPUT_CHANNELS,
    HistoryInput,
    board_from_history_input,
    combined_features_from_board,
    combined_features_with_history,
)
from .evaluator import Evaluation


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, squeeze_excitation: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(channels)
        self.se = SqueezeExcitation(channels) if squeeze_excitation else None

    def __call__(self, inputs: mx.array) -> mx.array:
        hidden = nn.relu(self.bn1(self.conv1(inputs)))
        hidden = self.bn2(self.conv2(hidden))
        if self.se is not None:
            hidden = self.se(hidden)
        return nn.relu(inputs + hidden)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.reduce = nn.Linear(channels, hidden)
        self.expand = nn.Linear(hidden, channels)

    def __call__(self, inputs: mx.array) -> mx.array:
        pooled = mx.mean(inputs, axis=(1, 2))
        gates = mx.sigmoid(self.expand(nn.silu(self.reduce(pooled))))
        return inputs * gates[:, None, None, :]


class SpatialTransformerBlock(nn.Module):
    """81-square attention with learned absolute and relative position terms."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        hidden = channels * 2
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiHeadAttention(channels, heads, bias=False)
        self.attention_gate = mx.zeros((channels,))
        self.absolute_position = mx.zeros((1, BOARD_SIZE * BOARD_SIZE, channels))
        self.relative_bias = mx.zeros((heads, BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE))
        self.norm2 = nn.LayerNorm(channels)
        self.gate_projection = nn.Linear(channels, hidden, bias=False)
        self.value_projection = nn.Linear(channels, hidden, bias=False)
        self.output_projection = nn.Linear(hidden, channels, bias=False)

    def __call__(self, inputs: mx.array) -> mx.array:
        batch = inputs.shape[0]
        sequence = mx.reshape(inputs, (batch, BOARD_SIZE * BOARD_SIZE, inputs.shape[-1]))
        positioned = sequence + self.absolute_position
        normalized = self.norm1(positioned)
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            self.relative_bias[None, :, :, :],
        )
        sequence = sequence + mx.sigmoid(self.attention_gate)[None, None, :] * attended
        normalized = self.norm2(sequence)
        swiglu = nn.silu(self.gate_projection(normalized)) * self.value_projection(normalized)
        sequence = sequence + self.output_projection(swiglu)
        return mx.reshape(sequence, inputs.shape)


@dataclass(frozen=True, slots=True)
class CanonicalModelOutput:
    """All canonical-target-contract-v2 heads from one shared trunk."""

    policy_play: mx.array
    value_play: mx.array
    policy_teachers: mx.array
    value_teachers: mx.array
    wdl_play_logits: mx.array
    uncertainty: mx.array


class PolicyValueResNet(nn.Module):
    """Official-shape dlshogi ResNet or Meteo hybrid; tensors enter as NCHW."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        input_channels = FEATURES1_NUM + FEATURES2_NUM
        self.stem = (
            None
            if config.dlshogi_legacy
            else nn.Conv2d(input_channels, config.channels, 3, padding=1, bias=False)
        )
        self.stem_feature_3x3 = (
            nn.Conv2d(FEATURES1_NUM, config.channels, 3, padding=1, bias=False)
            if config.dlshogi_legacy
            else None
        )
        self.stem_feature_1x1 = (
            nn.Conv2d(FEATURES1_NUM, config.channels, 1, bias=False)
            if config.dlshogi_legacy
            else None
        )
        self.stem_hand_1x1 = (
            nn.Conv2d(FEATURES2_NUM, config.channels, 1, bias=False)
            if config.dlshogi_legacy
            else None
        )
        self.history_stem = (
            nn.Conv2d(
                HISTORY_INPUT_CHANNELS,
                config.channels,
                3,
                padding=1,
                bias=False,
            )
            if config.history_input_version == 2
            else None
        )
        if self.history_stem is not None:
            self.history_stem.weight = mx.zeros_like(self.history_stem.weight)
        self.stem_bn = nn.BatchNorm(config.channels)
        blocks: list[nn.Module] = []
        for index in range(1, config.residual_blocks + 1):
            if config.transformer_interval and index % config.transformer_interval == 0:
                blocks.append(SpatialTransformerBlock(config.channels, config.attention_heads))
            else:
                blocks.append(
                    ResidualBlock(
                        config.channels,
                        squeeze_excitation=bool(
                            config.se_interval and index % config.se_interval == 0
                        ),
                    )
                )
        self.blocks = blocks
        self.policy = nn.Conv2d(
            config.channels,
            config.policy_channels,
            1,
            bias=not config.dlshogi_legacy,
        )
        self.policy_bias = mx.zeros((MOVE_LABEL_COUNT,)) if config.dlshogi_legacy else None
        self.value_conv = nn.Conv2d(config.channels, config.value_channels, 1, bias=False)
        self.value_bn = nn.BatchNorm(config.value_channels)
        self.value_hidden = nn.Linear(
            BOARD_SIZE * BOARD_SIZE * config.value_channels, config.value_hidden
        )
        self.value_output = nn.Linear(config.value_hidden, 1)
        self.teacher_policies: list[nn.Conv2d] | None = (
            [
                nn.Conv2d(
                    config.channels,
                    config.policy_channels,
                    1,
                    bias=not config.dlshogi_legacy,
                )
                for _ in range(2)
            ]
            if config.canonical_head_version == 2
            else None
        )
        self.teacher_policy_biases: list[mx.array] | None = (
            [mx.zeros((MOVE_LABEL_COUNT,)) for _ in range(2)]
            if config.canonical_head_version == 2 and config.dlshogi_legacy
            else None
        )
        self.teacher_value_outputs: list[nn.Linear] | None = (
            [nn.Linear(config.value_hidden, 1) for _ in range(2)]
            if config.canonical_head_version == 2
            else None
        )
        self.wdl_play_output = (
            nn.Linear(config.value_hidden, 3)
            if config.canonical_head_version == 2
            else None
        )
        self.uncertainty_output = (
            nn.Linear(config.value_hidden, 1)
            if config.canonical_head_version == 2
            else None
        )
        if config.canonical_head_version == 2:
            self.initialize_canonical_heads_from_play()

    @staticmethod
    def _copy_linear(source: nn.Linear, destination: nn.Linear) -> None:
        destination.weight = mx.array(source.weight)
        if source.bias is not None and destination.bias is not None:
            destination.bias = mx.array(source.bias)

    def initialize_canonical_heads_from_play(self) -> None:
        """Copy old play heads into both teacher heads during an explicit upgrade."""

        if self.config.canonical_head_version != 2:
            raise ValueError("canonical teacher heads are unavailable in a v1 model")
        if self.teacher_policies is None or self.teacher_value_outputs is None:
            raise AssertionError("canonical teacher heads were not initialized")
        for head in self.teacher_policies:
            head.weight = mx.array(self.policy.weight)
            if self.policy.bias is not None and head.bias is not None:
                head.bias = mx.array(self.policy.bias)
        if self.policy_bias is not None:
            self.teacher_policy_biases = [
                mx.array(self.policy_bias) for _head in self.teacher_policies
            ]
        for head in self.teacher_value_outputs:
            self._copy_linear(self.value_output, head)
        if self.wdl_play_output is None or self.uncertainty_output is None:
            raise AssertionError("canonical scalar heads were not initialized")
        self.wdl_play_output.weight = mx.zeros_like(self.wdl_play_output.weight)
        self.wdl_play_output.bias = mx.zeros_like(self.wdl_play_output.bias)
        self.uncertainty_output.weight = mx.zeros_like(self.uncertainty_output.weight)
        self.uncertainty_output.bias = mx.zeros_like(self.uncertainty_output.bias)

    def _trunk(self, inputs: mx.array) -> mx.array:
        expected_channels = FEATURES1_NUM + FEATURES2_NUM + (
            HISTORY_INPUT_CHANNELS if self.config.history_input_version == 2 else 0
        )
        if inputs.ndim != 4 or inputs.shape[1] != expected_channels:
            raise ValueError(
                f"model expected NCHW input with {expected_channels} channels, got {inputs.shape}"
            )
        board_inputs = inputs[:, : FEATURES1_NUM + FEATURES2_NUM]
        history_inputs = inputs[:, FEATURES1_NUM + FEATURES2_NUM :]
        history_hidden: mx.array | None = None
        if self.config.history_input_version == 2:
            if self.history_stem is None:
                raise AssertionError("history-input-v2 stem is not initialized")
            history_hidden = self.history_stem(mx.transpose(history_inputs, (0, 2, 3, 1)))
        if self.config.dlshogi_legacy:
            if (
                self.stem_feature_3x3 is None
                or self.stem_feature_1x1 is None
                or self.stem_hand_1x1 is None
            ):
                raise AssertionError("dlshogi legacy stems are not initialized")
            features = mx.transpose(board_inputs[:, :FEATURES1_NUM], (0, 2, 3, 1))
            hand = mx.transpose(board_inputs[:, FEATURES1_NUM:], (0, 2, 3, 1))
            hidden = (
                self.stem_feature_3x3(features)
                + self.stem_feature_1x1(features)
                + self.stem_hand_1x1(hand)
            )
        else:
            if self.stem is None:
                raise AssertionError("Meteo hybrid stem is not initialized")
            hidden = self.stem(mx.transpose(board_inputs, (0, 2, 3, 1)))
        if history_hidden is not None:
            hidden = hidden + history_hidden
        hidden = nn.relu(self.stem_bn(hidden))
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def _policy_logits(
        self,
        hidden: mx.array,
        head: nn.Conv2d,
        bias: mx.array | None,
        *,
        batch_size: int,
    ) -> mx.array:
        policy_nhwc = head(hidden)
        policy = mx.reshape(
            mx.transpose(policy_nhwc, (0, 3, 1, 2)),
            (batch_size, -1),
        )
        return policy if bias is None else policy + bias

    def _value_features(self, hidden: mx.array, *, batch_size: int) -> mx.array:
        value = nn.relu(self.value_bn(self.value_conv(hidden)))
        value = mx.reshape(value, (batch_size, -1))
        return nn.relu(self.value_hidden(value))

    def __call__(self, inputs: mx.array) -> tuple[mx.array, mx.array]:
        hidden = self._trunk(inputs)
        policy = self._policy_logits(
            hidden,
            self.policy,
            self.policy_bias,
            batch_size=inputs.shape[0],
        )
        value_features = self._value_features(hidden, batch_size=inputs.shape[0])
        value = mx.tanh(self.value_output(value_features))
        return policy, value

    def forward_canonical(self, inputs: mx.array) -> CanonicalModelOutput:
        """Return independent teacher heads plus guarded play/uncertainty heads."""

        if self.config.canonical_head_version != 2:
            raise ValueError("canonical-target-contract-v2 requires canonical head v2")
        if (
            self.teacher_policies is None
            or len(self.teacher_policies) != 2
            or self.teacher_value_outputs is None
            or len(self.teacher_value_outputs) != 2
            or self.wdl_play_output is None
            or self.uncertainty_output is None
        ):
            raise AssertionError("canonical-target-contract-v2 heads are incomplete")
        hidden = self._trunk(inputs)
        policy_play = self._policy_logits(
            hidden,
            self.policy,
            self.policy_bias,
            batch_size=inputs.shape[0],
        )
        # Auxiliary teacher heads intentionally receive a detached shared
        # representation.  Otherwise an unresolved teacher disagreement could
        # still alter the actual play policy through the shared trunk even when
        # the play-head loss is masked.
        teacher_hidden = mx.stop_gradient(hidden)
        teacher_policies = mx.stack(
            [
                self._policy_logits(
                    teacher_hidden,
                    head,
                    (
                        self.teacher_policy_biases[index]
                        if self.teacher_policy_biases is not None
                        else None
                    ),
                    batch_size=inputs.shape[0],
                )
                for index, head in enumerate(self.teacher_policies)
            ],
            axis=1,
        )
        value_features = self._value_features(hidden, batch_size=inputs.shape[0])
        value_play = mx.tanh(self.value_output(value_features))
        auxiliary_value_features = mx.stop_gradient(value_features)
        value_teachers = mx.concatenate(
            [
                mx.tanh(head(auxiliary_value_features))
                for head in self.teacher_value_outputs
            ],
            axis=1,
        )
        return CanonicalModelOutput(
            policy_play=policy_play,
            value_play=value_play,
            policy_teachers=teacher_policies,
            value_teachers=value_teachers,
            wdl_play_logits=self.wdl_play_output(value_features),
            uncertainty=mx.sigmoid(self.uncertainty_output(auxiliary_value_features)),
        )


class MLXEvaluator:
    def __init__(self, model: PolicyValueResNet) -> None:
        self.model = model
        self.model.eval()

    def evaluate(self, board: Board) -> Evaluation:
        return self.evaluate_batch([board])[0]

    def evaluate_batch(self, boards: list[Board]) -> list[Evaluation]:
        if not boards:
            return []
        encoded = [
            combined_features_from_board(
                board,
                history_input_version=self.model.config.history_input_version,
            )
            for board in boards
        ]
        return self._evaluate_encoded(boards, encoded)

    def evaluate_history_batch(self, histories: list[HistoryInput]) -> list[Evaluation]:
        """Evaluate exact recorded prefixes under history-input-v2."""

        if not histories:
            return []
        if self.model.config.history_input_version != 2:
            raise ValueError("exact-history evaluation requires a history-input-v2 checkpoint")
        boards = [board_from_history_input(history) for history in histories]
        encoded = [combined_features_with_history(history) for history in histories]
        return self._evaluate_encoded(boards, encoded)

    def _evaluate_encoded(
        self,
        boards: list[Board],
        encoded: list[np.ndarray[Any, np.dtype[np.float32]]],
    ) -> list[Evaluation]:
        inputs = mx.array(np.stack(encoded))
        logits, values = self.model(inputs)
        probabilities = np.asarray(mx.softmax(logits, axis=1), dtype=np.float64)
        value_array = np.asarray(values, dtype=np.float64)
        results: list[Evaluation] = []
        for row, board in enumerate(boards):
            policy = {
                move.to_usi(): float(probabilities[row, move_label(move, board.turn)])
                for move in board.legal_moves()
            }
            total = sum(policy.values())
            if total > 0:
                policy = {move: probability / total for move, probability in policy.items()}
            results.append(Evaluation(policy=policy, value=float(value_array[row, 0])))
        return results


def upgrade_model_to_canonical_v2(model: PolicyValueResNet) -> PolicyValueResNet:
    """Warm-start v2 while preserving every legacy play prediction exactly.

    The old 119-plane trunk and play heads are copied by name.  The additional
    history adapter is zero, both teacher heads start as exact copies of the
    play heads, and WDL/uncertainty start uninformative.  Optimizer state cannot
    be reused because the parameter tree changes.
    """

    if model.config.canonical_head_version == 2:
        if model.config.history_input_version != 2:
            raise AssertionError("canonical v2 model has an incompatible history input")
        return model
    upgraded = PolicyValueResNet(
        replace(
            model.config,
            history_input_version=2,
            canonical_head_version=2,
        )
    )
    legacy_weights = list(tree_flatten(model.parameters()))
    upgraded.load_weights(legacy_weights, strict=False)
    upgraded.initialize_canonical_heads_from_play()
    mx.eval(upgraded.parameters())
    return upgraded


def parameter_count(model: nn.Module) -> int:
    def count(value: Any) -> int:
        if isinstance(value, mx.array):
            return int(value.size)
        if isinstance(value, dict):
            return sum(count(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(count(item) for item in value)
        return 0

    return count(model.parameters())


def assert_model_shapes(model: PolicyValueResNet, batch_size: int = 2) -> None:
    input_channels = FEATURES1_NUM + FEATURES2_NUM + (
        HISTORY_INPUT_CHANNELS if model.config.history_input_version == 2 else 0
    )
    inputs = mx.zeros((batch_size, input_channels, BOARD_SIZE, BOARD_SIZE))
    policy, value = model(inputs)
    mx.eval(policy, value)
    if policy.shape != (batch_size, MOVE_LABEL_COUNT):
        raise AssertionError(f"unexpected policy shape: {policy.shape}")
    if value.shape != (batch_size, 1):
        raise AssertionError(f"unexpected value shape: {value.shape}")
    if model.config.canonical_head_version == 2:
        canonical = model.forward_canonical(inputs)
        mx.eval(
            canonical.policy_teachers,
            canonical.value_teachers,
            canonical.wdl_play_logits,
            canonical.uncertainty,
        )
        if canonical.policy_teachers.shape != (batch_size, 2, MOVE_LABEL_COUNT):
            raise AssertionError(
                f"unexpected canonical policy shape: {canonical.policy_teachers.shape}"
            )
        if canonical.value_teachers.shape != (batch_size, 2):
            raise AssertionError(
                f"unexpected canonical value shape: {canonical.value_teachers.shape}"
            )
        if canonical.wdl_play_logits.shape != (batch_size, 3):
            raise AssertionError(f"unexpected WDL shape: {canonical.wdl_play_logits.shape}")
        if canonical.uncertainty.shape != (batch_size, 1):
            raise AssertionError(
                f"unexpected uncertainty shape: {canonical.uncertainty.shape}"
            )
