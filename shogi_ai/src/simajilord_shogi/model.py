"""MLX policy-value ResNet with dlshogi-compatible input and output shapes."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from rsshogi.core import Board
from rsshogi.policy import MOVE_LABEL_COUNT, move_label

from .config import ModelConfig
from .encoding import BOARD_SIZE, FEATURES1_NUM, FEATURES2_NUM, encode_board
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

    def __call__(self, inputs: mx.array) -> tuple[mx.array, mx.array]:
        if self.config.dlshogi_legacy:
            if (
                self.stem_feature_3x3 is None
                or self.stem_feature_1x1 is None
                or self.stem_hand_1x1 is None
            ):
                raise AssertionError("dlshogi legacy stems are not initialized")
            features = mx.transpose(inputs[:, :FEATURES1_NUM], (0, 2, 3, 1))
            hand = mx.transpose(inputs[:, FEATURES1_NUM:], (0, 2, 3, 1))
            hidden = (
                self.stem_feature_3x3(features)
                + self.stem_feature_1x1(features)
                + self.stem_hand_1x1(hand)
            )
        else:
            if self.stem is None:
                raise AssertionError("Meteo hybrid stem is not initialized")
            hidden = self.stem(mx.transpose(inputs, (0, 2, 3, 1)))
        hidden = nn.relu(self.stem_bn(hidden))
        for block in self.blocks:
            hidden = block(hidden)
        policy_nhwc = self.policy(hidden)
        policy = mx.reshape(mx.transpose(policy_nhwc, (0, 3, 1, 2)), (inputs.shape[0], -1))
        if self.policy_bias is not None:
            policy = policy + self.policy_bias
        value = nn.relu(self.value_bn(self.value_conv(hidden)))
        value = mx.reshape(value, (inputs.shape[0], -1))
        value = mx.tanh(self.value_output(nn.relu(self.value_hidden(value))))
        return policy, value


class MLXEvaluator:
    def __init__(self, model: PolicyValueResNet) -> None:
        self.model = model
        self.model.eval()

    def evaluate(self, board: Board) -> Evaluation:
        return self.evaluate_batch([board])[0]

    def evaluate_batch(self, boards: list[Board]) -> list[Evaluation]:
        if not boards:
            return []
        encoded = []
        for board in boards:
            feature1, feature2 = encode_board(board)
            encoded.append(np.concatenate((feature1, feature2), axis=0))
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
    inputs = mx.zeros((batch_size, FEATURES1_NUM + FEATURES2_NUM, BOARD_SIZE, BOARD_SIZE))
    policy, value = model(inputs)
    mx.eval(policy, value)
    if policy.shape != (batch_size, MOVE_LABEL_COUNT):
        raise AssertionError(f"unexpected policy shape: {policy.shape}")
    if value.shape != (batch_size, 1):
        raise AssertionError(f"unexpected value shape: {value.shape}")
