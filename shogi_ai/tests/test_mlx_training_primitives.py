from __future__ import annotations

# ruff: noqa: E402 -- MLX-dependent imports must follow the platform skip.
import platform

import numpy as np
import pytest

if platform.system() != "Darwin" or platform.machine() != "arm64":
    pytest.skip("MLX numerical primitive tests require Apple Silicon", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from simajilord_shogi.mlx_training_primitives import (
    DaemonPrefetch,
    QuantisationSchedule,
    fake_quantise_raw,
    ste_floor,
    ste_round,
)


def test_daemon_prefetch_returns_values_and_propagates_errors() -> None:
    success = DaemonPrefetch(name="test-prefetch-success", producer=lambda: 42)
    assert success.result(timeout=1.0) == 42
    assert success.done is True

    def fail() -> int:
        raise RuntimeError("producer failed")

    failure = DaemonPrefetch(name="test-prefetch-failure", producer=fail)
    with pytest.raises(RuntimeError, match="producer failed"):
        failure.result(timeout=1.0)


def test_quantisation_schedule_has_an_explicit_boundary() -> None:
    schedule = QuantisationSchedule(warmup_optimizer_steps=1_024)

    assert schedule.phase_for_completed_steps(0) == "mixed_precision_warmup"
    assert schedule.phase_for_completed_steps(1_023) == "mixed_precision_warmup"
    assert schedule.phase_for_completed_steps(1_024) == "quantisation_aware"
    with pytest.raises(ValueError, match="non-negative"):
        schedule.phase_for_completed_steps(-1)


def test_fake_quantise_raw_uses_integer_forward_values_and_ste_gradient() -> None:
    source = mx.array([-3.0, -0.51, -0.49, 0.49, 0.51, 3.0], dtype=mx.float32)

    def objective(values: mx.array) -> mx.array:
        return mx.sum(fake_quantise_raw(values, scale=2.0, minimum=-4.0, maximum=4.0))

    quantised = fake_quantise_raw(source, scale=2.0, minimum=-4.0, maximum=4.0)
    gradient = mx.grad(objective)(source)
    mx.eval(quantised, gradient)

    np.testing.assert_array_equal(np.asarray(quantised), [-4.0, -1.0, -1.0, 1.0, 1.0, 4.0])
    np.testing.assert_array_equal(np.asarray(gradient), [0.0, 2.0, 2.0, 2.0, 2.0, 0.0])


def test_round_and_floor_forward_values_keep_identity_gradients() -> None:
    source = mx.array([0.25, 0.75, 1.25], dtype=mx.float32)
    rounded = ste_round(source)
    floored = ste_floor(source)
    round_gradient = mx.grad(lambda values: mx.sum(ste_round(values)))(source)
    floor_gradient = mx.grad(lambda values: mx.sum(ste_floor(values)))(source)
    mx.eval(rounded, floored, round_gradient, floor_gradient)

    np.testing.assert_array_equal(np.asarray(rounded), [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(np.asarray(floored), [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(np.asarray(round_gradient), [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(np.asarray(floor_gradient), [1.0, 1.0, 1.0])
