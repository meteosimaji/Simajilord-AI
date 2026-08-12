from __future__ import annotations

# ruff: noqa: E402 -- MLX-dependent imports must follow the platform skip.
import platform
from pathlib import Path

import numpy as np
import pytest

if platform.system() != "Darwin" or platform.machine() != "arm64":
    pytest.skip("MLX NNUE tests require Apple Silicon", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from simajilord_shogi.mlx_nnue import WrmLossParameters, _append_jsonl, wrm_probabilities
from simajilord_shogi.nnue_training import nagisa_wrm_contract


def _parameters() -> WrmLossParameters:
    return WrmLossParameters.from_training(
        {"score_scale": 600.0, "loss": nagisa_wrm_contract(600.0)}
    )


def test_wrm_target_is_the_original_sigmoid_distribution() -> None:
    parameters = _parameters()
    scores_np = np.array([-32767, -1200, -600, -1, 0, 1, 600, 1200, 32767], dtype=np.float32)
    scores = mx.array(scores_np)
    outputs = mx.zeros_like(scores)

    _, targets = wrm_probabilities(outputs, scores, parameters=parameters)
    mx.eval(targets)
    expected = 1.0 / (1.0 + np.exp(-scores_np / np.float32(600.0)))

    assert np.max(np.abs(np.asarray(targets) - expected)) <= 2.0e-7


def test_wrm_network_optimum_and_yaneuraou_scale_are_aligned() -> None:
    parameters = _parameters()
    scores_np = np.array([-32000, -1200, -100, 0, 100, 1200, 32000], dtype=np.float32)
    scores = mx.array(scores_np)
    outputs = scores / parameters.nnue2score

    predictions, targets = wrm_probabilities(outputs, scores, parameters=parameters)
    mx.eval(predictions, targets)

    assert parameters.nnue2score == 8128.0 / 16.0 == 508.0
    assert np.max(np.abs(np.asarray(predictions) - np.asarray(targets))) <= 2.0e-7


def test_wrm_plan_rejects_old_cp_output_loss_contract() -> None:
    with pytest.raises(ValueError, match="scale-aligned"):
        WrmLossParameters.from_training(
            {
                "score_scale": 600.0,
                "loss": {
                    "kind": "plain_sigmoid_mse",
                    "network_optimum": "score",
                },
            }
        )


def test_metric_writer_emits_one_json_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"

    _append_jsonl(path, {"event": "first", "value": 1})
    _append_jsonl(path, {"event": "second", "value": 2})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == '{"event":"first","value":1}'
    assert lines[1] == '{"event":"second","value":2}'
