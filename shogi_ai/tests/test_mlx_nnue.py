from __future__ import annotations

# ruff: noqa: E402 -- MLX-dependent imports must follow the platform skip.
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pytest

if platform.system() != "Darwin" or platform.machine() != "arm64":
    pytest.skip("MLX NNUE tests require Apple Silicon", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

import simajilord_shogi.mlx_nnue as mlx_nnue
from simajilord_shogi.mlx_nnue import (
    WrmLossParameters,
    _append_jsonl,
    _download_mlx_source_with_progress,
    _ensure_mlx_export_rounding_repair_receipt,
    _ensure_mlx_numeric_repair_receipt,
    _exact_sparse_ft_accumulation,
    _release_consumed_source_cache,
    _rounding_and_range_summary,
    audit_mlx_master_weight_ranges,
    wrm_probabilities,
)
from simajilord_shogi.nnue_training import (
    IncompleteShardDownloadError,
    PublicPsvShard,
    nagisa_wrm_contract,
)


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


def test_exact_sparse_ft_accumulation_has_int32_forward_and_scatter_vjp() -> None:
    ft = np.zeros((4, mlx_nnue.FT_OUT), dtype=np.float32)
    ft[0, :] = 4_097.0
    ft[1, :] = -4_096.0
    ft[2, :] = 3.0
    ft[3, :] = 7.0
    indices = np.full((2, mlx_nnue.MAX_ACTIVE), -1, dtype=np.int32)
    indices[0, :3] = (0, 1, 2)
    indices[1, :4] = (0, 0, 1, 3)
    nnz = mx.array([3, 4], dtype=mx.int32)
    mlx_indices = mx.array(indices)

    def total(values: mx.array) -> mx.array:
        return mx.sum(_exact_sparse_ft_accumulation(values, mlx_indices, nnz))

    loss, gradient = mx.value_and_grad(total)(mx.array(ft))
    output = _exact_sparse_ft_accumulation(mx.array(ft), mlx_indices, nnz)
    mx.eval(loss, gradient, output)

    observed = np.asarray(output)
    assert np.array_equal(observed[0], np.full(mlx_nnue.FT_OUT, 4.0))
    assert np.array_equal(observed[1], np.full(mlx_nnue.FT_OUT, 4_105.0))
    assert float(loss.item()) == (4.0 + 4_105.0) * mlx_nnue.FT_OUT
    expected_counts = np.asarray([3.0, 2.0, 1.0, 1.0], dtype=np.float32)
    assert np.array_equal(np.asarray(gradient)[:, 0], expected_counts)
    assert np.all(np.asarray(gradient) == expected_counts[:, None])


def test_legacy_fp16_failure_creates_one_content_bound_numeric_repair(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    checkpoint = root / "checkpoints" / "step"
    (root / "receipts").mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    model = checkpoint / "model.safetensors"
    optimizer = checkpoint / "optimizer.safetensors"
    model.write_bytes(b"model")
    optimizer.write_bytes(b"optimizer")
    manifest = {
        "optimizer": {"step": 61_032},
        "model_file": {"name": model.name, "sha256": hashlib.sha256(b"model").hexdigest()},
        "optimizer_file": {
            "name": optimizer.name,
            "sha256": hashlib.sha256(b"optimizer").hexdigest(),
        },
    }
    (checkpoint / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    plan = {
        "schema": mlx_nnue.MLX_BACKEND_PLAN_SCHEMA,
        "numerics": {
            "warmup_optimizer_steps": 1_024,
            "warmup_forward": "ft_float16_dense_float32",
            "deployment_graph": "tatara_yaneuraou_integer_equivalent_ste",
            "qat_feature_accumulation": mlx_nnue.LEGACY_QAT_FEATURE_ACCUMULATION,
        },
    }
    (root / "mlx-plan.json").write_text(json.dumps(plan), encoding="utf-8")
    state = {
        "failure": {
            "kind": "ValueError",
            "message": ("FP16 feature accumulation differs from the exported native network: {}"),
        }
    }

    first = _ensure_mlx_numeric_repair_receipt(
        root,
        plan=plan,
        state=state,
        checkpoint=checkpoint,
    )
    second = _ensure_mlx_numeric_repair_receipt(
        root,
        plan=plan,
        state={"failure": None},
        checkpoint=checkpoint,
    )

    assert first == second
    assert first is not None
    assert first["to_feature_accumulation"] == mlx_nnue.QAT_FEATURE_ACCUMULATION
    assert first["model_or_optimizer_values_changed"] is False


def test_legacy_rounding_failure_creates_content_bound_export_repair(tmp_path: Path) -> None:
    root = tmp_path / "run"
    checkpoint = root / "checkpoints" / "step"
    export = root / "exports" / "meteo-nagisa-v1-mlx-step-00000007"
    (root / "receipts").mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    export.mkdir(parents=True)
    model = checkpoint / "model.safetensors"
    optimizer = checkpoint / "optimizer.safetensors"
    model.write_bytes(b"model")
    optimizer.write_bytes(b"optimizer")
    manifest = {
        "optimizer": {"step": 7},
        "model_file": {"name": model.name, "sha256": hashlib.sha256(b"model").hexdigest()},
        "optimizer_file": {
            "name": optimizer.name,
            "sha256": hashlib.sha256(b"optimizer").hexdigest(),
        },
    }
    (checkpoint / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "mlx-plan.json").write_text(json.dumps({"numerics": {}}), encoding="utf-8")
    (export / "network.bin").write_bytes(b"network")
    (export / "nn.bin").write_bytes(b"yaneuraou")
    (export / "receipt.json").write_text(
        json.dumps(
            {
                "schema": "meteo-nagisa-mlx-yaneuraou-export-v1",
                "optimizer_step": 7,
                "checkpoint_model_sha256": hashlib.sha256(b"model").hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    plan = {"numerics": {}}
    state = {
        "failure": {
            "kind": "ValueError",
            "message": "MLX integer reference differs from the exported native network: {}",
        }
    }

    first = _ensure_mlx_export_rounding_repair_receipt(
        root,
        plan=plan,
        state=state,
        checkpoint=checkpoint,
    )
    second = _ensure_mlx_export_rounding_repair_receipt(
        root,
        plan=plan,
        state={"failure": None},
        checkpoint=checkpoint,
    )

    assert first == second
    assert first is not None
    assert first["to_export_rounding"] == mlx_nnue.TATARA_EXPORT_ROUNDING
    assert first["model_or_optimizer_values_changed"] is False
    assert first["source_checkpoint"]["optimizer_step"] == 7


def test_release_consumed_source_cache_requires_receipt_and_is_idempotent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    cache = root / "cache"
    receipts = root / "receipts"
    logs = root / "logs"
    cache.mkdir(parents=True)
    receipts.mkdir()
    logs.mkdir()
    shard = PublicPsvShard(
        filename="split_000.bin",
        byte_size=40,
        records=1,
        sha256="a" * 64,
        resolve_url="https://example.invalid/split_000.bin",
    )
    cached = cache / shard.filename
    cached.write_bytes(bytes(40))
    (receipts / f"source-{shard.sha256}.json").write_text(
        json.dumps(
            {
                "schema": "meteo-nagisa-source-probe-v1",
                "source": shard.to_dict(),
                "probe": {"bytes": 40, "records": 1, "sha256": shard.sha256},
            }
        ),
        encoding="utf-8",
    )

    assert _release_consumed_source_cache(root, shard) is True
    assert not cached.exists()
    assert _release_consumed_source_cache(root, shard) is False
    event = json.loads((logs / "mlx-training.jsonl").read_text(encoding="utf-8"))
    assert event["event"] == "source_cache_released"
    assert event["source"] == shard.filename
    assert event["cached_source_shards"] == 0


def test_weight_range_summary_separates_rounding_from_saturation() -> None:
    summary = _rounding_and_range_summary(
        np.asarray([-2.1, -2.0, -0.49, 0.49, 1.98, 2.1], dtype=np.float64),
        scale=64.0,
        minimum_raw=-128.0,
        maximum_raw=127.0,
    )

    assert summary["elements"] == 6
    assert summary["below_range_elements"] == 1
    assert summary["above_range_elements"] == 1
    assert summary["out_of_range_elements"] == 2
    assert summary["out_of_range_fraction"] == pytest.approx(2 / 6)
    assert summary["maximum_absolute_raw_rounding_or_clipping_residual"] > 1.0


def test_weight_range_summary_rejects_non_finite_or_empty_values() -> None:
    for values in (
        np.asarray([], dtype=np.float64),
        np.asarray([np.nan], dtype=np.float64),
    ):
        with pytest.raises(ValueError, match="finite non-empty"):
            _rounding_and_range_summary(
                values,
                scale=64.0,
                minimum_raw=-128.0,
                maximum_raw=127.0,
            )


def test_weight_range_summary_checks_the_rounded_integer_not_raw_float() -> None:
    summary = _rounding_and_range_summary(
        np.asarray([127.49 / 64.0, 127.51 / 64.0, -128.49 / 64.0, -128.51 / 64.0]),
        scale=64.0,
        minimum_raw=-128.0,
        maximum_raw=127.0,
    )

    assert summary["below_range_elements"] == 1
    assert summary["above_range_elements"] == 1
    assert summary["out_of_range_elements"] == 2


def test_master_weight_range_audit_is_read_only_and_combines_factorisers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model_path = checkpoint / "model.safetensors"
    model_bytes = b"synthetic-model-identity"
    model_path.write_bytes(model_bytes)
    manifest = {
        "schema": mlx_nnue.MLX_CHECKPOINT_SCHEMA,
        "complete": True,
        "model_file": {
            "name": model_path.name,
            "bytes": len(model_bytes),
            "sha256": hashlib.sha256(model_bytes).hexdigest(),
        },
    }
    manifest_path = checkpoint / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    arrays = {
        "ft_real": mx.zeros((45, 1)),
        "ft_virtual": mx.zeros((1, 1)),
        "ft_b": mx.zeros((1,)),
        "l1_w": mx.zeros((1, 1, 1)),
        "l1_b": mx.zeros((1, 1)),
        "l1f_w": mx.zeros((1, 1)),
        "l1f_b": mx.zeros((1,)),
        "l2_w": mx.zeros((1, 1, 1)),
        "l2_b": mx.zeros((1, 1)),
        "l3_w": mx.zeros((1, 1)),
        "l3_b": mx.zeros((1,)),
    }
    monkeypatch.setattr(mlx_nnue.mx, "load", lambda _path: arrays)
    before = {path.name: path.read_bytes() for path in sorted(checkpoint.iterdir())}

    result = audit_mlx_master_weight_ranges(checkpoint)

    after = {path.name: path.read_bytes() for path in sorted(checkpoint.iterdir())}
    assert result["read_only"] is True
    assert result["out_of_range_elements"] == 0
    assert result["audited_elements"] == 52
    assert set(result["tensors"]) == {
        "effective_ft_w",
        "ft_b",
        "merged_l1_w",
        "merged_l1_b",
        "l2_w",
        "l2_b",
        "l3_w",
        "l3_b",
    }
    assert before == after


def test_master_weight_range_audit_rejects_checkpoint_symlink(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    alias = tmp_path / "checkpoint-alias"
    alias.symlink_to(checkpoint, target_is_directory=True)

    with pytest.raises(ValueError, match="regular directory"):
        audit_mlx_master_weight_ranges(alias)


def test_mlx_download_retries_a_resumable_short_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    destination = root / "cache" / "split_003.bin"
    destination.parent.mkdir(parents=True)
    (root / "logs").mkdir()
    shard = PublicPsvShard(
        filename=destination.name,
        byte_size=40,
        records=1,
        sha256="a" * 64,
        resolve_url="https://example.invalid/split_003.bin",
    )
    attempts = 0
    delays: list[float] = []

    def flaky_download(*_args: object, **_kwargs: object) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise IncompleteShardDownloadError(
                "downloaded shard is incomplete: expected=40 observed=20"
            )
        destination.write_bytes(b"abcdefgh" * 5)
        return destination

    monkeypatch.setattr(mlx_nnue, "download_nagisa_shard", flaky_download)
    monkeypatch.setattr(mlx_nnue.time, "sleep", delays.append)

    result = _download_mlx_source_with_progress(root, shard, timeout_seconds=12.5)

    assert result == destination
    assert attempts == 2
    assert delays == [5.0]
    events = [
        json.loads(line) for line in (root / "logs" / "mlx-training.jsonl").read_text().splitlines()
    ]
    retries = [event for event in events if event["event"] == "source_download_retry"]
    assert len(retries) == 1
    assert retries[0]["error_type"] == "IncompleteShardDownloadError"
