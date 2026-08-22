from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypedDict, cast

import mlx.core as mx
import numpy as np
import pytest
from rsshogi.core import Board, Move
from rsshogi.numpy import PackedSfenValue

import simajilord_shogi.incremental_mlx as incremental_mlx
from simajilord_shogi.artifact_provenance import sha256_file as provenance_sha256_file
from simajilord_shogi.distillation_targets import (
    SOUJOU_TSEC7_SCORER_ID,
    SUISHO11PLUS_SCORER_ID,
)
from simajilord_shogi.incremental_mlx import (
    INCREMENTAL_MLX_CHECKPOINT_RECEIPT_SCHEMA,
    INCREMENTAL_MLX_PLAN_SCHEMA,
    INCREMENTAL_MLX_STATE_SCHEMA,
    incremental_mlx_status,
    load_incremental_mlx_plan,
    load_incremental_mlx_state,
    prepare_incremental_mlx_run,
    run_incremental_mlx,
    weighted_batch_source,
)
from simajilord_shogi.incremental_value_replay import IncrementalValueSplit
from simajilord_shogi.mlx_nnue import (
    MLX_BACKEND_PLAN_SCHEMA,
    MLX_CHECKPOINT_SCHEMA,
    QAT_FEATURE_ACCUMULATION,
)
from simajilord_shogi.nnue_training import DEFAULT_NET_ID, nagisa_wrm_contract


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _psv(path: Path, records: int, *, first_move: str | None = None) -> None:
    values = np.zeros(records, dtype=PackedSfenValue)
    board = Board()
    if first_move is not None:
        board.apply_move(Move.from_usi(first_move))
    for index in range(records):
        values[index]["sfen"] = np.frombuffer(board.to_packed_sfen(), dtype=np.uint8)
        values[index]["score"] = index * 10
        values[index]["move"] = 0
        values[index]["game_ply"] = index
        values[index]["game_result"] = 0
    path.write_bytes(values.tobytes())


def _replay_source(
    root: Path,
    name: str,
    *,
    position_ids: tuple[str, ...],
    split_id: str,
    first_move: str | None,
) -> tuple[Path, Path, Path]:
    directory = root / name
    directory.mkdir()
    psv = directory / "data.psv"
    split_path = directory / "split.json"
    receipt_path = directory / "receipt.json"
    _psv(psv, len(position_ids), first_move=first_move)
    split = IncrementalValueSplit(
        split_id=split_id,
        source_games_sha256=_digest(f"source-games-{name}"),
        position_ids=(*position_ids, _digest(f"unused-split-member-{name}")),
    )
    _json(split_path, split.to_dict())
    labels_path = directory / "labels.jsonl"
    labels_path.write_text(
        "".join(
            json.dumps(
                {
                    "position_id": position_id,
                    "kind": "anchor_search_exact",
                    "scorer_id": (
                        SUISHO11PLUS_SCORER_ID if name == "broad" else SOUJOU_TSEC7_SCORER_ID
                    ),
                    "requested_nodes": 1_000_000,
                },
                sort_keys=True,
            )
            + "\n"
            for position_id in position_ids
        ),
        encoding="utf-8",
    )
    receipt = {
        "schema": "meteo-incremental-value-replay-v1",
        "mode": "value_only_nnue_incremental_exact_labels",
        "complete": True,
        "records": len(position_ids),
        "policy_targets": 0,
        "cross_teacher_value_average": False,
        "bounds_written_as_point_targets": False,
        "unresolved_positions_written": False,
        "history_dependent_positions_written": False,
        "publication_allowed": False,
        "target_contract": {
            "schema": incremental_mlx.SCALAR_TARGET_CONTRACT_SCHEMA,
            "label_space": "calibrated_anchor_centipawn_v1",
            "score_perspective": "side_to_move",
            "target_anchor_scorer_id": SOUJOU_TSEC7_SCORER_ID,
            "target_anchor_identity_sha256": _digest("soujou-anchor-identity"),
            "calibration_sha256": _digest("meteo-soujou-calibration"),
            "score_scale": 600.0,
            "yaneuraou_fv_scale": 16,
            "source_label_space": (
                "dl_suisho_raw_centipawn_v1"
                if name == "broad"
                else "calibrated_anchor_centipawn_v1"
            ),
            "conversion_kind": (
                "calibrated_to_target" if name == "broad" else "identity_same_scale"
            ),
            "conversion_artifact_sha256": (
                _digest("dl-suisho-to-soujou-calibration")
                if name == "broad"
                else _digest("meteo-soujou-calibration")
            ),
        },
        "psv": {
            "sha256": _sha256(psv),
            "bytes": psv.stat().st_size,
        },
        "split": {
            "receipt_sha256": _sha256(split_path),
            "id": split_id,
        },
        "labels": {
            "file": labels_path.name,
            "sha256": _sha256(labels_path),
            "bytes": labels_path.stat().st_size,
        },
    }
    _json(receipt_path, receipt)
    return psv, receipt_path, split_path


def _base_run(root: Path, *, status: str = "complete") -> tuple[Path, Path]:
    root.mkdir()
    for child in ("checkpoints", "private", "dependencies"):
        (root / child).mkdir()
    checkpoint = root / "checkpoints" / "meteo-nagisa-mlx-step-00002000"
    checkpoint.mkdir()
    model = checkpoint / "model.safetensors"
    optimizer = checkpoint / "optimizer.safetensors"
    model.write_bytes(b"fake model master weights")
    optimizer.write_bytes(b"fake Ranger state at step 2000")
    manifest = {
        "schema": MLX_CHECKPOINT_SCHEMA,
        "complete": True,
        "optimizer": {"step": 2_000},
        "numerics": {
            "next_update_phase": "quantisation_aware",
            "warmup_optimizer_steps": 1_024,
            "deployment_graph": "tatara_yaneuraou_integer_equivalent_ste",
        },
        "model_file": {
            "name": model.name,
            "bytes": model.stat().st_size,
            "sha256": _sha256(model),
        },
        "optimizer_file": {
            "name": optimizer.name,
            "bytes": optimizer.stat().st_size,
            "sha256": _sha256(optimizer),
        },
    }
    _json(checkpoint / "manifest.json", manifest)
    common = {
        "schema": "fixture-compatible-source-plan",
        "training": {
            "score_scale": 600.0,
            "loss": nagisa_wrm_contract(600.0),
            "yaneuraou_fv_scale": 16,
        },
    }
    _json(root / "plan.json", common)
    native = root / "dependencies" / "fake-native"
    native.write_bytes(b"fake native binary")
    native.chmod(0o700)
    mlx_plan = {
        "schema": MLX_BACKEND_PLAN_SCHEMA,
        "source_plan_sha256": _sha256(root / "plan.json"),
        "numerics": {
            "warmup_optimizer_steps": 1_024,
            "warmup_forward": "ft_float16_dense_float32",
            "deployment_graph": "tatara_yaneuraou_integer_equivalent_ste",
            "qat_feature_accumulation": QAT_FEATURE_ACCUMULATION,
        },
        "execution": {
            "batch_size": 2,
            "native_decode_threads": 1,
            "native_prefetch_batches": 2,
            "mlx_memory_limit_bytes": 0,
            "mlx_cache_limit_bytes": 0,
        },
        "native": {
            "binary": {
                "path": str(native),
                "sha256": _sha256(native),
            }
        },
    }
    _json(root / "mlx-plan.json", mlx_plan)
    (root / "private" / "progress.bin").write_bytes(b"fixture progress")
    _json(
        root / "mlx-state.json",
        {
            "schema": "meteo-nagisa-mlx-backend-state-v1",
            "status": status,
            "latest_checkpoint": str(checkpoint),
            "durable_optimizer_step": 2_000,
        },
    )
    return root, checkpoint


class _SourceArguments(TypedDict):
    broad_anchor_psv: Path
    broad_anchor_receipt: Path
    broad_anchor_split_receipt: Path
    hard_exact_psv: Path
    hard_exact_receipt: Path
    hard_exact_split_receipt: Path
    calibration_psv: Path
    calibration_receipt: Path
    calibration_split_receipt: Path


def _source_arguments(tmp_path: Path) -> _SourceArguments:
    broad = _replay_source(
        tmp_path,
        "broad",
        position_ids=tuple(_digest(f"broad-{index}") for index in range(4)),
        split_id="train",
        first_move=None,
    )
    hard = _replay_source(
        tmp_path,
        "hard",
        position_ids=tuple(_digest(f"hard-{index}") for index in range(4)),
        split_id="train",
        first_move="7g7f",
    )
    calibration = _replay_source(
        tmp_path,
        "calibration",
        position_ids=tuple(_digest(f"calibration-{index}") for index in range(4)),
        split_id="calibration",
        first_move="2g2f",
    )
    return _SourceArguments(
        broad_anchor_psv=broad[0],
        broad_anchor_receipt=broad[1],
        broad_anchor_split_receipt=broad[2],
        hard_exact_psv=hard[0],
        hard_exact_receipt=hard[1],
        hard_exact_split_receipt=hard[2],
        calibration_psv=calibration[0],
        calibration_receipt=calibration[1],
        calibration_split_receipt=calibration[2],
    )


def _prepare(tmp_path: Path, *, additional_steps: int = 4) -> tuple[Path, Path, Path]:
    base, checkpoint = _base_run(tmp_path / "base")
    output = tmp_path / "incremental"
    prepare_incremental_mlx_run(
        output,
        base_run_directory=base,
        base_checkpoint=checkpoint,
        **_source_arguments(tmp_path),
        additional_optimizer_steps=additional_steps,
        learning_rate=1e-5,
        checkpoint_batches=2,
        log_batches=1,
        calibration_validation_positions=2,
        quantisation_audit_positions=1,
    )
    return output, base, checkpoint


def _tree_snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_prepare_is_separate_create_only_hash_bound_and_qat_continuing(
    tmp_path: Path,
) -> None:
    base, checkpoint = _base_run(tmp_path / "base")
    sources = _source_arguments(tmp_path)
    source_paths = tuple(cast(Path, value) for value in sources.values())
    base_before = _tree_snapshot(base)
    source_before = {str(path): _sha256(path) for path in source_paths}
    output = tmp_path / "incremental"

    plan = prepare_incremental_mlx_run(
        output,
        base_run_directory=base,
        base_checkpoint=checkpoint,
        **sources,
        additional_optimizer_steps=8,
        learning_rate=1e-5,
        broad_weight=3,
        hard_weight=1,
        checkpoint_batches=2,
        log_batches=1,
        calibration_validation_positions=2,
        quantisation_audit_positions=1,
    )

    assert plan["schema"] == INCREMENTAL_MLX_PLAN_SCHEMA
    assert plan["numerics"] == {
        "phase": "quantisation_aware",
        "phase_reset_allowed": False,
        "warmup_reentry_allowed": False,
        "float_master_weights_retained": True,
        "deployment_graph": "tatara_yaneuraou_integer_equivalent_ste",
    }
    schedule = cast(dict[str, object], plan["schedule"])
    duplicate_contract = cast(dict[str, object], plan["duplicate_contract"])
    sources_manifest = cast(dict[str, object], plan["sources"])
    assert schedule["cycle"] == [
        "broad_anchor",
        "broad_anchor",
        "broad_anchor",
        "hard_exact",
    ]
    assert schedule["physical_psv_copy"] is False
    assert duplicate_contract["duplicates_allowed"] is False
    assert "calibration" in sources_manifest
    assert "heldout" not in sources_manifest
    assert all(
        cast(int, cast(dict[str, object], source)["records"])
        < cast(
            int,
            cast(dict[str, object], cast(dict[str, object], source)["split"])["positions"],
        )
        for source in sources_manifest.values()
    )
    assert not any(path.suffix == ".psv" for path in output.rglob("*"))
    assert load_incremental_mlx_plan(output) == plan
    state = load_incremental_mlx_state(output)
    assert state["schema"] == INCREMENTAL_MLX_STATE_SCHEMA
    assert state["durable_optimizer_step"] == 2_000
    assert state["latest_checkpoint"] == str(checkpoint)
    assert _tree_snapshot(base) == base_before
    assert {str(path): _sha256(path) for path in source_paths} == source_before
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_incremental_mlx_run(
            output,
            base_run_directory=base,
            base_checkpoint=checkpoint,
            **sources,
            additional_optimizer_steps=8,
            learning_rate=1e-5,
            calibration_validation_positions=2,
            quantisation_audit_positions=1,
        )


def test_prepare_rejects_overlap_incomplete_base_and_base_root_output(tmp_path: Path) -> None:
    base, checkpoint = _base_run(tmp_path / "base", status="running")
    sources = _source_arguments(tmp_path)
    with pytest.raises(ValueError, match="must be complete"):
        prepare_incremental_mlx_run(
            tmp_path / "incremental-incomplete",
            base_run_directory=base,
            base_checkpoint=checkpoint,
            **sources,
            additional_optimizer_steps=1,
            learning_rate=1e-5,
            calibration_validation_positions=2,
            quantisation_audit_positions=1,
        )
    _json(
        base / "mlx-state.json",
        {
            "schema": "meteo-nagisa-mlx-backend-state-v1",
            "status": "complete",
            "latest_checkpoint": str(checkpoint),
        },
    )
    with pytest.raises(ValueError, match="outside"):
        prepare_incremental_mlx_run(
            base / "incremental-forbidden",
            base_run_directory=base,
            base_checkpoint=checkpoint,
            **sources,
            additional_optimizer_steps=1,
            learning_rate=1e-5,
            calibration_validation_positions=2,
            quantisation_audit_positions=1,
        )

    hard_split_path = sources["hard_exact_split_receipt"]
    hard_split = json.loads(hard_split_path.read_text(encoding="utf-8"))
    broad_split = json.loads(sources["broad_anchor_split_receipt"].read_text(encoding="utf-8"))
    hard_split["position_ids"][0] = broad_split["position_ids"][0]
    _json(hard_split_path, hard_split)
    hard_labels_path = hard_split_path.parent / "labels.jsonl"
    hard_labels = [
        json.loads(line) for line in hard_labels_path.read_text(encoding="utf-8").splitlines()
    ]
    hard_labels[0]["position_id"] = broad_split["position_ids"][0]
    hard_labels_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in hard_labels),
        encoding="utf-8",
    )
    hard_receipt_path = sources["hard_exact_receipt"]
    hard_receipt = json.loads(hard_receipt_path.read_text(encoding="utf-8"))
    hard_receipt["split"]["receipt_sha256"] = _sha256(hard_split_path)
    hard_receipt["labels"]["sha256"] = _sha256(hard_labels_path)
    hard_receipt["labels"]["bytes"] = hard_labels_path.stat().st_size
    _json(hard_receipt_path, hard_receipt)
    with pytest.raises(ValueError, match="duplicate normalized"):
        prepare_incremental_mlx_run(
            tmp_path / "incremental-overlap",
            base_run_directory=base,
            base_checkpoint=checkpoint,
            **sources,
            additional_optimizer_steps=1,
            learning_rate=1e-5,
            calibration_validation_positions=2,
            quantisation_audit_positions=1,
        )


def test_run_rejects_redirected_mutable_directory_and_incomplete_qat_plan(
    tmp_path: Path,
) -> None:
    output, base, _checkpoint = _prepare(tmp_path)
    base_before = _tree_snapshot(base)
    (output / "logs").rmdir()
    (output / "logs").symlink_to(base / "private", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        run_incremental_mlx(output)
    assert _tree_snapshot(base) == base_before

    other_root = tmp_path / "other"
    other_root.mkdir()
    other_base, other_checkpoint = _base_run(other_root / "base")
    mlx_plan_path = other_base / "mlx-plan.json"
    mlx_plan = json.loads(mlx_plan_path.read_text(encoding="utf-8"))
    del mlx_plan["numerics"]["warmup_forward"]
    _json(mlx_plan_path, mlx_plan)
    with pytest.raises(ValueError, match="QAT phase"):
        prepare_incremental_mlx_run(
            other_root / "incremental",
            base_run_directory=other_base,
            base_checkpoint=other_checkpoint,
            **_source_arguments(other_root),
            additional_optimizer_steps=1,
            learning_rate=1e-5,
            calibration_validation_positions=2,
            quantisation_audit_positions=1,
        )


def test_resume_rehashes_sources_while_status_remains_constant_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _base, _checkpoint = _prepare(tmp_path)
    plan = load_incremental_mlx_plan(output)
    sources = cast(dict[str, object], plan["sources"])
    broad = Path(cast(str, cast(dict[str, object], sources["broad_anchor"])["path"]))
    original_sha256 = provenance_sha256_file
    status_hashed: list[Path] = []

    def traced_sha256(path: Path) -> str:
        status_hashed.append(path)
        if path.suffix == ".psv":
            raise AssertionError("status must not re-hash a replay corpus")
        return original_sha256(path)

    monkeypatch.setattr(incremental_mlx, "sha256_file", traced_sha256)
    assert incremental_mlx_status(output)["status"] == "prepared"
    assert status_hashed
    monkeypatch.setattr(incremental_mlx, "sha256_file", original_sha256)

    payload = bytearray(broad.read_bytes())
    payload[-1] ^= 1
    broad.write_bytes(payload)
    with pytest.raises(ValueError, match="PSV identity changed"):
        run_incremental_mlx(output)


def test_stop_file_before_training_preserves_base_coordinate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _base, _checkpoint = _prepare(tmp_path)
    (output / "STOP").write_text("operator stop\n", encoding="utf-8")
    monkeypatch.setattr(
        incremental_mlx,
        "load_nagisa_plan",
        lambda _root: {
            "training": {
                "score_scale": 600.0,
                "loss": nagisa_wrm_contract(600.0),
                "yaneuraou_fv_scale": 16,
            }
        },
    )
    monkeypatch.setattr(
        incremental_mlx,
        "_native_binary_from_plan",
        lambda _root, _plan: output / "private" / "progress.bin",
    )
    monkeypatch.setattr(
        incremental_mlx,
        "load_mlx_checkpoint",
        lambda _checkpoint, *, learning_rate: (
            _FakeModel(),
            _FakeOptimizer(2_000),
            {"learning_rate": learning_rate},
        ),
    )

    status = run_incremental_mlx(output)

    assert status["status"] == "stopped_safely"
    assert status["durable_optimizer_step"] == 2_000
    assert status["durable_incremental_batches"] == 0


def test_weighted_schedule_is_deterministic_and_rejects_unknown_sources() -> None:
    cycle = ("broad_anchor", "broad_anchor", "broad_anchor", "hard_exact")
    assert [weighted_batch_source(cycle, index) for index in range(10)] == [
        "broad_anchor",
        "broad_anchor",
        "broad_anchor",
        "hard_exact",
        "broad_anchor",
        "broad_anchor",
        "broad_anchor",
        "hard_exact",
        "broad_anchor",
        "broad_anchor",
    ]
    with pytest.raises(ValueError, match="unknown replay source"):
        weighted_batch_source(("shadowed-source",), 0)


class _FakeScalar:
    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value


class _FakeOptimizer:
    def __init__(self, step: int) -> None:
        self.state: dict[str, _FakeScalar] = {"step": _FakeScalar(step)}


class _FakeModel:
    def __init__(self) -> None:
        self.state: dict[str, object] = {}


class _FakeBatch:
    def __init__(self, positions: int) -> None:
        self.positions = positions


def test_mock_executor_continues_optimizer_and_never_reenters_float_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _base, _checkpoint = _prepare(tmp_path, additional_steps=4)
    phases: list[str] = []
    stream_sources: list[str] = []
    lock_held = False
    original_load_plan = incremental_mlx.load_incremental_mlx_plan

    @contextmanager
    def fake_run_lock(_root: Path) -> Iterator[None]:
        nonlocal lock_held
        lock_held = True
        try:
            yield
        finally:
            lock_held = False

    def locked_load_plan(root: Path) -> dict[str, object]:
        assert lock_held, "plan/state resume coordinates must be loaded under the run lock"
        return original_load_plan(root)

    def fake_common(_root: Path) -> dict[str, object]:
        return {
            "training": {
                "score_scale": 600.0,
                "loss": nagisa_wrm_contract(600.0),
                "yaneuraou_fv_scale": 16,
            }
        }

    def fake_load_checkpoint(
        checkpoint: Path, *, learning_rate: float
    ) -> tuple[_FakeModel, _FakeOptimizer, dict[str, object]]:
        assert learning_rate == 1e-5
        step = 2_000
        manifest_path = checkpoint / "manifest.json"
        if checkpoint.is_relative_to(output) and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            step = manifest["optimizer"]["step"]
        return _FakeModel(), _FakeOptimizer(step), {}

    def fake_compile(
        _model: _FakeModel,
        _optimizer: _FakeOptimizer,
        *,
        parameters: object,
        forward_mode: str,
    ) -> object:
        del parameters
        phases.append(forward_mode)
        return object()

    class FakeStream:
        def __init__(self, _binary: Path, **kwargs: Any) -> None:
            stream_sources.append(Path(kwargs["psv"]).parent.name)
            self.count = int(kwargs["records"]) // int(kwargs["batch_size"])
            self.batch_size = int(kwargs["batch_size"])

        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self) -> Iterator[_FakeBatch]:
            for _ in range(self.count):
                yield _FakeBatch(self.batch_size)

    def fake_execute(
        _step_fn: object,
        _learning_rate: object,
        _batch: _FakeBatch,
        *,
        model: _FakeModel,
        optimizer: _FakeOptimizer,
    ) -> tuple[float, float, float]:
        del model
        optimizer.state["step"].value += 1
        return 0.01, 0.1, 0.5

    def fake_checkpoint(
        root: Path,
        *,
        model: _FakeModel,
        optimizer: _FakeOptimizer,
        optimizer_step: int,
        incremental_batches: int,
        **_kwargs: object,
    ) -> tuple[_FakeModel, _FakeOptimizer, Path, Path, dict[str, object]]:
        assert optimizer.state["step"].item() == optimizer_step
        checkpoint = root / "checkpoints" / f"{DEFAULT_NET_ID}-mlx-step-{optimizer_step:08d}"
        checkpoint.mkdir()
        _json(
            checkpoint / "manifest.json",
            {
                "schema": MLX_CHECKPOINT_SCHEMA,
                "complete": True,
                "optimizer": {"step": optimizer_step},
            },
        )
        export = root / "exports" / f"{DEFAULT_NET_ID}-mlx-step-{optimizer_step:08d}"
        export.mkdir()
        receipt = {
            "schema": INCREMENTAL_MLX_CHECKPOINT_RECEIPT_SCHEMA,
            "complete": True,
            "optimizer_step": optimizer_step,
            "incremental_batches": incremental_batches,
            "numeric_phase": "quantisation_aware",
            "checkpoint": {
                "path": str(checkpoint),
                "manifest_sha256": _sha256(checkpoint / "manifest.json"),
            },
            "calibration_validation": {"loss": 0.02},
            "export": {"path": str(export), "receipt": {"complete": True}},
            "quantisation_audit": {"accepted": True},
            "master_weight_audit": {"accepted": True},
            "source_manifest_sha256": _sha256(root / "incremental-plan.json"),
            "local_only": True,
        }
        _json(root / "receipts" / f"incremental-step-{optimizer_step:08d}.json", receipt)
        return model, optimizer, checkpoint, export, receipt

    monkeypatch.setattr(incremental_mlx, "load_nagisa_plan", fake_common)
    monkeypatch.setattr(incremental_mlx, "_exclusive_run_lock", fake_run_lock)
    monkeypatch.setattr(incremental_mlx, "load_incremental_mlx_plan", locked_load_plan)
    monkeypatch.setattr(
        incremental_mlx,
        "_native_binary_from_plan",
        lambda _root, _plan: output / "private" / "progress.bin",
    )
    monkeypatch.setattr(incremental_mlx, "load_mlx_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(incremental_mlx, "_compiled_training_step", fake_compile)
    monkeypatch.setattr(incremental_mlx, "NativeBatchStream", FakeStream)
    monkeypatch.setattr(incremental_mlx, "_execute_training_batch", fake_execute)
    monkeypatch.setattr(incremental_mlx, "_checkpoint_transaction", fake_checkpoint)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)

    status = run_incremental_mlx(output)

    assert status["status"] == "complete"
    assert status["durable_optimizer_step"] == 2_004
    assert status["durable_incremental_batches"] == 4
    assert phases and set(phases) == {"quantisation_aware"}
    # The broad fixture contains two batches, so the third weighted broad batch
    # deterministically wraps to a new native stream before the hard batch.
    assert stream_sources == ["broad", "broad", "hard"]
    assert len(cast(list[object], status["checkpoint_generations"])) == 2
    assert len(list((output / "receipts").glob("incremental-step-*.json"))) == 2
    base_step = load_incremental_mlx_state(output)["base_optimizer_step"]
    assert base_step == 2_000

    resumed = run_incremental_mlx(output)
    assert resumed["status"] == "complete"
    assert resumed["durable_optimizer_step"] == 2_004
    assert incremental_mlx_status(output)["eta_seconds"] == 0.0
