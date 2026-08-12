from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request

import numpy as np
import pytest
from rsshogi.core import Board
from rsshogi.numpy import PackedSfenValue

import simajilord_shogi.nnue_training as nnue_training
from simajilord_shogi.nnue_runner import _default_tatara_patch, build_parser
from simajilord_shogi.nnue_training import (
    DEFAULT_BATCH_SIZE,
    NAGISA_NNUE_PLAN_SCHEMA,
    NAGISA_NNUE_STATE_SCHEMA,
    PROGRESS_KPABS_BYTES,
    IncompleteShardDownloadError,
    NagisaNnueArchitecture,
    PublicPsvShard,
    download_nagisa_shard,
    extract_nagisa_progress,
    index_public_value_corpus,
    nagisa_run_status,
    nagisa_wrm_contract,
    parse_tatara_metrics,
    prepare_nagisa_nnue_run,
    probe_value_only_psv,
    prune_numbered_artifacts,
    tatara_command_for_segment,
)


def _psv(path: Path, count: int, *, move16: int = 0, invalid_result: bool = False) -> None:
    records = np.zeros(count, dtype=PackedSfenValue)
    board = Board()
    for index in range(count):
        records[index]["sfen"] = np.frombuffer(board.to_packed_sfen(), dtype=np.uint8)
        records[index]["score"] = index - count // 2
        records[index]["move"] = move16
        records[index]["game_ply"] = index
        records[index]["game_result"] = 7 if invalid_result and index == 1 else index % 3 - 1
    path.write_bytes(records.tobytes())


def _progress_archive(path: Path) -> bytes:
    payload = bytes(PROGRESS_KPABS_BYTES)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr("NAGISA_V3.1/eval/progress.bin", payload)
    return payload


def _full_corpus_api() -> bytes:
    revision = "4dfad115d4a808ebe20b6f65f6416ad75a69a6e7"
    sizes = [20_000_000_000] * 99 + [3_794_202_520]
    siblings = []
    for index, size in enumerate(sizes):
        siblings.append(
            {
                "rfilename": f"split_{index:03d}.bin",
                "size": size,
                "lfs": {
                    "size": size,
                    "sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                },
            }
        )
    return json.dumps({"sha": revision, "siblings": siblings}).encode()


def test_nagisa_architecture_is_value_only_runtime_shape() -> None:
    architecture = NagisaNnueArchitecture()

    assert architecture.feature_dimensions == 73_305
    assert (architecture.ft_out, architecture.l1_out, architecture.l2_out) == (1024, 16, 64)
    assert architecture.num_buckets == 9
    assert architecture.yaneuraou_header == (
        "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)"
        "[73305->1024x2],Network=SFNN_HALFKAHM2_1024_15_64_K3K3"
        "{LayerStack=9}"
    )


def test_nagisa_wrm_preserves_target_distribution_and_export_scale() -> None:
    contract = nagisa_wrm_contract(600.0)

    assert contract["kind"] == "wrm"
    assert contract["target_identity"] == "sigmoid(score/600.0)"
    assert contract["nnue2score"] == 508.0
    assert contract["effective_cp_per_float_output"] == 508.0
    assert contract["yaneuraou_fv_scale"] == 16
    assert contract["quantization_gain"] == 8128
    assert contract["systematic_export_scale_ratio"] == 1.0


def test_default_tatara_patch_is_packaged_with_the_nnue_runner() -> None:
    patch = _default_tatara_patch()

    assert patch.is_file()
    assert "assume_progress8kpabs" in patch.read_text()
    assert nnue_training.TATARA_REPOSITORY_URL == "https://github.com/nodchip/tatara.git"


def test_public_value_index_binds_all_4959_billion_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        nnue_training,
        "_fetch_url",
        lambda _url, *, timeout_seconds: _full_corpus_api(),
    )

    index = index_public_value_corpus(allow_user_attested_local_only=True)

    assert index["totals"] == {
        "shards": 100,
        "bytes": 1_983_794_202_520,
        "records": 49_594_855_063,
    }
    assert index["label_contract"] == {
        "move16": "zero_value_only_required",
        "scalar_score_is_training_target": True,
        "policy_target_available": False,
        "teacher_reanalysis_required_before_value_training": False,
    }


def test_public_value_index_requires_explicit_local_only_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden_fetch(_url: str, *, timeout_seconds: float) -> bytes:
        nonlocal called
        called = True
        return b"{}"

    monkeypatch.setattr(nnue_training, "_fetch_url", forbidden_fetch)
    with pytest.raises(PermissionError, match="acknowledgement"):
        index_public_value_corpus()
    assert not called


def test_incomplete_shard_download_is_retriable_and_resumes_exact_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    cache = root / "cache"
    cache.mkdir(parents=True)
    payload = b"abcdefgh" * 5
    shard = PublicPsvShard(
        filename="split_000.bin",
        byte_size=len(payload),
        records=1,
        sha256=hashlib.sha256(payload).hexdigest(),
        resolve_url="https://example.invalid/split_000.bin",
    )
    requests: list[tuple[str | None, float]] = []

    class Response(io.BytesIO):
        def __init__(self, body: bytes, *, status: int, content_range: str | None = None) -> None:
            super().__init__(body)
            self.status = status
            self.headers = {} if content_range is None else {"Content-Range": content_range}

        def getcode(self) -> int:
            return self.status

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    responses = iter(
        (
            Response(payload[:20], status=200),
            Response(payload[20:], status=206, content_range="bytes 20-39/40"),
        )
    )

    def fake_urlopen(request: Request, timeout: float) -> Response:
        range_header = request.get_header("Range")
        requests.append((range_header, timeout))
        return next(responses)

    monkeypatch.setattr(nnue_training.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        nnue_training.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=16 * 1024**3),
    )

    with pytest.raises(IncompleteShardDownloadError, match="expected=40 observed=20"):
        download_nagisa_shard(root, shard, timeout_seconds=12.5)

    part = cache / "split_000.bin.part"
    assert part.read_bytes() == payload[:20]
    completed = download_nagisa_shard(root, shard, timeout_seconds=12.5)

    assert completed.read_bytes() == payload
    assert not part.exists()
    assert requests == [(None, 12.5), ("bytes=20-39", 12.5)]


def test_progress_extraction_copies_router_but_not_teacher_weights(tmp_path: Path) -> None:
    archive = tmp_path / "nagisa.zip"
    payload = _progress_archive(archive)
    destination = tmp_path / "private" / "progress.bin"

    receipt = extract_nagisa_progress(archive, destination)

    assert destination.read_bytes() == payload
    assert receipt["bytes"] == PROGRESS_KPABS_BYTES
    assert receipt["copied_teacher_nnue_weights"] is False
    assert receipt["role"] == "fixed_progress_bucket_router_only"
    with pytest.raises(FileExistsError):
        extract_nagisa_progress(archive, destination)


def test_value_psv_probe_accepts_move_zero_and_rejects_policy_or_bad_result(
    tmp_path: Path,
) -> None:
    value_only = tmp_path / "value.psv"
    _psv(value_only, 12)
    probe = probe_value_only_psv(value_only, board_samples=5)

    assert probe["records"] == 12
    assert probe["move16_zero_records"] == 12
    assert probe["policy_targets"] == 0
    assert probe["game_result_counts"] == {"loss": 4, "draw": 4, "win": 4}

    policy = tmp_path / "policy.psv"
    _psv(policy, 3, move16=1)
    with pytest.raises(ValueError, match="Move16=0"):
        probe_value_only_psv(policy)

    bad_result = tmp_path / "bad-result.psv"
    _psv(bad_result, 3, invalid_result=True)
    with pytest.raises(ValueError, match="invalid game_result"):
        probe_value_only_psv(bad_result)


def test_prepare_plan_is_random_init_value_only_and_storage_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "nagisa.zip"
    _progress_archive(archive)
    shard_size = (DEFAULT_BATCH_SIZE * 4 + 100) * 40
    shard = PublicPsvShard(
        filename="split_000.bin",
        byte_size=shard_size,
        records=shard_size // 40,
        sha256="a" * 64,
        resolve_url="https://example.invalid/split_000.bin",
    )
    fake_index = {
        "schema": "meteo-public-value-psv-index-v1",
        "corpus_id": "fixture",
        "revision": "b" * 40,
        "shards": [shard.to_dict()],
        "totals": {"shards": 1, "bytes": shard.byte_size, "records": shard.records},
    }
    monkeypatch.setattr(
        nnue_training,
        "index_public_value_corpus",
        lambda *args, **kwargs: fake_index,
    )

    def fake_clone(destination: Path, *, patch: Path | None) -> dict[str, object]:
        destination.mkdir(parents=True)
        return {
            "repository": nnue_training.TATARA_REPOSITORY_URL,
            "commit": nnue_training.TATARA_PINNED_COMMIT,
            "license": "MIT",
            "patch": None if patch is None else {"path": str(patch)},
        }

    monkeypatch.setattr(nnue_training, "_clone_pinned_tatara", fake_clone)
    output = tmp_path / "run"
    plan = prepare_nagisa_nnue_run(
        output,
        nagisa_archive=archive,
        tatara_patch=None,
        target_presentations=DEFAULT_BATCH_SIZE * 5,
        heldout_positions=100,
        allow_user_attested_local_only=True,
    )

    assert plan["schema"] == NAGISA_NNUE_PLAN_SCHEMA
    assert plan["mode"] == "value_only_nnue_sfnn"
    assert plan["initialization"]["student_weights"] == "random"
    assert plan["initialization"]["teacher_nnue_weights_copied"] is False
    assert plan["training"]["policy_loss"] == 0.0
    assert plan["training"]["score_drop_abs"] is None
    assert plan["training"]["extreme_score_policy"] == "keep_including_mate_stamps"
    assert plan["training"]["yaneuraou_fv_scale"] == 16
    assert plan["training"]["loss"] == nagisa_wrm_contract(600.0)
    assert plan["training"]["planned_presentations"] >= DEFAULT_BATCH_SIZE * 5
    assert plan["storage_bound"]["raw_checkpoints_kept"] == 2
    assert plan["storage_bound"]["tatara_bins_kept"] == 2
    assert len(plan["schedule"]) == 2
    state = json.loads((output / "state.json").read_text())
    assert state["schema"] == NAGISA_NNUE_STATE_SCHEMA
    assert state["completed_superbatch"] == 0


def test_tatara_command_has_no_policy_or_teacher_weight_copy(tmp_path: Path) -> None:
    root = tmp_path / "run"
    for relative in ("checkpoints", "private"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    binary = root / "nnue-train"
    binary.write_bytes(b"fixture")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    psv = root / "source.psv"
    psv.write_bytes(b"x")
    heldout = root / "private" / "heldout.psv"
    heldout.write_bytes(b"x")
    progress = root / "private" / "progress.bin"
    progress.write_bytes(bytes(PROGRESS_KPABS_BYTES))
    plan = {
        "schema": NAGISA_NNUE_PLAN_SCHEMA,
        "architecture": NagisaNnueArchitecture().to_dict(),
        "training": {
            "net_id": "meteo-nagisa-v1",
            "batch_size": DEFAULT_BATCH_SIZE,
            "learning_rate": 8.75e-4,
            "learning_rate_schedule": "cosine",
            "learning_rate_final": 1e-5,
            "superbatches": 2,
            "wdl_lambda": 0.0,
            "score_scale": 600.0,
            "yaneuraou_fv_scale": 16,
            "loss": nagisa_wrm_contract(600.0),
            "score_drop_abs": None,
            "target_presentations": DEFAULT_BATCH_SIZE * 2,
        },
        "heldout": {
            "source_filename": "heldout-source.bin",
            "tail_positions": 1,
            "test_positions_per_superbatch": 1,
        },
    }
    (root / "plan.json").write_text(json.dumps(plan))
    segment = {
        "superbatch": 1,
        "filename": "train.bin",
        "batches": 1,
    }

    command = tatara_command_for_segment(
        root,
        segment,
        tatara_binary=binary,
        cached_psv=psv,
        resume_checkpoint=None,
    )

    joined = " ".join(command)
    assert "--feature-set halfka-hm-merged" in joined
    assert "--ft-out 1024" in joined
    assert "--l1 16" in joined
    assert "--l2 64" in joined
    assert "--num-buckets 9" in joined
    assert "--wdl 0.0" in joined
    assert "--scale 600.0" in joined
    assert "--win-rate-model" in command
    assert "--wrm-nnue2score 508.0" in joined
    assert "--wrm-in-scaling 600.0" in joined
    assert "--wrm-in-offset 0.0" in joined
    assert "--wrm-target-scaling 600.0" in joined
    assert "--wrm-target-offset 0.0" in joined
    assert "--fv-scale 16" in joined
    assert "--score-drop-abs" not in command
    assert "--resume" not in command
    assert "--init-from" not in command
    assert "policy" not in joined.lower()


def test_metrics_eta_and_generation_pruning(tmp_path: Path) -> None:
    metrics = parse_tatara_metrics(
        [
            "[train] superbatch 2/206 | loss 0.012345 | 2500000 pos/s | "
            "lr 1e-4 | wdl 0.000 | sb 200.0s | ETA 1h | test_loss 0.013000 | "
            "test_acc 0.7000"
        ]
    )
    assert metrics == {
        "superbatch": 2,
        "loss": 0.012345,
        "positions_per_second": 2_500_000.0,
        "test_loss": 0.013,
    }

    artifacts = tmp_path / "checkpoints"
    artifacts.mkdir()
    for generation in (1, 2, 9, 10):
        (artifacts / f"meteo-nagisa-v1-{generation}.ckpt").write_bytes(b"x")
    kept = prune_numbered_artifacts(artifacts, net_id="meteo-nagisa-v1", suffix=".ckpt", keep=2)
    assert [path.name for path in kept] == [
        "meteo-nagisa-v1-10.ckpt",
        "meteo-nagisa-v1-9.ckpt",
    ]
    assert sorted(path.name for path in artifacts.iterdir()) == sorted(path.name for path in kept)


def test_status_and_standalone_cli_do_not_require_legacy_mlx(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    plan = {
        "schema": NAGISA_NNUE_PLAN_SCHEMA,
        "training": {"target_presentations": 1000, "superbatches": 2},
    }
    state = {
        "schema": NAGISA_NNUE_STATE_SCHEMA,
        "status": "running",
        "completed_superbatch": 1,
        "presentations": 400,
        "last_metrics": {"positions_per_second": 100.0, "loss": 0.1},
        "failure": None,
    }
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "state.json").write_text(json.dumps(state))

    status_payload = nagisa_run_status(root)

    assert status_payload["completion_fraction"] == 0.4
    assert status_payload["compute_eta_seconds_excluding_download"] == 6.0
    args = build_parser().parse_args(["status", str(root)])
    assert args.command == "status"
    assert args.run_directory == root


def test_stop_marker_is_created_with_private_permissions(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    marker = root / "STOP"
    descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
