from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

import simajilord_shogi.nnue_runtime as nnue_runtime
from simajilord_shogi.nnue_runner import build_parser
from simajilord_shogi.nnue_runner import main as nnue_main
from simajilord_shogi.nnue_runtime import (
    NNUE_RUNTIME_CONTRACT_SCHEMA,
    NNUE_RUNTIME_REGISTRY_SCHEMA,
    SANITIZED_EXPORT_RECEIPT_SCHEMA,
    NnueRuntimeContract,
    RuntimeProfile,
    load_nnue_runtime_contract,
    load_nnue_runtime_registry,
    publish_nnue_runtime_registry,
    stage_nnue_runtime,
    verify_nnue_runtime,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_export(
    directory: Path,
    *,
    weight_marker: bytes,
    optimizer_step: int,
    private_marker: str,
) -> Path:
    directory.mkdir()
    nn_bin = directory / "nn.bin"
    progress_bin = directory / "progress.bin"
    eval_options = directory / "eval_options.txt"
    nn_bin.write_bytes(b"Meteo NNUE\0" + weight_marker)
    progress_bin.write_bytes(b"progress-8kpabs-v1")
    eval_options.write_text(
        "LS_BUCKET_MODE progress8kpabs\nLS_PROGRESS_COEFF progress.bin\nFV_SCALE 16\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": "meteo-nagisa-mlx-yaneuraou-export-v1",
        "architecture": "HalfKAv2_hm^ + progress LayerStack",
        "optimizer_step": optimizer_step,
        "local_only": True,
        "nn_bin": {"bytes": nn_bin.stat().st_size, "sha256": _sha256(nn_bin)},
        "progress_bin_sha256": _sha256(progress_bin),
        "routing": "progress8kpabs",
        "yaneuraou_fv_scale": 16,
        # These fields are deliberately private and must never be copied.
        "checkpoint": f"/Users/private/{private_marker}/master.npz",
        "exporter_stderr_tail": f"private diagnostic {private_marker}",
    }
    (directory / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return directory


def _write_fake_yaneuraou(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path
import sys

launch_log = os.environ.get("FAKE_NNUE_LAUNCH_LOG")
if launch_log:
    with open(launch_log, "a", encoding="utf-8") as stream:
        stream.write("launch\\n")

options = {
    "MultiPV": "1",
    "BookFile": "standard_book.db",
    "EnteringKingRule": "CSARule27",
    "EvalDir": "eval",
    "FV_SCALE": "16",
    "USI_Hash": "16",
    "LS_BUCKET_MODE": "progress8kpabs",
    "LS_PROGRESS_COEFF": "progress.bin",
    "PvInterval": "300",
    "Threads": "1",
    "USI_OwnBook": "true",
}

for raw in sys.stdin:
    command = raw.strip()
    if command == "usi":
        print("id name fake-yaneuraou-nnue", flush=True)
        print("id author fixture", flush=True)
        print("option name MultiPV type spin default 1 min 1 max 32", flush=True)
        print(
            "option name BookFile type combo default standard_book.db "
            "var no_book var standard_book.db",
            flush=True,
        )
        print(
            "option name EnteringKingRule type combo default CSARule27 "
            "var CSARule24 var CSARule27",
            flush=True,
        )
        print("option name EvalDir type string default eval", flush=True)
        print("option name FV_SCALE type spin default 16 min 1 max 128", flush=True)
        print(
            "option name USI_Hash type spin default 16 min 1 max 1048576",
            flush=True,
        )
        print(
            "option name LS_BUCKET_MODE type combo default progress8kpabs "
            "var progress8kpabs",
            flush=True,
        )
        print(
            "option name LS_PROGRESS_COEFF type string default progress.bin",
            flush=True,
        )
        print("option name PvInterval type spin default 300 min 0 max 100000", flush=True)
        print("option name Threads type spin default 1 min 1 max 512", flush=True)
        print("option name USI_OwnBook type check default true", flush=True)
        print("usiok", flush=True)
    elif command.startswith("setoption name "):
        setting = command.removeprefix("setoption name ")
        name, value = setting.split(" value ", 1)
        options[name] = value
    elif command == "isready":
        if os.environ.get("FAKE_NNUE_WRONG_FV_SCALE") == "1":
            options["FV_SCALE"] = "99"
        required = (
            Path(options["EvalDir"]) / "nn.bin",
            Path(options["EvalDir"]) / options["LS_PROGRESS_COEFF"],
        )
        if not all(item.is_file() for item in required):
            print("error NNUE runtime artifact not found", flush=True)
        print("readyok", flush=True)
    elif command.startswith("getoption "):
        name = command.removeprefix("getoption ")
        if name in options:
            print(f"Options[{name}] = {options[name]}", flush=True)
        else:
            print(f"No such option: {name}", flush=True)
    elif command.startswith("go nodes "):
        nodes = command.split()[2]
        print(
            f"info depth 4 seldepth 5 multipv 1 score cp -42 "
            f"nodes {nodes} pv 7g7f",
            flush=True,
        )
        print("bestmove 7g7f", flush=True)
    elif command == "quit":
        break
""",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


def _profile(profile_id: str = "meteo-local") -> RuntimeProfile:
    return RuntimeProfile(
        profile_id=profile_id,
        hash_megabytes=32,
        smoke_nodes=37,
        timeout_seconds=10.0,
    )


def test_stage_is_private_atomic_and_independently_reloadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_marker = "DO_NOT_COPY_THIS_PATH_OR_DIAGNOSTIC"
    export = _write_export(
        tmp_path / "export",
        weight_marker=b"generation-one",
        optimizer_step=101,
        private_marker=private_marker,
    )
    engine = _write_fake_yaneuraou(tmp_path / "source-yaneuraou")
    launch_log = tmp_path / "launches.txt"
    monkeypatch.setenv("FAKE_NNUE_LAUNCH_LOG", str(launch_log))
    destination = tmp_path / "runtimes" / "generation-one"

    contract = stage_nnue_runtime(export, engine, destination, profile=_profile())

    assert contract.schema == NNUE_RUNTIME_CONTRACT_SCHEMA
    assert contract.scope == "private_local_only"
    assert contract.publication_allowed is False
    assert contract.reload_consistent is True
    assert contract.nn_bin_sha256 == _sha256(export / "nn.bin")
    assert contract.initial_load_smoke.score_cp == -42
    assert contract.initial_load_smoke.bestmove == "7g7f"
    assert all(option.verified for option in contract.initial_load_smoke.applied_options)
    assert load_nnue_runtime_contract(destination) == contract
    assert launch_log.read_text(encoding="utf-8").splitlines() == ["launch", "launch"]

    verify_nnue_runtime(destination, repeat_load_smoke=True, timeout_seconds=10.0)
    assert launch_log.read_text(encoding="utf-8").splitlines() == [
        "launch",
        "launch",
        "launch",
    ]
    assert {item.name for item in destination.iterdir()} == {
        "engine",
        "eval",
        "runtime-contract.json",
        "source-export-receipt.json",
    }
    assert {item.name for item in (destination / "eval").iterdir()} == {
        "nn.bin",
        "progress.bin",
        "eval_options.txt",
    }
    sanitized = json.loads((destination / "source-export-receipt.json").read_text(encoding="utf-8"))
    assert sanitized["schema"] == SANITIZED_EXPORT_RECEIPT_SCHEMA
    assert sanitized["raw_source_metadata_copied"] is False
    persisted = b"".join(
        path.read_bytes()
        for path in (
            destination / "runtime-contract.json",
            destination / "source-export-receipt.json",
        )
    )
    assert private_marker.encode() not in persisted
    assert str(tmp_path).encode() not in persisted
    assert b"/Users/private" not in persisted
    assert not list(destination.parent.glob(f".{destination.name}.stage-*"))


def test_repeat_load_allows_informational_startup_stdout_to_vary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = _write_export(
        tmp_path / "export",
        weight_marker=b"dynamic-startup-info",
        optimizer_step=102,
        private_marker="dynamic-startup-info",
    )
    engine = _write_fake_yaneuraou(tmp_path / "source-yaneuraou")
    destination = tmp_path / "runtime"
    contract = stage_nnue_runtime(export, engine, destination, profile=_profile())
    observed = replace(
        contract.reload_load_smoke,
        startup_stdout_sha256=hashlib.sha256(b"process-specific info").hexdigest(),
        startup_stdout_bytes=len(b"process-specific info"),
    )
    monkeypatch.setattr(nnue_runtime, "_run_load_smoke", lambda *args, **kwargs: observed)

    assert (
        verify_nnue_runtime(destination, repeat_load_smoke=True, timeout_seconds=10.0) == contract
    )


def test_stage_rejects_symlinks_options_and_json_key_collisions(tmp_path: Path) -> None:
    engine = _write_fake_yaneuraou(tmp_path / "source-yaneuraou")
    export = _write_export(
        tmp_path / "export-symlink",
        weight_marker=b"symlink",
        optimizer_step=1,
        private_marker="symlink",
    )
    original_nn = export / "original-nn.bin"
    (export / "nn.bin").rename(original_nn)
    (export / "nn.bin").symlink_to(original_nn)
    with pytest.raises(ValueError, match="symlink"):
        stage_nnue_runtime(export, engine, tmp_path / "symlink-runtime", profile=_profile())

    with pytest.raises(ValueError, match="option collision"):
        RuntimeProfile(
            profile_id="duplicate-option",
            extra_options=(("fv_scale", 16),),
        )

    duplicate_export = _write_export(
        tmp_path / "export-duplicate-json",
        weight_marker=b"duplicate-json",
        optimizer_step=2,
        private_marker="duplicate-json",
    )
    receipt_path = duplicate_export / "receipt.json"
    receipt_text = receipt_path.read_text(encoding="utf-8")
    receipt_path.write_text(
        receipt_text.replace('"local_only": true,', '"local_only": true,\n  "LOCAL_ONLY": true,'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="key collision"):
        stage_nnue_runtime(
            duplicate_export,
            engine,
            tmp_path / "duplicate-json-runtime",
            profile=_profile(),
        )


def test_casefold_export_name_collision_is_rejected_when_filesystem_allows_it(
    tmp_path: Path,
) -> None:
    export = _write_export(
        tmp_path / "export",
        weight_marker=b"casefold",
        optimizer_step=3,
        private_marker="casefold",
    )
    collision = export / "NN.BIN"
    try:
        with collision.open("xb") as stream:
            stream.write(b"collision")
    except FileExistsError:
        pytest.skip("the test filesystem is case-insensitive")
    engine = _write_fake_yaneuraou(tmp_path / "source-yaneuraou")
    with pytest.raises(ValueError, match="case-insensitive name collision"):
        stage_nnue_runtime(export, engine, tmp_path / "runtime", profile=_profile())


def test_failed_getoption_smoke_rolls_back_new_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = _write_export(
        tmp_path / "export",
        weight_marker=b"bad-option",
        optimizer_step=4,
        private_marker="bad-option",
    )
    engine = _write_fake_yaneuraou(tmp_path / "source-yaneuraou")
    destination = tmp_path / "runtimes" / "must-not-survive"
    monkeypatch.setenv("FAKE_NNUE_WRONG_FV_SCALE", "1")

    with pytest.raises(RuntimeError, match="was not applied"):
        stage_nnue_runtime(export, engine, destination, profile=_profile())

    assert not destination.exists()
    assert not list(destination.parent.glob(f".{destination.name}.stage-*"))


def test_failed_post_rename_reload_removes_only_the_new_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = _write_export(
        tmp_path / "export",
        weight_marker=b"post-rename-failure",
        optimizer_step=41,
        private_marker="post-rename-failure",
    )
    engine = _write_fake_yaneuraou(tmp_path / "source-yaneuraou")
    destination = tmp_path / "runtimes" / "must-be-rolled-back"
    real_loader = nnue_runtime.load_nnue_runtime_contract
    load_count = 0

    def fail_installed_reload(runtime_directory: Path) -> NnueRuntimeContract:
        nonlocal load_count
        load_count += 1
        if load_count == 2:
            raise RuntimeError("fixture post-rename reload failure")
        return real_loader(runtime_directory)

    monkeypatch.setattr(
        nnue_runtime,
        "load_nnue_runtime_contract",
        fail_installed_reload,
    )

    with pytest.raises(RuntimeError, match="post-rename reload failure"):
        stage_nnue_runtime(export, engine, destination, profile=_profile())

    assert load_count == 2
    assert not destination.exists()
    assert not list(destination.parent.glob(f".{destination.name}.stage-*"))


def test_runtime_reload_detects_content_tampering(tmp_path: Path) -> None:
    export = _write_export(
        tmp_path / "export",
        weight_marker=b"before-tamper",
        optimizer_step=5,
        private_marker="tamper",
    )
    engine = _write_fake_yaneuraou(tmp_path / "source-yaneuraou")
    destination = tmp_path / "runtime"
    stage_nnue_runtime(export, engine, destination, profile=_profile())
    nn_bin = destination / "eval" / "nn.bin"
    nn_bin.chmod(0o600)
    nn_bin.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="artifact identity mismatch"):
        load_nnue_runtime_contract(destination)


def test_registry_enforces_latest_previous_champion_two_weight_window(
    tmp_path: Path,
) -> None:
    engine = _write_fake_yaneuraou(tmp_path / "source-yaneuraou")
    runtime_root = tmp_path / "runtimes"
    runtime_one = runtime_root / "generation-one"
    runtime_two = runtime_root / "generation-two"
    runtime_three = runtime_root / "generation-three"
    duplicate_one = runtime_root / "generation-one-copy"
    export_one = _write_export(
        tmp_path / "export-one",
        weight_marker=b"one",
        optimizer_step=10,
        private_marker="one",
    )
    export_two = _write_export(
        tmp_path / "export-two",
        weight_marker=b"two",
        optimizer_step=20,
        private_marker="two",
    )
    export_three = _write_export(
        tmp_path / "export-three",
        weight_marker=b"three",
        optimizer_step=30,
        private_marker="three",
    )
    stage_nnue_runtime(export_one, engine, runtime_one, profile=_profile())
    stage_nnue_runtime(export_two, engine, runtime_two, profile=_profile())
    stage_nnue_runtime(export_three, engine, runtime_three, profile=_profile())
    stage_nnue_runtime(export_one, engine, duplicate_one, profile=_profile())
    registry_path = runtime_root / "runtime-registry.json"

    registry = publish_nnue_runtime_registry(
        registry_path,
        latest_runtime=runtime_two,
        previous_runtime=runtime_one,
        champion_runtime=runtime_one,
    )
    assert registry.to_dict()["schema"] == NNUE_RUNTIME_REGISTRY_SCHEMA
    assert load_nnue_runtime_registry(registry_path) == registry
    registry_bytes = registry_path.read_bytes()
    assert str(tmp_path).encode() not in registry_bytes
    stable_registry_bytes = registry_bytes

    with pytest.raises(ValueError, match="champion must reference"):
        publish_nnue_runtime_registry(
            registry_path,
            latest_runtime=runtime_three,
            previous_runtime=runtime_two,
            champion_runtime=runtime_one,
        )
    assert registry_path.read_bytes() == stable_registry_bytes

    with pytest.raises(ValueError, match="distinct"):
        publish_nnue_runtime_registry(
            registry_path,
            latest_runtime=runtime_one,
            previous_runtime=duplicate_one,
            champion_runtime=runtime_one,
        )
    assert registry_path.read_bytes() == stable_registry_bytes


def test_destination_symlink_is_rejected_without_touching_target(tmp_path: Path) -> None:
    export = _write_export(
        tmp_path / "export",
        weight_marker=b"destination-symlink",
        optimizer_step=6,
        private_marker="destination-symlink",
    )
    engine = _write_fake_yaneuraou(tmp_path / "source-yaneuraou")
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "runtime-link"
    destination.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        stage_nnue_runtime(export, engine, destination, profile=_profile())

    assert not list(outside.iterdir())
    assert os.path.islink(destination)


def test_runtime_cli_subcommands_parse_the_public_contract() -> None:
    parser = build_parser()
    staged = parser.parse_args(
        [
            "stage-runtime",
            "export",
            "engine",
            "runtime",
            "--profile-id",
            "local-fast",
            "--threads",
            "2",
            "--hash-mb",
            "128",
            "--multipv",
            "3",
            "--smoke-nodes",
            "4096",
            "--option",
            "Contempt=0",
        ]
    )
    assert staged.command == "stage-runtime"
    assert staged.profile_id == "local-fast"
    assert staged.threads == 2
    assert staged.hash_megabytes == 128
    assert staged.multipv == 3
    assert staged.smoke_nodes == 4096
    assert staged.option == ["Contempt=0"]
    assert parser.parse_args(["verify-runtime", "runtime"]).command == "verify-runtime"
    assert (
        parser.parse_args(
            [
                "publish-runtime-registry",
                "registry.json",
                "--latest-runtime",
                "latest",
                "--champion-runtime",
                "latest",
            ]
        ).command
        == "publish-runtime-registry"
    )


def test_runtime_cli_stage_verify_and_registry_emit_machine_readable_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = _write_fake_yaneuraou(tmp_path / "source-yaneuraou")
    runtime_root = tmp_path / "runtimes"
    runtime_one = runtime_root / "generation-one"
    runtime_two = runtime_root / "generation-two"
    export_one = _write_export(
        tmp_path / "export-one",
        weight_marker=b"cli-one",
        optimizer_step=101,
        private_marker="cli-private-one",
    )
    export_two = _write_export(
        tmp_path / "export-two",
        weight_marker=b"cli-two",
        optimizer_step=102,
        private_marker="cli-private-two",
    )
    launch_log = tmp_path / "launches.txt"
    monkeypatch.setenv("FAKE_NNUE_LAUNCH_LOG", str(launch_log))

    assert (
        nnue_main(
            [
                "stage-runtime",
                str(export_one),
                str(engine),
                str(runtime_one),
                "--profile-id",
                "cli-local",
                "--hash-mb",
                "32",
                "--smoke-nodes",
                "37",
                "--timeout-seconds",
                "10",
            ]
        )
        == 0
    )
    staged = json.loads(capsys.readouterr().out)
    assert staged["schema"] == NNUE_RUNTIME_CONTRACT_SCHEMA
    assert staged["profile_id"] == "cli-local"

    assert nnue_main(["verify-runtime", str(runtime_one)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["contract_sha256"] == staged["contract_sha256"]
    assert (
        nnue_main(
            [
                "verify-runtime",
                str(runtime_one),
                "--repeat-load-smoke",
                "--timeout-seconds",
                "10",
            ]
        )
        == 0
    )
    reloaded = json.loads(capsys.readouterr().out)
    assert reloaded["runtime_content_sha256"] == staged["runtime_content_sha256"]

    assert (
        nnue_main(
            [
                "stage-runtime",
                str(export_two),
                str(engine),
                str(runtime_two),
                "--profile-id",
                "cli-local",
                "--smoke-nodes",
                "37",
                "--timeout-seconds",
                "10",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    registry_path = runtime_root / "registry.json"
    assert (
        nnue_main(
            [
                "publish-runtime-registry",
                str(registry_path),
                "--latest-runtime",
                str(runtime_two),
                "--previous-runtime",
                str(runtime_one),
                "--champion-runtime",
                str(runtime_one),
            ]
        )
        == 0
    )
    registry = json.loads(capsys.readouterr().out)
    assert registry["schema"] == NNUE_RUNTIME_REGISTRY_SCHEMA
    assert registry["latest"]["relative_directory"] == "generation-two"
    assert registry["previous"]["relative_directory"] == "generation-one"
    assert registry["champion"] == registry["previous"]


@pytest.mark.parametrize(
    "options, match",
    [
        (("Threads=2", "threads=4"), "duplicate runtime option"),
        (("Threads=2",), "runtime option collision"),
        (("MultiPV=2",), "MultiPV is reserved"),
        (("not-an-assignment",), "NAME=VALUE"),
    ],
)
def test_stage_runtime_cli_rejects_duplicate_reserved_and_malformed_options(
    options: tuple[str, ...], match: str
) -> None:
    arguments = [
        "stage-runtime",
        "export",
        "engine",
        "runtime",
        "--profile-id",
        "reject-options",
    ]
    for option in options:
        arguments.extend(("--option", option))
    with pytest.raises(ValueError, match=match):
        nnue_main(arguments)
