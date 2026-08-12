"""Standalone CLI for Meteo's NAGISA-style value-only NNUE production run.

Unlike :mod:`simajilord_shogi.cli`, this entrypoint does not import MLX.  It can
therefore be installed on the Linux/NVIDIA host used by Tatara while the
legacy Policy+Value engine remains a macOS-only compatibility surface.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .nnue_training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_HELDOUT_POSITIONS,
    DEFAULT_SCORE_SCALE,
    SOUJOU_DATASETS_1_CORPUS_ID,
    TARGET_PRESENTATIONS,
    ensure_nagisa_heldout,
    index_public_value_corpus,
    nagisa_hardware_preflight,
    nagisa_run_status,
    prepare_nagisa_nnue_run,
    probe_value_only_psv,
    run_nagisa_nnue,
)


def _default_tatara_patch() -> Path:
    return (
        Path(__file__).resolve().with_name("tatara_patches")
        / "progress8kpabs-yaneuraou-converter.patch"
    )


def _build_tatara(run_directory: Path, *, install_cuda_oxide: bool) -> dict[str, object]:
    root = run_directory.expanduser().resolve(strict=True)
    tatara = root / "dependencies" / "tatara"
    if not tatara.is_dir() or tatara.is_symlink():
        raise ValueError("run has no pinned regular Tatara checkout")
    commands: list[list[str]] = []
    if install_cuda_oxide:
        commands.append(["bash", "scripts/setup-cuda-oxide.sh"])
    commands.extend(
        [
            ["bash", "scripts/build-kernels.sh"],
            [
                "cargo",
                "build",
                "--release",
                "--bin",
                "nnue-train",
                "--bin",
                "net_to_yo",
            ],
        ]
    )
    for command in commands:
        completed = subprocess.run(command, cwd=tatara, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Tatara build command failed with {completed.returncode}: {command!r}"
            )
    preflight = nagisa_hardware_preflight(root)
    if not preflight["ready"]:
        raise RuntimeError(f"Tatara built but production preflight still fails: {preflight}")
    return {"commands": commands, "preflight": preflight}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simajilord-nnue",
        description="Meteo NAGISA-style value-only NNUE production supervisor",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser(
        "index-corpus", help="verify and print the pinned public value-PSV shard index"
    )
    index.add_argument("--corpus-id", default=SOUJOU_DATASETS_1_CORPUS_ID)
    index.add_argument("--timeout-seconds", type=float, default=60.0)
    index.add_argument("--allow-user-attested-local-only", action="store_true")

    probe = subparsers.add_parser("probe-psv", help="validate Move16=0 scalar-value PSV records")
    probe.add_argument("psv", type=Path)
    probe.add_argument("--board-samples", type=int, default=4_096)

    prepare = subparsers.add_parser(
        "prepare", help="create a random-init 100B NAGISA-style production plan"
    )
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--nagisa-archive", type=Path, required=True)
    prepare.add_argument("--tatara-patch", type=Path, default=_default_tatara_patch())
    prepare.add_argument("--target-presentations", type=int, default=TARGET_PRESENTATIONS)
    prepare.add_argument("--heldout-positions", type=int, default=DEFAULT_HELDOUT_POSITIONS)
    prepare.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    prepare.add_argument("--score-scale", type=float, default=DEFAULT_SCORE_SCALE)
    prepare.add_argument("--corpus-id", default=SOUJOU_DATASETS_1_CORPUS_ID)
    prepare.add_argument("--timeout-seconds", type=float, default=120.0)
    prepare.add_argument("--allow-user-attested-local-only", action="store_true")
    prepare.add_argument(
        "--skip-heldout-download",
        action="store_true",
        help="prepare metadata only; the run command will fetch the immutable heldout tail",
    )

    heldout = subparsers.add_parser(
        "fetch-heldout", help="fetch and verify the immutable heldout PSV tail"
    )
    heldout.add_argument("run_directory", type=Path)
    heldout.add_argument("--timeout-seconds", type=float, default=120.0)

    build = subparsers.add_parser(
        "build", help="build the pinned Tatara kernels and patched converter on NVIDIA Linux"
    )
    build.add_argument("run_directory", type=Path)
    build.add_argument("--install-cuda-oxide", action="store_true")

    preflight = subparsers.add_parser(
        "preflight", help="check NVIDIA, Tatara binaries, progress model, and immutable plan"
    )
    preflight.add_argument("run_directory", type=Path)

    run = subparsers.add_parser("run", help="run/resume until 100B presentations or a STOP marker")
    run.add_argument("run_directory", type=Path)
    run.add_argument("--timeout-seconds", type=float, default=300.0)
    run.add_argument("--build-if-needed", action="store_true")
    run.add_argument("--install-cuda-oxide", action="store_true")

    status = subparsers.add_parser("status", help="print loss, speed, progress, and compute ETA")
    status.add_argument("run_directory", type=Path)

    prepare_mlx = subparsers.add_parser(
        "prepare-mlx",
        help="bind the immutable NAGISA-style run to this Mac's local MLX GPU",
    )
    prepare_mlx.add_argument("run_directory", type=Path)
    prepare_mlx.add_argument("--checkpoint-batches", type=int, default=512)
    prepare_mlx.add_argument("--log-batches", type=int, default=32)
    prepare_mlx.add_argument("--validation-positions", type=int, default=262_144)
    prepare_mlx.add_argument("--timeout-seconds", type=float, default=300.0)

    preflight_mlx = subparsers.add_parser(
        "preflight-mlx", help="verify local Apple GPU, MLX, Tatara decoder, and inputs"
    )
    preflight_mlx.add_argument("run_directory", type=Path)

    smoke_mlx = subparsers.add_parser(
        "smoke-mlx",
        help="verify real labels, MLX update, exact save/resume, and YaneuraOu export",
    )
    smoke_mlx.add_argument("run_directory", type=Path)
    smoke_mlx.add_argument("--positions", type=int, default=8_192)
    smoke_mlx.add_argument("--training-steps", type=int, default=12)

    run_mlx = subparsers.add_parser(
        "run-mlx", help="run/resume the real 100B value-only schedule on the local Apple GPU"
    )
    run_mlx.add_argument("run_directory", type=Path)
    run_mlx.add_argument("--timeout-seconds", type=float, default=300.0)

    status_mlx = subparsers.add_parser(
        "status-mlx", help="print durable/live MLX loss, speed, ETA, and retained generations"
    )
    status_mlx.add_argument("run_directory", type=Path)

    stop = subparsers.add_parser(
        "request-stop", help="request a safe stop after the active shard/superbatch"
    )
    stop.add_argument("run_directory", type=Path)
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


def _plan_summary(plan: Mapping[str, object], output: Path) -> dict[str, object]:
    corpus = cast(dict[str, Any], plan["corpus"])
    training = cast(dict[str, Any], plan["training"])
    initialization = cast(dict[str, Any], plan["initialization"])
    return {
        "schema": plan["schema"],
        "path": str(output.expanduser().resolve(strict=True) / "plan.json"),
        "mode": plan["mode"],
        "student_weights": initialization["student_weights"],
        "teacher_nnue_weights_copied": initialization["teacher_nnue_weights_copied"],
        "corpus_revision": corpus["revision"],
        "source": corpus["totals"],
        "target_presentations": training["target_presentations"],
        "planned_presentations": training["planned_presentations"],
        "superbatches": training["superbatches"],
        "policy_loss": training["policy_loss"],
        "storage_bound": plan["storage_bound"],
        "public_checkpoint_release_allowed": cast(dict[str, Any], plan["distribution"])[
            "public_checkpoint_release_allowed"
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "index-corpus":
        _print(
            index_public_value_corpus(
                args.corpus_id,
                timeout_seconds=args.timeout_seconds,
                allow_user_attested_local_only=args.allow_user_attested_local_only,
            )
        )
        return 0
    if args.command == "probe-psv":
        _print(probe_value_only_psv(args.psv, board_samples=args.board_samples))
        return 0
    if args.command == "prepare":
        plan = prepare_nagisa_nnue_run(
            args.output,
            nagisa_archive=args.nagisa_archive,
            tatara_patch=args.tatara_patch,
            target_presentations=args.target_presentations,
            heldout_positions=args.heldout_positions,
            batch_size=args.batch_size,
            score_scale=args.score_scale,
            corpus_id=args.corpus_id,
            timeout_seconds=args.timeout_seconds,
            allow_user_attested_local_only=args.allow_user_attested_local_only,
        )
        result: dict[str, object] = {"plan": _plan_summary(plan, args.output)}
        if not args.skip_heldout_download:
            result["heldout"] = ensure_nagisa_heldout(
                args.output, timeout_seconds=args.timeout_seconds
            )
        result["status"] = nagisa_run_status(args.output)
        _print(result)
        return 0
    if args.command == "fetch-heldout":
        _print(ensure_nagisa_heldout(args.run_directory, timeout_seconds=args.timeout_seconds))
        return 0
    if args.command == "build":
        _print(
            _build_tatara(
                args.run_directory,
                install_cuda_oxide=args.install_cuda_oxide,
            )
        )
        return 0
    if args.command == "preflight":
        result = nagisa_hardware_preflight(args.run_directory)
        _print(result)
        return 0 if result["ready"] else 2
    if args.command == "run":
        preflight = nagisa_hardware_preflight(args.run_directory)
        if args.build_if_needed and not preflight["ready"]:
            _build_tatara(
                args.run_directory,
                install_cuda_oxide=args.install_cuda_oxide,
            )
        elif not preflight["ready"]:
            try:
                run_nagisa_nnue(args.run_directory, timeout_seconds=args.timeout_seconds)
            except RuntimeError:
                _print({"preflight": preflight, "status": nagisa_run_status(args.run_directory)})
                return 2
        _print(run_nagisa_nnue(args.run_directory, timeout_seconds=args.timeout_seconds))
        return 0
    if args.command == "status":
        _print(nagisa_run_status(args.run_directory))
        return 0
    if args.command == "prepare-mlx":
        from .mlx_nnue import prepare_mlx_backend

        _print(
            prepare_mlx_backend(
                args.run_directory,
                checkpoint_batches=args.checkpoint_batches,
                log_batches=args.log_batches,
                validation_positions=args.validation_positions,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0
    if args.command == "preflight-mlx":
        from .mlx_nnue import mlx_hardware_preflight

        result = mlx_hardware_preflight(args.run_directory)
        _print(result)
        return 0 if result["ready"] else 2
    if args.command == "smoke-mlx":
        from .mlx_nnue import smoke_mlx_backend

        _print(
            smoke_mlx_backend(
                args.run_directory,
                positions=args.positions,
                training_steps=args.training_steps,
            )
        )
        return 0
    if args.command == "run-mlx":
        from .mlx_nnue import run_mlx_backend

        _print(
            run_mlx_backend(
                args.run_directory,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0
    if args.command == "status-mlx":
        from .mlx_nnue import mlx_run_status

        _print(mlx_run_status(args.run_directory))
        return 0
    if args.command == "request-stop":
        root = args.run_directory.expanduser().resolve(strict=True)
        marker = root / "STOP"
        if marker.exists() or marker.is_symlink():
            raise FileExistsError(f"stop marker already exists: {marker}")
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        _print({"status": "stop_requested", "marker": str(marker)})
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
