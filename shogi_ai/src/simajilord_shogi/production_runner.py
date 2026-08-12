"""Persistent, bounded-log supervisor for one-generation improvement jobs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUNNER_STATUS_SCHEMA = "meteo-production-runner-status-v1"
DEFAULT_TARGET_POSITION_PRESENTATIONS = 100_000_000_000
DEFAULT_MINIMUM_DISK_FREE_BYTES = 32 * 1024**3
DEFAULT_MAXIMUM_WORKDIR_BYTES = 8 * 1024**3
PROGRESS_HEARTBEAT_SECONDS = 60.0


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _rotate_log(path: Path, *, max_bytes: int, backups: int) -> bool:
    if not path.exists() or path.stat().st_size < max_bytes:
        return False
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"production log must be a regular file: {path}")
    for index in range(backups, 0, -1):
        source = path if index == 1 else path.with_name(f"{path.name}.{index - 1}")
        destination = path.with_name(f"{path.name}.{index}")
        if not source.exists():
            continue
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"rotated production log must be a regular file: {source}")
        if destination.exists():
            if destination.is_symlink() or not destination.is_file():
                raise ValueError(
                    f"rotated production log destination must be a regular file: {destination}"
                )
            destination.unlink()
        source.replace(destination)
    return True


def _directory_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for root, _directories, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                if not file_path.is_symlink():
                    total += file_path.stat().st_size
            except FileNotFoundError:
                # Generation artifacts are atomically renamed and retention-pruned
                # while the supervisor is taking its status snapshot.
                continue
    return total


def _capacity_guard_reason(
    snapshot: dict[str, Any],
    *,
    minimum_disk_free_bytes: int,
    maximum_workdir_bytes: int,
) -> str | None:
    disk_free = int(snapshot["disk_free_bytes"])
    workdir_bytes = int(snapshot["workdir_bytes"])
    if disk_free < minimum_disk_free_bytes:
        return (
            "disk_free_below_minimum:"
            f"{disk_free}<{minimum_disk_free_bytes}"
        )
    if workdir_bytes > maximum_workdir_bytes:
        return (
            "workdir_above_maximum:"
            f"{workdir_bytes}>{maximum_workdir_bytes}"
        )
    return None


def _progress_log_line(snapshot: dict[str, Any]) -> bytes:
    stage_progress = snapshot.get("stage_progress")
    progress_suffix = (
        ""
        if stage_progress is None
        else " progress="
        + json.dumps(stage_progress, allow_nan=False, separators=(",", ":"), sort_keys=True)
    )
    return (
        f"[{snapshot['updated_at']}] PROGRESS "
        f"generation={snapshot['current_generation']} "
        f"stage={snapshot['current_stage']} "
        f"status={snapshot['current_status']} "
        f"positions={snapshot['lifetime_position_presentations']}/"
        f"{snapshot['target_position_presentations']} "
        f"workdir_bytes={snapshot['workdir_bytes']} "
        f"disk_free_bytes={snapshot['disk_free_bytes']}"
        f"{progress_suffix}\n"
    ).encode()


def _latest_manifest(workdir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    manifests = sorted(workdir.glob("generation-[0-9][0-9][0-9][0-9][0-9][0-9]/manifest.json"))
    if not manifests:
        return None, None
    path = manifests[-1]
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"generation manifest must be a regular file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"generation manifest must be a JSON object: {path}")
    return path, payload


def _progress_snapshot(
    workdir: Path,
    *,
    runner_pid: int,
    child_pid: int | None,
    runner_state: str,
    started_at: str,
    command_sha256: str,
    bootstrap_position_presentations: int,
    target_position_presentations: int,
    minimum_disk_free_bytes: int = DEFAULT_MINIMUM_DISK_FREE_BYTES,
    maximum_workdir_bytes: int = DEFAULT_MAXIMUM_WORKDIR_BYTES,
    last_exit_code: int | None = None,
) -> dict[str, Any]:
    state_path = workdir / "state.json"
    improvement_positions = 0
    completed_generation = 0
    champion: str | None = None
    if state_path.is_file() and not state_path.is_symlink():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("self-improvement state must be a JSON object")
        improvement_positions = int(state.get("cumulative_positions_seen", 0))
        completed_generation = int(state.get("generation", 0))
        champion_value = state.get("champion")
        champion = None if champion_value is None else str(champion_value)
    manifest_path, manifest = _latest_manifest(workdir)
    current_generation = completed_generation
    current_stage = "not_started"
    current_status = "pending"
    generation_seed: int | None = None
    stage_progress: dict[str, object] | None = None
    if manifest is not None:
        current_generation = int(manifest.get("generation", completed_generation))
        current_stage = str(manifest.get("stage", "unknown"))
        current_status = str(manifest.get("status", "unknown"))
        raw_seed = manifest.get("generation_seed")
        generation_seed = None if raw_seed is None else int(raw_seed)
        raw_progress = manifest.get("stage_progress")
        if raw_progress is not None:
            if not isinstance(raw_progress, dict):
                raise ValueError("generation stage_progress must be a JSON object")
            stage_progress = dict(raw_progress)
    lifetime_positions = bootstrap_position_presentations + improvement_positions
    disk = shutil.disk_usage(workdir.parent)
    snapshot: dict[str, Any] = {
        "schema": RUNNER_STATUS_SCHEMA,
        "updated_at": _timestamp(),
        "started_at": started_at,
        "runner_state": runner_state,
        "runner_pid": runner_pid,
        "child_pid": child_pid,
        "last_exit_code": last_exit_code,
        "command_sha256": command_sha256,
        "workdir": str(workdir),
        "workdir_bytes": _directory_bytes(workdir),
        "disk_free_bytes": disk.free,
        "completed_generation": completed_generation,
        "current_generation": current_generation,
        "current_stage": current_stage,
        "current_status": current_status,
        "generation_seed": generation_seed,
        "stage_progress": stage_progress,
        "latest_manifest": None if manifest_path is None else str(manifest_path),
        "champion": champion,
        "bootstrap_position_presentations": bootstrap_position_presentations,
        "improvement_position_presentations": improvement_positions,
        "lifetime_position_presentations": lifetime_positions,
        "target_position_presentations": target_position_presentations,
        "target_fraction": lifetime_positions / target_position_presentations,
    }
    capacity_reason = _capacity_guard_reason(
        snapshot,
        minimum_disk_free_bytes=minimum_disk_free_bytes,
        maximum_workdir_bytes=maximum_workdir_bytes,
    )
    snapshot["capacity_guard"] = {
        "minimum_disk_free_bytes": minimum_disk_free_bytes,
        "maximum_workdir_bytes": maximum_workdir_bytes,
        "passed": capacity_reason is None,
        "blocker": capacity_reason,
    }
    return snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="run resumable Meteo production generations until stopped"
    )
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--retry-seconds", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-log-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--log-backups", type=int, default=2)
    parser.add_argument(
        "--minimum-disk-free-bytes",
        type=int,
        default=DEFAULT_MINIMUM_DISK_FREE_BYTES,
    )
    parser.add_argument(
        "--maximum-workdir-bytes",
        type=int,
        default=DEFAULT_MAXIMUM_WORKDIR_BYTES,
    )
    parser.add_argument("--bootstrap-position-presentations", type=int, default=0)
    parser.add_argument(
        "--target-position-presentations",
        type=int,
        default=DEFAULT_TARGET_POSITION_PRESENTATIONS,
    )
    parser.add_argument("improve_args", nargs=argparse.REMAINDER)
    return parser


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _wait_until(deadline: float, *, stop_requested: list[bool]) -> None:
    while not stop_requested[0] and time.monotonic() < deadline:
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _positive("retry-seconds", args.retry_seconds)
    _positive("poll-seconds", args.poll_seconds)
    _positive("max-log-bytes", args.max_log_bytes)
    _positive("minimum-disk-free-bytes", args.minimum_disk_free_bytes)
    _positive("maximum-workdir-bytes", args.maximum_workdir_bytes)
    _positive("target-position-presentations", args.target_position_presentations)
    if args.log_backups < 1 or args.bootstrap_position_presentations < 0:
        raise ValueError("log-backups must be positive and bootstrap count non-negative")

    cli = args.cli.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    cwd = args.cwd.expanduser().resolve(strict=True)
    workdir = args.workdir.expanduser().resolve()
    log_path = args.log.expanduser().resolve()
    status_path = args.status.expanduser().resolve()
    stop_file = args.stop_file.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = status_path.with_suffix(status_path.suffix + ".lock")
    improve_args = list(args.improve_args)
    if improve_args[:1] == ["--"]:
        improve_args = improve_args[1:]
    if "--generations" in improve_args:
        raise ValueError("production runner owns --generations and requires one per child")
    command = [
        str(cli),
        "improve",
        str(checkpoint),
        str(workdir),
        "--generations",
        "1",
        *improve_args,
    ]
    command_sha256 = hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode()
    ).hexdigest()
    started_at = _timestamp()
    stop_requested = [False]
    child: subprocess.Popen[bytes] | None = None
    capacity_stop_reason: str | None = None

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested[0] = True
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another production runner owns {lock_path}") from error
        lock.seek(0)
        lock.truncate()
        lock.write(f"{os.getpid()}\n".encode())
        lock.flush()
        os.fsync(lock.fileno())
        last_exit_code: int | None = None
        while (
            not stop_requested[0]
            and not stop_file.exists()
            and capacity_stop_reason is None
        ):
            preflight_snapshot = _progress_snapshot(
                workdir,
                runner_pid=os.getpid(),
                child_pid=None,
                runner_state="capacity_preflight",
                started_at=started_at,
                command_sha256=command_sha256,
                bootstrap_position_presentations=args.bootstrap_position_presentations,
                target_position_presentations=args.target_position_presentations,
                minimum_disk_free_bytes=args.minimum_disk_free_bytes,
                maximum_workdir_bytes=args.maximum_workdir_bytes,
                last_exit_code=last_exit_code,
            )
            capacity_stop_reason = preflight_snapshot["capacity_guard"]["blocker"]
            if capacity_stop_reason is not None:
                _atomic_json(status_path, preflight_snapshot)
                break
            _rotate_log(
                log_path,
                max_bytes=args.max_log_bytes,
                backups=args.log_backups,
            )
            with log_path.open("ab", buffering=0) as log_stream:
                log_stream.write(
                    f"[{_timestamp()}] START one production generation\n".encode()
                )
                child = subprocess.Popen(
                    command,
                    cwd=cwd,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                last_progress_key: tuple[object, ...] | None = None
                last_progress_log_at = 0.0
                while child.poll() is None and not stop_requested[0]:
                    snapshot = _progress_snapshot(
                        workdir,
                        runner_pid=os.getpid(),
                        child_pid=child.pid,
                        runner_state="running_generation",
                        started_at=started_at,
                        command_sha256=command_sha256,
                        bootstrap_position_presentations=(
                            args.bootstrap_position_presentations
                        ),
                        target_position_presentations=(
                            args.target_position_presentations
                        ),
                        minimum_disk_free_bytes=args.minimum_disk_free_bytes,
                        maximum_workdir_bytes=args.maximum_workdir_bytes,
                        last_exit_code=last_exit_code,
                    )
                    _atomic_json(
                        status_path,
                        snapshot,
                    )
                    progress_key = (
                        snapshot["current_generation"],
                        snapshot["current_stage"],
                        snapshot["current_status"],
                        json.dumps(
                            snapshot.get("stage_progress"),
                            allow_nan=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    )
                    now = time.monotonic()
                    if (
                        progress_key != last_progress_key
                        or now - last_progress_log_at >= PROGRESS_HEARTBEAT_SECONDS
                    ):
                        log_stream.write(_progress_log_line(snapshot))
                        last_progress_key = progress_key
                        last_progress_log_at = now
                    capacity_stop_reason = snapshot["capacity_guard"]["blocker"]
                    if capacity_stop_reason is not None:
                        log_stream.write(
                            (
                                f"[{_timestamp()}] CAPACITY_GUARD "
                                f"{capacity_stop_reason}\n"
                            ).encode()
                        )
                        os.killpg(child.pid, signal.SIGINT)
                        break
                    _wait_until(
                        time.monotonic() + args.poll_seconds,
                        stop_requested=stop_requested,
                    )
                if (stop_requested[0] or capacity_stop_reason is not None) and child.poll() is None:
                    os.killpg(child.pid, signal.SIGINT)
                last_exit_code = child.wait()
                log_stream.write(
                    (
                        f"[{_timestamp()}] END one production generation "
                        f"exit={last_exit_code}\n"
                    ).encode()
                )
                child = None
            runner_state = (
                "stopped_by_capacity_guard"
                if capacity_stop_reason is not None
                else ("between_generations" if last_exit_code == 0 else "retry_wait")
            )
            _atomic_json(
                status_path,
                _progress_snapshot(
                    workdir,
                    runner_pid=os.getpid(),
                    child_pid=None,
                    runner_state=runner_state,
                    started_at=started_at,
                    command_sha256=command_sha256,
                    bootstrap_position_presentations=args.bootstrap_position_presentations,
                    target_position_presentations=args.target_position_presentations,
                    minimum_disk_free_bytes=args.minimum_disk_free_bytes,
                    maximum_workdir_bytes=args.maximum_workdir_bytes,
                    last_exit_code=last_exit_code,
                ),
            )
            if capacity_stop_reason is None:
                delay = 1.0 if last_exit_code == 0 else args.retry_seconds
                _wait_until(time.monotonic() + delay, stop_requested=stop_requested)

        final_state = (
            "stopped_by_capacity_guard"
            if capacity_stop_reason is not None
            else ("stopped_by_signal" if stop_requested[0] else "stopped_by_marker")
        )
        _atomic_json(
            status_path,
            _progress_snapshot(
                workdir,
                runner_pid=os.getpid(),
                child_pid=None,
                runner_state=final_state,
                started_at=started_at,
                command_sha256=command_sha256,
                bootstrap_position_presentations=args.bootstrap_position_presentations,
                target_position_presentations=args.target_position_presentations,
                minimum_disk_free_bytes=args.minimum_disk_free_bytes,
                maximum_workdir_bytes=args.maximum_workdir_bytes,
                last_exit_code=last_exit_code,
            ),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
