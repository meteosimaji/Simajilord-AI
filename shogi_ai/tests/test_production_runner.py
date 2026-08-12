from __future__ import annotations

import json
from pathlib import Path

from simajilord_shogi.production_runner import (
    _capacity_guard_reason,
    _progress_log_line,
    _progress_snapshot,
    _rotate_log,
)


def test_rotate_log_keeps_a_bounded_number_of_backups(tmp_path: Path) -> None:
    log = tmp_path / "production.log"
    log.write_bytes(b"current")
    log.with_name("production.log.1").write_bytes(b"previous")

    assert _rotate_log(log, max_bytes=4, backups=2)
    assert not log.exists()
    assert log.with_name("production.log.1").read_bytes() == b"current"
    assert log.with_name("production.log.2").read_bytes() == b"previous"


def test_progress_snapshot_reports_manifest_disk_and_position_counts(tmp_path: Path) -> None:
    workdir = tmp_path / "improve"
    generation = workdir / "generation-000003"
    generation.mkdir(parents=True)
    (generation / "manifest.json").write_text(
        json.dumps(
            {
                "generation": 3,
                "generation_seed": 123,
                "stage": "reanalysed",
                "status": "running",
                "stage_progress": {
                    "phase": "reanalysis",
                    "positions_completed": 64,
                    "positions_total": 128,
                },
            }
        ),
        encoding="utf-8",
    )
    (workdir / "state.json").write_text(
        json.dumps(
            {
                "generation": 2,
                "champion": "/private/checkpoint",
                "cumulative_positions_seen": 128_000,
            }
        ),
        encoding="utf-8",
    )

    snapshot = _progress_snapshot(
        workdir,
        runner_pid=10,
        child_pid=11,
        runner_state="running_generation",
        started_at="2026-08-11T00:00:00+00:00",
        command_sha256="a" * 64,
        bootstrap_position_presentations=64_000,
        target_position_presentations=100_000_000_000,
    )

    assert snapshot["completed_generation"] == 2
    assert snapshot["current_generation"] == 3
    assert snapshot["current_stage"] == "reanalysed"
    assert snapshot["generation_seed"] == 123
    assert snapshot["stage_progress"] == {
        "phase": "reanalysis",
        "positions_completed": 64,
        "positions_total": 128,
    }
    assert snapshot["improvement_position_presentations"] == 128_000
    assert snapshot["lifetime_position_presentations"] == 192_000
    assert snapshot["target_fraction"] == 0.00000192
    assert snapshot["workdir_bytes"] > 0
    assert snapshot["disk_free_bytes"] > 0
    assert snapshot["capacity_guard"]["passed"] is True
    assert b"generation=3 stage=reanalysed" in _progress_log_line(snapshot)
    assert b'progress={"phase":"reanalysis"' in _progress_log_line(snapshot)


def test_capacity_guard_reports_exact_disk_and_workdir_blockers() -> None:
    snapshot = {"disk_free_bytes": 9, "workdir_bytes": 20}

    assert _capacity_guard_reason(
        snapshot,
        minimum_disk_free_bytes=10,
        maximum_workdir_bytes=20,
    ) == "disk_free_below_minimum:9<10"
    assert _capacity_guard_reason(
        {"disk_free_bytes": 10, "workdir_bytes": 21},
        minimum_disk_free_bytes=10,
        maximum_workdir_bytes=20,
    ) == "workdir_above_maximum:21>20"
    assert (
        _capacity_guard_reason(
            {"disk_free_bytes": 10, "workdir_bytes": 20},
            minimum_disk_free_bytes=10,
            maximum_workdir_bytes=20,
        )
        is None
    )
