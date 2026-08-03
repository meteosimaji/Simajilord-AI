from __future__ import annotations

from pathlib import Path

import pytest

from simajilord.diagnostics.ci_junit import (
    PytestJUnitSummary,
    main,
    markdown_summary,
    summarize_junit,
)


def _report(path: Path, *, skipped: int = 0, failures: int = 0) -> None:
    path.write_text(
        "<?xml version='1.0'?>"
        "<testsuites><testsuite tests='5' errors='0' "
        f"failures='{failures}' skipped='{skipped}'/></testsuites>",
        encoding="utf-8",
    )


def test_junit_summary_counts_each_outcome(tmp_path: Path) -> None:
    report = tmp_path / "pytest.xml"
    _report(report, skipped=1, failures=1)

    summary = summarize_junit(report)

    assert summary == PytestJUnitSummary(
        collected=5,
        passed=3,
        skipped=1,
        failed=1,
        errors=0,
    )
    rendered = markdown_summary("Seatbelt", summary)
    assert "| 5 | 3 | 1 | 1 | 0 |" in rendered


def test_junit_gate_rejects_skips_and_writes_ci_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "pytest.xml"
    step_summary = tmp_path / "summary.md"
    _report(report, skipped=1)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))

    exit_code = main(
        [
            str(report),
            "--title",
            "macOS Seatbelt boundary tests",
            "--require-no-skips",
        ]
    )

    assert exit_code == 1
    rendered = step_summary.read_text(encoding="utf-8")
    assert "macOS Seatbelt boundary tests" in rendered
    assert "| 5 | 4 | 1 | 0 | 0 |" in rendered


def test_junit_gate_accepts_nonempty_success(tmp_path: Path) -> None:
    report = tmp_path / "pytest.xml"
    _report(report)

    assert main([str(report), "--require-no-skips"]) == 0


def test_junit_gate_rejects_missing_or_empty_report(tmp_path: Path) -> None:
    assert main([str(tmp_path / "missing.xml")]) == 2
    empty = tmp_path / "empty.xml"
    empty.write_text("<testsuites />", encoding="utf-8")
    assert main([str(empty)]) == 2
