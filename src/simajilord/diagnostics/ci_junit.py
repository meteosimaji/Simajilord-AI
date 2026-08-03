"""Summarize a pytest JUnit report and enforce CI collection guarantees."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


@dataclass(frozen=True, slots=True)
class PytestJUnitSummary:
    collected: int
    passed: int
    skipped: int
    failed: int
    errors: int


def summarize_junit(path: Path) -> PytestJUnitSummary:
    """Aggregate leaf test suites without double-counting nested suite totals."""

    root = ElementTree.parse(path).getroot()
    suites = [
        suite
        for suite in root.iter("testsuite")
        if not any(child.tag == "testsuite" for child in suite)
    ]
    if not suites:
        raise ValueError("JUnit report contains no test suites")

    collected = sum(_count(suite, "tests") for suite in suites)
    skipped = sum(_count(suite, "skipped") for suite in suites)
    failed = sum(_count(suite, "failures") for suite in suites)
    errors = sum(_count(suite, "errors") for suite in suites)
    passed = collected - skipped - failed - errors
    if passed < 0:
        raise ValueError("JUnit report totals are inconsistent")
    return PytestJUnitSummary(
        collected=collected,
        passed=passed,
        skipped=skipped,
        failed=failed,
        errors=errors,
    )


def _count(suite: ElementTree.Element, attribute: str) -> int:
    raw = suite.get(attribute, "0")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid JUnit {attribute} count: {raw!r}") from exc
    if value < 0:
        raise ValueError(f"negative JUnit {attribute} count")
    return value


def markdown_summary(title: str, summary: PytestJUnitSummary) -> str:
    return (
        f"### {title}\n\n"
        "| Collected | Passed | Skipped | Failed | Errors |\n"
        "| ---: | ---: | ---: | ---: | ---: |\n"
        f"| {summary.collected} | {summary.passed} | {summary.skipped} | "
        f"{summary.failed} | {summary.errors} |\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--title", default="pytest results")
    parser.add_argument("--require-no-skips", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = summarize_junit(args.report)
    except (OSError, ElementTree.ParseError, ValueError) as exc:
        print(f"Could not validate pytest JUnit report: {exc}", file=sys.stderr)
        return 2

    line = (
        f"{args.title}: collected={summary.collected} passed={summary.passed} "
        f"skipped={summary.skipped} failed={summary.failed} errors={summary.errors}"
    )
    print(line)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as output:
            output.write(markdown_summary(args.title, summary))

    if summary.collected == 0:
        print("No tests were collected.", file=sys.stderr)
        return 1
    if summary.failed or summary.errors:
        print("The pytest report contains failures or errors.", file=sys.stderr)
        return 1
    if args.require_no_skips and summary.skipped:
        print(
            f"Required tests were skipped ({summary.skipped}); refusing CI success.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
