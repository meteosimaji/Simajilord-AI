"""Local administrator CLI for the restart-safe feedback inbox."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from pathlib import Path

from simajilord.core.errors import UserError
from simajilord.services.feedback import (
    FeedbackKind,
    FeedbackReport,
    FeedbackService,
    FeedbackStatus,
)


def _report_payload(report: FeedbackReport) -> dict[str, object]:
    return {
        "report_id": report.report_id,
        "status": report.status.value,
        "kind": report.kind.value,
        "title": report.title,
        "details": report.details,
        "expected": report.expected,
        "reporter_actor_id": report.reporter_actor_id,
        "workspace_id": report.workspace_id,
        "source_transport": report.source_transport,
        "source_event_id": report.source_event_id,
        "source_channel_id": report.source_channel_id,
        "public_reference_id": report.public_reference_id,
        "duplicate_of": report.duplicate_of,
        "created_at": report.created_at.isoformat(),
        "updated_at": report.updated_at.isoformat(),
        "resolved_at": (
            report.resolved_at.isoformat()
            if report.resolved_at is not None
            else None
        ),
    }


def _parser() -> argparse.ArgumentParser:
    data_dir = Path(os.getenv("DATA_DIR", ".data")).expanduser().resolve()
    parser = argparse.ArgumentParser(
        description="Inspect and triage Simajilord's local feedback inbox."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=data_dir / "feedback.sqlite3",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List newest reports.")
    list_parser.add_argument(
        "--status",
        choices=tuple(item.value for item in FeedbackStatus),
    )
    list_parser.add_argument("--limit", type=int, default=50)
    list_parser.add_argument("--json", action="store_true")

    show_parser = subparsers.add_parser("show", help="Show one complete report.")
    show_parser.add_argument("report_id")
    show_parser.add_argument("--json", action="store_true")

    status_parser = subparsers.add_parser(
        "set-status",
        help="Change triage status.",
    )
    status_parser.add_argument("report_id")
    status_parser.add_argument(
        "status",
        choices=tuple(item.value for item in FeedbackStatus),
    )
    status_parser.add_argument("--duplicate-of")

    kind_parser = subparsers.add_parser("set-kind", help="Classify one report.")
    kind_parser.add_argument("report_id")
    kind_parser.add_argument(
        "kind",
        choices=tuple(item.value for item in FeedbackKind),
    )

    export_parser = subparsers.add_parser(
        "export",
        help="Export a bounded JSON array.",
    )
    export_parser.add_argument("--limit", type=int, default=500)
    export_parser.add_argument("--output", type=Path)
    return parser


async def _run(args: argparse.Namespace) -> tuple[int, object]:
    service = FeedbackService(args.database.expanduser().resolve())
    if args.command == "list":
        reports = await service.list(
            status=(FeedbackStatus(args.status) if args.status else None),
            limit=args.limit,
        )
        return 0, reports
    if args.command == "show":
        return 0, await service.get(args.report_id)
    if args.command == "set-status":
        return 0, await service.set_status(
            args.report_id,
            FeedbackStatus(args.status),
            duplicate_of=args.duplicate_of,
        )
    if args.command == "set-kind":
        return 0, await service.set_kind(
            args.report_id,
            FeedbackKind(args.kind),
        )
    if args.command == "export":
        reports = await service.list(limit=args.limit)
        return 0, reports
    raise AssertionError(f"Unhandled feedback command: {args.command}")


def _print_report(report: FeedbackReport) -> None:
    print(f"Report: {report.report_id}")
    print(f"Status: {report.status.value}")
    print(f"Kind: {report.kind.value}")
    print(f"Title: {report.title}")
    print(f"Reporter: {report.reporter_actor_id}")
    print(f"Workspace: {report.workspace_id}")
    print(f"Source: {report.source_transport}:{report.source_event_id}")
    print(f"Created: {report.created_at.isoformat()}")
    print("Details:")
    print(report.details)
    if report.expected is not None:
        print("Expected:")
        print(report.expected)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if hasattr(args, "limit") and not 1 <= args.limit <= 1_000:
        raise SystemExit("--limit must be between 1 and 1000")
    try:
        exit_code, result = asyncio.run(_run(args))
    except UserError as exc:
        print(f"Error: {exc.code}")
        return 1

    if isinstance(result, tuple):
        payload = [_report_payload(report) for report in result]
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        if args.command == "export" and args.output is not None:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            output.write_text(f"{encoded}\n", encoding="utf-8")
            os.chmod(output, 0o600)
            print(f"Exported {len(payload)} report(s) to {output}.")
        elif getattr(args, "json", False) or args.command == "export":
            print(encoded)
        elif not result:
            print("No feedback reports.")
        else:
            for report in result:
                print(
                    f"{report.report_id}  {report.status.value:<11} "
                    f"{report.kind.value:<10} {report.title}"
                )
        return exit_code

    assert isinstance(result, FeedbackReport)
    if getattr(args, "json", False):
        print(
            json.dumps(
                _report_payload(result),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_report(result)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
