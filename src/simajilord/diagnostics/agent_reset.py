"""Offline last-resort reset for saved Codex provider continuity."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from simajilord.agent.store import AgentConversationStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simajilord-agent-reset",
        description=(
            "Clear only saved provider-thread continuity. Stop the bot first; "
            "request, delivery, audit, feedback, memory, and action rows are preserved."
        ),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(".data/agent_conversations.sqlite3"),
        help="Path to agent_conversations.sqlite3.",
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--all",
        action="store_true",
        help="Reset every conversation with saved provider continuity.",
    )
    scope.add_argument(
        "--conversation",
        action="append",
        metavar="ID",
        help="Reset one conversation ID; repeat for multiple IDs.",
    )
    parser.add_argument(
        "--backup-path",
        type=Path,
        help="SQLite backup destination (default: timestamped sibling file).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the bot is stopped and execute the reset.",
    )
    return parser


async def _run(args: argparse.Namespace) -> dict[str, object]:
    database = args.database.expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"Agent conversation database does not exist: {database}")
    if not args.yes:
        raise ValueError("Refusing to reset without --yes after stopping the bot")
    backup_path = (
        args.backup_path.expanduser().resolve()
        if args.backup_path is not None
        else database.with_name(
            f"{database.stem}.backup-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
            f"{database.suffix}"
        )
    )
    if backup_path == database or backup_path.exists():
        raise FileExistsError(f"Backup destination is not available: {backup_path}")
    backup_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    await asyncio.to_thread(_backup_sqlite, database, backup_path)
    stored_epoch = await asyncio.to_thread(
        _stored_conversation_compatibility_epoch,
        database,
    )
    store = (
        AgentConversationStore(database)
        if stored_epoch is None
        else AgentConversationStore(database, compatibility_epoch=stored_epoch)
    )
    selected = None if args.all else tuple(args.conversation or ())
    reset_count = await store.reset_provider_continuity(selected)
    return {
        "backup_path": str(backup_path),
        "database": str(database),
        "preserved": [
            "agent_requests",
            "agent_host_deliveries",
            "audit",
            "feedback",
            "memory",
            "action_receipts",
        ],
        "reset_conversations": reset_count,
        "scope": "all" if args.all else "selected",
    }


def _backup_sqlite(source_path: Path, destination_path: Path) -> None:
    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    destination = sqlite3.connect(destination_path)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    os.chmod(destination_path, 0o600)


def _stored_conversation_compatibility_epoch(database: Path) -> int | None:
    """Read an existing epoch without triggering the store's migration path."""

    source = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        table = source.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'agent_runtime_metadata'
            """
        ).fetchone()
        if table is None:
            return None
        row = source.execute(
            """
            SELECT value FROM agent_runtime_metadata
            WHERE key = 'conversation_compatibility_epoch'
            """
        ).fetchone()
    finally:
        source.close()
    if row is None:
        return None
    try:
        epoch = int(str(row[0]))
    except ValueError as exc:
        raise ValueError("stored conversation compatibility epoch is invalid") from exc
    if epoch < 1 or epoch > 10_000:
        raise ValueError("stored conversation compatibility epoch is out of range")
    return epoch


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(_run(args))
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
