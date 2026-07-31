"""Resolve one public agent failure reference without exposing request bodies."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from pathlib import Path

from simajilord.agent.store import AgentConversationStore, AgentRequestRecord
from simajilord.observability import EventJournal, EventRecord

_CAPABILITY_INVOCATION_FIELDS = frozenset(
    {
        "capability",
        "outcome",
        "duration_ms",
        "error_type",
        "public_reference_id",
        "provider_thread_id",
        "provider_turn_id",
        "tool_call_id",
    }
)


async def inspect_agent_reference(
    *,
    reference_id: str,
    requests_database: Path,
    events_database: Path,
    limit: int = 500,
) -> dict[str, object] | None:
    """Return bounded metadata and correlated events for one public reference."""

    if not requests_database.is_file():
        return None
    store = AgentConversationStore(requests_database)
    record = await store.request_by_public_reference_id(reference_id)
    if record is None:
        return None
    events: tuple[EventRecord, ...] = ()
    if events_database.is_file():
        journal = EventJournal(events_database)
        try:
            events = await journal.agent_trace(
                request_id=record.event_id,
                public_reference_id=reference_id,
                limit=limit,
            )
        finally:
            await journal.close()
    return {
        "found": True,
        "request": _request_payload(record),
        "events": [_event_payload(event) for event in events],
        "event_count": len(events),
        "truncated": len(events) >= limit,
    }


def _request_payload(record: AgentRequestRecord) -> dict[str, object]:
    return {
        "public_reference_id": record.public_reference_id,
        "event_id": record.event_id,
        "conversation_id": record.conversation_id,
        "trigger": record.trigger.value,
        "actor_id": record.actor_id,
        "workspace_id": record.workspace_id,
        "channel_id": record.channel_id,
        "source_message_id": record.source_message_id,
        "model": record.model,
        "status": record.status,
        "provider_thread_id": record.provider_thread_id,
        "error_type": record.error_type,
        "occurred_at": record.occurred_at.isoformat(),
        "started_at": record.started_at.isoformat(),
        "completed_at": (
            record.completed_at.isoformat()
            if record.completed_at is not None
            else None
        ),
        "host_delivered_at": (
            record.host_delivered_at.isoformat()
            if record.host_delivered_at is not None
            else None
        ),
    }


def _event_payload(record: EventRecord) -> dict[str, object]:
    payload = record.payload
    if record.kind == "capability.invocation":
        payload = {
            key: value
            for key, value in payload.items()
            if key in _CAPABILITY_INVOCATION_FIELDS
        }
    return {
        "sequence": record.sequence,
        "occurred_at": record.occurred_at.isoformat(),
        "kind": record.kind,
        "actor_id": record.actor_id,
        "workspace_id": record.workspace_id,
        "transport": record.transport,
        "request_id": record.request_id,
        "payload": payload,
    }


def _parser() -> argparse.ArgumentParser:
    data_dir = Path(os.getenv("DATA_DIR", ".data")).expanduser().resolve()
    parser = argparse.ArgumentParser(
        description=(
            "Resolve an agt_ public reference to body-free request metadata and "
            "its durable agent/tool trace."
        )
    )
    parser.add_argument("reference_id")
    parser.add_argument(
        "--requests-database",
        type=Path,
        default=data_dir / "agent_conversations.sqlite3",
    )
    parser.add_argument(
        "--events-database",
        type=Path,
        default=data_dir / "events.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.limit <= 1_000:
        raise SystemExit("--limit must be between 1 and 1000")
    try:
        result = asyncio.run(
            inspect_agent_reference(
                reference_id=args.reference_id,
                requests_database=args.requests_database.expanduser().resolve(),
                events_database=args.events_database.expanduser().resolve(),
                limit=args.limit,
            )
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    payload = result or {
        "found": False,
        "public_reference_id": args.reference_id,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    elif result is None:
        print(f"No agent request found for {args.reference_id}.")
    else:
        request = result["request"]
        assert isinstance(request, dict)
        print(f"Reference: {request['public_reference_id']}")
        print(f"Status: {request['status']}")
        print(f"Event: {request['event_id']}")
        print(f"Actor: {request['actor_id']}")
        print(f"Workspace: {request['workspace_id']}")
        print(f"Started: {request['started_at']}")
        print(f"Events: {result['event_count']}")
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
