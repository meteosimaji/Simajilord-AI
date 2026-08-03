"""Body-free operator CLI for uncertain agent external effects."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from pathlib import Path

from simajilord.agent import (
    ActionReceiptStore,
    ExternalEffectRecord,
    ExternalEffectStatus,
)


def _parser() -> argparse.ArgumentParser:
    data_dir = Path(os.getenv("DATA_DIR", ".data")).expanduser().resolve()
    parser = argparse.ArgumentParser(
        prog="simajilord-agent-effects",
        description=(
            "Inspect body-free external-effect metadata and explicitly close UNKNOWN "
            "records without replaying provider calls."
        ),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=data_dir / "agent_actions.sqlite3",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list",
        help="List UNKNOWN effects by default.",
    )
    filter_group = list_parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--status",
        choices=tuple(status.value for status in ExternalEffectStatus),
        default=ExternalEffectStatus.UNKNOWN.value,
    )
    filter_group.add_argument(
        "--all",
        action="store_true",
        help="List every status instead of only UNKNOWN effects.",
    )
    list_parser.add_argument("--limit", type=int, default=100)
    list_parser.add_argument("--json", action="store_true")

    show_parser = subparsers.add_parser(
        "show",
        help="Show one body-free effect record.",
    )
    show_parser.add_argument("effect_id")
    show_parser.add_argument("--json", action="store_true")

    reconcile_parser = subparsers.add_parser(
        "reconcile",
        help=(
            "Mark one operator-verified UNKNOWN record reconciled; this never retries "
            "the provider call. Stop the bot first."
        ),
    )
    reconcile_parser.add_argument("effect_id")
    reconcile_parser.add_argument(
        "--action-id",
        help="Optional existing Action Receipt used as positive evidence.",
    )
    reconcile_parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the bot is stopped and the external outcome was checked.",
    )
    reconcile_parser.add_argument("--json", action="store_true")
    return parser


def _payload(effect: ExternalEffectRecord) -> dict[str, object]:
    return {
        "effect_id": effect.effect_id,
        "capability": effect.capability,
        "actor_id": effect.actor_id,
        "workspace_id": effect.workspace_id,
        "transport": effect.transport,
        "request_id": effect.request_id,
        "provider_thread_id": effect.provider_thread_id,
        "provider_turn_id": effect.provider_turn_id,
        "tool_call_id": effect.tool_call_id,
        "arguments_fingerprint": effect.arguments_fingerprint,
        "target_ids": dict(effect.target_ids),
        "authorization_reference": effect.authorization_reference,
        "summary": effect.summary,
        "status": effect.status.value,
        "action_id": effect.action_id,
        "created_at": effect.created_at.isoformat(),
        "updated_at": effect.updated_at.isoformat(),
        "expires_at": effect.expires_at.isoformat(),
    }


async def _run(
    args: argparse.Namespace,
) -> ExternalEffectRecord | tuple[ExternalEffectRecord, ...]:
    database = args.database.expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"Action database does not exist: {database}")
    # Diagnostics must not reinterpret a currently running process's DISPATCHED
    # row as a restart. Runtime composition keeps recovery enabled by default.
    store = ActionReceiptStore(database, recover_interrupted=False)
    if args.command == "list":
        status = None if args.all else ExternalEffectStatus(args.status)
        return await store.external_effects(status=status, limit=args.limit)
    if args.command == "show":
        effect = await store.external_effect(args.effect_id)
        if effect is None:
            raise ValueError("external effect does not exist")
        return effect
    if args.command == "reconcile":
        if not args.yes:
            raise ValueError(
                "refusing to reconcile without --yes after stopping the bot and "
                "checking the provider outcome"
            )
        return await store.reconcile_external_effect(
            args.effect_id,
            action_id=args.action_id,
        )
    raise AssertionError(f"Unhandled external-effect command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if hasattr(args, "limit") and not 1 <= args.limit <= 1_000:
        print("Error: --limit must be between 1 and 1000")
        return 2
    try:
        result = asyncio.run(_run(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    if isinstance(result, tuple):
        if args.json:
            print(
                json.dumps(
                    [_payload(effect) for effect in result],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        elif not result:
            print("No matching external effects.")
        else:
            for effect in result:
                targets = ",".join(
                    f"{name}={value}" for name, value in effect.target_ids
                ) or "none"
                print(
                    f"{effect.effect_id}  {effect.status.value:<10} "
                    f"{effect.capability}  targets={targets}  "
                    f"updated={effect.updated_at.isoformat()}"
                )
        return 0

    if args.json:
        print(
            json.dumps(
                _payload(result),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(_payload(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
