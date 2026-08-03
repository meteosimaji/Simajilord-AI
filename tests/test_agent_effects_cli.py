from __future__ import annotations

import asyncio
import json
from pathlib import Path

from simajilord.agent import ActionReceiptStore, ExternalEffectStatus
from simajilord.core import InvocationContext
from simajilord.diagnostics.agent_effects import main


async def _unknown_effect(path: Path) -> str:
    store = ActionReceiptStore(path)
    context = InvocationContext(
        actor_id="actor",
        workspace_id="guild",
        transport="agent",
        request_id="request",
        provider_thread_id="thread",
        provider_turn_id="turn",
        tool_call_id="tool",
    )
    planned = await store.plan_external_effect(
        capability="discord.send_message",
        request={
            "channel_id": "channel",
            "content": "body-that-must-not-appear-in-diagnostics",
        },
        context=context,
        authorization_reference="discord:message:event",
    )
    await store.dispatch_external_effect(planned.effect_id)
    await store.mark_external_effect_unknown(planned.effect_id)
    return planned.effect_id


def test_operator_cli_lists_and_reconciles_unknown_without_replay(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "actions.sqlite3"
    effect_id = asyncio.run(_unknown_effect(path))

    assert main(["--database", str(path), "list", "--json"]) == 0
    listed_output = capsys.readouterr().out
    listed = json.loads(listed_output)
    assert [item["effect_id"] for item in listed] == [effect_id]
    assert listed[0]["status"] == "unknown"
    assert "body-that-must-not-appear-in-diagnostics" not in listed_output

    assert main(["--database", str(path), "reconcile", effect_id]) == 1
    assert "--yes" in capsys.readouterr().out
    current = asyncio.run(
        ActionReceiptStore(
            path,
            recover_interrupted=False,
        ).external_effect(effect_id)
    )
    assert current is not None
    assert current.status is ExternalEffectStatus.UNKNOWN

    assert (
        main(
            [
                "--database",
                str(path),
                "reconcile",
                effect_id,
                "--yes",
                "--json",
            ]
        )
        == 0
    )
    reconciled = json.loads(capsys.readouterr().out)
    assert reconciled["status"] == "reconciled"
    assert reconciled["effect_id"] == effect_id

    stored = asyncio.run(
        ActionReceiptStore(
            path,
            recover_interrupted=False,
        ).external_effect(effect_id)
    )
    assert stored is not None
    assert stored.status is ExternalEffectStatus.RECONCILED
