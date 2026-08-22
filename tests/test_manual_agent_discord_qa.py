from __future__ import annotations

import importlib
from pathlib import Path

import pytest


class _ProviderConstructionReached(Exception):
    """Stop after the manual QA has built and registered all local contracts."""


@pytest.mark.asyncio
async def test_manual_agent_discord_qa_contracts_are_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    manual_agent_discord_qa = importlib.import_module(
        "scripts.manual_agent_discord_qa"
    )

    def stop_before_live_provider(**_kwargs: object) -> None:
        raise _ProviderConstructionReached

    monkeypatch.setattr(
        manual_agent_discord_qa,
        "CodexAppServerProvider",
        stop_before_live_provider,
    )

    with pytest.raises(_ProviderConstructionReached):
        await manual_agent_discord_qa.run("status")
