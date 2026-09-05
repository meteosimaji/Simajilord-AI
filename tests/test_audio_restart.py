from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import discord
import pytest
from test_audio import FakeOutput

from simajilord.domain.audio import AudioItem, LoopMode
from simajilord.integrations.discord.bot import SimajilordDiscordBot
from simajilord.services.audio import AudioSessionManager
from simajilord.services.audio_state import AudioStateStore, StoredAudioSession


@pytest.mark.asyncio
async def test_active_audio_restart_preserves_position_and_recovery_intent(tmp_path: Path) -> None:
    path = tmp_path / "audio.json"
    output = FakeOutput()
    manager = AudioSessionManager(
        max_active=2, max_pending_speech=3, state_store=AudioStateStore(path)
    )
    session = manager.get_or_create("1", lambda: output)
    await session.connect("55")
    await session.enqueue(
        AudioItem(
            "old-signed-url",
            "Music",
            "https://example.com/music",
            start_seconds=23.0,
        )
    )
    await manager.close()
    (saved,) = AudioStateStore(path).all()
    assert saved.resume_on_restart
    assert saved.items[0].start_seconds >= 23.0
    assert "old-signed-url" not in path.read_text()

    restored_output = FakeOutput()
    restored_output.connected = False
    restored = AudioSessionManager(
        max_active=2, max_pending_speech=3, state_store=AudioStateStore(path)
    )
    (session,) = restored.restore(lambda _: restored_output)
    assert session.restart_recovery_pending
    await restored.persist_restored_sessions((session,))
    await restored.close()
    # A failed or interrupted reconnect must not consume the recovery intent.
    (saved_again,) = AudioStateStore(path).all()
    assert saved_again.resume_on_restart
    assert not saved_again.voice_activation_required


@pytest.mark.asyncio
@pytest.mark.parametrize("held", [False, True])
async def test_restart_only_reconnects_previously_active_sessions(
    tmp_path: Path,
    held: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = StoredAudioSession(
        workspace_id="1",
        destination_id="55",
        waiting_actor_ids=(),
        loop_mode=LoopMode.NONE,
        auto_leave=True,
        speed=1.0,
        pitch=1.0,
        items=(),
        history=(),
        resume_on_restart=True,
        voice_activation_required=held,
    )
    store = AudioStateStore(tmp_path / "audio.json")
    await store.put(state)
    output = FakeOutput()
    output.connected = False
    manager = AudioSessionManager(max_active=2, max_pending_speech=3, state_store=store)
    channel = Mock(spec=discord.VoiceChannel)
    guild = SimpleNamespace(get_channel=lambda _: channel)
    bot = SimpleNamespace(runtime=SimpleNamespace(audio=manager), get_guild=lambda _: guild)
    monkeypatch.setattr("simajilord.integrations.discord.bot.DiscordAudioOutput", lambda *_: output)
    await SimajilordDiscordBot._restore_audio_sessions(bot)
    assert output.connected is (not held)
    await manager.close()


@pytest.mark.asyncio
async def test_paused_and_suspended_audio_stay_held(tmp_path: Path) -> None:
    output = FakeOutput()
    manager = AudioSessionManager(max_active=2, max_pending_speech=3)
    session = manager.get_or_create("1", lambda: output)
    await session.connect("55")
    output.paused = True
    assert not (await session.persisted_state()).resume_on_restart
    output.paused = False
    await session.suspend()
    state = await session.persisted_state()
    assert not state.resume_on_restart
    assert state.voice_activation_required
    # Legacy files have no explicit evidence of an active connection.
    store = AudioStateStore(tmp_path / "audio.json")
    await store.put(replace(state, resume_on_restart=False))
    await store.flush()
    assert not AudioStateStore(store.path).all()[0].resume_on_restart
    await manager.close()


@pytest.mark.asyncio
async def test_reconnect_failure_retains_saved_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AudioStateStore(tmp_path / "audio.json")
    await store.put(
        StoredAudioSession(
            workspace_id="1",
            destination_id="55",
            waiting_actor_ids=(),
            loop_mode=LoopMode.NONE,
            auto_leave=True,
            speed=1.0,
            pitch=1.0,
            items=(),
            history=(),
            resume_on_restart=True,
        )
    )
    manager = AudioSessionManager(max_active=2, max_pending_speech=3, state_store=store)
    output = FakeOutput()
    output.connected = False
    output.connect = AsyncMock(side_effect=RuntimeError("unavailable"))
    bot = SimpleNamespace(
        runtime=SimpleNamespace(audio=manager),
        get_guild=lambda _: SimpleNamespace(get_channel=lambda _: Mock(spec=discord.VoiceChannel)),
    )
    monkeypatch.setattr("simajilord.integrations.discord.bot.DiscordAudioOutput", lambda *_: output)
    await SimajilordDiscordBot._restore_audio_sessions(bot)
    await manager.close()
    assert AudioStateStore(store.path).all()[0].resume_on_restart
