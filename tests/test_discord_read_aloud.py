from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import discord
import pytest
from discord.ext import commands

from simajilord.agent import ReadAloudAudienceMode
from simajilord.integrations.discord.cogs import (
    _READ_ALOUD_BURST_BATCH_SIZE,
    _READ_ALOUD_BURST_DELAY_SECONDS,
    _READ_ALOUD_VOICE_DEBOUNCE_SECONDS,
    ReadAloudCog,
    _read_aloud_audience_allowed,
)
from simajilord.integrations.discord.permissions import (
    ReadAloudAudienceInspection,
    ReadAloudListenerCheck,
    inspect_read_aloud_audience,
    read_aloud_audience_relation,
)
from simajilord.integrations.discord.read_aloud import (
    ReadAloudMessageFormatter,
    ReadAloudMessageText,
    abbreviate_read_aloud_segments,
    merge_read_aloud_messages,
)
from simajilord.runtime import SimajilordRuntime
from simajilord.services.read_aloud import (
    ReadAloudMode,
    ReadAloudRoute,
    ReadAloudService,
)
from simajilord.services.speech import SpeechSegment, SpeechSegmentKind


def _read_permissions(*, readable: bool) -> discord.Permissions:
    return discord.Permissions(
        view_channel=readable,
        read_message_history=readable,
    )


def test_read_aloud_audience_relation_requires_every_current_listener() -> None:
    allowed = SimpleNamespace(id=10, bot=False, display_name="Allowed")
    denied = SimpleNamespace(id=11, bot=False, display_name="Denied")
    bot = SimpleNamespace(id=12, bot=True, display_name="Bot")
    guild = Mock(spec=discord.Guild)
    guild.me = bot
    guild.get_member.side_effect = {10: allowed, 11: denied, 12: bot}.get
    source = Mock(spec=discord.TextChannel)
    source.permissions_for.side_effect = lambda member: _read_permissions(
        readable=member is not denied
    )
    destination = Mock(spec=discord.VoiceChannel)
    destination.voice_states = {10: object(), 11: object(), 12: object()}

    assert (
        read_aloud_audience_relation(guild, source, destination) == "broader"
    )

    destination.voice_states = {10: object(), 12: object()}
    assert (
        read_aloud_audience_relation(guild, source, destination)
        == "same_or_narrower"
    )


def test_read_aloud_audience_relation_ignores_unrelated_incomplete_guild_cache() -> None:
    listener = SimpleNamespace(id=10, bot=False, display_name="Listener")
    bot = SimpleNamespace(id=12, bot=True, display_name="Bot")
    guild = Mock(spec=discord.Guild)
    guild.members = []
    guild.member_count = 100
    guild.chunked = False
    guild.me = bot
    guild.get_member.side_effect = {10: listener, 12: bot}.get
    source = Mock(spec=discord.TextChannel)
    source.permissions_for.return_value = _read_permissions(readable=True)
    destination = Mock(spec=discord.VoiceChannel)
    destination.voice_states = {10: object(), 12: object()}

    assert (
        read_aloud_audience_relation(guild, source, destination)
        == "same_or_narrower"
    )


def test_read_aloud_audience_relation_fails_closed_on_unresolved_listener() -> None:
    bot = SimpleNamespace(id=12, bot=True, display_name="Bot")
    guild = Mock(spec=discord.Guild)
    guild.me = bot
    guild.get_member.return_value = None
    source = Mock(spec=discord.TextChannel)
    destination = Mock(spec=discord.VoiceChannel)
    destination.voice_states = {10: object(), 12: object()}

    inspection = inspect_read_aloud_audience(guild, source, destination)

    assert inspection.relation == "uncertain"
    assert inspection.listeners == (
        ReadAloudListenerCheck(
            member_id=10,
            display_name=None,
            relation="unresolved",
        ),
    )


def test_read_aloud_audience_relation_fails_closed_on_listener_race() -> None:
    listener = SimpleNamespace(id=10, bot=False, display_name="Listener")
    guild = Mock(spec=discord.Guild)
    guild.me = None
    guild.get_member.return_value = listener
    source = Mock(spec=discord.TextChannel)
    source.permissions_for.return_value = _read_permissions(readable=True)

    class RacingDestination:
        def __init__(self) -> None:
            self.reads = 0

        @property
        def voice_states(self) -> dict[int, object]:
            self.reads += 1
            return {10: object()} if self.reads == 1 else {10: object(), 11: object()}

    destination = cast(discord.VoiceChannel, RacingDestination())

    inspection = inspect_read_aloud_audience(guild, source, destination)

    assert inspection.relation == "uncertain"
    assert inspection.stable is False


def test_read_aloud_audience_relation_checks_private_thread_membership() -> None:
    listener = SimpleNamespace(id=10, bot=False, display_name="Listener")
    guild = Mock(spec=discord.Guild)
    guild.me = None
    guild.get_member.return_value = listener
    source = Mock(spec=discord.Thread)
    source.type = discord.ChannelType.private_thread
    source.members = []
    source.permissions_for.return_value = _read_permissions(readable=True)
    destination = Mock(spec=discord.VoiceChannel)
    destination.voice_states = {10: object()}

    assert read_aloud_audience_relation(guild, source, destination) == "broader"


def test_read_aloud_audience_policy_is_reversible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Mock(spec=discord.TextChannel)
    source.id = 2
    destination = Mock(spec=discord.VoiceChannel)
    destination.id = 3
    guild = SimpleNamespace(id=1)
    message = cast(
        discord.Message,
        SimpleNamespace(guild=guild, channel=source),
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.cogs.inspect_read_aloud_audience",
        lambda *_: ReadAloudAudienceInspection(
            relation="broader",
            listeners=(
                ReadAloudListenerCheck(
                    member_id=7,
                    display_name="Listener",
                    relation="broader",
                ),
            ),
            stable=True,
        ),
    )

    for mode, allowed in (
        (ReadAloudAudienceMode.ENFORCE, False),
        (ReadAloudAudienceMode.AUDIT, True),
        (ReadAloudAudienceMode.DISABLED, True),
    ):
        runtime = cast(
            SimajilordRuntime,
            SimpleNamespace(settings=SimpleNamespace(read_aloud_audience_mode=mode)),
        )
        assert _read_aloud_audience_allowed(runtime, message, destination) is allowed


def test_read_aloud_audience_policy_blocks_unresolved_listener_in_enforce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Mock(spec=discord.TextChannel)
    source.id = 2
    destination = Mock(spec=discord.VoiceChannel)
    destination.id = 3
    message = cast(
        discord.Message,
        SimpleNamespace(guild=SimpleNamespace(id=1), channel=source),
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.cogs.inspect_read_aloud_audience",
        lambda *_: ReadAloudAudienceInspection(
            relation="uncertain",
            listeners=(
                ReadAloudListenerCheck(
                    member_id=7,
                    display_name=None,
                    relation="unresolved",
                ),
            ),
            stable=True,
        ),
    )
    runtime = cast(
        SimajilordRuntime,
        SimpleNamespace(
            settings=SimpleNamespace(
                read_aloud_audience_mode=ReadAloudAudienceMode.ENFORCE
            )
        ),
    )

    assert _read_aloud_audience_allowed(runtime, message, destination) is False


def _message(
    *,
    content: str = "",
    author_id: int = 10,
    author_name: str = "めてお",
    mentions: list[Any] | None = None,
    role_mentions: list[Any] | None = None,
    channel_mentions: list[Any] | None = None,
    attachments: list[Any] | None = None,
    stickers: list[Any] | None = None,
    reference: Any = None,
) -> discord.Message:
    async def fetch_message(_: int) -> Any:
        return SimpleNamespace(author=SimpleNamespace(display_name="返信元"))

    return cast(
        discord.Message,
        SimpleNamespace(
            guild=SimpleNamespace(id=1),
            channel=SimpleNamespace(id=2, fetch_message=fetch_message),
            author=SimpleNamespace(
                id=author_id,
                display_name=author_name,
                name=author_name,
            ),
            content=content,
            mentions=mentions or [],
            role_mentions=role_mentions or [],
            channel_mentions=channel_mentions or [],
            attachments=attachments or [],
            stickers=stickers or [],
            reference=reference,
        ),
    )


@pytest.mark.asyncio
async def test_formatter_resolves_discord_markup_before_speech(tmp_path) -> None:
    formatter = ReadAloudMessageFormatter(
        ReadAloudService(tmp_path / "read_aloud.json")
    )
    message = _message(
        content=(
            "<@20> <@&30> <#40> <:party_parrot:50> @everyone @here"
        ),
        mentions=[SimpleNamespace(id=20, display_name="田中")],
        role_mentions=[SimpleNamespace(id=30, name="管理者")],
        channel_mentions=[SimpleNamespace(id=40, name="雑談")],
    )

    prepared = await formatter.format(message)

    assert prepared is not None
    assert prepared.text == (
        "めておさん。田中さん 管理者へのメンション 雑談チャンネル "
        "party parrotの絵文字 全員へのメンション "
        "オンラインの皆さんへのメンション"
    )


@pytest.mark.asyncio
async def test_formatter_reads_attachments_without_message_content(tmp_path) -> None:
    formatter = ReadAloudMessageFormatter(
        ReadAloudService(tmp_path / "read_aloud.json")
    )
    message = _message(
        attachments=[
            SimpleNamespace(filename="cat.png", content_type="image/png"),
            SimpleNamespace(filename="clip.mp4", content_type="video/mp4"),
            SimpleNamespace(filename="../report.pdf", content_type="application/pdf"),
        ],
        stickers=[SimpleNamespace(name="にっこり")],
    )

    prepared = await formatter.format(message)

    assert prepared is not None
    assert prepared.text == (
        "めておさん。画像を1件送信しました。動画を1件送信しました。"
        "ファイル、report.pdfを送信しました。スタンプ、にっこりを送信しました"
    )


@pytest.mark.asyncio
async def test_formatter_reads_reply_author_and_uses_dictionary(tmp_path) -> None:
    service = ReadAloudService(tmp_path / "read_aloud.json")
    await service.upsert_dictionary_entry(
        workspace_id="1",
        surface="IUT",
        reading="あいゆーてぃー",
    )
    formatter = ReadAloudMessageFormatter(service)
    message = _message(
        content="IUTを確認しました",
        reference=SimpleNamespace(
            message_id=99,
            resolved=SimpleNamespace(
                author=SimpleNamespace(display_name="アリス"),
            ),
        ),
    )

    prepared = await formatter.format(message)

    assert prepared is not None
    assert prepared.text == (
        "めておさん。アリスさんへの返信。あいゆーてぃーを確認しました"
    )


@pytest.mark.asyncio
async def test_formatter_can_abbreviate_after_markup_and_dictionary_processing(
    tmp_path,
) -> None:
    service = ReadAloudService(tmp_path / "read_aloud.json")
    await service.upsert_dictionary_entry(
        workspace_id="1",
        surface="AI",
        reading="人工知能",
    )
    await service.set_semantic_options(
        workspace_id="1",
        abbreviate_long_messages=True,
        message_character_limit=20,
    )
    formatter = ReadAloudMessageFormatter(service)

    prepared = await formatter.format(_message(content="AI" * 20))

    assert prepared is not None
    assert prepared.segments[0].kind is SpeechSegmentKind.AUTHOR
    assert prepared.segments[-1] == SpeechSegment(
        SpeechSegmentKind.EVENT,
        "以下略",
        cache_key="read-aloud:omission",
    )
    retained = "".join(
        segment.text
        for segment in prepared.segments
        if segment.kind not in {SpeechSegmentKind.AUTHOR, SpeechSegmentKind.EVENT}
    )
    assert len(retained) == 20
    assert "AI" not in retained


def test_abbreviation_leaves_short_text_and_author_cache_unchanged() -> None:
    segments = (
        SpeechSegment(
            SpeechSegmentKind.AUTHOR,
            "めておさん",
            cache_key="author:10:めてお",
        ),
        SpeechSegment(SpeechSegmentKind.BODY, "短い本文"),
    )

    assert abbreviate_read_aloud_segments(segments, maximum=20) is segments


@pytest.mark.asyncio
async def test_formatter_fetches_unresolved_reply_only_when_needed(tmp_path) -> None:
    service = ReadAloudService(tmp_path / "read_aloud.json")
    formatter = ReadAloudMessageFormatter(service)
    unresolved = _message(
        content="了解",
        reference=SimpleNamespace(message_id=99, resolved=None),
    )

    prepared = await formatter.format(unresolved)

    assert prepared is not None
    assert prepared.text == "めておさん。返信元さんへの返信。了解"

    await service.set_semantic_options(workspace_id="1", replies=False)
    disabled = await formatter.format(
        _message(
            content="了解",
            author_id=11,
            reference=SimpleNamespace(message_id=99, resolved=None),
        )
    )
    assert disabled is not None
    assert disabled.text == "めておさん。了解"


@pytest.mark.asyncio
async def test_formatter_avoids_repeating_same_author_until_timeout(
    tmp_path,
) -> None:
    now = iter((10.0, 20.0, 120.0))
    formatter = ReadAloudMessageFormatter(
        ReadAloudService(tmp_path / "read_aloud.json"),
        repeat_author_after_seconds=90,
        clock=lambda: next(now),
    )

    first = await formatter.format(_message(content="一つ目"))
    second = await formatter.format(_message(content="二つ目"))
    third = await formatter.format(_message(content="三つ目"))

    assert first is not None and first.text == "めておさん。一つ目"
    assert second is not None and second.text == "二つ目"
    assert third is not None and third.text == "めておさん。三つ目"


def test_short_burst_compacts_exact_consecutive_spam_without_reordering() -> None:
    author = SpeechSegment(SpeechSegmentKind.AUTHOR, "めておさん")
    hello = SpeechSegment(SpeechSegmentKind.BODY, "こんにちは")
    other = SpeechSegment(SpeechSegmentKind.BODY, "別の内容")

    merged = merge_read_aloud_messages(
        (
            ("10", ReadAloudMessageText((author, hello), "一つ目")),
            ("10", ReadAloudMessageText((hello,), "二つ目")),
            ("11", ReadAloudMessageText((other,), "三つ目")),
        )
    )

    assert merged.title == "3件のメッセージ"
    assert merged.text == (
        "めておさん。こんにちは。同じ内容を2回送信しました。別の内容"
    )


@pytest.mark.asyncio
async def test_read_aloud_burst_uses_low_latency_debounce(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(
        "simajilord.integrations.discord.cogs.asyncio.sleep",
        record_sleep,
    )
    runtime = Mock(spec=SimajilordRuntime)
    runtime.read_aloud = ReadAloudService(tmp_path / "read_aloud.json")
    cog = ReadAloudCog(cast(commands.Bot, object()), runtime)
    key = (1, 2)
    message = SimpleNamespace(id=1, author=SimpleNamespace(id=10))
    prepared = ReadAloudMessageText(
        (SpeechSegment(SpeechSegmentKind.BODY, "すぐに読む"),),
        "Message",
    )
    cog._message_bursts[key] = [message]
    cog._message_formatter.format = AsyncMock(return_value=prepared)
    deliver = AsyncMock()
    monkeypatch.setattr(cog, "_deliver_read_aloud", deliver)

    await cog._flush_message_burst(key)

    assert delays == [_READ_ALOUD_BURST_DELAY_SECONDS]
    assert _READ_ALOUD_BURST_DELAY_SECONDS <= 0.1
    deliver.assert_awaited_once_with(message, prepared)


@pytest.mark.asyncio
async def test_read_aloud_burst_keeps_snowflake_order_and_drains_every_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "simajilord.integrations.discord.cogs._READ_ALOUD_BURST_DELAY_SECONDS",
        0.0,
    )
    runtime = Mock(spec=SimajilordRuntime)
    runtime.read_aloud = ReadAloudService(tmp_path / "read_aloud.json")
    cog = ReadAloudCog(cast(commands.Bot, object()), runtime)
    key = (1, 2)
    messages = [
        SimpleNamespace(
            id=index,
            channel=SimpleNamespace(id=2 + (index % 2)),
            author=SimpleNamespace(id=10),
        )
        for index in range(1, _READ_ALOUD_BURST_BATCH_SIZE + 2)
    ]
    cog._message_bursts[key] = [*reversed(messages)]

    async def format_message(message) -> ReadAloudMessageText:
        if message.id == 1:
            await asyncio.sleep(0.01)
        return ReadAloudMessageText(
            (SpeechSegment(SpeechSegmentKind.BODY, f"message-{message.id}"),),
            f"Message {message.id}",
        )

    cog._message_formatter.format = AsyncMock(side_effect=format_message)
    deliver = AsyncMock()
    monkeypatch.setattr(cog, "_deliver_read_aloud", deliver)

    await cog._flush_message_burst(key)

    assert deliver.await_count == 2
    first_message, first_prepared = deliver.await_args_list[0].args
    second_message, second_prepared = deliver.await_args_list[1].args
    assert first_message.id == 1
    assert [segment.text for segment in first_prepared.segments] == [
        f"message-{index}"
        for index in range(1, _READ_ALOUD_BURST_BATCH_SIZE + 1)
    ]
    assert second_message.id == _READ_ALOUD_BURST_BATCH_SIZE + 1
    assert [segment.text for segment in second_prepared.segments] == [
        f"message-{_READ_ALOUD_BURST_BATCH_SIZE + 1}"
    ]
    assert cog._message_bursts == {}


@pytest.mark.asyncio
async def test_read_aloud_formats_in_parallel_but_delivers_in_snowflake_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "simajilord.integrations.discord.cogs._READ_ALOUD_BURST_DELAY_SECONDS",
        0.0,
    )
    runtime = Mock(spec=SimajilordRuntime)
    runtime.read_aloud = ReadAloudService(tmp_path / "read_aloud.json")
    cog = ReadAloudCog(cast(commands.Bot, object()), runtime)
    key = (1, 55)
    messages = [
        SimpleNamespace(id=index, author=SimpleNamespace(id=10))
        for index in (2, 1)
    ]
    cog._message_bursts[key] = messages
    both_started = asyncio.Event()
    release = asyncio.Event()
    started: list[int] = []

    async def format_message(message) -> ReadAloudMessageText:
        started.append(message.id)
        if len(started) == 2:
            both_started.set()
        await release.wait()
        return ReadAloudMessageText(
            (SpeechSegment(SpeechSegmentKind.BODY, f"message-{message.id}"),),
            f"Message {message.id}",
        )

    cog._message_formatter.format = AsyncMock(side_effect=format_message)
    deliver = AsyncMock()
    monkeypatch.setattr(cog, "_deliver_read_aloud", deliver)
    flush = asyncio.create_task(cog._flush_message_burst(key))

    await asyncio.wait_for(both_started.wait(), timeout=1)
    release.set()
    await flush

    assert started == [1, 2]
    message, prepared = deliver.await_args.args
    assert message.id == 1
    assert [segment.text for segment in prepared.segments] == [
        "message-1",
        "message-2",
    ]


@pytest.mark.asyncio
async def test_multiple_read_aloud_sources_share_one_destination_fifo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    service = ReadAloudService(tmp_path / "read_aloud.json")
    await service.configure_sources(
        workspace_id="1",
        text_channel_ids=("2", "3"),
        audio_destination_id="55",
        mode=ReadAloudMode.QUEUE,
    )
    runtime = Mock(spec=SimajilordRuntime)
    runtime.read_aloud = service
    destination = Mock(spec=discord.VoiceChannel)
    destination.id = 55
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    guild.get_channel.return_value = destination
    author = SimpleNamespace(id=10, bot=False)

    def source_message(message_id: int, channel_id: int) -> discord.Message:
        return cast(
            discord.Message,
            SimpleNamespace(
                id=message_id,
                guild=guild,
                channel=SimpleNamespace(id=channel_id),
                author=author,
                webhook_id=None,
            ),
        )

    monkeypatch.setattr(
        "simajilord.integrations.discord.cogs._read_aloud_audience_allowed",
        lambda *_args: True,
    )
    cog = ReadAloudCog(cast(commands.Bot, object()), runtime)
    first = source_message(100, 2)
    second = source_message(101, 3)

    await cog.on_message(first)
    await cog.on_message(second)

    assert tuple(cog._message_bursts) == ((1, 55),)
    assert cog._message_bursts[(1, 55)] == [first, second]
    assert len(cog._message_burst_tasks) == 1
    await cog.cog_unload()


@pytest.mark.asyncio
async def test_message_does_not_reconnect_read_aloud_to_an_empty_voice_channel(
    tmp_path,
) -> None:
    service = ReadAloudService(tmp_path / "read_aloud.json")
    await service.configure(
        ReadAloudRoute("1", "2", "55", ReadAloudMode.QUEUE)
    )
    destination = Mock(spec=discord.VoiceChannel)
    destination.id = 55
    destination.members = []
    destination.voice_states = {}
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    guild.get_channel.return_value = destination
    author = SimpleNamespace(
        id=10,
        display_name="めてお",
        name="めてお",
        bot=False,
    )
    message = cast(
        discord.Message,
        SimpleNamespace(
            id=99,
            guild=guild,
            channel=SimpleNamespace(id=2),
            author=author,
            content="誰もいないVCへ接続しない",
            mentions=[],
            role_mentions=[],
            channel_mentions=[],
            attachments=[],
            stickers=[],
            reference=None,
            webhook_id=None,
        ),
    )
    runtime = Mock(spec=SimajilordRuntime)
    runtime.read_aloud = service
    runtime.registry = Mock()
    runtime.registry.invoke = AsyncMock()
    runtime.audio = Mock()
    session = Mock()
    session.voice_activation_required = False
    session.current = None
    session.output.connected = False
    runtime.audio.get_or_create.return_value = session
    cog = ReadAloudCog(cast(commands.Bot, object()), runtime)

    await cog.on_message(message)

    runtime.audio.get_or_create.assert_not_called()
    runtime.registry.invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_message_content_does_not_reconnect_with_an_allowed_listener(
    tmp_path,
) -> None:
    service = ReadAloudService(tmp_path / "read_aloud.json")
    await service.configure(ReadAloudRoute("1", "2", "55", ReadAloudMode.QUEUE))
    listener = SimpleNamespace(id=7, bot=False, display_name="Listener")
    bot_member = SimpleNamespace(id=99, bot=True, display_name="Bot")
    source = Mock(spec=discord.TextChannel)
    source.id = 2
    source.permissions_for.return_value = _read_permissions(readable=True)
    destination = Mock(spec=discord.VoiceChannel)
    destination.id = 55
    destination.voice_states = {7: object(), 99: object()}
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    guild.me = bot_member
    guild.get_member.side_effect = {7: listener, 99: bot_member}.get
    guild.get_channel.return_value = destination
    author = SimpleNamespace(id=10, bot=False)
    message = cast(
        discord.Message,
        SimpleNamespace(id=99, guild=guild, channel=source, author=author),
    )
    runtime = Mock(spec=SimajilordRuntime)
    runtime.settings = SimpleNamespace(read_aloud_audience_mode="enforce")
    runtime.read_aloud = service
    runtime.registry = Mock()
    runtime.registry.invoke = AsyncMock()
    runtime.audio = Mock()
    runtime.audio.connect = AsyncMock()
    session = Mock()
    session.voice_activation_required = False
    session.current = None
    session.output.connected = False
    runtime.audio.get_or_create.return_value = session
    cog = ReadAloudCog(cast(commands.Bot, object()), runtime)

    await cog._deliver_read_aloud(
        message,
        ReadAloudMessageText(
            (SpeechSegment(SpeechSegmentKind.BODY, "接続しない"),),
            "Message",
        ),
    )

    runtime.audio.get_or_create.assert_called_once()
    runtime.audio.connect.assert_not_awaited()
    runtime.registry.invoke.assert_not_awaited()


async def _announcement_cog(tmp_path) -> tuple[
    ReadAloudCog,
    SimajilordRuntime,
    discord.Member,
    discord.VoiceChannel,
]:
    service = ReadAloudService(tmp_path / "read_aloud.json")
    await service.configure(
        ReadAloudRoute("1", "2", "55", ReadAloudMode.QUEUE)
    )
    await service.set_announcements(
        workspace_id="1",
        join=True,
        leave=True,
        move=True,
    )
    runtime = Mock(spec=SimajilordRuntime)
    runtime.read_aloud = service
    runtime.registry = Mock()
    runtime.registry.invoke = AsyncMock()
    runtime.audio = Mock()
    runtime.audio.connect = AsyncMock()
    session = Mock()
    session.has_music = False
    session.voice_activation_required = False
    session.destination_id = "55"
    session.output.connected = False
    runtime.audio.get_or_create.return_value = session

    destination = Mock(spec=discord.VoiceChannel)
    destination.id = 55
    destination.name = "一般"
    listener = Mock(spec=discord.Member)
    listener.bot = False
    destination.members = [listener]
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    guild.get_channel.return_value = destination
    member = Mock(spec=discord.Member)
    member.id = 7
    member.bot = False
    member.display_name = "アリス"
    member.name = "alice"
    member.guild = guild
    cog = ReadAloudCog(cast(commands.Bot, object()), runtime)
    return cog, runtime, member, destination


@pytest.mark.asyncio
async def test_join_announcement_does_not_connect_a_passive_voice_route(tmp_path) -> None:
    cog, runtime, member, destination = await _announcement_cog(tmp_path)

    await cog._announce_voice_transition(
        member,
        before_channel=None,
        after_channel=destination,
    )

    runtime.audio.connect.assert_not_awaited()
    runtime.registry.invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_join_announcement_uses_shared_speech_api_when_already_connected(
    tmp_path,
) -> None:
    cog, runtime, member, destination = await _announcement_cog(tmp_path)
    runtime.audio.get_or_create.return_value.output.connected = True

    await cog._announce_voice_transition(
        member,
        before_channel=None,
        after_channel=destination,
    )

    runtime.audio.connect.assert_not_awaited()
    runtime.registry.invoke.assert_awaited_once()
    capability, request, context = runtime.registry.invoke.await_args.args
    assert capability == "speech.speak"
    assert request.text == "アリスさんがボイスチャンネルに参加しました"
    assert request.title == "VCの入退室通知"
    assert context.workspace_id == "1"


@pytest.mark.asyncio
async def test_voice_transition_uses_low_latency_debounce(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(
        "simajilord.integrations.discord.cogs.asyncio.sleep",
        record_sleep,
    )
    cog, _runtime, member, destination = await _announcement_cog(tmp_path)
    key = (member.guild.id, member.id)
    cog._voice_transitions[key] = (member, None, destination)
    announce = AsyncMock()
    monkeypatch.setattr(cog, "_announce_voice_transition", announce)

    await cog._flush_voice_transition(key)

    assert delays == [_READ_ALOUD_VOICE_DEBOUNCE_SECONDS]
    assert _READ_ALOUD_VOICE_DEBOUNCE_SECONDS <= 0.15
    announce.assert_awaited_once_with(
        member,
        before_channel=None,
        after_channel=destination,
    )


@pytest.mark.asyncio
async def test_leave_announcement_is_not_generated_for_an_empty_channel(
    tmp_path,
) -> None:
    cog, runtime, member, destination = await _announcement_cog(tmp_path)
    destination.members = []

    await cog._announce_voice_transition(
        member,
        before_channel=destination,
        after_channel=None,
    )

    runtime.audio.connect.assert_not_awaited()
    runtime.registry.invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_move_announcement_names_both_channels(tmp_path) -> None:
    cog, runtime, member, destination = await _announcement_cog(tmp_path)
    other = Mock(spec=discord.VoiceChannel)
    other.id = 77
    other.name = "ゲーム"
    runtime.audio.get_or_create.return_value.output.connected = True

    await cog._announce_voice_transition(
        member,
        before_channel=other,
        after_channel=destination,
    )

    request = runtime.registry.invoke.await_args.args[1]
    assert request.text == "アリスさんが、ゲームから一般へ移動しました"
