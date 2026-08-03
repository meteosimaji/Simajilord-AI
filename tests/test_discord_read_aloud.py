from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import discord
import pytest
from discord.ext import commands

from simajilord.agent import ReadAloudAudienceMode
from simajilord.integrations.discord.cogs import (
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
