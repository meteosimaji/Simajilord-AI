from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import discord
import pytest
from discord.ext import commands

from simajilord.integrations.discord.cogs import ReadAloudCog
from simajilord.integrations.discord.read_aloud import ReadAloudMessageFormatter
from simajilord.runtime import SimajilordRuntime
from simajilord.services.read_aloud import (
    ReadAloudMode,
    ReadAloudRoute,
    ReadAloudService,
)


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
async def test_join_announcement_connects_and_uses_shared_speech_api(tmp_path) -> None:
    cog, runtime, member, destination = await _announcement_cog(tmp_path)

    await cog._announce_voice_transition(
        member,
        before_channel=None,
        after_channel=destination,
    )

    runtime.audio.connect.assert_awaited_once_with("1", "55")
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
