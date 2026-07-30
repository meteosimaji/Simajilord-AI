from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import Mock

import discord
import pytest

from simajilord.agent.providers import CodexAppServerProvider
from simajilord.config import load_settings
from simajilord.core import CapabilityRegistry
from simajilord.integrations.discord.capabilities import build_discord_endpoints
from simajilord.runtime import SimajilordRuntime

# These endpoints exist for host behavior, backward compatibility, or Action Receipt Undo.
# They are deliberately registered but are not offered through AgentToolCatalog.
INTERNAL_DISCORD_CAPABILITIES = frozenset(
    {
        "discord.expand_message",
        "discord.delete_created_role",
        "discord.delete_created_channel",
        "discord.delete_own_messages",
        "discord.control_audio",
        "discord.manage_read_aloud",
    }
)


# Every model-facing endpoint gets an independently written natural Japanese and English
# discovery query. The cases intentionally describe user goals instead of echoing capability
# identifiers, so a descriptor rename cannot make this test pass accidentally.
DISCOVERY_CASES: dict[str, tuple[str, str]] = {
    "discord.list_servers": (
        "私とBOTが参加しているサーバーを一覧にして",
        "list the Discord servers shared by me and the bot",
    ),
    "discord.inspect_server": (
        "このサーバーの構成と基本情報を確認して",
        "inspect this server structure and basic information",
    ),
    "discord.inspect_user": (
        "このユーザーの公開プロフィールと役職を見て",
        "inspect this user's public profile and server roles",
    ),
    "discord.list_voice_states": (
        "今VCに誰がいるか一覧で見せて",
        "list everyone currently connected to voice or stage",
    ),
    "discord.list_roles": (
        "このサーバーにある既存ロールを一覧にして",
        "list the existing roles in this server",
    ),
    "discord.list_channels": (
        "このサーバーのチャンネル一覧を見せて",
        "list the channels and threads in this server",
    ),
    "discord.list_archived_threads": (
        "アーカイブ済みのフォーラム投稿を一覧にして",
        "list archived threads and forum posts",
    ),
    "discord.read_messages": (
        "このチャンネルの最近の会話を順番に読んで",
        "read the recent conversation in this channel",
    ),
    "discord.search_messages": (
        "過去のメッセージから障害報告を検索して",
        "search message history for the incident report",
    ),
    "discord.get_message": (
        "このメッセージIDの原文を取得して",
        "get the original message by its ID",
    ),
    "discord.translate_message": (
        "このメッセージをドイツ語に翻訳して",
        "translate this Discord message into German",
    ),
    "discord.post_expanded_message": (
        "このメッセージを文章のまま引用して別チャンネルに再投稿して",
        "repost this message as a text quotation with a jump link",
    ),
    "discord.create_quote_image": (
        "このメッセージをスクショ風の引用画像にして",
        "render this message as a screenshot-style quote image",
    ),
    "discord.view_custom_emoji": (
        "このカスタム絵文字のアニメーションを見せて",
        "show the full animation of this custom emoji",
    ),
    "discord.view_sticker": (
        "この動くスタンプの3フレーム目を見せて",
        "show frame three of this animated sticker",
    ),
    "discord.analyze_attachment": (
        "この添付画像がAI生成かディープフェイクか調べて",
        "analyze this attachment for AI-generated or deepfake signals",
    ),
    "discord.import_attachment": (
        "この添付PDFを作業領域に取り込んで",
        "import this PDF attachment into the workspace",
    ),
    "discord.view_image_attachment": (
        "この添付画像を実際に見て内容を確認して",
        "inspect this attached image with model vision",
    ),
    "discord.add_reaction": (
        "このメッセージに絵文字でリアクションを付けて",
        "add an emoji reaction to this message",
    ),
    "discord.remove_own_reaction": (
        "このメッセージからBOT自身のリアクションを外して",
        "remove the bot's own reaction from this message",
    ),
    "discord.send_message": (
        "このチャンネルに通常メッセージを一件送って",
        "send one plain message to this channel",
    ),
    "discord.send_embed": (
        "結果を見やすい埋め込みカードで投稿して",
        "post the result as a structured embed card",
    ),
    "discord.reply_message": (
        "このメッセージへ返信して",
        "reply directly to this Discord message",
    ),
    "discord.edit_own_message": (
        "さっきBOTが送ったメッセージを訂正して",
        "edit the message previously authored by the bot",
    ),
    "discord.pin_message": (
        "このメッセージをピン留めして",
        "pin this message in the channel",
    ),
    "discord.unpin_message": (
        "このメッセージのピン留めを解除して",
        "unpin this message from the channel",
    ),
    "discord.create_thread": (
        "この議論を新しいスレッドに分けて",
        "create a new thread for this discussion",
    ),
    "discord.update_thread": (
        "このスレッドの名前を変更して",
        "rename this Discord thread",
    ),
    "discord.add_thread_member": (
        "田中さんをこのスレッドに招待して",
        "invite this member into the thread",
    ),
    "discord.remove_thread_member": (
        "田中さんをこのスレッドから外して",
        "remove this member from the thread",
    ),
    "discord.create_forum_post": (
        "このバグ報告を新しいフォーラム投稿にして",
        "create a forum post for this bug report",
    ),
    "discord.create_role": (
        "新しいサーバーロールを作って",
        "create a new server role",
    ),
    "discord.assign_role": (
        "田中さんにこのロールを付与して",
        "assign this role to the member",
    ),
    "discord.remove_role": (
        "田中さんからこのロールを外して",
        "remove this role from the member",
    ),
    "discord.update_channel_settings": (
        "このテキストチャンネルのトピックを編集して",
        "edit this text channel topic",
    ),
    "discord.create_channel": (
        "新しいテキストチャンネルを作って",
        "create a new text channel",
    ),
    "discord.set_timeout": (
        "このメンバーを10分間タイムアウトして",
        "timeout this member for ten minutes",
    ),
    "discord.delete_message": (
        "違反しているこの一件のメッセージを削除して",
        "moderate and delete this one offending message",
    ),
    "discord.bulk_delete_messages": (
        "指定した荒らしメッセージをまとめて一括削除して",
        "bulk delete these exact spam message IDs",
    ),
    "discord.kick_member": (
        "このメンバーをサーバーからキックして",
        "kick this member from the server",
    ),
    "discord.ban_member": (
        "この荒らしをサーバーからBANして",
        "ban this abusive member from the server",
    ),
    "discord.unban_member": (
        "このユーザーのBANを解除して",
        "unban this user from the server",
    ),
    "discord.delete_own_message": (
        "BOT自身が送ったこの投稿を消して",
        "delete this message authored by the bot",
    ),
    "discord.send_files": (
        "この3つのファイルを添付してまとめて送って",
        "send these three workspace files as attachments",
    ),
    "discord.send_file": (
        "このファイルを一つ添付して送って",
        "send this one workspace file as an attachment",
    ),
    "discord.create_poll": (
        "この質問で投票アンケートを作って",
        "create a native poll for this question",
    ),
    "discord.connect_voice": (
        "私がいるVCにBOTを参加させて",
        "connect the bot to my voice channel",
    ),
    "discord.play_audio": (
        "この公開URLの曲をVCで流して",
        "play music from this public URL in voice",
    ),
    "discord.play_attachment": (
        "この添付動画の音声をVCで流して",
        "play the audio from this video attachment in voice",
    ),
    "discord.pause_audio": (
        "今流れている音楽を一時停止して",
        "pause the music currently playing",
    ),
    "discord.resume_audio": (
        "一時停止した音楽を続きから再開して",
        "resume the paused music",
    ),
    "discord.skip_audio": (
        "今の曲を飛ばして次の曲へ進めて",
        "skip the current track",
    ),
    "discord.stop_audio": (
        "音楽を停止してキューも空にして",
        "stop playback and clear the music queue",
    ),
    "discord.leave_audio": (
        "音楽を止めてBOTをVCから切断して",
        "disconnect the bot from voice and clear the queue",
    ),
    "discord.set_audio_loop": (
        "今の曲をリピート再生にして",
        "set the current track to repeat",
    ),
    "discord.remove_audio": (
        "待機キューの2曲目を削除して",
        "remove the second waiting track from the queue",
    ),
    "discord.set_audio_auto_leave": (
        "誰もいなくなったら自動切断するようにして",
        "enable automatic voice disconnect when no listeners remain",
    ),
    "discord.shuffle_audio": (
        "待機中の音楽キューをシャッフルして",
        "shuffle the waiting music queue",
    ),
    "discord.seek_audio": (
        "今の曲の再生位置を1分30秒へ移して",
        "seek the current track to ninety seconds",
    ),
    "discord.tune_audio": (
        "音楽の再生速度とピッチを変更して",
        "change the music speed and pitch",
    ),
    "discord.set_audio_volume": (
        "音楽の音量を80パーセントにして",
        "set the music volume to eighty percent",
    ),
    "discord.set_audio_radio": (
        "この曲から関連曲を自動再生するラジオを始めて",
        "start related-track radio autoplay from this song",
    ),
    "discord.move_audio": (
        "待機キューの3曲目を先頭へ並べ替えて",
        "move the third waiting track to the front of the queue",
    ),
    "discord.clear_my_audio": (
        "私が追加した待機中の曲だけ取り消して",
        "clear only the waiting tracks that I requested",
    ),
    "discord.speak": (
        "この短い文章をVOICEVOXでVCにしゃべって",
        "speak this short passage with voice synthesis in my VC",
    ),
    "discord.read_aloud_status": (
        "現在の読み上げ経路と状態を確認して",
        "inspect the current read-aloud routes and status",
    ),
    "discord.read_aloud_add_sources": (
        "このチャンネルを読み上げ対象に追加して",
        "add this conversation channel as a read-aloud source",
    ),
    "discord.read_aloud_remove_source": (
        "このチャンネルを読み上げ対象から外して",
        "remove this channel from the read-aloud sources",
    ),
    "discord.read_aloud_disable": (
        "このサーバーの読み上げを全部無効にして",
        "disable all read-aloud routes for this server",
    ),
    "discord.read_aloud_policy_status": (
        "読み上げ辞書と除外設定をまとめて確認して",
        "inspect the pronunciation and exclusion policy settings",
    ),
    "discord.read_aloud_dictionary_list": (
        "登録済みの読み上げ辞書を一覧にして",
        "list the registered pronunciation dictionary",
    ),
    "discord.read_aloud_dictionary_set": (
        "OpenAIをオープンエーアイと読むようにして",
        "register a pronunciation for this written form",
    ),
    "discord.read_aloud_dictionary_remove": (
        "OpenAIの読み方を読み上げ辞書から削除して",
        "remove this pronunciation dictionary entry",
    ),
    "discord.read_aloud_exclusion_set": (
        "このユーザーを読み上げ対象外にして",
        "exclude this user from read aloud",
    ),
    "discord.read_aloud_announcements_set": (
        "VCの入退室アナウンス設定を変更して",
        "configure voice join and leave announcements",
    ),
    "discord.read_aloud_content_mode_set": (
        "読み上げ内容をメッセージだけに切り替えて",
        "change the read-aloud content mode to messages only",
    ),
    "discord.read_aloud_semantics_set": (
        "返信元と添付も読み上げるようにして",
        "configure read-aloud semantics for replies and attachments",
    ),
    "discord.list_members": (
        "オンライン状態とVC参加を含めてメンバー一覧を見せて",
        "list members with presence activity and voice participation",
    ),
    "discord.inspect_channel": (
        "このチャンネルの設定と実効権限を確認して",
        "inspect this channel settings and effective permissions",
    ),
    "discord.list_pins": (
        "このチャンネルのピン留め済みメッセージを一覧にして",
        "list the pinned messages in this channel",
    ),
    "discord.list_reaction_users": (
        "このリアクションを付けた人を一覧にして",
        "list the users represented by this reaction",
    ),
    "discord.list_thread_members": (
        "このスレッドの参加者一覧を見せて",
        "list the current members of this thread",
    ),
    "discord.list_poll_voters": (
        "この投票の選択肢に投票した人を一覧にして",
        "list the voters for this poll answer",
    ),
    "discord.list_platform_resources": (
        "このサーバーのオンボーディング設定を確認して",
        "inspect this server's onboarding resource",
    ),
    "discord.inspect_application": (
        "このBOTのインテントとレイテンシを確認して",
        "inspect this bot application's intents and latency",
    ),
    "discord.create_guild_resource": (
        "新しいボイスチャンネルを作って",
        "create a new voice channel",
    ),
    "discord.update_guild_resource": (
        "この予定イベントの内容を更新して",
        "update this scheduled event",
    ),
    "discord.delete_guild_resource": (
        "このウェブフックを完全に削除して",
        "delete this webhook permanently",
    ),
    "discord.message_action": (
        "このアナウンスメッセージを公開して",
        "publish this announcement message",
    ),
    "discord.set_channel_overwrite": (
        "このチャンネルのロール権限上書きを設定して",
        "set a role permission overwrite on this channel",
    ),
    "discord.create_platform_asset": (
        "この画像ファイルからサーバー絵文字を作成して",
        "create a guild emoji from this workspace image",
    ),
    "discord.update_platform_asset": (
        "このサーバースタンプの名前を編集して",
        "edit this guild sticker",
    ),
    "discord.delete_platform_asset": (
        "このサウンドボード音源を削除して",
        "delete this soundboard sound",
    ),
    "discord.create_automod_rule": (
        "スパムを防ぐ自動モデレーションルールを作成して",
        "create an AutoMod spam rule",
    ),
    "discord.update_automod_rule": (
        "この自動モデレーションルールを編集して",
        "update this AutoMod rule",
    ),
    "discord.delete_automod_rule": (
        "この自動モデレーションルールを削除して",
        "delete this AutoMod rule",
    ),
    "discord.channel_operation": (
        "このアナウンスチャンネルをフォローして",
        "follow this announcement channel",
    ),
    "discord.forward_message": (
        "Discordの転送機能でこのメッセージを転送して",
        "forward this message using Discord's native forwarding",
    ),
    "discord.send_direct_message": (
        "このメンバーへDMを送って",
        "send this member a direct message",
    ),
    "discord.set_bot_presence": (
        "ボットのステータスを取り込み中にして",
        "set the bot presence to do not disturb",
    ),
}


CRITICAL_COLLISION_CASES = (
    ("メッセージを送って", "discord.send_message"),
    ("テキストチャンネルを作って", "discord.create_channel"),
    ("ボイスチャンネルを作って", "discord.create_guild_resource"),
    ("ステージチャンネルを作って", "discord.create_guild_resource"),
    ("フォーラムチャンネルを作って", "discord.create_guild_resource"),
    ("カテゴリを作って", "discord.create_guild_resource"),
    ("ステージを開始して", "discord.create_guild_resource"),
    ("アナウンスチャンネルをフォローして", "discord.channel_operation"),
    ("スレッドに参加して", "discord.channel_operation"),
    ("スレッドから退出して", "discord.channel_operation"),
    ("フォーラムタグを作って", "discord.channel_operation"),
    ("サウンドボードを鳴らして", "discord.channel_operation"),
    ("オンボーディング設定を確認して", "discord.list_platform_resources"),
    ("ウェルカム画面を確認して", "discord.list_platform_resources"),
    ("サーバーウィジェットを確認して", "discord.list_platform_resources"),
    ("バニティURLを確認して", "discord.list_platform_resources"),
    ("アクティブスレッドを一覧にして", "discord.list_platform_resources"),
    ("このメッセージを引用画像にして", "discord.create_quote_image"),
    ("このメッセージを文章で引用して再投稿して", "discord.post_expanded_message"),
    ("カスタム絵文字のアニメを見せて", "discord.view_custom_emoji"),
    ("スタンプの3フレーム目を見せて", "discord.view_sticker"),
    ("この画像がAI生成か調べて", "discord.analyze_attachment"),
    ("読み上げ辞書に登録して", "discord.read_aloud_dictionary_set"),
    ("DMを送って", "discord.send_direct_message"),
    ("ボットを取り込み中にして", "discord.set_bot_presence"),
)


def _discord_registry() -> tuple[CapabilityRegistry, tuple[str, ...]]:
    registry = CapabilityRegistry()
    names: list[str] = []
    for endpoint in build_discord_endpoints(
        cast(discord.Client, Mock(spec=discord.Client)),
        Mock(spec=SimajilordRuntime),
    ):
        names.append(endpoint.descriptor.name)
        registry.register(endpoint)
    return registry, tuple(names)


def _contains_japanese(value: str) -> bool:
    return any(
        "\u3040" <= character <= "\u30ff"
        or "\u3400" <= character <= "\u9fff"
        for character in value
    )


def test_all_106_discord_capabilities_have_one_explicit_discovery_classification() -> None:
    registry, names = _discord_registry()

    assert len(names) == 106
    assert len(names) == len(set(names))
    assert set(names) == set(DISCOVERY_CASES) | INTERNAL_DISCORD_CAPABILITIES
    assert not set(DISCOVERY_CASES) & INTERNAL_DISCORD_CAPABILITIES

    for capability_name in DISCOVERY_CASES:
        descriptor = registry.endpoint(capability_name).descriptor
        assert len(descriptor.keywords) == len(set(descriptor.keywords)), capability_name
        assert any(_contains_japanese(keyword) for keyword in descriptor.keywords), (
            capability_name
        )
        assert any(
            keyword.isascii() and any(character.isalpha() for character in keyword)
            for keyword in descriptor.keywords
        ), capability_name


@pytest.mark.asyncio
async def test_runtime_exposes_exactly_the_100_model_facing_discord_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "123")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_ENABLED", "true")
    monkeypatch.setenv("AGENT_FILE_SANDBOX_ENABLED", "true")
    monkeypatch.setenv("HIVE_API_KEY", "test-key")
    monkeypatch.setenv("IMAGE_GENERATION_ACCESS", "disabled")
    monkeypatch.setenv("AGENT_WEB_SEARCH_ACCESS", "disabled")
    settings = load_settings(dotenv_path=tmp_path / "missing.env")
    runtime = SimajilordRuntime.build(settings)
    assert runtime.agent is not None
    provider = cast(CodexAppServerProvider, runtime.agent.provider)

    exposed_discord_capabilities = {
        name
        for name in provider.tools.allowed_capabilities
        if name.startswith("discord.")
    }

    assert exposed_discord_capabilities == set(DISCOVERY_CASES)
    assert exposed_discord_capabilities.isdisjoint(INTERNAL_DISCORD_CAPABILITIES)
    await runtime.close()


@pytest.mark.parametrize(
    ("capability_name", "query"),
    [
        (capability_name, query)
        for capability_name, queries in DISCOVERY_CASES.items()
        for query in queries
    ],
)
def test_every_model_facing_discord_capability_ranks_first(
    capability_name: str,
    query: str,
) -> None:
    registry, _ = _discord_registry()

    matches = registry.search(query, limit=1)

    assert matches, query
    assert matches[0].descriptor.name == capability_name, query


@pytest.mark.parametrize(("query", "capability_name"), CRITICAL_COLLISION_CASES)
def test_ambiguous_natural_japanese_selects_the_specific_discord_capability(
    query: str,
    capability_name: str,
) -> None:
    registry, _ = _discord_registry()

    matches = registry.search(query, limit=3)

    assert matches, query
    assert matches[0].descriptor.name == capability_name, query
