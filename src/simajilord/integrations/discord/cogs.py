"""Discord slash commands as thin capability adapters."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shutil
import tempfile
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeAlias, cast

import discord
from discord import app_commands
from discord.ext import commands

from simajilord.agent import (
    AGENT_AUDIO_GRANT,
    AGENT_AUDIO_WRITE_CAPABILITIES,
    AGENT_AUTONOMY_ACTOR_ID,
    AGENT_FILE_GRANT,
    AGENT_IMAGE_GRANT,
    AGENT_MESSAGE_BREAK,
    AGENT_MESSAGE_GRANT,
    AGENT_MODERATION_GRANT,
    AGENT_NO_ACTION_CONTENT,
    AGENT_QUOTE_GRANT,
    AGENT_REPOST_GRANT,
    AGENT_WEB_GRANT,
    AgentBusyError,
    AgentProgressStage,
    AgentRateLimitError,
    AgentRequest,
    AgentTrigger,
    AgentUnavailableError,
)
from simajilord.capabilities.audio import (
    AudioAction,
    AudioControlRequest,
    AudioControlResponse,
    AudioHistoryRequest,
    AudioHistoryResponse,
    AudioMixRequest,
    AudioMixResponse,
    AudioPlayRequest,
    AudioPlayResponse,
    AudioQueueRequest,
    AudioQueueResponse,
    AudioSearchItem,
    AudioSearchRequest,
    AudioSearchResponse,
    FreshMixEnqueueRequest,
    FreshMixEnqueueResponse,
    FreshMixPlanRequest,
    FreshMixPreviewResponse,
    FreshMixReviseRequest,
    audio_queue_response,
)
from simajilord.capabilities.media import DownloadRequest, DownloadResponse
from simajilord.capabilities.moderation import (
    SyntheticMediaAnalyzeRequest,
    SyntheticMediaAnalyzeResponse,
)
from simajilord.capabilities.read_aloud import (
    ReadAloudAction,
    ReadAloudAnnouncementsSetRequest,
    ReadAloudContentModeSetRequest,
    ReadAloudDictionaryListRequest,
    ReadAloudDictionaryRemoveRequest,
    ReadAloudDictionarySetRequest,
    ReadAloudExclusionSetRequest,
    ReadAloudExclusionTarget,
    ReadAloudPolicyResponse,
    ReadAloudRequest,
    ReadAloudResponse,
    ReadAloudSemanticsSetRequest,
    ReadAloudStatusRequest,
)
from simajilord.capabilities.speech import SpeechSpeakRequest
from simajilord.capabilities.status import StatusRequest, StatusResponse
from simajilord.capabilities.system import (
    CapabilitySearchRequest,
    CapabilitySearchResponse,
    PingRequest,
    PingResponse,
    UptimeRequest,
    UptimeResponse,
)
from simajilord.capabilities.utility import (
    ChooseRequest,
    ChooseResponse,
    RollRequest,
    RollResponse,
)
from simajilord.capabilities.web import (
    WebFetchRequest,
    WebFetchResponse,
    WebFindRequest,
    WebFindResponse,
    WebSearchRequest,
    WebSearchResponse,
)
from simajilord.config import AgentFeatureAccess
from simajilord.core import InvocationContext
from simajilord.core.errors import MediaError, ModerationError, UserError, WebError
from simajilord.domain.audio import AudioKind, LoopMode
from simajilord.domain.media import DownloadFormat
from simajilord.domain.web import SearchDepth, WebSource, WebTextMatch
from simajilord.observability import EventRecord
from simajilord.runtime import SimajilordRuntime
from simajilord.services.audio import AudioSession
from simajilord.services.fresh_mix import FreshMixEnergy, FreshMixVocals
from simajilord.services.read_aloud import (
    ReadAloudContentMode,
    ReadAloudMode,
    ReadAloudRoute,
)
from simajilord.services.speech import SpeechSegment, SpeechSegmentKind

from .audio import DiscordAudioOutput
from .capabilities import (
    DiscordConnectVoiceRequest,
    DiscordCreateQuoteImageRequest,
    DiscordPollRequest,
    DiscordPollResponse,
    DiscordPostExpandedMessageRequest,
    DiscordPostExpandedMessageResponse,
    DiscordServerRequest,
    DiscordServerResponse,
    DiscordUserRequest,
    DiscordUserResponse,
    agent_readable_channel_ids,
    parse_discord_message_link,
    quote_message_has_animation,
)
from .presenter import (
    EmbedField,
    EmbedTone,
    command_embed,
)
from .read_aloud import ReadAloudMessageFormatter

log = logging.getLogger(__name__)
BotContext: TypeAlias = commands.Context[commands.Bot]
_QUOTE_CONTEXT_MENU_NAME = "Quote"
_MUSIC_DASHBOARD_ATTRIBUTE = "_simajilord_music_dashboard"
_MUSIC_DASHBOARD_STATE_FILE = "discord_music_dashboards.json"

_ERROR_MESSAGES = {
    "audio.auto_leave_value_required": "自動退出を有効にするか選んでください。",
    "audio.capacity_reached": "同時に接続できる音声サーバー数の上限に達しています。",
    "audio.history_limit_invalid": "履歴の表示件数は1〜25件で指定してください。",
    "audio.loop_mode_required": "ループ方法を選んでください。",
    "audio.loop_mix_conflict": (
        "Mixがオンです。ループへ切り替える場合は、先にMixを停止してください。"
    ),
    "audio.mix_loop_conflict": (
        "ループがオンです。Mixへ切り替える場合は、先にループをオフにしてください。"
    ),
    "audio.mix_seed_limit": "Mixの起点にできる曲は最大8曲です。",
    "audio.mix_seed_required": (
        "Mixの起点にするYouTube曲を指定するか、先に曲をキューへ追加してください。"
    ),
    "audio.mix_unavailable": "この環境では自動Mixを利用できません。",
    "audio.not_paused": "現在、一時停止していません。",
    "audio.nothing_playing": "現在再生している曲はありません。",
    "audio.output_disconnected": "ボイスチャンネルに接続されていません。",
    "audio.queue_position_invalid": "キューに表示されている有効な番号を指定してください。",
    "audio.queue_full": "このサーバーの音楽キューは上限に達しています。",
    "audio.user_queue_full": "自分が追加できる待機曲数の上限に達しています。",
    "audio.duplicate_limit": (
        "同じ曲がすでに複数回待機しています。繰り返し再生にはループを利用してください。"
    ),
    "audio.fresh_mix_artist_limit_invalid": "同じ投稿者の曲数上限は1〜4曲で指定してください。",
    "audio.fresh_mix_draft_busy": "このFresh Mixは別の操作で更新中です。少し待ってください。",
    "audio.fresh_mix_draft_capacity": (
        "保存中のFresh Mixが多いため、少し待ってから作成してください。"
    ),
    "audio.fresh_mix_draft_not_found": ("このFresh Mix previewは期限切れか、すでに追加済みです。"),
    "audio.fresh_mix_duration_invalid": "Fresh Mixの長さは15〜240分で指定してください。",
    "audio.fresh_mix_history_must_be_off": "Fresh Mixでは再生履歴を使用できません。",
    "audio.fresh_mix_no_candidates": (
        "条件を満たす実在曲を十分に確認できませんでした。内容を少し具体的にしてください。"
    ),
    "audio.fresh_mix_no_replacement": "置き換え条件に合う別の実在曲が見つかりませんでした。",
    "audio.fresh_mix_phase_duration_mismatch": "区間の合計時間がMix全体の時間と一致しません。",
    "audio.fresh_mix_phase_invalid": "各区間には時間と1〜4件の検索意図が必要です。",
    "audio.fresh_mix_phase_limit": "Fresh Mixの区間は最大6件です。",
    "audio.fresh_mix_position_invalid": "previewに表示されている有効な曲番号を指定してください。",
    "audio.fresh_mix_prompt_required": "作業内容や聴きたい雰囲気を入力してください。",
    "audio.fresh_mix_query_invalid": "検索意図が空か、長すぎます。",
    "audio.fresh_mix_query_limit": "Fresh Mix全体の検索意図は最大8件です。",
    "audio.seek_position_required": "移動先の再生位置を指定してください。",
    "audio.search_empty": "一致する曲が見つかりませんでした。",
    "audio.search_limit_invalid": "検索件数は1〜10件で指定してください。",
    "audio.session_closed": "この音声セッションは終了しています。",
    "audio.session_missing": "このサーバーには音声セッションがありません。",
    "audio.same_voice_required": (
        "音楽を操作するには、BOTと同じボイスチャンネルに参加してください。"
    ),
    "audio.waiting_queue_restricted": (
        "待機中のキューを開始・変更できるのは、曲を追加したユーザーだけです。"
    ),
    "audio.tune_range_invalid": "再生速度とピッチは、それぞれ0.5〜2.0で指定してください。",
    "audio.tune_values_required": "再生速度とピッチの両方を指定してください。",
    "audio.volume_range_invalid": "音量は0〜200%で指定してください。",
    "audio.volume_value_required": "音楽または読み上げの音量を指定してください。",
    "media.reference_required": "メディアのURLまたは検索キーワードを入力してください。",
    "media.reference_too_long": "URLまたは検索キーワードが長すぎます。",
    "media.query_url_not_allowed": "検索キーワードにURLは含められません。",
    "media.url_private": "非公開・ローカルネットワークのアドレスは使用できません。",
    "media.url_unresolvable": "メディアの配信元へ接続できませんでした。",
    "media.url_unsupported": ("認証情報や独自ポートを含まない、公開HTTPS URLを指定してください。"),
    "moderation.daily_limit_reached": (
        "本日分のHIVE API利用枠を使い切りました。日本時間の午前9時にリセットされます。"
    ),
    "moderation.filename_invalid": "添付ファイル名を読み取れませんでした。",
    "moderation.media_empty": "添付ファイルが空です。",
    "moderation.media_too_large": "添付ファイルが大きすぎるため、HIVEで解析できません。",
    "moderation.media_type_unsupported": ("HIVEで解析できるのは、一般的な画像・動画ファイルです。"),
    "moderation.not_configured": "この環境ではHIVE Moderationがまだ設定されていません。",
    "discord.message_limit_invalid": "メッセージ履歴の取得件数は1〜100件で指定してください。",
    "discord.message_chunk_limit_invalid": (
        "メッセージの取得文字数は1〜1000文字で指定してください。"
    ),
    "discord.message_offset_invalid": "有効なメッセージ開始位置を指定してください。",
    "discord.member_lookup_failed": (
        "Discordからユーザー情報を取得できませんでした。少し待ってから試してください。"
    ),
    "discord.member_required": (
        "サーバー上のユーザー情報を確認できませんでした。もう一度お試しください。"
    ),
    "discord.expand_cross_guild_forbidden": ("別のサーバーにあるメッセージは展開できません。"),
    "discord.expand_unavailable": (
        "このメッセージは展開できません。リンク先を閲覧できるか確認してください。"
    ),
    "discord.expand_destination_unavailable": (
        "このチャンネルへ展開結果を投稿できません。"
        "BOTのメッセージ送信・埋め込みリンク権限を確認してください。"
    ),
    "discord.expand_failed": (
        "Discordから元のメッセージを取得できませんでした。時間を空けて試してください。"
    ),
    "discord.quote_destination_unavailable": (
        "このチャンネルへ引用画像を投稿できません。"
        "BOTのメッセージ送信・ファイル添付権限を確認してください。"
    ),
    "discord.quote_failed": (
        "引用画像をDiscordへ送信できませんでした。少し待ってから試してください。"
    ),
    "discord.quote_render_failed": (
        "引用画像を描画できませんでした。メッセージ内容を確認してください。"
    ),
    "discord.manage_guild_required": (
        "この操作には「サーバー管理」権限が必要です。"
        "読み上げ先を変更する場合は、管理者が `/readaloud setup` を実行してください。"
    ),
    "read_aloud.route_fields_required": (
        "読み上げる会話チャンネルと、音声を流すボイスチャンネルを選んでください。"
    ),
    "read_aloud.destination_conflict": ("読み上げは別のボイスチャンネルに設定されています。"),
    "read_aloud.source_channel_required": "読み上げる会話チャンネルを選んでください。",
    "read_aloud.source_channel_limit": "会話チャンネルは1〜25個まで選べます。",
    "read_aloud.source_channels_required": "会話チャンネルを1つ以上選んでください。",
    "read_aloud.dictionary_surface_required": "辞書へ登録する表記を入力してください。",
    "read_aloud.dictionary_surface_too_long": "辞書の表記は100文字以内で入力してください。",
    "read_aloud.dictionary_reading_required": "表記の読み方を入力してください。",
    "read_aloud.dictionary_reading_too_long": "読み方は200文字以内で入力してください。",
    "read_aloud.announcement_value_required": "変更する入退室通知を1つ以上選んでください。",
    "read_aloud.semantic_value_required": "変更する読み上げ項目を1つ以上選んでください。",
    "read_aloud.ignore_bot_unnecessary": "BOTのメッセージは最初から読み上げません。",
    "read_aloud.role_not_found": "指定したロールがこのサーバーに見つかりません。",
    "discord.message_channel_unavailable": (
        "選択したすべてのチャンネルを、あなたとBOTの両方が閲覧できる必要があります。"
    ),
    "speech.no_readable_text": "読み上げられる文章がありません。",
    "speech.queue_full": "読み上げが混み合っています。少し待ってから試してください。",
    "utility.dice_count_invalid": "サイコロの個数は1〜20個で指定してください。",
    "utility.dice_sides_invalid": "サイコロの面数は2〜1000面で指定してください。",
    "utility.option_count_invalid": "選択肢は2〜20個入力してください。",
    "utility.option_too_long": "選択肢は1つ100文字以内にしてください。",
    "web.chunk_limit_invalid": "ページの取得文字数は200〜6000文字で指定してください。",
    "web.context_limit_invalid": "検索箇所の前後は40〜300文字で指定してください。",
    "web.match_limit_invalid": "ページ内検索の表示件数は1〜10件で指定してください。",
    "web.offset_invalid": "ページ本文の有効な開始位置を指定してください。",
    "web.pattern_required": "ページ内で探す語句を入力してください。",
    "web.pattern_too_long": "検索語句は300文字以内にしてください。",
    "web.query_required": "検索キーワードを入力してください。",
    "web.query_too_long": "検索キーワードは500文字以内にしてください。",
    "web.safesearch_invalid": "セーフサーチは0・1・2のいずれかを指定してください。",
    "web.time_range_invalid": "期間はday・month・yearのいずれかを指定してください。",
    "workspace.required": "この操作はDiscordサーバー内で実行してください。",
}

_MEDIA_ERROR_MESSAGES = {
    "cookie_required": (
        "このメディアの取得にはログインが必要です。ホスト側でCookieを設定してください。"
    ),
    "geo_restricted": "この地域からはメディアを利用できません。",
    "rate_limited": "配信元の利用制限に達しました。時間を空けて試してください。",
    "timeout": "メディアの処理が時間内に終わりませんでした。",
    "too_large": "ファイルがこのサーバーのアップロード上限を超えています。",
    "unavailable": "メディアが非公開、削除済み、または利用できない状態です。",
    "unsafe_path": "配信元から安全でない処理結果が返されました。",
    "unsupported": "このメディアURLには対応していません。",
    "unknown": "メディアの処理を完了できませんでした。",
}

_WEB_ERROR_MESSAGES = {
    "content_empty": "ページ内に読み取れる文章がありません。",
    "content_invalid": "ページ本文を読み取れませんでした。",
    "content_type_unsupported": "この種類のページにはまだ対応していません。",
    "fetch_failed": "ページを取得できませんでした。",
    "redirect_invalid": "ページの転送先が無効です。",
    "request_too_broad": "検索範囲が広すぎます。キーワードを絞ってください。",
    "response_too_large": "ページが大きすぎるため、安全に読み取れません。",
    "search_backend_error": "ローカル検索サービスでエラーが発生しました。",
    "search_invalid_response": "ローカル検索サービスから不正な応答が返されました。",
    "search_response_too_large": "検索結果が大きすぎます。キーワードを絞ってください。",
    "search_unavailable": "ローカル検索サービスを一時的に利用できません。",
    "timeout": "Web処理が時間内に終わりませんでした。",
    "too_many_redirects": "ページの転送回数が多すぎます。",
    "upstream_unavailable": "Webサイトを一時的に利用できません。",
    "url_invalid": "公開されているHTTPまたはHTTPS URLを指定してください。",
    "url_private": "非公開・ローカルネットワークのアドレスは開けません。",
    "url_rejected": "Webサイトからアクセスを拒否されました。",
    "url_unresolvable": "Webサイトへ接続できませんでした。",
}

_MODERATION_ERROR_MESSAGES = {
    "authentication_failed": "HIVEの認証に失敗しました。ホスト側のAPIキーを確認してください。",
    "invalid_response": "HIVEから正しくない解析結果が返されました。",
    "media_rejected": "HIVEでこの添付ファイルを解析できませんでした。",
    "provider_unavailable": "HIVEを一時的に利用できません。",
    "rate_limited": "HIVEの利用制限に達しています。時間を空けて試してください。",
    "response_too_large": "HIVEの解析結果が想定上限を超えました。",
    "timeout": "HIVEの解析が時間内に終わりませんでした。",
}

_AUDIO_ACTION_MESSAGES = {
    AudioAction.PAUSE.value: "一時停止しました。",
    AudioAction.RESUME.value: "再生を再開しました。",
    AudioAction.SKIP.value: "再生中の曲をスキップしました。",
    AudioAction.STOP.value: "再生を停止し、キューを空にしました。",
    AudioAction.LEAVE.value: "ボイスチャンネルから退出しました。",
}


def _duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3_600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes}:{remaining_seconds:02d}"


def _read_aloud_mode_label(mode: str | None) -> str:
    if mode == ReadAloudMode.SKIP_DURING_MUSIC.value:
        return "音楽の再生中は読み上げない"
    return "すべて読み上げる・読み上げ中は音楽を自動調整"


def _on_off(enabled: bool) -> str:
    return "オン" if enabled else "オフ"


def _speech_voice_label(runtime: SimajilordRuntime) -> str:
    settings = runtime.settings
    if settings.tts_provider == "voicevox":
        return f"VOICEVOX・スタイルID {settings.voicevox_speaker_id}"
    return f"macOS · {settings.tts_voice}"


def _parse_position(value: str) -> tuple[float, bool]:
    text = value.strip()
    relative = text.startswith(("+", "-"))
    sign = -1.0 if text.startswith("-") else 1.0
    unsigned = text[1:] if relative else text
    parts = unsigned.split(":")
    if not 1 <= len(parts) <= 3 or any(not part.isdigit() for part in parts):
        raise UserError("`1:23`、`+30`、`-10` のように再生位置を入力してください。")
    numbers = [int(part) for part in parts]
    seconds = 0
    for number in numbers:
        seconds = seconds * 60 + number
    return sign * float(seconds), relative


def _requester(name: str | None) -> str:
    return discord.utils.escape_markdown(name) if name else "Unknown"


def _queue_requester(item: object) -> str:
    lane = getattr(item, "queue_lane", "request")
    if lane == "autoplay":
        return "Auto Mix"
    return _requester(getattr(item, "requested_by_name", None))


def _loop_mode_label(mode: str) -> str:
    return {
        LoopMode.NONE.value: "Off",
        LoopMode.TRACK.value: "Track",
        LoopMode.QUEUE.value: "Queue",
    }.get(mode, mode)


def _compact_panel_text(value: str, *, maximum: int) -> str:
    """Keep control-panel copy readable without altering the underlying value."""

    cleaned = " ".join(value.split())
    return cleaned if len(cleaned) <= maximum else f"{cleaned[: maximum - 1].rstrip()}…"


def _read_aloud_status(route: ReadAloudRoute | None) -> str:
    if route is None:
        return "Off · Select **Read aloud** to choose channels."
    source_count = len(route.text_channel_ids)
    channel_label = "channel" if source_count == 1 else "channels"
    return (
        f"On · **{source_count} {channel_label}** → <#{route.audio_destination_id}>\n"
        f"Mode **{route.mode.value.replace('_', ' ').title()}**"
    )


def _active_read_aloud_route(
    runtime: SimajilordRuntime,
    workspace_id: str,
) -> ReadAloudRoute | None:
    route = runtime.read_aloud.get(workspace_id)
    return route if isinstance(route, ReadAloudRoute) else None


def _risk_label(risk: str) -> str:
    return {
        "read": "参照",
        "write": "変更",
        "external": "外部通信",
        "destructive": "破壊的",
    }.get(risk, risk)


def _approval_label(approval: str) -> str:
    return {
        "never": "不要",
        "when_requested": "依頼時",
        "always": "常に必要",
    }.get(approval, approval)


def _search_candidate_line(candidate: AudioSearchItem) -> str:
    title = discord.utils.escape_markdown(candidate.title)
    source = f"\n{discord.utils.escape_markdown(candidate.uploader)}" if candidate.uploader else ""
    return f"[{title}]({candidate.reference}){source}"


def music_added_embed(response: AudioPlayResponse) -> discord.Embed:
    if response.playback_state == "playing":
        title = "Now playing"
        playback = "Playing now"
    elif response.playback_state == "waiting_for_voice":
        title = "Added to queue"
        playback = "Ready · Playback starts when you join a voice channel."
    else:
        title = "Added to queue"
        playback = f"Waiting · Position **{response.queue_position}**"
    fields = [
        EmbedField("Status", playback, inline=False),
        EmbedField("Duration", _duration(response.duration_seconds)),
        EmbedField("Requested by", _requester(response.requested_by_name)),
    ]
    if response.uploader:
        fields.append(EmbedField("Source", discord.utils.escape_markdown(response.uploader)))
    fields.append(
        EmbedField(
            "Voice channel",
            f"<#{response.destination_id}>"
            if response.destination_id
            else "Connects when the requester joins",
        )
    )
    embed = command_embed(
        title,
        description=(
            "### ["
            f"{discord.utils.escape_markdown(_compact_panel_text(response.title, maximum=100))}"
            "]"
            f"({response.page_url})"
        ),
        fields=tuple(fields),
        tone=EmbedTone.SUCCESS,
    )
    if response.thumbnail_url:
        embed.set_thumbnail(url=response.thumbnail_url)
    return embed


def music_queue_embed(
    response: AudioQueueResponse,
    *,
    page: int = 1,
    page_size: int = 10,
    read_aloud_route: ReadAloudRoute | None = None,
) -> discord.Embed:
    fields: list[EmbedField] = []
    upcoming = tuple(item for item in response.pending if item.kind == AudioKind.MUSIC.value)
    if response.current is None:
        title = "Audio"
        description_lines = ["No track is playing."]
    else:
        title = "Audio · Now playing"
        current = response.current
        elapsed = min(response.position_seconds, current.duration_seconds)
        uploader = discord.utils.escape_markdown(current.uploader or "Unknown uploader")
        timing = f"`{_duration(elapsed)} / {_duration(current.duration_seconds)}`"
        if not response.paused and current.duration_seconds > elapsed:
            remaining = (current.duration_seconds - elapsed) / max(response.speed, 0.01)
            timing += f" · ends <t:{int(time.time() + remaining)}:R>"
        description_lines = [
            (
                "### ["
                f"{discord.utils.escape_markdown(_compact_panel_text(current.title, maximum=96))}"
                "]"
                f"({current.page_url})"
            ),
            timing,
            (
                f"{_compact_panel_text(uploader, maximum=54)} · "
                f"requested by {_queue_requester(current)}"
            ),
        ]

    if response.resume_confirmation_required:
        description_lines.append("**Ready to resume** · Join the VC and press **Start**.")
    elif response.waiting_for_voice:
        description_lines.append("**Queued** · Playback starts when the requester joins the VC.")
    elif response.paused:
        description_lines.append("**Paused**")

    fields.extend(
        (
            EmbedField(
                "Playback",
                f"Queue **{len(upcoming)}**\n"
                f"Loop **{_loop_mode_label(response.loop_mode)}** · "
                f"Mix **{'On' if response.autoplay_enabled else 'Off'}**",
            ),
            EmbedField(
                "Levels",
                f"Music **{response.music_volume_percent}%**\n"
                f"Read aloud **{response.speech_volume_percent}%**",
            ),
            EmbedField(
                "Read aloud",
                _read_aloud_status(read_aloud_route),
                inline=False,
            ),
        )
    )

    if upcoming:
        page_count = max(1, (len(upcoming) + page_size - 1) // page_size)
        selected_page = min(max(1, page), page_count)
        start = (selected_page - 1) * page_size
        visible = upcoming[start : start + page_size]
        lines = [
            f"`{index:02d}` [{item.title}]({item.page_url}) · "
            f"`{_duration(item.duration_seconds)}` · {_queue_requester(item)}"
            for index, item in enumerate(visible, start=start + 1)
        ]
        fields.append(
            EmbedField(
                f"Up Next · {selected_page}/{page_count}",
                "\n".join(lines),
                inline=False,
            )
        )
    if response.autoplay_enabled:
        autoplay_text = (
            f"[{discord.utils.escape_markdown(response.autoplay_next.title)}]"
            f"({response.autoplay_next.page_url})"
            if response.autoplay_next is not None
            else "Finding the next track…"
        )
        fields.append(
            EmbedField(
                "Mix",
                f"{autoplay_text}\nManual requests take priority.",
                inline=False,
            )
        )
    if response.speed != 1.0 or response.pitch != 1.0:
        fields.append(
            EmbedField(
                "Tuning",
                f"Speed {response.speed:.2f}x · Pitch {response.pitch:.2f}x",
            )
        )
    embed = command_embed(
        title,
        description="\n".join(description_lines),
        fields=tuple(fields),
    )
    if response.current and response.current.thumbnail_url:
        embed.set_thumbnail(url=response.current.thumbnail_url)
    return embed


def music_now_playing_embed(response: AudioQueueResponse) -> discord.Embed:
    return music_queue_embed(response)


def fresh_mix_preview_embed(response: FreshMixPreviewResponse) -> discord.Embed:
    lines: list[str] = []
    used = 0
    for index, track in enumerate(response.tracks, start=1):
        line = (
            f"`{index:02d}` [{discord.utils.escape_markdown(track.title)}]"
            f"({_safe_markdown_url(track.reference)}) · "
            f"`{_duration(track.duration_seconds)}` · "
            f"{discord.utils.escape_markdown(track.artist)}"
        )
        if used + len(line) + 1 > 3_300:
            remaining = len(response.tracks) - len(lines)
            lines.append(f"ほか **{remaining}曲**")
            break
        lines.append(line)
        used += len(line) + 1
    duration_delta = round(response.duration_seconds / 60 - response.target_minutes)
    duration_note = (
        "目標範囲内"
        if abs(duration_delta) <= 5
        else f"目標より {abs(duration_delta)}分{'長い' if duration_delta > 0 else '短い'}"
    )
    return command_embed(
        "Fresh Mix preview",
        description="\n".join(lines),
        fields=(
            EmbedField(
                "構成",
                f"**{len(response.tracks)}曲・{_duration(response.duration_seconds)}**\n"
                f"目標 {response.target_minutes}分・{duration_note}",
                inline=False,
            ),
            EmbedField(
                "検証",
                "履歴不使用・実在候補を検索済み・重複/同一投稿者/版を検査済み",
                inline=False,
            ),
        ),
        tone=(
            EmbedTone.SUCCESS
            if "duration_within_5_minutes" in response.checks
            else EmbedTone.WARNING
        ),
    )


class FreshMixPreviewView(discord.ui.View):
    """Require an explicit click before a verified draft changes the queue."""

    def __init__(
        self,
        runtime: SimajilordRuntime,
        dashboard: MusicDashboardManager,
        response: FreshMixPreviewResponse,
        *,
        requester_id: int,
        requester_name: str,
    ) -> None:
        super().__init__(timeout=10 * 60)
        self.runtime = runtime
        self.dashboard = dashboard
        self.response = response
        self.requester_id = requester_id
        self.requester_name = requester_name
        self._lock = asyncio.Lock()
        self._finished = False

    async def _require_requester(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            embed=command_embed(
                "このpreviewはほかのユーザーが作成しました",
                description="自分用のMixは `/freshmix` から作成できます。",
                tone=EmbedTone.WARNING,
            ),
            ephemeral=True,
        )
        return False

    async def revise(
        self,
        interaction: discord.Interaction,
        *,
        position: int,
        query: str,
    ) -> None:
        if not await self._require_requester(interaction):
            return
        async with self._lock:
            if self._finished:
                await interaction.response.send_message(
                    "このpreviewは処理済みです。",
                    ephemeral=True,
                )
                return
            await interaction.response.defer(thinking=True)
            try:
                self.response = cast(
                    FreshMixPreviewResponse,
                    await self.runtime.registry.invoke(
                        "discord.revise_fresh_mix",
                        FreshMixReviseRequest(
                            draft_id=self.response.draft_id,
                            position=position,
                            query=query,
                        ),
                        invocation_context(interaction),
                    ),
                )
                await interaction.edit_original_response(
                    embed=fresh_mix_preview_embed(self.response),
                    view=self,
                )
            except Exception as exc:
                await send_error(interaction, exc)

    @discord.ui.button(
        label="Replace Track",
        style=discord.ButtonStyle.primary,
        custom_id="simajilord:fresh-mix:revise",
    )
    async def revise_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[FreshMixPreviewView],
    ) -> None:
        if not await self._require_requester(interaction):
            return
        await interaction.response.send_modal(FreshMixReviseModal(self))

    @discord.ui.button(
        label="Play Mix",
        style=discord.ButtonStyle.success,
        custom_id="simajilord:fresh-mix:enqueue",
    )
    async def enqueue_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[FreshMixPreviewView],
    ) -> None:
        if not await self._require_requester(interaction):
            return
        async with self._lock:
            if self._finished:
                await interaction.response.send_message(
                    "このpreviewは処理済みです。",
                    ephemeral=True,
                )
                return
            await interaction.response.defer(thinking=True)
            try:
                self.dashboard.bind(interaction.guild_id, interaction.channel_id)
                response = cast(
                    FreshMixEnqueueResponse,
                    await self.runtime.registry.invoke(
                        "discord.enqueue_fresh_mix",
                        FreshMixEnqueueRequest(
                            draft_id=self.response.draft_id,
                            requested_by_name=self.requester_name,
                        ),
                        invocation_context(interaction),
                    ),
                )
                self._finished = True
                self.stop()
                await interaction.edit_original_response(
                    embed=command_embed(
                        "Fresh Mixをキューに追加しました",
                        description=(
                            f"**{response.track_count}曲・"
                            f"{_duration(response.duration_seconds)}**\n"
                            f"キュー {response.first_queue_position}〜"
                            f"{response.last_queue_position}番"
                        ),
                        tone=EmbedTone.SUCCESS,
                    ),
                    view=None,
                )
            except Exception as exc:
                await send_error(interaction, exc)

    @discord.ui.button(
        label="Cancel",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:fresh-mix:cancel",
    )
    async def cancel_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[FreshMixPreviewView],
    ) -> None:
        if not await self._require_requester(interaction):
            return
        self._finished = True
        self.stop()
        await interaction.response.edit_message(view=None)

    async def on_timeout(self) -> None:
        self._finished = True
        self.stop()


class FreshMixReviseModal(discord.ui.Modal, title="Fresh Mixの1曲を変更"):
    position: discord.ui.TextInput[FreshMixReviseModal] = discord.ui.TextInput(
        label="曲番号",
        placeholder="preview左側の番号 (例: 3)",
        min_length=1,
        max_length=2,
    )
    query: discord.ui.TextInput[FreshMixReviseModal] = discord.ui.TextInput(
        label="代わりに探す曲",
        placeholder="曲名、投稿者、雰囲気など",
        min_length=1,
        max_length=150,
    )

    def __init__(self, preview: FreshMixPreviewView) -> None:
        super().__init__()
        self.preview = preview

    async def on_submit(self, interaction: discord.Interaction) -> None:
        try:
            position = int(str(self.position).strip())
        except ValueError:
            await interaction.response.send_message(
                "曲番号は数字で入力してください。",
                ephemeral=True,
            )
            return
        await self.preview.revise(
            interaction,
            position=position,
            query=str(self.query).strip(),
        )


class MusicDashboardManager:
    """Keep one current, silent music panel near the bottom of each bound channel."""

    _DEBOUNCE_SECONDS = 0.8
    _IDLE_SECONDS = 10 * 60

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime
        self._channel_ids: dict[str, int] = {}
        self._messages: dict[str, discord.Message] = {}
        self._fingerprints: dict[str, tuple[object, ...]] = {}
        self._repost_tasks: dict[str, asyncio.Task[None]] = {}
        self._expiry_tasks: dict[str, asyncio.Task[None]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._store_lock = asyncio.Lock()
        data_dir = getattr(getattr(runtime, "settings", None), "data_dir", None)
        self._state_path = (
            data_dir / _MUSIC_DASHBOARD_STATE_FILE if isinstance(data_dir, Path) else None
        )
        self._stored_messages = self._load_stored_messages()
        self._channel_ids.update(
            {
                workspace_id: channel_id
                for workspace_id, (channel_id, _) in self._stored_messages.items()
            }
        )
        runtime.audio.add_state_listener(self.on_audio_state_changed)
        for workspace_id in self._channel_ids:
            session = runtime.audio.find(workspace_id)
            if session is not None:
                self._schedule_repost(session)

    def bind(self, guild_id: int | None, channel_id: int | None) -> None:
        if guild_id is None or channel_id is None:
            return
        workspace_id = str(guild_id)
        changed = self._channel_ids.get(workspace_id) != channel_id
        if changed:
            self._channel_ids[workspace_id] = channel_id
            self._fingerprints.pop(workspace_id, None)
        if changed or workspace_id not in self._messages:
            session = self.runtime.audio.find(workspace_id)
            if session is not None:
                self._schedule_repost(session)

    async def on_audio_state_changed(self, session: AudioSession) -> None:
        self._schedule_repost(session)

    async def publish(
        self,
        session: AudioSession,
        *,
        obsolete_message: discord.Message | None = None,
        force: bool = False,
    ) -> None:
        """Publish one canonical panel immediately, cancelling a queued duplicate."""

        workspace_id = session.workspace_id
        previous = self._repost_tasks.pop(workspace_id, None)
        if previous is not None and previous is not asyncio.current_task():
            previous.cancel()
        async with self._locks.setdefault(workspace_id, asyncio.Lock()):
            await self._repost(
                session,
                obsolete_message=obsolete_message,
                force=force,
            )

    def _schedule_repost(self, session: AudioSession) -> None:
        workspace_id = session.workspace_id
        if workspace_id not in self._channel_ids:
            return
        previous = self._repost_tasks.pop(workspace_id, None)
        if previous is not None and previous is not asyncio.current_task():
            previous.cancel()
        self._repost_tasks[workspace_id] = asyncio.create_task(
            self._debounced_repost(session),
            name=f"simajilord-music-dashboard-{workspace_id}",
        )

    async def close(self) -> None:
        self.runtime.audio.remove_state_listener(self.on_audio_state_changed)
        tasks = (*self._repost_tasks.values(), *self._expiry_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._repost_tasks.clear()
        self._expiry_tasks.clear()

    async def _debounced_repost(self, session: AudioSession) -> None:
        workspace_id = session.workspace_id
        try:
            await asyncio.sleep(self._DEBOUNCE_SECONDS)
            async with self._locks.setdefault(workspace_id, asyncio.Lock()):
                await self._repost(session)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Could not repost music dashboard guild=%s", workspace_id)
        finally:
            if self._repost_tasks.get(workspace_id) is asyncio.current_task():
                self._repost_tasks.pop(workspace_id, None)

    async def _repost(
        self,
        session: AudioSession,
        *,
        obsolete_message: discord.Message | None = None,
        force: bool = False,
    ) -> None:
        workspace_id = session.workspace_id
        channel_id = self._channel_ids.get(workspace_id)
        if channel_id is None:
            return
        snapshot = await session.snapshot()
        response = audio_queue_response(snapshot)
        read_aloud_route = _active_read_aloud_route(self.runtime, workspace_id)
        fingerprint = _music_dashboard_fingerprint(response, read_aloud_route)
        current_message = self._messages.get(workspace_id)
        if current_message is None:
            current_message = await self._fetch_stored_message(workspace_id)
        if (
            not force
            and response.current is None
            and not any(
                item.kind == AudioKind.MUSIC.value for item in response.pending
            )
            and not response.autoplay_enabled
            and response.destination_id is None
            and not response.waiting_for_voice
            and not response.resume_confirmation_required
        ):
            for message in _unique_messages(current_message, obsolete_message):
                with suppress(discord.DiscordException):
                    await message.delete()
            self._messages.pop(workspace_id, None)
            self._fingerprints.pop(workspace_id, None)
            if self._stored_messages.pop(workspace_id, None) is not None:
                await self._persist_stored_messages()
            return
        if self._fingerprints.get(workspace_id) == fingerprint and current_message:
            if obsolete_message is not None and obsolete_message.id != current_message.id:
                with suppress(discord.DiscordException):
                    await obsolete_message.delete()
            return
        embed = music_queue_embed(
            response,
            read_aloud_route=read_aloud_route,
        )
        view = MusicControlsView(self.runtime, self, response=response)
        stored = self._stored_messages.get(workspace_id)
        current_is_in_bound_channel = stored is None or stored[0] == channel_id
        if current_message is not None and current_is_in_bound_channel and not force:
            try:
                edited = await current_message.edit(embed=embed, view=view)
            except discord.DiscordException:
                log.warning(
                    "Could not edit music dashboard; posting a replacement guild=%s",
                    workspace_id,
                    exc_info=True,
                )
            else:
                if isinstance(edited, discord.Message):
                    current_message = edited
                self._messages[workspace_id] = current_message
                self._fingerprints[workspace_id] = fingerprint
                self._stored_messages[workspace_id] = (channel_id, current_message.id)
                await self._persist_stored_messages()
                if (
                    obsolete_message is not None
                    and obsolete_message.id != current_message.id
                ):
                    with suppress(discord.DiscordException):
                        await obsolete_message.delete()
                self._refresh_expiry(
                    workspace_id,
                    current_message,
                    idle=_audio_dashboard_is_idle(response),
                )
                return

        channel = self.bot.get_channel(channel_id)
        if channel is None:
            with suppress(discord.DiscordException):
                channel = await self.bot.fetch_channel(channel_id)
        if not isinstance(channel, discord.abc.Messageable):
            return

        # Sending first avoids losing the working panel when Discord rejects a post.
        new_message = await channel.send(
            embed=embed,
            view=view,
            silent=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self._messages[workspace_id] = new_message
        self._fingerprints[workspace_id] = fingerprint
        self._stored_messages[workspace_id] = (channel_id, new_message.id)
        await self._persist_stored_messages()
        for message in _unique_messages(current_message, obsolete_message):
            if message.id == new_message.id:
                continue
            with suppress(discord.DiscordException):
                await message.delete()

        self._refresh_expiry(
            workspace_id,
            new_message,
            idle=_audio_dashboard_is_idle(response),
        )

    def _refresh_expiry(
        self,
        workspace_id: str,
        message: discord.Message,
        *,
        idle: bool,
    ) -> None:
        expiry = self._expiry_tasks.pop(workspace_id, None)
        if expiry is not None and expiry is not asyncio.current_task():
            expiry.cancel()
        if not idle:
            return
        self._expiry_tasks[workspace_id] = asyncio.create_task(
            self._expire(workspace_id, message),
            name=f"simajilord-music-dashboard-expiry-{workspace_id}",
        )

    async def _expire(self, workspace_id: str, message: discord.Message) -> None:
        try:
            await asyncio.sleep(self._IDLE_SECONDS)
            if self._messages.get(workspace_id) is not message:
                return
            with suppress(discord.DiscordException):
                await message.delete()
            self._messages.pop(workspace_id, None)
            self._fingerprints.pop(workspace_id, None)
            stored = self._stored_messages.get(workspace_id)
            if stored is not None and stored[1] == message.id:
                self._stored_messages.pop(workspace_id, None)
                await self._persist_stored_messages()
        except asyncio.CancelledError:
            raise
        finally:
            if self._expiry_tasks.get(workspace_id) is asyncio.current_task():
                self._expiry_tasks.pop(workspace_id, None)

    async def _fetch_stored_message(
        self,
        workspace_id: str,
    ) -> discord.Message | None:
        stored = self._stored_messages.get(workspace_id)
        if stored is None:
            return None
        channel_id, message_id = stored
        channel = self.bot.get_channel(channel_id)
        if channel is None:
            with suppress(discord.DiscordException):
                channel = await self.bot.fetch_channel(channel_id)
        fetch_message = getattr(channel, "fetch_message", None)
        if not callable(fetch_message):
            return None
        try:
            message = await fetch_message(message_id)
        except (discord.DiscordException, TypeError):
            return None
        return message if isinstance(message, discord.Message) else None

    def _load_stored_messages(self) -> dict[str, tuple[int, int]]:
        path = self._state_path
        if path is None or not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("version") != 1:
                return {}
            raw_messages = payload.get("messages")
            if not isinstance(raw_messages, dict):
                return {}
            restored: dict[str, tuple[int, int]] = {}
            for workspace_id, value in raw_messages.items():
                if (
                    isinstance(workspace_id, str)
                    and isinstance(value, dict)
                    and isinstance(value.get("channel_id"), int)
                    and isinstance(value.get("message_id"), int)
                ):
                    restored[workspace_id] = (
                        value["channel_id"],
                        value["message_id"],
                    )
            return restored
        except (OSError, TypeError, ValueError):
            return {}

    async def _persist_stored_messages(self) -> None:
        path = self._state_path
        if path is None:
            return
        async with self._store_lock:
            payload = {
                "version": 1,
                "messages": {
                    workspace_id: {
                        "channel_id": channel_id,
                        "message_id": message_id,
                    }
                    for workspace_id, (channel_id, message_id) in sorted(
                        self._stored_messages.items()
                    )
                },
            }
            await asyncio.to_thread(
                _write_private_json,
                path,
                payload,
            )


def _write_private_json(path: Path, payload: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def _music_dashboard_fingerprint(
    response: AudioQueueResponse,
    read_aloud_route: ReadAloudRoute | None = None,
) -> tuple[object, ...]:
    """Ignore elapsed seconds and read-aloud overlays; retain visible music state."""

    current = response.current
    autoplay_next = response.autoplay_next
    return (
        None if current is None else current.page_url,
        tuple(
            (
                item.page_url,
                item.requested_by_name,
                item.queue_lane,
            )
            for item in response.pending
            if item.kind == AudioKind.MUSIC.value
        ),
        response.paused,
        response.loop_mode,
        response.destination_id,
        response.auto_leave,
        response.waiting_for_voice,
        response.resume_confirmation_required,
        response.music_volume_percent,
        response.speech_volume_percent,
        response.speed,
        response.pitch,
        response.autoplay_enabled,
        None if autoplay_next is None else autoplay_next.page_url,
        None
        if read_aloud_route is None
        else (
            read_aloud_route.text_channel_ids,
            read_aloud_route.audio_destination_id,
            read_aloud_route.mode.value,
        ),
    )


def _audio_dashboard_is_idle(response: AudioQueueResponse) -> bool:
    """Expire only a connected shell with no active, queued, or held work."""

    return (
        response.current is None
        and not any(
            item.kind == AudioKind.MUSIC.value
            for item in response.pending
        )
        and not response.autoplay_enabled
        and not response.waiting_for_voice
        and not response.resume_confirmation_required
    )


def _unique_messages(
    *messages: discord.Message | None,
) -> tuple[discord.Message, ...]:
    unique: dict[int, discord.Message] = {}
    for message in messages:
        if message is not None:
            unique[message.id] = message
    return tuple(unique.values())


def _schedule_message_delete(message: discord.Message, *, delay: float) -> None:
    async def delete_later() -> None:
        await asyncio.sleep(delay)
        with suppress(discord.DiscordException):
            await message.delete()

    task = asyncio.create_task(
        delete_later(),
        name=f"simajilord-discord-delete-{message.id}",
    )
    task.add_done_callback(
        lambda completed: None if completed.cancelled() else completed.exception()
    )


def music_search_embed(response: AudioSearchResponse) -> discord.Embed:
    fields = tuple(
        EmbedField(
            f"{index} · {_duration(candidate.duration_seconds)}",
            _search_candidate_line(candidate),
            inline=False,
        )
        for index, candidate in enumerate(response.candidates, start=1)
    )
    embed = command_embed(
        "再生する曲を選んでください",
        description=(
            f"**{discord.utils.escape_markdown(response.query)}** に近い候補が"
            "複数見つかりました。一度選ぶと履歴に残り、次回から自動で選びやすくなります。"
        ),
        fields=fields,
        tone=EmbedTone.WARNING,
    )
    if response.candidates and response.candidates[0].thumbnail_url:
        embed.set_thumbnail(url=response.candidates[0].thumbnail_url)
    return embed


def music_history_embed(response: AudioHistoryResponse) -> discord.Embed:
    if not response.items:
        return command_embed(
            "再生履歴",
            description="まだ再生履歴はありません。",
        )
    lines = []
    for index, item in enumerate(response.items, start=1):
        when = f" · <t:{item.played_at_epoch}:R>" if item.played_at_epoch else ""
        lines.append(
            f"`{index:02d}` [{item.title}]({item.page_url}) · "
            f"`{_duration(item.duration_seconds)}` · {_requester(item.requested_by_name)}{when}"
        )
    return command_embed("再生履歴", description="\n".join(lines))


def _safe_markdown_url(value: str) -> str:
    return value.replace("(", "%28").replace(")", "%29")


def _web_source_text(index: int, source: WebSource) -> str:
    title = discord.utils.escape_markdown(source.title[:160])
    host = discord.utils.escape_markdown(source.host[:100])
    snippet = discord.utils.escape_markdown(source.snippet[:240])
    line = f"`{index:02d}` **[{title}]({_safe_markdown_url(source.url)})**\n{host}"
    return f"{line} · {snippet}" if snippet else line


def web_search_embed(response: WebSearchResponse) -> discord.Embed:
    if not response.sources:
        return command_embed(
            "検索結果が見つかりませんでした",
            description=(
                f"**{discord.utils.escape_markdown(response.query)}** に一致する情報源を"
                "取得できませんでした。検索サービス側の一時的な不調も考えられます。"
                "同じ検索をすぐ繰り返すより、少し待つかキーワードを変えてみてください。"
            ),
            fields=(
                EmbedField("検索エンジン", response.backend),
                EmbedField("警告", str(len(response.warnings))),
            ),
            tone=EmbedTone.WARNING,
        )
    lines: list[str] = []
    used = 0
    for index, source in enumerate(response.sources, start=1):
        line = _web_source_text(index, source)
        if used + len(line) + 2 > 3_700:
            break
        lines.append(line)
        used += len(line) + 2
    coverage = f"候補 {response.candidate_count}件・表示 {len(response.sources)}件"
    if response.maybe_more:
        coverage += "・ほかにも候補あり"
    fields = [EmbedField("検索範囲", coverage, inline=False)]
    if response.warnings:
        fields.append(
            EmbedField(
                "検索サービスの状態",
                f"{len(response.warnings)}件の情報源で問題が報告されました。",
                inline=False,
            )
        )
    return command_embed(
        "検索結果",
        description="\n\n".join(lines),
        fields=tuple(fields),
        tone=EmbedTone.SUCCESS,
    )


def web_fetch_embed(response: WebFetchResponse) -> discord.Embed:
    excerpt = discord.utils.escape_markdown(response.text[:3_500])
    if not excerpt:
        excerpt = "この範囲には読み取れる文章がありませんでした。"
    fields = [
        EmbedField("出典", f"[元のページを開く]({_safe_markdown_url(response.url)})"),
        EmbedField("形式", response.content_type),
        EmbedField(
            "本文",
            f"{response.offset + len(response.text):,} / {response.total_characters:,}文字",
        ),
    ]
    if response.next_offset is not None:
        fields.append(EmbedField("続きの開始位置", str(response.next_offset)))
    return command_embed(
        response.title[:256],
        description=excerpt,
        fields=tuple(fields),
        tone=EmbedTone.SUCCESS,
    )


class WebFetchContinueView(discord.ui.View):
    """Continue a bounded Fetch result without making the user re-enter its URL."""

    def __init__(
        self,
        runtime: SimajilordRuntime,
        response: WebFetchResponse,
    ) -> None:
        super().__init__(timeout=600)
        self.runtime = runtime
        self.url = response.url
        self.next_offset = response.next_offset

    @discord.ui.button(
        label="続きを読む",
        style=discord.ButtonStyle.primary,
        custom_id="simajilord:web:fetch:continue",
    )
    async def continue_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[WebFetchContinueView],
    ) -> None:
        if self.next_offset is None:
            await interaction.response.edit_message(view=None)
            return
        await interaction.response.defer()
        try:
            response = cast(
                WebFetchResponse,
                await self.runtime.registry.invoke(
                    "web.fetch",
                    WebFetchRequest(
                        url=self.url,
                        offset=self.next_offset,
                        max_characters=3_500,
                    ),
                    invocation_context(interaction),
                ),
            )
            view = (
                WebFetchContinueView(self.runtime, response)
                if response.next_offset is not None
                else None
            )
            await interaction.edit_original_response(
                embed=web_fetch_embed(response),
                view=view,
            )
        except Exception as exc:
            await interaction.followup.send(
                embed=command_embed(
                    "処理できませんでした",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                ),
                ephemeral=True,
            )


def _web_match_text(index: int, match: WebTextMatch) -> str:
    before = discord.utils.escape_markdown(match.before)
    found = discord.utils.escape_markdown(match.match)
    after = discord.utils.escape_markdown(match.after)
    return f"`{index:02d}` …{before}**{found}**{after}…"


def web_find_embed(response: WebFindResponse) -> discord.Embed:
    if not response.matches:
        description = (
            f"[{discord.utils.escape_markdown(response.title)}]"
            f"({_safe_markdown_url(response.url)}) 内に "
            f"**{discord.utils.escape_markdown(response.pattern)}** は見つかりませんでした。"
        )
        return command_embed(
            "ページ内に見つかりませんでした",
            description=description,
            tone=EmbedTone.WARNING,
        )
    lines = tuple(
        _web_match_text(index, match) for index, match in enumerate(response.matches, start=1)
    )
    return command_embed(
        f"{response.title} 内の検索結果"[:256],
        description="\n\n".join(lines)[:3_800],
        fields=(
            EmbedField(
                "一致",
                f"全{response.total_matches}件・{len(response.matches)}件を表示",
            ),
            EmbedField("出典", f"[元のページを開く]({_safe_markdown_url(response.url)})"),
        ),
        tone=EmbedTone.SUCCESS,
    )


def _discord_audio_session(
    bot: commands.Bot,
    runtime: SimajilordRuntime,
    guild_id: int | None,
) -> AudioSession:
    if guild_id is None:
        raise UserError("workspace.required")
    return runtime.audio.get_or_create(
        str(guild_id),
        lambda: DiscordAudioOutput(bot, guild_id),
    )


def _member_voice_channel(
    member: discord.abc.User,
) -> discord.VoiceChannel | discord.StageChannel | None:
    if not isinstance(member, discord.Member):
        return None
    state = member.voice
    if state is None or not isinstance(state.channel, (discord.VoiceChannel, discord.StageChannel)):
        return None
    return state.channel


def _require_same_voice(session: AudioSession, member: discord.abc.User) -> None:
    if not session.output.connected:
        if session.waiting_for_voice and not session.can_control_while_waiting(str(member.id)):
            raise UserError("audio.waiting_queue_restricted")
        return
    channel = _member_voice_channel(member)
    if (
        channel is None
        or session.destination_id is None
        or str(channel.id) != session.destination_id
    ):
        raise UserError("audio.same_voice_required")


async def _enqueue_interaction_track(
    runtime: SimajilordRuntime,
    interaction: discord.Interaction,
    *,
    reference: str,
    requested_by_name: str,
) -> AudioPlayResponse:
    return cast(
        AudioPlayResponse,
        await runtime.registry.invoke(
            "discord.play_audio",
            AudioPlayRequest(
                reference=reference,
                requested_by_name=requested_by_name,
            ),
            invocation_context(interaction),
        ),
    )


class MusicCandidateButton(discord.ui.Button["MusicSearchChoiceView"]):
    def __init__(self, index: int, candidate: AudioSearchItem, token: str) -> None:
        source = candidate.uploader or candidate.title
        label = f"{index + 1} · {source}"
        super().__init__(
            label=label[:80],
            style=(discord.ButtonStyle.primary if index == 0 else discord.ButtonStyle.secondary),
            custom_id=f"simajilord:music:choice:{token}:{index}",
            row=0,
        )
        self.index = index

    async def callback(self, interaction: discord.Interaction) -> None:
        if self.view is not None:
            await self.view.choose(interaction, self.index)


class MusicSearchChoiceView(discord.ui.View):
    """One-click disambiguation used only when zero-click selection is unsafe."""

    def __init__(
        self,
        bot: commands.Bot,
        runtime: SimajilordRuntime,
        response: AudioSearchResponse,
        *,
        requester_id: int,
        requester_name: str,
    ) -> None:
        super().__init__(timeout=90)
        self.bot = bot
        self.runtime = runtime
        self.search = response
        self.requester_id = requester_id
        self.requester_name = requester_name
        self.message: discord.Message | None = None
        self._lock = asyncio.Lock()
        self._selected = False
        token = secrets.token_hex(5)
        for index, candidate in enumerate(response.candidates[:5]):
            self.add_item(MusicCandidateButton(index, candidate, token))

    async def choose(self, interaction: discord.Interaction, index: int) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                embed=command_embed(
                    "ほかのユーザーが検索した候補です",
                    description="自分で曲を追加するには `/play` を実行してください。",
                    tone=EmbedTone.WARNING,
                ),
                ephemeral=True,
            )
            return
        async with self._lock:
            if self._selected:
                await interaction.response.send_message(
                    embed=command_embed(
                        "選択済みです",
                        description="選んだ曲をキューへ追加しています。",
                        tone=EmbedTone.WARNING,
                    ),
                    ephemeral=True,
                )
                return
            try:
                candidate = self.search.candidates[index]
            except IndexError:
                await interaction.response.send_message(
                    embed=command_embed(
                        "候補の有効期限が切れました",
                        description="もう一度 `/play` を実行して検索し直してください。",
                        tone=EmbedTone.ERROR,
                    ),
                    ephemeral=True,
                )
                return
            self._selected = True
            await interaction.response.defer(thinking=True)
            self._set_disabled(True)
            await interaction.edit_original_response(view=self)
            try:
                response = await _enqueue_interaction_track(
                    self.runtime,
                    interaction,
                    reference=candidate.reference,
                    requested_by_name=self.requester_name,
                )
                await interaction.edit_original_response(
                    embed=music_added_embed(response),
                    view=None,
                )
                self.stop()
            except Exception as exc:
                self._selected = False
                self._set_disabled(False)
                await interaction.edit_original_response(view=self)
                await send_error(interaction, exc)

    async def on_timeout(self) -> None:
        self._set_disabled(True)
        if self.message is not None:
            with suppress(discord.DiscordException):
                await self.message.edit(view=self)

    def _set_disabled(self, disabled: bool) -> None:
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = disabled


class MusicControlsView(discord.ui.View):
    """Persistent controls backed by the same capability API as commands and agents."""

    def __init__(
        self,
        runtime: SimajilordRuntime,
        dashboard: MusicDashboardManager | None = None,
        *,
        response: AudioQueueResponse | None = None,
    ) -> None:
        super().__init__(timeout=None)
        self.runtime = runtime
        self.dashboard = dashboard
        self._apply_state(response)

    def _apply_state(self, response: AudioQueueResponse | None) -> None:
        """Keep the panel concise while retaining one persistent callback set."""

        if response is None:
            return
        active = response.current is not None
        has_manual_queue = any(
            item.kind == AudioKind.MUSIC.value for item in response.pending
        )
        can_start = response.waiting_for_voice or response.resume_confirmation_required
        if not can_start:
            self.remove_item(self.start_button)
        if not active:
            self.remove_item(self.pause_button)
            self.remove_item(self.skip_button)
        elif response.paused or response.resume_confirmation_required:
            self.pause_button.label = "Resume"
            self.pause_button.style = discord.ButtonStyle.success
        else:
            self.pause_button.label = "Pause"
            self.pause_button.style = discord.ButtonStyle.secondary
        self.loop_button.label = f"Loop: {_loop_mode_label(response.loop_mode)}"
        self.mix_button.label = f"Mix: {'On' if response.autoplay_enabled else 'Off'}"
        self.mix_button.style = (
            discord.ButtonStyle.success
            if response.autoplay_enabled
            else discord.ButtonStyle.secondary
        )
        if not (active or has_manual_queue or response.autoplay_enabled):
            self.remove_item(self.stop_button)
        if response.destination_id is None:
            self.remove_item(self.leave_button)

    def _bind_dashboard(self, interaction: discord.Interaction) -> None:
        dashboard = self._dashboard_manager(interaction)
        if isinstance(dashboard, MusicDashboardManager):
            dashboard.bind(interaction.guild_id, interaction.channel_id)

    def _dashboard_manager(
        self,
        interaction: discord.Interaction,
    ) -> MusicDashboardManager | None:
        dashboard = self.dashboard
        if dashboard is None:
            dashboard = getattr(
                interaction.client,
                _MUSIC_DASHBOARD_ATTRIBUTE,
                None,
            )
        return dashboard if isinstance(dashboard, MusicDashboardManager) else None

    async def _publish_dashboard(
        self,
        interaction: discord.Interaction,
        session: AudioSession,
        response: AudioQueueResponse,
    ) -> None:
        dashboard = self._dashboard_manager(interaction)
        if dashboard is not None:
            await dashboard.publish(
                session,
                obsolete_message=interaction.message,
            )
            return
        await interaction.edit_original_response(
            embed=music_queue_embed(
                response,
                read_aloud_route=_active_read_aloud_route(
                    self.runtime,
                    session.workspace_id,
                ),
            ),
            view=MusicControlsView(
                self.runtime,
                self.dashboard,
                response=response,
            ),
        )

    async def _run(
        self,
        interaction: discord.Interaction,
        action: AudioAction,
        *,
        loop_mode: LoopMode | None = None,
        position_seconds: float | None = None,
        speed: float | None = None,
        pitch: float | None = None,
    ) -> None:
        try:
            self._bind_dashboard(interaction)
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            _require_same_voice(session, interaction.user)
            await interaction.response.defer()
            await self.runtime.registry.invoke(
                "audio.control",
                AudioControlRequest(
                    action=action,
                    loop_mode=loop_mode,
                    position_seconds=position_seconds,
                    speed=speed,
                    pitch=pitch,
                ),
                invocation_context(interaction),
            )
            response = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    invocation_context(interaction),
                ),
            )
            await self._publish_dashboard(interaction, session, response)
        except Exception as exc:
            await send_error(interaction, exc)

    @discord.ui.button(
        label="Start",
        style=discord.ButtonStyle.success,
        custom_id="simajilord:music:start",
        row=0,
    )
    async def start_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        try:
            self._bind_dashboard(interaction)
            channel = _member_voice_channel(interaction.user)
            if channel is None:
                raise UserError("先にボイスチャンネルへ参加してください。")
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            if session.output.connected:
                _require_same_voice(session, interaction.user)
            elif not session.can_start_for(str(interaction.user.id)):
                raise UserError("audio.waiting_queue_restricted")
            await interaction.response.defer()
            await self.runtime.registry.invoke(
                "discord.connect_voice",
                DiscordConnectVoiceRequest(channel_id=str(channel.id)),
                invocation_context(interaction),
            )
            response = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    invocation_context(interaction),
                ),
            )
            await self._publish_dashboard(interaction, session, response)
        except Exception as exc:
            await send_error(interaction, exc)

    @discord.ui.button(
        label="Pause",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:music:pause-resume",
        row=0,
    )
    async def pause_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        response = cast(
            AudioQueueResponse,
            await self.runtime.registry.invoke(
                "audio.queue",
                AudioQueueRequest(),
                invocation_context(interaction),
            ),
        )
        action = (
            AudioAction.RESUME
            if response.paused or response.resume_confirmation_required
            else AudioAction.PAUSE
        )
        await self._run(interaction, action)

    @discord.ui.button(
        label="Skip",
        style=discord.ButtonStyle.primary,
        custom_id="simajilord:music:skip",
        row=0,
    )
    async def skip_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        await self._run(interaction, AudioAction.SKIP)

    @discord.ui.button(
        label="Loop",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:music:loop",
        row=1,
    )
    async def loop_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        response = cast(
            AudioQueueResponse,
            await self.runtime.registry.invoke(
                "audio.queue",
                AudioQueueRequest(),
                invocation_context(interaction),
            ),
        )
        modes = (LoopMode.NONE, LoopMode.TRACK, LoopMode.QUEUE)
        current = LoopMode(response.loop_mode)
        next_mode = modes[(modes.index(current) + 1) % len(modes)]
        if response.autoplay_enabled and next_mode is not LoopMode.NONE:
            await interaction.response.send_message(
                embed=command_embed(
                    "Mix is on",
                    description=(
                        "Loop and Mix cannot run together.\n"
                        "Would you like to stop Mix and switch to "
                        f"**{_loop_mode_label(next_mode.value)} loop**?"
                    ),
                    tone=EmbedTone.WARNING,
                ),
                view=LoopMixConflictView(
                    self.runtime,
                    self.dashboard,
                    requester_id=interaction.user.id,
                    loop_mode=next_mode,
                ),
                ephemeral=True,
            )
            return
        await self._run(
            interaction,
            AudioAction.LOOP,
            loop_mode=next_mode,
        )

    @discord.ui.button(
        label="Mix",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:music:mix",
        row=1,
    )
    async def mix_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        try:
            self._bind_dashboard(interaction)
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            _require_same_voice(session, interaction.user)
            current = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    invocation_context(interaction),
                ),
            )
            if (
                not current.autoplay_enabled
                and LoopMode(current.loop_mode) is not LoopMode.NONE
            ):
                await interaction.response.send_message(
                    embed=command_embed(
                        "Loop is on",
                        description=(
                            "Loop and Mix cannot run together.\n"
                            "Would you like to turn off Loop and switch to **Mix**?"
                        ),
                        tone=EmbedTone.WARNING,
                    ),
                    view=LoopMixConflictView(
                        self.runtime,
                        self.dashboard,
                        requester_id=interaction.user.id,
                    ),
                    ephemeral=True,
                )
                return
            await interaction.response.defer()
            await self.runtime.registry.invoke(
                "audio.mix",
                AudioMixRequest(enabled=not current.autoplay_enabled),
                invocation_context(interaction),
            )
            response = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    invocation_context(interaction),
                ),
            )
            await self._publish_dashboard(interaction, session, response)
        except Exception as exc:
            await send_error(interaction, exc)

    @discord.ui.button(
        label="Stop",
        style=discord.ButtonStyle.danger,
        custom_id="simajilord:music:stop",
        row=0,
    )
    async def stop_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        await self._run(interaction, AudioAction.STOP)

    @discord.ui.button(
        label="Add music",
        style=discord.ButtonStyle.primary,
        custom_id="simajilord:audio:add-music",
        row=1,
    )
    async def add_music_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        await interaction.response.send_modal(
            MusicAddModal(self.runtime, self._dashboard_manager(interaction))
        )

    @discord.ui.button(
        label="Read aloud",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:audio:read-aloud",
        row=1,
    )
    async def read_aloud_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        await _send_read_aloud_setup(interaction, self.runtime)

    @discord.ui.button(
        label="Leave",
        style=discord.ButtonStyle.danger,
        custom_id="simajilord:music:leave",
        row=1,
    )
    async def leave_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        await self._run(interaction, AudioAction.LEAVE)


class MusicAddModal(discord.ui.Modal, title="Add music"):
    reference: discord.ui.TextInput[MusicAddModal] = discord.ui.TextInput(
        label="Song, artist, or public URL",
        placeholder="What would you like to hear?",
        min_length=1,
        max_length=500,
    )

    def __init__(
        self,
        runtime: SimajilordRuntime,
        dashboard: MusicDashboardManager | None,
    ) -> None:
        super().__init__(timeout=5 * 60)
        self.runtime = runtime
        self.dashboard = dashboard

    async def on_submit(self, interaction: discord.Interaction) -> None:
        try:
            await interaction.response.defer(thinking=True, ephemeral=True)
            if self.dashboard is not None:
                self.dashboard.bind(interaction.guild_id, interaction.channel_id)
            bot = cast(commands.Bot, interaction.client)
            _discord_audio_session(bot, self.runtime, interaction.guild_id)
            reference = str(self.reference).strip()
            selected_reference = reference
            if "://" not in reference:
                search = cast(
                    AudioSearchResponse,
                    await self.runtime.registry.invoke(
                        "audio.search",
                        AudioSearchRequest(query=reference, limit=5),
                        invocation_context(interaction),
                    ),
                )
                if search.selection_required:
                    await interaction.edit_original_response(
                        embed=music_search_embed(search),
                        view=MusicSearchChoiceView(
                            bot,
                            self.runtime,
                            search,
                            requester_id=interaction.user.id,
                            requester_name=interaction.user.display_name,
                        ),
                    )
                    return
                if search.selected_index is None:
                    raise UserError("audio.search_empty")
                selected_reference = search.candidates[search.selected_index].reference
            response = await _enqueue_interaction_track(
                self.runtime,
                interaction,
                reference=selected_reference,
                requested_by_name=interaction.user.display_name,
            )
            await interaction.edit_original_response(
                embed=music_added_embed(response),
                view=None,
            )
        except Exception as exc:
            await edit_deferred_error(interaction, exc)


class LoopMixConflictView(discord.ui.View):
    """Ask before replacing one mutually exclusive playback mode."""

    def __init__(
        self,
        runtime: SimajilordRuntime,
        dashboard: MusicDashboardManager | None,
        *,
        requester_id: int,
        loop_mode: LoopMode | None = None,
        seed_references: tuple[str, ...] = (),
    ) -> None:
        super().__init__(timeout=60)
        self.runtime = runtime
        self.dashboard = dashboard
        self.requester_id = requester_id
        self.loop_mode = loop_mode
        self.seed_references = seed_references
        self.confirm_button.label = (
            "Switch to Mix" if loop_mode is None else "Switch to Loop"
        )

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "Only the person who requested this change can confirm it.",
            ephemeral=True,
        )
        return False

    @discord.ui.button(
        label="Switch",
        style=discord.ButtonStyle.primary,
        custom_id="simajilord:music:replace-conflict",
    )
    async def confirm_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[LoopMixConflictView],
    ) -> None:
        try:
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            _require_same_voice(session, interaction.user)
            await interaction.response.defer()
            if self.loop_mode is None:
                await self.runtime.registry.invoke(
                    "audio.mix",
                    AudioMixRequest(
                        enabled=True,
                        seed_references=self.seed_references,
                        replace_loop=True,
                    ),
                    invocation_context(interaction),
                )
                title = "Switched to Mix"
                description = "Loop is off. Mix will supply related tracks continuously."
            else:
                await self.runtime.registry.invoke(
                    "audio.control",
                    AudioControlRequest(
                        action=AudioAction.LOOP,
                        loop_mode=self.loop_mode,
                        replace_mix=True,
                    ),
                    invocation_context(interaction),
                )
                title = "Switched to Loop"
                description = (
                    "Mix is off. Loop is now "
                    f"**{_loop_mode_label(self.loop_mode.value)}**."
                )
            await interaction.edit_original_response(
                embed=command_embed(
                    title,
                    description=description,
                    tone=EmbedTone.SUCCESS,
                ),
                view=None,
            )
        except Exception as exc:
            await edit_deferred_error(interaction, exc)

    @discord.ui.button(
        label="Keep current mode",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:music:keep-conflict",
    )
    async def keep_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[LoopMixConflictView],
    ) -> None:
        await interaction.response.edit_message(
            embed=command_embed(
                "No changes made",
                description="The current playback mode is unchanged.",
            ),
            view=None,
        )

def invocation_context(interaction: discord.Interaction) -> InvocationContext:
    return InvocationContext(
        actor_id=str(interaction.user.id),
        workspace_id=str(interaction.guild_id) if interaction.guild_id else None,
        transport="discord",
        request_id=str(interaction.id),
        origin_resource_id=(
            str(interaction.channel_id) if interaction.channel_id is not None else None
        ),
    )


def prefix_context(context: BotContext) -> InvocationContext:
    return InvocationContext(
        actor_id=str(context.author.id),
        workspace_id=str(context.guild.id) if context.guild else None,
        transport="discord",
        request_id=str(context.message.id),
        origin_resource_id=str(context.channel.id),
    )


def message_context(message: discord.Message) -> InvocationContext:
    return InvocationContext(
        actor_id=str(message.author.id),
        workspace_id=str(message.guild.id) if message.guild is not None else None,
        transport="discord",
        request_id=str(message.id),
        origin_resource_id=str(message.channel.id),
    )


def error_message(error: Exception) -> str:
    if isinstance(error, MediaError):
        return _MEDIA_ERROR_MESSAGES.get(error.category, _MEDIA_ERROR_MESSAGES["unknown"])
    if isinstance(error, WebError):
        return _WEB_ERROR_MESSAGES.get(
            error.category,
            "Web処理を完了できませんでした。",
        )
    if isinstance(error, ModerationError):
        return _MODERATION_ERROR_MESSAGES.get(
            error.category,
            "HIVEの解析を完了できませんでした。",
        )
    if isinstance(error, UserError):
        return _ERROR_MESSAGES.get(error.code, error.code)
    log.exception("Unhandled Discord command error", exc_info=error)
    return "予期しないエラーが発生しました。ホスト側のログを確認してください。"


async def send_error(interaction: discord.Interaction, error: Exception) -> None:
    embed = command_embed(
        "処理できませんでした",
        description=error_message(error),
        tone=EmbedTone.ERROR,
    )
    if interaction.response.is_done():
        await interaction.followup.send(embed=embed, ephemeral=True)
    else:
        await interaction.response.send_message(embed=embed, ephemeral=True)


async def edit_deferred_error(
    interaction: discord.Interaction,
    error: Exception,
) -> None:
    await interaction.edit_original_response(
        embed=command_embed(
            "処理できませんでした",
            description=error_message(error),
            tone=EmbedTone.ERROR,
        )
    )


class SystemCog(commands.Cog):
    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime

    @app_commands.command(
        name="ping",
        description="BOTの稼働状態とDiscordへの応答速度を確認します。",
    )
    async def ping(self, interaction: discord.Interaction) -> None:
        response = cast(
            PingResponse,
            await self.runtime.registry.invoke(
                "system.ping",
                PingRequest(transport_latency_ms=round(self.bot.latency * 1_000, 1)),
                invocation_context(interaction),
            ),
        )
        await interaction.response.send_message(
            embed=command_embed(
                "稼働状況",
                fields=(
                    EmbedField("状態", "正常" if response.status == "ok" else response.status),
                    EmbedField(
                        "Discord応答時間",
                        f"{response.transport_latency_ms:.1f} ms",
                    ),
                ),
                tone=EmbedTone.SUCCESS,
            )
        )

    @app_commands.command(
        name="capabilities",
        description="目的に合うSimajilordの機能を探します。",
    )
    @app_commands.describe(query="やりたいことを入力してください")
    async def capabilities(self, interaction: discord.Interaction, query: str = "") -> None:
        response = cast(
            CapabilitySearchResponse,
            await self.runtime.registry.invoke(
                "system.discover_capabilities",
                CapabilitySearchRequest(query=query, limit=8),
                invocation_context(interaction),
            ),
        )
        if not response.capabilities:
            await interaction.response.send_message(
                embed=command_embed(
                    "利用できる機能",
                    description="条件に合う機能は見つかりませんでした。",
                    tone=EmbedTone.WARNING,
                )
            )
            return
        lines = [
            f"• `{item.name}` — {item.summary} "
            f"— 危険度: **{_risk_label(item.risk)}** / "
            f"承認: **{_approval_label(item.approval)}**"
            for item in response.capabilities
        ]
        await interaction.response.send_message(
            embed=command_embed(
                "利用できる機能",
                description="\n".join(lines),
                fields=(EmbedField("検索内容", query or "すべて", inline=False),),
            )
        )

    @app_commands.command(
        name="about",
        description="Simajilord AIとDiscord BOTの役割を表示します。",
    )
    async def about(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            embed=command_embed(
                "Simajilord AIについて",
                description=(
                    "Simajilord AIは、AIと各種機能を共通APIでつなぐ基盤です。"
                    "このBOTはDiscordへの窓口であり、音楽・読み上げ・メディア処理・"
                    "AIの判断はDiscordに依存しないSimajilord基盤上で動作します。"
                ),
            )
        )

    @app_commands.command(name="uptime", description="BOTの起動時刻と連続稼働時間を表示します。")
    async def uptime(self, interaction: discord.Interaction) -> None:
        response = cast(
            UptimeResponse,
            await self.runtime.registry.invoke(
                "system.uptime",
                UptimeRequest(),
                invocation_context(interaction),
            ),
        )
        total_seconds = int(response.uptime_seconds)
        days, remainder = divmod(total_seconds, 86_400)
        hours, remainder = divmod(remainder, 3_600)
        minutes, seconds = divmod(remainder, 60)
        await interaction.response.send_message(
            embed=command_embed(
                "稼働時間",
                fields=(
                    EmbedField(
                        "起動日時",
                        f"<t:{int(response.started_at.timestamp())}:F>",
                        inline=False,
                    ),
                    EmbedField(
                        "連続稼働",
                        f"{days}日 {hours}時間 {minutes}分 {seconds}秒",
                        inline=False,
                    ),
                ),
                tone=EmbedTone.SUCCESS,
            )
        )

    @app_commands.command(name="status", description="Simajilord基盤の詳しい状態を表示します。")
    async def status(self, interaction: discord.Interaction) -> None:
        response = cast(
            StatusResponse,
            await self.runtime.registry.invoke(
                "system.status",
                StatusRequest(),
                invocation_context(interaction),
            ),
        )
        await interaction.response.send_message(
            embed=command_embed(
                "Simajilordの状態",
                fields=(
                    EmbedField(
                        "稼働状態",
                        f"システム: **{'正常' if response.status == 'ok' else response.status}**\n"
                        f"AI: **{'有効' if response.model_runtime == 'enabled' else '無効'}**",
                    ),
                    EmbedField(
                        "機能",
                        f"登録済みAPI: **{response.capability_count}**\n"
                        "音声セッション: "
                        f"{response.active_audio_session_count}/"
                        f"{response.audio_session_count}件が稼働中\n"
                        f"読み上げ: **{response.speech_provider.upper()} "
                        f"{response.speech_voice}**",
                    ),
                    EmbedField(
                        "Web検索",
                        "状態: "
                        f"**{'利用可能' if response.web_search_ready else '一部制限'}**\n"
                        f"検索エンジン: **{response.web_search_backend}**",
                    ),
                ),
                tone=EmbedTone.SUCCESS,
            )
        )


class MusicCog(commands.Cog):
    music = app_commands.Group(
        name="music",
        description="音楽の再生と詳細な操作を行います。",
    )

    def __init__(
        self,
        bot: commands.Bot,
        runtime: SimajilordRuntime,
        dashboard: MusicDashboardManager | None = None,
    ) -> None:
        self.bot = bot
        self.runtime = runtime
        existing = getattr(bot, _MUSIC_DASHBOARD_ATTRIBUTE, None)
        self.dashboard = dashboard or (
            existing
            if isinstance(existing, MusicDashboardManager)
            else MusicDashboardManager(bot, runtime)
        )

    async def _send_play(self, interaction: discord.Interaction, reference: str) -> None:
        try:
            await interaction.response.defer(thinking=True)
            self.dashboard.bind(interaction.guild_id, interaction.channel_id)
            _discord_audio_session(self.bot, self.runtime, interaction.guild_id)
            selected_reference = reference
            if "://" not in reference:
                search = cast(
                    AudioSearchResponse,
                    await self.runtime.registry.invoke(
                        "audio.search",
                        AudioSearchRequest(query=reference, limit=5),
                        invocation_context(interaction),
                    ),
                )
                if search.selection_required:
                    view = MusicSearchChoiceView(
                        self.bot,
                        self.runtime,
                        search,
                        requester_id=interaction.user.id,
                        requester_name=interaction.user.display_name,
                    )
                    message = await interaction.followup.send(
                        embed=music_search_embed(search),
                        view=view,
                        silent=True,
                        wait=True,
                    )
                    view.message = message
                    return
                if search.selected_index is None:
                    raise UserError("audio.search_empty")
                selected_reference = search.candidates[search.selected_index].reference
            response = await _enqueue_interaction_track(
                self.runtime,
                interaction,
                reference=selected_reference,
                requested_by_name=interaction.user.display_name,
            )
            message = await interaction.followup.send(
                embed=music_added_embed(response),
                silent=True,
                wait=True,
            )
            _schedule_message_delete(message, delay=8)
        except Exception as exc:
            await send_error(interaction, exc)

    @app_commands.command(
        name="play",
        description="URLまたは曲名から音楽を再生します。",
    )
    @app_commands.describe(reference="公開URLまたは曲名")
    async def quick_play(self, interaction: discord.Interaction, reference: str) -> None:
        await self._send_play(interaction, reference)

    @music.command(
        name="play",
        description="URLまたは曲名から音楽を再生します。",
    )
    @app_commands.describe(reference="公開URLまたは曲名")
    async def play(self, interaction: discord.Interaction, reference: str) -> None:
        await self._send_play(interaction, reference)

    async def _send_queue(self, interaction: discord.Interaction, page: int = 1) -> None:
        try:
            await interaction.response.defer(ephemeral=True)
            self.dashboard.bind(interaction.guild_id, interaction.channel_id)
            session = _discord_audio_session(self.bot, self.runtime, interaction.guild_id)
            if page == 1:
                await self.dashboard.publish(session, force=True)
                await interaction.delete_original_response()
                return
            response = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    invocation_context(interaction),
                ),
            )
            await interaction.edit_original_response(
                embed=music_queue_embed(
                    response,
                    page=page,
                    read_aloud_route=_active_read_aloud_route(
                        self.runtime,
                        session.workspace_id,
                    ),
                ),
                view=MusicControlsView(
                    self.runtime,
                    self.dashboard,
                    response=response,
                ),
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @app_commands.command(
        name="audio",
        description="Open music controls and read-aloud setup in one panel.",
    )
    async def audio(self, interaction: discord.Interaction) -> None:
        await self._send_queue(interaction)

    @app_commands.command(
        name="queue",
        description="再生中の曲とキューを表示します。",
    )
    async def quick_queue(
        self,
        interaction: discord.Interaction,
        page: app_commands.Range[int, 1, 50] = 1,
    ) -> None:
        await self._send_queue(interaction, int(page))

    @music.command(
        name="queue",
        description="再生中の曲とキューを表示します。",
    )
    async def queue(
        self,
        interaction: discord.Interaction,
        page: app_commands.Range[int, 1, 50] = 1,
    ) -> None:
        await self._send_queue(interaction, int(page))

    @app_commands.command(
        name="nowplaying",
        description="現在再生している曲を表示します。",
    )
    async def now_playing(self, interaction: discord.Interaction) -> None:
        try:
            self.dashboard.bind(interaction.guild_id, interaction.channel_id)
            _discord_audio_session(self.bot, self.runtime, interaction.guild_id)
            response = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=music_queue_embed(
                    response,
                    read_aloud_route=_active_read_aloud_route(
                        self.runtime,
                        str(interaction.guild_id)
                    ),
                ),
                view=MusicControlsView(
                    self.runtime,
                    self.dashboard,
                    response=response,
                ),
                silent=True,
            )
        except Exception as exc:
            await send_error(interaction, exc)

    async def _send_history(self, interaction: discord.Interaction, limit: int) -> None:
        try:
            self.dashboard.bind(interaction.guild_id, interaction.channel_id)
            _discord_audio_session(self.bot, self.runtime, interaction.guild_id)
            response = cast(
                AudioHistoryResponse,
                await self.runtime.registry.invoke(
                    "audio.history",
                    AudioHistoryRequest(limit=limit),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=music_history_embed(response),
                silent=True,
            )
        except Exception as exc:
            await send_error(interaction, exc)

    async def _send_mix(
        self,
        interaction: discord.Interaction,
        *,
        enabled: bool,
        seeds: str | None,
    ) -> None:
        try:
            self.dashboard.bind(interaction.guild_id, interaction.channel_id)
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            _require_same_voice(session, interaction.user)
            references = tuple(part for part in seeds.split() if part) if seeds is not None else ()
            if enabled:
                current = cast(
                    AudioQueueResponse,
                    await self.runtime.registry.invoke(
                        "audio.queue",
                        AudioQueueRequest(),
                        invocation_context(interaction),
                    ),
                )
                if LoopMode(current.loop_mode) is not LoopMode.NONE:
                    await interaction.response.send_message(
                        embed=command_embed(
                            "Loop is on",
                            description=(
                                "Loop and Mix cannot run together.\n"
                                "Would you like to turn off Loop and switch to **Mix**?"
                            ),
                            tone=EmbedTone.WARNING,
                        ),
                        view=LoopMixConflictView(
                            self.runtime,
                            self.dashboard,
                            requester_id=interaction.user.id,
                            seed_references=references,
                        ),
                        ephemeral=True,
                    )
                    return
            response = cast(
                AudioMixResponse,
                await self.runtime.registry.invoke(
                    "audio.mix",
                    AudioMixRequest(
                        enabled=enabled,
                        seed_references=references,
                    ),
                    invocation_context(interaction),
                ),
            )
            if response.enabled:
                description = (
                    f"**{len(response.seed_references)}曲**を起点に自動選曲します。"
                    "希望曲は常にMixより先に再生されます。"
                )
            else:
                description = "自動選曲を停止しました。希望曲のキューはそのままです。"
            await interaction.response.send_message(
                embed=command_embed(
                    "Mixを開始しました" if response.enabled else "Mixを停止しました",
                    description=description,
                    tone=EmbedTone.SUCCESS,
                ),
                silent=True,
                delete_after=8,
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @app_commands.command(
        name="mix",
        description="希望曲を優先しながら、関連曲を途切れず自動再生します。",
    )
    @app_commands.describe(
        enabled="オンで開始、オフで停止",
        seeds="起点にするYouTube URL (空白区切り・最大8件)",
    )
    async def quick_mix(
        self,
        interaction: discord.Interaction,
        enabled: bool = True,
        seeds: str | None = None,
    ) -> None:
        await self._send_mix(interaction, enabled=enabled, seeds=seeds)

    @music.command(
        name="mix",
        description="希望曲を優先しながら、関連曲を途切れず自動再生します。",
    )
    @app_commands.describe(
        enabled="オンで開始、オフで停止",
        seeds="起点にするYouTube URL (空白区切り・最大8件)",
    )
    async def mix(
        self,
        interaction: discord.Interaction,
        enabled: bool = True,
        seeds: str | None = None,
    ) -> None:
        await self._send_mix(interaction, enabled=enabled, seeds=seeds)

    async def _send_fresh_mix(
        self,
        interaction: discord.Interaction,
        *,
        prompt: str,
        minutes: int,
        energy: Literal["calm", "steady", "rising"],
        vocals: Literal["low", "balanced", "any"],
    ) -> None:
        try:
            await interaction.response.defer(thinking=True)
            self.dashboard.bind(interaction.guild_id, interaction.channel_id)
            _discord_audio_session(self.bot, self.runtime, interaction.guild_id)
            response = cast(
                FreshMixPreviewResponse,
                await self.runtime.registry.invoke(
                    "discord.plan_fresh_mix",
                    FreshMixPlanRequest(
                        prompt=prompt,
                        target_minutes=minutes,
                        energy=FreshMixEnergy(energy),
                        vocals=FreshMixVocals(vocals),
                        history_policy="ignore",
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.edit_original_response(
                embed=fresh_mix_preview_embed(response),
                view=FreshMixPreviewView(
                    self.runtime,
                    self.dashboard,
                    response,
                    requester_id=interaction.user.id,
                    requester_name=interaction.user.display_name,
                ),
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @app_commands.command(
        name="freshmix",
        description="作業内容から履歴を使わず、新しいMixを組み立てます。",
    )
    @app_commands.describe(
        prompt="作業内容、雰囲気、ジャンルなど",
        minutes="目標の長さ",
        energy="全体の盛り上がり方",
        vocals="歌声の量",
    )
    async def quick_fresh_mix(
        self,
        interaction: discord.Interaction,
        prompt: str,
        minutes: app_commands.Range[int, 15, 240] = 60,
        energy: Literal["calm", "steady", "rising"] = "steady",
        vocals: Literal["low", "balanced", "any"] = "balanced",
    ) -> None:
        await self._send_fresh_mix(
            interaction,
            prompt=prompt,
            minutes=int(minutes),
            energy=energy,
            vocals=vocals,
        )

    @music.command(
        name="freshmix",
        description="作業内容から履歴を使わず、新しいMixを組み立てます。",
    )
    @app_commands.describe(
        prompt="作業内容、雰囲気、ジャンルなど",
        minutes="目標の長さ",
        energy="全体の盛り上がり方",
        vocals="歌声の量",
    )
    async def fresh_mix(
        self,
        interaction: discord.Interaction,
        prompt: str,
        minutes: app_commands.Range[int, 15, 240] = 60,
        energy: Literal["calm", "steady", "rising"] = "steady",
        vocals: Literal["low", "balanced", "any"] = "balanced",
    ) -> None:
        await self._send_fresh_mix(
            interaction,
            prompt=prompt,
            minutes=int(minutes),
            energy=energy,
            vocals=vocals,
        )

    @app_commands.command(
        name="history",
        description="最近再生した曲と、追加したユーザーを表示します。",
    )
    async def quick_history(
        self,
        interaction: discord.Interaction,
        limit: app_commands.Range[int, 1, 25] = 10,
    ) -> None:
        await self._send_history(interaction, int(limit))

    @music.command(
        name="history",
        description="最近再生した曲と、追加したユーザーを表示します。",
    )
    async def history(
        self,
        interaction: discord.Interaction,
        limit: app_commands.Range[int, 1, 25] = 10,
    ) -> None:
        await self._send_history(interaction, int(limit))

    async def _control(
        self,
        interaction: discord.Interaction,
        action: AudioAction,
        loop_mode: LoopMode | None = None,
        enabled: bool | None = None,
        position_seconds: float | None = None,
        speed: float | None = None,
        pitch: float | None = None,
        position: int | None = None,
        to_position: int | None = None,
        music_percent: int | None = None,
        speech_percent: int | None = None,
    ) -> None:
        try:
            self.dashboard.bind(interaction.guild_id, interaction.channel_id)
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            _require_same_voice(session, interaction.user)
            response = cast(
                AudioControlResponse,
                await self.runtime.registry.invoke(
                    "audio.control",
                    AudioControlRequest(
                        action=action,
                        loop_mode=loop_mode,
                        enabled=enabled,
                        position_seconds=position_seconds,
                        speed=speed,
                        pitch=pitch,
                        position=position,
                        to_position=to_position,
                        music_percent=music_percent,
                        speech_percent=speech_percent,
                    ),
                    invocation_context(interaction),
                ),
            )
            if response.action == AudioAction.LOOP.value:
                message = f"ループを **{_loop_mode_label(response.loop_mode or '')}** にしました。"
            elif response.action == AudioAction.REMOVE.value:
                message = f"**{response.affected_title}** をキューから削除しました。"
            elif response.action == AudioAction.AUTO_LEAVE.value:
                message = f"自動退出を **{'オン' if response.enabled else 'オフ'}** にしました。"
            elif response.action == AudioAction.SHUFFLE.value:
                message = "再生待ちの曲をシャッフルしました。"
            elif response.action == AudioAction.SEEK.value:
                formatted_position = _duration(response.position_seconds or 0)
                message = f"再生位置を `{formatted_position}` に移動しました。"
            elif response.action == AudioAction.TUNE.value:
                message = (
                    f"速度 **{response.speed:.2f}倍**・"
                    f"ピッチ **{response.pitch:.2f}倍** にしました。"
                    if response.speed is not None and response.pitch is not None
                    else "再生設定を変更しました。"
                )
            elif response.action == AudioAction.VOLUME.value:
                message = (
                    f"音楽 **{response.music_volume_percent}%**・"
                    f"読み上げ **{response.speech_volume_percent}%** にしました。"
                )
            elif response.action == AudioAction.MOVE.value:
                message = f"**{response.affected_title}** の再生順を変更しました。"
            elif response.action == AudioAction.CLEAR_MINE.value:
                removed_count = response.removed_count or 0
                message = f"自分が追加した待機曲を **{removed_count}曲** 削除しました。"
            else:
                message = _AUDIO_ACTION_MESSAGES[response.action]
            await interaction.response.send_message(
                embed=command_embed(
                    "音楽を操作しました",
                    description=message,
                    tone=EmbedTone.SUCCESS,
                ),
                silent=True,
                delete_after=8,
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @music.command(name="pause", description="再生を一時停止します。")
    async def pause(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.PAUSE)

    @music.command(name="resume", description="一時停止した曲を再開します。")
    async def resume(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.RESUME)

    @music.command(name="skip", description="再生中の曲をスキップします。")
    async def skip(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.SKIP)

    @music.command(name="stop", description="再生を停止し、キューを空にします。")
    async def stop(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.STOP)

    @music.command(name="leave", description="BOTをボイスチャンネルから退出させます。")
    async def leave(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.LEAVE)

    @music.command(name="loop", description="音楽のループ方法を設定します。")
    async def loop(
        self,
        interaction: discord.Interaction,
        mode: Literal["none", "track", "queue"],
    ) -> None:
        await self._control(interaction, AudioAction.LOOP, LoopMode(mode))

    @music.command(name="remove", description="指定した番号の曲をキューから削除します。")
    @app_commands.describe(position="「次に再生」に表示されている番号")
    async def remove(self, interaction: discord.Interaction, position: int) -> None:
        try:
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            _require_same_voice(session, interaction.user)
            response = cast(
                AudioControlResponse,
                await self.runtime.registry.invoke(
                    "audio.control",
                    AudioControlRequest(
                        action=AudioAction.REMOVE,
                        position=position,
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "キューから削除しました",
                    description=f"**{response.affected_title}**",
                    tone=EmbedTone.SUCCESS,
                ),
                silent=True,
                delete_after=8,
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @music.command(
        name="autoleave",
        description="最後のユーザーが退出したとき、キューを残してBOTも退出します。",
    )
    async def autoleave(self, interaction: discord.Interaction, enabled: bool) -> None:
        await self._control(
            interaction,
            AudioAction.AUTO_LEAVE,
            enabled=enabled,
        )

    @music.command(name="shuffle", description="再生待ちの曲をシャッフルします。")
    async def shuffle(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.SHUFFLE)

    @music.command(name="seek", description="再生中の曲の位置を移動します。")
    @app_commands.describe(position="1:23、+30、-10 のように指定")
    async def seek(self, interaction: discord.Interaction, position: str) -> None:
        try:
            parsed, relative = _parse_position(position)
            if relative:
                snapshot = cast(
                    AudioQueueResponse,
                    await self.runtime.registry.invoke(
                        "audio.queue",
                        AudioQueueRequest(),
                        invocation_context(interaction),
                    ),
                )
                parsed += snapshot.position_seconds
            await self._control(
                interaction,
                AudioAction.SEEK,
                position_seconds=max(0.0, parsed),
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @music.command(name="tune", description="再生速度とピッチを調整します。")
    async def tune(
        self,
        interaction: discord.Interaction,
        speed: app_commands.Range[float, 0.5, 2.0] = 1.0,
        pitch: app_commands.Range[float, 0.5, 2.0] = 1.0,
    ) -> None:
        await self._control(
            interaction,
            AudioAction.TUNE,
            speed=float(speed),
            pitch=float(pitch),
        )

    @music.command(
        name="volume",
        description="音楽と読み上げの音量を個別に設定します。",
    )
    @app_commands.describe(
        music="音楽の音量 (0〜200%・省略時は変更しない)",
        read_aloud="読み上げの音量 (0〜200%・省略時は変更しない)",
    )
    async def volume(
        self,
        interaction: discord.Interaction,
        music: app_commands.Range[int, 0, 200] | None = None,
        read_aloud: app_commands.Range[int, 0, 200] | None = None,
    ) -> None:
        await self._control(
            interaction,
            AudioAction.VOLUME,
            music_percent=None if music is None else int(music),
            speech_percent=None if read_aloud is None else int(read_aloud),
        )

    @music.command(name="move", description="待機曲の再生順を変更します。")
    @app_commands.describe(
        source="現在のキュー番号",
        destination="移動後のキュー番号",
    )
    async def move(
        self,
        interaction: discord.Interaction,
        source: int,
        destination: int,
    ) -> None:
        await self._control(
            interaction,
            AudioAction.MOVE,
            position=source,
            to_position=destination,
        )

    @music.command(
        name="clear-mine",
        description="自分が追加した待機曲だけをキューから削除します。",
    )
    async def clear_mine(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.CLEAR_MINE)


_READ_ALOUD_CHANNEL_TYPES = [
    discord.ChannelType.text,
    discord.ChannelType.news,
    discord.ChannelType.voice,
    discord.ChannelType.stage_voice,
    discord.ChannelType.news_thread,
    discord.ChannelType.public_thread,
    discord.ChannelType.private_thread,
]


class ReadAloudChannelSelect(discord.ui.ChannelSelect[discord.ui.View]):
    """Stage a bounded set of conversation channels for explicit confirmation."""

    def __init__(
        self,
        runtime: SimajilordRuntime,
        *,
        requester_id: int,
        destination_id: int,
        default_values: tuple[discord.abc.GuildChannel | discord.Thread, ...],
    ) -> None:
        self.runtime = runtime
        self.requester_id = requester_id
        self.destination_id = destination_id
        self.selected_channel_ids = tuple(str(channel.id) for channel in default_values)
        super().__init__(
            custom_id="simajilord:readaloud:channels",
            channel_types=_READ_ALOUD_CHANNEL_TYPES,
            placeholder="読み上げるチャンネルを選択",
            min_values=1,
            max_values=25,
            default_values=default_values,
            row=0,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the person who opened this setup can change the selection.",
                ephemeral=True,
            )
            return
        self.selected_channel_ids = tuple(str(channel.id) for channel in self.values)
        await interaction.response.defer()

    async def commit(self, interaction: discord.Interaction) -> None:
        """Save the staged route and establish voice only after Start is pressed."""

        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the person who opened this setup can start read aloud.",
                ephemeral=True,
            )
            return
        configured: ReadAloudResponse | None = None
        try:
            await interaction.response.defer()
            if not self.selected_channel_ids:
                raise UserError("read_aloud.source_channels_required")
            configured = cast(
                ReadAloudResponse,
                await self.runtime.registry.invoke(
                    "discord.manage_read_aloud",
                    ReadAloudRequest(
                        action=ReadAloudAction.ADD_SOURCES,
                        text_channel_ids=self.selected_channel_ids,
                        audio_destination_id=str(self.destination_id),
                    ),
                    invocation_context(interaction),
                ),
            )
            await self.runtime.registry.invoke(
                "discord.connect_voice",
                DiscordConnectVoiceRequest(channel_id=str(self.destination_id)),
                invocation_context(interaction),
            )
            await interaction.edit_original_response(
                embed=command_embed(
                    "Read aloud is ready",
                    description=(
                        "New messages from the selected channels will be spoken automatically."
                    ),
                    fields=(
                        EmbedField(
                            "Reading from",
                            "\n".join(
                                f"<#{channel_id}>" for channel_id in configured.text_channel_ids
                            ),
                            inline=False,
                        ),
                        EmbedField(
                            "Speaking in",
                            f"<#{configured.audio_destination_id}>",
                        ),
                        EmbedField("Connection", "Ready"),
                        EmbedField("Voice", _speech_voice_label(self.runtime)),
                    ),
                    tone=EmbedTone.SUCCESS,
                ),
                view=None,
            )
            dashboard = getattr(
                interaction.client,
                _MUSIC_DASHBOARD_ATTRIBUTE,
                None,
            )
            session = self.runtime.audio.find(str(interaction.guild_id))
            if isinstance(dashboard, MusicDashboardManager) and session is not None:
                await dashboard.publish(session)
        except Exception as exc:
            if configured is None:
                await send_error(interaction, exc)
                return
            log.exception(
                "Read-aloud route was saved but eager voice connection failed guild=%s channel=%s",
                interaction.guild_id,
                self.destination_id,
            )
            await interaction.edit_original_response(
                embed=command_embed(
                    "Read aloud was saved",
                    description=(
                        "The channels are configured, but the voice connection is not ready yet. "
                        "Simajilord will retry when the next message arrives."
                    ),
                    fields=(
                        EmbedField(
                            "Speaking in",
                            f"<#{configured.audio_destination_id}>",
                        ),
                        EmbedField("Connection", error_message(exc)),
                    ),
                    tone=EmbedTone.WARNING,
                ),
                view=None,
            )


class ReadAloudChannelSelectView(discord.ui.View):
    def __init__(
        self,
        runtime: SimajilordRuntime,
        *,
        requester_id: int,
        destination_id: int,
        default_values: tuple[discord.abc.GuildChannel | discord.Thread, ...],
    ) -> None:
        super().__init__(timeout=300)
        self.selector = ReadAloudChannelSelect(
            runtime,
            requester_id=requester_id,
            destination_id=destination_id,
            default_values=default_values,
        )
        self.add_item(self.selector)

    @discord.ui.button(
        label="Start",
        style=discord.ButtonStyle.success,
        custom_id="simajilord:readaloud:start",
        row=1,
    )
    async def start(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[ReadAloudChannelSelectView],
    ) -> None:
        await self.selector.commit(interaction)


def _read_aloud_setup(
    interaction: discord.Interaction,
    runtime: SimajilordRuntime,
) -> tuple[discord.Embed, ReadAloudChannelSelectView]:
    member = interaction.user
    source = interaction.channel
    if not isinstance(member, discord.Member):
        raise UserError("Use this control inside a server.")
    if not isinstance(
        source,
        (
            discord.TextChannel,
            discord.Thread,
            discord.VoiceChannel,
            discord.StageChannel,
        ),
    ):
        raise UserError("Open this control in a server conversation channel.")
    destination = member.voice.channel if member.voice is not None else None
    if not isinstance(destination, (discord.VoiceChannel, discord.StageChannel)):
        raise UserError("Join the voice channel where read aloud should speak first.")

    defaults: list[discord.abc.GuildChannel | discord.Thread] = []
    route = _active_read_aloud_route(runtime, str(member.guild.id))
    candidate_ids = (
        (*route.text_channel_ids, str(source.id))
        if route is not None and route.audio_destination_id == str(destination.id)
        else (str(source.id),)
    )
    for channel_id in dict.fromkeys(candidate_ids):
        selected = member.guild.get_channel_or_thread(int(channel_id))
        if selected is not None:
            defaults.append(selected)

    view = ReadAloudChannelSelectView(
        runtime,
        requester_id=member.id,
        destination_id=destination.id,
        default_values=tuple(defaults[:25]),
    )
    embed = command_embed(
        "Audio · Read aloud",
        description=(
            "Choose up to 25 text channels, threads, or voice-channel chats, "
            "then select **Start**."
        ),
        fields=(
            EmbedField("Current channel", source.mention),
            EmbedField("Speaking in", destination.mention),
            EmbedField("Voice", _speech_voice_label(runtime)),
        ),
    )
    return embed, view


async def _send_read_aloud_setup(
    interaction: discord.Interaction,
    runtime: SimajilordRuntime,
) -> None:
    try:
        embed, view = _read_aloud_setup(interaction, runtime)
        await interaction.response.send_message(
            embed=embed,
            view=view,
            ephemeral=True,
        )
    except Exception as exc:
        await send_error(interaction, exc)


class ReadAloudCog(commands.Cog):
    readaloud = app_commands.Group(
        name="readaloud",
        description="DiscordのメッセージをVCで自動読み上げします。",
    )

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime
        self._message_formatter = ReadAloudMessageFormatter(runtime.read_aloud)
        self._voice_transitions: dict[
            tuple[int, int],
            tuple[
                discord.Member,
                discord.VoiceChannel | discord.StageChannel | None,
                discord.VoiceChannel | discord.StageChannel | None,
            ],
        ] = {}
        self._announcement_tasks: dict[
            tuple[int, int],
            asyncio.Task[None],
        ] = {}

    async def cog_unload(self) -> None:
        for task in self._announcement_tasks.values():
            task.cancel()
        self._announcement_tasks.clear()
        self._voice_transitions.clear()

    @app_commands.command(
        name="join",
        description="選んだ会話チャンネルを、参加中のVCで読み上げます。",
    )
    async def join(self, interaction: discord.Interaction) -> None:
        await _send_read_aloud_setup(interaction, self.runtime)

    @readaloud.command(
        name="setup",
        description="読み上げる会話チャンネルとVCを設定し直します。",
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(
        text_channel="自動で読み上げるテキストチャンネルまたはVCチャット",
        voice_channel="読み上げ音声を流すボイスチャンネル",
        mode="音楽再生中のメッセージを待機させるか、読み飛ばすか",
    )
    async def setup(
        self,
        interaction: discord.Interaction,
        text_channel: discord.TextChannel | discord.VoiceChannel | None = None,
        voice_channel: discord.VoiceChannel | None = None,
        mode: Literal["queue", "skip_during_music"] = "queue",
    ) -> None:
        try:
            member = interaction.user
            if not isinstance(member, discord.Member) or not member.guild_permissions.manage_guild:
                raise UserError("この設定には「サーバー管理」権限が必要です。")
            selected_text = text_channel
            if selected_text is None and isinstance(
                interaction.channel,
                (discord.TextChannel, discord.VoiceChannel),
            ):
                selected_text = interaction.channel
            if selected_text is None:
                raise UserError("会話チャンネルを選んでください。")
            selected_voice = voice_channel
            if selected_voice is None and member.voice is not None:
                candidate = member.voice.channel
                if isinstance(candidate, discord.VoiceChannel):
                    selected_voice = candidate
            if selected_voice is None:
                raise UserError("ボイスチャンネルを選ぶか、先に参加してください。")

            response = cast(
                ReadAloudResponse,
                await self.runtime.registry.invoke(
                    "discord.manage_read_aloud",
                    ReadAloudRequest(
                        action=ReadAloudAction.CONFIGURE,
                        text_channel_id=str(selected_text.id),
                        audio_destination_id=str(selected_voice.id),
                        mode=ReadAloudMode(mode),
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "読み上げを設定しました",
                    fields=(
                        EmbedField("読み上げるチャンネル", selected_text.mention),
                        EmbedField("音声を流すVC", selected_voice.mention),
                        EmbedField("動作", _read_aloud_mode_label(response.mode)),
                        EmbedField("音声", _speech_voice_label(self.runtime)),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(name="status", description="現在の読み上げ設定を表示します。")
    async def status(self, interaction: discord.Interaction) -> None:
        try:
            context = invocation_context(interaction)
            route_result, policy_result = await asyncio.gather(
                self.runtime.registry.invoke(
                    "discord.read_aloud_status",
                    ReadAloudStatusRequest(),
                    context,
                ),
                self.runtime.registry.invoke(
                    "discord.read_aloud_policy_status",
                    ReadAloudStatusRequest(),
                    context,
                ),
            )
            response = cast(ReadAloudResponse, route_result)
            policy = cast(ReadAloudPolicyResponse, policy_result)
            route_fields: tuple[EmbedField, ...] = ()
            if response.enabled:
                route_fields = (
                    EmbedField(
                        "読み上げるチャンネル",
                        "\n".join(f"<#{channel_id}>" for channel_id in response.text_channel_ids),
                    ),
                    EmbedField(
                        "音声を流すVC",
                        f"<#{response.audio_destination_id}>",
                    ),
                    EmbedField("動作", _read_aloud_mode_label(response.mode)),
                )
            await interaction.response.send_message(
                embed=command_embed(
                    "読み上げ設定",
                    description=(
                        None
                        if response.enabled
                        else "現在、読み上げ経路は無効です。辞書などの設定は保持されています。"
                    ),
                    fields=(
                        *route_fields,
                        EmbedField(
                            "対象",
                            {
                                "all": "メッセージと入退室",
                                "messages": "メッセージ",
                                "events": "入退室",
                                "off": "オフ",
                            }.get(policy.content_mode, policy.content_mode),
                        ),
                        EmbedField(
                            "入退室通知",
                            (
                                f"参加 {_on_off(policy.announce_join)} · "
                                f"退出 {_on_off(policy.announce_leave)} · "
                                f"移動 {_on_off(policy.announce_move)}"
                            ),
                        ),
                        EmbedField(
                            "メッセージの読み方",
                            (
                                f"投稿者名 {_on_off(policy.read_author_names)} · "
                                f"返信先 {_on_off(policy.read_replies)} · "
                                f"添付 {_on_off(policy.read_attachments)}"
                            ),
                        ),
                        EmbedField(
                            "辞書と除外",
                            (
                                f"辞書 {len(policy.dictionary)}件 · "
                                f"ユーザー除外 {len(policy.ignored_user_ids)}人 · "
                                f"ロール除外 {len(policy.ignored_role_ids)}件"
                            ),
                        ),
                        EmbedField("音声", _speech_voice_label(self.runtime)),
                    ),
                    tone=(EmbedTone.SUCCESS if response.enabled else EmbedTone.WARNING),
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="mode",
        description="メッセージ・入退室・両方・オフを一度に選びます。",
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(mode="読み上げる内容")
    async def content_mode(
        self,
        interaction: discord.Interaction,
        mode: Literal["all", "messages", "events", "off"],
    ) -> None:
        try:
            policy = cast(
                ReadAloudPolicyResponse,
                await self.runtime.registry.invoke(
                    "discord.read_aloud_content_mode_set",
                    ReadAloudContentModeSetRequest(mode=ReadAloudContentMode(mode)),
                    invocation_context(interaction),
                ),
            )
            labels = {
                "all": "メッセージと入退室",
                "messages": "メッセージのみ",
                "events": "入退室のみ",
                "off": "オフ",
            }
            await interaction.response.send_message(
                embed=command_embed(
                    "読み上げ対象を変更しました",
                    fields=(EmbedField("対象", labels[policy.content_mode]),),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="dictionary",
        description="このサーバーの読み上げ辞書を表示します。",
    )
    async def dictionary(self, interaction: discord.Interaction) -> None:
        try:
            policy = cast(
                ReadAloudPolicyResponse,
                await self.runtime.registry.invoke(
                    "discord.read_aloud_dictionary_list",
                    ReadAloudDictionaryListRequest(),
                    invocation_context(interaction),
                ),
            )
            entries = policy.dictionary[:20]
            await interaction.response.send_message(
                embed=command_embed(
                    "読み上げ辞書",
                    description=(
                        "\n".join(f"`{item.surface}` → {item.reading}" for item in entries)
                        if entries
                        else "登録されている単語はありません。"
                    ),
                    fields=(
                        EmbedField(
                            "登録数",
                            (
                                f"{len(policy.dictionary)}件"
                                + ("、先頭20件を表示" if len(policy.dictionary) > 20 else "")
                            ),
                        ),
                    ),
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="dictionary-add",
        description="単語の読み方をこのサーバーの辞書へ登録します。",
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(
        word="メッセージに現れる表記",
        reading="VOICEVOXに読ませる読み方",
    )
    async def dictionary_add(
        self,
        interaction: discord.Interaction,
        word: str,
        reading: str,
    ) -> None:
        try:
            policy = cast(
                ReadAloudPolicyResponse,
                await self.runtime.registry.invoke(
                    "discord.read_aloud_dictionary_set",
                    ReadAloudDictionarySetRequest(
                        surface=word,
                        reading=reading,
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "読み方を登録しました",
                    fields=(
                        EmbedField("表記", word.strip()),
                        EmbedField("読み", reading.strip()),
                        EmbedField("辞書", f"{len(policy.dictionary)}件"),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="dictionary-remove",
        description="登録済みの表記を読み上げ辞書から削除します。",
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(word="辞書から削除する表記")
    async def dictionary_remove(
        self,
        interaction: discord.Interaction,
        word: str,
    ) -> None:
        try:
            policy = cast(
                ReadAloudPolicyResponse,
                await self.runtime.registry.invoke(
                    "discord.read_aloud_dictionary_remove",
                    ReadAloudDictionaryRemoveRequest(surface=word),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "読み方の登録を解除しました",
                    fields=(
                        EmbedField("表記", word.strip()),
                        EmbedField("辞書", f"{len(policy.dictionary)}件"),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="mute",
        description="自分のメッセージを読み上げるか選びます。",
    )
    @app_commands.describe(ignored="オンにすると、自分のメッセージを読み上げません")
    async def mute(
        self,
        interaction: discord.Interaction,
        ignored: bool,
    ) -> None:
        try:
            await self.runtime.registry.invoke(
                "discord.read_aloud_exclusion_set",
                ReadAloudExclusionSetRequest(
                    target=ReadAloudExclusionTarget.USER,
                    target_id=str(interaction.user.id),
                    ignored=ignored,
                ),
                invocation_context(interaction),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "自分の読み上げ設定を変更しました",
                    description=(
                        "あなたのメッセージは読み上げません。"
                        if ignored
                        else "あなたのメッセージを読み上げます。"
                    ),
                    tone=EmbedTone.SUCCESS,
                ),
                ephemeral=True,
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="ignore-user",
        description="指定したユーザーを読み上げ対象から外すか選びます。",
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(
        user="読み上げ設定を変更するユーザー",
        ignored="オンにすると読み上げません",
    )
    async def ignore_user(
        self,
        interaction: discord.Interaction,
        user: discord.Member,
        ignored: bool = True,
    ) -> None:
        try:
            await self.runtime.registry.invoke(
                "discord.read_aloud_exclusion_set",
                ReadAloudExclusionSetRequest(
                    target=ReadAloudExclusionTarget.USER,
                    target_id=str(user.id),
                    ignored=ignored,
                ),
                invocation_context(interaction),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "ユーザーの読み上げ設定を変更しました",
                    fields=(
                        EmbedField("ユーザー", user.mention),
                        EmbedField("読み上げ", "しない" if ignored else "する"),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="ignore-role",
        description="指定したロールを読み上げ対象から外すか選びます。",
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(
        role="読み上げ設定を変更するロール",
        ignored="オンにすると、このロールのメンバーを読み上げません",
    )
    async def ignore_role(
        self,
        interaction: discord.Interaction,
        role: discord.Role,
        ignored: bool = True,
    ) -> None:
        try:
            await self.runtime.registry.invoke(
                "discord.read_aloud_exclusion_set",
                ReadAloudExclusionSetRequest(
                    target=ReadAloudExclusionTarget.ROLE,
                    target_id=str(role.id),
                    ignored=ignored,
                ),
                invocation_context(interaction),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "ロールの読み上げ設定を変更しました",
                    fields=(
                        EmbedField("ロール", role.mention),
                        EmbedField("読み上げ", "しない" if ignored else "する"),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="announcements",
        description="VCへの参加・退出・移動を読み上げるか選びます。",
    )
    @app_commands.default_permissions(manage_guild=True)
    async def announcements(
        self,
        interaction: discord.Interaction,
        join: bool | None = None,
        leave: bool | None = None,
        move: bool | None = None,
    ) -> None:
        try:
            policy = cast(
                ReadAloudPolicyResponse,
                await self.runtime.registry.invoke(
                    "discord.read_aloud_announcements_set",
                    ReadAloudAnnouncementsSetRequest(
                        join=join,
                        leave=leave,
                        move=move,
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "入退室通知を変更しました",
                    fields=(
                        EmbedField("参加", _on_off(policy.announce_join)),
                        EmbedField("退出", _on_off(policy.announce_leave)),
                        EmbedField("移動", _on_off(policy.announce_move)),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="message-style",
        description="投稿者名・返信先・添付を読み上げるか選びます。",
    )
    @app_commands.default_permissions(manage_guild=True)
    async def message_style(
        self,
        interaction: discord.Interaction,
        author_names: bool | None = None,
        replies: bool | None = None,
        attachments: bool | None = None,
    ) -> None:
        try:
            policy = cast(
                ReadAloudPolicyResponse,
                await self.runtime.registry.invoke(
                    "discord.read_aloud_semantics_set",
                    ReadAloudSemanticsSetRequest(
                        author_names=author_names,
                        replies=replies,
                        attachments=attachments,
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "メッセージの読み方を変更しました",
                    fields=(
                        EmbedField("投稿者名", _on_off(policy.read_author_names)),
                        EmbedField("返信先", _on_off(policy.read_replies)),
                        EmbedField("添付", _on_off(policy.read_attachments)),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="remove",
        description="指定した会話チャンネルの読み上げを解除します。",
    )
    @app_commands.describe(
        channel="解除する会話チャンネル。省略時は現在のチャンネル",
    )
    async def remove(
        self,
        interaction: discord.Interaction,
        channel: discord.TextChannel | discord.VoiceChannel | None = None,
    ) -> None:
        try:
            selected = channel
            if selected is None and isinstance(
                interaction.channel,
                (discord.TextChannel, discord.VoiceChannel),
            ):
                selected = interaction.channel
            if selected is None:
                raise UserError("解除する会話チャンネルを選んでください。")
            response = cast(
                ReadAloudResponse,
                await self.runtime.registry.invoke(
                    "discord.manage_read_aloud",
                    ReadAloudRequest(
                        action=ReadAloudAction.REMOVE_SOURCE,
                        text_channel_id=str(selected.id),
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "読み上げ対象から外しました",
                    description=(
                        "読み上げるチャンネルがなくなったため、読み上げを無効にしました。"
                        if not response.enabled
                        else "ほかに設定されているチャンネルの読み上げは続きます。"
                    ),
                    fields=(
                        EmbedField("解除したチャンネル", selected.mention),
                        EmbedField(
                            "残り",
                            f"{len(response.text_channel_ids)}チャンネル",
                        ),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(name="disable", description="このサーバーの自動読み上げを停止します。")
    @app_commands.default_permissions(manage_guild=True)
    async def disable(self, interaction: discord.Interaction) -> None:
        try:
            member = interaction.user
            if not isinstance(member, discord.Member) or not member.guild_permissions.manage_guild:
                raise UserError("この設定には「サーバー管理」権限が必要です。")
            await self.runtime.registry.invoke(
                "discord.manage_read_aloud",
                ReadAloudRequest(action=ReadAloudAction.DISABLE),
                invocation_context(interaction),
            )
            if interaction.guild_id is not None:
                self._message_formatter.forget_workspace(str(interaction.guild_id))
            await interaction.response.send_message(
                embed=command_embed(
                    "読み上げを停止しました",
                    description="このサーバーの自動読み上げ設定を無効にしました。",
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None:
            return
        workspace_id = str(message.guild.id)
        if not self.runtime.read_aloud.matches(workspace_id, str(message.channel.id)):
            return
        route = self.runtime.read_aloud.get(workspace_id)
        if route is None:
            return
        role_ids = (
            tuple(str(role.id) for role in message.author.roles)
            if isinstance(message.author, discord.Member)
            else ()
        )
        if not self.runtime.read_aloud.allows_message(
            workspace_id=workspace_id,
            author_id=str(message.author.id),
            role_ids=role_ids,
            is_bot=message.author.bot,
            is_webhook=message.webhook_id is not None,
        ):
            return
        prepared = await self._message_formatter.format(message)
        if prepared is None:
            return
        guild_id = message.guild.id
        session = self.runtime.audio.get_or_create(
            workspace_id,
            lambda: DiscordAudioOutput(self.bot, guild_id),
        )
        if session.resume_confirmation_required and not session.output.connected:
            return
        if (
            route.mode is ReadAloudMode.SKIP_DURING_MUSIC
            and session.current is not None
            and session.current.kind is AudioKind.MUSIC
        ):
            return
        output = cast(DiscordAudioOutput, session.output)
        try:
            destination = message.guild.get_channel(int(route.audio_destination_id))
            if not isinstance(
                destination,
                (discord.VoiceChannel, discord.StageChannel),
            ) or not any(not listener.bot for listener in destination.members):
                return
            if (
                output.connected
                and output.destination_id != int(route.audio_destination_id)
                and session.current is not None
            ):
                return
            await self.runtime.registry.invoke(
                "discord.connect_voice",
                DiscordConnectVoiceRequest(channel_id=route.audio_destination_id),
                InvocationContext(
                    actor_id=str(message.author.id),
                    workspace_id=workspace_id,
                    transport="discord",
                    request_id=f"read-aloud:{message.id}",
                ),
            )
            await self.runtime.registry.invoke(
                "speech.speak",
                SpeechSpeakRequest(
                    title=prepared.title,
                    segments=prepared.segments,
                ),
                InvocationContext(
                    actor_id=str(message.author.id),
                    workspace_id=workspace_id,
                    transport="discord",
                    request_id=f"read-aloud:{message.id}",
                ),
            )
        except Exception:
            log.exception(
                "Automatic read-aloud failed guild=%s channel=%s",
                message.guild.id,
                message.channel.id,
            )

    @commands.Cog.listener()
    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Debounce voice transitions before adding an announcement."""

        if member.bot or before.channel == after.channel:
            return
        workspace_id = str(member.guild.id)
        route = self.runtime.read_aloud.get(workspace_id)
        if route is None:
            return
        destination_id = int(route.audio_destination_id)
        before_relevant = before.channel is not None and before.channel.id == destination_id
        after_relevant = after.channel is not None and after.channel.id == destination_id
        if not before_relevant and not after_relevant:
            return

        key = (member.guild.id, member.id)
        previous = self._voice_transitions.get(key)
        initial_channel = previous[1] if previous is not None else before.channel
        self._voice_transitions[key] = (member, initial_channel, after.channel)
        existing = self._announcement_tasks.get(key)
        if existing is not None:
            existing.cancel()
        self._announcement_tasks[key] = asyncio.create_task(
            self._flush_voice_transition(key),
            name=f"simajilord-read-aloud-voice-{member.guild.id}-{member.id}",
        )

    async def _flush_voice_transition(self, key: tuple[int, int]) -> None:
        try:
            await asyncio.sleep(0.7)
            transition = self._voice_transitions.get(key)
            if transition is None:
                return
            member, before_channel, after_channel = transition
            await self._announce_voice_transition(
                member,
                before_channel=before_channel,
                after_channel=after_channel,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "Voice read-aloud announcement failed guild=%s member=%s",
                key[0],
                key[1],
            )
        finally:
            if self._announcement_tasks.get(key) is asyncio.current_task():
                self._announcement_tasks.pop(key, None)
                self._voice_transitions.pop(key, None)

    async def _announce_voice_transition(
        self,
        member: discord.Member,
        *,
        before_channel: discord.VoiceChannel | discord.StageChannel | None,
        after_channel: discord.VoiceChannel | discord.StageChannel | None,
    ) -> None:
        workspace_id = str(member.guild.id)
        route = self.runtime.read_aloud.get(workspace_id)
        if route is None or before_channel == after_channel:
            return
        policy = self.runtime.read_aloud.policy(workspace_id)
        destination_id = int(route.audio_destination_id)
        before_id = before_channel.id if before_channel is not None else None
        after_id = after_channel.id if after_channel is not None else None
        name = member.display_name.strip() or member.name

        if before_channel is not None and after_channel is not None:
            if not policy.announce_move or destination_id not in (before_id, after_id):
                return
            text = f"{name}さんが、{before_channel.name}から{after_channel.name}へ移動しました"
        elif after_id == destination_id:
            if not policy.announce_join:
                return
            text = f"{name}さんがボイスチャンネルに参加しました"
        elif before_id == destination_id:
            if not policy.announce_leave:
                return
            text = f"{name}さんがボイスチャンネルから退出しました"
        else:
            return

        destination = member.guild.get_channel(destination_id)
        if not isinstance(destination, (discord.VoiceChannel, discord.StageChannel)):
            return
        if not any(not listener.bot for listener in destination.members):
            return

        session = self.runtime.audio.get_or_create(
            workspace_id,
            lambda: DiscordAudioOutput(self.bot, member.guild.id),
        )
        if session.resume_confirmation_required and not session.output.connected:
            return
        if (
            session.has_music
            and session.destination_id is not None
            and session.destination_id != route.audio_destination_id
        ):
            return
        if not session.output.connected:
            await self.runtime.audio.connect(workspace_id, route.audio_destination_id)
        spoken_text = self.runtime.read_aloud.apply_dictionary(workspace_id, text)
        await self.runtime.registry.invoke(
            "speech.speak",
            SpeechSpeakRequest(
                text=spoken_text,
                title="VCの入退室通知",
                segments=(
                    SpeechSegment(
                        SpeechSegmentKind.EVENT,
                        spoken_text,
                    ),
                ),
            ),
            InvocationContext(
                actor_id=str(member.id),
                workspace_id=workspace_id,
                transport="discord",
                request_id=f"read-aloud-voice:{member.id}:{time.time_ns()}",
            ),
        )


class VoiceLifecycleCog(commands.Cog):
    """Keep voice presence aligned with listeners without losing the music queue."""

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime
        self._leave_tasks: dict[str, asyncio.Task[None]] = {}
        dashboard = getattr(bot, _MUSIC_DASHBOARD_ATTRIBUTE, None)
        self.dashboard = dashboard if isinstance(dashboard, MusicDashboardManager) else None

    async def cog_unload(self) -> None:
        for task in self._leave_tasks.values():
            task.cancel()
        self._leave_tasks.clear()

    @commands.Cog.listener()
    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        if member.bot:
            return
        workspace_id = str(member.guild.id)
        route = self.runtime.read_aloud.get(workspace_id)
        joined_read_aloud_destination = (
            route is not None
            and after.channel is not None
            and str(after.channel.id) == route.audio_destination_id
        )
        session = self.runtime.audio.find(workspace_id)
        if session is None and joined_read_aloud_destination:
            session = self.runtime.audio.get_or_create(
                workspace_id,
                lambda: DiscordAudioOutput(self.bot, member.guild.id),
            )
        if session is None:
            return

        destination_id = int(session.destination_id) if session.destination_id is not None else None
        joined_expected_channel = after.channel is not None and (
            (session.waiting_for_voice and session.can_start_for(str(member.id)))
            or after.channel.id == destination_id
            or joined_read_aloud_destination
        )
        if joined_expected_channel and after.channel is not None:
            if self.dashboard is not None and (
                session.has_music
                or session.waiting_for_voice
                or session.resume_confirmation_required
            ):
                self.dashboard.bind(member.guild.id, after.channel.id)
            task = self._leave_tasks.pop(workspace_id, None)
            if task is not None:
                task.cancel()
            music_targets_another_channel = (
                session.has_music
                and session.destination_id is not None
                and str(after.channel.id) != session.destination_id
            )
            should_connect = (
                session.has_music or joined_read_aloud_destination
            ) and not music_targets_another_channel and not session.resume_confirmation_required
            if should_connect and not session.output.connected:
                try:
                    await self.runtime.audio.connect(
                        workspace_id,
                        str(after.channel.id),
                    )
                    log.info(
                        "Prepared audio after a listener joined guild=%s read_aloud=%s music=%s",
                        workspace_id,
                        joined_read_aloud_destination,
                        session.has_music,
                    )
                except Exception:
                    log.exception(
                        "Could not prepare audio after a listener joined guild=%s",
                        workspace_id,
                    )
            return

        if destination_id is None:
            return
        if before.channel is None or before.channel.id != destination_id:
            return
        existing = self._leave_tasks.pop(workspace_id, None)
        if existing is not None:
            existing.cancel()

        async def leave_if_lonely() -> None:
            try:
                await asyncio.sleep(10)
                output = session.output
                if not output.connected or not session.auto_leave:
                    return
                guild = member.guild
                channel = guild.get_channel(destination_id)
                if not isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
                    await session.suspend()
                    return
                if any(not listener.bot for listener in channel.members):
                    return
                await session.suspend()
                log.info(
                    "Auto-left empty voice channel while preserving queue guild=%s channel=%s",
                    workspace_id,
                    destination_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Voice auto-leave failed guild=%s", workspace_id)
            finally:
                if self._leave_tasks.get(workspace_id) is asyncio.current_task():
                    self._leave_tasks.pop(workspace_id, None)

        self._leave_tasks[workspace_id] = asyncio.create_task(
            leave_if_lonely(),
            name=f"simajilord-auto-leave-{workspace_id}",
        )


class WebCog(commands.Cog):
    """Discord presentation for the same web APIs available to the agent."""

    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    @app_commands.command(
        name="search",
        description="Simajilordのローカル検索サービスでWebを検索します。",
    )
    @app_commands.describe(
        query="調べたい話題、質問、または完全一致で探す語句",
        depth="検索の深さ: quick / standard / deep",
    )
    async def search(
        self,
        interaction: discord.Interaction,
        query: str,
        depth: Literal["quick", "standard", "deep"] = "standard",
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            response = cast(
                WebSearchResponse,
                await self.runtime.registry.invoke(
                    "web.search",
                    WebSearchRequest(
                        query=query,
                        depth=SearchDepth(depth),
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.edit_original_response(embed=web_search_embed(response))
        except Exception as exc:
            await edit_deferred_error(interaction, exc)

    @app_commands.command(
        name="fetch",
        description="公開Webページを開き、読み取れる本文を表示します。",
    )
    @app_commands.describe(
        url="開く公開HTTPまたはHTTPS URL",
        offset="本文を途中から読む場合の開始位置",
    )
    async def fetch(
        self,
        interaction: discord.Interaction,
        url: str,
        offset: app_commands.Range[int, 0, 40_000] = 0,
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            response = cast(
                WebFetchResponse,
                await self.runtime.registry.invoke(
                    "web.fetch",
                    WebFetchRequest(
                        url=url,
                        offset=offset,
                        max_characters=3_500,
                    ),
                    invocation_context(interaction),
                ),
            )
            view = (
                WebFetchContinueView(self.runtime, response)
                if response.next_offset is not None
                else None
            )
            await interaction.edit_original_response(
                embed=web_fetch_embed(response),
                view=view,
            )
        except Exception as exc:
            await edit_deferred_error(interaction, exc)

    @app_commands.command(
        name="find",
        description="公開Webページの本文から語句を探します。",
    )
    @app_commands.describe(
        url="調べる公開HTTPまたはHTTPS URL",
        phrase="大文字と小文字を区別せずに探す語句",
    )
    async def find(
        self,
        interaction: discord.Interaction,
        url: str,
        phrase: str,
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            response = cast(
                WebFindResponse,
                await self.runtime.registry.invoke(
                    "web.find",
                    WebFindRequest(url=url, pattern=phrase),
                    invocation_context(interaction),
                ),
            )
            await interaction.edit_original_response(embed=web_find_embed(response))
        except Exception as exc:
            await edit_deferred_error(interaction, exc)


def _percentage(score: float) -> str:
    return f"{score * 100:.1f}%"


_HIVE_SOURCE_LABELS = {
    "4o": "GPT-4o",
    "adobefirefly": "Adobe Firefly",
    "amused": "Amused",
    "aniportrait": "AniPortrait",
    "bagel": "Bagel",
    "bingimagecreator": "Bing Image Creator",
    "blip3o": "BLIP-3o",
    "bria": "Bria",
    "cogvideos": "CogVideo",
    "cogview": "CogView",
    "cosmos": "Cosmos",
    "dalle": "DALL-E",
    "deepfloyd": "DeepFloyd",
    "dmd2": "DMD2",
    "dreamid": "DreamID",
    "emu3": "Emu3",
    "flashvideo": "FlashVideo",
    "flux": "FLUX",
    "flux2": "FLUX.2",
    "gan": "GAN",
    "gemini": "Gemini",
    "gemini3": "Gemini 3",
    "glide": "GLIDE",
    "gptimage1_5": "GPT Image 1.5",
    "grok": "Grok",
    "grokimagine": "Grok Imagine",
    "haiper": "Haiper",
    "hailuo": "Hailuo",
    "hallo": "Hallo",
    "happyhorse": "HappyHorse",
    "hedra": "Hedra",
    "heygen": "HeyGen",
    "hidream": "HiDream",
    "higgsfield": "Higgsfield",
    "hunyuan": "Hunyuan",
    "ideogram": "Ideogram",
    "imagen": "Imagen",
    "imagen4": "Imagen 4",
    "imagineart": "ImagineArt",
    "infinity": "Infinity",
    "janus": "Janus",
    "kandinsky": "Kandinsky",
    "kling": "Kling",
    "krea": "Krea",
    "lcm": "LCM",
    "leonardo": "Leonardo",
    "liveportrait": "LivePortrait",
    "longcat": "LongCat",
    "ltx": "LTX",
    "lucid": "Lucid",
    "luma": "Luma",
    "luminagpt": "Lumina GPT",
    "mai": "MAI",
    "makeittalk": "MakeItTalk",
    "mcnet": "MCNet",
    "meta": "Meta",
    "midjourney": "Midjourney",
    "mochi": "Mochi",
    "moonvalley": "Moonvalley",
    "omnigen": "OmniGen",
    "other_image_generators": "Other image generator",
    "ovis": "Ovis",
    "personalive": "PersonaLive",
    "pika": "Pika",
    "pixart": "PixArt",
    "pixverse": "PixVerse",
    "pyramidflows": "Pyramid Flow",
    "qwen": "Qwen",
    "ray3": "Ray 3",
    "recraft": "Recraft",
    "reve": "Reve",
    "runway": "Runway",
    "sadtalker": "SadTalker",
    "sana": "Sana",
    "sanavideo": "Sana Video",
    "scail": "SCAIL",
    "sdxlinpaint": "SDXL Inpainting",
    "seedance": "Seedance",
    "seedance2": "Seedance 2",
    "seedream": "Seedream",
    "sora": "Sora",
    "sora2": "Sora 2",
    "stablecascade": "Stable Cascade",
    "stablediffusion": "Stable Diffusion",
    "stablediffusioninpaint": "Stable Diffusion Inpainting",
    "stablediffusionxl": "Stable Diffusion XL",
    "steadydancer": "SteadyDancer",
    "switti": "Switti",
    "titan": "Titan",
    "transpixar": "TransPixar",
    "var": "VAR",
    "veo3": "Veo 3",
    "vibe": "Vibe",
    "viduq2": "Vidu Q2",
    "vqdiffusion": "VQ Diffusion",
    "wan": "Wan",
    "wuerstchen": "Würstchen",
    "zimage": "Z-Image",
}


def _source_name(value: str) -> str:
    known = _HIVE_SOURCE_LABELS.get(value)
    if known is not None:
        return known
    return " ".join(part.capitalize() for part in value.replace("_", " ").split())


def _likelihood(score: float, *, high_threshold: float) -> str:
    if score >= high_threshold:
        return "高"
    if score >= 0.5:
        return "中"
    return "低"


def synthetic_media_embed(
    response: SyntheticMediaAnalyzeResponse,
    *,
    attachment_url: str | None = None,
) -> discord.Embed:
    ai_likelihood = _likelihood(
        response.ai_generated_score,
        high_threshold=response.threshold,
    )
    deepfake_likelihood = (
        "高"
        if response.deepfake_likely
        else _likelihood(
            response.deepfake_score,
            high_threshold=response.threshold,
        )
    )
    severity_labels = {ai_likelihood, deepfake_likelihood}
    if "高" in severity_labels:
        tone = EmbedTone.ERROR
    elif "中" in severity_labels:
        tone = EmbedTone.WARNING
    else:
        tone = EmbedTone.SUCCESS
    media_label = {"image": "画像", "video": "動画"}[response.modality.value]
    conclusion_lines = [f"**AI生成{media_label}の可能性: {ai_likelihood}**"]
    # Discordでは小数第1位まで表示するため、表示上0.0%になる値を
    # 結論文で「ディープフェイクの可能性あり」と強調しない。
    if response.deepfake_score * 100 >= 0.05:
        conclusion_lines.append(f"ディープフェイクの可能性: {deepfake_likelihood}")
    conclusion = "\n".join(conclusion_lines)
    fields = [
        EmbedField(
            "AI生成",
            f"**{_percentage(response.ai_generated_score)}**・可能性 {ai_likelihood}",
        ),
        EmbedField(
            "ディープフェイク",
            f"**{_percentage(response.deepfake_score)}**・可能性 {deepfake_likelihood}",
        ),
    ]
    if response.top_source is not None and response.top_source_score >= 0.5:
        fields.append(
            EmbedField(
                "推定生成モデル",
                (
                    f"**{_source_name(response.top_source)}** · "
                    f"{_percentage(response.top_source_score)}"
                ),
                inline=False,
            )
        )
    cache_line = (
        "保存済みの解析結果を再利用・HIVE APIの追加消費なし"
        if response.cached
        else "HIVE APIで新しく解析"
    )
    sample_name = {
        "image": "枚の画像",
        "video": "フレーム",
    }[response.modality.value]
    fields.append(
        EmbedField(
            "解析",
            f"{response.sample_count}{sample_name}・{cache_line}",
            inline=False,
        )
    )
    embed = command_embed(
        "HIVE AIコンテンツ解析",
        description=conclusion,
        fields=tuple(fields),
        tone=tone,
    )
    if attachment_url is not None and response.content_type.startswith("image/"):
        embed.set_thumbnail(url=attachment_url)
    model_label = response.model.removeprefix("hive/")
    embed.set_footer(text=f"HIVE Moderationによる解析・{model_label}")
    return embed


class ModerationCog(commands.Cog):
    """Discord upload adapter for the shared synthetic-media capability."""

    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    @app_commands.command(
        name="detectai",
        description="画像・動画がAI生成かどうかをHIVEで解析します。",
    )
    @app_commands.describe(media="HIVEで解析する画像または動画")
    async def detectai(
        self,
        interaction: discord.Interaction,
        media: discord.Attachment,
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            if media.size > self.runtime.settings.hive_max_media_bytes:
                raise UserError("moderation.media_too_large")
            content = await media.read(use_cached=True)
            response = cast(
                SyntheticMediaAnalyzeResponse,
                await self.runtime.registry.invoke(
                    "moderation.detect_synthetic_media",
                    SyntheticMediaAnalyzeRequest(
                        filename=media.filename,
                        content_type=media.content_type,
                        content=content,
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.edit_original_response(
                embed=synthetic_media_embed(
                    response,
                    attachment_url=media.url,
                )
            )
        except Exception as exc:
            await edit_deferred_error(interaction, exc)


class DownloadCog(commands.Cog):
    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime
        self._last_request: dict[int, float] = {}

    @app_commands.command(
        name="download",
        description="対応サイトの公開URLから動画または音声を保存します。",
    )
    async def download(
        self,
        interaction: discord.Interaction,
        url: str,
        media_type: Literal["video", "audio"] = "video",
    ) -> None:
        temporary: Path | None = None
        try:
            now = time.monotonic()
            previous = self._last_request.get(interaction.user.id, 0.0)
            if now - previous < 30:
                raise UserError("次のダウンロードまで30秒お待ちください。")
            self._last_request[interaction.user.id] = now
            await interaction.response.defer(thinking=True)
            download_root = self.runtime.settings.data_dir / "downloads"
            download_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix="request-", dir=download_root))
            guild_limit = interaction.guild.filesize_limit if interaction.guild else 10_000_000
            max_bytes = max(1_000_000, guild_limit - 1_000_000)
            response = cast(
                DownloadResponse,
                await self.runtime.registry.invoke(
                    "media.download",
                    DownloadRequest(
                        url=url,
                        media_type=DownloadFormat(media_type),
                        destination=temporary,
                        max_bytes=max_bytes,
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.followup.send(
                embed=command_embed(
                    "メディアを保存しました",
                    description=(
                        f"### {discord.utils.escape_markdown(response.title)}\n"
                        f"[配信元を開く]({response.source_url})"
                    ),
                    fields=(
                        EmbedField(
                            "サイズ",
                            f"{response.size_bytes / 1_000_000:.1f} MB",
                        ),
                        EmbedField(
                            "種類",
                            "動画" if media_type == "video" else "音声",
                        ),
                        EmbedField("形式", response.path.suffix.lstrip(".").upper()),
                    ),
                    tone=EmbedTone.SUCCESS,
                ),
                file=discord.File(response.path),
            )
        except Exception as exc:
            await send_error(interaction, exc)
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)


class UtilityCog(commands.Cog):
    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    @app_commands.command(name="roll", description="指定した個数・面数のサイコロを振ります。")
    async def roll(
        self,
        interaction: discord.Interaction,
        dice: app_commands.Range[int, 1, 20] = 1,
        sides: app_commands.Range[int, 2, 1_000] = 6,
    ) -> None:
        try:
            response = cast(
                RollResponse,
                await self.runtime.registry.invoke(
                    "utility.roll",
                    RollRequest(dice=dice, sides=sides),
                    invocation_context(interaction),
                ),
            )
            values = ", ".join(str(value) for value in response.rolls)
            await interaction.response.send_message(
                embed=command_embed(
                    "サイコロ",
                    fields=(
                        EmbedField("出目", values, inline=False),
                        EmbedField("合計", str(response.total)),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @app_commands.command(name="choose", description="カンマ区切りの候補から1つ選びます。")
    async def choose(self, interaction: discord.Interaction, options: str) -> None:
        try:
            parsed = tuple(item.strip() for item in options.split(","))
            response = cast(
                ChooseResponse,
                await self.runtime.registry.invoke(
                    "utility.choose",
                    ChooseRequest(options=parsed),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "選択結果",
                    description=response.choice,
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)


class DiscordInfoCog(commands.Cog):
    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    @app_commands.command(name="serverinfo", description="このサーバーの公開情報を表示します。")
    async def serverinfo(self, interaction: discord.Interaction) -> None:
        try:
            response = cast(
                DiscordServerResponse,
                await self.runtime.registry.invoke(
                    "discord.inspect_server",
                    DiscordServerRequest(),
                    invocation_context(interaction),
                ),
            )
            embed = discord.Embed(
                title=response.name,
                description=f"サーバーID: `{response.server_id}`",
                colour=discord.Colour.blurple(),
                timestamp=discord.utils.utcnow(),
            )
            embed.add_field(name="メンバー", value=response.member_count or "不明")
            embed.add_field(name="テキストチャンネル", value=response.text_channel_count)
            embed.add_field(name="ボイスチャンネル", value=response.voice_channel_count)
            embed.add_field(name="ロール", value=response.role_count)
            embed.add_field(
                name="作成日時",
                value=f"<t:{int(discord.utils.parse_time(response.created_at_iso).timestamp())}:F>",
            )
            if response.icon_url:
                embed.set_thumbnail(url=response.icon_url)
            await interaction.response.send_message(embed=embed)
        except Exception as exc:
            await send_error(interaction, exc)

    @app_commands.command(name="userinfo", description="Discordユーザーの公開情報を表示します。")
    async def userinfo(
        self,
        interaction: discord.Interaction,
        user: discord.User | None = None,
    ) -> None:
        try:
            target = user or interaction.user
            response = cast(
                DiscordUserResponse,
                await self.runtime.registry.invoke(
                    "discord.inspect_user",
                    DiscordUserRequest(user_id=str(target.id)),
                    invocation_context(interaction),
                ),
            )
            embed = discord.Embed(
                title=response.display_name,
                description=f"ユーザーID: `{response.user_id}`",
                colour=discord.Colour.blurple(),
                timestamp=discord.utils.utcnow(),
            )
            embed.set_thumbnail(url=response.avatar_url)
            embed.add_field(
                name="アカウント作成日時",
                value=f"<t:{int(discord.utils.parse_time(response.created_at_iso).timestamp())}:F>",
                inline=False,
            )
            if response.joined_at_iso:
                joined = discord.utils.parse_time(response.joined_at_iso)
                embed.add_field(
                    name="サーバー参加日時",
                    value=f"<t:{int(joined.timestamp())}:F>",
                    inline=False,
                )
            if response.top_role:
                embed.add_field(name="最高位ロール", value=response.top_role)
            embed.add_field(
                name="アカウント種別",
                value="BOT" if response.bot else "ユーザー",
            )
            await interaction.response.send_message(embed=embed)
        except Exception as exc:
            await send_error(interaction, exc)


class DiscordActionCog(commands.Cog):
    """Discord-native presentation actions."""

    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    @app_commands.command(name="avatar", description="Discordユーザーのアイコンを表示します。")
    async def avatar(
        self,
        interaction: discord.Interaction,
        user: discord.User | None = None,
    ) -> None:
        try:
            target = user or interaction.user
            response = cast(
                DiscordUserResponse,
                await self.runtime.registry.invoke(
                    "discord.inspect_user",
                    DiscordUserRequest(user_id=str(target.id)),
                    invocation_context(interaction),
                ),
            )
            embed = discord.Embed(
                title=response.display_name,
                colour=discord.Colour.blurple(),
                timestamp=discord.utils.utcnow(),
            )
            embed.set_image(url=response.avatar_url)
            await interaction.response.send_message(embed=embed)
        except Exception as exc:
            await send_error(interaction, exc)

    @app_commands.command(name="poll", description="Discordの投票を作成します。")
    async def poll(
        self,
        interaction: discord.Interaction,
        question: str,
        options: str,
        hours: app_commands.Range[int, 1, 168] = 24,
        multiple: bool = False,
    ) -> None:
        try:
            if interaction.channel_id is None:
                raise UserError("テキストチャンネルで実行してください。")
            response = cast(
                DiscordPollResponse,
                await self.runtime.registry.invoke(
                    "discord.create_poll",
                    DiscordPollRequest(
                        channel_id=str(interaction.channel_id),
                        question=question,
                        options=tuple(item.strip() for item in options.split(",")),
                        duration_hours=hours,
                        multiple=multiple,
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "投票を作成しました",
                    description=(
                        f"[投票を開く](https://discord.com/channels/{interaction.guild_id}/"
                        f"{response.channel_id}/{response.message_id})"
                    ),
                    tone=EmbedTone.SUCCESS,
                ),
                ephemeral=True,
            )
        except Exception as exc:
            await send_error(interaction, exc)


class MessageExpandCog(commands.Cog):
    """Expand one bare Discord message link with least-privilege checks."""

    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None or message.author.bot or message.webhook_id is not None:
            return
        link = parse_discord_message_link(message.content)
        if link is None:
            return
        try:
            cast(
                DiscordPostExpandedMessageResponse,
                await self.runtime.registry.invoke(
                    "discord.post_expanded_message",
                    DiscordPostExpandedMessageRequest(
                        source_guild_id=link.guild_id,
                        source_channel_id=link.channel_id,
                        source_message_id=link.message_id,
                        destination_channel_id=str(message.channel.id),
                    ),
                    message_context(message),
                ),
            )
            try:
                await message.delete()
            except discord.DiscordException:
                log.info(
                    "Expanded link retained because the source post could not be deleted "
                    "guild=%s channel=%s message=%s",
                    message.guild.id,
                    message.channel.id,
                    message.id,
                )
        except UserError as exc:
            await message.reply(
                embed=command_embed(
                    "メッセージを展開できません",
                    description=error_message(exc),
                    tone=EmbedTone.WARNING,
                ),
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
                silent=True,
            )
        except discord.DiscordException:
            log.exception(
                "Could not publish expanded Discord message guild=%s channel=%s "
                "message=%s source_message=%s",
                message.guild.id,
                message.channel.id,
                message.id,
                link.message_id,
            )


class QuoteCog(commands.Cog):
    """Create one local quote image from Discord's message context menu."""

    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    async def create_quote(
        self,
        interaction: discord.Interaction,
        message: discord.Message,
    ) -> None:
        if interaction.guild_id is None or interaction.channel_id is None:
            await send_error(
                interaction,
                UserError("この操作はサーバー内のメッセージで使用してください。"),
            )
            return
        view = QuoteComposerView(
            self.runtime,
            requester_id=interaction.user.id,
            source_channel_id=message.channel.id,
            source_message_id=message.id,
            destination_channel_id=interaction.channel_id,
            has_animation=quote_message_has_animation(message),
        )
        await interaction.response.send_message(
            embed=view.embed(),
            view=view,
            ephemeral=True,
        )


class QuoteComposerView(discord.ui.View):
    """Private, short-lived quote options without cluttering the result message."""

    def __init__(
        self,
        runtime: SimajilordRuntime,
        *,
        requester_id: int,
        source_channel_id: int,
        source_message_id: int,
        destination_channel_id: int,
        has_animation: bool = False,
    ) -> None:
        super().__init__(timeout=300)
        self.runtime = runtime
        self.requester_id = requester_id
        self.source_channel_id = source_channel_id
        self.source_message_id = source_message_id
        self.destination_channel_id = destination_channel_id
        self.color = False
        self.vertical = False
        self.bold = False
        self.flip = False
        self.has_animation = has_animation
        self.animate = False
        self.include_jump = True
        if not has_animation:
            self.remove_item(self.animation_button)
        self._sync_labels()

    def embed(self) -> discord.Embed:
        layout = "Vertical · 4:5" if self.vertical else "Landscape · 40:21"
        appearance = "Color" if self.color else "Black / White"
        enabled = [
            label
            for label, active in (
                ("Bold", self.bold),
                ("Flip", self.flip),
                ("Animation", self.animate),
                ("Jump", self.include_jump),
            )
            if active
        ]
        return discord.Embed(
            title="Quote",
            description="スタイルを選び、Generateで投稿します。",
            color=discord.Colour.green(),
        ).add_field(
            name="Preview settings",
            value=f"{layout}\n{appearance}\n{' · '.join(enabled) or 'Standard'}",
            inline=False,
        )

    def request(self) -> DiscordCreateQuoteImageRequest:
        return DiscordCreateQuoteImageRequest(
            source_channel_id=str(self.source_channel_id),
            source_message_id=str(self.source_message_id),
            destination_channel_id=str(self.destination_channel_id),
            color=self.color,
            vertical=self.vertical,
            bold=self.bold,
            flip=self.flip,
            animate=self.animate,
            include_jump=self.include_jump,
        )

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "このQuoteメニューは、開いた本人だけが操作できます。",
            ephemeral=True,
        )
        return False

    def _sync_labels(self) -> None:
        self.color_button.label = f"Color {'On' if self.color else 'Off'}"
        self.vertical_button.label = "Vertical" if self.vertical else "Landscape"
        self.bold_button.label = f"Bold {'On' if self.bold else 'Off'}"
        self.flip_button.label = f"Flip {'On' if self.flip else 'Off'}"
        self.jump_button.label = f"Jump {'On' if self.include_jump else 'Off'}"
        self.animation_button.label = f"Animation {'On' if self.animate else 'Off'}"
        self.color_button.style = (
            discord.ButtonStyle.primary if self.color else discord.ButtonStyle.secondary
        )
        self.vertical_button.style = (
            discord.ButtonStyle.primary if self.vertical else discord.ButtonStyle.secondary
        )
        self.bold_button.style = (
            discord.ButtonStyle.primary if self.bold else discord.ButtonStyle.secondary
        )
        self.flip_button.style = (
            discord.ButtonStyle.primary if self.flip else discord.ButtonStyle.secondary
        )
        self.jump_button.style = (
            discord.ButtonStyle.primary if self.include_jump else discord.ButtonStyle.secondary
        )
        self.animation_button.style = (
            discord.ButtonStyle.primary if self.animate else discord.ButtonStyle.secondary
        )

    async def _refresh_composer(self, interaction: discord.Interaction) -> None:
        self._sync_labels()
        await interaction.response.edit_message(embed=self.embed(), view=self)

    @discord.ui.button(label="Color Off", emoji="🎨", row=0)
    async def color_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        self.color = not self.color
        await self._refresh_composer(interaction)

    @discord.ui.button(label="Landscape", emoji="↔️", row=0)
    async def vertical_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        self.vertical = not self.vertical
        await self._refresh_composer(interaction)

    @discord.ui.button(label="Bold Off", emoji="🅱️", row=0)
    async def bold_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        self.bold = not self.bold
        await self._refresh_composer(interaction)

    @discord.ui.button(label="Flip Off", emoji="🔄", row=0)
    async def flip_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        self.flip = not self.flip
        await self._refresh_composer(interaction)

    @discord.ui.button(label="Jump On", emoji="↗️", row=0)
    async def jump_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        self.include_jump = not self.include_jump
        await self._refresh_composer(interaction)

    @discord.ui.button(label="Animation Off", emoji="🎞️", row=1)
    async def animation_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        self.animate = not self.animate
        await self._refresh_composer(interaction)

    @discord.ui.button(
        label="Generate",
        emoji="✨",
        style=discord.ButtonStyle.success,
        row=1,
    )
    async def generate_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        await interaction.response.defer()
        try:
            await self.runtime.registry.invoke(
                "discord.create_quote_image",
                self.request(),
                invocation_context(interaction),
            )
            await interaction.delete_original_response()
            self.stop()
        except Exception as exc:
            await interaction.edit_original_response(
                embed=command_embed(
                    "生成できませんでした",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                ),
                view=self,
            )


def discord_conversation_id(
    *,
    guild_id: int | None,
    channel_id: int,
    grants: frozenset[str] = frozenset(),
) -> str:
    """Map one Discord channel and capability profile to one agent conversation."""

    scope = f"guild:{guild_id}" if guild_id is not None else "direct"
    base = f"discord:{scope}:channel:{channel_id}"
    if not grants:
        return base
    profile = "+".join(sorted(grants))
    return f"{base}:profile:{profile}"


def _agent_grants(
    runtime: SimajilordRuntime,
    *,
    actor_id: str,
    autonomous: bool = False,
) -> frozenset[str]:
    settings = runtime.settings
    grants: set[str] = {
        AGENT_AUDIO_GRANT,
        AGENT_QUOTE_GRANT,
        AGENT_REPOST_GRANT,
    }
    if not autonomous:
        grants.add(AGENT_MESSAGE_GRANT)
    if not autonomous and settings.agent_file_sandbox_enabled and runtime.files is not None:
        grants.add(AGENT_FILE_GRANT)
    web_access = settings.agent_web_search_access
    if web_access is AgentFeatureAccess.EVERYONE or (
        web_access is AgentFeatureAccess.ADMINS and actor_id in settings.agent_admin_user_ids
    ):
        grants.add(AGENT_WEB_GRANT)
    if runtime.moderation.provider is not None:
        grants.add(AGENT_MODERATION_GRANT)
    image_access = settings.image_generation_access
    if (
        not autonomous
        and runtime.image.provider is not None
        and (
            image_access is AgentFeatureAccess.EVERYONE
            or (
                image_access is AgentFeatureAccess.ADMINS
                and actor_id in settings.agent_admin_user_ids
            )
        )
    ):
        grants.add(AGENT_IMAGE_GRANT)
    return frozenset(grants)


def _discord_message_chunks(content: str, *, maximum: int = 1_900) -> tuple[str, ...]:
    """Bound Discord output without asking the model to repeat a long answer."""

    text = content.strip()
    if not text:
        return ()
    chunks: list[str] = []
    while text:
        if len(text) <= maximum:
            chunks.append(text)
            break
        boundary = text.rfind("\n", 0, maximum + 1)
        if boundary < maximum // 2:
            boundary = text.rfind(" ", 0, maximum + 1)
        if boundary < maximum // 2:
            boundary = maximum
        chunks.append(text[:boundary].rstrip())
        text = text[boundary:].lstrip()
    return tuple(chunk for chunk in chunks if chunk)


def _agent_message_groups(content: str) -> tuple[str, ...]:
    """Convert every explicit boundary into an independent Discord post."""

    groups = content.split(AGENT_MESSAGE_BREAK)
    messages: list[str] = []
    for group in groups:
        messages.extend(_discord_message_chunks(group))
    return tuple(messages)


def _agent_error_text(error: Exception) -> str:
    if isinstance(error, AgentBusyError):
        return "AIへの依頼が混み合っています。少し待ってからもう一度お試しください。"
    if isinstance(error, AgentRateLimitError):
        if error.retry_after_seconds is not None:
            return (
                "AIの利用間隔を調整しています。"
                f"あと{_retry_after_text(error.retry_after_seconds)}ほどお待ちください。"
            )
        return "AIの利用間隔を調整しています。時間を空けてもう一度お試しください。"
    if isinstance(error, AgentUnavailableError):
        return "現在、このホストではSimajilord AIを利用できません。"
    return "AIの処理を完了できませんでした。"


def _retry_after_text(total_seconds: int) -> str:
    seconds = max(1, total_seconds)
    hours, remainder = divmod(seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}時間")
    if minutes:
        parts.append(f"{minutes}分")
    if seconds or not parts:
        parts.append(f"{seconds}秒")
    return "".join(parts)


_AGENT_PROGRESS_MESSAGES = {
    AgentProgressStage.QUEUED: "先に受け付けたAI処理が終わるのを待っています…",
    AgentProgressStage.STARTING: "依頼内容を確認しています…",
    AgentProgressStage.READING_DISCORD: "必要なDiscord上の会話を確認しています…",
    AgentProgressStage.SEARCHING_WEB: "Webを検索しています…",
    AgentProgressStage.COMPUTING: "計算を実行しています…",
    AgentProgressStage.ANALYZING_MEDIA: "添付ファイルをHIVEで解析しています…",
    AgentProgressStage.GENERATING_IMAGE: "ローカル画像生成の準備をしています…",
    AgentProgressStage.USING_AUDIO: "サーバーの音声機能を準備しています…",
    AgentProgressStage.PREPARING_RESPONSE: "回答をまとめています…",
}


class _AgentProgressMessage:
    """Coalesce real execution stages into one low-frequency Discord message."""

    def __init__(
        self,
        source: discord.Message,
        *,
        initial_delay_seconds: float = 1.0,
        minimum_update_seconds: float = 2.5,
    ) -> None:
        self.source = source
        self.initial_delay_seconds = initial_delay_seconds
        self.minimum_update_seconds = minimum_update_seconds
        self.message: discord.Message | None = None
        self._latest: AgentProgressStage | None = None
        self._published: AgentProgressStage | None = None
        self._last_update = 0.0
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def update(self, stage: AgentProgressStage) -> None:
        if self._closed:
            return
        self._latest = stage
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(
                self._flush_later(),
                name=f"simajilord-agent-progress-{self.source.id}",
            )

    async def finish(self, content: str) -> None:
        self._closed = True
        await self._cancel_pending()
        if content.strip() == AGENT_NO_ACTION_CONTENT:
            async with self._lock:
                if self.message is not None:
                    with suppress(discord.DiscordException):
                        await self.message.delete()
                    self.message = None
            return
        messages = _agent_message_groups(content)
        if not messages:
            return
        async with self._lock:
            if self.message is None:
                self.message = await self.source.reply(
                    messages[0],
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            else:
                await self.message.edit(
                    content=messages[0],
                    embed=None,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
        for message in messages[1:]:
            await self.source.channel.send(
                message,
                allowed_mentions=discord.AllowedMentions.none(),
            )

    async def fail(self, content: str) -> None:
        self._closed = True
        await self._cancel_pending()
        async with self._lock:
            if self.message is None:
                self.message = await self.source.reply(
                    content,
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            else:
                await self.message.edit(
                    content=content,
                    embed=None,
                    allowed_mentions=discord.AllowedMentions.none(),
                )

    async def _flush_later(self) -> None:
        try:
            if self.message is None:
                delay = self.initial_delay_seconds
            else:
                elapsed = time.monotonic() - self._last_update
                delay = max(0.0, self.minimum_update_seconds - elapsed)
            if delay:
                await asyncio.sleep(delay)
            if self._closed or self._latest is None or self._latest is self._published:
                return
            stage = self._latest
            embed = command_embed(
                "処理中",
                description=_AGENT_PROGRESS_MESSAGES[stage],
                tone=EmbedTone.INFO,
            )
            async with self._lock:
                if self._closed:
                    return
                if self.message is None:
                    self.message = await self.source.reply(
                        embed=embed,
                        mention_author=False,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                else:
                    await self.message.edit(content=None, embed=embed)
                self._published = stage
                self._last_update = time.monotonic()
        except asyncio.CancelledError:
            raise
        except discord.DiscordException:
            log.exception("Could not publish agent progress message.")
        finally:
            self._task = None
            if (
                not self._closed
                and self._latest is not None
                and self._latest is not self._published
            ):
                self._task = asyncio.create_task(
                    self._flush_later(),
                    name=f"simajilord-agent-progress-{self.source.id}",
                )

    async def _cancel_pending(self) -> None:
        task = self._task
        self._task = None
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class AgentCog(commands.Cog):
    """Wake one shared agent conversation only for explicit mentions."""

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        agent = self.runtime.agent
        bot_user = self.bot.user
        if (
            agent is None
            or bot_user is None
            or message.author.bot
            or message.webhook_id is not None
            or message.mention_everyone
            or bot_user not in message.mentions
        ):
            return
        if (
            message.guild is None
            or str(message.guild.id) not in self.runtime.settings.agent_allowed_guild_ids
        ):
            return
        member = (
            message.author
            if isinstance(message.author, discord.Member)
            else message.guild.get_member(message.author.id)
        )
        resource_ids = agent_readable_channel_ids(
            message.guild,
            member,
            trusted_guild=(str(message.guild.id) in self.runtime.settings.agent_trusted_guild_ids),
            trigger_channel_id=message.channel.id,
        )
        if str(message.channel.id) not in resource_ids:
            log.info(
                "Mention agent turn rejected by channel scope guild=%s channel=%s "
                "channel_type=%s actor=%s",
                message.guild.id,
                message.channel.id,
                type(message.channel).__name__,
                message.author.id,
            )
            return
        actor_id = str(message.author.id)
        grants = _agent_grants(self.runtime, actor_id=actor_id)
        approvals = frozenset(AGENT_AUDIO_WRITE_CAPABILITIES)
        request = AgentRequest(
            conversation_id=discord_conversation_id(
                guild_id=message.guild.id if message.guild else None,
                channel_id=message.channel.id,
                grants=grants,
            ),
            event_id=f"discord:message:{message.id}",
            trigger=AgentTrigger.MENTION,
            actor_id=actor_id,
            actor_name=message.author.display_name,
            workspace_id=str(message.guild.id) if message.guild else None,
            channel_id=str(message.channel.id),
            message_id=str(message.id),
            occurred_at=message.created_at,
            resource_ids=resource_ids,
            grants=grants,
            approvals=approvals,
        )
        progress = _AgentProgressMessage(message)
        try:
            async with message.channel.typing():
                response = await agent.respond(
                    request,
                    on_progress=progress.update,
                )
            await progress.finish(response.content)
        except Exception as exc:
            log.exception("Mention agent turn failed message=%s", message.id)
            await progress.fail(_agent_error_text(exc))


class ObservationCog(commands.Cog):
    """Feed Discord changes into the platform event stream for agent reconciliation."""

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None:
            return
        bot_user = self.bot.user
        await self.runtime.journal.append(
            kind="discord.message.created",
            actor_id=str(message.author.id),
            workspace_id=str(message.guild.id),
            transport="discord",
            request_id=str(message.id),
            payload={
                "message_id": str(message.id),
                "channel_id": str(message.channel.id),
                "author_name": message.author.display_name,
                "author_is_bot": message.author.bot,
                "content_length": len(message.content),
                "mentions_bot": bot_user in message.mentions if bot_user else False,
                "attachments": [
                    {
                        "id": str(attachment.id),
                        "filename": attachment.filename,
                        "size": attachment.size,
                    }
                    for attachment in message.attachments
                ],
            },
        )


class AgentAutonomyCog(commands.Cog):
    """Bounded default-off patrol over metadata-only Discord events."""

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime
        self._task: asyncio.Task[None] | None = None

    async def cog_load(self) -> None:
        settings = self.runtime.settings
        if (
            self.runtime.agent is not None
            and settings.agent_autonomy_enabled
            and bool(settings.agent_autonomy_guild_ids)
            and self._task is None
        ):
            self._task = asyncio.create_task(
                self._run(),
                name="simajilord-agent-autonomy",
            )

    async def cog_unload(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        settings = self.runtime.settings
        cursor = await self.runtime.journal.latest_sequence()
        await self.runtime.journal.append(
            kind="agent.autonomy.started",
            transport="agent",
            payload={
                "interval_seconds": settings.agent_autonomy_interval_seconds,
                "max_runs": settings.agent_autonomy_max_runs,
                "candidate_limit": settings.agent_autonomy_candidate_limit,
            },
        )
        completed_runs = 0
        try:
            while completed_runs < settings.agent_autonomy_max_runs:
                await asyncio.sleep(settings.agent_autonomy_interval_seconds)
                records = await self.runtime.journal.recent(
                    after_sequence=cursor,
                    limit=min(1_000, settings.agent_autonomy_candidate_limit * 20),
                )
                if records:
                    cursor = records[-1].sequence
                candidates = [
                    record
                    for record in records
                    if record.kind == "discord.message.created"
                    and record.actor_id is not None
                    and not bool(record.payload.get("author_is_bot"))
                    and not bool(record.payload.get("mentions_bot"))
                    and isinstance(record.payload.get("message_id"), str)
                    and isinstance(record.payload.get("channel_id"), str)
                    and record.workspace_id in settings.agent_autonomy_guild_ids
                ][-settings.agent_autonomy_candidate_limit :]
                completed_runs += 1
                await self.runtime.journal.append(
                    kind="agent.autonomy.checked",
                    transport="agent",
                    payload={
                        "run": completed_runs,
                        "candidate_count": len(candidates),
                        "model_called": bool(candidates),
                    },
                )
                if candidates:
                    await self._inspect(candidates[-1])
        except asyncio.CancelledError:
            raise
        finally:
            await self.runtime.journal.append(
                kind="agent.autonomy.stopped",
                transport="agent",
                payload={"completed_runs": completed_runs},
            )

    async def _inspect(self, record: EventRecord) -> None:
        agent = self.runtime.agent
        if agent is None:
            return
        payload = record.payload
        channel_id = str(payload["channel_id"])
        message_id = str(payload["message_id"])
        workspace_id = record.workspace_id
        occurred_at = record.occurred_at
        guild = self.bot.get_guild(int(workspace_id)) if workspace_id else None
        if guild is None:
            return
        resource_ids = agent_readable_channel_ids(
            guild,
            None,
            trusted_guild=True,
            trigger_channel_id=int(channel_id),
        )
        if channel_id not in resource_ids:
            return
        grants = _agent_grants(
            self.runtime,
            actor_id=AGENT_AUTONOMY_ACTOR_ID,
            autonomous=True,
        )
        request = AgentRequest(
            conversation_id=discord_conversation_id(
                guild_id=int(workspace_id) if workspace_id else None,
                channel_id=int(channel_id),
                grants=grants,
            ),
            event_id=f"autonomy:event:{record.sequence}",
            trigger=AgentTrigger.AUTONOMOUS,
            actor_id=AGENT_AUTONOMY_ACTOR_ID,
            actor_name="Simajilord autonomy",
            workspace_id=workspace_id,
            channel_id=channel_id,
            message_id=message_id,
            occurred_at=occurred_at if isinstance(occurred_at, datetime) else datetime.now(UTC),
            resource_ids=resource_ids,
            grants=grants,
        )
        try:
            response = await agent.respond(request)
            if response.content.strip() == AGENT_NO_ACTION_CONTENT:
                return
            channel = self.bot.get_channel(int(channel_id))
            if not isinstance(
                channel,
                (
                    discord.TextChannel,
                    discord.Thread,
                    discord.VoiceChannel,
                    discord.StageChannel,
                ),
            ):
                return
            messages = _agent_message_groups(response.content)
            if not messages:
                return
            try:
                target = await channel.fetch_message(int(message_id))
            except discord.DiscordException:
                target = None
            if target is not None:
                await target.reply(
                    messages[0],
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            else:
                await channel.send(
                    messages[0],
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            for message in messages[1:]:
                await channel.send(
                    message,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
        except (AgentBusyError, AgentRateLimitError):
            log.info("Autonomous agent check skipped by local budget.")
        except Exception:
            log.exception("Autonomous agent check failed message=%s", message_id)


class PrefixCog(commands.Cog):
    """Prefix presentation for the same APIs used by slash commands."""

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime
        dashboard = getattr(bot, _MUSIC_DASHBOARD_ATTRIBUTE, None)
        self.dashboard = dashboard if isinstance(dashboard, MusicDashboardManager) else None

    def _bind_music_dashboard(self, context: BotContext) -> None:
        if self.dashboard is not None:
            guild_id = context.guild.id if context.guild is not None else None
            self.dashboard.bind(guild_id, context.channel.id)

    @commands.command(name="ping")
    async def ping(self, context: BotContext) -> None:
        response = cast(
            PingResponse,
            await self.runtime.registry.invoke(
                "system.ping",
                PingRequest(transport_latency_ms=round(self.bot.latency * 1_000, 1)),
                prefix_context(context),
            ),
        )
        await context.send(
            embed=command_embed(
                "稼働状況",
                fields=(
                    EmbedField("状態", "正常" if response.status == "ok" else response.status),
                    EmbedField(
                        "Discord応答時間",
                        f"{response.transport_latency_ms:.1f} ms",
                    ),
                ),
                tone=EmbedTone.SUCCESS,
            )
        )

    @commands.command(name="capabilities", aliases=("help",))
    async def capabilities(self, context: BotContext, *, query: str = "") -> None:
        response = cast(
            CapabilitySearchResponse,
            await self.runtime.registry.invoke(
                "system.discover_capabilities",
                CapabilitySearchRequest(query=query, limit=8),
                prefix_context(context),
            ),
        )
        description = (
            "\n".join(
                f"• `{item.name}` — {item.summary} — 危険度: **{_risk_label(item.risk)}**"
                for item in response.capabilities
            )
            or "条件に合う機能は見つかりませんでした。"
        )
        await context.send(embed=command_embed("利用できる機能", description=description))

    @commands.command(name="search")
    async def search(self, context: BotContext, *, query: str) -> None:
        try:
            async with context.typing():
                response = cast(
                    WebSearchResponse,
                    await self.runtime.registry.invoke(
                        "web.search",
                        WebSearchRequest(query=query),
                        prefix_context(context),
                    ),
                )
            await context.send(embed=web_search_embed(response))
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "処理できませんでした",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                )
            )

    @commands.command(name="fetch")
    async def fetch(self, context: BotContext, url: str, offset: int = 0) -> None:
        try:
            async with context.typing():
                response = cast(
                    WebFetchResponse,
                    await self.runtime.registry.invoke(
                        "web.fetch",
                        WebFetchRequest(
                            url=url,
                            offset=offset,
                            max_characters=3_500,
                        ),
                        prefix_context(context),
                    ),
                )
            view = (
                WebFetchContinueView(self.runtime, response)
                if response.next_offset is not None
                else None
            )
            if view is None:
                await context.send(embed=web_fetch_embed(response))
            else:
                await context.send(
                    embed=web_fetch_embed(response),
                    view=view,
                )
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "処理できませんでした",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                )
            )

    @commands.command(name="find")
    async def find(
        self,
        context: BotContext,
        url: str,
        *,
        phrase: str,
    ) -> None:
        try:
            async with context.typing():
                response = cast(
                    WebFindResponse,
                    await self.runtime.registry.invoke(
                        "web.find",
                        WebFindRequest(url=url, pattern=phrase),
                        prefix_context(context),
                    ),
                )
            await context.send(embed=web_find_embed(response))
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "処理できませんでした",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                )
            )

    @commands.command(name="detectai")
    async def detectai(self, context: BotContext) -> None:
        try:
            if not context.message.attachments:
                raise UserError("moderation.media_empty")
            media = context.message.attachments[0]
            if media.size > self.runtime.settings.hive_max_media_bytes:
                raise UserError("moderation.media_too_large")
            async with context.typing():
                response = cast(
                    SyntheticMediaAnalyzeResponse,
                    await self.runtime.registry.invoke(
                        "moderation.detect_synthetic_media",
                        SyntheticMediaAnalyzeRequest(
                            filename=media.filename,
                            content_type=media.content_type,
                            content=await media.read(use_cached=True),
                        ),
                        prefix_context(context),
                    ),
                )
            await context.send(
                embed=synthetic_media_embed(
                    response,
                    attachment_url=media.url,
                )
            )
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "処理できませんでした",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                )
            )

    @commands.command(name="play")
    async def play(self, context: BotContext, *, reference: str) -> None:
        try:
            self._bind_music_dashboard(context)
            guild_id = context.guild.id if context.guild else None
            _discord_audio_session(self.bot, self.runtime, guild_id)
            selected_reference = reference
            if "://" not in reference:
                search = cast(
                    AudioSearchResponse,
                    await self.runtime.registry.invoke(
                        "audio.search",
                        AudioSearchRequest(query=reference, limit=5),
                        prefix_context(context),
                    ),
                )
                if search.selection_required:
                    view = MusicSearchChoiceView(
                        self.bot,
                        self.runtime,
                        search,
                        requester_id=context.author.id,
                        requester_name=context.author.display_name,
                    )
                    message = await context.send(
                        embed=music_search_embed(search),
                        view=view,
                        silent=True,
                    )
                    view.message = message
                    return
                if search.selected_index is None:
                    raise UserError("audio.search_empty")
                selected_reference = search.candidates[search.selected_index].reference
            response = cast(
                AudioPlayResponse,
                await self.runtime.registry.invoke(
                    "discord.play_audio",
                    AudioPlayRequest(
                        reference=selected_reference,
                        requested_by_name=context.author.display_name,
                    ),
                    prefix_context(context),
                ),
            )
            await context.send(
                embed=music_added_embed(response),
                silent=True,
                delete_after=8,
            )
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "処理できませんでした",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                )
            )

    @commands.command(name="queue")
    async def queue(self, context: BotContext, page: int = 1) -> None:
        try:
            self._bind_music_dashboard(context)
            guild_id = context.guild.id if context.guild else None
            _discord_audio_session(self.bot, self.runtime, guild_id)
            response = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    prefix_context(context),
                ),
            )
            await context.send(
                embed=music_queue_embed(response, page=page),
                view=MusicControlsView(
                    self.runtime,
                    self.dashboard,
                    response=response,
                ),
                silent=True,
            )
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "処理できませんでした",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                )
            )

    @commands.command(name="history")
    async def history(self, context: BotContext, limit: int = 10) -> None:
        try:
            self._bind_music_dashboard(context)
            guild_id = context.guild.id if context.guild else None
            _discord_audio_session(self.bot, self.runtime, guild_id)
            response = cast(
                AudioHistoryResponse,
                await self.runtime.registry.invoke(
                    "audio.history",
                    AudioHistoryRequest(limit=limit),
                    prefix_context(context),
                ),
            )
            await context.send(embed=music_history_embed(response), silent=True)
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "処理できませんでした",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                )
            )

    @commands.command(name="nowplaying", aliases=("np",))
    async def nowplaying(self, context: BotContext) -> None:
        try:
            self._bind_music_dashboard(context)
            guild_id = context.guild.id if context.guild else None
            _discord_audio_session(self.bot, self.runtime, guild_id)
            response = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    prefix_context(context),
                ),
            )
            await context.send(
                embed=music_now_playing_embed(response),
                view=MusicControlsView(
                    self.runtime,
                    self.dashboard,
                    response=response,
                ),
                silent=True,
            )
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "処理できませんでした",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                )
            )

    async def _control(
        self,
        context: BotContext,
        action: AudioAction,
        *,
        position: int | None = None,
        to_position: int | None = None,
        music_percent: int | None = None,
        speech_percent: int | None = None,
    ) -> None:
        try:
            self._bind_music_dashboard(context)
            if context.guild is None:
                raise UserError("workspace.required")
            session = self.runtime.audio.require(str(context.guild.id))
            _require_same_voice(session, context.author)
            response = cast(
                AudioControlResponse,
                await self.runtime.registry.invoke(
                    "audio.control",
                    AudioControlRequest(
                        action=action,
                        position=position,
                        to_position=to_position,
                        music_percent=music_percent,
                        speech_percent=speech_percent,
                    ),
                    prefix_context(context),
                ),
            )
            description = _AUDIO_ACTION_MESSAGES.get(
                response.action,
                "音楽の状態を更新しました。",
            )
            if response.action == AudioAction.MOVE.value:
                description = (
                    f"**{response.affected_title}** を再生待ちの "
                    f"**{to_position}番目**へ移動しました。"
                )
            elif response.action == AudioAction.CLEAR_MINE.value:
                description = (
                    f"あなたが追加した曲を **{response.removed_count or 0}曲** "
                    "キューから削除しました。"
                )
            elif response.action == AudioAction.VOLUME.value:
                description = (
                    f"音楽 **{response.music_volume_percent}%**・"
                    f"読み上げ **{response.speech_volume_percent}%** にしました。"
                )
            await context.send(
                embed=command_embed(
                    "音楽を操作しました",
                    description=description,
                    tone=EmbedTone.SUCCESS,
                ),
                silent=True,
                delete_after=8,
            )
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "処理できませんでした",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                )
            )

    @commands.command(name="pause")
    async def pause(self, context: BotContext) -> None:
        await self._control(context, AudioAction.PAUSE)

    @commands.command(name="resume")
    async def resume(self, context: BotContext) -> None:
        await self._control(context, AudioAction.RESUME)

    @commands.command(name="skip")
    async def skip(self, context: BotContext) -> None:
        await self._control(context, AudioAction.SKIP)

    @commands.command(name="stop")
    async def stop(self, context: BotContext) -> None:
        await self._control(context, AudioAction.STOP)

    @commands.command(name="leave")
    async def leave(self, context: BotContext) -> None:
        await self._control(context, AudioAction.LEAVE)

    @commands.command(name="volume")
    async def volume(
        self,
        context: BotContext,
        music: int | None = None,
        read_aloud: int | None = None,
    ) -> None:
        await self._control(
            context,
            AudioAction.VOLUME,
            music_percent=music,
            speech_percent=read_aloud,
        )

    @commands.command(name="move")
    async def move(
        self,
        context: BotContext,
        from_position: int,
        to_position: int,
    ) -> None:
        await self._control(
            context,
            AudioAction.MOVE,
            position=from_position,
            to_position=to_position,
        )

    @commands.command(name="clear-mine", aliases=("clearmine",))
    async def clear_mine(self, context: BotContext) -> None:
        await self._control(context, AudioAction.CLEAR_MINE)


async def setup_cogs(bot: commands.Bot, runtime: SimajilordRuntime) -> None:
    dashboard = MusicDashboardManager(bot, runtime)
    setattr(bot, _MUSIC_DASHBOARD_ATTRIBUTE, dashboard)
    bot.add_view(MusicControlsView(runtime, dashboard))
    await bot.add_cog(SystemCog(bot, runtime))
    await bot.add_cog(MusicCog(bot, runtime, dashboard))
    await bot.add_cog(ReadAloudCog(bot, runtime))
    await bot.add_cog(VoiceLifecycleCog(bot, runtime))
    await bot.add_cog(WebCog(runtime))
    await bot.add_cog(ModerationCog(runtime))
    await bot.add_cog(DownloadCog(runtime))
    await bot.add_cog(UtilityCog(runtime))
    await bot.add_cog(DiscordInfoCog(runtime))
    await bot.add_cog(DiscordActionCog(runtime))
    await bot.add_cog(MessageExpandCog(runtime))
    quote_cog = QuoteCog(runtime)
    await bot.add_cog(quote_cog)
    bot.tree.add_command(
        app_commands.ContextMenu(
            name=_QUOTE_CONTEXT_MENU_NAME,
            callback=quote_cog.create_quote,
        )
    )
    await bot.add_cog(AgentCog(bot, runtime))
    await bot.add_cog(ObservationCog(bot, runtime))
    await bot.add_cog(AgentAutonomyCog(bot, runtime))
    await bot.add_cog(PrefixCog(bot, runtime))
