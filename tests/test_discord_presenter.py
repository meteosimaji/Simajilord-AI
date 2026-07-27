from __future__ import annotations

from simajilord.capabilities.audio import (
    AudioHistoryItem,
    AudioHistoryResponse,
    AudioPlayResponse,
    AudioQueueItem,
    AudioQueueResponse,
    AudioSearchItem,
    AudioSearchReason,
    AudioSearchResponse,
)
from simajilord.capabilities.moderation import SyntheticMediaAnalyzeResponse
from simajilord.capabilities.web import (
    WebFetchResponse,
    WebSearchResponse,
)
from simajilord.domain.image import (
    ImageGenerationJob,
    ImageGenerationPrompt,
    ImageJobStatus,
    ImageRendering,
)
from simajilord.domain.moderation import SyntheticMediaModality, SyntheticMediaVerdict
from simajilord.domain.web import WebSource
from simajilord.integrations.discord.bot import _image_result_embed
from simajilord.integrations.discord.capabilities import (
    DiscordExpandedAttachmentRecord,
    DiscordExpandMessageResponse,
)
from simajilord.integrations.discord.cogs import (
    music_added_embed,
    music_history_embed,
    music_queue_embed,
    music_search_embed,
    synthetic_media_embed,
    web_fetch_embed,
    web_search_embed,
)
from simajilord.integrations.discord.presenter import (
    EmbedField,
    EmbedTone,
    command_embed,
    expanded_message_embeds,
    expanded_message_view,
)


def test_command_embed_keeps_useful_timestamp_without_meta_footer() -> None:
    embed = command_embed(
        "Platform status",
        fields=(EmbedField("Status", "ok"),),
        tone=EmbedTone.SUCCESS,
    )
    assert embed.title == "Platform status"
    assert embed.timestamp is not None
    assert embed.footer.text is None
    assert embed.fields[0].name == "Status"
    assert embed.fields[0].value == "ok"


def test_expanded_message_keeps_jump_and_bounds_visible_images() -> None:
    response = DiscordExpandMessageResponse(
        guild_id="1",
        channel_id="2",
        channel_name="general",
        message_id="3",
        jump_url="https://discord.com/channels/1/2/3",
        author_id="4",
        author_name="Alice",
        author_avatar_url="https://cdn.example.com/avatar.png",
        author_is_bot=False,
        content="hello",
        created_at_iso="2026-07-27T00:00:00+00:00",
        edited_at_iso=None,
        attachments=tuple(
            DiscordExpandedAttachmentRecord(
                filename=f"{index}.png",
                content_type="image/png",
                size_bytes=100,
                url=f"https://cdn.example.com/{index}.png",
                proxy_url=f"https://proxy.example.com/{index}.png",
                spoiler=False,
            )
            for index in range(6)
        ),
        embeds=(),
        sticker_names=(),
        poll=None,
        reply_author_name=None,
        reply_content_preview=None,
    )
    embeds = expanded_message_embeds(response)
    view = expanded_message_view(response.jump_url)
    assert len(embeds) == 4
    assert embeds[0].title == "元のメッセージへ"
    assert embeds[0].url == response.jump_url
    assert embeds[0].image.url == "https://proxy.example.com/0.png"
    button = view.children[0]
    assert getattr(button, "label", None) == "Jump"
    assert getattr(button, "url", None) == response.jump_url


def test_generated_image_embed_shows_the_actual_creative_brief() -> None:
    job = ImageGenerationJob(
        job_id="job",
        actor_id="actor",
        workspace_id="workspace",
        delivery_target_id="channel",
        reply_to_message_id=None,
        prompt=ImageGenerationPrompt(
            subject="Exactly one orange cat with amber eyes",
            scene="A rainy apartment window and a low walnut table",
            composition="Landscape portrait with the cat on the left third",
            style="Natural editorial pet photography",
            lighting="Cool window light with a warm lamp rim",
            rendering=ImageRendering.PHOTO,
        ),
        caption_json="{}",
        status=ImageJobStatus.COMPLETED,
        output_path=None,
        width=768,
        height=512,
        seed=42,
        created_at_iso="2026-07-27T00:00:00+00:00",
        generation_seconds=153.0,
    )

    embed = _image_result_embed(job, filename="image.png")

    assert "Exactly one orange cat" in (embed.description or "")
    assert "rainy apartment window" in (embed.description or "")
    assert "Natural editorial pet photography" in (embed.description or "")
    assert "Cool window light" in (embed.description or "")


def test_hive_embed_separates_ai_deepfake_and_generator_signals() -> None:
    embed = synthetic_media_embed(
        SyntheticMediaAnalyzeResponse(
            sha256="a" * 64,
            filename="sample.png",
            content_type="image/png",
            modality=SyntheticMediaModality.IMAGE,
            ai_generated_score=0.999987,
            not_ai_generated_score=0.000013,
            deepfake_score=0.0002827,
            deepfake_likely=False,
            sample_count=1,
            model="hive/ai-generated-and-deepfake-content-detection",
            threshold=0.9,
            top_source="stablediffusionxl",
            top_source_score=0.99166,
            verdict=SyntheticMediaVerdict.AI_GENERATED,
            version="1",
            cached=False,
            quota_used=4,
            quota_remaining=96,
            quota_limit=100,
            quota_reset_at_epoch=1_800_000_000,
        ),
        attachment_url="https://cdn.example.com/sample.png",
    )

    assert embed.title == "HIVE AIコンテンツ解析"
    assert embed.description == "**AI生成画像の可能性: 高**"
    assert "ディープフェイク" not in (embed.description or "")
    fields = {field.name: field.value for field in embed.fields}
    assert fields["AI生成"] == "**100.0%**・可能性 高"
    assert fields["ディープフェイク"] == "**0.0%**・可能性 低"
    assert fields["推定生成モデル"] == "**Stable Diffusion XL** · 99.2%"
    assert "本日のHIVE API利用枠" not in fields
    assert "HIVE Moderation" in embed.footer.text
    assert embed.thumbnail.url == "https://cdn.example.com/sample.png"


def test_music_embed_contains_track_progress_queue_and_operational_state() -> None:
    embed = music_queue_embed(
        AudioQueueResponse(
            current=AudioQueueItem(
                title="Current",
                page_url="https://example.com/current",
                kind="music",
                duration_seconds=180,
                requested_by_name="Alice",
                uploader="Current Artist",
                thumbnail_url="https://img.example.com/current.jpg",
            ),
            pending=(
                AudioQueueItem(
                    title="Next",
                    page_url="https://example.com/next",
                    kind="music",
                    duration_seconds=90,
                    requested_by_name="Bob",
                ),
            ),
            paused=False,
            loop_mode="queue",
            destination_id="123",
            auto_leave=True,
            position_seconds=45,
            speed=1.25,
            pitch=1.0,
            waiting_for_voice=False,
        )
    )
    assert embed.title == "音楽"
    assert embed.timestamp is not None
    assert embed.footer.text is None
    fields = {field.name: field.value for field in embed.fields}
    assert "0:45 / 3:00" in fields["再生位置"]
    assert "Next" in fields["次に再生"]
    assert fields["状態"] == "再生中"
    assert fields["ループ"] == "キュー全体"
    assert fields["自動退出"] == "オン"
    assert fields["再生先"] == "<#123>"
    assert "速度 1.25倍" in fields["再生調整"]
    assert fields["追加した人"] == "Alice"
    assert fields["投稿者"] == "Current Artist"
    assert "Bob" in fields["次に再生"]
    assert embed.thumbnail.url == "https://img.example.com/current.jpg"


def test_waiting_play_embed_explains_that_no_reentry_is_needed() -> None:
    embed = music_added_embed(
        AudioPlayResponse(
            title="Queued track",
            page_url="https://example.com/queued",
            queue_position=1,
            duration_seconds=90,
            destination_id=None,
            playback_state="waiting_for_voice",
            requested_by_name="Alice",
            uploader="Artist",
            thumbnail_url="https://img.example.com/queued.jpg",
        )
    )
    fields = {field.name: field.value for field in embed.fields}
    assert embed.title == "キューに追加しました"
    assert "自動で再生" in fields["状態"]
    assert fields["追加した人"] == "Alice"
    assert fields["再生先"] == "追加したユーザーがVCに入ると接続"
    assert fields["投稿者"] == "Artist"
    assert embed.thumbnail.url == "https://img.example.com/queued.jpg"


def test_music_history_embed_shows_requester_and_played_time() -> None:
    embed = music_history_embed(
        AudioHistoryResponse(
            items=(
                AudioHistoryItem(
                    title="Played track",
                    page_url="https://example.com/played",
                    duration_seconds=120,
                    requested_by_name="Alice",
                    played_at_epoch=1_700_000_000,
                ),
            )
        )
    )
    assert embed.title == "再生履歴"
    assert "Alice" in (embed.description or "")
    assert "<t:1700000000:R>" in (embed.description or "")


def test_ambiguous_search_embed_is_compact_and_actionable() -> None:
    embed = music_search_embed(
        AudioSearchResponse(
            query="Hello",
            candidates=(
                AudioSearchItem(
                    "https://example.com/one",
                    "Artist One - Hello",
                    180,
                    uploader="Artist One",
                    thumbnail_url="https://img.example.com/one.jpg",
                ),
                AudioSearchItem(
                    "https://example.com/two",
                    "Artist Two - Hello",
                    190,
                    uploader="Artist Two",
                ),
            ),
            selected_index=None,
            selection_required=True,
            reason=AudioSearchReason.AMBIGUOUS_TITLE,
        )
    )
    assert embed.title == "再生する曲を選んでください"
    assert len(embed.fields) == 2
    assert embed.fields[0].name == "1 · 3:00"
    assert "Artist One" in embed.fields[0].value
    assert "次回から自動で選びやすく" in (embed.description or "")
    assert embed.thumbnail.url == "https://img.example.com/one.jpg"


def test_web_search_embed_shows_sources_coverage_and_timestamp() -> None:
    embed = web_search_embed(
        WebSearchResponse(
            query="example",
            backend="searxng",
            sources=(
                WebSource(
                    source_id="S1",
                    title="Example result",
                    url="https://example.com/page",
                    host="example.com",
                    snippet="A concise source excerpt.",
                    category="general",
                ),
            ),
            raw_candidate_count=37,
            candidate_count=37,
            maybe_more=True,
            warnings=("one provider failed",),
        )
    )
    assert embed.title == "検索結果"
    assert embed.timestamp is not None
    assert "Example result" in (embed.description or "")
    fields = {field.name: field.value for field in embed.fields}
    assert "候補 37件" in fields["検索範囲"]
    assert "ほかにも候補あり" in fields["検索範囲"]
    assert "1件の情報源" in fields["検索サービスの状態"]


def test_web_fetch_embed_preserves_continuation_offset() -> None:
    embed = web_fetch_embed(
        WebFetchResponse(
            title="Opened page",
            url="https://example.com/page",
            content_type="text/html",
            text="Readable text",
            offset=200,
            total_characters=900,
            next_offset=213,
            links=(),
        )
    )
    fields = {field.name: field.value for field in embed.fields}
    assert embed.title == "Opened page"
    assert fields["続きの開始位置"] == "213"
    assert "213 / 900文字" in fields["本文"]
