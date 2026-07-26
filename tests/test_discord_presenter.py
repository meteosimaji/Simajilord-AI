from __future__ import annotations

from simajilord.capabilities.audio import (
    AudioHistoryItem,
    AudioHistoryResponse,
    AudioPlayResponse,
    AudioQueueItem,
    AudioQueueResponse,
)
from simajilord.integrations.discord.cogs import (
    music_added_embed,
    music_history_embed,
    music_queue_embed,
)
from simajilord.integrations.discord.presenter import (
    EmbedField,
    EmbedTone,
    command_embed,
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


def test_music_embed_contains_track_progress_queue_and_operational_state() -> None:
    embed = music_queue_embed(
        AudioQueueResponse(
            current=AudioQueueItem(
                title="Current",
                page_url="https://example.com/current",
                kind="music",
                duration_seconds=180,
                requested_by_name="Alice",
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
    assert embed.title == "Music"
    assert embed.timestamp is not None
    assert embed.footer.text is None
    fields = {field.name: field.value for field in embed.fields}
    assert "0:45 / 3:00" in fields["Progress"]
    assert "Next" in fields["Up next"]
    assert fields["State"] == "Playing"
    assert fields["Loop"] == "Queue"
    assert fields["Auto leave"] == "On"
    assert fields["Voice"] == "<#123>"
    assert "1.25x speed" in fields["Tuning"]
    assert fields["Requested by"] == "Alice"
    assert "Bob" in fields["Up next"]


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
        )
    )
    fields = {field.name: field.value for field in embed.fields}
    assert embed.title == "Added to queue"
    assert "Join a voice channel to start automatically" in fields["Playback"]
    assert fields["Requested by"] == "Alice"
    assert fields["Voice"] == "Not connected yet"


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
    assert embed.title == "Recently played"
    assert "Alice" in (embed.description or "")
    assert "<t:1700000000:R>" in (embed.description or "")
