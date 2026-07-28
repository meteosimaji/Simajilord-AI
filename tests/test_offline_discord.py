from __future__ import annotations

from simajilord.diagnostics.offline_discord import (
    _discord_markdown,
    _sample_panels,
    render_preview_html,
)


def test_offline_preview_uses_exact_embed_and_adaptive_components() -> None:
    panels = _sample_panels()

    assert [panel.name for panel in panels] == [
        "Idle · disconnected",
        "Queued · waiting for voice",
        "Playing · radio and read aloud",
        "Quote · main menu",
        "YouTube · audio actions",
        "Help · overview",
        "Help · /play",
        "Server info · populated",
        "User info · populated",
    ]
    idle_labels = tuple(
        str(component.get("label"))
        for component in panels[0].components
        if component.get("type") == 2
    )
    waiting_labels = tuple(
        str(component.get("label"))
        for component in panels[1].components
        if component.get("type") == 2
    )
    active_labels = tuple(
        str(component.get("label"))
        for component in panels[2].components
        if component.get("type") == 2
    )
    assert idle_labels == ("Add music",)
    assert waiting_labels == (
        "Start",
        "Add music",
    )
    assert "Pause" in active_labels
    assert "Skip" in active_labels
    assert "Stop" in active_labels
    assert "Add music" in active_labels
    assert panels[0].embed["title"] == "Audio"
    assert tuple(
        str(component.get("label"))
        for component in panels[3].components
        if component.get("type") == 2
    ) == (
        "Layout · Landscape",
        "Style · B/W",
        "More · 1 On",
        "Generate",
        "Cancel",
    )
    assert tuple(
        str(component.get("label"))
        for component in panels[4].components
        if component.get("type") == 2
    ) == ("Play", "Add", "Radio")
    assert panels[5].embed["title"] == "Help"
    assert panels[6].embed["title"] == "/play"
    assert panels[7].embed["title"] == "Simajilord Audio Lab"
    assert panels[8].embed["title"] == "Meteo in Simajilord"


def test_offline_html_is_local_and_contains_audio_controls() -> None:
    output = render_preview_html(
        _sample_panels(),
        speech_filename="speech.wav",
        mixed_audio_filename="mixed.wav",
    )

    assert "No Discord gateway, webhook," in output
    assert "token, or server send is used." in output
    assert 'src="speech.wav"' in output
    assert 'src="mixed.wav"' in output
    assert "discord.com/api" not in output
    assert "https://cdn.discordapp.com" not in output
    assert "<script" in output
    assert 'class="component-button ' in output
    assert "max-width: 516px" in output
    assert "padding: 2px 16px 16px 12px" in output
    assert "font-size: 14px" in output
    assert "line-height: 18px" in output
    assert "--discord-control-primary: rgb(88 101 242)" in output
    assert "border-radius: 8px" in output
    assert "&lt;t:" not in output


def test_markdown_link_escapes_query_string_once() -> None:
    output = _discord_markdown("[track](https://example.invalid/watch?a=1&b=2)")

    assert output == '<a href="https://example.invalid/watch?a=1&amp;b=2">track</a>'


def test_markdown_heading_matches_discord_block_flow_without_extra_break() -> None:
    output = _discord_markdown("### Track\n`1:05 / 4:09`\nArtist")

    assert output == "<h3>Track</h3><code>1:05 / 4:09</code><br>Artist"
