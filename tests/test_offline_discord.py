from __future__ import annotations

import discord

from simajilord.diagnostics.offline_discord import (
    _discord_markdown,
    _render_component,
    _sample_panels,
    render_preview_html,
    serialize_view,
)


def test_offline_preview_uses_exact_embed_and_adaptive_components() -> None:
    panels = _sample_panels()

    assert [panel.name for panel in panels] == [
        "Idle · disconnected",
        "Queued · waiting for voice",
        "Playing · radio and read aloud",
        "Quote · main menu",
        "Help · overview",
        "Help · /play",
        "Server info · populated",
        "User info · populated",
        "Audio hub · ready",
        "Audio hub · outside VC",
        "Audio settings · personal",
        "Audio settings · shared",
        "Audio settings · sources",
    ]
    idle_labels = tuple(
        str(component.get("label"))
        for row in panels[0].components
        for component in row["components"]
        if component.get("type") == 2
    )
    waiting_labels = tuple(
        str(component.get("label"))
        for row in panels[1].components
        for component in row["components"]
        if component.get("type") == 2
    )
    active_labels = tuple(
        str(component.get("label"))
        for row in panels[2].components
        for component in row["components"]
        if component.get("type") == 2
    )
    assert idle_labels == ("Add music", "Audio setup")
    assert waiting_labels == (
        "Resume music",
        "Add music",
        "Audio setup",
    )
    assert "Pause" in active_labels
    assert "Skip" in active_labels
    assert "Stop" in active_labels
    assert "Add music" in active_labels
    assert panels[0].embed["title"] == "Audio"
    assert tuple(
        str(component.get("label"))
        for row in panels[3].components
        for component in row["components"]
        if component.get("type") == 2
    ) == (
        "Layout · Landscape",
        "Style · B/W",
        "More · 1 On",
        "Generate",
        "Cancel",
    )
    assert panels[4].embed["title"] == "Help"
    overview_fields = panels[4].embed["fields"]
    assert all("commands" in str(field["name"]) for field in overview_fields)
    assert all("`/" not in str(field["value"]) for field in overview_fields)
    assert panels[5].embed["title"] == "/play"
    assert [field["name"] for field in panels[5].embed["fields"]] == [
        "Usage",
        "Examples",
        "Required permissions",
        "Side effects",
        "Behaviour notes",
        "Common errors",
    ]
    assert panels[6].embed["title"] == "Simajilord Audio Lab"
    assert panels[7].embed["title"] == "Meteo in Simajilord"


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


def test_preview_preserves_discord_action_rows_and_order() -> None:
    view = discord.ui.View()
    view.add_item(discord.ui.Button(label="Second row", row=1))
    view.add_item(discord.ui.Button(label="First row", row=0))
    assert serialize_view(view) == tuple(view.to_components())
    assert serialize_view(view)[0]["components"][0]["label"] == "First row"


def test_preview_select_preserves_disabled_options_and_defaults() -> None:
    control = discord.ui.Select(
        placeholder="Choose a command",
        disabled=True,
        options=[discord.SelectOption(label="Play <music>", value="play", default=True)],
    )
    rendered = _render_component(control.to_component_dict())
    assert "<select " in rendered
    assert " disabled" in rendered
    assert '<option value="play" selected>Play &lt;music&gt;</option>' in rendered
    assert 'aria-label="Choose a command"' in rendered


def test_preview_preserves_button_and_option_emoji_text() -> None:
    labelled = discord.ui.Button(label="Play", emoji="▶️")
    icon_only = discord.ui.Button(emoji="▶️")
    control = discord.ui.Select(options=[discord.SelectOption(label="Radio", emoji="📻")])
    assert "▶️ Play</button>" in _render_component(labelled.to_component_dict())
    assert ">▶️</button>" in _render_component(icon_only.to_component_dict())
    assert "📻 Radio</option>" in _render_component(control.to_component_dict())
