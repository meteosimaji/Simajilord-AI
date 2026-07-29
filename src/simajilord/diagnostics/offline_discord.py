"""Render Discord-facing UI and audio locally without a bot connection or token."""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import discord

from simajilord.capabilities.audio import AudioQueueItem, AudioQueueResponse
from simajilord.domain.audio import AudioKind
from simajilord.integrations.discord.capabilities import (
    DiscordServerResponse,
    DiscordUserResponse,
)
from simajilord.integrations.discord.cogs import (
    HelpView,
    MusicControlsView,
    QuoteComposerView,
    _help_entry_embed,
    _help_overview_embed,
    music_queue_embed,
    server_info_embed,
    user_info_embed,
)
from simajilord.integrations.discord.help_catalog import HELP_ENTRIES_BY_TOPIC
from simajilord.providers.speech import MacOSSayProvider
from simajilord.runtime import SimajilordRuntime
from simajilord.services.speech import SpeechService

_MARKDOWN_LINK = re.compile(r"\[([^\]]+)]\((https?://[^)\s]+)\)")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_CHANNEL = re.compile(r"&lt;#(\d+)&gt;")
_TIMESTAMP = re.compile(r"&lt;t:(\d+)(?::([tTdDfFR]))?&gt;")
_LINK_PLACEHOLDER = "\u0000SIMAJILORD_LINK_{index}\u0000"


@dataclass(frozen=True, slots=True)
class PreviewPanel:
    """One exact embed/view pair produced by the Discord adapter."""

    name: str
    embed: dict[str, Any]
    components: tuple[dict[str, Any], ...]


def serialize_view(view: discord.ui.View) -> tuple[dict[str, Any], ...]:
    """Serialize the same component objects Discord.py would send."""

    return tuple(dict(child.to_component_dict()) for child in view.children)


def render_preview_html(
    panels: tuple[PreviewPanel, ...],
    *,
    speech_filename: str,
    mixed_audio_filename: str,
) -> str:
    """Return an offline simulator calibrated against the current Discord Web UI."""

    panel_markup = "\n".join(_render_panel(panel) for panel in panels)
    case_buttons = "\n".join(
        (
            f'<button class="case-button{" selected" if index == 0 else ""}" '
            f'data-case-target="{index}" type="button">'
            f"{html.escape(panel.name)}</button>"
        )
        for index, panel in enumerate(panels)
    )
    return f"""<!doctype html>
<html lang="ja" data-discord-client="web-2026-07-28">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline';
               img-src 'self' data:; media-src 'self'">
<title>Simajilord offline Discord preview</title>
<style>
:root {{
  color-scheme: dark;
  font-family: "gg sans", "Hiragino Sans", "ヒラギノ角ゴ ProN W3",
    "Hiragino Kaku Gothic ProN", Meiryo, Osaka, "MS PGothic",
    "Noto Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
  --discord-background-base-lowest: hsl(240 5.263% 7.451%);
  --discord-background-base-lower: hsl(240 7.143% 10.98%);
  --discord-background-surface-high: hsl(240 6.494% 15.098%);
  --discord-background-mod-subtle: hsl(240 4% 60.784% / 12.1569%);
  --discord-border-subtle: hsl(240 4% 60.784% / 12.1569%);
  --discord-border-normal: hsl(240 4% 60.784% / 20%);
  --discord-text-default: hsl(240 6.667% 94.118%);
  --discord-text-strong: hsl(0 0% 98.431%);
  --discord-text-muted: hsl(232.5 3.96% 60.392%);
  --discord-text-link: hsl(212.795 82.564% 61.765%);
  --discord-control-primary: rgb(88 101 242);
  --discord-control-connected: rgb(0 133 69);
  --discord-control-critical: rgb(210 45 57);
  --discord-control-border: hsl(0 0% 100% / 7.8431%);
  background: var(--discord-background-base-lowest);
  color: var(--discord-text-default);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  min-height: 100vh;
  background: var(--discord-background-base-lowest);
  font-size: 16px;
  line-height: 16px;
}}
.simulator {{
  display: grid;
  grid-template-columns: 248px minmax(0, 1fr);
  min-height: 100vh;
}}
.simulator-sidebar {{
  padding: 20px 16px;
  background: var(--discord-background-base-lowest);
  border-right: 1px solid var(--discord-border-subtle);
}}
.simulator-sidebar h1 {{
  margin: 0 0 5px;
  color: var(--discord-text-strong);
  font-size: 17px;
  line-height: 22px;
}}
.simulator-sidebar p, .muted {{
  margin: 0;
  color: var(--discord-text-muted);
  font-size: 12px;
  line-height: 16px;
}}
.case-switcher {{
  display: grid;
  gap: 6px;
  margin-top: 18px;
}}
.case-button {{
  width: 100%;
  min-height: 32px;
  padding: 5px 8px;
  border: 0;
  border-radius: 4px;
  color: var(--discord-text-default);
  background: transparent;
  font: inherit;
  font-size: 14px;
  font-weight: 500;
  line-height: 18px;
  text-align: left;
  cursor: pointer;
}}
.case-button:hover, .case-button.selected {{
  background: var(--discord-background-mod-subtle);
  color: var(--discord-text-strong);
}}
.local-audio {{
  margin-top: 22px;
  padding-top: 16px;
  border-top: 1px solid var(--discord-border-subtle);
}}
.audio-card {{ margin-top: 12px; }}
.audio-card strong {{
  display: block;
  margin-bottom: 5px;
  color: var(--discord-text-default);
  font-size: 12px;
  line-height: 16px;
}}
audio {{ width: 100%; height: 28px; }}
.discord-window {{
  min-width: 0;
  background: var(--discord-background-base-lower);
}}
.channel-header {{
  display: flex;
  align-items: center;
  height: 48px;
  padding: 0 16px;
  border-bottom: 1px solid var(--discord-border-subtle);
  color: var(--discord-text-strong);
  font-size: 16px;
  font-weight: 600;
  line-height: 22px;
}}
.channel-hash {{
  margin-right: 8px;
  color: var(--discord-text-muted);
  font-size: 24px;
  font-weight: 400;
}}
.chat-scroller {{
  min-height: calc(100vh - 48px);
  overflow: auto;
  padding: 24px 0 40px;
}}
.case {{
  display: none;
  min-width: 0;
}}
.case.active {{ display: block; }}
.message {{
  position: relative;
  width: 100%;
  margin-top: 17px;
  padding: 2px 24px 2px 72px;
}}
.avatar {{
  position: absolute;
  top: 2px;
  left: 16px;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  margin-top: 2px;
  background: var(--discord-text-default);
  color: var(--discord-background-base-lowest);
  font-size: 13px;
  font-weight: 700;
}}
.message-header {{
  height: 22px;
  line-height: 22px;
}}
.author {{
  display: inline;
  color: var(--discord-text-strong);
  font-size: 16px;
  font-weight: 500;
  line-height: 22px;
}}
.bot {{
  display: inline-flex;
  align-items: center;
  height: 15px;
  margin: 2px 0 0 4px;
  padding: 0 4.4px;
  border-radius: 4px;
  background: #5865f2;
  color: #fff;
  font-size: 10px;
  font-weight: 400;
  line-height: 15px;
  vertical-align: top;
}}
.message-time {{
  margin-left: 4px;
  color: var(--discord-text-muted);
  font-size: 12px;
  font-weight: 500;
  line-height: 22px;
}}
.embed-container {{
  display: grid;
  grid-template-columns: minmax(0, 516px);
  gap: 4px;
  padding: 2px 0;
}}
.embed {{
  display: grid;
  max-width: max-content;
  margin-top: 8px;
  overflow: hidden;
  border: 1px solid var(--discord-border-subtle);
  border-left: 4px solid var(--embed-color, var(--discord-border-normal));
  border-radius: 4px;
  background: var(--discord-background-surface-high);
  box-shadow: none;
}}
.embed-grid {{
  display: grid;
  grid-template-columns: auto;
  grid-template-rows: auto;
  overflow: hidden;
  padding: 2px 16px 16px 12px;
}}
.embed-grid.has-thumbnail {{
  grid-template-columns: minmax(0, 1fr) 80px;
  column-gap: 16px;
}}
.embed-grid.has-thumbnail > :not(.embed-thumbnail) {{ grid-column: 1; }}
.embed-thumbnail {{
  grid-column: 2;
  grid-row: 1 / span 4;
  width: 80px;
  height: 80px;
  margin-top: 8px;
  border-radius: 4px;
  display: grid;
  place-items: center;
  background:
    linear-gradient(135deg, hsl(235 85.6% 64.7%), hsl(262 83% 58%));
  color: white;
  font-size: 26px;
  font-weight: 700;
}}
.embed-title {{
  min-width: 0;
  margin-top: 8px;
  color: var(--discord-text-strong);
  font-size: 16px;
  font-weight: 600;
  line-height: 22px;
}}
.description {{
  min-width: 0;
  margin-top: 8px;
  color: var(--discord-text-default);
  font-size: 14px;
  font-weight: 400;
  line-height: 18px;
  overflow-wrap: anywhere;
  text-align: start;
  unicode-bidi: plaintext;
  white-space: pre-line;
}}
.description h3 {{
  margin: 4px 0 8px;
  color: var(--discord-text-strong);
  font-size: 14px;
  font-weight: 700;
  line-height: 18px;
}}
.description a, .field a {{
  color: var(--discord-text-link);
  text-decoration: none;
}}
.description code, .field code {{
  padding: 0 4px;
  border-radius: 3px;
  background: var(--discord-background-base-lowest);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 85%;
}}
.fields {{
  display: grid;
  grid-template-columns: repeat(12, minmax(0, 1fr));
  grid-auto-flow: row;
  gap: 8px;
  margin-top: 8px;
}}
.field {{
  grid-column: span 6;
  min-width: 0;
}}
.field.wide {{ grid-column: 1 / -1; }}
.field-name {{
  min-width: 0;
  margin-bottom: 2px;
  color: var(--discord-text-strong);
  font-size: 14px;
  font-weight: 600;
  line-height: 18px;
  text-align: start;
  unicode-bidi: plaintext;
}}
.field-value {{
  min-width: 0;
  color: var(--discord-text-default);
  font-size: 14px;
  font-weight: 400;
  line-height: 18px;
  overflow-wrap: anywhere;
  text-align: start;
  unicode-bidi: plaintext;
  white-space: pre-line;
}}
.timestamp {{
  margin-top: 8px;
  color: var(--discord-text-muted);
  font-size: 12px;
  font-weight: 500;
  line-height: 16px;
}}
.controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  max-width: 516px;
  margin-top: 4px;
}}
.component-button, .select {{
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: auto;
  min-width: 60px;
  min-height: 32px;
  height: 32px;
  margin: 0;
  padding: 3px 11px;
  border: 1px solid var(--discord-border-subtle);
  border-radius: 8px;
  color: #fbfbfb;
  background: var(--discord-background-mod-subtle);
  font: inherit;
  font-size: 14px;
  font-weight: 500;
  line-height: 18px;
  white-space: nowrap;
}}
.component-button.primary {{
  border-color: var(--discord-control-border);
  background: var(--discord-control-primary);
}}
.component-button.success {{
  border-color: var(--discord-control-border);
  background: var(--discord-control-connected);
}}
.component-button.danger {{
  border-color: var(--discord-control-border);
  background: var(--discord-control-critical);
}}
.component-button:disabled {{ opacity: .45; }}
.select {{
  justify-content: space-between;
  width: min(100%, 516px);
  min-height: 40px;
  height: 40px;
  padding: 8px 12px;
}}
.interaction-log {{
  min-height: 18px;
  margin-top: 8px;
  color: var(--discord-text-muted);
  font-size: 12px;
  line-height: 18px;
}}
@media (max-width: 760px) {{
  .simulator {{ grid-template-columns: 1fr; }}
  .simulator-sidebar {{ border-right: 0; border-bottom: 1px solid var(--discord-border-subtle); }}
  .case-switcher {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
  .case-button {{ text-align: center; }}
}}
@media (max-width: 520px) {{
  .message {{ padding-right: 16px; }}
}}
</style>
</head>
<body>
<main class="simulator">
  <aside class="simulator-sidebar">
    <h1>Discord UI simulator</h1>
    <p>Measured from the current Web client. No Discord gateway, webhook,
       token, or server send is used.</p>
    <nav class="case-switcher" aria-label="Audio panel states">
      {case_buttons}
    </nav>
    <section class="local-audio">
      <p class="muted">Local speech and music ducking</p>
    <div class="audio-card">
        <strong>Read aloud</strong>
      <audio controls preload="metadata" src="{html.escape(speech_filename)}"></audio>
    </div>
    <div class="audio-card">
        <strong>Music + ducking</strong>
      <audio controls preload="metadata" src="{html.escape(mixed_audio_filename)}"></audio>
    </div>
    </section>
  </aside>
  <section class="discord-window" aria-label="Discord message preview">
    <header class="channel-header"><span class="channel-hash">#</span>audio-lab</header>
    <div class="chat-scroller">
      {panel_markup}
    </div>
  </section>
</main>
<script>
(() => {{
  const cases = [...document.querySelectorAll(".case")];
  const selectors = [...document.querySelectorAll(".case-button")];
  const activate = (index) => {{
    cases.forEach((item, itemIndex) => item.classList.toggle("active", itemIndex === index));
    selectors.forEach((item, itemIndex) => {{
      item.classList.toggle("selected", itemIndex === index);
      item.setAttribute("aria-pressed", String(itemIndex === index));
    }});
  }};
  selectors.forEach((button, index) => button.addEventListener("click", () => activate(index)));
  document.querySelectorAll(".component-button").forEach((button) => {{
    button.addEventListener("click", () => {{
      const log = button.closest(".message")?.querySelector(".interaction-log");
      if (log) log.textContent = `${{button.textContent.trim()}} · simulated locally`;
    }});
  }});
  const updateRelativeTimes = () => {{
    const now = Date.now();
    document.querySelectorAll(".relative-time").forEach((time) => {{
      const deltaSeconds = Math.round((Number(time.dataset.unix) * 1000 - now) / 1000);
      const absolute = Math.abs(deltaSeconds);
      const amount = absolute < 60
        ? Math.max(1, absolute)
        : absolute < 3600
          ? Math.max(1, Math.round(absolute / 60))
          : Math.max(1, Math.round(absolute / 3600));
      const unit = absolute < 60 ? "秒" : absolute < 3600 ? "分" : "時間";
      time.textContent = deltaSeconds >= 0 ? `${{amount}}${{unit}}後` : `${{amount}}${{unit}}前`;
    }});
  }};
  updateRelativeTimes();
  window.setInterval(updateRelativeTimes, 1000);
  activate(0);
}})();
</script>
</body>
</html>
"""


async def build_offline_preview(
    output_dir: Path,
    *,
    speech_text: str,
    voice: str,
) -> dict[str, Any]:
    """Generate visual/audio artifacts without constructing a Discord client."""

    destination = output_dir.expanduser().resolve()
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    panels = _sample_panels()
    speech_path = await _synthesize_speech(destination, speech_text=speech_text, voice=voice)
    mixed_path = await _build_ducked_audio(destination, speech_path)
    html_path = destination / "index.html"
    manifest_path = destination / "manifest.json"
    html_path.write_text(
        render_preview_html(
            panels,
            speech_filename=speech_path.name,
            mixed_audio_filename=mixed_path.name,
        ),
        encoding="utf-8",
    )
    manifest = {
        "offline": True,
        "discord_send_count": 0,
        "discord_client_code_copied": False,
        "discord_client_reason": (
            "The installed bootstrapper and desktop core packages are UNLICENSED "
            "and do not contain a reusable embed renderer."
        ),
        "discord_style_calibration": {
            "observed_client": "Discord Web, 2026-07-28",
            "source_copied": False,
            "method": "computed styles and loaded CSSOM declarations",
            "embed_max_width_px": 516,
            "embed_grid_padding_px": [2, 16, 16, 12],
            "embed_title_typography_px": [16, 22],
            "embed_body_typography_px": [14, 18],
            "embed_footer_typography_px": [12, 16],
            "button_height_px": 32,
            "button_radius_px": 8,
            "button_padding_px": [3, 11],
            "message_padding_px": [2, 24, 2, 72],
            "avatar_size_px": 40,
        },
        "panels": [
            {
                "name": panel.name,
                "embed": panel.embed,
                "components": panel.components,
            }
            for panel in panels
        ],
        "speech": {
            "text": speech_text,
            "voice": voice,
            "path": speech_path.name,
        },
        "mixed_audio": mixed_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "html": str(html_path),
        "manifest": str(manifest_path),
        "speech": str(speech_path),
        "mixed_audio": str(mixed_path),
    }


def _sample_panels() -> tuple[PreviewPanel, ...]:
    cases = (
        (
            "Idle · disconnected",
            AudioQueueResponse(
                current=None,
                pending=(),
                paused=False,
                loop_mode="none",
                destination_id=None,
                auto_leave=True,
                position_seconds=0,
                speed=1,
                pitch=1,
                waiting_for_voice=False,
                autoplay_enabled=False,
                connected=False,
            ),
        ),
        (
            "Queued · waiting for voice",
            AudioQueueResponse(
                current=None,
                pending=(
                    AudioQueueItem(
                        title="Primary Colors",
                        page_url="https://example.invalid/primary-colors",
                        kind=AudioKind.MUSIC.value,
                        duration_seconds=249,
                        requested_by_name="Meteo",
                        uploader="PELICAN FANCLUB",
                    ),
                ),
                paused=False,
                loop_mode="none",
                destination_id=None,
                auto_leave=True,
                position_seconds=0,
                speed=1,
                pitch=1,
                waiting_for_voice=True,
                autoplay_enabled=True,
                connected=False,
            ),
        ),
        (
            "Playing · radio and read aloud",
            AudioQueueResponse(
                current=AudioQueueItem(
                    title="Primary Colors",
                    page_url="https://example.invalid/primary-colors",
                    kind=AudioKind.MUSIC.value,
                    duration_seconds=249,
                    requested_by_name="Meteo",
                    uploader="PELICAN FANCLUB",
                ),
                pending=(
                    AudioQueueItem(
                        title="Good Morning World!",
                        page_url="https://example.invalid/good-morning-world",
                        kind=AudioKind.MUSIC.value,
                        duration_seconds=249,
                        requested_by_name="Guest",
                        uploader="BURNOUT SYNDROMES",
                    ),
                ),
                paused=False,
                loop_mode="none",
                destination_id="1415260808103067671",
                auto_leave=True,
                position_seconds=65,
                speed=1,
                pitch=1,
                waiting_for_voice=False,
                autoplay_enabled=True,
                autoplay_next=AudioQueueItem(
                    title="Kaibutsu",
                    page_url="https://example.invalid/kaibutsu",
                    kind=AudioKind.MUSIC.value,
                    duration_seconds=206,
                    requested_by_name=None,
                    uploader="YOASOBI",
                    queue_lane="autoplay",
                ),
                connected=True,
            ),
        ),
    )
    panels: list[PreviewPanel] = []
    for name, response in cases:
        embed = music_queue_embed(response)
        # The runtime is never invoked during view construction or serialization.
        view = MusicControlsView(
            cast(SimajilordRuntime, object()),
            response=response,
        )
        panels.append(
            PreviewPanel(
                name=name,
                embed=cast(dict[str, Any], embed.to_dict()),
                components=serialize_view(view),
            )
        )
    quote_view = QuoteComposerView(
        cast(SimajilordRuntime, object()),
        requester_id=1,
        source_channel_id=2,
        source_message_id=3,
        destination_channel_id=2,
        has_animation=True,
    )
    panels.append(
        PreviewPanel(
            name="Quote · main menu",
            embed=cast(dict[str, Any], quote_view.embed().to_dict()),
            components=serialize_view(quote_view),
        )
    )
    panels.extend(
        (
            PreviewPanel(
                name="Help · overview",
                embed=cast(dict[str, Any], _help_overview_embed().to_dict()),
                components=serialize_view(HelpView(requester_id=1)),
            ),
            PreviewPanel(
                name="Help · /play",
                embed=cast(
                    dict[str, Any],
                    _help_entry_embed(HELP_ENTRIES_BY_TOPIC["play"]).to_dict(),
                ),
                components=serialize_view(HelpView(requester_id=1)),
            ),
            PreviewPanel(
                name="Server info · populated",
                embed=cast(
                    dict[str, Any],
                    server_info_embed(
                        DiscordServerResponse(
                            server_id="1415260807494766627",
                            name="Simajilord Audio Lab",
                            owner_id="1300000000000000000",
                            owner_name="Meteo",
                            member_count=128,
                            human_count=113,
                            bot_count=15,
                            text_channel_count=18,
                            voice_channel_count=5,
                            stage_channel_count=1,
                            forum_channel_count=2,
                            category_count=7,
                            role_count=22,
                            emoji_count=74,
                            sticker_count=12,
                            created_at_iso="2025-09-10T09:01:00+00:00",
                            icon_url="https://cdn.discordapp.com/embed/avatars/0.png",
                            description=(
                                "A deliberately long public description used to verify "
                                "wrapping, thumbnail spacing, and information density."
                            ),
                            boost_level=2,
                            boost_count=17,
                            preferred_locale="ja",
                            verification_level="medium",
                            explicit_content_filter="all_members",
                            features=(
                                "COMMUNITY",
                                "ANIMATED_ICON",
                                "BANNER",
                                "MEMBER_VERIFICATION_GATE_ENABLED",
                            ),
                        )
                    ).to_dict(),
                ),
                components=(),
            ),
            PreviewPanel(
                name="User info · populated",
                embed=cast(
                    dict[str, Any],
                    user_info_embed(
                        DiscordUserResponse(
                            user_id="1300000000000000000",
                            display_name="Meteo in Simajilord",
                            username="meteo_in_simajilord",
                            global_name="Meteo",
                            nickname="Meteo in Simajilord",
                            bot=False,
                            created_at_iso="2024-11-16T14:02:00+00:00",
                            joined_at_iso="2025-09-10T09:03:00+00:00",
                            top_role="Administrator",
                            avatar_url="https://cdn.discordapp.com/embed/avatars/1.png",
                            role_names=(
                                "Member",
                                "Audio tester",
                                "Developer",
                                "Administrator",
                            ),
                            role_count=4,
                            status="online",
                            key_permissions=(
                                "Administrator",
                                "Manage Server",
                                "Manage Channels",
                            ),
                            colour_value=0x57F287,
                        ),
                        mention="<@1300000000000000000>",
                    ).to_dict(),
                ),
                components=(),
            ),
        )
    )
    return tuple(panels)


async def _synthesize_speech(
    output_dir: Path,
    *,
    speech_text: str,
    voice: str,
) -> Path:
    service = SpeechService(
        MacOSSayProvider(voice),
        output_dir=output_dir / "speech-work",
        chunk_characters=180,
        max_concurrent=1,
        file_suffix=".aiff",
    )
    try:
        item = await service.synthesize(
            speech_text,
            title="Offline read aloud",
            workspace_id="offline-preview",
        )
        source = Path(item.source)
        destination = output_dir / "read-aloud.wav"
        await _run_process(
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ar",
            "48000",
            "-ac",
            "2",
            str(destination),
        )
        item.cleanup()
        return destination
    finally:
        await service.close()


async def _build_ducked_audio(output_dir: Path, speech_path: Path) -> Path:
    duration = await _audio_duration(speech_path)
    music_path = output_dir / "music-bed.wav"
    mixed_path = output_dir / "music-with-read-aloud.wav"
    total_duration = max(5.0, duration + 2.0)
    speech_end = duration + 1.0
    await _run_process(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=220:sample_rate=48000:duration={total_duration:.3f}",
        "-filter:a",
        "volume=0.08",
        "-ac",
        "2",
        str(music_path),
    )
    volume_expression = (
        f"if(lt(t,0.85),1,if(lt(t,{speech_end + 0.15:.3f}),0.22,1))"
    )
    await _run_process(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(music_path),
        "-i",
        str(speech_path),
        "-filter_complex",
        (
            f"[0:a]volume='{volume_expression}':eval=frame[music];"
            "[1:a]adelay=1000|1000[speech];"
            "[music][speech]amix=inputs=2:duration=longest:normalize=0[out]"
        ),
        "-map",
        "[out]",
        "-ar",
        "48000",
        "-ac",
        "2",
        str(mixed_path),
    )
    return mixed_path


async def _audio_duration(path: Path) -> float:
    stdout = await _run_process(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
        capture_stdout=True,
    )
    return max(0.1, float(stdout.strip()))


async def _run_process(
    executable: str,
    *arguments: str,
    capture_stdout: bool = False,
) -> str:
    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(f"{executable} is required for the offline preview.")
    process = await asyncio.create_subprocess_exec(
        resolved,
        *arguments,
        stdout=(asyncio.subprocess.PIPE if capture_stdout else asyncio.subprocess.DEVNULL),
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        detail = stderr.decode(errors="replace").strip()[:800]
        raise RuntimeError(f"{executable} failed: {detail or process.returncode}")
    return stdout.decode(errors="replace") if capture_stdout else ""


def _render_panel(panel: PreviewPanel) -> str:
    embed = panel.embed
    colour = f"#{int(embed.get('color', 0x5865F2)):06x}"
    title = html.escape(str(embed.get("title", "")))
    description = _discord_markdown(str(embed.get("description", "")))
    field_markup = "\n".join(
        (
            f'<div class="field{" wide" if not field.get("inline", True) else ""}">'
            f'<div class="field-name">{html.escape(str(field.get("name", "")))}</div>'
            f'<div class="field-value">{_discord_markdown(str(field.get("value", "")))}</div>'
            "</div>"
        )
        for field in embed.get("fields", ())
    )
    controls = "\n".join(_render_component(component) for component in panel.components)
    timestamp = html.escape(_discord_timestamp(str(embed.get("timestamp", ""))))
    thumbnail = embed.get("thumbnail")
    has_thumbnail = isinstance(thumbnail, dict) and bool(thumbnail.get("url"))
    thumbnail_markup = (
        '<div class="embed-thumbnail" aria-label="Embed thumbnail">S</div>'
        if has_thumbnail
        else ""
    )
    return f"""<section class="case" data-case-name="{html.escape(panel.name)}">
  <div class="message">
    <div class="avatar">S</div>
    <div class="message-body">
      <div class="message-header">
        <span class="author">SIMAJILORD</span><span class="bot">APP</span>
        <span class="message-time">13:47</span>
      </div>
      <div class="embed-container">
        <div class="embed" style="--embed-color:{colour}">
          <div class="embed-grid{" has-thumbnail" if has_thumbnail else ""}">
        {f'<div class="embed-title">{title}</div>' if title else ''}
        {f'<div class="description">{description}</div>' if description else ''}
        {f'<div class="fields">{field_markup}</div>' if field_markup else ''}
        {f'<div class="timestamp">{timestamp}</div>' if timestamp else ''}
        {thumbnail_markup}
          </div>
        </div>
      </div>
      <div class="controls">{controls}</div>
      <div class="interaction-log" aria-live="polite"></div>
    </div>
  </div>
</section>"""


def _render_component(component: dict[str, Any]) -> str:
    component_type = int(component.get("type", 0))
    if component_type == int(discord.ComponentType.button.value):
        style = {
            int(discord.ButtonStyle.primary.value): "primary",
            int(discord.ButtonStyle.success.value): "success",
            int(discord.ButtonStyle.danger.value): "danger",
        }.get(int(component.get("style", 0)), "")
        label = html.escape(str(component.get("label") or component.get("emoji") or "Button"))
        disabled = " disabled" if component.get("disabled") else ""
        return (
            f'<button class="component-button {style}" type="button"{disabled}>'
            f"{label}</button>"
        )
    if component_type in {
        int(discord.ComponentType.select.value),
        int(discord.ComponentType.user_select.value),
        int(discord.ComponentType.role_select.value),
        int(discord.ComponentType.mentionable_select.value),
        int(discord.ComponentType.channel_select.value),
    }:
        placeholder = html.escape(str(component.get("placeholder") or "Select"))
        return f'<div class="select">{placeholder}⌄</div>'
    return ""


def _discord_markdown(value: str) -> str:
    links: list[tuple[str, str]] = []

    def hold_link(match: re.Match[str]) -> str:
        links.append((match.group(1), match.group(2)))
        return _LINK_PLACEHOLDER.format(index=len(links) - 1)

    escaped = html.escape(_MARKDOWN_LINK.sub(hold_link, value))
    escaped = _BOLD.sub(r"<strong>\1</strong>", escaped)
    escaped = _INLINE_CODE.sub(r"<code>\1</code>", escaped)
    escaped = _CHANNEL.sub(r'<span class="channel">#voice-\1</span>', escaped)
    escaped = _TIMESTAMP.sub(_render_discord_timestamp_tag, escaped)
    for index, (label, url) in enumerate(links):
        anchor = (
            f'<a href="{html.escape(url, quote=True)}">'
            f"{html.escape(label)}</a>"
        )
        escaped = escaped.replace(
            html.escape(_LINK_PLACEHOLDER.format(index=index)),
            anchor,
        )
    lines = escaped.splitlines()
    rendered: list[str] = []
    for index, line in enumerate(lines):
        if index > 0 and not lines[index - 1].startswith("### "):
            rendered.append("<br>")
        if line.startswith("### "):
            rendered.append(f"<h3>{line[4:]}</h3>")
        else:
            rendered.append(line)
    return "".join(rendered)


def _render_discord_timestamp_tag(match: re.Match[str]) -> str:
    epoch = int(match.group(1))
    style = match.group(2) or "f"
    if style == "R":
        return f'<time class="relative-time" data-unix="{epoch}"></time>'
    moment = datetime.fromtimestamp(epoch).astimezone()
    formats = {
        "t": "%H:%M",
        "T": "%H:%M:%S",
        "d": "%Y/%m/%d",
        "D": "%Y年%m月%d日",
        "f": "%Y年%m月%d日 %H:%M",
        "F": "%Y年%m月%d日 %H:%M:%S",
    }
    return f"<time>{moment:{formats[style]}}</time>"


def _discord_timestamp(value: str) -> str:
    if not value:
        return ""
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone()
    except ValueError:
        return value
    return f"今日 {moment:%H:%M}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render Simajilord embeds, controls, read-aloud, and ducked audio "
            "without Discord network access."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".data/offline_discord_preview"),
    )
    parser.add_argument(
        "--speech",
        default="読み上げのローカルテストです。音楽の音量を下げて再生します。",
    )
    parser.add_argument("--voice", default=os.getenv("TTS_VOICE", "Kyoko"))
    arguments = parser.parse_args()
    output = asyncio.run(
        build_offline_preview(
            arguments.output_dir,
            speech_text=arguments.speech,
            voice=arguments.voice,
        )
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
