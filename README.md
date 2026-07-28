# Simajilord AI

Simajilord AI is a local-first capability platform. The AI uses Discord; it is not an AI
feature embedded inside a Discord bot.

The capability platform runs independently of its optional language-model adapter. Useful,
auditable operations remain available when the model is disabled; human commands and the
agent call the same APIs.

## Architecture

The dependency direction is one-way:

```text
local agent / human commands / future transports
                         |
                  capability registry
                         |
        domain + services + provider boundaries
                         |
       Discord adapter / future transport adapters
```

- `src/simajilord/capabilities` defines transport-neutral request and response APIs.
- `src/simajilord/domain` owns reusable models and policy state.
- `src/simajilord/services` owns media, speech, read-aloud, audio, and web orchestration.
- `src/simajilord/providers` and `src/simajilord/media/providers` isolate concrete providers.
- `src/simajilord/integrations/discord` converts Discord input into API requests and presents
  structured results back to Discord.
- `src/simajilord/agent` holds model-independent event, conversation, context-budget,
  permission-grant, and action contracts.
- `vendor/yt-dlp` is a platform-owned upstream snapshot. Discord never imports it directly.

Conversation text is not hard-coded into capabilities. Only the Discord presenter owns fixed
operational feedback such as command success, validation, and permission errors. The agent
generates conversational text and chooses `discord.send_message` separately.
Explicit human commands use compact, natural Japanese embeds with useful result fields and a
Discord timestamp; implementation labels and decorative footer text are intentionally omitted.

## Current Discord capabilities

- Health, uptime, and searchable capability discovery
- Per-server durable music queues with play, pause, resume, seek, tuning, separate
  music/read-aloud volume, shuffle, move, requester-only clearing, skip, stop, leave, loop,
  bounded per-server/per-user admission, and persistent button controls
- Zero-click track search when one result is clear, with direct one-click choices only for
  genuinely ambiguous same-name tracks
- Automatic stream re-resolution/retry, restart recovery, listener-aware reconnect, and
  queue-preserving auto-leave
- Globally bounded, priority-aware media work and guild-fair TTS work, with per-guild
  connection reservations, debounced durable audio state, and wait/duration metrics
- Voice-free queueing: add a track before joining, then start automatically when one of its
  requesters enters voice
- A silent, five-minute YouTube link card with direct `Play`, `Add`, and `Radio` actions;
  simply posting the link does not resolve media or change the queue
- Durable Radio mode fed one related track at a time, while manual requests always take
  priority; the old `mix` name remains a compatibility alias
- Durable requester attribution and a bounded recently-played history
- Automatic read-aloud from text channels, threads, and voice-channel chats with persistent
  many-conversations-to-one-voice routing
- Eager read-aloud voice preparation after `/join` and whenever a listener returns, avoiding
  a first-message connection delay
- Local VOICEVOX speech synthesis with BOT-owned engine startup/shutdown and a macOS
  `say` fallback; long messages are sentence-chunked and joined without truncation, while
  raw URLs and Discord mention markup are normalized before synthesis
- Speech-over-music sidechain ducking in the active Discord player, without restarting
  the music stream; standalone speech is used only as an overlay-failure fallback
- Opt-in VC-member-only read aloud, short-burst merging and spam suppression, plus durable
  server and user voice presets
- Restart-safe Focus Timers with retryable text delivery, optional VC-aware speech, and
  temporary read-aloud focus mode
- Bounded video/audio downloads across the vendored provider's built-in public-site extractors
- Server, user, and avatar information
- Native polls, dice, and bounded random choice
- Bounded Discord channel listing, reply-chain retrieval, and message-history reads—including
  voice-channel chat—for the agent without injecting entire messages into its initial context
- A bare Discord message link expands in place only after actor and BOT permission checks;
  the replacement preserves a Jump link and the original link post is deleted only after the
  replacement succeeds
- Custom emoji and sticker metadata stays text-only by default. The agent can request one
  selected asset as a preview, full GIF/APNG animation, or exact animation frame only when
  visual inspection is actually needed
- The `Quote` message action renders locally without an external image API. Static output is
  the default; animated custom emoji or stickers can be preserved as GIF on request. A known
  Discord CDN asset ID can be rendered even when the BOT is not a member of the asset's
  original server, but reading that server's message still requires normal Discord access.
  Its private composer groups controls under `Layout`, `Style`, and `More`, then exposes only
  `Generate` and `Cancel` on the main screen
- Local-first web Search / Fetch / Find with source diversity, readable HTML/PDF extraction,
  one-click chunk continuation, short-lived caching, and private-network/redirect blocking
- Plain message sending and voice connection as independently invokable Discord APIs
- Permission-guarded agent audio playback/control, VOICEVOX speech, and read-aloud routing;
  third parties outside the active VC cannot control playback. Capability scope and
  per-turn write approval are separate, and the exact triggering message must be read before
  any approved write
- A bounded FIFO AI-turn queue with durable conversation IDs, context-budget rotation,
  exact-message verification, progressive status updates, and corrective retries after
  failed writes
- Autonomous turns retain bounded audio inspection and a provenance-locked message-repost API,
  but receive no arbitrary Discord-message write scope, audio-write approval,
  image-generation scope, or file-write scope
- Structured, append-only local command/capability/message events in SQLite

One independent audio session is created per Discord server. Different servers may use voice
concurrently up to the configured process limit; one server never owns multiple Discord voice
connections.

The optional agent is default-off. Its Codex runtime has browser control, shell execution,
personal-file access, plugins, sub-agents, and automatic browser-cookie extraction disabled.

## Requirements

- A local VOICEVOX Engine installation, or macOS with the built-in `say` fallback
- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- [FFmpeg](https://ffmpeg.org/)
- A loopback [SearXNG](https://docs.searxng.org/) JSON endpoint for web search
- A Discord bot token

## Setup

```bash
cp .env.example .env
chmod 600 .env
uv sync --all-groups
uv run simajilord-discord
```

The real `.env`, local databases, generated speech, downloads, cookies, and `notes/` are
ignored by Git.

For private development, `COMMAND_SCOPE=guild` synchronizes commands to each connected server.
It also removes stale global commands for the same application so users do not see old and new
commands together. Use `global` only when global publication is intended. Automatic read-aloud
requires the Discord Message Content intent. Presence and Server Members intents are not used.

Set `TTS_PROVIDER=voicevox`, `VOICEVOX_SPEAKER_ID` to a VOICEVOX style ID, and
`VOICEVOX_ENGINE_PATH` to the local engine executable. The provider only accepts a loopback
HTTP endpoint. With `VOICEVOX_AUTO_START=true`, the BOT starts the engine on first speech and
stops only the process it owns during clean shutdown.

An optional Netscape cookie file may be configured with `MEDIA_COOKIE_FILE`. It must be an
existing local file with mode `0600`. The platform never extracts cookies from a browser.

## Commands

- `/ping`, `/uptime`, `/about`, `/capabilities`
- `/audio` opens the shared music and read-aloud control panel
- `/play`, `/queue`, `/nowplaying`, `/history` for the shortest everyday paths
- `/radio` starts continuous related playback; `/mix` remains a compatibility alias
- `/music play`, `/music queue`, `/music history`, `/music pause`, `/music resume`
- `/music skip`, `/music stop`, `/music leave`, `/music loop`
- `/music remove`, `/music move`, `/music clear-mine`, `/music shuffle`
- `/music seek`, `/music tune`, `/music volume`, `/music autoleave`
- Right-click or long-press a message, then choose `Apps` → `Quote` for local quote rendering
- `/join` to select up to 25 conversations and read them in your current VC
- `/timer` creates, lists, or cancels a persistent Focus Timer
- `/readaloud setup`, `/readaloud status`, `/readaloud remove`, `/readaloud disable` for
  advanced route management
- `/download`
- `/search`, `/fetch`, `/find`
- `/serverinfo`, `/userinfo`, `/avatar`, `/poll`
- `/roll`, `/choose`

The configurable prefix remains available as a manual adapter. Its operations and slash
commands call the same capability APIs.

## Local event journal

`.data/events.sqlite3` records Discord command receipt, capability outcomes, and new Discord
messages. Each row has a monotonic sequence cursor, actor, workspace, transport, request ID,
and structured payload. Sensitive field names such as token, password, secret, authorization,
and cookie are redacted before storage. The agent can read only the rows after its last
cursor and reconcile user-driven state changes before acting.

`.data/audio_sessions.json` stores only stable media page references, requester attribution,
recent history, and queue policy. Signed stream URLs are deliberately excluded. A restart
reloads the queue, resolves each stream again immediately before playback, and reconnects only
when an eligible human listener is present.

## Verification

```bash
uv run ruff check src tests
uv run mypy src/simajilord
uv run pytest
uv run simajilord-audio-doctor
uv run simajilord-web-doctor "Python"
uv run python -m compileall -q src
```

The same gate runs automatically on every GitHub push and pull request. Startup also performs
a real FFmpeg-to-Opus self-test before Discord commands become available.

For a provider regression, test the complete resolver/transcoder path without joining a VC:

```bash
uv run simajilord-audio-doctor "https://public.example/media/..."
```

This does not require a Discord token and never sends audio or messages to Discord.

To inspect Discord embeds, adaptive buttons, local read-aloud, and music ducking without
connecting a Discord client:

```bash
uv run simajilord-offline-preview
python3 -m http.server 8765 --bind 127.0.0.1 \
  --directory .data/offline_discord_preview
```

Open `http://127.0.0.1:8765`. The simulator renders the same `discord.Embed` dictionaries and
component payloads used by the adapter. Its independent CSS is calibrated from the current
Discord Web client's loaded CSS declarations and computed styles; no Discord client source,
token, gateway, webhook, or server send is used. Discord can change its client styling, so
the dated calibration recorded in `manifest.json` should be refreshed when visual drift is
observed.

Third-party license information is in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
