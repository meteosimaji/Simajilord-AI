# Simajilord AI

Simajilord AI is a local-first capability platform. The AI uses Discord; it is not an AI
feature embedded inside a Discord bot.

The capability platform runs independently of its optional language-model adapter. Useful,
auditable operations remain available when the model is disabled; human commands and the
agent call the same APIs.

## Architecture

The dependency direction is one-way:

```text
Discord adapter / local agent / human commands / future transports
                              |
                       capability registry
                              |
                    domain + services
                              |
                     provider boundaries
```

- `src/simajilord/capabilities` defines transport-neutral request and response APIs.
- `src/simajilord/domain` owns reusable models and policy state.
- `src/simajilord/services` owns media, speech, read-aloud, audio, and web orchestration.
- `src/simajilord/providers` and `src/simajilord/media/providers` isolate concrete providers.
- `src/simajilord/integrations/discord` converts Discord input into API requests and presents
  structured results back to Discord. AI progress delivery and Discord permission policy live
  in separate `agent_ui` and `permissions` modules rather than the command Cog.
- `src/simajilord/agent` holds model-independent event, conversation, context-budget,
  permission-grant, and action contracts.
- `vendor/yt-dlp` is a platform-owned upstream snapshot. Discord never imports it directly.

Conversation text is not hard-coded into capabilities. Only the Discord presenter owns fixed
operational feedback such as command success, validation, and permission errors. The agent
generates conversational text and chooses `discord.send_message` separately.
Agent replies match the depth of the request: concise wording removes filler, but substantive
questions still receive a direct answer, reasoning, context, and important limitations.
Explicit human commands use compact English embeds with useful result fields and a Discord
timestamp; implementation labels and decorative footer text are intentionally omitted.
Unexpected command errors show the same reference ID that is recorded in the host log, so a
user and an administrator can correlate one failed request without exposing internal details.
Every slash command, prefix command, button, select, and modal also has a final error boundary;
validation failures use stable machine codes and unexpected callback failures cannot end only
with Discord's generic “interaction failed” banner.

## Current Discord capabilities

- Health, uptime, and searchable capability discovery
- Per-server durable music queues with play, pause, resume, seek, tuning, separate
  music/read-aloud volume, shuffle, move, requester-only clearing, skip, stop, leave, loop,
  bounded per-server/per-user admission, and persistent button controls. A Discord 403 removes
  the denied dashboard binding from both memory and durable state so restart does not revive it
- Zero-click track search when one result is clear, with direct one-click choices only for
  genuinely ambiguous same-name tracks. A selection disables and updates the same visible
  result message while the track is being added
- Automatic stream re-resolution/retry, restart recovery, listener-aware reconnect, and
  queue-preserving auto-leave
- Globally bounded, priority-aware media work and guild-fair TTS work, with per-guild
  connection reservations, debounced durable audio state, and wait/duration metrics
- Voice-free queueing: add a track before joining, then start automatically when one of its
  requesters enters voice
- A silent, five-minute YouTube link card with direct `Play`, `Add`, and `Radio` actions;
  simply posting the link does not resolve media or change the queue
- Durable Radio mode fed one related track at a time, while manual requests always take
  priority
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
  replacement succeeds. The complete Embed text budget, including author, footer, title,
  description, field names, and field values, is kept within Discord's 6,000-character limit
- Discord audio/video attachments can be imported through `/play file:` or
  `Apps` → `Play Audio`. Files are probed before admission, stored under a private
  content-addressed local cache, and retained while a durable queue still references them
- On macOS 26 or newer, `/translate` and `Apps` → `Translate` use Apple's on-device
  Natural Language and Translation frameworks through a small local Swift helper. Message
  text is not sent to a cloud translation API
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
- Agent research uses Codex first-party live web search through the host's existing ChatGPT OAuth
  session when `AGENT_WEB_SEARCH_ACCESS` grants access. Local SearXNG remains available to
  explicit `/web` and prefix commands, but is not exposed to autonomous Codex turns
- Plain message sending and voice connection as independently invokable Discord APIs
- Permission-guarded agent audio playback/control, VOICEVOX speech, and read-aloud routing;
  third parties outside the active VC cannot control playback. Capability scope and
  per-turn write approval are separate, and the exact triggering message must be read before
  any approved write
- Bounded per-server FIFO AI-turn queues with durable conversation IDs, context-budget
  rotation, exact-message verification, progressive status updates, and corrective retries
  after failed writes. Separate servers can run turns concurrently without allowing two turns
  from the same server to overtake each other. Codex notifications and tool budgets are routed
  per thread/turn, so concurrent servers cannot consume each other's completion events.
  Mentions posted in the same channel while a turn is active are delivered with `turn/steer`
  as pointer-only follow-ups; the AI must fetch the exact Discord message, and actor ID/name
  remain attached so another user's read-only contribution is distinguishable. Queue and
  host execution progress uses short English status messages. The temporary `Working` embed is
  deleted before the final answer is posted as a new reply, preserving the order of visible
  milestone updates. Accepted-follow-up acknowledgements are tied to the parent turn and removed
  on either completion or failure. Multi-step work also posts task-specific progress in the
  conversation language, naming checked evidence and the next step without exposing private
  reasoning
- Restart-safe local image generation with atomic user/server/pending admission and a resident
  exponential-backoff Discord delivery retry loop. A completed image is not considered
  delivered until Discord accepts it, without waiting for the next service restart
- An optional read-only Now Playing Activity built with Discord's official Embedded App SDK.
  OAuth identity and same-VC membership are checked by the backend; the browser receives no
  stream URLs, authorization headers, local paths, or playback controls
- Autonomous turns retain bounded audio inspection and a provenance-locked message-repost API,
  but receive no arbitrary Discord-message write scope, audio-write approval,
  image-generation scope, or file-write scope
- Structured, append-only local command/capability/message events in SQLite

One independent audio session is created per Discord server. Different servers may use voice
concurrently up to the configured process limit; one server never owns multiple Discord voice
connections.

The optional agent is default-off. Its Codex runtime has browser control, shell execution,
personal-file access, plugins, sub-agents, and automatic browser-cookie extraction disabled.
The default runtime profile is `gpt-5.6-terra` with `medium` reasoning. Context protection is
rotation, not lossy in-place compression: when the configured turn or context-ratio limit is
reached, a new provider thread starts while durable audit records remain stored.

## Requirements

- A local VOICEVOX Engine installation, or macOS with the built-in `say` fallback
- macOS 26 or newer with Xcode Command Line Tools for optional on-device translation
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

### Optional Now Playing Activity

The Activity uses Discord's official Embedded App SDK and public
`LAUNCH_ACTIVITY` interaction response. The initial release is display-only.

Build the bundled frontend:

```bash
cd activity
npm ci
npm run build
cd ..
```

In the Discord Developer Portal, enable Activities, add the public HTTPS endpoint that maps
to `DISCORD_ACTIVITY_HOST:DISCORD_ACTIVITY_PORT`, and keep the OAuth client secret only in
the real `.env`:

```dotenv
DISCORD_ACTIVITY_ENABLED=true
DISCORD_CLIENT_SECRET=replace-in-local-env-only
DISCORD_ACTIVITY_HOST=127.0.0.1
DISCORD_ACTIVITY_PORT=8787
```

When enabled, `Open Player` appears in the `/audio` panel and uses discord.py's public
`launch_activity()` interaction response. A viewer must authenticate with Discord and be in
the same active voice channel. The Activity is intentionally display-only. See Discord's
[Activity tutorial](https://docs.discord.com/developers/activities/building-an-activity) and
[networking guide](https://docs.discord.com/developers/activities/development-guides/networking)
for URL Mapping and production-hosting requirements.

## Commands

- `/help`, `/status`
- `/audio` opens the shared music and read-aloud control panel
- `/play` adds a track and `/radio` starts or stops continuous related playback
- `/join` selects up to 25 conversations to read in the current VC
- `/timer` creates a persistent Focus Timer
- `/system ping`, `/system uptime`, `/system about`, `/system capabilities`
- `/readaloud ...` manages advanced read-aloud routes, voices, dictionaries, and exclusions
- `/web search`, `/web fetch`, `/web find`
- `/translate` translates supplied text or the latest visible message locally
- `/media download`, `/media detect-ai`
- `/info server`, `/info user`, `/info avatar`
- `/utility poll`, `/utility roll`, `/utility choose`
- Right-click or long-press a message, then choose `Apps` → `Play Audio`, `Translate`, or
  `Quote` for attachment playback, private translation, or local quote rendering

Playback controls such as pause, skip, loop, queue, levels, and leave are kept in the
`/audio` panel instead of being duplicated as a large slash-command group.

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

The code-quality commands above are the local gate; the web doctor additionally requires the
configured SearXNG endpoint. GitHub Actions runs lint, type checks, pytest, the audio doctor,
secret scanning, compile/wheel checks, macOS translation integration, and Activity build,
dependency-audit, desktop, and mobile checks on every push and pull request. Startup also
performs a real FFmpeg-to-Opus self-test before Discord commands become available.

For a provider regression, test the complete resolver/transcoder path without joining a VC:

```bash
uv run simajilord-audio-doctor "https://public.example/media/..."
```

This does not require a Discord token and never sends audio or messages to Discord.

The real Codex app-server and typed Discord message capabilities can be checked manually
without attaching the check to CI or consuming Discord API rate limits:

```bash
uv run python scripts/manual_agent_discord_qa.py
```

This one-shot test gives the AI an exact-message research task and verifies first-party Codex
web search, multiple concrete intermediate messages, and the final sourced answer. It consumes
one model turn and live search, and is intentionally not run on push.

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
