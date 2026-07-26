# Simajilord AI

Simajilord AI is a local-first capability platform. The AI uses Discord; it is not an AI
feature embedded inside a Discord bot.

The current milestone deliberately runs without a language model. It first provides useful,
auditable capabilities that a future local agent can discover and invoke directly. Human
slash commands are another caller of the same APIs.

## Architecture

The dependency direction is one-way:

```text
future local agent / human commands / future transports
                         |
                  capability registry
                         |
        domain + services + provider boundaries
                         |
       Discord adapter / future transport adapters
```

- `src/simajilord/capabilities` defines transport-neutral request and response APIs.
- `src/simajilord/domain` owns reusable models and policy state.
- `src/simajilord/services` owns media, speech, read-aloud, and audio orchestration.
- `src/simajilord/providers` and `src/simajilord/media/providers` isolate concrete providers.
- `src/simajilord/integrations/discord` converts Discord input into API requests and presents
  structured results back to Discord.
- `src/simajilord/agent` holds model-independent contracts for future goals, events, context
  compaction, and action proposals.
- `vendor/yt-dlp` is a platform-owned upstream snapshot. Discord never imports it directly.

Conversation text is not hard-coded into capabilities. Only the Discord presenter owns fixed
operational feedback such as command success, validation, and permission errors. A future
agent will generate conversational text and choose `discord.send_message` separately.
Explicit human commands use compact English embeds with useful result fields and a Discord
timestamp; implementation labels and decorative footer text are intentionally omitted.

## Current Discord capabilities

- Health, uptime, and searchable capability discovery
- Per-server music queues with structured queue inspection, play, pause, resume, skip, stop,
  leave, and loop controls
- Automatic channel read-aloud with persistent text-to-voice routing
- Local, offline speech synthesis through the macOS `say` executable
- Bounded YouTube and TikTok video/audio downloads
- Server, user, and avatar information
- Native polls, dice, and bounded random choice
- Bounded Discord channel listing and message-history reads for a future agent
- Plain message sending and voice connection as independently invokable Discord APIs
- Structured, append-only local command/capability/message events in SQLite

One independent audio session is created per Discord server. Different servers may use voice
concurrently up to the configured process limit; one server never owns multiple Discord voice
connections.

No AI model, autonomous loop, browser control, shell execution, administrator-role command,
or automatic browser-cookie extraction is enabled in this milestone.

## Requirements

- macOS with the built-in `say` executable
- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- [FFmpeg](https://ffmpeg.org/)
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

An optional Netscape cookie file may be configured with `MEDIA_COOKIE_FILE`. It must be an
existing local file with mode `0600`. The platform never extracts cookies from a browser.

## Commands

- `/ping`, `/uptime`, `/about`, `/capabilities`
- `/music play`, `/music queue`, `/music pause`, `/music resume`
- `/music skip`, `/music stop`, `/music leave`, `/music loop`
- `/readaloud setup`, `/readaloud status`, `/readaloud disable`
- `/download`
- `/serverinfo`, `/userinfo`, `/avatar`, `/poll`
- `/roll`, `/choose`

The configurable prefix is retained for future manual adapters. Every implementation must call
the same capability API as its slash-command equivalent.

## Local event journal

`.data/events.sqlite3` records Discord command receipt, capability outcomes, and new Discord
messages. Each row has a monotonic sequence cursor, actor, workspace, transport, request ID,
and structured payload. Sensitive field names such as token, password, secret, authorization,
and cookie are redacted before storage. A future agent can read only the rows after its last
cursor and reconcile user-driven state changes before acting.

## Verification

```bash
uv run ruff check src tests
uv run mypy src/simajilord
uv run pytest
uv run python -m compileall -q src
```

Third-party license information is in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
