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
Before every turn and accepted follow-up, the host declares the exact total response-character
budget and Discord's safe per-message boundary. The agent chooses whether one host reply, semantic
multi-message breaks, a plain post, selected-message reply, authorized channel, embed, file, DM,
or VC speech best fits the task. Transport-side splitting remains only a final Discord safety net;
an over-budget provider response is logged explicitly instead of being an invisible truncation.
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
- The Now Playing panel reports Discord's live relative end time for an actively playing,
  non-track-looping item (for example, `1分後`) and recalculates it after seek, resume, or speed
  changes; paused, disconnected, waiting, and per-track loop states keep invariant timing text
- Zero-click track search when one result is clear, with direct one-click choices only for
  genuinely ambiguous same-name tracks. A selection disables and updates the same visible
  result message while the track is being added
- Automatic stream re-resolution/retry, restart recovery, listener-aware reconnect, and
  queue-preserving auto-leave
- Globally bounded, priority-aware media work and guild-fair TTS work, with per-guild
  connection reservations, debounced durable audio state, and wait/duration metrics
- Voice-free queueing: add a track before joining, then start automatically when one of its
  requesters enters voice
- There is no YouTube URL listener or generic `Play` / `Add` / `Radio` link card. A bare URL is
  ignored by the BOT; `!play URL` runs only the normal explicit play flow and does not receive a
  second fixed card
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
- Staged Discord research without injecting whole histories into the initial context:
  `discord.list_servers` pages through BOT servers and live-checks any uncached requester
  membership while reporting incomplete lookups instead of silently treating them as absence;
  `discord.list_channels` pages through the channels currently readable by both principals,
  `discord.search_messages` searches authorized channels by phrase/author/message-ID or ISO time
  range in pages of at most 25, and `discord.get_message` retrieves a selected original in
  chunks. A directly supplied server ID is checked live too. `discord.read_messages` supplies
  chronological pages for sampling—including voice-channel chat. Results expose stable
  server/channel/message IDs, explicit completeness, continuation cursors, and
  reaction/reply/thread signals so the agent can continue a long search without guessing
- Every explicit AI turn records a typed semantic evidence plan after reading the exact active
  message. The AI—not a host keyword list—decides whether it needs earlier channel conversation
  or current Simajilord source. Earlier messages are fetched only when required, anchored before
  the active message, and remain interpretation evidence rather than additional requests to
  answer. Current implementation questions use bounded, read-only `source.search` /
  `source.read` access to allowlisted package, Activity, native-helper and script source plus
  tests, workflows, and top-level docs; generated static assets, vendor code, runtime data,
  `.env`, secrets, symlinks, and arbitrary host paths are excluded. A required evidence source
  must actually be read before a host reply or write can be finalized
- Read-only Discord research may cross to another shared server when both the active requester
  and BOT still have View Channel and Read Message History access (and private-thread membership
  where applicable). Each result includes a time-local source-visibility/disclosure advisory.
  `broader` flags a currently known audience expansion and `uncertain` means the member cache
  cannot prove the complete audience; neither is a mechanical disclosure block or a source of
  write authority. A write to any explicitly selected shared server still requires fresh
  requester and BOT permissions there; historical reads never grant write authority. When an
  agent supplies a globally unique cached channel ID but omits its server ID, the write resolver
  may infer that shared server and then repeats the same live membership and permission checks;
  an explicit mismatched server ID is never silently corrected
- The Discord adapter currently registers 106 typed capability endpoints covering the bot-visible
  conversation, moderation, audio, and common server-management areas listed below. This is a
  broad capability catalog, not a one-to-one implementation of every route in Discord's official
  REST API: six endpoints are internal compatibility/Undo helpers, and file or synthetic-media
  capabilities are exposed only when their provider is configured. Application-command
  deployment, bot identity/lifecycle changes, credential or token disclosure, OAuth-only private
  user data, webhook-token execution, member pruning, integration deletion, incident mutation,
  and test-commerce mutation remain host-managed or intentionally unavailable to the model
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
- A quota-bound file workspace per Discord server: the agent can import Discord attachments,
  download a bounded public document through the same SSRF-safe fetcher, inspect PDF/ZIP/text
  content in chunks, edit text, and send a selected result back to Discord. PDF reads select
  1–20 pages at a time and return `total_pages` plus `next_page`; character-level `next_offset`
  continues within the selected page range, so documents longer than 20 pages remain readable
- Optional `AGENT_SAFE_COMPUTE_ACCESS` runs only an argv-based Python script that the agent first
  stores in that server workspace. On macOS, Seatbelt blocks network access, child processes,
  personal/project host files, and writes outside a temporary staged workspace; wall time, CPU,
  resident memory, output, open-file, per-file, file-count, and workspace-byte limits are
  host-enforced.
  Validated output files are committed through the existing file quota, while failed or
  over-limit runs discard the staging directory. No general shell capability is exposed
- Optional `AGENT_ISOLATED_SHELL_ACCESS` is a separate, higher-risk host capability for explicit
  mention turns only. It is disabled by default, can be scoped to administrators, requires
  per-event write approval, and uses a Seatbelt-confined workspace isolated by Discord actor and
  durable task. Autonomous turns never receive this grant
- Optional `AGENT_CONNECTOR_ACCESS` keeps Figma, Canva, Adobe, Adobe Express, and BioRender usable
  through Simajilord's existing capability, authorization, effect-ledger, and audit boundary.
  Native model-facing Codex Apps and plugin MCP servers remain disabled. The broker reads the live
  tool inventory for every operation, omits unknown/unclassified tools, separates read and write
  endpoints, and requires a current actor/request/thread/schema-bound contract; connector writes
  are approved and receipted as non-idempotent external effects
- Optional `AGENT_CURATED_SKILLS_ENABLED` adds a small package-owned `workflow.search`
  catalog for community research, long web/PDF reading, file plus safe-compute transforms,
  generic media saving, selective memory, and Action Receipt/Undo flows. These are typed
  Simajilord-tool recipes, not Codex user/global skills: no plugin, MCP, shell, or host path
  is loaded, and workflows requiring unavailable capabilities or grants are omitted
- At most five full eager schemas are shown at turn start: exact-message read, bounded nearby
  message read, the semantic evidence plan, selective memory search, and (when granted) local web
  search. General ability
  questions use compact cursor-paged `capability_list`, modeled after MCP `tools/list`; concrete
  goals use `capability_search` for short ranked summaries plus a complete, compact index of all
  capabilities available in that turn. The AI selects from that index by meaning, loads only one
  exact contract with `capability_describe`, and copies its opaque contract ID into
  `capability_invoke`; guessed contracts and IDs from another request are rejected. A semantic
  no-match conclusion is recorded against the request-bound catalog ID with
  `capability_resolution`.
  A concrete search left without either selection or resolution receives one bounded corrective
  turn. File, workflow, image, audio, Discord mutation, and the rest stay deferred regardless of
  how many grants the caller has. Natural-language command strings and per-platform URL trigger
  tables are not the capability router
- A plan that requires earlier Discord context is provisional. After the bounded history read
  anchored to the active message, the host invalidates that plan and any discovery contract, then
  requires a fresh AI semantic plan before capability browsing. History stays selective while a
  pre-context plan cannot ground a current ability claim
- `AGENT_WEB_SEARCH_ACCESS` exposes both Codex first-party live search through the host's existing
  ChatGPT OAuth session and Simajilord's local `web.search`, `web.fetch`, and `web.find` tools.
  This lets the agent discover sources, continue through long HTML/PDF text, locate a passage,
  and locally fetch a public URL that first-party search could not open. The same grant policy
  applies to autonomous turns; explicit `/web` and prefix commands remain available independently
- The 106 registered Discord endpoints span server/member Presence and activities, live VC state,
  channels/threads/pins/reactions/poll voters, effective permissions and overwrites,
  audit/bans/invites/events/AutoMod, emojis/stickers/soundboard, token-free webhooks,
  templates/integrations/onboarding/widget/application metadata, message/file/embed delivery,
  global and per-server voice regions, non-mutating prune estimates, SKUs/entitlements/
  subscriptions, application role-connection metadata, voice-channel status, moderation,
  resource mutation, and audio. Active threads and guild preview data are fetched from REST
  rather than assumed from cache. The count describes Simajilord's typed abstractions, some of
  which intentionally group several Discord routes; it does not claim complete official Discord
  API coverage
- Plain message sending and voice connection as independently invokable Discord APIs
- Permission-guarded agent audio playback/control, VOICEVOX speech, and read-aloud routing;
  third parties outside the active VC cannot control playback. Capability scope and
  per-turn write approval are separate, and the exact triggering message must be read before
  any approved write
- Agent actions include intentional reactions, own-message reply/edit/delete, pinning, polls,
  threads and forum posts, roles, channel settings/creation, audio and read-aloud controls, and
  permission-checked moderation. Destructive actions require the explicit capability policy,
  the active contributor's permissions, a reason/evidence where applicable, and a journaled
  result; retrieving an old administrator message can never authorize a new action
- Bounded per-server FIFO AI-turn queues with durable conversation IDs, Codex-native retained
  context compaction, exact-message verification, progressive status updates, and corrective
  retries after failed writes. Native `contextCompaction` lifecycle events renew the inactivity
  watchdog, update the temporary progress message, and are recorded in the body-free agent trace.
  The host does not discard a healthy provider thread at a fixed turn count or token percentage.
  Separate servers can run turns concurrently
  without allowing two
  turns from the same server to overtake each other. Codex notifications, inactivity watchdogs,
  and tool budgets are routed per thread/turn, so concurrent servers cannot consume each other's
  completion events.
  A mention posted in the same channel while a turn is active is persisted first as an independent
  task candidate, then delivered with `turn/steer` as a pointer only. It is not authoritative until
  the AI fetches the exact current Discord revision and calls the typed `turn.route_task_event`
  capability with `attach`, `separate`, `finish`, or `cancel`. The host never classifies message
  text with keyword or fixed-phrase rules—including natural corrections such as “やっぱりいい”.
  Edits and resends retain independent event IDs; a superseded revision cannot
  silently reuse its earlier authorization. `attach` keeps the contributor's own grants and live
  Discord permissions, every task receives its own provider-continuity key, and `finish`
  lets the AI conclude when a typo correction or resend adds no work. An authorized `cancel`
  interrupts the unfinished original model turn without pretending to undo already-confirmed
  external effects. Missing, timed-out, or crash-interrupted decisions default to a recoverable
  separate task, so no mention disappears.
  Each candidate receives a bounded evidence allowance for its exact read, route decision, and
  any required refreshed semantic evidence plan. Rate limits are checked before admission, while
  `MAX_ACTIVE_AGENT_TURNS`, `MAX_PENDING_AGENT_TURNS`, and
  `MAX_PENDING_AGENT_TURNS_PER_USER` independently bound active provider work, waiting work,
  and one actor's share of waiting work. `AGENT_INTERACTIVE_RESERVE_PERCENT` (25 by default)
  prevents autonomous turns from consuming the final active, pending, and rolling-token capacity
  reserved for explicit mentions. Active turns do not consume either pending allowance,
  so the main queue can hold up to the active limit plus the waiting limit. Queue and host
  execution progress uses short English status messages. The temporary `Working` embed contains
  only the current progress and active duration—no internal task/reference IDs, quota diagnostics,
  routing manual, or Cancel button. Accepted follow-ups do not add acknowledgement embeds. The AI
  can post useful progress directly; task, route, receipt, delivery, and audit evidence remains in
  the local ledgers instead of adding short-lived inspection commands or panels. The Working
  message is deleted before the
  final answer is posted as a new reply, preserving the order of visible
  milestone updates. Discord typing is renewed only while a model notification or
  an actually running capability keeps the activity lease alive; a quiet app-server does not
  leave the BOT typing indefinitely. Multi-step work also posts task-specific progress in the
  conversation language, naming checked evidence and the next step without exposing private
  reasoning
- The AI chooses its final Discord delivery instead of being forced into the triggering
  mention's reply. The normal default remains a reply, but it may instead use a plain post in an
  authorized channel, an embed, a reply to another selected message, one or more attachments, a
  DM, VC speech, or deliberate silence. A successful tool-owned final delivery suppresses the
  host fallback exactly once; a rejected or failed delivery leaves the normal durable host reply
  available, so an attempted alternative cannot lose the answer
- Every successful agent write returns an Action Receipt whose `tracked` field states whether
  the local ledger commit also succeeded. A ledger failure returns `tracked=false`,
  `action_id=null`, and no Undo claim instead of inventing an ID; the external write remains
  succeeded and is never blindly repeated. Reactions, pin state, role membership,
  timeout state, thread membership/settings, channel topic/slowmode, audio volume, selected
  read-aloud settings, and Focus Timer create/cancel operations have restart-safe inverse or
  compensating actions; newly created BOT messages can be deleted. `action.undo` accepts a
  receipt ID or resolves the same actor's most recent undoable action; repeating the same Undo
  does not execute the inverse twice. Destructive operations or writes that would require
  retaining deleted content/file bodies are explicitly receipted as non-undoable.
  Final replies and autonomous host posts are also receipted after each Discord send using only
  channel/message IDs, even though they bypass the model tool catalog. If ledger persistence
  fails, their already-sent delivery evidence stays pending and only receipt persistence is
  retried; the Discord message is not posted again. Recovery uses a known message ID directly;
  only a crash before that ID is saved falls back to nonce reconciliation, and one recovery pass
  shares that bounded history read across pending deliveries in the same channel. A
  single-source autonomous reply is attributed to that source actor for natural same-actor
  Undo; a mixed-source batch remains BOT-owned instead of arbitrarily granting one member
  control over the post. If a final confirmation follows a substantive write in the same turn,
  ID-less Undo prefers that write; a reply-only turn instead removes its latest host post.
  The bounded `.data/agent_actions.sqlite3` ledger retains at most 2,000 records, at most 100 per
  actor, for seven days. It stores IDs and a small scalar inverse (maximum 4 KiB), never a file
  body, deleted-message body, or large snapshot. The same database places every provider write
  behind a body-free `planned → dispatched → confirmed|unknown → reconciled` effect ledger.
  A process restart converts an unconfirmed dispatch to `unknown`; interrupted mention recovery
  then refuses to replay that whole model turn, instead of risking a duplicate external action.
- Durable agent memory is independent of Codex provider threads in
  `.data/agent_memory.sqlite3`. Typed `user`, `channel`, `workspace`, and verified-success/failure
  `procedure` scopes enforce same-user/same-channel/same-server visibility. Records contain only a
  short summary, exact source Discord message IDs, confidence, timestamps, and optional expiry;
  message bodies, attachments, secret-like values, inferred profiles, and low-confidence guesses
  are rejected. `memory.search`, `memory.remember`, and `memory.update` tools let the agent
  retrieve relevant context and capture at most one reusable outcome after substantive work
  without recording every turn. Recent rows are not injected into every prompt: the agent first
  reads the exact active Discord event and searches memory only when that task needs it. Search
  uses a bounded bilingual lexical matcher tolerant of
  case/width/punctuation and common English/CJK wording variants, with scope, evidence basis,
  minimum-confidence, update-time, and `next_offset` filters; an empty query returns the most
  recently used accessible records. Normalized keys upsert duplicates, search updates
  `last_used_at`, and expiry plus total/workspace/user/channel/procedure caps keep the database
  bounded. Memory writes still require the exact authorizing event and return non-undoable
  Action Receipts; forgetting is intentionally irreversible. Provider conversation continuity
  and this distilled memory are separate, so resetting provider continuity does not turn raw
  chat history into durable memory.
- Event-driven autonomy stores content-free message/edit/reaction/thread/voice/timer/audio
  pointers in `.data/agent_autonomy.sqlite3`. The first event opens one fixed 5–15 second
  same-channel batch window
  and each turn receives the oldest candidates up to `AGENT_AUTONOMY_CANDIDATE_LIMIT`; excess
  events stay queued for a later turn instead of being skipped. `AGENT_AUTONOMY_MAX_RUNS=0`
  means no artificial run-count cutoff. `observe` suppresses spontaneous turns, `assist` can
  answer, react, and manage timers, and `act` can use all otherwise granted write capabilities
  subject to live Discord permissions and the same admission/rate limits. The autonomous
  principal is the BOT itself, never the user who produced an event; source actors in the batch
  provide context but not borrowed authority.
  Global, per-channel, and per-human-source queue caps, per-server autonomous turn budgets,
  stable deduplication keys, leased fixed-membership batches, and durable exponential retry
  timestamps prevent gateway bursts from bypassing admission or disappearing on restart.
  Timer/audio system events do not consume the human-source cap. No built-in GitHub or RSS
  adapter is connected yet; a future adapter can publish through the same typed
  `AutonomyEventQueue` without adding a second scheduler
- Restart-safe image generation through the saved Codex login, with atomic user/server/pending
  admission. Agent requests wait in the same turn and receive a bounded model-visible preview
  plus the full server-scoped workspace file. Generation and publication stay independent: the
  model may inspect, compare, describe, or iterate privately, and uses `discord.send_file` only
  when the user asks to publish. Legacy automatic-delivery jobs reuse one durable Discord
  message and a hidden deterministic nonce for crash-safe reconciliation
- An optional read-only Now Playing Activity built with Discord's official Embedded App SDK.
  OAuth identity and same-VC membership are checked by the backend; the browser receives no
  stream URLs, authorization headers, local paths, or playback controls
- Structured, append-only local command/capability/message events in SQLite

One independent audio session is created per Discord server. Different servers may use voice
concurrently up to the configured process limit; one server never owns multiple Discord voice
connections.

The optional agent and event autonomy are both default-off. Event autonomy is inert until the agent
is enabled, `AGENT_AUTONOMY_ENABLED=true`, and
`AGENT_AUTONOMY_GUILD_IDS` contains an allowed server, even though its mode defaults to `act`
and `AGENT_AUTONOMY_MAX_RUNS=0` has no artificial run-count cutoff. Its native Codex runtime has
browser control, model-facing Apps, plugin MCP servers, internal shell execution, personal-file
access, remote plugin installation, sub-agents, and automatic browser-cookie extraction disabled.
Optional host-brokered shell and design connector grants are independent of those native paths and
are never given to autonomous turns. The default runtime profile is `gpt-5.6-luna` with `high`
reasoning. Codex keeps one durable
provider thread per task in its stable `legacy` history mode and compacts retained context natively; the
host automatically resets it only when the saved thread is genuinely unavailable. An explicit
`AGENT_CONVERSATION_COMPATIBILITY_EPOCH` is persisted with the local store; operators bump it
only for an intentionally incompatible prompt/tool/permission protocol, which clears only saved
provider continuity while preserving tasks, deliveries, receipts, memory, and audit evidence.
The current single-broker authority and actor/task workspace protocol uses epoch `5`.
Model or capability-list changes do not silently fingerprint-reset every conversation. Existing
conversations are never proactively rotated at a fixed host threshold. Agent turns have no wall-clock
deadline. A turn is stopped only after its configured inactivity window, while a running
capability receives its own declared timeout. Conversation and image generation share one
app-server rather than a second image runtime. Its JSONL reader accepts bounded large image
notifications, reports line size, reader/process state, active tools, recent activity, and
sanitized stderr on failure, and wakes the affected turn immediately if the transport dies
instead of waiting for the inactivity window. Discord link previews are suppressed for
AI-authored text without altering its URLs.

As an offline last resort after stopping the bot (for example, after a broken provider/tool
migration), reset only provider continuity and keep request, delivery, audit, memory, feedback,
and action evidence:

```bash
uv run simajilord-agent-reset --all --yes
```

The command creates a timestamped SQLite backup before clearing provider thread IDs and context
counters. Use repeated `--conversation ID` arguments instead of `--all` for a selective reset.

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
commands together. Each scope has a durable canonical manifest hash: an unchanged restart makes
no Discord command API write, while a changed local manifest is compared with the fetched remote
manifest and only a real difference is synchronized. Use `global` only when global publication is
intended. Message Content is the
documented minimum and is always requested because mentions, prefix commands, message inspection,
and automatic read-aloud need bodies. Server Members and Presence are opt-in with
`DISCORD_MEMBERS_INTENT_ENABLED` and `DISCORD_PRESENCE_INTENT_ENABLED`; enable each matching
Developer Portal toggle before setting it true. The adapter does not request disabled privileged
intents, because Discord closes a Gateway connection with code 4014 when a requested privileged
intent is not enabled or approved. Presence/activity and complete cached member inspection require
the corresponding opt-ins; normal non-privileged guild, channel, message, reaction, and VC events
remain enabled.

Set `TTS_PROVIDER=voicevox`, `VOICEVOX_SPEAKER_ID` to a VOICEVOX style ID, and
`VOICEVOX_ENGINE_PATH` to the local engine executable. The provider only accepts a loopback
HTTP endpoint. With `VOICEVOX_AUTO_START=true`, the BOT starts the engine on first speech and
stops only the process it owns during clean shutdown.

For on-device macOS translation, a source checkout can build
`native/macos/TranslationHelper` lazily with Swift. The platform wheel deliberately remains
portable and does not contain a Mach-O executable. When running from an installed wheel, build
the helper separately, keep it private and executable, and configure its absolute path:

```bash
swift build --package-path native/macos/TranslationHelper -c release
chmod 700 native/macos/TranslationHelper/.build/release/TranslationHelper
export TRANSLATION_HELPER_PATH="$PWD/native/macos/TranslationHelper/.build/release/TranslationHelper"
simajilord-translation-doctor
```

An installed wheel with no `TRANSLATION_HELPER_PATH` reports
`translation.helper_missing` instead of guessing a nonexistent source-tree path.

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
- `/feedback` opens a private Modal and saves the report locally without asking for a triage kind
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

`.data/events.sqlite3` records Discord command receipt, capability outcomes, and Discord/agent
events. Each row has a monotonic sequence, actor, workspace, transport, request ID, and
structured payload. Sensitive field names such as token, password, secret, authorization, and
cookie are redacted before storage. Capability outcomes enter a bounded audit queue and a
background writer commits them in batches, so unrelated capability results do not wait for one
SQLite commit or a process-wide read lock. Explicit transport events remain durable before their
call returns; history reads, retention, and clean shutdown flush queued audits. `/status` reads a
single-row operation projection and the committed cursor without scanning or decoding the event
history; the `(kind, sequence)` index and projection are created/backfilled during migration and
rebuilt after retention pruning. Autonomous delivery state is separate in
`.data/agent_autonomy.sqlite3`, so advancing an observability cursor cannot discard candidates.
Undo state is separately bounded in `.data/agent_actions.sqlite3`; inverse behavior remains a
static code policy, so the database does not retain source file or message content.
Durable memory state is separately bounded in `.data/agent_memory.sqlite3` and never stores the
source Discord message body or attachment.
Feedback is separate in `.data/feedback.sqlite3`. `/feedback` and an explicitly authorized AI
request both call the same `feedback.create` capability; reporter, workspace, channel, event, and
optional `agt_` reference come only from the host invocation context. Submitters cannot choose
kind or status. Administrators can run `simajilord-feedback list`, `show`, `set-kind`,
`set-status`, or `export`; the model has no inbox-read or triage capability.

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
uv run simajilord-translation-doctor
uv run simajilord-feedback list
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
uv run python scripts/manual_agent_discord_qa.py --scenario context
uv run python scripts/manual_agent_discord_qa.py --scenario capability
uv run python scripts/manual_agent_discord_qa.py --scenario bot
uv run python scripts/manual_agent_discord_qa.py --scenario status
uv run python scripts/manual_agent_discord_qa.py --scenario capability_context
uv run python scripts/manual_agent_discord_qa.py \
  --scenario handoff --escalation-model gpt-5.6-luna
uv run python scripts/manual_agent_discord_qa.py \
  --scenario handoff --escalation-model gpt-5.6-terra
```

This one-shot test gives the AI an exact-message research task and verifies first-party Codex
web search, the AI-authored primary-first evidence plan, Luna-high completion, concrete
intermediate messaging, and the final sourced answer. It consumes live model/search usage and
is intentionally not run on push. The context scenario reuses one provider thread, leaves an
unrelated prior turn in it, injects an instruction-like third-party history message, and verifies
that Luna retrieves a bounded page anchored before `↑これどう思う？` without treating history as
authority or confusing provider-thread order with the typed Discord
`immediate_predecessor_message_id`. The capability scenario uses the actual failed wording
`今流れてる曲について解説`; its Japanese text is absent from the English descriptor, so the check
passes only when Luna selects `audio.queue` semantically from the complete index, describes it,
invokes it, and reports the typed current-track result. The BOT and status cases exercise
`discord.inspect_application` and `system.status`; `capability_context` combines the exact
`↑これどう思う？` regression with an instruction-like history item and live audio discovery.
All capability cases also verify the persisted body-free `search → describe → invoke` trace, not
only the final prose. The two handoff commands form an A/B pair:
Luna investigates in both runs, while only the second finalization model changes.

To reproduce the official Discord HTTP-route comparison, clone Discord's documentation and pass
that exact checkout to the audit. The report records the documentation commit, every declared
route and classification, and implementation/type/permission evidence for all 106 typed
endpoints; an unknown future route fails closed:

```bash
git clone --depth 1 https://github.com/discord/discord-api-docs.git /tmp/discord-api-docs
uv run python scripts/audit_discord_api_coverage.py \
  --discord-docs /tmp/discord-api-docs \
  --output /tmp/simajilord-discord-api-coverage.json
```

The live audit invokes all 106 endpoints against a real connected client. Its first phase uses a
non-existent workspace to prove every safety boundary. With the explicit write flag, its second
phase creates one temporary channel in the selected test server, exercises safe reads, replies,
embeds, files, reactions, pins, threads, polls, and platform-resource reads. It also creates a
temporary voice channel to set and clear its status; both disposable channels are deleted in
`finally`. Never point this at a server that is not approved for disposable tests:

```bash
uv run python scripts/live_discord_capability_audit.py \
  --guild-id TEST_GUILD_ID \
  --actor-id ADMIN_USER_ID \
  --allow-safe-writes \
  --output .data/audits/discord-live-capabilities.json
```

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
