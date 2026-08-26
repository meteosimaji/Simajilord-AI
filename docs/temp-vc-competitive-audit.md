# TempVC competitive and issue audit

Reviewed on 2026-08-26. This document separates observed product behaviour and issue evidence
from METEOBOT's implementation decisions.

## Product references

- [PartyBeast](https://www.partybeast.xyz/) uses permanent creator channels that create a room
  when joined and removes the room after it becomes empty. Its official command documentation
  covers lock/unlock, rename, user limit, invite, kick, ban, owner display, and ownership transfer;
  its [initial setup guide](https://krankey.gitbook.io/partybeast/helptopics/initial-setup)
  creates a category and a `Join Here` channel and supports multiple creators/categories.
- [TempVoice](https://easy.tempvoice.xyz/getting-started/setup) supports multiple independently
  configured join-to-create channels and either slash commands or an interface message. Its
  current [interface](https://easy.tempvoice.xyz/commands) includes name, limit, privacy,
  trust/block, invite/kick, claim, transfer, and delete controls. Its owner-setting documentation
  explicitly supports restoring the previous name and privacy mode on recreation.
- [Serenity VoiceMaster](https://docs.serenity.wtf/server-setup/voicemaster) uses one shared
  interface, validates missing setup permissions before half-building the module, supports claim
  and transfer, preserves presets, and documents Discord's category capacity and channel-edit
  rate limits.
- [Bored VoiceMaster](https://docs.bored.rest/voicemaster) separates public/private creators and
  exposes an explicit policy for whether music bots may stay after the owner leaves.
- [Astro generators](https://github.com/bot-astro/docs/blob/main/src/temporary-voice-channels/generators.md)
  make permission inheritance explicit: none, creator channel, or category. They also support
  multiple generators and a fallback when a 50-channel category is full.
- [Robotnic](https://github.com/jack-schultz/Robotnic) stores creator and temporary channels in
  SQLite, cleans stale records on startup, performs a periodic empty-room sweep, offers creator
  versus category permission inheritance, and queues renames rather than assuming channel edits
  are unlimited.

## Issue and request evidence

| Evidence | Observed failure or request | METEOBOT response |
| --- | --- | --- |
| [VoiceMaster #29](https://github.com/SamSanai/VoiceMaster-Discord-Bot/issues/29) | The BOT forgot that the member was the controller but still remembered that they owned a room, leaving both creation and controls unusable. | Active room and owner live in one SQLite row; every control rechecks the current Discord room and membership. |
| [VoiceMaster #43](https://github.com/SamSanai/VoiceMaster-Discord-Bot/issues/43) | An invalid channel name could abort creation/rename and leave the member in the creator channel indefinitely. | Names are normalized and reject control/surrogate/replacement characters before a Discord mutation; failed create/move rolls back the new channel and DB row. |
| [VoiceMaster #44](https://github.com/SamSanai/VoiceMaster-Discord-Bot/issues/44) and [AutoRoom #70](https://github.com/PhasecoreX/PCXCogs/issues/70) | Owners could lock themselves out, and users requested their renamed room/settings to return next time. | Owner view/connect is a safety overwrite. Name, limit, and lock are stored as a durable owner profile and restored on recreation. |
| [VoiceMaster #52](https://github.com/SamSanai/VoiceMaster-Discord-Bot/issues/52), [AutoRoom #95](https://github.com/PhasecoreX/PCXCogs/issues/95), and [AutoRoom #142](https://github.com/PhasecoreX/PCXCogs/issues/142) | Servers need the creator lobby's visibility to differ from the generated room and need role/member overwrite copying to be predictable. | Each creator stores an optional existing voice-channel permission source. The private setup panel can switch between a specific VC and live category permissions; a missing source blocks creation instead of broadening access. |
| [VoiceMaster #3](https://github.com/SamSanai/VoiceMaster-Discord-Bot/issues/3) | Public command acknowledgements accumulated in a shared channel; the request was for private feedback. | Setup, creation retry, and room controls are ephemeral interaction responses. Successful automatic creation needs no acknowledgement. |
| [AutoRoom #119](https://github.com/PhasecoreX/PCXCogs/issues/119) | A member could leave during join-triggered creation, causing Discord error `40032` and leaving an empty room unless explicitly rolled back. | Automatic creation rechecks that the member is still in the exact lobby under a per-member lock. A failed move deletes the new Discord channel and closes its SQLite row, while `/tempvc` offers a private retry if the member remains in the lobby. |
| [AutoRoom #165](https://github.com/PhasecoreX/PCXCogs/issues/165) | Moving to another VC sometimes supplied incomplete leave state and left old rooms behind, while a normal disconnect cleaned them up. | Voice events remain the fast path, but startup and five-minute reconciliation independently inspect only durable BOT-owned room IDs and reschedule missed empty rooms. |
| [YapHub #6](https://github.com/diese-tech/lab-yaphub/issues/6) | The design review called for restart-safe SQLite cleanup, stale-record repair, one active owned room, and interaction-driven private notices instead of voice-event DMs/public fallbacks. | Automatic creation stays silent on success, stale/missing rooms are reconciled, occupied owned rooms block duplicates, and a recoverable empty room is reused. Expected automatic failures are journaled; the member can open a private `/tempvc` retry panel without a public or DM fallback. |
| [AutoRoom #116](https://github.com/PhasecoreX/PCXCogs/issues/116) | Copying a stale bitrate after a server boost reduction made channel creation fail. | Permission copy intentionally copies only role/member overwrites. Bitrate, name, and user limit are separately bounded METEOBOT settings. |

## Discord constraints

- Discord's [channel resource](https://docs.discord.com/developers/resources/channel) bounds names
  to 1-100 characters and categories to 50 children. METEOBOT validates both before creation.
- Discord's [permission documentation](https://docs.discord.com/developers/topics/permissions)
  distinguishes channel overwrites from category permission syncing. METEOBOT therefore stores
  whether a creator uses a fixed VC snapshot or the destination category's current overwrites.
- Discord's [rate-limit documentation](https://docs.discord.com/developers/topics/rate-limits)
  says limits can change and clients must honor returned bucket/reset headers. METEOBOT serializes
  room edits and relies on discord.py's HTTP limiter rather than hard-coding an assumed route
  quota.

## Resulting METEOBOT design

1. A Manage Channels member runs `/tempvc` and creates a permanent category plus creator lobby,
   or adds a creator to an existing category.
2. The administrator may select a different existing voice channel whose role/member permission
   overwrites each creator copies. New creators copy their own lobby by default. Locking overlays
   the copied `@everyone` Connect value; unlocking restores that exact tri-state value instead of
   silently widening a source VC that was already private.
3. Joining the lobby starts creation immediately. Under a per-member lock, the BOT rechecks the
   exact lobby, creator record, category capacity, copy source, and BOT permissions before creating
   anything. If the member remains in the lobby after an expected failure or missed event,
   `/tempvc` exposes the same operation as a requester-only **Retry room creation** button.
4. `/tempvc` inside a managed room opens the owner controls. The callback rechecks that the
   requester is still in that exact room and is its current owner or a channel manager.
5. The last human leaving starts a durable recovery window rather than immediate deletion.
   Rejoining cancels the task; restart/periodic reconciliation restores it. Only a registry row
   authorizes deletion, and an administrator can detach a room permanently.
6. Existing audio auto-leave remains the first response to an empty room. TempVC expiry also
   suspends a connected session when auto-leave was disabled, preserves its queue, forgets only a
   dashboard bound to the expiring channel, and keeps recreated destinations passive until an
   explicit audio action.

## Functional comparison after review

| Area | Common competitor model | METEOBOT decision |
| --- | --- | --- |
| Creation | Join a `Join Here` / `Join to create` channel and get moved immediately. | Same one-action path. New hubs use `Join to create`; existing tracked lobby names are not silently renamed. |
| Recovery from a failed create | Usually a DM, command, or second generator. | Stay in the lobby and run `/tempvc` for a requester-only retry with the exact same validation and rollback. |
| Everyday controls | Persistent interface or commands for name, limit, privacy, members, and ownership. | One contextual `/tempvc` surface for rename, limit, lock, invite, remove/deny, transfer, and administrator-only permanence. Ownership also transfers automatically if the owner leaves people behind. |
| Saved preferences | Product-dependent presets or previous-room restoration. | Name, limit, lock, permission source, empty recovery deadline, and remapped audio/read-aloud destinations are durable. |
| Category capacity | Fail, queue, or use another configured generator. | Fail before mutation with a private administrator action; multiple hubs/categories are supported, but automatic cross-category fallback is deliberately not inferred. |
| Cleanup safety | Delete a channel when it becomes empty. | Delete only a channel ID present in the BOT-owned SQLite registry, after a configurable recovery window and periodic reconciliation. |

This matches the dominant `Join Here` / `Join to create` model in PartyBeast, TempVoice, Serenity,
Bored VoiceMaster, Astro, and VoiceMaster. METEOBOT keeps its stricter differences: durable BOT-only
deletion authority, per-member serialization, exact-lobby revalidation, rollback on failed moves,
and a private retry/control surface rather than public command acknowledgements.
