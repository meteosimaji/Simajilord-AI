# TempVC competitive and issue audit

Reviewed on 2026-08-25. This document separates observed product behaviour and issue evidence
from METEOBOT's implementation decisions.

## Product references

- [PartyBeast](https://www.partybeast.xyz/) uses permanent creator channels that create a room
  when joined and removes the room after it becomes empty. Its official command documentation
  covers lock/unlock, rename, user limit, invite, kick, ban, owner display, and ownership transfer;
  its [initial setup guide](https://krankey.gitbook.io/partybeast/helptopics/initial-setup)
  creates a category and a `Join Here` channel and supports multiple creators/categories.
- [TempVoice](https://easy.tempvoice.xyz/) supports multiple independently configured creator
  channels and either slash commands or an interface message. Its current
  [interface](https://easy.tempvoice.xyz/commands/interface) includes name, limit, privacy,
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
| [VoiceMaster #3](https://github.com/SamSanai/VoiceMaster-Discord-Bot/issues/3) | Public command acknowledgements accumulated in a shared channel; the request was for private feedback. | Setup, creation confirmation, and room controls are ephemeral interaction responses. |
| [AutoRoom #119](https://github.com/PhasecoreX/PCXCogs/issues/119) | A member could leave during join-triggered creation, causing Discord error `40032` and leaving an empty room unless explicitly rolled back. | Joining the lobby is inert. Creation requires a requester-bound button, rechecks that the member is still in the selected lobby, and rolls back both Discord and SQLite if the move fails. |
| [AutoRoom #165](https://github.com/PhasecoreX/PCXCogs/issues/165) | Moving to another VC sometimes supplied incomplete leave state and left old rooms behind, while a normal disconnect cleaned them up. | Voice events remain the fast path, but startup and five-minute reconciliation independently inspect only durable BOT-owned room IDs and reschedule missed empty rooms. |
| [YapHub #6](https://github.com/diese-tech/lab-yaphub/issues/6) | The design review called for restart-safe SQLite cleanup, stale-record repair, one active owned room, and interaction-driven private notices instead of voice-event DMs/public fallbacks. | Creation runs in an interaction, stale/missing rooms are reconciled, occupied owned rooms block duplicate creation, and a recoverable empty room is reused. |
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
   overwrites each creator copies. New creators copy their own lobby by default.
3. Joining the lobby does nothing by itself. The member runs `/tempvc` and presses the private
   **Create my room** button. The BOT verifies the exact lobby, creator record, category capacity,
   copy source, and BOT permissions before creating anything.
4. `/tempvc` inside a managed room opens the owner controls. The callback rechecks that the
   requester is still in that exact room and is its current owner or a channel manager.
5. The last human leaving starts a durable recovery window rather than immediate deletion.
   Rejoining cancels the task; restart/periodic reconciliation restores it. Only a registry row
   authorizes deletion, and an administrator can detach a room permanently.
6. Existing audio auto-leave remains the first response to an empty room. TempVC expiry also
   suspends a connected session when auto-leave was disabled, preserves its queue, forgets only a
   dashboard bound to the expiring channel, and keeps recreated destinations passive until an
   explicit audio action.

Immediate join-to-create remains common in other products, but it was not selected for METEOBOT:
the explicit interaction supplies private errors, removes accidental creation, and closes the
documented create-versus-move race without adding separate public command families.
