# Deleted Discord channel recovery

Read-aloud settings include an active route and saved profiles for previous voice
channels. Deleted IDs must be removed from both stores; otherwise Start can revive
a stale source and reject the entire request with `discord.message_destination_invalid`.

The Discord adapter reconciles saved references on READY, when opening the audio
hub, and before resuming or moving audio. A channel absent from the gateway cache
is checked using Discord REST. Only `404 / 10003 Unknown Channel` removes its
references. Missing access, transient failures, and uncached existing threads keep
the saved configuration. Fetch results are used when validating uncached sources;
normal permissions and audience checks still apply.

Gateway channel and raw thread deletion events perform the same durable cleanup.
A deleted source is removed while preserving remaining sources and mode. A deleted
destination or last source removes the route and saved profile. Explicit Start can
then use the conversation in which it was requested. Cleanup never automatically
chooses another voice channel or starts audio. Dashboard bindings to channels
received in deletion events are also forgotten.

Expected user-facing rejections are logged by error code and request ID, without
message contents. They previously bypassed the exception log, so absence of ERROR
lines did not mean the UI had no failures.

Verification: tests cover active and inactive profile persistence, repeated cleanup,
guild isolation, REST-confirmed missing sources, permission failures, transient
failures, existing uncached threads, gateway events, and one-operation Start recovery.
