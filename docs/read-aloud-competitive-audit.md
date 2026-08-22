# Read-aloud competitive audit

Checked on 2026-08-22. Public product pages change, so this is a dated engineering
baseline rather than a permanent ranking. A feature is marked present only when it is
documented by the operator or verified in the Simajilord checkout.

## Public baselines

| Product | Publicly documented strengths | Paid or boosted features |
| --- | --- | --- |
| [ずんだもんβ](https://lenlino.com/zunda-beta/) | Per-user voice, speed and pitch, auto-start, dictionary, mute, alarms, queue speed-up, join/leave announcements, multiple voice engines and a web dashboard | [Premium](https://lenlino.com/zunda-beta/premium/) raises one-message output from 50 to 100 characters and advertises GPU generation, extra BOT instances, COEIROINK/Aivis/AquesTalk, regex and voice dictionaries, translation, alerts, mentions, temporary voices and API access across ¥100–¥1,000 plans |
| [VOISCORD](https://voiscord.net/) | A.I.VOICE, dashboard configuration, per-member settings, server dictionary, auto-join and per-VC source routing | The official comparison shows 5,000 characters/month, 60 characters/message, 50 dictionary slots and one BOT for free; one ¥200/month boost advertises 100,000 characters, 140 characters/message, speed/pitch, BOT-message reading, 300 dictionary slots and four BOTs |
| [Vocalis](https://vocalis.click/) | Per-user speaker, dictionary, dashboard, original voice models and a claimed approximately 0.1-second-class response. Its current landing page says 70+ speakers while its pricing page says 150+, so the catalog count is not treated as settled | [Pricing](https://vocalis.click/pricing) lists a ¥350 pay-as-you-go tier with personal VC creation, an original branded BOT and 50 additional characters per message |
| [Swiftly](https://swiftlybot.com/commands) | Per-user voice, speed, user/server dictionaries, dictionary sharing and moderation, auto-join, overlap mode, user/role ignore panels, status and bug-report forms | No paid feature tier was found on the checked official command and sponsor pages; GPU hosting is sponsor-supported |
| [0kqBee](https://zenn.dev/home/articles/919eece59d9ad6) | 100+ characters across multiple engines, per-user speed/pitch, safe content filters, regex/importable dictionaries, multiple source channels, auto-join, entry/exit notices and a dashboard | Free is documented as unlimited total characters/dictionary entries with an 80-character message limit; Mini is ¥100 and Standard or higher is ¥500 with all voices and a 160-character limit |
| [KuronekoServer](https://tts.krnk.org/) | Multiple synthesis engines/BOT instances, dashboard, dictionary, auto-entry and user/role exclusion | Free/¥110/¥330/¥550 plans advertise 100/150/200/300 characters per message, 50/75/100/unlimited dictionary entries, progressively larger auto-entry and exclusion limits, plus BOT/webhook reading, priority and regex in higher tiers |
| [VOICEVOX-読み上げさん](https://yomiage.org/) | Advertises every feature free: 100+ voices, VOICEVOX/AivisSpeech selection, per-user speed/pitch/intonation/volume/suffix, translation, dictionary permissions, auto-join and a web dashboard | No paid tier is advertised |
| [読みちゃん](https://yomi-chan.jp/) | Four engines, custom uploaded speakers, speed/volume/speaker settings and a simple slash-command flow | No public price table was found on the checked landing page |

Self-reported latency claims are not interchangeable benchmarks. The checked Vocalis page
does not publish the message length, client region, load, percentile, timer boundaries or
whether a result was cached. Simajilord therefore reports its own server-side and live-VC
measurements separately.

## Simajilord verified surface

- One durable route can combine up to 25 text channels, threads and VC chats into one
  destination while preserving Discord snowflake order.
- One guild FIFO covers messages, join/leave/move announcements and operator speech.
- Exact message-ID de-duplication and consecutive-content compaction prevent common double
  reads without conflating distinct messages.
- Five curated, durable VOICEVOX presets can be selected per user, with a server fallback.
  The installed engine currently reports 118 styles, but the complete catalog is not yet
  exposed in the Discord picker.
- Per-user VOICEVOX speed (0.5-2.0) and pitch (-0.15 to 0.15) are durable and available from
  the setup panel. Provider calls and reusable speech cache keys include both values, so one
  member's tuning cannot contaminate another member's cached announcement.
- Literal pronunciation dictionaries, user/role exclusion, reply/attachment semantics,
  VC-member-only mode and optional full-text or 20–400-character `以下略` behavior persist
  across restarts.
- The ephemeral setup panel now exposes channel selection, long-message behavior, personal
  voice, message-detail behavior and pronunciation entry. It does not modify the persistent
  music/Queue component tree.
- Rich Discord content is normalized locally: Markdown decoration is removed, links and
  replies receive spoken labels, code blocks are compacted, and repeated custom emoji are
  counted. This adds no network fetch to the latency-critical path.
- Speech is mixed into an already-running Opus music source. Music remains on the same
  decoder/source instead of reconnecting to YouTube after every utterance.
- Read-aloud listener/source visibility can be enforced, audited or disabled. The live host
  used for this audit currently has enforcement disabled; that is an operator configuration
  warning, not a missing implementation.

## Measured latency on the live M4 Pro host

The local VOICEVOX 0.25.1 engine was warm and style 3 was used.

| Input | VOICEVOX `/audio_query` | `/synthesis` first byte / complete | Live Simajilord request to mixer-ready |
| --- | ---: | ---: | ---: |
| 11 Japanese characters, five runs | 2.1–2.6 ms | 330–338 ms | comparable 16-character live request: about 459 ms |
| 51 Japanese characters, five runs | 3.5–5.4 ms | 1,119–1,166 ms | comparable 54-character live request: about 1,121 ms |

VOICEVOX returned the first response byte only when synthesis was essentially complete, so
streaming the same endpoint cannot by itself reach sub-100 ms onset. The remaining practical
speed work is to remove local decode/startup overhead, keep the engine warm, cache repeated
semantic segments and evaluate a second low-latency engine with a clearly labelled quality
trade-off. Message order and natural single-utterance output remain hard regression gates.

The optional `--enable_cancellable_synthesis` route was also tested in an isolated engine on
port 50031. It still returned a complete `Content-Length` WAV only after synthesis: at 44
characters, first-byte/complete was about 990.1/991.1 ms; at 216 characters it was about
5,220.8/5,224.8 ms. The regular route on the same isolated engine was about 996.0/997.0 ms
and 5,043.4/5,047.8 ms respectively. Cancellation changes process control, not incremental
audio delivery. Segmenting text is therefore the only pseudo-streaming option in this engine;
it remains disabled because the requested natural single-utterance output takes priority.

## Priorities after this patch

1. Validate the expanded panel and rich-content formatting in Discord, including non-manager
   access, permission revocation, multiple selected channels and the unchanged Queue panel.
2. Expose the installed VOICEVOX catalog with searchable/paginated selection and attribution.
3. Add dictionary JSON import/export. Regex requires a bounded engine or timeouts; Python's
   unbounded backtracking regex must not be put on the automatic message path.
4. Evaluate optional BOT-message reading with loop/duplicate protections. Keep it off by
   default because it can recreate the double-reading class of bug.
5. Benchmark a direct WAV-to-48-kHz mixer path and alternative local synthesis engines. Report
   p50 and p95 from Discord message creation to first audible packet; do not market an
   incomparable best-case number.
