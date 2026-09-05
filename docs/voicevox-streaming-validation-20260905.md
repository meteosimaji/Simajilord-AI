# METEOBOT streaming validation — September 5, 2026

The implementation is in commit `f36c571a4e63f71466478bc74a2dfea1a73e1923`.
Experiments used the existing BOT's private local operator input and its real
Discord voice output with background music. No user-account action was used to
submit test messages. Times are measured at the BOT; they exclude the receiver's
Discord network/jitter buffer, audio device, and human reaction time.

## Same-text comparison

The same 66-character Japanese passage and `clear` voice were used. Speaking rate
and pitch stayed at 1.0 and 0.0; music volume 100% and speech volume 200% were retained.
The speech duration was 10.347 seconds in both versions.

> これは読み上げ開始時間の比較テストです。文章全体の発音と抑揚を保ちながら、最後まで途切れず自然につながって聞こえることを確認します。

| Measurement | Original BOT (1 run) | Final BOT (3-run mean) | Final range | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Input to playback-ready | 12.942 s | 1.138 s | 1.029–1.276 s | 91.21% |
| Input to playback-complete | 23.361 s | 11.525 s | 11.417–11.663 s | 50.67% |

The original input timestamp is reconstructed from the capability completion
record and its measured duration. Final timestamps come from the operator client
and correlated BOT log records. These are observed samples under the host's
concurrent workload, not guaranteed percentiles or a controlled population study.

| Final run | Input (UTC) | Playback-ready | Playback-complete | Full synthesis complete |
| --- | --- | ---: | ---: | ---: |
| 1 | 2026-09-05T05:30:17.420257+00:00 | 1.108 s | 11.494 s | 3.100 s |
| 2 | 2026-09-05T05:30:29.518140+00:00 | 1.029 s | 11.417 s | 3.183 s |
| 3 | 2026-09-05T05:30:41.591541+00:00 | 1.276 s | 11.663 s | 3.248 s |

Each final same-text run emitted 518 Opus packets, began before synthesis finished,
and completed with FFmpeg exit code 0. No read exceeded the 20 ms packet interval.

## Long text, real prior messages, and concurrency

- The final 308-character passage, with added line breaks, began in
  1.310 s and finished in 55.997 s.
  Its entire waveform was generated at 16.394 s:
  playback therefore began about 15.08 s before generation completed.
  All 2,733 packets were consumed. Maximum speech read wait was 3.153 ms;
  reads exceeding 20 ms: zero. No per-line or per-sentence synthesis was performed.
- Five texts previously read in the same VC were replayed through the BOT, including
  a multiline task list, an author prefix, a Latin-letter input, and percentage
  expressions. All completed without a speech-data read exceeding 20 ms.
- Three submissions 50 ms apart queued as 1, 2, 3 while music played. The first
  contained the 308-character passage. Their playback-ready times were 2.244,
  56.967, and 66.588 s from each input; completion was 56.934, 66.555, and 73.396 s.
  Each previous speech completed before the next began. Long waits for entries
  2 and 3 represent the required queue order, not synthesis starvation.
  Packet counts were 2,733 / 478 / 339; slow reads: 0 / 0 / 0.
- One earlier long-text test queued behind a real user read-aloud message and
  started after full generation. It is retained as queue-contention evidence,
  rather than counted as proof of early streaming playback.
- Across 23 controlled live utterances after fixing process priority, including
  the final buffer optimization, no speech-data read exceeded 20 ms. This excludes
  the failed pre-fix experiment described below and is not a guarantee against
  arbitrary future CPU starvation or network loss.

## What was fixed and how it was checked

The first streaming deployment was slower: about 16.4 s to start and 26.9 s to
finish, with a 1.495 s speech read stall. Its launchd job used `ProcessType=Background`,
which was inherited by the engine and FFmpeg. Terminal experiments did not have
that restriction. The job now uses `Interactive`; measured process priority moved
from 4 to 31. Inference threads are explicitly configured to 4. More threads were
not assumed to be better.

FFmpeg's default one-second Ogg page accumulation was reduced to 20 ms pages with
packet flushing. With identical paced PCM input, first-packet time changed from
1.667 / 1.669 s to 1.067 / 1.066 s. All 919 encoded packet payloads had the same
SHA-256 (`8c165a3963b2471258ac542421b3351646dfd329b27d634939c721b3c058dbed`)
before and after. Loudness filters and user volumes were unchanged.

Official normal and streaming synthesis of one fixed query produced the same
440,576 samples at 24 kHz; maximum PCM difference was 2 integer units out of 32,768.
A separate 48 kHz stereo, speed 1.2, pitch 0.04 test returned 177,152 frames in both
paths with maximum PCM difference 1. The engine's manifest, speaker info, version,
and core-version endpoints returned HTTP 200. All 127 previously available talk
style IDs remain available across 43 speakers. This checks engine API compatibility;
it does not claim that the editor UI implements streaming itself.

Graceful restart saved the active music at 166.804 seconds and automatically
reconnected the same voice destination on September 5 at 14:29:11 JST. The music
resumed without an operator connect action. Manually held sessions stayed held.
Earlier restart validation also restored the same Vivaldi track from its saved
122.290-second position. SIGTERM now reaches graceful cleanup before Discord voice
transports are torn down; forcibly killing a process still cannot produce a final
position checkpoint.

## Validation and evidence

- Local complete suite: 1,391 passed; one existing `audioop` deprecation warning.
- mypy: 140 source modules; Ruff, lock check, package build, secret scan, and diff
  whitespace check passed.
- Added real FFmpeg regression verifies a first packet before the withheld WAV
  tail, then compares every Opus packet with ordinary complete-file playback.
- Added tests cover multiline/full-text submission, split HTTP headers, size and
  PCM validation, temporary buffer exhaustion, cancellation, failed synthesis,
  queue recovery, held routes, failed reconnect, and SIGTERM cleanup.
- GitHub Actions initially passed Linux Python 3.11/3.12/3.13, macOS, and Activity;
  one Python 3.14 test reached its overly short 2-second CI wait. The timeout was
  raised to 10 seconds while still withholding the tail and requiring exact packet
  equality. Final remote status is verified separately after the follow-up push.

Detailed request IDs, raw timestamp evidence, JSON summaries, WAV comparisons,
asset hashes, restart snapshots, and pytest XML are retained locally under
`.data/benchmarks/read-aloud-20260905/`. Private channel data, configuration, and
raw logs are not included in this repository report. No PR was created.
