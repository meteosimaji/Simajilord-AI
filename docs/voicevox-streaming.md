# VOICEVOX streaming and playback recovery

METEOBOT can consume the official `/streaming_synthesis` WAV response while the
engine is still generating it. Enable `VOICEVOX_STREAMING_ENABLED=true` only with
an engine/core/model combination supporting `streaming_talk` styles. The provider
checks `/speakers` and keeps the ordinary synthesis route for unsupported voices.
The default remains disabled for existing installations.

The streaming path submits a single complete AudioQuery, preserving the text's
punctuation and author/body order. It does not perform sentence-by-sentence
synthesis. Engine waveform frames share the full utterance's intermediate features.
FFmpeg emits Ogg pages every 20 ms instead of collecting a full second; regression
coverage compares every encoded packet with the ordinary complete-file path.
A bounded private disk spool supplies a blocking FFmpeg input: temporary exhaustion
is not EOF, cancellation releases the reader, and incomplete/failed responses are
reported as playback errors. PCM format and declared length are checked. A measured
initial buffer accounts for synthesis slower than realtime; it cannot guarantee
uninterrupted playback under arbitrary CPU starvation or network outages.

Existing VOICEVOX speed/pitch presets, Discord speech volume/loudness processing,
and live music mixing continue to apply. The ordinary `/synthesis` endpoint is
retained by the unmodified official engine; editor-native streaming is separate
from this BOT integration.

## Tested local installation (Apple Silicon, September 5, 2026)

- Engine: official `VOICEVOX/voicevox_engine` commit
  `7aed82202a2fcff59d35def8899733db63c3fc7f`, including PRs 1823 and 1876.
- Core: official `VOICEVOX/voicevox_core` release archive
  `voicevox_core-osx-arm64-0.17.0.zip`.
- Runtime: official VOICEVOX ONNX Runtime `1.23.2`, macOS arm64.
- Models: official `VOICEVOX/voicevox_vvm` release `0.17.0`, numbered models
  `0.vvm` through `25.vvm`. Do not mix the optional Nemo `n0.vvm` into this engine's
  character catalogue.
- Character resources: `VOICEVOX/voicevox_resource`, commit
  `8674f04cd160cdb0b08817020b40c5774257b6f9`. Package character directories by UUID (remove the name and
  underscore prefix), retaining their original metadata, policies, and samples.
- Python: Homebrew 3.14.7 with the engine's declared dependencies in a dedicated
  virtual environment. The upstream project currently pins 3.14.6; this host uses
  the next Python patch release and validates it through real synthesis.

Keep the engine checkout at `.data/voicevox-engine-streaming`, its Python at
`.venv/bin/python` within that checkout, and libraries/models/resources at
`.data/voicevox-streaming-assets`. The core resolves its actual library directory:
that directory must also contain a `model` symlink and the ONNX Runtime library.
Retain the official model README and terms. None of these binaries or resources
are vendored into the BOT repository.

Set `VOICEVOX_ENGINE_PATH` to the absolute path of
`scripts/run_voicevox_streaming.sh`, `VOICEVOX_AUTO_START=true`, and
`VOICEVOX_CPU_NUM_THREADS=4`. Thread count is configurable (0 means engine default);
benchmark on the target machine rather than assuming more threads is faster.

For a launchd service handling realtime Discord audio, use `ProcessType` =
`Interactive`, not `Background`. Background resource restrictions propagate to
the engine and FFmpeg and can make generation slower than playback even when a
terminal benchmark is fast. Apply this in the user's existing LaunchAgent, retain
its other settings, and reload it gracefully. `ExitTimeOut` = 60 accommodates child
shutdown. The BOT handles SIGTERM as a graceful close, preserving audio state
before Discord disconnects. SIGKILL cannot provide a final position checkpoint.

## Measuring the running BOT without operating a user account

Use the existing private local operator socket:

```sh
uv run --locked simajilord-vc-speak --guild GUILD_ID --voice clear --json '固定の比較文。'
```

Use `--connect-if-needed` only when reconnecting an authorized saved destination;
exclude connection setup from a steady-state latency comparison. Reuse exactly the
same text, voice, and tuning for both paths. Correlate the returned request ID with
`capability.invocation`, `Speech stream ready`, first Opus packet, live overlay
ready, synthesis completion, and `Speech overlay completed` records. Report input
to playback-ready and input to playback-complete separately. A queued response
alone is not an audible-start receipt. BOT-side timestamps do not include the
listener's Discord network/jitter buffer or device latency.

The live mixer records consumed packets and read durations exceeding the 20 ms
packet interval. Validate waveform length/tail, normal-versus-streaming PCM
continuity, FFmpeg exit status, and listener feedback in addition to latency.
Do not declare success if a faster start introduces underruns or distorted audio.

## Restart recovery

A durable `resume_on_restart` flag identifies previously connected, running audio
sessions. Graceful shutdown saves music, queue position, volume, speed, and routing
before stopping transports. Startup reconnects those sessions and resolves saved
media references again. Paused, manually held, and legacy unmarked sessions remain
held; failed reconnects retain recovery intent. Transient speech spools are not
replayed after restart. Existing durable image/focus jobs retain their own recovery
mechanisms.
