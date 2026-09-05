#!/bin/sh
# Local official engine checkout and runtime installed as documented in docs/voicevox-streaming.md.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
engine_dir="$project_dir/.data/voicevox-engine-streaming"
assets_dir="$project_dir/.data/voicevox-streaming-assets"
exec "$engine_dir/.venv/bin/python" "$engine_dir/run.py" \
    --voicevox_dir "$assets_dir" --runtime_dir "$assets_dir" "$@"
