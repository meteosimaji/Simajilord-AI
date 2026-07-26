# Vendored upstream software

## yt-dlp

- Upstream repository: <https://github.com/yt-dlp/yt-dlp>
- Upstream commit: `fdcc954df4955267ec1627cbeb347b661a110e7c`
- Snapshot date: 2026-07-26
- License: Unlicense (`vendor/yt-dlp/LICENSE`)
- Import policy: full repository snapshot, without upstream Git history

The platform installs this local snapshot through the root `pyproject.toml`. Arbitrary
external yt-dlp plugins are disabled by default. Simajilord-specific provider behavior belongs
outside the vendored tree so that the snapshot can be compared with and refreshed from
upstream. Transport adapters, including Discord, never import this directory directly.
