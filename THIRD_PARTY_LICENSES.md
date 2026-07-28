# Third-party licenses

This project directly depends on the following open-source packages.
The package distributions and linked repositories contain the complete license texts.

| Package | Purpose | License |
| --- | --- | --- |
| discord.py | Discord API and voice client | MIT |
| davey | Discord DAVE media encryption binding | MIT |
| aiohttp | Bounded asynchronous HTTP client | Apache-2.0 / MIT |
| pypdf | Text extraction from bounded PDF sources | BSD-3-Clause |
| yt-dlp | Media metadata and stream resolution | Unlicense |
| yt-dlp-ejs | YouTube JavaScript challenge support | Unlicense (bundled MIT / ISC components) |
| python-dotenv | Local environment configuration | BSD-3-Clause |
| PyNaCl | Discord voice encryption support | Apache-2.0 |

The complete yt-dlp upstream repository is vendored at `vendor/yt-dlp`; its license text and
third-party notices are preserved in that directory. `davey` is also pinned directly because
the experimental video media layer requires codec-aware DAVE frame encryption in addition to
the audio support provided through `discord.py[voice]`. `yt-dlp-ejs` is pinned because current
yt-dlp releases require it, together with a supported JavaScript runtime, for complete YouTube
support.

FFmpeg is invoked as an external executable. Its available license configuration depends on
the installed build; see `ffmpeg -L` on the target system.

SearXNG is used as a separately running local metasearch service and is not copied into this
repository. Its source and AGPL-3.0-or-later license are available from the upstream project.

VOICEVOX Engine is used as a separately installed local speech service and is not copied into
this repository. The engine source is LGPL-3.0. Individual voices and character names are also
subject to the terms shown by the installed VOICEVOX distribution.
