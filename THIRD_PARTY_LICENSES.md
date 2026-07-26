# Third-party licenses

This project directly depends on the following open-source packages.
The package distributions and linked repositories contain the complete license texts.

| Package | Purpose | License |
| --- | --- | --- |
| discord.py | Discord API and voice client | MIT |
| yt-dlp | Media metadata and stream resolution | Unlicense |
| python-dotenv | Local environment configuration | BSD-3-Clause |
| PyNaCl | Discord voice encryption support | Apache-2.0 |

The complete yt-dlp upstream repository is vendored at `vendor/yt-dlp`; its license text and
third-party notices are preserved in that directory. `davey` is installed through the
`discord.py[voice]` extra to support Discord's current voice encryption protocol.

FFmpeg is invoked as an external executable. Its available license configuration depends on
the installed build; see `ffmpeg -L` on the target system.
