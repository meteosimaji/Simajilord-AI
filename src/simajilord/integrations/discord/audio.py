"""Discord implementation of the platform audio-output port."""

from __future__ import annotations

import asyncio
import logging
import shlex
import shutil
from time import monotonic

import discord

from simajilord.core.errors import ProviderError, UserError
from simajilord.domain.audio import AudioItem

log = logging.getLogger(__name__)


async def verify_ffmpeg_opus() -> None:
    """Fail fast when the host cannot produce the Opus packets Discord expects."""

    executable = shutil.which("ffmpeg")
    if executable is None:
        raise ProviderError("FFmpeg is not installed or is not available on PATH.")
    process = await asyncio.create_subprocess_exec(
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=48000:cl=stereo",
        "-t",
        "0.05",
        "-c:a",
        "libopus",
        "-f",
        "opus",
        "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10.0)
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise ProviderError("FFmpeg Opus self-test timed out.") from exc
    if process.returncode != 0 or not stdout.startswith(b"OggS"):
        detail = stderr.decode(errors="replace")[-500:]
        raise ProviderError(f"FFmpeg Opus self-test failed: {detail or 'invalid output'}")
    log.info("FFmpeg Opus self-test passed using %s", executable)


class DiscordAudioOutput:
    """One Discord voice connection for one guild-owned audio session."""

    def __init__(self, bot: discord.Client, guild_id: int) -> None:
        self.bot = bot
        self.guild_id = guild_id
        self.destination_id: int | None = None
        self._voice: discord.VoiceClient | None = None

    @property
    def connected(self) -> bool:
        voice = self._adopt_voice_client()
        return voice is not None and voice.is_connected()

    @property
    def paused(self) -> bool:
        voice = self._adopt_voice_client()
        return voice is not None and voice.is_paused()

    async def connect(self, destination_id: str) -> None:
        try:
            channel_id = int(destination_id)
        except ValueError as exc:
            raise UserError("音声の出力先が正しくありません。") from exc
        guild = self.bot.get_guild(self.guild_id)
        if guild is None:
            raise UserError("Discordサーバーを利用できません。")
        channel = guild.get_channel(channel_id)
        if not isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
            raise UserError("設定されたボイスチャンネルは現在存在しません。")

        voice = self._adopt_voice_client()
        if voice is not None and voice.is_connected():
            if voice.channel.id != channel.id:
                await voice.move_to(channel)
            self.destination_id = channel.id
            return

        if voice is not None:
            try:
                await voice.disconnect(force=True)
            except discord.DiscordException:
                log.warning("Could not discard a stale Discord voice client", exc_info=True)
        try:
            protocol: discord.VoiceProtocol = await channel.connect(
                timeout=20.0,
                reconnect=True,
                self_deaf=True,
            )
        except (TimeoutError, discord.DiscordException) as exc:
            raise UserError("ボイスチャンネルへ接続できませんでした。") from exc
        if not isinstance(protocol, discord.VoiceClient):
            await protocol.disconnect(force=True)
            raise ProviderError("Discord returned an unsupported voice protocol.")
        self._voice = protocol
        self.destination_id = channel.id

    async def play(self, item: AudioItem) -> None:
        voice = self._adopt_voice_client()
        if voice is None or not voice.is_connected():
            raise UserError("BOTはボイスチャンネルに接続していません。")
        if voice.is_playing() or voice.is_paused():
            raise ProviderError("The Discord audio output is already busy.")

        source = build_discord_audio_source(item)
        loop = asyncio.get_running_loop()
        completed: asyncio.Future[None] = loop.create_future()

        def after(error: Exception | None) -> None:
            def finish() -> None:
                if completed.done():
                    return
                if error is None:
                    completed.set_result(None)
                else:
                    completed.set_exception(error)

            loop.call_soon_threadsafe(finish)

        try:
            # FFmpegOpusAudio reports is_opus=True, so discord.py sends the packets
            # directly instead of constructing its native libopus PCM encoder.
            voice.play(source, after=after)
            started_at = monotonic()
            item.cleanup_speech_overlay()
            overlay_duration = item.speech_overlay_duration_seconds
            if item.speech_overlay_source is None or overlay_duration <= 0:
                await completed
            else:
                overlay_finished = asyncio.create_task(
                    asyncio.sleep(overlay_duration + 0.35),
                    name="simajilord-speech-overlay",
                )
                try:
                    done, _ = await asyncio.wait(
                        (completed, overlay_finished),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if overlay_finished in done and not completed.done():
                        elapsed = max(0.0, monotonic() - started_at)
                        voice.stop()
                        await completed
                        item.start_seconds += elapsed * item.speed
                        item.resume_after_overlay = True
                    else:
                        await completed
                finally:
                    overlay_finished.cancel()
                    await asyncio.gather(overlay_finished, return_exceptions=True)
        finally:
            source.cleanup()

    def pause(self) -> None:
        voice = self._adopt_voice_client()
        if voice is None or not voice.is_playing():
            raise UserError("現在再生している曲はありません。")
        voice.pause()

    def resume(self) -> None:
        voice = self._adopt_voice_client()
        if voice is None or not voice.is_paused():
            raise UserError("現在、一時停止していません。")
        voice.resume()

    def stop(self) -> None:
        voice = self._adopt_voice_client()
        if voice is not None and (voice.is_playing() or voice.is_paused()):
            voice.stop()

    async def disconnect(self) -> None:
        voice = self._adopt_voice_client()
        if voice is not None:
            await voice.disconnect(force=True)
        self._voice = None
        self.destination_id = None

    def _adopt_voice_client(self) -> discord.VoiceClient | None:
        if self._voice is not None:
            return self._voice
        guild = self.bot.get_guild(self.guild_id)
        if guild is None or guild.voice_client is None:
            return None
        if isinstance(guild.voice_client, discord.VoiceClient):
            self._voice = guild.voice_client
        return self._voice


def build_discord_audio_source(item: AudioItem) -> discord.FFmpegOpusAudio:
    """Create an Opus source without invoking discord.py's native PCM encoder."""

    before_parts = ["-nostdin"]
    if item.speech_overlay_source is not None:
        before_parts.extend(("-i", item.speech_overlay_source))
    if item.kind.value == "music" and item.source.startswith(("http://", "https://")):
        before_parts.extend(
            ("-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5")
        )
    header_value = _header_value(item.http_headers)
    if header_value:
        before_parts.extend(("-headers", header_value))
    if item.start_seconds > 0:
        before_parts.extend(("-ss", f"{item.start_seconds:.3f}"))
    filters: list[str] = []
    if item.pitch != 1.0:
        filters.extend(
            (
                "aresample=48000",
                f"asetrate=48000*{item.pitch:.6f}",
                "aresample=48000",
            )
        )
    tempo = item.speed / item.pitch
    filters.extend(_atempo_filters(tempo))
    options: list[str]
    if item.speech_overlay_source is not None:
        music_filter = ",".join(filters) if filters else "anull"
        filter_graph = (
            "[0:a]aresample=48000,asplit=2[speech_sc][speech_mix];"
            f"[1:a]{music_filter}[music];"
            "[music][speech_sc]sidechaincompress="
            "threshold=0.015:ratio=8:attack=20:release=350[ducked];"
            "[ducked][speech_mix]amix="
            "inputs=2:duration=longest:dropout_transition=0:normalize=0[mixed]"
        )
        options = ["-filter_complex", filter_graph, "-map", "[mixed]", "-vn"]
    else:
        options = ["-vn"]
        if filters:
            options.extend(("-filter:a", ",".join(filters)))
    return discord.FFmpegOpusAudio(
        item.source,
        before_options=shlex.join(before_parts),
        options=shlex.join(options),
    )


def _atempo_filters(tempo: float) -> list[str]:
    filters: list[str] = []
    remaining = tempo
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    if abs(remaining - 1.0) > 1e-6:
        filters.append(f"atempo={remaining:.6f}")
    return filters


def _header_value(headers: dict[str, str] | None) -> str | None:
    if not headers:
        return None
    lines = []
    for key, value in headers.items():
        if key.lower() in {"cookie", "authorization"}:
            continue
        safe_key = key.replace("\r", "").replace("\n", "")
        safe_value = value.replace("\r", " ").replace("\n", " ")
        lines.append(f"{safe_key}: {safe_value}")
    return "\r\n".join(lines) + "\r\n" if lines else None
