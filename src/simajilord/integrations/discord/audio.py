"""Discord implementation of the platform audio-output port."""

from __future__ import annotations

import asyncio
import logging
import shlex
import shutil
from dataclasses import replace
from time import monotonic

import discord

from simajilord.core.errors import EarlyPlaybackEnd, ProviderError, UserError
from simajilord.domain.audio import AudioItem, AudioKind

log = logging.getLogger(__name__)
_SOURCE_PREFLIGHT_TIMEOUT_SECONDS = 8.0
_EARLY_EOF_MINIMUM_EXPECTED_SECONDS = 15.0
_READ_ALOUD_LOUDNESS_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"
_MUSIC_DUCK_GAIN = 0.25


class _PrefetchedAudioSource(discord.AudioSource):
    """Replay one packet consumed while validating a replacement source."""

    def __init__(self, source: discord.AudioSource, first_packet: bytes) -> None:
        self._source = source
        self._first_packet = first_packet
        self._cleaned = False

    def read(self) -> bytes:
        if self._first_packet:
            packet = self._first_packet
            self._first_packet = b""
            return packet
        return self._source.read()

    def is_opus(self) -> bool:
        return self._source.is_opus()

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        self._source.cleanup()


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
        self._source_lock = asyncio.Lock()
        self._intentional_stop_generation = 0

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
        started_at = monotonic()
        stop_generation = self._intentional_stop_generation

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
            await completed
            elapsed = max(0.0, monotonic() - started_at)
            expected = max(
                0.0,
                (item.duration_seconds - item.start_seconds) / max(item.speed, 0.01),
            )
            tolerance = max(5.0, expected * 0.1)
            if (
                item.kind is AudioKind.MUSIC
                and expected >= _EARLY_EOF_MINIMUM_EXPECTED_SECONDS
                and self._intentional_stop_generation == stop_generation
                and elapsed + tolerance < expected
            ):
                raise EarlyPlaybackEnd(
                    elapsed_seconds=elapsed,
                    expected_seconds=expected,
                )
        finally:
            source.cleanup()

    async def overlay_speech(
        self,
        music: AudioItem,
        speech: AudioItem,
        *,
        position_seconds: float,
    ) -> None:
        """Hot-swap the active source without completing the Discord player."""

        overlay = replace(
            music,
            start_seconds=position_seconds,
            speech_overlay_source=speech.source,
            speech_overlay_owned_file=None,
            speech_overlay_duration_seconds=speech.duration_seconds,
            speech_overlay_volume=speech.volume,
            resume_after_overlay=False,
        )
        await self._swap_music_source(overlay)
        await asyncio.sleep(max(0.0, speech.duration_seconds) + 0.15)

    async def update_music(
        self,
        music: AudioItem,
        *,
        position_seconds: float,
    ) -> None:
        """Apply a new gain/tuning position while retaining the current stream URL."""

        updated = replace(
            music,
            start_seconds=position_seconds,
            speech_overlay_source=None,
            speech_overlay_owned_file=None,
            speech_overlay_duration_seconds=0.0,
            resume_after_overlay=False,
        )
        await self._swap_music_source(updated)

    async def fade_out(
        self,
        music: AudioItem,
        *,
        position_seconds: float,
        duration_seconds: float,
    ) -> None:
        """Replace the active source briefly with a bounded fade-out."""

        faded = replace(
            music,
            start_seconds=position_seconds,
            fade_in_seconds=0.0,
            fade_out_seconds=max(0.0, duration_seconds),
            speech_overlay_source=None,
            speech_overlay_owned_file=None,
            speech_overlay_duration_seconds=0.0,
            resume_after_overlay=False,
        )
        await self._swap_music_source(faded)
        await asyncio.sleep(max(0.0, duration_seconds))

    async def _swap_music_source(self, item: AudioItem) -> None:
        async with self._source_lock:
            voice = self._adopt_voice_client()
            if voice is None or not voice.is_connected() or not voice.is_playing():
                raise ProviderError("The Discord music source is not active.")
            replacement = build_discord_audio_source(item)
            prepared: _PrefetchedAudioSource | None = None
            previous = voice.source
            try:
                first_packet = await asyncio.wait_for(
                    asyncio.to_thread(replacement.read),
                    timeout=_SOURCE_PREFLIGHT_TIMEOUT_SECONDS,
                )
                if not first_packet:
                    raise ProviderError(
                        "The replacement audio source produced no Opus packet."
                    )
                prepared = _PrefetchedAudioSource(replacement, first_packet)
                voice.source = prepared
            except Exception:
                if prepared is None:
                    replacement.cleanup()
                else:
                    prepared.cleanup()
                raise
            if previous is not None and previous is not prepared:
                previous.cleanup()

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
        self._intentional_stop_generation += 1
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
    if item.kind is AudioKind.SPEECH:
        # VOICEVOX output is materially quieter than mastered music at the same
        # nominal volume. Normalise speech first, then retain the user-facing
        # speech volume as a predictable multiplier.
        filters.append(_READ_ALOUD_LOUDNESS_FILTER)
    if item.volume != 1.0:
        filters.append(f"volume={item.volume:.6f}")
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
    if item.fade_in_seconds > 0:
        filters.append(f"afade=t=in:st=0:d={item.fade_in_seconds:.3f}")
    if item.fade_out_seconds > 0:
        filters.append(f"afade=t=out:st=0:d={item.fade_out_seconds:.3f}")
    options: list[str]
    if item.speech_overlay_source is not None:
        music_filter = ",".join(filters) if filters else "anull"
        speech_filter = f"aresample=48000,{_READ_ALOUD_LOUDNESS_FILTER}"
        if item.speech_overlay_volume != 1.0:
            speech_filter += f",volume={item.speech_overlay_volume:.6f}"
        filter_graph = (
            f"[0:a]{speech_filter}[speech];"
            f"[1:a]{music_filter},volume={_MUSIC_DUCK_GAIN:.6f}[ducked];"
            "[ducked][speech]amix="
            "inputs=2:duration=longest:dropout_transition=0:normalize=0[sum];"
            "[sum]alimiter=limit=0.95:attack=5:release=50[mixed]"
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
