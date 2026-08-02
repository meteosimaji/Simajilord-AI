"""Discord implementation of the platform audio-output port."""

from __future__ import annotations

import asyncio
import logging
import shlex
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import replace
from time import monotonic

import discord

from simajilord.core.errors import EarlyPlaybackEnd, ProviderError, UserError
from simajilord.domain.audio import AudioItem, AudioKind

log = logging.getLogger(__name__)
_SOURCE_PREFLIGHT_TIMEOUT_SECONDS = 8.0
_SOURCE_PREFLIGHT_CLEANUP_SECONDS = 2.0
_PLAYBACK_WATCHDOG_INTERVAL_SECONDS = 1.0
_PLAYBACK_STOPPED_GRACE_SECONDS = 2.0
_PLAYBACK_COMPLETION_GRACE_SECONDS = 30.0
_PLAYBACK_MAX_ACTIVE_SECONDS = 6 * 60 * 60
_EARLY_EOF_MINIMUM_EXPECTED_SECONDS = 15.0
_READ_ALOUD_LOUDNESS_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"
_MUSIC_DUCK_GAIN = 0.25


class _ManagedFFmpegOpusAudio(discord.FFmpegOpusAudio):
    """Close discord.py's child pipes even when FFmpeg already reached EOF."""

    def cleanup(self) -> None:
        streams = tuple(
            getattr(self, name, None)
            for name in ("_stdout", "_stdin", "_stderr")
        )
        super().cleanup()
        for stream in streams:
            close = getattr(stream, "close", None)
            if callable(close):
                with suppress(OSError):
                    close()


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
        self._preflight_executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"simajilord-audio-preflight-{guild_id}",
        )
        self._preflight_poisoned = False

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
            raise UserError("discord.voice_destination_invalid") from exc
        guild = self.bot.get_guild(self.guild_id)
        if guild is None:
            raise UserError("discord.guild_unavailable")
        channel = guild.get_channel(channel_id)
        if not isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
            raise UserError("discord.voice_channel_unavailable")

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
            raise UserError("discord.voice_connect_failed") from exc
        if not isinstance(protocol, discord.VoiceClient):
            await protocol.disconnect(force=True)
            raise ProviderError("Discord returned an unsupported voice protocol.")
        self._voice = protocol
        self.destination_id = channel.id

    async def play(self, item: AudioItem) -> None:
        voice = self._adopt_voice_client()
        if voice is None or not voice.is_connected():
            raise UserError("audio.output_disconnected")
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
            expected = max(
                0.0,
                (item.duration_seconds - item.start_seconds)
                / max(item.speed, 0.01),
            )
            try:
                await self._await_playback_completion(
                    completed,
                    voice=voice,
                    expected_seconds=expected,
                )
            except BaseException:
                if not completed.done():
                    completed.cancel()
                self._intentional_stop_generation += 1
                if voice.is_playing() or voice.is_paused():
                    voice.stop()
                raise
            elapsed = max(0.0, monotonic() - started_at)
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

    async def _await_playback_completion(
        self,
        completed: asyncio.Future[None],
        *,
        voice: discord.VoiceClient,
        expected_seconds: float,
    ) -> None:
        """Bound a missing callback without counting intentional pause time."""

        active_seconds = 0.0
        last_checked = monotonic()
        stopped_since: float | None = None
        maximum_active_seconds = min(
            _PLAYBACK_MAX_ACTIVE_SECONDS,
            (
                expected_seconds
                + max(
                    _PLAYBACK_COMPLETION_GRACE_SECONDS,
                    expected_seconds * 0.2,
                )
                if expected_seconds > 0
                else _PLAYBACK_MAX_ACTIVE_SECONDS
            ),
        )
        while True:
            try:
                async with asyncio.timeout(
                    _PLAYBACK_WATCHDOG_INTERVAL_SECONDS
                ):
                    await asyncio.shield(completed)
                return
            except TimeoutError:
                now = monotonic()
                if not voice.is_paused():
                    active_seconds += max(0.0, now - last_checked)
                last_checked = now
                if not voice.is_connected():
                    raise UserError("audio.output_disconnected") from None
                if not voice.is_playing() and not voice.is_paused():
                    if stopped_since is None:
                        stopped_since = now
                    elif now - stopped_since >= _PLAYBACK_STOPPED_GRACE_SECONDS:
                        raise ProviderError(
                            "Discord stopped playback without reporting completion."
                        ) from None
                else:
                    stopped_since = None
                if active_seconds >= maximum_active_seconds:
                    raise ProviderError(
                        "Discord audio playback exceeded its bounded completion window."
                    ) from None

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
            if self._preflight_poisoned:
                replacement.cleanup()
                raise ProviderError(
                    "The Discord audio preflight worker is unavailable until reconnect."
                )
            executor = self._preflight_executor
            if executor is None:
                executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix=(
                        f"simajilord-audio-preflight-{self.guild_id}"
                    ),
                )
                self._preflight_executor = executor
            read_future = asyncio.get_running_loop().run_in_executor(
                executor,
                replacement.read,
            )
            try:
                async with asyncio.timeout(_SOURCE_PREFLIGHT_TIMEOUT_SECONDS):
                    first_packet = await asyncio.shield(read_future)
                if not first_packet:
                    raise ProviderError(
                        "The replacement audio source produced no Opus packet."
                    )
                prepared = _PrefetchedAudioSource(replacement, first_packet)
                voice.source = prepared
            except TimeoutError as exc:
                await self._abort_preflight_reader(replacement, read_future)
                raise ProviderError(
                    "The replacement audio source preflight timed out."
                ) from exc
            except asyncio.CancelledError:
                if prepared is None:
                    await self._abort_preflight_reader(replacement, read_future)
                else:
                    prepared.cleanup()
                raise
            except BaseException:
                if prepared is None:
                    replacement.cleanup()
                else:
                    prepared.cleanup()
                raise
            if previous is not None and previous is not prepared:
                previous.cleanup()

    async def _abort_preflight_reader(
        self,
        replacement: discord.AudioSource,
        read_future: asyncio.Future[bytes],
    ) -> None:
        """Close a timed-out source and bound the dedicated reader cleanup."""

        try:
            replacement.cleanup()
        except Exception:
            self._preflight_poisoned = True
            log.exception("Discord replacement source cleanup failed")
        try:
            async with asyncio.timeout(_SOURCE_PREFLIGHT_CLEANUP_SECONDS):
                await asyncio.shield(read_future)
        except TimeoutError:
            self._preflight_poisoned = True
        except Exception:
            # Closing FFmpeg's pipes commonly makes the outstanding read fail;
            # the dedicated worker has still terminated and is reusable.
            pass

    def pause(self) -> None:
        voice = self._adopt_voice_client()
        if voice is None or not voice.is_playing():
            raise UserError("audio.nothing_playing")
        voice.pause()

    def resume(self) -> None:
        voice = self._adopt_voice_client()
        if voice is None or not voice.is_paused():
            raise UserError("audio.not_paused")
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
        executor = self._preflight_executor
        self._preflight_executor = None
        self._preflight_poisoned = False
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

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
    return _ManagedFFmpegOpusAudio(
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
