"""Composition root for the model-independent Simajilord platform."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic

from simajilord.capabilities import (
    build_audio_endpoints,
    build_download_endpoint,
    build_read_aloud_endpoint,
    build_system_endpoints,
    build_utility_endpoints,
)
from simajilord.capabilities.status import build_status_endpoint
from simajilord.config import Settings
from simajilord.core.capabilities import CapabilityRegistry
from simajilord.media.providers import YtDlpProvider
from simajilord.observability import EventJournal
from simajilord.providers.speech import MacOSSayProvider
from simajilord.services import (
    AudioSessionManager,
    AudioStateStore,
    MediaService,
    ReadAloudService,
    SpeechService,
)


@dataclass(slots=True)
class SimajilordRuntime:
    """Shared services and endpoints consumed by all current and future adapters."""

    settings: Settings
    registry: CapabilityRegistry
    media: MediaService
    audio: AudioSessionManager
    speech: SpeechService
    read_aloud: ReadAloudService
    journal: EventJournal
    started_at: datetime
    started_monotonic: float

    @classmethod
    def build(cls, settings: Settings) -> SimajilordRuntime:
        media = MediaService(
            YtDlpProvider(
                cookie_file=settings.media_cookie_file,
                download_timeout_seconds=settings.download_timeout_seconds,
            )
        )
        audio = AudioSessionManager(
            max_active=settings.max_active_voice_guilds,
            max_pending_speech=settings.max_pending_speech,
            resolver=media.resolve_audio,
            state_store=AudioStateStore(settings.data_dir / "audio_sessions.json"),
        )
        speech = SpeechService(
            MacOSSayProvider(settings.tts_voice),
            output_dir=settings.data_dir / "speech",
            max_characters=settings.max_read_aloud_characters,
            max_concurrent=settings.max_concurrent_tts,
        )
        read_aloud = ReadAloudService(settings.data_dir / "read_aloud.json")
        journal = EventJournal(settings.data_dir / "events.sqlite3")
        registry = CapabilityRegistry(journal=journal)
        started_at = datetime.now(UTC)
        started_monotonic = monotonic()
        runtime = cls(
            settings=settings,
            registry=registry,
            media=media,
            audio=audio,
            speech=speech,
            read_aloud=read_aloud,
            journal=journal,
            started_at=started_at,
            started_monotonic=started_monotonic,
        )
        for item in (
            *build_system_endpoints(
                registry,
                started_at=started_at,
                started_monotonic=started_monotonic,
            ),
            *build_utility_endpoints(),
            *build_audio_endpoints(media, audio),
            build_download_endpoint(media),
            build_read_aloud_endpoint(read_aloud),
        ):
            registry.register(item)
        registry.register(build_status_endpoint(registry, journal, audio))
        return runtime

    async def close(self) -> None:
        await self.audio.close()
