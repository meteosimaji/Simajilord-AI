"""Composition root for the model-independent Simajilord platform."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from simajilord.agent import (
    AGENT_AUDIO_CONTROL_CAPABILITIES,
    AGENT_AUDIO_GRANT,
    AGENT_AUDIO_WRITE_CAPABILITIES,
    AGENT_COMPUTE_GRANT,
    AGENT_DISCORD_DESTRUCTIVE_CAPABILITIES,
    AGENT_DISCORD_MODERATION_CAPABILITIES,
    AGENT_DISCORD_REQUESTED_WRITE_CAPABILITIES,
    AGENT_FILE_GRANT,
    AGENT_HIVE_GRANT,
    AGENT_IMAGE_GRANT,
    AGENT_MEDIA_GRANT,
    AGENT_MEMORY_GRANT,
    AGENT_MEMORY_WRITE_CAPABILITIES,
    AGENT_MESSAGE_GRANT,
    AGENT_MODERATION_GRANT,
    AGENT_QUOTE_GRANT,
    AGENT_REACTION_GRANT,
    AGENT_REPOST_GRANT,
    AGENT_REQUESTED_WRITE_CAPABILITIES,
    AGENT_WEB_GRANT,
    ActionReceiptService,
    ActionReceiptStore,
    AgentAutonomyMode,
    AgentMemoryService,
    AgentMemoryStore,
    AutonomyEnqueueResult,
    AutonomyEventKind,
    AutonomyEventQueue,
    build_action_undo_endpoint,
    build_curated_workflow_endpoint,
    build_memory_endpoints,
)
from simajilord.agent.providers import CodexAppServerProvider
from simajilord.agent.service import AgentLimits, AgentService
from simajilord.agent.store import AgentConversationStore
from simajilord.agent.tools import AgentToolCatalog
from simajilord.capabilities import (
    build_audio_endpoints,
    build_compute_endpoints,
    build_download_endpoint,
    build_file_endpoints,
    build_focus_timer_endpoints,
    build_image_endpoints,
    build_media_save_endpoint,
    build_moderation_endpoints,
    build_read_aloud_endpoint,
    build_read_aloud_policy_endpoints,
    build_read_aloud_route_endpoints,
    build_speech_endpoint,
    build_system_endpoints,
    build_translation_endpoints,
    build_utility_endpoints,
    build_web_endpoints,
)
from simajilord.capabilities.status import build_status_endpoint
from simajilord.config import AgentFeatureAccess, Settings
from simajilord.core.capabilities import CapabilityRegistry
from simajilord.domain.audio import AudioItem, AudioQueueLane
from simajilord.media.providers import RoutingMediaProvider, YtDlpProvider
from simajilord.observability import EventJournal
from simajilord.providers.image import IdeogramMlxProvider
from simajilord.providers.moderation import HiveSyntheticMediaProvider
from simajilord.providers.speech import MacOSSayProvider, VoicevoxSpeechProvider
from simajilord.providers.translation import MacOSTranslationProvider
from simajilord.providers.web import AiohttpPublicWebFetcher, SearxngSearchProvider
from simajilord.services import (
    AgentFileSandbox,
    AudioSessionManager,
    AudioStateStore,
    DataMaintenanceService,
    FocusTimerService,
    ImageGenerationService,
    ImageGenerationStore,
    LocalMediaStore,
    MediaPriority,
    MediaService,
    ModerationService,
    ModerationStore,
    QuoteImageService,
    ReadAloudService,
    ServiceOperationMetric,
    SpeechService,
    TranslationService,
    TranslationStore,
    WebService,
    WorkspaceComputeService,
)


@dataclass(slots=True)
class SimajilordRuntime:
    """Shared services and endpoints consumed by all current and future adapters."""

    settings: Settings
    registry: CapabilityRegistry
    media: MediaService
    local_media: LocalMediaStore
    audio: AudioSessionManager
    focus_timer: FocusTimerService
    speech: SpeechService
    read_aloud: ReadAloudService
    web: WebService
    moderation: ModerationService
    image: ImageGenerationService
    quote: QuoteImageService
    translation: TranslationService
    files: AgentFileSandbox | None
    compute: WorkspaceComputeService | None
    memory: AgentMemoryService
    journal: EventJournal
    autonomy_events: AutonomyEventQueue
    action_receipts: ActionReceiptService | None
    agent_store: AgentConversationStore
    maintenance: DataMaintenanceService
    agent: AgentService | None
    started_at: datetime
    started_monotonic: float

    @classmethod
    def build(cls, settings: Settings) -> SimajilordRuntime:
        journal = EventJournal(settings.data_dir / "events.sqlite3")
        autonomy_events = AutonomyEventQueue(
            settings.data_dir / "agent_autonomy.sqlite3",
            max_pending_events=settings.agent_autonomy_max_pending_events,
            max_pending_events_per_channel=(
                settings.agent_autonomy_max_pending_events_per_channel
            ),
            max_pending_events_per_actor=(
                settings.agent_autonomy_max_pending_events_per_actor
            ),
        )
        agent_store = AgentConversationStore(
            settings.data_dir / "agent_conversations.sqlite3"
        )
        memory_store = AgentMemoryStore(
            settings.data_dir / "agent_memory.sqlite3"
        )
        memory = AgentMemoryService(memory_store)

        async def record_service_metric(metric: ServiceOperationMetric) -> None:
            await journal.append(
                kind="service.operation",
                workspace_id=metric.workspace_id,
                payload={
                    "operation": metric.operation,
                    "wait_ms": round(metric.wait_ms, 3),
                    "duration_ms": round(metric.duration_ms, 3),
                    "outcome": metric.outcome,
                    "resource_id": metric.resource_id,
                },
            )
            if (
                settings.agent_enabled
                and settings.agent_autonomy_enabled
                and settings.agent_autonomy_mode is not AgentAutonomyMode.OBSERVE
                and metric.workspace_id in settings.agent_autonomy_guild_ids
                and metric.outcome in {"failed", "fallback_standalone"}
                and metric.operation.startswith(("audio.", "voice."))
            ):
                session = audio.find(metric.workspace_id)
                channel_id = metric.resource_id or (
                    session.destination_id if session is not None else None
                )
                if channel_id is not None:
                    occurred_at = datetime.now(UTC)
                    failure_bucket = int(occurred_at.timestamp()) // 30
                    enqueue_result = await autonomy_events.enqueue(
                        kind=AutonomyEventKind.AUDIO_ERROR,
                        deduplication_key=(
                            f"audio-error:{metric.workspace_id}:{metric.operation}:"
                            f"{metric.outcome}:{failure_bucket}"
                        ),
                        workspace_id=metric.workspace_id,
                        channel_id=channel_id,
                        actor_id=None,
                        message_id=None,
                        occurred_at=occurred_at,
                        payload={
                            "operation": metric.operation,
                            "outcome": metric.outcome,
                        },
                    )
                    if enqueue_result in {
                        AutonomyEnqueueResult.QUEUE_FULL,
                        AutonomyEnqueueResult.CHANNEL_QUEUE_FULL,
                    }:
                        await journal.append(
                            kind="agent.autonomy.event_rejected",
                            workspace_id=metric.workspace_id,
                            transport="agent",
                            request_id=(
                                f"audio-error:{metric.workspace_id}:"
                                f"{metric.operation}:{failure_bucket}"
                            ),
                            payload={
                                "event_kind": AutonomyEventKind.AUDIO_ERROR.value,
                                "channel_id": channel_id,
                                "reason": enqueue_result.value,
                            },
                        )

        local_media = LocalMediaStore(
            settings.data_dir / "local_media",
            max_file_bytes=settings.local_media_max_file_bytes,
            max_cache_bytes=settings.local_media_cache_bytes,
            max_duration_seconds=settings.local_media_max_duration_seconds,
            audio_state_path=settings.data_dir / "audio_sessions.json",
        )
        media = MediaService(
            RoutingMediaProvider(
                YtDlpProvider(
                    cookie_file=settings.media_cookie_file,
                    download_timeout_seconds=settings.download_timeout_seconds,
                ),
                local_media,
            ),
            max_concurrent=settings.max_concurrent_media,
            max_per_workspace=settings.max_concurrent_media_per_guild,
            metric_hook=record_service_metric,
        )

        async def supply_autoplay(
            workspace_id: str,
            seeds: tuple[str, ...],
            limit: int,
        ) -> tuple[AudioItem, ...]:
            candidates = await media.mix_audio(
                seeds,
                limit=limit,
                workspace_id=workspace_id,
                priority=MediaPriority.BACKGROUND,
            )
            requested_at = int(datetime.now(UTC).timestamp())
            return tuple(
                AudioItem(
                    source="",
                    title=candidate.title,
                    page_url=candidate.reference,
                    duration_seconds=candidate.duration_seconds,
                    resolver_reference=candidate.reference,
                    queue_lane=AudioQueueLane.AUTOPLAY,
                    request_source="youtube_mix",
                    requested_at_epoch=requested_at,
                    uploader=candidate.uploader,
                    thumbnail_url=candidate.thumbnail_url,
                )
                for candidate in candidates
            )

        async def resolve_for_workspace(
            workspace_id: str,
            reference: str,
        ) -> AudioItem:
            return await media.resolve_audio(
                reference,
                workspace_id=workspace_id,
                priority=MediaPriority.INTERACTIVE,
            )

        audio = AudioSessionManager(
            max_active=settings.max_active_voice_guilds,
            max_pending_speech=settings.max_pending_speech,
            max_pending_music=settings.max_pending_music,
            max_pending_music_per_actor=settings.max_pending_music_per_user,
            workspace_resolver=resolve_for_workspace,
            workspace_autoplay_supplier=supply_autoplay,
            state_store=AudioStateStore(settings.data_dir / "audio_sessions.json"),
            metric_hook=record_service_metric,
        )
        speech_provider = (
            VoicevoxSpeechProvider(
                base_url=settings.voicevox_base_url,
                speaker_id=settings.voicevox_speaker_id,
                timeout_seconds=settings.voicevox_timeout_seconds,
                engine_path=settings.voicevox_engine_path,
                auto_start=settings.voicevox_auto_start,
                readiness_ttl_seconds=settings.voicevox_readiness_ttl_seconds,
            )
            if settings.tts_provider == "voicevox"
            else MacOSSayProvider(settings.tts_voice)
        )
        speech = SpeechService(
            speech_provider,
            output_dir=settings.data_dir / "speech",
            chunk_characters=settings.read_aloud_chunk_characters,
            max_concurrent=settings.max_concurrent_tts,
            max_provider_calls=settings.max_concurrent_tts_provider_calls,
            metric_hook=record_service_metric,
            voice_presets={
                "clear": settings.voicevox_preset_clear_id,
                "calm": settings.voicevox_preset_calm_id,
                "energetic": settings.voicevox_preset_energetic_id,
                "cute": settings.voicevox_preset_cute_id,
                "narrator": settings.voicevox_preset_narrator_id,
            },
            file_suffix=".wav" if settings.tts_provider == "voicevox" else ".aiff",
        )
        focus_timer = FocusTimerService(settings.data_dir / "focus_timers.sqlite3")
        read_aloud = ReadAloudService(settings.data_dir / "read_aloud.json")
        web = WebService(
            search_provider=SearxngSearchProvider(
                base_url=settings.web_search_base_url,
                timeout_seconds=settings.web_request_timeout_seconds,
                shared_secret=settings.web_search_shared_secret,
            ),
            page_fetcher=AiohttpPublicWebFetcher(
                timeout_seconds=settings.web_request_timeout_seconds,
            ),
            max_fetch_bytes=settings.web_fetch_max_bytes,
        )
        moderation_provider = (
            HiveSyntheticMediaProvider(
                api_key=settings.hive_api_key,
                timeout_seconds=settings.hive_timeout_seconds,
            )
            if settings.hive_api_key is not None
            else None
        )
        moderation_store = ModerationStore(settings.data_dir / "moderation.sqlite3")
        moderation = ModerationService(
            provider=moderation_provider,
            store=moderation_store,
            daily_limit=settings.hive_daily_limit,
            max_media_bytes=settings.hive_max_media_bytes,
            threshold=settings.hive_threshold,
        )
        image_provider = (
            IdeogramMlxProvider(
                model_path=settings.image_model_path,
                mflux_source=settings.image_mflux_source,
                timeout_seconds=settings.image_timeout_seconds,
                mlx_cache_limit_gb=settings.image_mlx_cache_limit_gb,
            )
            if settings.image_model_path is not None
            else None
        )
        image_store = ImageGenerationStore(
            settings.data_dir / "image_generation.sqlite3"
        )
        image_output_dir = settings.data_dir / "generated_images"
        image = ImageGenerationService(
            provider=image_provider,
            store=image_store,
            journal=journal,
            output_dir=image_output_dir,
            per_user_requests=settings.image_per_user_requests,
            per_user_window_seconds=settings.image_per_user_window_seconds,
            per_workspace_requests=settings.image_per_workspace_requests,
            per_workspace_window_seconds=settings.image_per_workspace_window_seconds,
            max_pending_jobs=settings.image_max_pending_jobs,
            rate_limit_exempt_actor_ids=settings.agent_rate_limit_exempt_user_ids,
        )
        quote = QuoteImageService()
        translation_package = (
            Path(__file__).resolve().parents[2]
            / "native"
            / "macos"
            / "TranslationHelper"
        )
        translation = TranslationService(
            (
                MacOSTranslationProvider(
                    translation_package,
                    executable_path=settings.translation_helper_path,
                    timeout_seconds=settings.translation_timeout_seconds,
                ),
            )
            if settings.translation_enabled and sys.platform == "darwin"
            else (),
            max_characters=settings.translation_max_characters,
            store=TranslationStore(settings.data_dir / "translations.sqlite3"),
        )
        files = (
            AgentFileSandbox(settings.data_dir / "agent_files")
            if settings.agent_file_sandbox_enabled
            else None
        )
        compute = (
            WorkspaceComputeService(
                files=files,
                run_root=settings.data_dir / "agent_compute",
                web_fetcher=web.page_fetcher,
                max_download_bytes=settings.web_fetch_max_bytes,
            )
            if (
                files is not None
                and settings.agent_safe_compute_access
                is not AgentFeatureAccess.DISABLED
            )
            else None
        )
        registry = CapabilityRegistry(journal=journal)
        action_receipts: ActionReceiptService | None = None
        agent: AgentService | None = None
        curated_workflow_endpoint = None
        if settings.agent_enabled:
            action_receipts = ActionReceiptService(
                store=ActionReceiptStore(
                    settings.data_dir / "agent_actions.sqlite3"
                ),
                registry=registry,
                journal=journal,
            )
            agent_capabilities = [
                "action.undo",
                "audio.history",
                "audio.queue",
                "audio.search",
                "discord.list_servers",
                "discord.inspect_server",
                "discord.inspect_user",
                "discord.get_message",
                "discord.list_channels",
                "discord.list_archived_threads",
                "discord.list_roles",
                "discord.search_messages",
                "discord.read_aloud_status",
                "discord.read_aloud_add_sources",
                "discord.read_aloud_remove_source",
                "discord.read_aloud_disable",
                "discord.read_aloud_policy_status",
                "discord.read_aloud_dictionary_list",
                "discord.read_aloud_dictionary_set",
                "discord.read_aloud_dictionary_remove",
                "discord.read_aloud_exclusion_set",
                "discord.read_aloud_announcements_set",
                "discord.read_aloud_semantics_set",
                "discord.read_aloud_content_mode_set",
                "discord.play_audio",
                "discord.play_attachment",
                *AGENT_AUDIO_CONTROL_CAPABILITIES,
                "discord.read_messages",
                "discord.add_reaction",
                *AGENT_DISCORD_REQUESTED_WRITE_CAPABILITIES,
                "discord.delete_own_message",
                "discord.remove_own_reaction",
                "discord.send_message",
                "discord.speak",
                "discord.translate_message",
                "discord.post_expanded_message",
                "discord.create_quote_image",
                "discord.view_custom_emoji",
                "discord.view_image_attachment",
                "discord.view_sticker",
                "timer.create",
                "timer.list",
                "timer.cancel",
                "memory.search",
                *AGENT_MEMORY_WRITE_CAPABILITIES,
                "moderation.status",
                "system.ping",
                "system.status",
                "system.uptime",
                "translation.detect",
                "translation.languages",
                "translation.translate",
                "translation.translate_batch",
                "utility.choose",
                "utility.roll",
                "web.search",
                "web.fetch",
                "web.find",
                "web.status",
            ]
            required_grants: dict[str, str] = {
                "action.undo": AGENT_MESSAGE_GRANT,
                "memory.search": AGENT_MEMORY_GRANT,
                **{
                    name: AGENT_MEMORY_GRANT
                    for name in AGENT_MEMORY_WRITE_CAPABILITIES
                },
                "audio.history": AGENT_AUDIO_GRANT,
                "audio.queue": AGENT_AUDIO_GRANT,
                "audio.search": AGENT_AUDIO_GRANT,
                "discord.play_audio": AGENT_AUDIO_GRANT,
                "discord.play_attachment": AGENT_AUDIO_GRANT,
                **{
                    name: AGENT_AUDIO_GRANT
                    for name in AGENT_AUDIO_CONTROL_CAPABILITIES
                },
                "discord.read_aloud_status": AGENT_AUDIO_GRANT,
                "discord.read_aloud_add_sources": AGENT_AUDIO_GRANT,
                "discord.read_aloud_remove_source": AGENT_AUDIO_GRANT,
                "discord.read_aloud_disable": AGENT_AUDIO_GRANT,
                "discord.read_aloud_policy_status": AGENT_AUDIO_GRANT,
                "discord.read_aloud_dictionary_list": AGENT_AUDIO_GRANT,
                "discord.read_aloud_dictionary_set": AGENT_AUDIO_GRANT,
                "discord.read_aloud_dictionary_remove": AGENT_AUDIO_GRANT,
                "discord.read_aloud_exclusion_set": AGENT_AUDIO_GRANT,
                "discord.read_aloud_announcements_set": AGENT_AUDIO_GRANT,
                "discord.read_aloud_semantics_set": AGENT_AUDIO_GRANT,
                "discord.read_aloud_content_mode_set": AGENT_AUDIO_GRANT,
                "discord.speak": AGENT_AUDIO_GRANT,
                "discord.send_message": AGENT_MESSAGE_GRANT,
                **{
                    name: (
                        AGENT_AUDIO_GRANT
                        if name == "discord.connect_voice"
                        else (
                            AGENT_MODERATION_GRANT
                            if name in AGENT_DISCORD_MODERATION_CAPABILITIES
                            else AGENT_MESSAGE_GRANT
                        )
                    )
                    for name in AGENT_DISCORD_REQUESTED_WRITE_CAPABILITIES
                },
                "discord.add_reaction": AGENT_REACTION_GRANT,
                "discord.delete_own_message": AGENT_MESSAGE_GRANT,
                "discord.remove_own_reaction": AGENT_REACTION_GRANT,
                "discord.post_expanded_message": AGENT_REPOST_GRANT,
                "discord.create_quote_image": AGENT_QUOTE_GRANT,
                "discord.view_image_attachment": AGENT_MESSAGE_GRANT,
                "timer.create": AGENT_MESSAGE_GRANT,
                "timer.list": AGENT_MESSAGE_GRANT,
                "timer.cancel": AGENT_MESSAGE_GRANT,
                "web.search": AGENT_WEB_GRANT,
                "web.fetch": AGENT_WEB_GRANT,
                "web.find": AGENT_WEB_GRANT,
            }
            if settings.hive_api_key is not None:
                capability_name = "discord.analyze_attachment"
                agent_capabilities.append(capability_name)
                required_grants[capability_name] = AGENT_HIVE_GRANT
            if (
                image.provider is not None
                and settings.image_generation_access is not AgentFeatureAccess.DISABLED
            ):
                for capability_name in ("image.generate", "image.status"):
                    agent_capabilities.append(capability_name)
                    required_grants[capability_name] = AGENT_IMAGE_GRANT
            if files is not None:
                file_capabilities = (
                    "files.list",
                    "files.read",
                    "files.write_text",
                    "files.replace_text",
                    "discord.import_attachment",
                    "discord.send_file",
                )
                agent_capabilities.extend(file_capabilities)
                required_grants.update(
                    {name: AGENT_FILE_GRANT for name in file_capabilities}
                )
                agent_capabilities.append("media.save")
                required_grants["media.save"] = AGENT_MEDIA_GRANT
            if compute is not None:
                compute_capabilities = (
                    "compute.run",
                    "files.download_url",
                )
                agent_capabilities.extend(compute_capabilities)
                required_grants.update(
                    {
                        name: AGENT_COMPUTE_GRANT
                        for name in compute_capabilities
                    }
                )
            if settings.agent_curated_skills_enabled:
                curated_workflow_endpoint = build_curated_workflow_endpoint(
                    frozenset(agent_capabilities),
                    capability_grants=required_grants,
                    approval_capabilities=frozenset(
                        AGENT_REQUESTED_WRITE_CAPABILITIES
                    ),
                )
                agent_capabilities.append(
                    curated_workflow_endpoint.descriptor.name
                )
            agent_tools = AgentToolCatalog(
                registry,
                tuple(agent_capabilities),
                required_grants=required_grants,
                eager_capabilities=(
                    "action.undo",
                    "discord.list_servers",
                    "discord.list_channels",
                    "discord.list_archived_threads",
                    "discord.list_roles",
                    "discord.get_message",
                    "discord.read_messages",
                    "discord.search_messages",
                    "discord.add_reaction",
                    "discord.delete_own_message",
                    "discord.remove_own_reaction",
                    "discord.send_message",
                    "discord.view_image_attachment",
                    "memory.search",
                    "web.search",
                    "web.fetch",
                    "web.find",
                    *(
                        ("workflow.search",)
                        if curated_workflow_endpoint is not None
                        else ()
                    ),
                    *(
                        (
                            "files.list",
                            "files.read",
                            "discord.import_attachment",
                            "discord.send_file",
                        )
                        if files is not None
                        else ()
                    ),
                    *(
                        ("compute.run", "files.download_url")
                        if compute is not None
                        else ()
                    ),
                ),
                write_capabilities=(
                    (
                        "action.undo",
                        "discord.send_message",
                        "discord.add_reaction",
                        "discord.delete_own_message",
                        "discord.remove_own_reaction",
                        "discord.post_expanded_message",
                        "discord.create_quote_image",
                        "timer.create",
                        "timer.cancel",
                        *AGENT_DISCORD_REQUESTED_WRITE_CAPABILITIES,
                        *AGENT_MEMORY_WRITE_CAPABILITIES,
                        *AGENT_AUDIO_WRITE_CAPABILITIES,
                    )
                    + (
                        ("image.generate",)
                        if "image.generate" in agent_capabilities
                        else ()
                    )
                    + (
                        ("discord.analyze_attachment",)
                        if "discord.analyze_attachment" in agent_capabilities
                        else ()
                    )
                    + (
                        (
                            "files.write_text",
                            "files.replace_text",
                            "discord.import_attachment",
                            "discord.send_file",
                            "media.save",
                        )
                        if files is not None
                        else ()
                    )
                    + (
                        ("compute.run", "files.download_url")
                        if compute is not None
                        else ()
                    )
                ),
                destructive_capabilities=AGENT_DISCORD_DESTRUCTIVE_CAPABILITIES,
                image_output_capabilities=(
                    "discord.view_custom_emoji",
                    "discord.view_image_attachment",
                    "discord.view_sticker",
                ),
                action_receipts=action_receipts,
            )
            agent = AgentService(
                provider=CodexAppServerProvider(
                    executable=settings.codex_executable,
                    model=settings.agent_model,
                    workspace_dir=settings.data_dir / "agent_workspace",
                    timeout_seconds=settings.agent_timeout_seconds,
                    reasoning_effort=settings.agent_reasoning_effort,
                    tools=agent_tools,
                    max_tool_calls=settings.agent_max_tool_calls,
                    max_tool_output_characters=settings.agent_max_tool_output_characters,
                    escalation_model=settings.agent_escalation_model,
                ),
                store=agent_store,
                journal=journal,
                limits=AgentLimits(
                    per_user_requests=settings.agent_per_user_requests,
                    per_user_window_seconds=settings.agent_per_user_window_seconds,
                    per_workspace_requests=settings.agent_per_workspace_requests,
                    per_workspace_window_seconds=settings.agent_per_workspace_window_seconds,
                    max_tokens_per_24_hours=settings.agent_max_tokens_per_24_hours,
                    max_conversation_turns=settings.agent_max_conversation_turns,
                    max_context_ratio=settings.agent_max_context_ratio,
                    max_response_characters=settings.agent_max_response_characters,
                    max_active_turns=settings.agent_max_active_turns,
                    max_pending_turns=settings.agent_max_pending_turns,
                    max_pending_turns_per_user=(
                        settings.agent_max_pending_turns_per_user
                    ),
                    rate_limit_exempt_actor_ids=(
                        settings.agent_rate_limit_exempt_user_ids
                    ),
                ),
            )
        maintenance = DataMaintenanceService(
            data_dir=settings.data_dir,
            retention_days=settings.data_retention_days,
            max_data_bytes=settings.max_data_size_bytes,
            journal=journal,
            agent_store=agent_store,
            memory_store=memory_store,
            focus_timers=focus_timer,
            image_store=image_store,
            image_output_dir=image_output_dir,
            moderation_store=moderation_store,
            speech=speech,
            local_media=local_media,
        )
        started_at = datetime.now(UTC)
        started_monotonic = monotonic()
        runtime = cls(
            settings=settings,
            registry=registry,
            media=media,
            local_media=local_media,
            audio=audio,
            focus_timer=focus_timer,
            speech=speech,
            read_aloud=read_aloud,
            web=web,
            moderation=moderation,
            image=image,
            quote=quote,
            translation=translation,
            files=files,
            compute=compute,
            memory=memory,
            journal=journal,
            autonomy_events=autonomy_events,
            action_receipts=action_receipts,
            agent_store=agent_store,
            maintenance=maintenance,
            agent=agent,
            started_at=started_at,
            started_monotonic=started_monotonic,
        )
        for item in (
            *build_system_endpoints(
                registry,
                started_at=started_at,
                started_monotonic=started_monotonic,
            ),
            *build_translation_endpoints(translation),
            *build_utility_endpoints(),
            *build_audio_endpoints(media, audio),
            *build_focus_timer_endpoints(focus_timer, read_aloud),
            build_download_endpoint(media),
            *(
                (build_media_save_endpoint(media, files),)
                if files is not None
                else ()
            ),
            build_read_aloud_endpoint(read_aloud),
            *build_read_aloud_route_endpoints(read_aloud),
            *build_read_aloud_policy_endpoints(read_aloud),
            build_speech_endpoint(speech, audio),
            *build_web_endpoints(web),
            *build_moderation_endpoints(moderation),
            *build_image_endpoints(image),
            *(build_file_endpoints(files) if files is not None else ()),
            *(build_compute_endpoints(compute) if compute is not None else ()),
            *build_memory_endpoints(memory),
            *((curated_workflow_endpoint,) if curated_workflow_endpoint else ()),
            *(
                (build_action_undo_endpoint(action_receipts),)
                if action_receipts is not None
                else ()
            ),
        ):
            registry.register(item)
        registry.register(
            build_status_endpoint(
                registry,
                journal,
                audio,
                web,
                maintenance,
                agent_enabled=settings.agent_enabled,
                speech_provider=settings.tts_provider,
                speech_voice=(
                    f"style {settings.voicevox_speaker_id}"
                    if settings.tts_provider == "voicevox"
                    else settings.tts_voice
                ),
            )
        )
        return runtime

    async def close(self) -> None:
        if self.agent is not None:
            await self.agent.close()
        await self.image.close()
        await self.moderation.close()
        await self.web.close()
        await self.translation.close()
        await self.audio.close()
        await self.media.close()
        await self.speech.close()
