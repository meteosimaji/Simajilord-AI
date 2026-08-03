from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from dotenv import dotenv_values

from simajilord.agent import (
    AgentAutonomyMode,
    AgentAutonomyPolicyMode,
    AgentFileWorkspaceMode,
    AgentHighRiskAuthorizationMode,
    AgentInformationFlowMode,
    ReadAloudAudienceMode,
)
from simajilord.config import (
    AgentFeatureAccess,
    AgentSecurityPreset,
    effective_security_policy,
    load_settings,
    security_policy_warnings,
)
from simajilord.core.errors import ConfigurationError
from simajilord.integrations.discord.bot import _gateway_intents

_AGENT_ENVIRONMENT_NAMES = (
    "AGENT_SECURITY_PRESET",
    "AGENT_SECURITY_PRESET_EXPIRES_AT",
    "AGENT_ENABLED",
    "AGENT_ALLOWED_GUILD_IDS",
    "AGENT_TRUSTED_GUILD_IDS",
    "AGENT_ADMIN_USER_IDS",
    "AGENT_RATE_LIMIT_EXEMPT_USER_IDS",
    "AGENT_WEB_SEARCH_ACCESS",
    "AGENT_SAFE_COMPUTE_ACCESS",
    "AGENT_ISOLATED_SHELL_ACCESS",
    "AGENT_CONNECTOR_ACCESS",
    "IMAGE_GENERATION_ACCESS",
    "AGENT_FILE_SANDBOX_ENABLED",
    "AGENT_FILE_WORKSPACE_MODE",
    "AGENT_INFORMATION_FLOW_MODE",
    "AGENT_HIGH_RISK_AUTHORIZATION_MODE",
    "AGENT_HIGH_RISK_CONFIRMATION_TIMEOUT_SECONDS",
    "AGENT_CURATED_SKILLS_ENABLED",
    "AGENT_MODEL",
    "AGENT_ESCALATION_MODEL",
    "CODEX_EXPECTED_VERSION_PREFIX",
    "AGENT_REASONING_EFFORT",
    "AGENT_IDLE_TIMEOUT_SECONDS",
    "AGENT_TIMEOUT_SECONDS",
    "AGENT_MAX_RESPONSE_CHARACTERS",
    "AGENT_MAX_PENDING_TURNS",
    "MAX_ACTIVE_AGENT_TURNS",
    "MAX_PENDING_AGENT_TURNS",
    "MAX_PENDING_AGENT_TURNS_PER_USER",
    "AGENT_INTERACTIVE_RESERVE_PERCENT",
    "AGENT_CONVERSATION_COMPATIBILITY_EPOCH",
    "AGENT_AUTONOMY_ENABLED",
    "AGENT_AUTONOMY_GUILD_IDS",
    "AGENT_AUTONOMY_MODE",
    "AGENT_AUTONOMY_POLICY_MODE",
    "AGENT_AUTONOMY_BATCH_SECONDS",
    "AGENT_AUTONOMY_INTERVAL_SECONDS",
    "AGENT_AUTONOMY_MAX_RUNS",
    "AGENT_AUTONOMY_CANDIDATE_LIMIT",
    "AGENT_AUTONOMY_MAX_PENDING_EVENTS",
    "AGENT_AUTONOMY_MAX_PENDING_EVENTS_PER_CHANNEL",
    "AGENT_AUTONOMY_MAX_PENDING_EVENTS_PER_ACTOR",
    "WEB_SEARCH_BASE_URL",
    "WEB_SEARCH_SHARED_SECRET",
    "WEB_REQUEST_TIMEOUT_SECONDS",
    "WEB_FETCH_MAX_BYTES",
    "HIVE_API_KEY",
    "HIVE_DAILY_LIMIT",
    "HIVE_TIMEOUT_SECONDS",
    "HIVE_MAX_MEDIA_BYTES",
    "HIVE_THRESHOLD",
    "TTS_PROVIDER",
    "TTS_VOICE",
    "VOICEVOX_BASE_URL",
    "VOICEVOX_SPEAKER_ID",
    "VOICEVOX_PRESET_CLEAR_ID",
    "VOICEVOX_PRESET_CALM_ID",
    "VOICEVOX_PRESET_ENERGETIC_ID",
    "VOICEVOX_PRESET_CUTE_ID",
    "VOICEVOX_PRESET_NARRATOR_ID",
    "VOICEVOX_ENGINE_PATH",
    "VOICEVOX_AUTO_START",
    "VOICEVOX_TIMEOUT_SECONDS",
    "VOICEVOX_READINESS_TTL_SECONDS",
    "TRANSLATION_HELPER_PATH",
    "READ_ALOUD_CHUNK_CHARACTERS",
    "READ_ALOUD_AUDIENCE_MODE",
    "MAX_PENDING_MUSIC",
    "MAX_PENDING_MUSIC_PER_USER",
    "MAX_CONCURRENT_MEDIA",
    "MAX_CONCURRENT_MEDIA_PER_GUILD",
    "DISCORD_EMOJI_LOADING_ID",
    "DISCORD_EMOJI_SUCCESS_ID",
    "DISCORD_EMOJI_WARNING_ID",
    "DISCORD_EMOJI_AUDIO_WAVE_ID",
    "DISCORD_EMOJI_RADIO_ID",
    "DISCORD_MEMBERS_INTENT_ENABLED",
    "DISCORD_PRESENCE_INTENT_ENABLED",
    "DISCORD_ACTIVITY_ENABLED",
    "DISCORD_CLIENT_SECRET",
    "DISCORD_ACTIVITY_HOST",
    "DISCORD_ACTIVITY_PORT",
    "DATA_RETENTION_DAYS",
    "MAX_DATA_SIZE_GB",
    "LOCAL_MEDIA_SIZE_GB",
    "LOCAL_MEDIA_CACHE_BYTES",
)


def _prepare_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    for name in _AGENT_ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "123")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "missing.env"


def test_checked_in_env_example_loads_without_optional_voicevox_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    example = Path(__file__).parents[1] / ".env.example"
    values = dotenv_values(example)
    for name in values:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    settings = load_settings(dotenv_path=example)

    assert settings.tts_provider == "macos"
    assert settings.voicevox_auto_start is False
    assert settings.agent_model == "gpt-5.6-luna"
    assert settings.agent_escalation_model == "gpt-5.6-terra"
    assert settings.agent_reasoning_effort == "high"
    assert settings.agent_idle_timeout_seconds == 600
    assert settings.agent_max_response_characters == 7_600
    assert settings.agent_max_active_turns == 4
    assert settings.agent_max_pending_turns == 20
    assert settings.agent_max_pending_turns_per_user == 2
    assert settings.agent_interactive_reserve_percent == 25
    assert settings.agent_conversation_compatibility_epoch == 6
    assert settings.agent_autonomy_enabled is False
    assert settings.agent_autonomy_mode is AgentAutonomyMode.OBSERVE
    assert settings.agent_autonomy_policy_mode is AgentAutonomyPolicyMode.STRICT
    assert settings.agent_autonomy_batch_seconds == 10
    assert settings.agent_autonomy_max_runs == 10
    assert settings.agent_autonomy_max_pending_events_per_actor == 50
    assert settings.agent_security_preset is AgentSecurityPreset.GUILD_ASSISTANT
    assert settings.agent_effective_security_preset is AgentSecurityPreset.GUILD_ASSISTANT
    assert settings.agent_security_preset_expires_at is None
    assert settings.agent_security_preset_expired is False
    assert settings.agent_isolated_shell_access is AgentFeatureAccess.DISABLED
    assert settings.agent_connector_access is AgentFeatureAccess.DISABLED
    assert settings.codex_expected_version_prefix == "0.146."
    assert settings.discord_members_intent_enabled is False
    assert settings.discord_presence_intent_enabled is False


def test_gateway_privileged_intents_are_explicit_opt_ins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    defaults = load_settings(dotenv_path=dotenv_path)
    default_intents = _gateway_intents(defaults)

    assert defaults.agent_max_response_characters == 7_600
    assert default_intents.message_content is True
    assert default_intents.members is False
    assert default_intents.presences is False
    assert default_intents.guilds is True
    assert default_intents.voice_states is True

    monkeypatch.setenv("DISCORD_MEMBERS_INTENT_ENABLED", "true")
    monkeypatch.setenv("DISCORD_PRESENCE_INTENT_ENABLED", "true")
    opted_in = _gateway_intents(load_settings(dotenv_path=dotenv_path))
    assert opted_in.message_content is True
    assert opted_in.members is True
    assert opted_in.presences is True


def test_agent_idle_watchdog_prefers_explicit_name_and_accepts_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "125")

    legacy = load_settings(dotenv_path=dotenv_path)
    assert legacy.agent_idle_timeout_seconds == 125

    monkeypatch.setenv("AGENT_IDLE_TIMEOUT_SECONDS", "90")
    explicit = load_settings(dotenv_path=dotenv_path)
    assert explicit.agent_idle_timeout_seconds == 90


def test_agent_security_policies_are_explicit_and_typed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_ALLOWED_GUILD_IDS", "10,20")
    monkeypatch.setenv("AGENT_TRUSTED_GUILD_IDS", "10")
    monkeypatch.setenv("AGENT_ADMIN_USER_IDS", "30")
    monkeypatch.setenv("AGENT_RATE_LIMIT_EXEMPT_USER_IDS", "40")
    monkeypatch.setenv("AGENT_WEB_SEARCH_ACCESS", "everyone")
    monkeypatch.setenv("AGENT_SAFE_COMPUTE_ACCESS", "admins")
    monkeypatch.setenv("AGENT_ISOLATED_SHELL_ACCESS", "admins")
    monkeypatch.setenv("AGENT_CONNECTOR_ACCESS", "admins")
    monkeypatch.setenv("AGENT_FILE_SANDBOX_ENABLED", "true")

    settings = load_settings(dotenv_path=dotenv_path)

    assert settings.agent_allowed_guild_ids == frozenset({"10", "20"})
    assert settings.agent_trusted_guild_ids == frozenset({"10"})
    assert settings.agent_admin_user_ids == frozenset({"30"})
    assert settings.agent_rate_limit_exempt_user_ids == frozenset({"40"})
    assert settings.agent_web_search_access is AgentFeatureAccess.EVERYONE
    assert settings.agent_safe_compute_access is AgentFeatureAccess.ADMINS
    assert settings.agent_isolated_shell_access is AgentFeatureAccess.ADMINS
    assert settings.agent_connector_access is AgentFeatureAccess.ADMINS
    assert settings.agent_file_sandbox_enabled is True
    assert settings.agent_file_workspace_mode is AgentFileWorkspaceMode.ACTOR_TASK
    assert settings.agent_information_flow_mode is AgentInformationFlowMode.ENFORCE
    assert settings.agent_high_risk_authorization_mode is AgentHighRiskAuthorizationMode.BOUND_ONCE
    assert settings.agent_high_risk_confirmation_timeout_seconds == 120
    assert settings.agent_curated_skills_enabled is False
    assert settings.web_search_base_url == "http://127.0.0.1:8888"
    assert settings.web_search_shared_secret is None
    assert settings.hive_api_key is None
    assert settings.hive_daily_limit == 100
    assert settings.hive_threshold == 0.9
    assert settings.read_aloud_chunk_characters == 400
    assert settings.read_aloud_audience_mode is ReadAloudAudienceMode.ENFORCE
    assert settings.voicevox_readiness_ttl_seconds == 5.0
    assert settings.max_pending_music == 100
    assert settings.max_pending_music_per_user == 20
    assert settings.max_concurrent_media == 4
    assert settings.max_concurrent_media_per_guild == 2
    assert settings.data_retention_days == 30
    assert settings.max_data_size_bytes == 10_000_000_000
    assert settings.local_media_cache_bytes == 5_000_000_000
    assert settings.discord_emoji_loading_id is None
    assert settings.discord_emoji_success_id is None
    assert settings.discord_emoji_warning_id is None
    assert settings.discord_emoji_audio_wave_id is None
    assert settings.discord_emoji_radio_id is None
    assert settings.activity_enabled is False
    assert settings.activity_client_secret is None
    assert security_policy_warnings(settings) == ()


@pytest.mark.parametrize(
    ("preset", "workspace_mode", "web_access", "shell_access"),
    (
        (
            AgentSecurityPreset.GUILD_ASSISTANT,
            AgentFileWorkspaceMode.ACTOR_TASK,
            AgentFeatureAccess.DISABLED,
            AgentFeatureAccess.DISABLED,
        ),
        (
            AgentSecurityPreset.TRUSTED_ADMIN,
            AgentFileWorkspaceMode.ACTOR,
            AgentFeatureAccess.ADMINS,
            AgentFeatureAccess.ADMINS,
        ),
        (
            AgentSecurityPreset.PERSONAL_LAB,
            AgentFileWorkspaceMode.ACTOR,
            AgentFeatureAccess.EVERYONE,
            AgentFeatureAccess.EVERYONE,
        ),
    ),
)
def test_reviewed_security_presets_supply_typed_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    preset: AgentSecurityPreset,
    workspace_mode: AgentFileWorkspaceMode,
    web_access: AgentFeatureAccess,
    shell_access: AgentFeatureAccess,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_SECURITY_PRESET", preset.value)
    if preset is AgentSecurityPreset.TRUSTED_ADMIN:
        monkeypatch.setenv("AGENT_ADMIN_USER_IDS", "30")

    settings = load_settings(dotenv_path=dotenv_path)

    assert settings.agent_security_preset is preset
    assert settings.agent_effective_security_preset is preset
    assert settings.agent_file_workspace_mode is workspace_mode
    assert settings.agent_web_search_access is web_access
    assert settings.agent_isolated_shell_access is shell_access
    assert settings.agent_autonomy_enabled is False
    assert settings.agent_autonomy_mode is AgentAutonomyMode.OBSERVE


def test_legacy_compatibility_requires_expiry_and_reverts_to_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    now = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
    monkeypatch.setenv(
        "AGENT_SECURITY_PRESET",
        AgentSecurityPreset.LEGACY_COMPATIBILITY.value,
    )

    with pytest.raises(ConfigurationError, match="EXPIRES_AT is required"):
        load_settings(dotenv_path=dotenv_path, now=now)

    monkeypatch.setenv(
        "AGENT_SECURITY_PRESET_EXPIRES_AT",
        (now + timedelta(hours=2)).isoformat(),
    )
    active = load_settings(dotenv_path=dotenv_path, now=now)
    assert active.agent_effective_security_preset is AgentSecurityPreset.LEGACY_COMPATIBILITY
    assert active.agent_file_workspace_mode is AgentFileWorkspaceMode.GUILD_SHARED
    assert active.agent_information_flow_mode is AgentInformationFlowMode.DISABLED
    assert active.agent_high_risk_authorization_mode is AgentHighRiskAuthorizationMode.LEGACY_EVENT
    assert active.agent_autonomy_enabled is False
    assert active.agent_autonomy_policy_mode is AgentAutonomyPolicyMode.LEGACY

    expired = load_settings(dotenv_path=dotenv_path, now=now + timedelta(hours=3))
    assert expired.agent_security_preset is AgentSecurityPreset.LEGACY_COMPATIBILITY
    assert expired.agent_effective_security_preset is AgentSecurityPreset.GUILD_ASSISTANT
    assert expired.agent_security_preset_expired is True
    assert expired.agent_file_workspace_mode is AgentFileWorkspaceMode.ACTOR_TASK
    assert expired.agent_information_flow_mode is AgentInformationFlowMode.ENFORCE
    assert expired.agent_high_risk_authorization_mode is AgentHighRiskAuthorizationMode.BOUND_ONCE
    assert expired.agent_autonomy_policy_mode is AgentAutonomyPolicyMode.STRICT
    assert any("expired" in item for item in security_policy_warnings(expired))


def test_security_preset_individual_environment_overrides_are_effective(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_SECURITY_PRESET", "personal_lab")
    monkeypatch.setenv("AGENT_ISOLATED_SHELL_ACCESS", "disabled")
    monkeypatch.setenv("AGENT_CONNECTOR_ACCESS", "admins")
    monkeypatch.setenv("AGENT_ADMIN_USER_IDS", "30")
    monkeypatch.setenv("AGENT_FILE_WORKSPACE_MODE", "actor_task")

    settings = load_settings(dotenv_path=dotenv_path)
    policy = effective_security_policy(settings)

    assert settings.agent_isolated_shell_access is AgentFeatureAccess.DISABLED
    assert settings.agent_connector_access is AgentFeatureAccess.ADMINS
    assert settings.agent_file_workspace_mode is AgentFileWorkspaceMode.ACTOR_TASK
    assert policy.effective_preset == "personal_lab"
    assert policy.shell_access == "disabled"
    assert policy.connector_access == "admins"
    assert policy.file_workspace_mode == "actor_task"
    assert policy.override_names == (
        "AGENT_ISOLATED_SHELL_ACCESS",
        "AGENT_CONNECTOR_ACCESS",
        "AGENT_FILE_WORKSPACE_MODE",
    )
    rendered = repr(policy)
    assert "30" not in rendered
    assert "test-token" not in rendered


def test_safe_compute_requires_the_isolated_file_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_SAFE_COMPUTE_ACCESS", "everyone")
    monkeypatch.setenv("AGENT_FILE_SANDBOX_ENABLED", "false")

    with pytest.raises(
        ConfigurationError,
        match="AGENT_FILE_SANDBOX_ENABLED must be true",
    ):
        load_settings(dotenv_path=dotenv_path)


def test_autonomy_mode_batching_and_unbounded_runs_are_typed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_ALLOWED_GUILD_IDS", "10")
    monkeypatch.setenv("AGENT_AUTONOMY_GUILD_IDS", "10")
    monkeypatch.setenv("AGENT_AUTONOMY_MODE", "assist")
    monkeypatch.setenv("AGENT_AUTONOMY_ENABLED", "true")
    monkeypatch.setenv("AGENT_AUTONOMY_BATCH_SECONDS", "5")
    monkeypatch.setenv("AGENT_AUTONOMY_MAX_RUNS", "0")

    settings = load_settings(dotenv_path=dotenv_path)

    assert settings.agent_autonomy_enabled is True
    assert settings.agent_autonomy_mode is AgentAutonomyMode.ASSIST
    assert settings.agent_autonomy_policy_mode is AgentAutonomyPolicyMode.STRICT
    assert settings.agent_autonomy_batch_seconds == 5
    assert settings.agent_autonomy_max_runs == 0

    monkeypatch.setenv("AGENT_AUTONOMY_MODE", "unrestricted")
    with pytest.raises(ConfigurationError, match="observe, assist, act"):
        load_settings(dotenv_path=dotenv_path)


def test_security_policy_compatibility_modes_are_typed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("READ_ALOUD_AUDIENCE_MODE", "audit")
    monkeypatch.setenv("AGENT_FILE_WORKSPACE_MODE", "guild_shared")
    monkeypatch.setenv("AGENT_INFORMATION_FLOW_MODE", "disabled")
    monkeypatch.setenv("AGENT_HIGH_RISK_AUTHORIZATION_MODE", "legacy_event")
    monkeypatch.setenv("AGENT_AUTONOMY_POLICY_MODE", "legacy")

    settings = load_settings(dotenv_path=dotenv_path)

    assert settings.read_aloud_audience_mode is ReadAloudAudienceMode.AUDIT
    assert settings.agent_file_workspace_mode is AgentFileWorkspaceMode.GUILD_SHARED
    assert settings.agent_information_flow_mode is AgentInformationFlowMode.DISABLED
    assert (
        settings.agent_high_risk_authorization_mode is AgentHighRiskAuthorizationMode.LEGACY_EVENT
    )
    assert settings.agent_autonomy_policy_mode is AgentAutonomyPolicyMode.LEGACY
    warnings = security_policy_warnings(settings)
    assert any(
        "AGENT_FILE_WORKSPACE_MODE=guild_shared" in warning
        and "Per-actor ownership checks remain enforced" in warning
        for warning in warnings
    )
    assert any("Information-flow enforcement is disabled" in item for item in warnings)
    assert any("Read-aloud audience enforcement is audit" in item for item in warnings)
    assert any("legacy_event" in item for item in warnings)
    assert any("Unsafe combination: guild_shared" in item for item in warnings)


def test_unsafe_autonomy_and_effect_combinations_are_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_ADMIN_USER_IDS", "30")
    monkeypatch.setenv("AGENT_FILE_SANDBOX_ENABLED", "true")
    monkeypatch.setenv("AGENT_ISOLATED_SHELL_ACCESS", "admins")
    monkeypatch.setenv("AGENT_HIGH_RISK_AUTHORIZATION_MODE", "legacy_event")
    monkeypatch.setenv("AGENT_AUTONOMY_ENABLED", "true")
    monkeypatch.setenv("AGENT_AUTONOMY_MODE", "act")
    monkeypatch.setenv("AGENT_AUTONOMY_POLICY_MODE", "legacy")

    warnings = security_policy_warnings(load_settings(dotenv_path=dotenv_path))

    assert any("shell or connector effects" in item for item in warnings)
    assert any("autonomy act and the legacy" in item for item in warnings)


def test_autonomy_per_channel_queue_cannot_exceed_global_queue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_AUTONOMY_MAX_PENDING_EVENTS", "5")
    monkeypatch.setenv("AGENT_AUTONOMY_MAX_PENDING_EVENTS_PER_CHANNEL", "6")

    with pytest.raises(
        ConfigurationError,
        match="MAX_PENDING_EVENTS_PER_CHANNEL must not exceed",
    ):
        load_settings(dotenv_path=dotenv_path)


def test_autonomy_per_actor_queue_cannot_exceed_global_queue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_AUTONOMY_MAX_PENDING_EVENTS", "5")
    monkeypatch.setenv("AGENT_AUTONOMY_MAX_PENDING_EVENTS_PER_CHANNEL", "5")
    monkeypatch.setenv("AGENT_AUTONOMY_MAX_PENDING_EVENTS_PER_ACTOR", "6")

    with pytest.raises(
        ConfigurationError,
        match="MAX_PENDING_EVENTS_PER_ACTOR must not exceed",
    ):
        load_settings(dotenv_path=dotenv_path)


def test_activity_requires_a_hidden_client_secret_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCORD_ACTIVITY_ENABLED", "true")

    with pytest.raises(
        ConfigurationError,
        match="DISCORD_CLIENT_SECRET is required",
    ):
        load_settings(dotenv_path=dotenv_path)

    monkeypatch.setenv("DISCORD_CLIENT_SECRET", "private-activity-secret")
    settings = load_settings(dotenv_path=dotenv_path)

    assert settings.activity_enabled is True
    assert settings.activity_client_secret == "private-activity-secret"
    assert settings.activity_host == "127.0.0.1"
    assert settings.activity_port == 8_787
    assert "private-activity-secret" not in repr(settings)


def test_application_emoji_ids_are_optional_validated_snowflakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCORD_EMOJI_LOADING_ID", "123456789")

    settings = load_settings(dotenv_path=dotenv_path)

    assert settings.discord_emoji_loading_id == 123456789

    monkeypatch.setenv("DISCORD_EMOJI_LOADING_ID", "loading")
    with pytest.raises(
        ConfigurationError,
        match="DISCORD_EMOJI_LOADING_ID must be a positive Discord ID",
    ):
        load_settings(dotenv_path=dotenv_path)


def test_media_per_guild_concurrency_cannot_exceed_global_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("MAX_CONCURRENT_MEDIA", "2")
    monkeypatch.setenv("MAX_CONCURRENT_MEDIA_PER_GUILD", "3")

    with pytest.raises(
        ConfigurationError,
        match="MAX_CONCURRENT_MEDIA_PER_GUILD cannot exceed",
    ):
        load_settings(dotenv_path=dotenv_path)


def test_hive_secret_is_hidden_and_daily_limit_cannot_exceed_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("HIVE_API_KEY", "private-hive-key")
    settings = load_settings(dotenv_path=dotenv_path)
    assert settings.hive_api_key == "private-hive-key"
    assert "private-hive-key" not in repr(settings)

    monkeypatch.setenv("HIVE_DAILY_LIMIT", "101")
    with pytest.raises(ConfigurationError, match="between 1 and 100"):
        load_settings(dotenv_path=dotenv_path)


def test_trusted_agent_guild_must_also_be_allowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_ALLOWED_GUILD_IDS", "10")
    monkeypatch.setenv("AGENT_TRUSTED_GUILD_IDS", "20")

    with pytest.raises(
        ConfigurationError,
        match="AGENT_TRUSTED_GUILD_IDS must be a subset",
    ):
        load_settings(dotenv_path=dotenv_path)


def test_admin_only_feature_requires_a_fixed_admin_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_WEB_SEARCH_ACCESS", "admins")

    with pytest.raises(
        ConfigurationError,
        match="AGENT_ADMIN_USER_IDS is required",
    ):
        load_settings(dotenv_path=dotenv_path)


def test_remote_web_search_backend_requires_https(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("WEB_SEARCH_BASE_URL", "http://search.example.com")

    with pytest.raises(
        ConfigurationError,
        match="must use HTTPS",
    ):
        load_settings(dotenv_path=dotenv_path)


def test_voicevox_settings_require_loopback_and_executable_for_auto_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    engine = tmp_path / "voicevox-run"
    engine.write_text("#!/bin/sh\n", encoding="utf-8")
    engine.chmod(0o700)
    monkeypatch.setenv("TTS_PROVIDER", "voicevox")
    monkeypatch.setenv("VOICEVOX_ENGINE_PATH", str(engine))
    monkeypatch.setenv("VOICEVOX_SPEAKER_ID", "14")

    settings = load_settings(dotenv_path=dotenv_path)

    assert settings.tts_provider == "voicevox"
    assert settings.voicevox_engine_path == engine
    assert settings.voicevox_speaker_id == 14
    assert settings.voicevox_auto_start is True

    monkeypatch.setenv("VOICEVOX_BASE_URL", "https://voice.example.com")
    with pytest.raises(ConfigurationError, match="loopback HTTP URL"):
        load_settings(dotenv_path=dotenv_path)


def test_translation_helper_requires_private_execute_permissions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    helper = tmp_path / "TranslationHelper"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("TRANSLATION_HELPER_PATH", str(helper))

    helper.chmod(0o600)
    with pytest.raises(ConfigurationError, match="chmod 700"):
        load_settings(dotenv_path=dotenv_path)

    helper.chmod(0o755)
    with pytest.raises(ConfigurationError, match="chmod 700"):
        load_settings(dotenv_path=dotenv_path)

    helper.chmod(0o700)
    settings = load_settings(dotenv_path=dotenv_path)
    assert settings.translation_helper_path == helper.resolve()
