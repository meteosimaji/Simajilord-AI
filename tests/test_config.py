from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import dotenv_values

from simajilord.agent import AgentAutonomyMode
from simajilord.config import AgentFeatureAccess, load_settings
from simajilord.core.errors import ConfigurationError
from simajilord.integrations.discord.bot import _gateway_intents

_AGENT_ENVIRONMENT_NAMES = (
    "AGENT_ENABLED",
    "AGENT_ALLOWED_GUILD_IDS",
    "AGENT_TRUSTED_GUILD_IDS",
    "AGENT_ADMIN_USER_IDS",
    "AGENT_RATE_LIMIT_EXEMPT_USER_IDS",
    "AGENT_WEB_SEARCH_ACCESS",
    "AGENT_SAFE_COMPUTE_ACCESS",
    "AGENT_FILE_SANDBOX_ENABLED",
    "AGENT_CURATED_SKILLS_ENABLED",
    "AGENT_MODEL",
    "AGENT_ESCALATION_MODEL",
    "AGENT_REASONING_EFFORT",
    "AGENT_IDLE_TIMEOUT_SECONDS",
    "AGENT_TIMEOUT_SECONDS",
    "AGENT_MAX_TOOL_CALLS",
    "AGENT_MAX_TOOL_OUTPUT_CHARACTERS",
    "AGENT_MAX_PENDING_TURNS",
    "MAX_ACTIVE_AGENT_TURNS",
    "MAX_PENDING_AGENT_TURNS",
    "MAX_PENDING_AGENT_TURNS_PER_USER",
    "AGENT_AUTONOMY_ENABLED",
    "AGENT_AUTONOMY_GUILD_IDS",
    "AGENT_AUTONOMY_MODE",
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
    "READ_ALOUD_CHUNK_CHARACTERS",
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
    assert settings.agent_model == "gpt-5.6-sol"
    assert settings.agent_escalation_model == "gpt-5.6-sol"
    assert settings.agent_reasoning_effort == "medium"
    assert settings.agent_idle_timeout_seconds == 600
    assert settings.agent_max_tool_calls == 32
    assert settings.agent_max_tool_output_characters == 24_000
    assert settings.agent_max_active_turns == 4
    assert settings.agent_max_pending_turns == 20
    assert settings.agent_max_pending_turns_per_user == 2
    assert settings.agent_autonomy_enabled is True
    assert settings.agent_autonomy_mode is AgentAutonomyMode.ACT
    assert settings.agent_autonomy_batch_seconds == 10
    assert settings.agent_autonomy_max_runs == 0
    assert settings.agent_autonomy_max_pending_events_per_actor == 50
    assert settings.discord_members_intent_enabled is False
    assert settings.discord_presence_intent_enabled is False


def test_gateway_privileged_intents_are_explicit_opt_ins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    defaults = load_settings(dotenv_path=dotenv_path)
    default_intents = _gateway_intents(defaults)

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
    monkeypatch.setenv("AGENT_FILE_SANDBOX_ENABLED", "true")

    settings = load_settings(dotenv_path=dotenv_path)

    assert settings.agent_allowed_guild_ids == frozenset({"10", "20"})
    assert settings.agent_trusted_guild_ids == frozenset({"10"})
    assert settings.agent_admin_user_ids == frozenset({"30"})
    assert settings.agent_rate_limit_exempt_user_ids == frozenset({"40"})
    assert settings.agent_web_search_access is AgentFeatureAccess.EVERYONE
    assert settings.agent_safe_compute_access is AgentFeatureAccess.ADMINS
    assert settings.agent_file_sandbox_enabled is True
    assert settings.agent_curated_skills_enabled is False
    assert settings.web_search_base_url == "http://127.0.0.1:8888"
    assert settings.web_search_shared_secret is None
    assert settings.hive_api_key is None
    assert settings.hive_daily_limit == 100
    assert settings.hive_threshold == 0.9
    assert settings.read_aloud_chunk_characters == 400
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
    monkeypatch.setenv("AGENT_AUTONOMY_BATCH_SECONDS", "5")
    monkeypatch.setenv("AGENT_AUTONOMY_MAX_RUNS", "0")

    settings = load_settings(dotenv_path=dotenv_path)

    assert settings.agent_autonomy_enabled is True
    assert settings.agent_autonomy_mode is AgentAutonomyMode.ASSIST
    assert settings.agent_autonomy_batch_seconds == 5
    assert settings.agent_autonomy_max_runs == 0

    monkeypatch.setenv("AGENT_AUTONOMY_MODE", "unrestricted")
    with pytest.raises(ConfigurationError, match="observe, assist, act"):
        load_settings(dotenv_path=dotenv_path)


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


def test_agent_tool_output_budget_accepts_eighty_thousand_character_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_path = _prepare_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_MAX_TOOL_OUTPUT_CHARACTERS", "80000")

    settings = load_settings(dotenv_path=dotenv_path)
    assert settings.agent_max_tool_output_characters == 80_000

    monkeypatch.setenv("AGENT_MAX_TOOL_OUTPUT_CHARACTERS", "80001")
    with pytest.raises(ConfigurationError, match="must be between 500 and 80000"):
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
