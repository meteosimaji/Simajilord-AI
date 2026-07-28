from __future__ import annotations

from pathlib import Path

import pytest

from simajilord.config import AgentFeatureAccess, load_settings
from simajilord.core.errors import ConfigurationError

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
    "AGENT_AUTONOMY_ENABLED",
    "AGENT_AUTONOMY_GUILD_IDS",
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
