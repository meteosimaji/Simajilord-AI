"""Validated local configuration."""

from __future__ import annotations

import os
import stat
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlsplit

from dotenv import load_dotenv

from .agent.contracts import (
    AgentAutonomyMode,
    AgentAutonomyPolicyMode,
    AgentFileWorkspaceMode,
    AgentHighRiskAuthorizationMode,
    AgentInformationFlowMode,
    ReadAloudAudienceMode,
)
from .core.errors import ConfigurationError


class CommandScope(StrEnum):
    """Where application commands are synchronized."""

    GUILD = "guild"
    GLOBAL = "global"


class AgentFeatureAccess(StrEnum):
    """Who may use an optional agent capability."""

    DISABLED = "disabled"
    ADMINS = "admins"
    EVERYONE = "everyone"


class AgentSecurityPreset(StrEnum):
    """Reviewed starting points for the agent's effective security boundary."""

    PERSONAL_LAB = "personal_lab"
    TRUSTED_ADMIN = "trusted_admin"
    GUILD_ASSISTANT = "guild_assistant"
    LEGACY_COMPATIBILITY = "legacy_compatibility"


@dataclass(frozen=True, slots=True)
class EffectiveSecurityPolicy:
    """Secret-free effective policy suitable for logs and status presenters."""

    configured_preset: str
    effective_preset: str
    preset_expires_at_epoch: int | None
    preset_expired: bool
    override_names: tuple[str, ...]
    agent_enabled: bool
    file_sandbox_enabled: bool
    file_workspace_mode: str
    information_flow_mode: str
    read_aloud_audience_mode: str
    high_risk_authorization_mode: str
    autonomy_enabled: bool
    autonomy_mode: str
    autonomy_policy_mode: str
    autonomy_max_runs: int
    web_access: str
    compute_access: str
    image_access: str
    shell_access: str
    connector_access: str
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SecurityPresetDefaults:
    file_sandbox_enabled: bool
    file_workspace_mode: AgentFileWorkspaceMode
    information_flow_mode: AgentInformationFlowMode
    read_aloud_audience_mode: ReadAloudAudienceMode
    high_risk_authorization_mode: AgentHighRiskAuthorizationMode
    autonomy_enabled: bool
    autonomy_mode: AgentAutonomyMode
    autonomy_policy_mode: AgentAutonomyPolicyMode
    autonomy_max_runs: int
    web_access: AgentFeatureAccess
    compute_access: AgentFeatureAccess
    image_access: AgentFeatureAccess
    shell_access: AgentFeatureAccess
    connector_access: AgentFeatureAccess


_SECURITY_PRESETS: dict[AgentSecurityPreset, _SecurityPresetDefaults] = {
    AgentSecurityPreset.GUILD_ASSISTANT: _SecurityPresetDefaults(
        file_sandbox_enabled=False,
        file_workspace_mode=AgentFileWorkspaceMode.ACTOR_TASK,
        information_flow_mode=AgentInformationFlowMode.ENFORCE,
        read_aloud_audience_mode=ReadAloudAudienceMode.ENFORCE,
        high_risk_authorization_mode=AgentHighRiskAuthorizationMode.BOUND_ONCE,
        autonomy_enabled=False,
        autonomy_mode=AgentAutonomyMode.OBSERVE,
        autonomy_policy_mode=AgentAutonomyPolicyMode.STRICT,
        autonomy_max_runs=10,
        web_access=AgentFeatureAccess.DISABLED,
        compute_access=AgentFeatureAccess.DISABLED,
        image_access=AgentFeatureAccess.DISABLED,
        shell_access=AgentFeatureAccess.DISABLED,
        connector_access=AgentFeatureAccess.DISABLED,
    ),
    AgentSecurityPreset.TRUSTED_ADMIN: _SecurityPresetDefaults(
        file_sandbox_enabled=True,
        file_workspace_mode=AgentFileWorkspaceMode.ACTOR,
        information_flow_mode=AgentInformationFlowMode.ENFORCE,
        read_aloud_audience_mode=ReadAloudAudienceMode.ENFORCE,
        high_risk_authorization_mode=AgentHighRiskAuthorizationMode.BOUND_ONCE,
        autonomy_enabled=False,
        autonomy_mode=AgentAutonomyMode.OBSERVE,
        autonomy_policy_mode=AgentAutonomyPolicyMode.STRICT,
        autonomy_max_runs=10,
        web_access=AgentFeatureAccess.ADMINS,
        compute_access=AgentFeatureAccess.ADMINS,
        image_access=AgentFeatureAccess.ADMINS,
        shell_access=AgentFeatureAccess.ADMINS,
        connector_access=AgentFeatureAccess.ADMINS,
    ),
    AgentSecurityPreset.PERSONAL_LAB: _SecurityPresetDefaults(
        file_sandbox_enabled=True,
        file_workspace_mode=AgentFileWorkspaceMode.ACTOR,
        information_flow_mode=AgentInformationFlowMode.ENFORCE,
        read_aloud_audience_mode=ReadAloudAudienceMode.ENFORCE,
        high_risk_authorization_mode=AgentHighRiskAuthorizationMode.BOUND_ONCE,
        autonomy_enabled=False,
        autonomy_mode=AgentAutonomyMode.OBSERVE,
        autonomy_policy_mode=AgentAutonomyPolicyMode.STRICT,
        autonomy_max_runs=10,
        web_access=AgentFeatureAccess.EVERYONE,
        compute_access=AgentFeatureAccess.EVERYONE,
        image_access=AgentFeatureAccess.EVERYONE,
        shell_access=AgentFeatureAccess.EVERYONE,
        connector_access=AgentFeatureAccess.EVERYONE,
    ),
    AgentSecurityPreset.LEGACY_COMPATIBILITY: _SecurityPresetDefaults(
        file_sandbox_enabled=True,
        file_workspace_mode=AgentFileWorkspaceMode.GUILD_SHARED,
        information_flow_mode=AgentInformationFlowMode.DISABLED,
        read_aloud_audience_mode=ReadAloudAudienceMode.DISABLED,
        high_risk_authorization_mode=AgentHighRiskAuthorizationMode.LEGACY_EVENT,
        autonomy_enabled=False,
        autonomy_mode=AgentAutonomyMode.OBSERVE,
        autonomy_policy_mode=AgentAutonomyPolicyMode.LEGACY,
        autonomy_max_runs=0,
        web_access=AgentFeatureAccess.EVERYONE,
        compute_access=AgentFeatureAccess.EVERYONE,
        image_access=AgentFeatureAccess.EVERYONE,
        shell_access=AgentFeatureAccess.EVERYONE,
        connector_access=AgentFeatureAccess.EVERYONE,
    ),
}

_SECURITY_OVERRIDE_NAMES = (
    "AGENT_WEB_SEARCH_ACCESS",
    "AGENT_SAFE_COMPUTE_ACCESS",
    "IMAGE_GENERATION_ACCESS",
    "AGENT_ISOLATED_SHELL_ACCESS",
    "AGENT_CONNECTOR_ACCESS",
    "AGENT_FILE_SANDBOX_ENABLED",
    "AGENT_FILE_WORKSPACE_MODE",
    "AGENT_INFORMATION_FLOW_MODE",
    "READ_ALOUD_AUDIENCE_MODE",
    "AGENT_HIGH_RISK_AUTHORIZATION_MODE",
    "AGENT_AUTONOMY_ENABLED",
    "AGENT_AUTONOMY_MODE",
    "AGENT_AUTONOMY_POLICY_MODE",
    "AGENT_AUTONOMY_MAX_RUNS",
)


PolicyModeT = TypeVar("PolicyModeT", bound=StrEnum)


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings. Secret fields are excluded from representations."""

    token: str = field(repr=False)
    application_id: int
    discord_members_intent_enabled: bool
    discord_presence_intent_enabled: bool
    discord_emoji_loading_id: int | None
    discord_emoji_success_id: int | None
    discord_emoji_warning_id: int | None
    discord_emoji_audio_wave_id: int | None
    discord_emoji_radio_id: int | None
    activity_enabled: bool
    activity_client_secret: str | None = field(repr=False)
    activity_host: str
    activity_port: int
    command_scope: CommandScope
    command_prefix: str
    log_level: str
    data_dir: Path
    discord_agent_workspace_dir: Path
    data_retention_days: int
    max_data_size_bytes: int
    media_cookie_file: Path | None
    tts_provider: str
    tts_voice: str
    voicevox_base_url: str
    voicevox_speaker_id: int
    voicevox_preset_clear_id: int
    voicevox_preset_calm_id: int
    voicevox_preset_energetic_id: int
    voicevox_preset_cute_id: int
    voicevox_preset_narrator_id: int
    voicevox_engine_path: Path | None
    voicevox_auto_start: bool
    voicevox_timeout_seconds: float
    voicevox_readiness_ttl_seconds: float
    read_aloud_chunk_characters: int
    read_aloud_audience_mode: ReadAloudAudienceMode
    max_pending_speech: int
    max_pending_music: int
    max_pending_music_per_user: int
    max_concurrent_tts: int
    max_concurrent_tts_provider_calls: int
    max_concurrent_media: int
    max_concurrent_media_per_guild: int
    max_active_voice_guilds: int
    download_timeout_seconds: float
    local_media_max_file_bytes: int
    local_media_cache_bytes: int
    local_media_max_duration_seconds: float
    translation_enabled: bool
    translation_helper_path: Path | None
    translation_timeout_seconds: float
    translation_max_characters: int
    web_search_base_url: str
    web_search_shared_secret: str | None = field(repr=False)
    web_request_timeout_seconds: float
    web_fetch_max_bytes: int
    hive_api_key: str | None = field(repr=False)
    hive_daily_limit: int
    hive_timeout_seconds: float
    hive_max_media_bytes: int
    hive_threshold: float
    image_timeout_seconds: float
    image_generation_access: AgentFeatureAccess
    image_per_user_requests: int
    image_per_user_window_seconds: int
    image_per_workspace_requests: int
    image_per_workspace_window_seconds: int
    image_max_pending_jobs: int
    agent_security_preset: AgentSecurityPreset
    agent_effective_security_preset: AgentSecurityPreset
    agent_security_preset_expires_at: datetime | None
    agent_security_preset_expired: bool
    agent_security_override_names: tuple[str, ...]
    agent_enabled: bool
    agent_allowed_guild_ids: frozenset[str]
    agent_trusted_guild_ids: frozenset[str]
    agent_admin_user_ids: frozenset[str]
    agent_rate_limit_exempt_user_ids: frozenset[str]
    agent_web_search_access: AgentFeatureAccess
    agent_safe_compute_access: AgentFeatureAccess
    agent_isolated_shell_access: AgentFeatureAccess
    agent_connector_access: AgentFeatureAccess
    agent_file_sandbox_enabled: bool
    agent_file_workspace_mode: AgentFileWorkspaceMode
    agent_information_flow_mode: AgentInformationFlowMode
    agent_high_risk_authorization_mode: AgentHighRiskAuthorizationMode
    agent_high_risk_confirmation_timeout_seconds: int
    agent_curated_skills_enabled: bool
    agent_provider: str
    agent_model: str
    agent_escalation_model: str
    codex_executable: str
    codex_expected_version_prefix: str
    agent_idle_timeout_seconds: float
    agent_reasoning_effort: str
    agent_per_user_requests: int
    agent_per_user_window_seconds: int
    agent_per_workspace_requests: int
    agent_per_workspace_window_seconds: int
    agent_max_tokens_per_24_hours: int
    agent_max_response_characters: int
    agent_max_active_turns: int
    agent_max_pending_turns: int
    agent_max_pending_turns_per_user: int
    agent_interactive_reserve_percent: int
    agent_conversation_compatibility_epoch: int
    agent_autonomy_enabled: bool
    agent_autonomy_guild_ids: frozenset[str]
    agent_autonomy_mode: AgentAutonomyMode
    agent_autonomy_policy_mode: AgentAutonomyPolicyMode
    agent_autonomy_batch_seconds: int
    agent_autonomy_max_runs: int
    agent_autonomy_candidate_limit: int
    agent_autonomy_max_pending_events: int
    agent_autonomy_max_pending_events_per_channel: int
    agent_autonomy_max_pending_events_per_actor: int


def security_policy_warnings(settings: Settings) -> tuple[str, ...]:
    """Describe effective combinations that need immediate operator attention."""

    warnings: list[str] = []
    if settings.agent_security_preset_expired:
        warnings.append(
            "The configured security preset expired and its preset defaults were "
            "replaced by guild_assistant. Review any individual environment overrides."
        )
    elif settings.agent_security_preset is AgentSecurityPreset.LEGACY_COMPATIBILITY:
        expires_at = settings.agent_security_preset_expires_at
        assert expires_at is not None
        warnings.append(
            "legacy_compatibility is active only until "
            f"{expires_at.isoformat().replace('+00:00', 'Z')}; it restores wider "
            "ambient authority and must remain temporary."
        )
    if settings.agent_file_workspace_mode is AgentFileWorkspaceMode.GUILD_SHARED:
        warnings.append(
            "AGENT_FILE_WORKSPACE_MODE=guild_shared uses one physical guild file "
            "namespace. Per-actor ownership checks remain enforced, but name "
            "collisions and trusted-internal maintenance paths have a wider blast "
            "radius; prefer actor or actor_task when compatibility is unnecessary."
        )
    if settings.agent_information_flow_mode is not AgentInformationFlowMode.ENFORCE:
        warnings.append(
            "Information-flow enforcement is "
            f"{settings.agent_information_flow_mode.value}; broader or uncertain "
            "source-to-target disclosures are not blocked by the strict policy kernel."
        )
    if settings.read_aloud_audience_mode is not ReadAloudAudienceMode.ENFORCE:
        warnings.append(
            "Read-aloud audience enforcement is "
            f"{settings.read_aloud_audience_mode.value}; voice listeners may receive "
            "content they cannot read in its source channel."
        )
    if settings.agent_high_risk_authorization_mode is AgentHighRiskAuthorizationMode.LEGACY_EVENT:
        warnings.append(
            "High-risk authorization uses legacy_event instead of a one-use binding "
            "to the exact capability, arguments, target, and message revision."
        )
    if settings.agent_isolated_shell_access is AgentFeatureAccess.EVERYONE:
        warnings.append(
            "The isolated host shell is granted to everyone; Seatbelt limits impact "
            "but does not make arbitrary process execution a low-risk capability."
        )
    if settings.agent_connector_access is AgentFeatureAccess.EVERYONE:
        warnings.append(
            "External connector writes are granted to everyone; connector contracts "
            "and receipts do not replace a narrow requester policy."
        )
    if (
        settings.agent_file_workspace_mode is AgentFileWorkspaceMode.GUILD_SHARED
        and settings.agent_information_flow_mode is not AgentInformationFlowMode.ENFORCE
    ):
        warnings.append(
            "Unsafe combination: guild_shared files and non-enforcing information "
            "flow widen both the data namespace and its possible disclosure path."
        )
    if (
        settings.agent_high_risk_authorization_mode is AgentHighRiskAuthorizationMode.LEGACY_EVENT
        and (
            settings.agent_isolated_shell_access is not AgentFeatureAccess.DISABLED
            or settings.agent_connector_access is not AgentFeatureAccess.DISABLED
        )
    ):
        warnings.append(
            "Unsafe combination: legacy_event authorization is active while shell or "
            "connector effects are enabled; exact arguments and targets are not bound once."
        )
    if settings.agent_autonomy_enabled:
        if settings.agent_autonomy_mode is AgentAutonomyMode.ACT:
            warnings.append(
                "Autonomy act mode is enabled; spontaneous events may produce external "
                "effects within the effective autonomy policy and finite budgets."
            )
        if settings.agent_autonomy_policy_mode is AgentAutonomyPolicyMode.LEGACY:
            warnings.append(
                "Autonomy uses the legacy BOT-principal authority profile instead of "
                "event-scoped strict authority."
            )
        if settings.agent_autonomy_max_runs == 0:
            warnings.append(
                "Autonomy has no run-count ceiling; configure a finite "
                "AGENT_AUTONOMY_MAX_RUNS before unattended operation."
            )
        if (
            settings.agent_autonomy_mode is AgentAutonomyMode.ACT
            and settings.agent_autonomy_policy_mode is AgentAutonomyPolicyMode.LEGACY
        ):
            warnings.append(
                "Unsafe combination: autonomy act and the legacy BOT-principal policy "
                "compose spontaneous effects with ambient service authority."
            )
    return tuple(warnings)


def effective_security_policy(settings: Settings) -> EffectiveSecurityPolicy:
    """Return the exact effective boundary without secrets or Discord identifiers."""

    return EffectiveSecurityPolicy(
        configured_preset=settings.agent_security_preset.value,
        effective_preset=settings.agent_effective_security_preset.value,
        preset_expires_at_epoch=(
            int(settings.agent_security_preset_expires_at.timestamp())
            if settings.agent_security_preset_expires_at is not None
            else None
        ),
        preset_expired=settings.agent_security_preset_expired,
        override_names=settings.agent_security_override_names,
        agent_enabled=settings.agent_enabled,
        file_sandbox_enabled=settings.agent_file_sandbox_enabled,
        file_workspace_mode=settings.agent_file_workspace_mode.value,
        information_flow_mode=settings.agent_information_flow_mode.value,
        read_aloud_audience_mode=settings.read_aloud_audience_mode.value,
        high_risk_authorization_mode=(settings.agent_high_risk_authorization_mode.value),
        autonomy_enabled=settings.agent_autonomy_enabled,
        autonomy_mode=settings.agent_autonomy_mode.value,
        autonomy_policy_mode=settings.agent_autonomy_policy_mode.value,
        autonomy_max_runs=settings.agent_autonomy_max_runs,
        web_access=settings.agent_web_search_access.value,
        compute_access=settings.agent_safe_compute_access.value,
        image_access=settings.image_generation_access.value,
        shell_access=settings.agent_isolated_shell_access.value,
        connector_access=settings.agent_connector_access.value,
        warnings=security_policy_warnings(settings),
    )


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"{name} is required.")
    return value


def _positive_int(name: str, default: int, *, maximum: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc
    if not 1 <= value <= maximum:
        raise ConfigurationError(f"{name} must be between 1 and {maximum}.")
    return value


def _positive_int_with_legacy_name(
    name: str,
    legacy_name: str,
    default: int,
    *,
    maximum: int,
) -> int:
    """Prefer the explicit queue setting while preserving existing deployments."""

    if name in os.environ:
        return _positive_int(name, default, maximum=maximum)
    return _positive_int(legacy_name, default, maximum=maximum)


def _positive_float(name: str, default: float, *, maximum: float) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number.") from exc
    if not 0 < value <= maximum:
        raise ConfigurationError(f"{name} must be greater than 0 and at most {maximum}.")
    return value


def _positive_float_with_legacy_name(
    name: str,
    legacy_name: str,
    default: float,
    *,
    maximum: float,
) -> float:
    """Prefer an explicit setting while preserving existing deployments."""

    if name in os.environ:
        return _positive_float(name, default, maximum=maximum)
    return _positive_float(legacy_name, default, maximum=maximum)


def _bounded_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number.") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _bounded_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw_value = os.getenv(name, "true" if default else "false").strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false.")


def _text(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ConfigurationError(f"{name} cannot be empty.")
    if len(value) > 200:
        raise ConfigurationError(f"{name} is too long.")
    return value


def _optional_secret(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    if len(value) > 2_000:
        raise ConfigurationError(f"{name} is too long.")
    return value


def _web_search_base_url() -> str:
    value = _text("WEB_SEARCH_BASE_URL", "http://127.0.0.1:8888").rstrip("/")
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ConfigurationError("WEB_SEARCH_BASE_URL is invalid.") from exc
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not host:
        raise ConfigurationError("WEB_SEARCH_BASE_URL must be an HTTP(S) URL.")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigurationError("WEB_SEARCH_BASE_URL must not contain credentials.")
    if parsed.scheme == "http" and host not in {"127.0.0.1", "::1", "localhost"}:
        raise ConfigurationError("Remote WEB_SEARCH_BASE_URL values must use HTTPS.")
    return value


def _voicevox_base_url() -> str:
    value = _text("VOICEVOX_BASE_URL", "http://127.0.0.1:50021").rstrip("/")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ConfigurationError("VOICEVOX_BASE_URL is invalid.") from exc
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "http"
        or host not in {"127.0.0.1", "::1", "localhost"}
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigurationError(
            "VOICEVOX_BASE_URL must be a credential-free loopback HTTP URL with a port."
        )
    return value


def _optional_voicevox_engine_path() -> Path | None:
    raw_path = os.getenv("VOICEVOX_ENGINE_PATH", "").strip()
    if not raw_path:
        default_path = Path.home() / "Applications" / "voicevox-engine" / "macos-arm64" / "run"
        return default_path.resolve() if default_path.is_file() else None
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError("VOICEVOX_ENGINE_PATH does not point to an existing executable.")
    if not os.access(path, os.X_OK):
        raise ConfigurationError("VOICEVOX_ENGINE_PATH is not executable.")
    return path


def _feature_access(name: str, default: AgentFeatureAccess) -> AgentFeatureAccess:
    raw_value = os.getenv(name, default.value).strip().lower()
    try:
        return AgentFeatureAccess(raw_value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in AgentFeatureAccess)
        raise ConfigurationError(f"{name} must be one of: {choices}.") from exc


def _autonomy_mode(name: str, default: AgentAutonomyMode) -> AgentAutonomyMode:
    raw_value = os.getenv(name, default.value).strip().lower()
    try:
        return AgentAutonomyMode(raw_value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in AgentAutonomyMode)
        raise ConfigurationError(f"{name} must be one of: {choices}.") from exc


def _policy_mode(
    name: str,
    default: PolicyModeT,
    mode_type: type[PolicyModeT],
) -> PolicyModeT:
    raw_value = os.getenv(name, default.value).strip().lower()
    try:
        return mode_type(raw_value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in mode_type)
        raise ConfigurationError(f"{name} must be one of: {choices}.") from exc


def _security_preset(
    *,
    now: datetime,
) -> tuple[AgentSecurityPreset, AgentSecurityPreset, datetime | None, bool]:
    raw_value = (
        os.getenv(
            "AGENT_SECURITY_PRESET",
            AgentSecurityPreset.GUILD_ASSISTANT.value,
        )
        .strip()
        .lower()
    )
    try:
        configured = AgentSecurityPreset(raw_value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in AgentSecurityPreset)
        raise ConfigurationError(f"AGENT_SECURITY_PRESET must be one of: {choices}.") from exc

    raw_expiry = os.getenv("AGENT_SECURITY_PRESET_EXPIRES_AT", "").strip()
    expires_at: datetime | None = None
    if raw_expiry:
        try:
            expires_at = datetime.fromisoformat(raw_expiry.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ConfigurationError(
                "AGENT_SECURITY_PRESET_EXPIRES_AT must be an RFC 3339 timestamp."
            ) from exc
        if expires_at.tzinfo is None or expires_at.utcoffset() is None:
            raise ConfigurationError("AGENT_SECURITY_PRESET_EXPIRES_AT must include a UTC offset.")
        expires_at = expires_at.astimezone(UTC)
    if configured is AgentSecurityPreset.LEGACY_COMPATIBILITY and expires_at is None:
        raise ConfigurationError(
            "AGENT_SECURITY_PRESET_EXPIRES_AT is required when "
            "AGENT_SECURITY_PRESET=legacy_compatibility."
        )
    if configured is not AgentSecurityPreset.LEGACY_COMPATIBILITY and expires_at is not None:
        raise ConfigurationError(
            "AGENT_SECURITY_PRESET_EXPIRES_AT is valid only when "
            "AGENT_SECURITY_PRESET=legacy_compatibility. Remove the stale expiry "
            "before selecting another preset."
        )

    normalized_now = now.astimezone(UTC)
    expired = (
        configured is AgentSecurityPreset.LEGACY_COMPATIBILITY
        and expires_at is not None
        and expires_at <= normalized_now
    )
    effective = AgentSecurityPreset.GUILD_ASSISTANT if expired else configured
    return configured, effective, expires_at, expired


def _snowflake_set(name: str) -> frozenset[str]:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return frozenset()
    values = frozenset(item.strip() for item in raw_value.split(",") if item.strip())
    if any(not value.isdigit() or int(value) <= 0 for value in values):
        raise ConfigurationError(f"{name} must be a comma-separated list of Discord IDs.")
    return values


def _optional_snowflake(name: str) -> int | None:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return None
    if not raw_value.isdigit() or int(raw_value) <= 0:
        raise ConfigurationError(f"{name} must be a positive Discord ID.")
    value = int(raw_value)
    if value >= 1 << 64:
        raise ConfigurationError(f"{name} exceeds Discord's snowflake range.")
    return value


def _optional_private_file(name: str) -> Path | None:
    raw_path = os.getenv(name, "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"{name} does not point to an existing file.")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ConfigurationError(
            f"{name} must not be readable or writable by group or other users. "
            f"Run: chmod 600 {path}"
        )
    return path


def _optional_private_executable(name: str) -> Path | None:
    raw_path = os.getenv(name, "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"{name} does not point to an existing executable.")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077 or not os.access(path, os.X_OK):
        raise ConfigurationError(
            f"{name} must be an owner-executable private file. Run: chmod 700 {path}"
        )
    return path


def load_settings(
    *,
    dotenv_path: str | Path = ".env",
    now: datetime | None = None,
) -> Settings:
    """Load settings without overriding variables already set by the process."""

    load_dotenv(dotenv_path=dotenv_path, override=False)

    security_now = now or datetime.now(UTC)
    if security_now.tzinfo is None or security_now.utcoffset() is None:
        raise ConfigurationError("load_settings now must include a UTC offset.")
    (
        agent_security_preset,
        agent_effective_security_preset,
        agent_security_preset_expires_at,
        agent_security_preset_expired,
    ) = _security_preset(now=security_now)
    security_defaults = _SECURITY_PRESETS[agent_effective_security_preset]
    agent_security_override_names = tuple(
        name for name in _SECURITY_OVERRIDE_NAMES if name in os.environ
    )

    raw_application_id = _required("DISCORD_APPLICATION_ID")
    try:
        application_id = int(raw_application_id)
    except ValueError as exc:
        raise ConfigurationError("DISCORD_APPLICATION_ID must be an integer.") from exc
    if application_id <= 0:
        raise ConfigurationError("DISCORD_APPLICATION_ID must be positive.")

    activity_enabled = _boolean("DISCORD_ACTIVITY_ENABLED", False)
    activity_client_secret = _optional_secret("DISCORD_CLIENT_SECRET")
    if activity_enabled and activity_client_secret is None:
        raise ConfigurationError(
            "DISCORD_CLIENT_SECRET is required when DISCORD_ACTIVITY_ENABLED is true."
        )

    raw_scope = os.getenv("COMMAND_SCOPE", CommandScope.GUILD.value).strip().lower()
    try:
        command_scope = CommandScope(raw_scope)
    except ValueError as exc:
        raise ConfigurationError("COMMAND_SCOPE must be guild or global.") from exc

    command_prefix = os.getenv("COMMAND_PREFIX", "!").strip()
    if not 1 <= len(command_prefix) <= 5 or any(
        character.isspace() for character in command_prefix
    ):
        raise ConfigurationError("COMMAND_PREFIX must contain 1 to 5 non-space characters.")

    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if log_level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        raise ConfigurationError("LOG_LEVEL is invalid.")

    data_dir = Path(os.getenv("DATA_DIR", ".data")).expanduser().resolve()
    data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with suppress(OSError):
        data_dir.chmod(0o700)
    raw_discord_agent_workspace_dir = os.getenv(
        "DISCORD_AGENT_WORKSPACE_DIR",
        "",
    ).strip()
    discord_agent_workspace_dir = (
        Path(raw_discord_agent_workspace_dir).expanduser().resolve()
        if raw_discord_agent_workspace_dir
        else data_dir / "discord_agent_workspaces"
    )

    tts_provider = os.getenv("TTS_PROVIDER", "macos").strip().lower()
    if tts_provider not in {"macos", "voicevox"}:
        raise ConfigurationError("TTS_PROVIDER must be macos or voicevox.")

    tts_voice = os.getenv("TTS_VOICE", "Samantha").strip()
    if not tts_voice:
        raise ConfigurationError("TTS_VOICE cannot be empty.")
    voicevox_base_url = _voicevox_base_url()
    voicevox_engine_path = _optional_voicevox_engine_path()
    voicevox_auto_start = _boolean("VOICEVOX_AUTO_START", True)
    if tts_provider == "voicevox" and voicevox_auto_start and voicevox_engine_path is None:
        raise ConfigurationError(
            "VOICEVOX_ENGINE_PATH is required when VOICEVOX_AUTO_START is enabled."
        )

    agent_provider = _text("AGENT_PROVIDER", "codex").lower()
    if agent_provider != "codex":
        raise ConfigurationError("AGENT_PROVIDER must be codex in this milestone.")
    agent_allowed_guild_ids = _snowflake_set("AGENT_ALLOWED_GUILD_IDS")
    agent_trusted_guild_ids = _snowflake_set("AGENT_TRUSTED_GUILD_IDS")
    agent_admin_user_ids = _snowflake_set("AGENT_ADMIN_USER_IDS")
    agent_rate_limit_exempt_user_ids = _snowflake_set("AGENT_RATE_LIMIT_EXEMPT_USER_IDS")
    agent_autonomy_guild_ids = _snowflake_set("AGENT_AUTONOMY_GUILD_IDS")
    if not agent_trusted_guild_ids <= agent_allowed_guild_ids:
        raise ConfigurationError(
            "AGENT_TRUSTED_GUILD_IDS must be a subset of AGENT_ALLOWED_GUILD_IDS."
        )
    if not agent_autonomy_guild_ids <= agent_allowed_guild_ids:
        raise ConfigurationError(
            "AGENT_AUTONOMY_GUILD_IDS must be a subset of AGENT_ALLOWED_GUILD_IDS."
        )
    agent_autonomy_max_pending_events = _positive_int(
        "AGENT_AUTONOMY_MAX_PENDING_EVENTS",
        1_000,
        maximum=100_000,
    )
    agent_autonomy_max_pending_events_per_channel = _positive_int(
        "AGENT_AUTONOMY_MAX_PENDING_EVENTS_PER_CHANNEL",
        100,
        maximum=10_000,
    )
    agent_autonomy_max_pending_events_per_actor = _positive_int(
        "AGENT_AUTONOMY_MAX_PENDING_EVENTS_PER_ACTOR",
        50,
        maximum=10_000,
    )
    if agent_autonomy_max_pending_events_per_channel > agent_autonomy_max_pending_events:
        raise ConfigurationError(
            "AGENT_AUTONOMY_MAX_PENDING_EVENTS_PER_CHANNEL must not exceed "
            "AGENT_AUTONOMY_MAX_PENDING_EVENTS."
        )
    if agent_autonomy_max_pending_events_per_actor > agent_autonomy_max_pending_events:
        raise ConfigurationError(
            "AGENT_AUTONOMY_MAX_PENDING_EVENTS_PER_ACTOR must not exceed "
            "AGENT_AUTONOMY_MAX_PENDING_EVENTS."
        )
    agent_web_search_access = _feature_access(
        "AGENT_WEB_SEARCH_ACCESS",
        security_defaults.web_access,
    )
    agent_safe_compute_access = _feature_access(
        "AGENT_SAFE_COMPUTE_ACCESS",
        security_defaults.compute_access,
    )
    agent_isolated_shell_access = _feature_access(
        "AGENT_ISOLATED_SHELL_ACCESS",
        security_defaults.shell_access,
    )
    agent_connector_access = _feature_access(
        "AGENT_CONNECTOR_ACCESS",
        security_defaults.connector_access,
    )
    agent_file_sandbox_enabled = _boolean(
        "AGENT_FILE_SANDBOX_ENABLED",
        security_defaults.file_sandbox_enabled,
    )
    if (
        agent_safe_compute_access is not AgentFeatureAccess.DISABLED
        and not agent_file_sandbox_enabled
    ):
        raise ConfigurationError(
            "AGENT_FILE_SANDBOX_ENABLED must be true when AGENT_SAFE_COMPUTE_ACCESS is enabled."
        )
    image_generation_access = _feature_access(
        "IMAGE_GENERATION_ACCESS",
        security_defaults.image_access,
    )
    if (
        AgentFeatureAccess.ADMINS
        in {
            agent_web_search_access,
            agent_safe_compute_access,
            agent_isolated_shell_access,
            agent_connector_access,
        }
        and not agent_admin_user_ids
    ):
        raise ConfigurationError(
            "AGENT_ADMIN_USER_IDS is required when an agent feature is admin-only."
        )
    if image_generation_access is AgentFeatureAccess.ADMINS and not agent_admin_user_ids:
        raise ConfigurationError(
            "AGENT_ADMIN_USER_IDS is required when image generation is admin-only."
        )
    max_concurrent_media = _positive_int(
        "MAX_CONCURRENT_MEDIA",
        4,
        maximum=32,
    )
    max_concurrent_media_per_guild = _positive_int(
        "MAX_CONCURRENT_MEDIA_PER_GUILD",
        2,
        maximum=16,
    )
    if max_concurrent_media_per_guild > max_concurrent_media:
        raise ConfigurationError(
            "MAX_CONCURRENT_MEDIA_PER_GUILD cannot exceed MAX_CONCURRENT_MEDIA."
        )
    local_media_max_file_bytes = _bounded_int(
        "LOCAL_MEDIA_MAX_FILE_BYTES",
        100_000_000,
        minimum=1_000_000,
        maximum=500_000_000,
    )
    local_media_size_gb = os.getenv("LOCAL_MEDIA_SIZE_GB", "").strip()
    local_media_cache_bytes = (
        _positive_int("LOCAL_MEDIA_SIZE_GB", 5, maximum=20) * 1_000_000_000
        if local_media_size_gb
        else _bounded_int(
            "LOCAL_MEDIA_CACHE_BYTES",
            5_000_000_000,
            minimum=100_000_000,
            maximum=20_000_000_000,
        )
    )
    if local_media_cache_bytes < local_media_max_file_bytes:
        raise ConfigurationError(
            "LOCAL_MEDIA_CACHE_BYTES cannot be smaller than LOCAL_MEDIA_MAX_FILE_BYTES."
        )

    data_retention_days = _positive_int(
        "DATA_RETENTION_DAYS",
        30,
        maximum=3_650,
    )
    max_data_size_bytes = _positive_int("MAX_DATA_SIZE_GB", 10, maximum=100) * 1_000_000_000
    if local_media_cache_bytes > max_data_size_bytes:
        raise ConfigurationError(
            "LOCAL_MEDIA_SIZE_GB/LOCAL_MEDIA_CACHE_BYTES cannot exceed MAX_DATA_SIZE_GB."
        )

    return Settings(
        token=_required("DISCORD_TOKEN"),
        application_id=application_id,
        discord_members_intent_enabled=_boolean(
            "DISCORD_MEMBERS_INTENT_ENABLED",
            False,
        ),
        discord_presence_intent_enabled=_boolean(
            "DISCORD_PRESENCE_INTENT_ENABLED",
            False,
        ),
        discord_emoji_loading_id=_optional_snowflake("DISCORD_EMOJI_LOADING_ID"),
        discord_emoji_success_id=_optional_snowflake("DISCORD_EMOJI_SUCCESS_ID"),
        discord_emoji_warning_id=_optional_snowflake("DISCORD_EMOJI_WARNING_ID"),
        discord_emoji_audio_wave_id=_optional_snowflake("DISCORD_EMOJI_AUDIO_WAVE_ID"),
        discord_emoji_radio_id=_optional_snowflake("DISCORD_EMOJI_RADIO_ID"),
        activity_enabled=activity_enabled,
        activity_client_secret=activity_client_secret,
        activity_host=_text("DISCORD_ACTIVITY_HOST", "127.0.0.1"),
        activity_port=_bounded_int(
            "DISCORD_ACTIVITY_PORT",
            8_787,
            minimum=1_024,
            maximum=65_535,
        ),
        command_scope=command_scope,
        command_prefix=command_prefix,
        log_level=log_level,
        data_dir=data_dir,
        discord_agent_workspace_dir=discord_agent_workspace_dir,
        data_retention_days=data_retention_days,
        max_data_size_bytes=max_data_size_bytes,
        media_cookie_file=_optional_private_file("MEDIA_COOKIE_FILE"),
        tts_provider=tts_provider,
        tts_voice=tts_voice,
        voicevox_base_url=voicevox_base_url,
        voicevox_speaker_id=_bounded_int("VOICEVOX_SPEAKER_ID", 3, minimum=0, maximum=65_535),
        voicevox_preset_clear_id=_bounded_int(
            "VOICEVOX_PRESET_CLEAR_ID", 2, minimum=0, maximum=65_535
        ),
        voicevox_preset_calm_id=_bounded_int(
            "VOICEVOX_PRESET_CALM_ID", 14, minimum=0, maximum=65_535
        ),
        voicevox_preset_energetic_id=_bounded_int(
            "VOICEVOX_PRESET_ENERGETIC_ID", 8, minimum=0, maximum=65_535
        ),
        voicevox_preset_cute_id=_bounded_int(
            "VOICEVOX_PRESET_CUTE_ID", 3, minimum=0, maximum=65_535
        ),
        voicevox_preset_narrator_id=_bounded_int(
            "VOICEVOX_PRESET_NARRATOR_ID", 13, minimum=0, maximum=65_535
        ),
        voicevox_engine_path=voicevox_engine_path,
        voicevox_auto_start=voicevox_auto_start,
        voicevox_timeout_seconds=_positive_float("VOICEVOX_TIMEOUT_SECONDS", 30.0, maximum=120.0),
        voicevox_readiness_ttl_seconds=_positive_float(
            "VOICEVOX_READINESS_TTL_SECONDS",
            5.0,
            maximum=300.0,
        ),
        read_aloud_chunk_characters=_positive_int(
            "READ_ALOUD_CHUNK_CHARACTERS", 400, maximum=2_000
        ),
        read_aloud_audience_mode=_policy_mode(
            "READ_ALOUD_AUDIENCE_MODE",
            security_defaults.read_aloud_audience_mode,
            ReadAloudAudienceMode,
        ),
        max_pending_speech=_positive_int("MAX_PENDING_SPEECH", 20, maximum=100),
        max_pending_music=_positive_int("MAX_PENDING_MUSIC", 100, maximum=500),
        max_pending_music_per_user=_positive_int(
            "MAX_PENDING_MUSIC_PER_USER",
            20,
            maximum=100,
        ),
        max_concurrent_tts=_positive_int("MAX_CONCURRENT_TTS", 2, maximum=10),
        max_concurrent_tts_provider_calls=_positive_int(
            "MAX_CONCURRENT_TTS_PROVIDER_CALLS",
            2,
            maximum=10,
        ),
        max_concurrent_media=max_concurrent_media,
        max_concurrent_media_per_guild=max_concurrent_media_per_guild,
        max_active_voice_guilds=_positive_int("MAX_ACTIVE_VOICE_GUILDS", 8, maximum=100),
        download_timeout_seconds=_positive_float("DOWNLOAD_TIMEOUT_SECONDS", 180.0, maximum=900.0),
        local_media_max_file_bytes=local_media_max_file_bytes,
        local_media_cache_bytes=local_media_cache_bytes,
        local_media_max_duration_seconds=_positive_float(
            "LOCAL_MEDIA_MAX_DURATION_SECONDS",
            21_600.0,
            maximum=86_400.0,
        ),
        translation_enabled=_boolean("TRANSLATION_ENABLED", True),
        translation_helper_path=_optional_private_executable("TRANSLATION_HELPER_PATH"),
        translation_timeout_seconds=_positive_float(
            "TRANSLATION_TIMEOUT_SECONDS",
            30.0,
            maximum=120.0,
        ),
        translation_max_characters=_positive_int(
            "TRANSLATION_MAX_CHARACTERS",
            8_000,
            maximum=20_000,
        ),
        web_search_base_url=_web_search_base_url(),
        web_search_shared_secret=_optional_secret("WEB_SEARCH_SHARED_SECRET"),
        web_request_timeout_seconds=_positive_float(
            "WEB_REQUEST_TIMEOUT_SECONDS",
            12.0,
            maximum=60.0,
        ),
        web_fetch_max_bytes=_bounded_int(
            "WEB_FETCH_MAX_BYTES",
            2_000_000,
            minimum=64_000,
            maximum=10_000_000,
        ),
        hive_api_key=_optional_secret("HIVE_API_KEY"),
        hive_daily_limit=_bounded_int(
            "HIVE_DAILY_LIMIT",
            100,
            minimum=1,
            maximum=100,
        ),
        hive_timeout_seconds=_positive_float(
            "HIVE_TIMEOUT_SECONDS",
            90.0,
            maximum=180.0,
        ),
        hive_max_media_bytes=_bounded_int(
            "HIVE_MAX_MEDIA_BYTES",
            25_000_000,
            minimum=64_000,
            maximum=100_000_000,
        ),
        hive_threshold=_bounded_float(
            "HIVE_THRESHOLD",
            0.9,
            minimum=0.01,
            maximum=1.0,
        ),
        image_timeout_seconds=_positive_float(
            "IMAGE_TIMEOUT_SECONDS",
            600.0,
            maximum=1_800.0,
        ),
        image_generation_access=image_generation_access,
        image_per_user_requests=_positive_int(
            "IMAGE_PER_USER_REQUESTS",
            5,
            maximum=100,
        ),
        image_per_user_window_seconds=_bounded_int(
            "IMAGE_PER_USER_WINDOW_SECONDS",
            3_600,
            minimum=60,
            maximum=86_400,
        ),
        image_per_workspace_requests=_positive_int(
            "IMAGE_PER_WORKSPACE_REQUESTS",
            30,
            maximum=1_000,
        ),
        image_per_workspace_window_seconds=_bounded_int(
            "IMAGE_PER_WORKSPACE_WINDOW_SECONDS",
            86_400,
            minimum=60,
            maximum=604_800,
        ),
        image_max_pending_jobs=_positive_int(
            "IMAGE_MAX_PENDING_JOBS",
            10,
            maximum=100,
        ),
        agent_security_preset=agent_security_preset,
        agent_effective_security_preset=agent_effective_security_preset,
        agent_security_preset_expires_at=agent_security_preset_expires_at,
        agent_security_preset_expired=agent_security_preset_expired,
        agent_security_override_names=agent_security_override_names,
        agent_enabled=_boolean("AGENT_ENABLED", False),
        agent_allowed_guild_ids=agent_allowed_guild_ids,
        agent_trusted_guild_ids=agent_trusted_guild_ids,
        agent_admin_user_ids=agent_admin_user_ids,
        agent_rate_limit_exempt_user_ids=agent_rate_limit_exempt_user_ids,
        agent_web_search_access=agent_web_search_access,
        agent_safe_compute_access=agent_safe_compute_access,
        agent_isolated_shell_access=agent_isolated_shell_access,
        agent_connector_access=agent_connector_access,
        agent_file_sandbox_enabled=agent_file_sandbox_enabled,
        agent_file_workspace_mode=_policy_mode(
            "AGENT_FILE_WORKSPACE_MODE",
            security_defaults.file_workspace_mode,
            AgentFileWorkspaceMode,
        ),
        agent_information_flow_mode=_policy_mode(
            "AGENT_INFORMATION_FLOW_MODE",
            security_defaults.information_flow_mode,
            AgentInformationFlowMode,
        ),
        agent_high_risk_authorization_mode=_policy_mode(
            "AGENT_HIGH_RISK_AUTHORIZATION_MODE",
            security_defaults.high_risk_authorization_mode,
            AgentHighRiskAuthorizationMode,
        ),
        agent_high_risk_confirmation_timeout_seconds=_bounded_int(
            "AGENT_HIGH_RISK_CONFIRMATION_TIMEOUT_SECONDS",
            120,
            minimum=30,
            maximum=600,
        ),
        agent_curated_skills_enabled=_boolean("AGENT_CURATED_SKILLS_ENABLED", False),
        agent_provider=agent_provider,
        agent_model=_text("AGENT_MODEL", "gpt-5.6-luna"),
        agent_escalation_model=_text(
            "AGENT_ESCALATION_MODEL",
            "gpt-5.6-terra",
        ),
        codex_executable=_text("CODEX_EXECUTABLE", "codex"),
        codex_expected_version_prefix=_text(
            "CODEX_EXPECTED_VERSION_PREFIX",
            "0.146.",
        ),
        agent_idle_timeout_seconds=_positive_float_with_legacy_name(
            "AGENT_IDLE_TIMEOUT_SECONDS",
            "AGENT_TIMEOUT_SECONDS",
            600.0,
            maximum=1_800.0,
        ),
        agent_reasoning_effort=_text("AGENT_REASONING_EFFORT", "high"),
        agent_per_user_requests=_positive_int(
            "AGENT_PER_USER_REQUESTS",
            3,
            maximum=100,
        ),
        agent_per_user_window_seconds=_bounded_int(
            "AGENT_PER_USER_WINDOW_SECONDS",
            600,
            minimum=10,
            maximum=86_400,
        ),
        agent_per_workspace_requests=_positive_int(
            "AGENT_PER_WORKSPACE_REQUESTS",
            10,
            maximum=1_000,
        ),
        agent_per_workspace_window_seconds=_bounded_int(
            "AGENT_PER_WORKSPACE_WINDOW_SECONDS",
            3_600,
            minimum=10,
            maximum=86_400,
        ),
        agent_max_tokens_per_24_hours=_bounded_int(
            "AGENT_MAX_TOKENS_PER_24_HOURS",
            150_000,
            minimum=1_000,
            maximum=10_000_000,
        ),
        agent_max_response_characters=_bounded_int(
            "AGENT_MAX_RESPONSE_CHARACTERS",
            7_600,
            minimum=200,
            maximum=8_000,
        ),
        agent_max_active_turns=_positive_int(
            "MAX_ACTIVE_AGENT_TURNS",
            4,
            maximum=20,
        ),
        agent_max_pending_turns=_positive_int_with_legacy_name(
            "MAX_PENDING_AGENT_TURNS",
            "AGENT_MAX_PENDING_TURNS",
            20,
            maximum=100,
        ),
        agent_max_pending_turns_per_user=_positive_int(
            "MAX_PENDING_AGENT_TURNS_PER_USER",
            2,
            maximum=20,
        ),
        agent_interactive_reserve_percent=_bounded_int(
            "AGENT_INTERACTIVE_RESERVE_PERCENT",
            25,
            minimum=0,
            maximum=90,
        ),
        agent_conversation_compatibility_epoch=_positive_int(
            "AGENT_CONVERSATION_COMPATIBILITY_EPOCH",
            6,
            maximum=10_000,
        ),
        agent_autonomy_enabled=_boolean(
            "AGENT_AUTONOMY_ENABLED",
            security_defaults.autonomy_enabled,
        ),
        agent_autonomy_guild_ids=agent_autonomy_guild_ids,
        agent_autonomy_mode=_autonomy_mode(
            "AGENT_AUTONOMY_MODE",
            security_defaults.autonomy_mode,
        ),
        agent_autonomy_policy_mode=_policy_mode(
            "AGENT_AUTONOMY_POLICY_MODE",
            security_defaults.autonomy_policy_mode,
            AgentAutonomyPolicyMode,
        ),
        agent_autonomy_batch_seconds=_bounded_int(
            "AGENT_AUTONOMY_BATCH_SECONDS",
            10,
            minimum=5,
            maximum=15,
        ),
        agent_autonomy_max_runs=_bounded_int(
            "AGENT_AUTONOMY_MAX_RUNS",
            security_defaults.autonomy_max_runs,
            minimum=0,
            maximum=1_000,
        ),
        agent_autonomy_candidate_limit=_bounded_int(
            "AGENT_AUTONOMY_CANDIDATE_LIMIT",
            5,
            minimum=1,
            maximum=20,
        ),
        agent_autonomy_max_pending_events=agent_autonomy_max_pending_events,
        agent_autonomy_max_pending_events_per_channel=(
            agent_autonomy_max_pending_events_per_channel
        ),
        agent_autonomy_max_pending_events_per_actor=(agent_autonomy_max_pending_events_per_actor),
    )


def _optional_directory(name: str) -> Path | None:
    raw_path = os.getenv(name, "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise ConfigurationError(f"{name} does not point to an existing directory.")
    return path
