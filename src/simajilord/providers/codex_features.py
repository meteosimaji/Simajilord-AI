"""Shared Codex feature isolation for embedded app-server processes."""

from __future__ import annotations

# Legacy remains durable, resumable, and compactable. Experimental paginated
# projection in bundled Codex runtimes can reject valid fractional rate-limit JSON.
CODEX_THREAD_HISTORY_MODE = "legacy"

_DISABLED_CODEX_FEATURES = (
    "apps",
    "auth_elicitation",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "code_mode_host",
    "computer_use",
    "fast_mode",
    "goals",
    "guardian_approval",
    "hooks",
    "image_generation",
    "in_app_browser",
    "memories",
    "multi_agent",
    "multi_agent_v2",
    "personality",
    "plugin_sharing",
    "plugins",
    "remote_plugin",
    "shell_snapshot",
    "shell_tool",
    "skill_mcp_dependency_install",
    "skill_search",
    "tool_call_mcp_elicitation",
    "tool_suggest",
    "unified_exec",
    "workspace_dependencies",
)


def codex_feature_arguments(
    *,
    allow_image_generation: bool = False,
    allow_discord_extensions: bool = False,
) -> tuple[str, ...]:
    """Disable unrelated hosted features and opt into images explicitly."""

    # Codex 0.147 presents namespace dynamic tools through the local code-mode
    # host.  Discord still owns every actual capability invocation and keeps
    # shell_tool/unified_exec disabled; enabling this host only gives the model
    # the isolated JavaScript dispatcher required to reach those typed tools.
    enabled_features = {
        "apps",
        "code_mode_host",
        "plugins",
        "skill_search",
    } if allow_discord_extensions else set()
    if allow_image_generation:
        enabled_features.add("image_generation")
    disabled = tuple(
        feature
        for feature in _DISABLED_CODEX_FEATURES
        if feature not in enabled_features
    )
    enabled_arguments = tuple(
        argument
        for feature in sorted(enabled_features)
        for argument in ("--enable", feature)
    )
    disabled_arguments = tuple(
        argument
        for feature in disabled
        for argument in ("--disable", feature)
    )
    return (*enabled_arguments, *disabled_arguments)
