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
) -> tuple[str, ...]:
    """Disable unrelated hosted features and opt into images explicitly."""

    disabled = tuple(
        feature
        for feature in _DISABLED_CODEX_FEATURES
        if not (allow_image_generation and feature == "image_generation")
    )
    arguments = tuple(
        argument
        for feature in disabled
        for argument in ("--disable", feature)
    )
    if allow_image_generation:
        return ("--enable", "image_generation", *arguments)
    return arguments
