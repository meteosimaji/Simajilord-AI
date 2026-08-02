"""Discord-only Codex app-server policy overrides.

These settings are passed on the embedded app-server command line. They never
write the user's global ``~/.codex/config.toml``. Native model-facing Apps and
plugin MCP servers stay disabled; reviewed design tools are reached only through
Simajilord's capability and effect broker.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DiscordCodexAppPolicy:
    """One connector and its Discord-only model-visible tool policy."""

    app_id: str
    name: str
    enabled_tools: tuple[str, ...] | None = None
    prompt_tools: tuple[str, ...] = ()


DISCORD_CODEX_PERMISSION_PROFILE = "simajilord_discord"


class DiscordCodexAppActionClass(StrEnum):
    """Conservative native-app classification retained for breach telemetry."""

    READ = "read"
    MUTATING = "mutating"
    UNKNOWN = "unknown"


_GITHUB_READ_TOOLS = (
    "compare_commits",
    "download_user_content",
    "download_workflow_artifact",
    "fetch",
    "fetch_blob",
    "fetch_commit",
    "fetch_commit_workflow_runs",
    "fetch_file",
    "fetch_issue",
    "fetch_issue_comments",
    "fetch_pr",
    "fetch_pr_comments",
    "fetch_pr_file_patch",
    "fetch_pr_patch",
    "fetch_workflow_job_logs",
    "fetch_workflow_job_steps",
    "fetch_workflow_run_artifacts",
    "fetch_workflow_run_jobs",
    "get_commit_combined_status",
    "get_issue_comment_reactions",
    "get_pr_diff",
    "get_pr_info",
    "get_pr_reactions",
    "get_pr_review_comment_reactions",
    "get_profile",
    "get_repo",
    "get_repo_collaborator_permission",
    "get_user_login",
    "get_users_recent_prs_in_repo",
    "list_installations",
    "list_installed_accounts",
    "list_pr_changed_filenames",
    "list_pull_request_review_threads",
    "list_pull_request_reviews",
    "list_recent_issues",
    "list_repositories",
    "list_repositories_by_affiliation",
    "list_repositories_by_installation",
    "list_user_org_memberships",
    "list_user_orgs",
    "search",
    "search_branches",
    "search_commits",
    "search_installed_repositories_streaming",
    "search_installed_repositories_v2",
    "search_issues",
    "search_prs",
    "search_repositories",
    "oai_user_search",
    "oai_user_fetch",
)

_HUGGING_FACE_READ_TOOLS = (
    "hf_whoami",
    "space_search",
    "model_search",
    "paper_search",
    "dataset_search",
    "hub_repo_details",
    "hf_doc_search",
    "hf_doc_fetch",
)

_PLUGIN_MANAGEMENT_READ_TOOLS = (
    "get_app_permissions",
    "get_plugin_dependencies",
)

# This list now describes reviewed connector identities for the host broker.
# Every native model-facing app is disabled below, including these entries.
DISCORD_CODEX_APP_POLICIES = (
    DiscordCodexAppPolicy(
        "asdk_app_6938a94a61d881918ef32cb999ff937c",
        "Apple Music",
    ),
    DiscordCodexAppPolicy(
        "asdk_app_6944570636288191b7944d8c4a3fb857",
        "Shazam",
    ),
    DiscordCodexAppPolicy(
        "asdk_app_69fe0bf66c8481919c513d799406436e",
        "Wolfram",
    ),
    DiscordCodexAppPolicy(
        "connector_691e3de0d2708191a6476a7b36e38779",
        "BioRender",
    ),
    DiscordCodexAppPolicy(
        "asdk_app_6939e86417648191b7bda087d872685b",
        "Hugging Face",
        _HUGGING_FACE_READ_TOOLS,
    ),
    DiscordCodexAppPolicy(
        "connector_76869538009648d5b282a4bb21c3d157",
        "GitHub",
        _GITHUB_READ_TOOLS,
    ),
    DiscordCodexAppPolicy(
        "connector_68df038e0ba48191908c8434991bbac2",
        "Figma",
    ),
    DiscordCodexAppPolicy(
        "asdk_app_699d522f170c81919c824678c7c03732",
        "Adobe Express",
    ),
    DiscordCodexAppPolicy(
        "connector_69312da8e4dc81919370cb86fd172b6c",
        "Adobe",
        prompt_tools=("asset_share_link", "asset_invite_collaborators"),
    ),
    DiscordCodexAppPolicy(
        "connector_68df33b1a2d081918778431a9cfca8ba",
        "Canva",
    ),
    DiscordCodexAppPolicy(
        "connector_openai_codex_document_control",
        "Codex Document Control",
    ),
    DiscordCodexAppPolicy(
        "connector_openai_plugin_management",
        "Plugin Management",
        _PLUGIN_MANAGEMENT_READ_TOOLS,
    ),
)

DISCORD_CODEX_BROKER_CONNECTORS = {
    policy.app_id: policy.name
    for policy in DISCORD_CODEX_APP_POLICIES
    if policy.name in {"Adobe", "Adobe Express", "BioRender", "Canva", "Figma"}
}
if len(DISCORD_CODEX_BROKER_CONNECTORS) != 5:
    raise RuntimeError("Discord connector broker identities collided")

DISCORD_CODEX_ENABLED_PLUGINS = (
    "build-ios-apps@openai-curated",
    "build-macos-apps@openai-curated",
    "build-web-apps@openai-curated",
    "canva@openai-curated",
    "codex-security@openai-curated",
    "documents@openai-primary-runtime",
    "game-studio@openai-curated",
    "github@openai-curated",
    "hugging-face@openai-curated",
    "latex@openai-bundled",
    "life-science-research@openai-curated",
    "nvidia@openai-curated",
    "openai-developers@openai-curated",
    "pdf@openai-primary-runtime",
    "presentations@openai-primary-runtime",
    "remotion@openai-curated",
    "spreadsheets@openai-primary-runtime",
    "template-creator@openai-primary-runtime",
    "test-android-apps@openai-curated",
    "visualize@openai-bundled",
)

DISCORD_CODEX_DISABLED_PLUGINS = (
    "browser@openai-bundled",
    "chrome@openai-bundled",
    "cloudflare@openai-curated",
    "computer-use@openai-bundled",
    "gmail@openai-curated",
    "google-calendar@openai-curated",
    "google-drive@openai-curated",
    "outlook-email@openai-curated",
    "record-and-replay@openai-bundled",
    "sites@openai-bundled",
)

# Plugin MCP processes execute outside the model shell sandbox. Shadow them
# with disabled inert definitions so they cannot inherit host environment or
# reach files outside the dedicated Discord workspace. Their plugin skills
# remain available and can use the sandboxed shell instead.
_DISCORD_CODEX_DISABLED_MCP_SERVERS = (
    "creative_production_mcp",
    "dataAnalyticsWidgets",
    "event-stream",
    "openai-api-key-local-confirmation",
)
_DISCORD_CODEX_ALWAYS_DISABLED_MCP_SERVERS = frozenset(
    {
        "computer-use",
        "node_repl",
        "openaiDeveloperDocs",
        "playwright",
    }
)

_APP_WRITE_TOOLS: dict[str, frozenset[str]] = {
    "connector_68df038e0ba48191908c8434991bbac2": frozenset(
        {
            "generate_figma_design",
            "generate_diagram",
            "generate_deck",
            "add_code_connect_map",
            "send_code_connect_mappings",
            "use_figma",
            "create_new_file",
            "upload_assets",
        }
    ),
    "asdk_app_699d522f170c81919c824678c7c03732": frozenset(
        {"fill_text", "replace_image", "change_background_color", "animate_design"}
    ),
    "connector_68df33b1a2d081918778431a9cfca8ba": frozenset(
        {
            "create-folder",
            "move-item-to-folder",
            "copy-design",
            "create-design-from-brand-template",
            "autofill-design",
            "import-design-from-url",
            "upload-asset-from-url",
            "resize-design",
            "start-editing-transaction",
            "perform-editing-operations",
            "commit-editing-transaction",
            "cancel-editing-transaction",
            "create-design-from-candidate",
            "image-to-design",
        }
    ),
    "connector_openai_codex_document_control": frozenset(
        {"execute_document_command"}
    ),
    "connector_69312da8e4dc81919370cb86fd172b6c": frozenset(
        {
            "asset_copy_assets",
            "asset_create_folders",
            "asset_share_link",
            "asset_invite_collaborators",
            "asset_openai_file_upload",
            "image_apply_lens_blur",
            "image_apply_preset",
            "image_auto_straighten",
            "image_apply_auto_tone",
            "image_invert_selection",
            "image_fill_area",
            "image_crop_and_resize",
            "image_crop_to_bounds",
            "image_remove_background",
            "image_apply_halftone",
            "image_apply_glitch_effect",
            "image_apply_gaussian_blur",
            "image_add_noise",
            "image_add_grain",
            "image_apply_color_overlay",
            "image_apply_monochromatic_tint",
            "image_apply_adjustments",
            "image_generative_expand",
            "image_generate",
            "image_instruct_edit",
            "image_remove_blemishes",
            "animate_design",
            "fill_text",
            "change_background_color",
            "replace_image",
            "image_vectorize",
            "document_merge_data_vector",
            "document_render_vector",
            "document_convert_pdf",
            "document_render_layout",
            "document_merge_data_layout",
            "convert_pdf_to_indd",
            "export_idml",
            "prepare_indd_merge_template",
            "pdf_create",
            "pdf_export",
            "pdf_to_image",
            "markdown_to_pdf",
            "pdf_ocr",
            "pdf_compress",
            "boards_create_new_board",
            "boards_add_items_to_board",
        }
    ),
}


def discord_codex_app_tool_action_class(
    app_id: str | None,
    tool: str | None,
) -> DiscordCodexAppActionClass:
    """Classify only statically reviewed native actions; unknown fails closed."""

    if app_id is None or tool is None:
        return DiscordCodexAppActionClass.UNKNOWN
    action = tool.rsplit(".", 1)[-1]
    if action in _APP_WRITE_TOOLS.get(app_id, frozenset()):
        return DiscordCodexAppActionClass.MUTATING
    policy = next(
        (candidate for candidate in DISCORD_CODEX_APP_POLICIES if candidate.app_id == app_id),
        None,
    )
    if policy is not None and policy.enabled_tools is not None and action in policy.enabled_tools:
        return DiscordCodexAppActionClass.READ
    return DiscordCodexAppActionClass.UNKNOWN


def discord_codex_app_tool_is_write(app_id: str | None, tool: str | None) -> bool:
    """Treat unknown native app actions as writes for retry and audit safety."""

    return (
        discord_codex_app_tool_action_class(app_id, tool)
        is not DiscordCodexAppActionClass.READ
    )

_DISABLED_SKILL_NAMES = frozenset(
    {
        "cloudflare",
        "gh-address-comments",
        "github-gh-address-comments",
        "github-yeet",
        "hugging-face-cli",
        "hugging-face-community-evals",
        "hugging-face-jobs",
        "hugging-face-llm-trainer",
        "hugging-face-paper-publisher",
        "hugging-face-trackio",
        "hugging-face-vision-trainer",
        "openai-docs",  # duplicate user copy; the system copy remains available
        "playwright",
        "playwright-interactive",
        "send-chatgpt-message",
        "yeet",
    }
)

_DISABLED_PLUGIN_SKILLS: dict[str, frozenset[str] | None] = {
    "browser": None,
    "chrome": None,
    "cloudflare": None,
    "computer-use": None,
    "gmail": None,
    "google-calendar": None,
    "google-drive": None,
    "outlook-email": None,
    "record-and-replay": None,
    "sites": None,
    "github": frozenset({"gh-address-comments", "yeet"}),
    "hugging-face": frozenset(
        {
            "cli",
            "community-evals",
            "jobs",
            "llm-trainer",
            "paper-publisher",
            "trackio",
            "vision-trainer",
        }
    ),
}

_DISABLED_SKILL_PREFIXES = (
    "browser-",
    "chrome-",
    "cloudflare-",
    "computer-use-",
    "gmail-",
    "google-calendar-",
    "google-drive-",
    "outlook-email-",
    "record-and-replay-",
    "sites-",
)


def discord_codex_policy_arguments(*, codex_home: Path) -> tuple[str, ...]:
    """Return process-local config overrides for the Discord app-server."""

    settings: list[tuple[str, str]] = [
        ("apps._default.enabled", "false"),
        ("apps._default.destructive_enabled", "false"),
        ("apps._default.open_world_enabled", "false"),
        ("apps._default.default_tools_approval_mode", _toml_string("approve")),
        ("shell_environment_policy.inherit", _toml_string("none")),
        ("shell_environment_policy.ignore_default_excludes", "false"),
        (
            "shell_environment_policy.set.PATH",
            _toml_string("/usr/bin:/bin:/usr/sbin:/sbin"),
        ),
        ("shell_environment_policy.set.HOME", _toml_string("/nonexistent")),
        ("allow_login_shell", "false"),
        (
            f"permissions.{DISCORD_CODEX_PERMISSION_PROFILE}",
            "{"
            'description="Discord-only isolated workspace",'
            'extends=":workspace",'
            'filesystem={":minimal"="read",'
            '":workspace_roots"={"."="write"}},'
            "network={enabled=false}"
            "}",
        ),
    ]
    configured_mcp_servers = _configured_mcp_server_names(codex_home)
    plugin_mcp_servers = _plugin_mcp_server_names(codex_home)
    disabled_mcp_servers = (
        configured_mcp_servers
        | plugin_mcp_servers
        | _DISCORD_CODEX_ALWAYS_DISABLED_MCP_SERVERS
    )
    for server in sorted(disabled_mcp_servers):
        settings.append((f"mcp_servers.{server}.enabled", "false"))
    for server in sorted(plugin_mcp_servers - configured_mcp_servers):
        settings.extend(
            (
                (f"mcp_servers.{server}.command", _toml_string("/usr/bin/false")),
            )
        )
    for policy in DISCORD_CODEX_APP_POLICIES:
        prefix = f"apps.{policy.app_id}"
        settings.extend(
            (
                (f"{prefix}.enabled", "false"),
                (f"{prefix}.destructive_enabled", "false"),
                (f"{prefix}.open_world_enabled", "false"),
                (f"{prefix}.default_tools_enabled", "false"),
                (f"{prefix}.default_tools_approval_mode", _toml_string("prompt")),
            )
        )
    # The CLI override parser does not apply quoted segments inside a dotted
    # key (``plugins."name@market".enabled``). Replace the whole process-local
    # table instead, which also prevents values from the user's global table
    # from leaking into the Discord policy.
    plugin_settings = (
        *((plugin, True) for plugin in DISCORD_CODEX_ENABLED_PLUGINS),
        *((plugin, False) for plugin in DISCORD_CODEX_DISABLED_PLUGINS),
    )
    settings.append(
        (
            "plugins",
            "{"
            + ",".join(
                _plugin_toml_entry(plugin, enabled)
                for plugin, enabled in plugin_settings
            )
            + "}",
        )
    )

    disabled_skill_paths = _disabled_skill_paths(codex_home)
    if disabled_skill_paths:
        skill_config = ",".join(
            f"{{path={_toml_string(str(path))},enabled=false}}"
            for path in disabled_skill_paths
        )
        settings.append(("skills.config", f"[{skill_config}]"))

    return tuple(
        argument
        for key, value in settings
        for argument in ("-c", f"{key}={value}")
    )


def _disabled_skill_paths(codex_home: Path) -> tuple[Path, ...]:
    skills_root = codex_home / "skills"
    disabled: set[Path] = set()
    if skills_root.is_dir():
        for skill_file in skills_root.glob("*/SKILL.md"):
            name = skill_file.parent.name
            if name in _DISABLED_SKILL_NAMES or name.startswith(
                _DISABLED_SKILL_PREFIXES
            ):
                disabled.add(skill_file.resolve())

    plugin_cache = codex_home / "plugins" / "cache"
    if plugin_cache.is_dir():
        for skill_file in plugin_cache.glob("*/*/*/skills/*/SKILL.md"):
            relative = skill_file.relative_to(plugin_cache)
            plugin_name = relative.parts[1]
            skill_name = skill_file.parent.name
            disabled_names = _DISABLED_PLUGIN_SKILLS.get(plugin_name, frozenset())
            if disabled_names is None or skill_name in disabled_names:
                disabled.add(skill_file.resolve())
    return tuple(sorted(disabled))


_MCP_SERVER_NAME = re.compile(r"[A-Za-z0-9_-]{1,200}\Z")


def _configured_mcp_server_names(codex_home: Path) -> set[str]:
    """Discover every user-configured MCP so the Discord process can disable it."""

    config = codex_home / "config.toml"
    if not config.is_file():
        return set()
    try:
        if config.stat().st_size > 2_000_000:
            raise RuntimeError("Codex config is too large to validate safely")
        raw = tomllib.loads(config.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError("Codex MCP configuration could not be validated") from exc
    servers = raw.get("mcp_servers")
    if servers is None:
        return set()
    if not isinstance(servers, dict):
        raise RuntimeError("Codex mcp_servers configuration is invalid")
    return _validated_mcp_server_names(servers, source="Codex config")


def _plugin_mcp_server_names(codex_home: Path) -> set[str]:
    """Discover every cached plugin MCP while retaining its instruction skills."""

    names = set(_DISCORD_CODEX_DISABLED_MCP_SERVERS)
    plugin_cache = codex_home / "plugins" / "cache"
    if not plugin_cache.is_dir():
        return names
    for manifest in plugin_cache.rglob(".mcp.json"):
        try:
            if manifest.stat().st_size > 1_000_000:
                raise RuntimeError(f"Plugin MCP manifest is too large: {manifest}")
            raw = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Plugin MCP manifest is invalid: {manifest}") from exc
        servers = raw.get("mcpServers") if isinstance(raw, dict) else None
        if not isinstance(servers, dict):
            raise RuntimeError(f"Plugin MCP manifest has no valid mcpServers: {manifest}")
        names.update(_validated_mcp_server_names(servers, source=str(manifest)))
    return names


def _validated_mcp_server_names(
    servers: dict[object, object],
    *,
    source: str,
) -> set[str]:
    names: set[str] = set()
    for name in servers:
        if not isinstance(name, str) or _MCP_SERVER_NAME.fullmatch(name) is None:
            raise RuntimeError(f"MCP server name in {source} cannot be disabled safely")
        names.add(name)
    return names


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _plugin_toml_entry(plugin: str, enabled: bool) -> str:
    fields = [f"enabled={'true' if enabled else 'false'}"]
    if plugin == "record-and-replay@openai-bundled":
        fields.append('mcp_servers={"event-stream"={enabled=false}}')
    return f"{_toml_string(plugin)}={{{','.join(fields)}}}"
