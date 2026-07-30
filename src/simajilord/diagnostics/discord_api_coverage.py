"""Audit Simajilord's typed Discord surface against Discord's official docs.

The official documentation repository is deliberately an input rather than a
vendored snapshot. This keeps the comparison reproducible while making upstream
route drift fail visibly instead of silently changing a claimed coverage count.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import subprocess
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, cast, get_args, get_origin, get_type_hints
from unittest.mock import Mock

import discord

from simajilord.integrations.discord.capabilities import build_discord_endpoints
from simajilord.runtime import SimajilordRuntime

CoverageCategory = Literal[
    "model_full",
    "model_partial",
    "host_adapter",
    "oauth_user",
    "webhook_token",
    "social_sdk",
    "intentionally_unavailable",
]
ProbePolicy = Literal[
    "safe_read",
    "safe_disposable_write",
    "guarded_or_destructive",
    "host_runtime",
    "credential_unavailable",
    "environment_limited",
]

EXPECTED_DECLARATIONS = 238
EXPECTED_UNIQUE_ROUTES = 229
EXPECTED_CAPABILITIES = 106
INTERNAL_CAPABILITIES = frozenset(
    {
        "discord.control_audio",
        "discord.delete_created_channel",
        "discord.delete_created_role",
        "discord.delete_own_messages",
        "discord.expand_message",
        "discord.manage_read_aloud",
    }
)
_ROUTE_PATTERN = re.compile(r'<Route\s+method="([A-Z]+)">(.*?)</Route>')
_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^]]+)\]\([^)]*\)")
_PERMISSION_GUARD_PATTERN = re.compile(r"\b(_(?:assert|can|require|write)_[A-Za-z0-9_]+)\b")


@dataclass(frozen=True, slots=True)
class OfficialRoute:
    method: str
    path: str
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RouteCoverage:
    category: CoverageCategory
    probe_policy: ProbePolicy
    capabilities: tuple[str, ...] = ()
    note: str = ""


def extract_official_routes(docs_root: Path) -> tuple[OfficialRoute, ...]:
    """Extract and deduplicate every documented HTTP route declaration."""

    developers = docs_root / "developers"
    if not developers.is_dir():
        raise ValueError(f"Discord docs developers directory not found: {developers}")
    declarations: dict[tuple[str, str], list[str]] = {}
    for path in sorted(developers.rglob("*.mdx")):
        relative_path = path.relative_to(docs_root).as_posix()
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            for match in _ROUTE_PATTERN.finditer(line):
                route_path = _normalize_route_path(match.group(2))
                declarations.setdefault(
                    (match.group(1), route_path),
                    [],
                ).append(f"{relative_path}:{line_number}")
    return tuple(
        OfficialRoute(method=method, path=route_path, sources=tuple(sources))
        for (method, route_path), sources in sorted(declarations.items())
    )


def declaration_count(routes: tuple[OfficialRoute, ...]) -> int:
    return sum(len(route.sources) for route in routes)


def classify_route(route: OfficialRoute) -> RouteCoverage:
    """Classify one official route without a silent catch-all."""

    method = route.method
    path = route.path

    if path.startswith("/lobbies"):
        return RouteCoverage(
            "social_sdk",
            "environment_limited",
            note="Discord Social SDK lobby state is outside this bot/Gateway adapter.",
        )
    if _is_interaction_webhook(path) or path.startswith("/interactions/"):
        return RouteCoverage(
            "host_adapter",
            "host_runtime",
            note="discord.py owns interaction acknowledgement and follow-up tokens.",
        )
    if _is_webhook_token_route(path):
        return RouteCoverage(
            "webhook_token",
            "credential_unavailable",
            note="Webhook tokens are intentionally never returned to or accepted from the model.",
        )
    if path.startswith("/oauth2/"):
        if path == "/oauth2/applications/@me":
            return RouteCoverage(
                "host_adapter",
                "host_runtime",
                ("discord.inspect_application",),
                "Application identity is exposed through a token-free typed record.",
            )
        return RouteCoverage(
            "oauth_user",
            "credential_unavailable",
            note="Requires an OAuth bearer identity rather than the configured Bot token.",
        )
    if path.startswith("/applications/") and "/commands" in path:
        return RouteCoverage(
            "host_adapter",
            "host_runtime",
            note="Application-command registration and permissions are startup-owned.",
        )
    if path in {"/gateway", "/gateway/bot"}:
        return RouteCoverage(
            "host_adapter",
            "host_runtime",
            note="discord.py owns Gateway discovery, identify, resume, and shard state.",
        )
    if path == "/channels/{channel.id}/typing":
        return RouteCoverage(
            "host_adapter",
            "host_runtime",
            note="Typing is emitted only while observable model/tool work is active.",
        )
    if path == "/applications/{application.id}/activity-instances/{instance_id}":
        return RouteCoverage(
            "host_adapter",
            "environment_limited",
            note="Activity instances belong to the optional Embedded App host.",
        )

    oauth_user_paths = {
        "/channels/{channel.id}/recipients/{user.id}",
        "/users/@me/applications/{application.id}/role-connection",
        "/users/@me/connections",
        "/users/@me/guilds",
        "/users/@me/guilds/{guild.id}/member",
    }
    if path in oauth_user_paths:
        return RouteCoverage(
            "oauth_user",
            "credential_unavailable",
            note="This route needs a user OAuth scope that Simajilord does not request.",
        )
    if method == "PUT" and path == "/guilds/{guild.id}/members/{user.id}":
        return RouteCoverage(
            "oauth_user",
            "credential_unavailable",
            note="Adding a member requires that user's OAuth access token.",
        )

    intentional_exact = {
        ("PATCH", "/applications/@me"): "Bot identity/developer settings stay host-owned.",
        (
            "PUT",
            "/applications/{application.id}/role-connections/metadata",
        ): "Role-verification schema deployment stays host-owned.",
        (
            "DELETE",
            "/guilds/{guild.id}/integrations/{integration.id}",
        ): "Deleting external integrations is intentionally unavailable.",
        ("POST", "/guilds/{guild.id}/prune"): "Member pruning is intentionally unavailable.",
        (
            "PUT",
            "/guilds/{guild.id}/incident-actions",
        ): "Guild incident mutation is intentionally unavailable.",
        (
            "PUT",
            "/guilds/{guild.id}/onboarding",
        ): "Community onboarding deployment stays host-owned.",
        (
            "PATCH",
            "/guilds/{guild.id}/welcome-screen",
        ): "Community welcome-screen deployment stays host-owned.",
        (
            "PATCH",
            "/guilds/{guild.id}/widget",
        ): "Public widget configuration stays host-owned.",
        (
            "DELETE",
            "/users/@me/guilds/{guild.id}",
        ): "The model cannot make the bot leave a server.",
        ("PATCH", "/users/@me"): "The model cannot mutate the bot account identity.",
        (
            "GET",
            "/invites/{invite.code}/target-users",
        ): "Targeted-invite CSV membership is not exposed to the model.",
        (
            "GET",
            "/invites/{invite.code}/target-users/job-status",
        ): "Targeted-invite upload jobs are not exposed to the model.",
        (
            "PUT",
            "/invites/{invite.code}/target-users",
        ): "Targeted-invite CSV uploads are intentionally unavailable.",
    }
    if (method, path) in intentional_exact:
        return RouteCoverage(
            "intentionally_unavailable",
            "guarded_or_destructive",
            note=intentional_exact[(method, path)],
        )
    if path.startswith("/applications/{application.id}/entitlements") and method != "GET":
        return RouteCoverage(
            "intentionally_unavailable",
            "environment_limited",
            note="Test entitlement creation, deletion, and consumption are not model tools.",
        )

    model_coverage = _model_route_coverage(method, path)
    if model_coverage is not None:
        return model_coverage
    raise ValueError(f"Unclassified official Discord route: {method} {path}")


def capability_inventory(repo_root: Path) -> list[dict[str, object]]:
    """Introspect all typed Discord endpoints and their concrete handler locations."""

    endpoints = build_discord_endpoints(
        cast(discord.Client, Mock(spec=discord.Client)),
        Mock(spec=SimajilordRuntime),
    )
    inventory: list[dict[str, object]] = []
    for capability in sorted(endpoints, key=lambda item: item.descriptor.name):
        handler = inspect.getclosurevars(capability.invoke).nonlocals.get("handler")
        if not callable(handler):
            raise RuntimeError(f"Could not recover handler for {capability.descriptor.name}")
        source_file = inspect.getsourcefile(handler)
        source_lines, source_line = inspect.getsourcelines(handler)
        source_text = "".join(source_lines)
        implementation_path = (
            Path(source_file).resolve().relative_to(repo_root.resolve()).as_posix()
            if source_file is not None
            else None
        )
        inventory.append(
            {
                "name": capability.descriptor.name,
                "model_facing": capability.descriptor.name not in INTERNAL_CAPABILITIES,
                "implementation": {
                    "path": implementation_path,
                    "line": source_line,
                    "handler": getattr(handler, "__name__", type(handler).__name__),
                },
                "request": _model_schema(capability.request_type),
                "response": _model_schema(capability.response_type),
                "risk": capability.descriptor.risk.value,
                "approval": capability.descriptor.approval.value,
                "idempotency": capability.descriptor.idempotency,
                "requires_workspace": capability.descriptor.requires_workspace,
                "requires_voice": capability.descriptor.requires_voice,
                "requires_same_voice": capability.descriptor.requires_same_voice,
                "timeout_seconds": capability.descriptor.timeout_seconds,
                "summary": capability.descriptor.summary,
                "side_effects": capability.descriptor.side_effects,
                "expected_errors": capability.descriptor.expected_errors,
                "direct_guard_calls": sorted(set(_PERMISSION_GUARD_PATTERN.findall(source_text))),
            }
        )
    return inventory


def build_report(docs_root: Path, repo_root: Path) -> dict[str, object]:
    routes = extract_official_routes(docs_root)
    capabilities = capability_inventory(repo_root)
    capability_names = {str(item["name"]) for item in capabilities}
    classified_routes: list[dict[str, object]] = []
    coverage_counts: dict[str, int] = {}
    unknown_capability_references: dict[str, list[str]] = {}
    for route in routes:
        coverage = classify_route(route)
        coverage_counts[coverage.category] = coverage_counts.get(coverage.category, 0) + 1
        missing = sorted(set(coverage.capabilities) - capability_names)
        if missing:
            unknown_capability_references[f"{route.method} {route.path}"] = missing
        classified_routes.append(
            {
                **asdict(route),
                **asdict(coverage),
            }
        )
    if unknown_capability_references:
        raise RuntimeError(
            "Official route map references unknown capabilities: "
            + json.dumps(unknown_capability_references, ensure_ascii=False)
        )
    duplicate_declarations = declaration_count(routes) - len(routes)
    return {
        "official_docs": {
            "path": str(docs_root.resolve()),
            "git_commit": _git_commit(docs_root),
            "route_declarations": declaration_count(routes),
            "unique_routes": len(routes),
            "duplicate_declarations": duplicate_declarations,
        },
        "simajilord": {
            "capability_count": len(capabilities),
            "model_facing_capabilities": sum(item["model_facing"] is True for item in capabilities),
            "internal_capabilities": sorted(INTERNAL_CAPABILITIES),
        },
        "coverage_counts": dict(sorted(coverage_counts.items())),
        "routes": classified_routes,
        "capabilities": capabilities,
    }


def validate_report(report: dict[str, object]) -> None:
    official = cast(dict[str, object], report["official_docs"])
    simajilord = cast(dict[str, object], report["simajilord"])
    problems: list[str] = []
    if official["route_declarations"] != EXPECTED_DECLARATIONS:
        problems.append(
            "official route declaration drift: "
            f"{official['route_declarations']} != {EXPECTED_DECLARATIONS}"
        )
    if official["unique_routes"] != EXPECTED_UNIQUE_ROUTES:
        problems.append(
            f"official unique route drift: {official['unique_routes']} != {EXPECTED_UNIQUE_ROUTES}"
        )
    if simajilord["capability_count"] != EXPECTED_CAPABILITIES:
        problems.append(
            "Discord capability count drift: "
            f"{simajilord['capability_count']} != {EXPECTED_CAPABILITIES}"
        )
    if problems:
        raise RuntimeError("; ".join(problems))


def _model_route_coverage(method: str, path: str) -> RouteCoverage | None:
    full: tuple[str, ...] | None = None
    partial: tuple[str, ...] | None = None

    if path == "/applications/@me" and method == "GET":
        full = ("discord.inspect_application",)
    elif path.startswith("/applications/{application.id}/emojis"):
        full = _crud_capabilities(
            method,
            read="discord.list_platform_resources",
            create="discord.create_platform_asset",
            update="discord.update_platform_asset",
            delete="discord.delete_platform_asset",
        )
    elif method == "GET" and (
        path.startswith("/applications/{application.id}/entitlements")
        or path == "/applications/{application.id}/role-connections/metadata"
        or path == "/applications/{application.id}/skus"
        or path.startswith("/skus/{sku.id}/subscriptions")
    ):
        full = ("discord.list_platform_resources",)
    elif path == "/channels/{channel.id}":
        full = {
            "GET": ("discord.inspect_channel",),
            "PATCH": (
                "discord.update_channel_settings",
                "discord.update_guild_resource",
                "discord.update_thread",
            ),
            "DELETE": (
                "discord.delete_created_channel",
                "discord.delete_guild_resource",
            ),
        }.get(method)
    elif path == "/channels/{channel.id}/invites":
        full = {
            "GET": ("discord.list_platform_resources",),
            "POST": ("discord.create_guild_resource",),
        }.get(method)
    elif (
        path
        in {
            "/channels/{channel.id}/messages/pins",
            "/channels/{channel.id}/pins",
        }
        and method == "GET"
    ):
        full = ("discord.list_pins",)
    elif path in {
        "/channels/{channel.id}/messages/pins/{message.id}",
        "/channels/{channel.id}/pins/{message.id}",
    }:
        full = {
            "PUT": ("discord.pin_message",),
            "DELETE": ("discord.unpin_message",),
        }.get(method)
    elif path == "/channels/{channel.id}/messages":
        full = {
            "GET": ("discord.read_messages", "discord.search_messages"),
            "POST": (
                "discord.send_message",
                "discord.send_embed",
                "discord.reply_message",
                "discord.send_file",
                "discord.send_files",
                "discord.create_poll",
                "discord.forward_message",
            ),
        }.get(method)
    elif path == "/channels/{channel.id}/messages/{message.id}":
        full = {
            "GET": ("discord.get_message",),
            "PATCH": ("discord.edit_own_message",),
            "DELETE": ("discord.delete_message", "discord.delete_own_message"),
        }.get(method)
    elif path == "/channels/{channel.id}/messages/bulk-delete" and method == "POST":
        full = ("discord.bulk_delete_messages",)
    elif (path.endswith("/crosspost") and method == "POST") or (
        path.endswith("/reactions") and method == "DELETE"
    ):
        full = ("discord.message_action",)
    elif "/reactions/{emoji.id}" in path:
        if path.endswith("/@me"):
            full = {
                "PUT": ("discord.add_reaction",),
                "DELETE": ("discord.remove_own_reaction",),
            }.get(method)
        elif path.endswith("/{user.id}") and method == "DELETE":
            partial = ("discord.message_action",)
        elif method == "GET":
            full = ("discord.list_reaction_users",)
        elif method == "DELETE":
            full = ("discord.message_action",)
    elif "/polls/" in path:
        full = {
            "GET": ("discord.list_poll_voters",),
            "POST": ("discord.message_action",),
        }.get(method)
    elif path.endswith("/threads") and method == "POST":
        full = ("discord.create_thread", "discord.create_forum_post")
    elif "/threads/archived/" in path and method == "GET":
        full = ("discord.list_archived_threads",)
    elif "/thread-members" in path:
        if method == "GET":
            full = ("discord.list_thread_members",)
        elif path.endswith("/@me"):
            full = ("discord.channel_operation",)
        else:
            full = {
                "PUT": ("discord.add_thread_member",),
                "DELETE": ("discord.remove_thread_member",),
            }.get(method)
    elif (method == "POST" and path.endswith(("/followers", "/send-soundboard-sound"))) or (
        method == "PUT" and path.endswith("/voice-status")
    ):
        full = ("discord.channel_operation",)
    elif "/permissions/{overwrite.id}" in path:
        full = ("discord.set_channel_overwrite",)
    elif path.endswith("/webhooks"):
        full = {
            "GET": ("discord.list_platform_resources",),
            "POST": ("discord.create_guild_resource",),
        }.get(method)

    if path == "/guilds/templates/{template.code}" and method == "GET":
        partial = ("discord.list_platform_resources",)
    elif path == "/guilds/{guild.id}":
        full = {
            "GET": ("discord.inspect_server",),
            "PATCH": ("discord.update_guild_resource",),
        }.get(method)
    elif path == "/guilds/{guild.id}/audit-logs" and method == "GET":
        full = ("discord.list_platform_resources",)
    elif "/auto-moderation/rules" in path:
        full = _crud_capabilities(
            method,
            read="discord.list_platform_resources",
            create="discord.create_automod_rule",
            update="discord.update_automod_rule",
            delete="discord.delete_automod_rule",
        )
    elif "/bans" in path:
        if (path.endswith("/bans") and method == "GET") or method == "GET":
            full = ("discord.list_platform_resources",)
        elif method == "PUT":
            full = ("discord.ban_member",)
        elif method == "DELETE":
            full = ("discord.unban_member",)
    elif path.endswith("/bulk-ban") and method == "POST":
        partial = ("discord.ban_member",)
    elif path == "/guilds/{guild.id}/channels":
        full = {
            "GET": ("discord.list_channels",),
            "POST": (
                "discord.create_channel",
                "discord.create_guild_resource",
            ),
        }.get(method)
        if method == "PATCH":
            partial = ("discord.update_guild_resource",)
    elif "/emojis" in path:
        full = _crud_capabilities(
            method,
            read="discord.list_platform_resources",
            create="discord.create_platform_asset",
            update="discord.update_platform_asset",
            delete="discord.delete_platform_asset",
        )
    elif method == "GET" and path in {
        "/guilds/{guild.id}/integrations",
        "/guilds/{guild.id}/invites",
    }:
        full = ("discord.list_platform_resources",)
    elif path == "/guilds/{guild.id}/members":
        full = ("discord.list_members",) if method == "GET" else None
    elif path == "/guilds/{guild.id}/members/search" and method == "GET":
        full = ("discord.list_members",)
    elif path == "/guilds/{guild.id}/members/{user.id}":
        full = {
            "GET": ("discord.inspect_user",),
            "PATCH": (
                "discord.set_timeout",
                "discord.update_guild_resource",
            ),
            "DELETE": ("discord.kick_member",),
        }.get(method)
    elif (
        path
        in {
            "/guilds/{guild.id}/members/@me",
            "/guilds/{guild.id}/members/@me/nick",
        }
        and method == "PATCH"
    ):
        partial = ("discord.update_guild_resource",)
    elif path.endswith("/members/{user.id}/roles/{role.id}"):
        full = {
            "PUT": ("discord.assign_role",),
            "DELETE": ("discord.remove_role",),
        }.get(method)
    elif path == "/guilds/{guild.id}/messages/search" and method == "GET":
        full = ("discord.search_messages",)
    elif method == "GET" and path in {
        "/guilds/{guild.id}/onboarding",
        "/guilds/{guild.id}/preview",
        "/guilds/{guild.id}/prune",
        "/guilds/{guild.id}/regions",
    }:
        full = ("discord.list_platform_resources",)
    elif path.startswith("/guilds/{guild.id}/roles"):
        if method == "GET":
            full = (
                "discord.list_roles",
                "discord.list_platform_resources",
            )
        elif method == "POST":
            full = ("discord.create_role",)
        elif method == "PATCH":
            full = ("discord.update_guild_resource",)
        elif method == "DELETE":
            full = ("discord.delete_guild_resource",)
    elif "/scheduled-events" in path:
        full = _crud_capabilities(
            method,
            read="discord.list_platform_resources",
            create="discord.create_guild_resource",
            update="discord.update_guild_resource",
            delete="discord.delete_guild_resource",
        )
    elif "/soundboard-sounds" in path or ("/stickers" in path and path.startswith("/guilds/")):
        full = _crud_capabilities(
            method,
            read="discord.list_platform_resources",
            create="discord.create_platform_asset",
            update="discord.update_platform_asset",
            delete="discord.delete_platform_asset",
        )
    elif "/templates" in path and path.startswith("/guilds/"):
        full = _crud_capabilities(
            method,
            read="discord.list_platform_resources",
            create="discord.create_guild_resource",
            update="discord.update_guild_resource",
            delete="discord.delete_guild_resource",
        )
    elif method == "GET" and path in {
        "/guilds/{guild.id}/threads/active",
        "/guilds/{guild.id}/vanity-url",
    }:
        full = ("discord.list_platform_resources",)
    elif "/voice-states/" in path:
        if method == "GET":
            full = ("discord.list_voice_states",)
        elif method == "PATCH":
            full = ("discord.update_guild_resource",)
    elif method == "GET" and path in {
        "/guilds/{guild.id}/webhooks",
        "/guilds/{guild.id}/welcome-screen",
        "/guilds/{guild.id}/widget",
        "/guilds/{guild.id}/widget.json",
        "/guilds/{guild.id}/widget.png",
    }:
        full = ("discord.list_platform_resources",)

    if path == "/invites/{invite.code}":
        full = {
            "GET": ("discord.list_platform_resources",),
            "DELETE": ("discord.delete_guild_resource",),
        }.get(method)
    elif path == "/soundboard-default-sounds" and method == "GET":
        full = ("discord.list_platform_resources",)
    elif path.startswith("/stage-instances"):
        full = _crud_capabilities(
            method,
            read="discord.list_platform_resources",
            create="discord.create_guild_resource",
            update="discord.update_guild_resource",
            delete="discord.delete_guild_resource",
        )
    elif path.startswith("/sticker-packs") and method == "GET":
        full = ("discord.list_platform_resources",)
    elif path == "/stickers/{sticker.id}" and method == "GET":
        partial = ("discord.view_sticker",)
    elif path == "/users/@me" and method == "GET":
        full = ("discord.inspect_application",)
    elif path == "/users/{user.id}" and method == "GET":
        full = ("discord.inspect_user",)
    elif path == "/users/@me/channels" and method == "POST":
        full = ("discord.send_direct_message",)
    elif path == "/voice/regions" and method == "GET":
        full = ("discord.list_platform_resources",)
    elif path == "/webhooks/{webhook.id}":
        full = _crud_capabilities(
            method,
            read="discord.list_platform_resources",
            update="discord.update_guild_resource",
            delete="discord.delete_guild_resource",
        )

    if full is not None:
        probe_policy: ProbePolicy = (
            "safe_read"
            if method == "GET"
            else "guarded_or_destructive"
            if method == "DELETE"
            else "safe_disposable_write"
        )
        return RouteCoverage("model_full", probe_policy, full)
    if partial is not None:
        probe_policy = (
            "safe_read" if method == "GET" else "guarded_or_destructive"
        )
        return RouteCoverage(
            "model_partial",
            probe_policy,
            partial,
            (
                "The typed capability covers the common bounded operation, "
                "not every raw route variant."
            ),
        )
    return None


def _crud_capabilities(
    method: str,
    *,
    read: str | None = None,
    create: str | None = None,
    update: str | None = None,
    delete: str | None = None,
) -> tuple[str, ...] | None:
    capability = {
        "GET": read,
        "POST": create,
        "PUT": update,
        "PATCH": update,
        "DELETE": delete,
    }.get(method)
    return (capability,) if capability is not None else None


def _is_interaction_webhook(path: str) -> bool:
    return path.startswith("/webhooks/{application.id}/{interaction.token}")


def _is_webhook_token_route(path: str) -> bool:
    return path.startswith("/webhooks/{webhook.id}/{webhook.token}")


def _normalize_route_path(value: str) -> str:
    return _MARKDOWN_LINK_PATTERN.sub(r"\1", value).replace("\\{", "{").replace("\\}", "}").strip()


def _model_schema(model: type[Any]) -> dict[str, object]:
    hints = get_type_hints(model)
    return {
        "type": model.__name__,
        "fields": [
            {
                "name": item.name,
                "type": _type_name(hints.get(item.name, Any)),
                "required": item.default is MISSING and item.default_factory is MISSING,
                "description": item.metadata.get("description"),
            }
            for item in fields(model)
        ],
    }


def _type_name(annotation: object) -> str:
    origin = get_origin(annotation)
    if origin in {UnionType, Union}:
        return " | ".join(_type_name(item) for item in get_args(annotation))
    if origin is not None:
        arguments = get_args(annotation)
        base = getattr(origin, "__name__", str(origin).replace("typing.", ""))
        return f"{base}[{', '.join(_type_name(item) for item in arguments)}]" if arguments else base
    return getattr(annotation, "__name__", str(annotation).replace("typing.", ""))


def _git_commit(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--discord-docs",
        required=True,
        type=Path,
        help="Checkout of discord/discord-api-docs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. The parent directory must already exist.",
    )
    parser.add_argument(
        "--allow-upstream-drift",
        action="store_true",
        help="Report changed official route counts without failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    report = build_report(args.discord_docs, repo_root)
    if not args.allow_upstream_drift:
        validate_report(report)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    summary = {
        "official_docs": report["official_docs"],
        "simajilord": report["simajilord"],
        "coverage_counts": report["coverage_counts"],
        "output": str(args.output.resolve()) if args.output is not None else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
