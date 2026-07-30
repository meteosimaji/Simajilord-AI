from __future__ import annotations

from pathlib import Path

import pytest

from simajilord.diagnostics.discord_api_coverage import (
    INTERNAL_CAPABILITIES,
    OfficialRoute,
    capability_inventory,
    classify_route,
    declaration_count,
    extract_official_routes,
)


@pytest.mark.parametrize(
    ("method", "path", "category", "capability"),
    (
        (
            "GET",
            "/guilds/{guild.id}/audit-logs",
            "model_full",
            "discord.list_platform_resources",
        ),
        (
            "PUT",
            "/channels/{channel.id}/voice-status",
            "model_full",
            "discord.channel_operation",
        ),
        (
            "GET",
            "/guilds/templates/{template.code}",
            "model_partial",
            "discord.list_platform_resources",
        ),
        ("GET", "/gateway/bot", "host_adapter", None),
        ("GET", "/users/@me/connections", "oauth_user", None),
        (
            "POST",
            "/webhooks/{webhook.id}/{webhook.token}",
            "webhook_token",
            None,
        ),
        ("POST", "/lobbies", "social_sdk", None),
        (
            "POST",
            "/guilds/{guild.id}/prune",
            "intentionally_unavailable",
            None,
        ),
    ),
)
def test_official_route_classifier_has_explicit_security_categories(
    method: str,
    path: str,
    category: str,
    capability: str | None,
) -> None:
    coverage = classify_route(OfficialRoute(method, path, ("fixture.mdx:1",)))

    assert coverage.category == category
    if capability is not None:
        assert capability in coverage.capabilities


def test_official_route_extractor_normalizes_links_and_retains_duplicates(
    tmp_path: Path,
) -> None:
    developers = tmp_path / "developers" / "resources"
    developers.mkdir(parents=True)
    (developers / "message.mdx").write_text(
        "\n".join(
            (
                '<Route method="GET">/channels/'
                "[\\{channel.id\\}](/developers/resources/channel)"
                "/messages</Route>",
                '<Route method="GET">/channels/'
                "[\\{channel.id\\}](/developers/resources/channel)"
                "/messages</Route>",
            )
        ),
        encoding="utf-8",
    )

    routes = extract_official_routes(tmp_path)

    assert len(routes) == 1
    assert declaration_count(routes) == 2
    assert routes[0].path == "/channels/{channel.id}/messages"
    assert routes[0].sources == (
        "developers/resources/message.mdx:1",
        "developers/resources/message.mdx:2",
    )


def test_all_discord_capabilities_have_unique_typed_implementation_evidence() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    inventory = capability_inventory(repo_root)
    names = [str(item["name"]) for item in inventory]

    assert len(inventory) == 106
    assert len(names) == len(set(names))
    assert set(names) > INTERNAL_CAPABILITIES
    assert sum(item["model_facing"] is True for item in inventory) == 100
    for item in inventory:
        implementation = item["implementation"]
        request = item["request"]
        response = item["response"]
        assert isinstance(implementation, dict)
        assert isinstance(request, dict)
        assert isinstance(response, dict)
        implementation_path = implementation["path"]
        assert isinstance(implementation_path, str)
        assert (repo_root / implementation_path).is_file()
        assert isinstance(implementation["line"], int)
        assert isinstance(request["type"], str)
        assert isinstance(request["fields"], list)
        assert isinstance(response["type"], str)
        assert isinstance(response["fields"], list)


def test_unclassified_official_route_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unclassified official Discord route"):
        classify_route(OfficialRoute("GET", "/future/unknown", ("future.mdx:1",)))
