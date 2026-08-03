from __future__ import annotations

import json

import pytest

from simajilord.agent import (
    AGENT_HIGH_RISK_CAPABILITIES,
    AgentHighRiskReviewField,
)
from simajilord.agent.high_risk import (
    HighRiskPresentationError,
    high_risk_presentation,
)


def test_every_high_risk_capability_has_one_structured_presenter() -> None:
    presentations = {
        capability: high_risk_presentation(capability, {})
        for capability in AGENT_HIGH_RISK_CAPABILITIES
    }

    assert presentations.keys() == AGENT_HIGH_RISK_CAPABILITIES
    assert all(item.public_action for item in presentations.values())
    assert all(item.public_target for item in presentations.values())
    assert all(item.review_fields for item in presentations.values())


def test_sensitive_direct_message_values_exist_only_in_private_fields() -> None:
    secret = "private launch details"
    presentation = high_risk_presentation(
        "discord.send_direct_message",
        {
            "user_id": "123456789",
            "content": secret,
            "purpose": "requested_action",
            "authorization_event_id": "discord:message:999",
        },
    )

    public_snapshot = json.dumps(
        {
            "action": presentation.public_action,
            "target": presentation.public_target,
        }
    )
    private_snapshot = "\n".join(
        field.value for field in presentation.review_fields
    )
    assert secret not in public_snapshot
    assert "123456789" not in public_snapshot
    assert secret in private_snapshot
    assert "123456789" in private_snapshot
    assert "requested_action" in private_snapshot
    assert "authorization_event_id" not in private_snapshot


def test_shell_and_connector_payloads_are_complete_and_structured() -> None:
    shell = high_risk_presentation(
        "system.shell",
        {
            "argv": ["/usr/bin/swift", "build", "--configuration", "release"],
            "working_directory": "native/helper",
            "timeout_seconds": 120,
        },
    )
    connector = high_risk_presentation(
        "connector.destructive",
        {
            "connector_id": "design",
            "tool": "delete_asset",
            "contract_id": "contract-bound-to-this-turn",
            "arguments": {"asset_id": "asset-7", "force": True},
        },
    )

    shell_fields = {field.name: field.value for field in shell.review_fields}
    connector_fields = {
        field.name: field.value for field in connector.review_fields
    }
    assert shell_fields["Provider or process payload"] == (
        'argv: ["/usr/bin/swift","build","--configuration","release"]\n'
        "timeout_seconds: 120"
    )
    assert shell_fields["Exact target"] == 'working_directory: "native/helper"'
    assert connector_fields["Exact target"] == (
        'connector_id: "design"\n'
        'tool: "delete_asset"\n'
        'contract_id: "contract-bound-to-this-turn"'
    )
    assert connector_fields["Provider or process payload"] == (
        'arguments: {"asset_id":"asset-7","force":true}'
    )


@pytest.mark.parametrize(
    ("capability", "arguments", "secret"),
    (
        (
            "discord.ban_member",
            {"user_id": "7", "reason": "private moderation reason"},
            "private moderation reason",
        ),
        (
            "system.shell",
            {"argv": ["/bin/echo", "private shell argument"]},
            "private shell argument",
        ),
        (
            "connector.destructive",
            {
                "connector_id": "design",
                "tool": "delete",
                "contract_id": "contract",
                "arguments": {"private": "connector payload"},
            },
            "connector payload",
        ),
    ),
)
def test_sensitive_high_risk_fields_never_enter_public_summary(
    capability: str,
    arguments: dict[str, object],
    secret: str,
) -> None:
    presentation = high_risk_presentation(capability, arguments)

    public_snapshot = presentation.public_action + presentation.public_target
    private_snapshot = "\n".join(
        field.value for field in presentation.review_fields
    )
    assert secret not in public_snapshot
    assert secret in private_snapshot


def test_private_review_never_truncates_oversize_values() -> None:
    with pytest.raises(HighRiskPresentationError):
        high_risk_presentation(
            "discord.send_direct_message",
            {"user_id": "7", "content": "x" * 1_000},
        )
    with pytest.raises(ValueError):
        AgentHighRiskReviewField("Payload", "x" * 951)
