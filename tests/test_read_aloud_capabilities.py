from __future__ import annotations

import pytest

from simajilord.capabilities.read_aloud import (
    ReadAloudAddSourcesRequest,
    ReadAloudAnnouncementsSetRequest,
    ReadAloudDictionarySetRequest,
    ReadAloudExclusionSetRequest,
    ReadAloudExclusionTarget,
    ReadAloudPolicyResponse,
    ReadAloudResponse,
    ReadAloudSemanticsSetRequest,
    ReadAloudServerVoiceSetRequest,
    ReadAloudStatusRequest,
    ReadAloudUserVoiceSetRequest,
    build_read_aloud_policy_endpoints,
    build_read_aloud_route_endpoints,
)
from simajilord.core import ApprovalMode, InvocationContext, RiskLevel
from simajilord.services.read_aloud import (
    ReadAloudMode,
    ReadAloudService,
    ReadAloudVoicePreset,
)


def _context() -> InvocationContext:
    return InvocationContext(
        actor_id="user",
        workspace_id="guild",
        transport="test",
        request_id="request",
    )


@pytest.mark.asyncio
async def test_split_route_capabilities_have_one_action_per_schema(tmp_path) -> None:
    service = ReadAloudService(tmp_path / "read_aloud.json")
    endpoints = {
        item.descriptor.name: item
        for item in build_read_aloud_route_endpoints(service)
    }

    added = await endpoints["speech.read_aloud_add_sources"].invoke(
        ReadAloudAddSourcesRequest(
            text_channel_ids=("one", "two"),
            audio_destination_id="voice",
            mode=ReadAloudMode.QUEUE,
        ),
        _context(),
    )
    status = await endpoints["speech.read_aloud_status"].invoke(
        ReadAloudStatusRequest(),
        _context(),
    )

    assert isinstance(added, ReadAloudResponse)
    assert added.text_channel_ids == ("one", "two")
    assert status == ReadAloudResponse(
        action="status",
        enabled=True,
        text_channel_id="one",
        text_channel_ids=("one", "two"),
        audio_destination_id="voice",
        mode="queue",
    )
    assert endpoints["speech.read_aloud_status"].descriptor.risk is RiskLevel.READ
    assert (
        endpoints["speech.read_aloud_add_sources"].descriptor.approval
        is ApprovalMode.WHEN_REQUESTED
    )


@pytest.mark.asyncio
async def test_split_policy_capabilities_share_one_durable_policy(tmp_path) -> None:
    service = ReadAloudService(tmp_path / "read_aloud.json")
    endpoints = {
        item.descriptor.name: item
        for item in build_read_aloud_policy_endpoints(service)
    }
    context = _context()

    await endpoints["speech.read_aloud_dictionary_set"].invoke(
        ReadAloudDictionarySetRequest("IUT", "あいゆーてぃー"),
        context,
    )
    await endpoints["speech.read_aloud_exclusion_set"].invoke(
        ReadAloudExclusionSetRequest(
            target=ReadAloudExclusionTarget.USER,
            target_id="user-two",
            ignored=True,
        ),
        context,
    )
    response = await endpoints["speech.read_aloud_announcements_set"].invoke(
        ReadAloudAnnouncementsSetRequest(join=True, leave=True),
        context,
    )

    assert isinstance(response, ReadAloudPolicyResponse)
    assert response.dictionary[0].surface == "IUT"
    assert response.ignored_user_ids == ("user-two",)
    assert response.announce_join is True
    assert response.announce_leave is True
    assert response.announce_move is False
    assert response.previous_announce_join is False
    assert response.previous_announce_leave is False
    assert response.previous_announce_move is False
    semantics = await endpoints["speech.read_aloud_semantics_set"].invoke(
        ReadAloudSemanticsSetRequest(author_names=True, vc_members_only=True),
        context,
    )
    assert semantics.previous_read_author_names is True
    assert semantics.previous_read_replies is True
    assert semantics.previous_read_attachments is True
    assert semantics.previous_vc_members_only is False
    await endpoints["speech.read_aloud_server_voice_set"].invoke(
        ReadAloudServerVoiceSetRequest(ReadAloudVoicePreset.NARRATOR),
        context,
    )
    voice_response = await endpoints["speech.read_aloud_user_voice_set"].invoke(
        ReadAloudUserVoiceSetRequest(ReadAloudVoicePreset.CUTE),
        context,
    )
    assert voice_response.default_voice_preset == "narrator"
    assert voice_response.user_voice_presets == (("user", "cute"),)
    assert ReadAloudService(service.state_file).policy("guild") == service.policy(
        "guild"
    )


@pytest.mark.asyncio
async def test_read_aloud_compare_and_set_accepts_satisfied_undo_target(tmp_path) -> None:
    service = ReadAloudService(tmp_path / "read_aloud.json")
    endpoints = {
        item.descriptor.name: item
        for item in build_read_aloud_policy_endpoints(service)
    }
    context = _context()
    await endpoints["speech.read_aloud_announcements_set"].invoke(
        ReadAloudAnnouncementsSetRequest(join=True),
        context,
    )
    await endpoints["speech.read_aloud_announcements_set"].invoke(
        ReadAloudAnnouncementsSetRequest(join=False),
        context,
    )

    response = await endpoints["speech.read_aloud_announcements_set"].invoke(
        ReadAloudAnnouncementsSetRequest(
            join=False,
            expected_join=True,
        ),
        context,
    )

    assert response.announce_join is False
    assert response.previous_announce_join is False
    assert service.policy("guild").announce_join is False


def test_split_capability_names_are_unique_and_non_shadowing(tmp_path) -> None:
    service = ReadAloudService(tmp_path / "read_aloud.json")
    endpoints = (
        *build_read_aloud_route_endpoints(service),
        *build_read_aloud_policy_endpoints(service),
    )
    names = [item.descriptor.name for item in endpoints]

    assert len(names) == len(set(names))
    assert all(name.startswith("speech.read_aloud_") for name in names)
