from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from simajilord.core.errors import UserError
from simajilord.integrations.discord.file_manager import (
    FileManagerCallbacks,
    FileManagerDeleteConfirmationView,
    FileManagerLauncherView,
    FileManagerPrivateView,
    FileManagerPublishConfirmationView,
    FileManagerPublishReview,
    _confirmation_timeout,
    file_manager_launcher_embed,
    render_file_action_history,
)
from simajilord.services.files import (
    WorkspaceFileAction,
    WorkspaceManagedFile,
    WorkspaceManagedFileCatalog,
)


def _managed_file(
    *,
    section: Literal["my", "task", "shared"] = "task",
) -> WorkspaceManagedFile:
    now = datetime.now(UTC).isoformat()
    return WorkspaceManagedFile(
        file_ref="fil_" + "a" * 32,
        section=section,
        filename="private-report.txt",
        kind="text",
        size_bytes=1_234,
        sha256="b" * 64,
        owner="You",
        origin="Discord message",
        sensitivity="restricted",
        created_task="Current task",
        share_state="private",
        target_display_name=None,
        revision=1,
        created_at=now,
        updated_at=now,
    )


def _callbacks(catalog: WorkspaceManagedFileCatalog) -> FileManagerCallbacks:
    async def load(_: discord.Interaction) -> WorkspaceManagedFileCatalog:
        return catalog

    async def action(
        _: WorkspaceManagedFile,
        __: discord.Interaction,
    ) -> str:
        return "done"

    async def inspect(
        _: WorkspaceManagedFile,
        __: discord.Interaction,
    ) -> FileManagerPublishReview:
        return FileManagerPublishReview(
            target_display_name="#review",
            new_reader_count=2,
            copy_expires_at_iso=datetime.now(UTC).isoformat(),
            confirmation_expires_at_iso=datetime.now(UTC).isoformat(),
            payload=object(),
        )

    async def publish(
        _: WorkspaceManagedFile,
        __: FileManagerPublishReview,
        ___: discord.Interaction,
    ) -> str:
        return "published"

    async def history(
        _: WorkspaceManagedFile,
        __: discord.Interaction,
    ) -> tuple[WorkspaceFileAction, ...]:
        return ()

    async def recent_activity(
        _: discord.Interaction,
    ) -> tuple[WorkspaceFileAction, ...]:
        return ()

    return FileManagerCallbacks(
        catalog=load,
        copy_to_task=action,
        inspect_publish=inspect,
        publish=publish,
        send=action,
        delete_or_revoke=action,
        history=history,
        recent_activity=recent_activity,
    )


def _interaction(user_id: int = 7) -> Mock:
    interaction = Mock(spec=discord.Interaction)
    interaction.id = 100
    interaction.user = SimpleNamespace(id=user_id)
    interaction.guild_id = 10
    interaction.channel_id = 20
    interaction.response = Mock()
    interaction.response.defer = AsyncMock()
    interaction.response.edit_message = AsyncMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.is_done = Mock(return_value=False)
    interaction.followup = Mock()
    interaction.followup.send = AsyncMock()
    interaction.edit_original_response = AsyncMock()
    return interaction


@pytest.mark.asyncio
async def test_file_manager_public_launcher_is_metadata_free() -> None:
    item = _managed_file()
    catalog = WorkspaceManagedFileCatalog(
        files=(item,),
        my_count=0,
        task_count=1,
        shared_count=0,
    )
    embed = file_manager_launcher_embed()
    launcher = FileManagerLauncherView(
        requester_id=7,
        callbacks=_callbacks(catalog),
        task_available=True,
    )
    rendered = f"{embed.title}\n{embed.description}"

    assert "private-report" not in rendered
    assert item.file_ref not in rendered
    assert item.sha256 not in rendered
    assert "actor" not in rendered.casefold()
    assert "workspace" not in rendered.casefold()
    labels = [child.label for child in launcher.children if isinstance(child, discord.ui.Button)]
    assert labels == ["Open private file manager"]


@pytest.mark.asyncio
async def test_private_file_manager_uses_opaque_select_and_required_metadata() -> None:
    item = _managed_file()
    catalog = WorkspaceManagedFileCatalog(
        files=(item,),
        my_count=0,
        task_count=1,
        shared_count=0,
    )
    view = FileManagerPrivateView(
        requester_id=7,
        callbacks=_callbacks(catalog),
        catalog=catalog,
        task_available=True,
        timeout=60,
    )
    view.selected_ref = item.file_ref
    view._rebuild_items()
    embed = view.render_embed()
    field_names = {field.name for field in embed.fields}
    file_select = next(
        child
        for child in view.children
        if isinstance(child, discord.ui.Select)
        and child.placeholder == "Choose a file (no path required)"
    )

    assert {
        "File",
        "Owner",
        "Origin",
        "Sensitivity",
        "Size",
        "Created task",
        "Share state",
    } <= field_names
    assert [option.value for option in file_select.options] == [item.file_ref]
    assert all("/" not in option.value for option in file_select.options)
    assert item.filename in {field.value for field in embed.fields}
    assert item.file_ref not in str(embed.to_dict())


@pytest.mark.asyncio
@pytest.mark.parametrize("label", ("Send here", "Copy to task"))
async def test_private_mutating_action_claims_before_first_await(label: str) -> None:
    item = _managed_file()
    catalog = WorkspaceManagedFileCatalog(files=(item,), my_count=0, task_count=1, shared_count=0)
    entered = asyncio.Event()
    release = asyncio.Event()
    call_count = 0

    async def action(
        _: WorkspaceManagedFile,
        __: discord.Interaction,
    ) -> str:
        nonlocal call_count
        call_count += 1
        entered.set()
        await release.wait()
        return "done"

    base = _callbacks(catalog)
    callbacks = FileManagerCallbacks(
        catalog=base.catalog,
        copy_to_task=action,
        inspect_publish=base.inspect_publish,
        publish=base.publish,
        send=action,
        delete_or_revoke=base.delete_or_revoke,
        history=base.history,
        recent_activity=base.recent_activity,
    )
    view = FileManagerPrivateView(
        requester_id=7,
        callbacks=callbacks,
        catalog=catalog,
        task_available=True,
        timeout=60,
    )
    view.selected_ref = item.file_ref
    view._rebuild_items()
    button = next(
        child
        for child in view.children
        if isinstance(child, discord.ui.Button) and child.label == label
    )
    first = _interaction()
    second = _interaction()

    running = asyncio.create_task(button.callback(first))
    await entered.wait()
    await button.callback(second)
    release.set()
    await running

    assert call_count == 1
    second.response.send_message.assert_awaited_once()
    assert second.response.send_message.await_args.kwargs["ephemeral"] is True


@pytest.mark.asyncio
async def test_publish_and_delete_confirmations_are_one_way_claims() -> None:
    item = _managed_file()
    catalog = WorkspaceManagedFileCatalog(files=(item,), my_count=0, task_count=1, shared_count=0)
    entered = asyncio.Event()
    release = asyncio.Event()
    publish_calls = 0
    delete_calls = 0

    async def publish(
        _: WorkspaceManagedFile,
        __: FileManagerPublishReview,
        ___: discord.Interaction,
    ) -> str:
        nonlocal publish_calls
        publish_calls += 1
        entered.set()
        await release.wait()
        return "published"

    async def delete(
        _: WorkspaceManagedFile,
        __: discord.Interaction,
    ) -> str:
        nonlocal delete_calls
        delete_calls += 1
        return "deleted"

    base = _callbacks(catalog)
    callbacks = FileManagerCallbacks(
        catalog=base.catalog,
        copy_to_task=base.copy_to_task,
        inspect_publish=base.inspect_publish,
        publish=publish,
        send=base.send,
        delete_or_revoke=delete,
        history=base.history,
        recent_activity=base.recent_activity,
    )
    origin = _interaction()
    review = FileManagerPublishReview(
        target_display_name="#review",
        new_reader_count=1,
        copy_expires_at_iso=datetime.now(UTC).isoformat(),
        confirmation_expires_at_iso=datetime.now(UTC).isoformat(),
        payload=object(),
    )
    publish_view = FileManagerPublishConfirmationView(
        requester_id=7,
        callbacks=callbacks,
        file=item,
        review=review,
        timeout=60,
        origin_interaction=origin,
    )
    publish_button = next(
        child
        for child in publish_view.children
        if isinstance(child, discord.ui.Button) and child.label == "Publish exact copy"
    )
    first = _interaction()
    second = _interaction()
    running = asyncio.create_task(publish_button.callback(first))
    await entered.wait()
    await publish_button.callback(second)
    assert all(getattr(child, "disabled", False) for child in publish_view.children)
    release.set()
    await running
    assert publish_calls == 1
    second.response.send_message.assert_awaited_once()

    delete_view = FileManagerDeleteConfirmationView(
        requester_id=7,
        callback=delete,
        file=item,
        timeout=60,
        origin_interaction=origin,
    )
    delete_button = next(
        child
        for child in delete_view.children
        if isinstance(child, discord.ui.Button) and child.label == "Confirm"
    )
    await delete_button.callback(_interaction())
    await delete_button.callback(_interaction())
    assert delete_calls == 1


@pytest.mark.asyncio
async def test_publish_review_distinguishes_copy_and_confirmation_expiry() -> None:
    item = _managed_file()
    catalog = WorkspaceManagedFileCatalog(files=(item,), my_count=0, task_count=1, shared_count=0)
    now = datetime.now(UTC)

    async def inspect(
        _: WorkspaceManagedFile,
        __: discord.Interaction,
    ) -> FileManagerPublishReview:
        return FileManagerPublishReview(
            target_display_name="#review",
            new_reader_count=2,
            copy_expires_at_iso=(now.replace(microsecond=0)).isoformat(),
            confirmation_expires_at_iso=(now.replace(microsecond=1)).isoformat(),
            payload=object(),
        )

    base = _callbacks(catalog)
    callbacks = FileManagerCallbacks(
        catalog=base.catalog,
        copy_to_task=base.copy_to_task,
        inspect_publish=inspect,
        publish=base.publish,
        send=base.send,
        delete_or_revoke=base.delete_or_revoke,
        history=base.history,
        recent_activity=base.recent_activity,
    )
    view = FileManagerPrivateView(
        requester_id=7,
        callbacks=callbacks,
        catalog=catalog,
        task_available=True,
        timeout=900,
    )
    view.selected_ref = item.file_ref
    interaction = _interaction()
    await view._inspect_publish(interaction)
    embed = interaction.edit_original_response.await_args.kwargs["embed"]
    confirmation_view = interaction.edit_original_response.await_args.kwargs["view"]

    assert {field.name for field in embed.fields} >= {"Copy expires", "Confirm by"}
    assert isinstance(confirmation_view, FileManagerPublishConfirmationView)
    assert 0 < float(confirmation_view.timeout or 0) <= 1.0


@pytest.mark.parametrize(
    ("code", "fragment"),
    (
        ("files.publication_confirmation_expired", "expired"),
        ("files.publication_audience_changed", "audience changed"),
        ("files.hash_conflict", "selected file changed"),
        ("files.publication_revision_conflict", "publication changed"),
    ),
)
@pytest.mark.asyncio
async def test_confirmation_error_is_visible_and_removes_controls(
    code: str,
    fragment: str,
) -> None:
    item = _managed_file()
    catalog = WorkspaceManagedFileCatalog(files=(item,), my_count=0, task_count=1, shared_count=0)
    origin = _interaction()
    view = FileManagerDeleteConfirmationView(
        requester_id=7,
        callback=_callbacks(catalog).delete_or_revoke,
        file=item,
        timeout=60,
        origin_interaction=origin,
    )
    interaction = _interaction()
    button = next(child for child in view.children if isinstance(child, discord.ui.Button))

    await view.on_error(interaction, UserError(code), button)

    kwargs = interaction.edit_original_response.await_args.kwargs
    assert fragment in kwargs["content"].casefold()
    assert kwargs["view"] is None
    assert view.is_finished()


@pytest.mark.asyncio
async def test_confirmation_timeout_removes_stale_controls() -> None:
    item = _managed_file()
    catalog = WorkspaceManagedFileCatalog(files=(item,), my_count=0, task_count=1, shared_count=0)
    origin = _interaction()
    view = FileManagerDeleteConfirmationView(
        requester_id=7,
        callback=_callbacks(catalog).delete_or_revoke,
        file=item,
        timeout=60,
        origin_interaction=origin,
    )
    await view.on_timeout()
    kwargs = origin.edit_original_response.await_args.kwargs
    assert "expired" in kwargs["content"].casefold()
    assert kwargs["view"] is None


def test_confirmation_timeout_never_exceeds_token_lifetime() -> None:
    soon = datetime.now(UTC).timestamp() + 12
    expiry = datetime.fromtimestamp(soon, UTC).isoformat()
    assert 0 < _confirmation_timeout(expiry, maximum=900) <= 12
    assert _confirmation_timeout("invalid", maximum=900) == 1.0


def _action(index: int, summary: str) -> WorkspaceFileAction:
    return WorkspaceFileAction(
        action_id=f"fact_{index:032x}",
        file_ref="fil_" + "a" * 32,
        action="sent",
        summary=summary,
        occurred_at=datetime.now(UTC).isoformat(),
        display_filename=f"report-{index}.txt",
    )


def test_history_renderer_is_line_aware_and_bounded() -> None:
    actions = tuple(_action(index, "x" * 200) for index in range(20))
    rendered = render_file_action_history(actions)
    lines = rendered.splitlines()

    assert len(rendered) <= 4_096
    assert lines[0].endswith("x" * 200)
    assert lines[-1].startswith("… ")
    omitted = int(lines[-1].split()[1])
    assert omitted == 20 - (len(lines) - 1)
    assert render_file_action_history(()) == ""
    short = render_file_action_history((_action(1, "Sent once."),))
    assert "Sent once." in short and "omitted" not in short


def test_history_renderer_omission_marker_boundaries() -> None:
    actions = (_action(1, "first"), _action(2, "second"))
    first_line = render_file_action_history(actions[:1])
    with_marker = render_file_action_history(
        actions,
        maximum=len(first_line) + len("\n… 1 more actions omitted"),
    )
    assert with_marker.endswith("… 1 more actions omitted")
    marker_cannot_fit = render_file_action_history(actions, maximum=1)
    assert marker_cannot_fit == ""
