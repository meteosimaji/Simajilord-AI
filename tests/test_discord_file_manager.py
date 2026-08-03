from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

import discord
import pytest

from simajilord.integrations.discord.file_manager import (
    FileManagerCallbacks,
    FileManagerLauncherView,
    FileManagerPrivateView,
    FileManagerPublishReview,
    file_manager_launcher_embed,
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
            expires_at_iso=datetime.now(UTC).isoformat(),
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

    return FileManagerCallbacks(
        catalog=load,
        copy_to_task=action,
        inspect_publish=inspect,
        publish=publish,
        send=action,
        delete_or_revoke=action,
        history=history,
    )


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
    labels = [
        child.label
        for child in launcher.children
        if isinstance(child, discord.ui.Button)
    ]
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
