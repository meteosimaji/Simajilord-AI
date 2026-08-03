"""Requester-private Discord controls for opaque managed file references."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

import discord

from simajilord.services.files import (
    WorkspaceFileAction,
    WorkspaceManagedFile,
    WorkspaceManagedFileCatalog,
)

log = logging.getLogger(__name__)

CatalogCallback = Callable[[discord.Interaction], Awaitable[WorkspaceManagedFileCatalog]]
FileActionCallback = Callable[
    [WorkspaceManagedFile, discord.Interaction],
    Awaitable[str],
]
HistoryCallback = Callable[
    [WorkspaceManagedFile, discord.Interaction],
    Awaitable[tuple[WorkspaceFileAction, ...]],
]
InteractionHandler = Callable[[discord.Interaction], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class FileManagerPublishReview:
    """Private exact-audience review with a host-only bound payload."""

    target_display_name: str
    new_reader_count: int
    expires_at_iso: str
    payload: object


PublishInspectCallback = Callable[
    [WorkspaceManagedFile, discord.Interaction],
    Awaitable[FileManagerPublishReview],
]
PublishExecuteCallback = Callable[
    [WorkspaceManagedFile, FileManagerPublishReview, discord.Interaction],
    Awaitable[str],
]


@dataclass(frozen=True, slots=True)
class FileManagerCallbacks:
    """Typed host actions shared by the agent endpoint and Discord UI."""

    catalog: CatalogCallback
    copy_to_task: FileActionCallback
    inspect_publish: PublishInspectCallback
    publish: PublishExecuteCallback
    send: FileActionCallback
    delete_or_revoke: FileActionCallback
    history: HistoryCallback


def file_manager_launcher_embed() -> discord.Embed:
    """Return the only public surface; it deliberately contains no file metadata."""

    return discord.Embed(
        title="Files",
        description=(
            "Open your private My / Task / Shared file manager. "
            "File names and metadata are visible only to you."
        ),
        colour=discord.Colour.blurple(),
    )


class FileManagerLauncherView(discord.ui.View):
    """Public-safe launcher that opens a requester-only ephemeral manager."""

    def __init__(
        self,
        *,
        requester_id: int,
        callbacks: FileManagerCallbacks,
        task_available: bool,
        timeout: float = 900,
    ) -> None:
        super().__init__(timeout=timeout)
        self.requester_id = requester_id
        self.callbacks = callbacks
        self.task_available = task_available

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "Only the requester can open this private file manager.",
            ephemeral=True,
        )
        return False

    async def on_error(
        self,
        interaction: discord.Interaction,
        error: Exception,
        item: discord.ui.Item[discord.ui.View],
    ) -> None:
        del item
        log.exception("Discord file manager launcher failed", exc_info=error)
        message = "The private file manager could not be opened. Ask the AI to retry."
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)

    @discord.ui.button(
        label="Open private file manager",
        style=discord.ButtonStyle.primary,
        custom_id="simajilord:files:open-private",
    )
    async def open_private(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[FileManagerLauncherView],
    ) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        catalog = await self.callbacks.catalog(interaction)
        view = FileManagerPrivateView(
            requester_id=self.requester_id,
            callbacks=self.callbacks,
            catalog=catalog,
            task_available=self.task_available,
            timeout=float(self.timeout or 900),
        )
        await interaction.edit_original_response(
            embed=view.render_embed(),
            view=view,
            allowed_mentions=discord.AllowedMentions.none(),
        )


class _FileManagerSelect(discord.ui.Select["FileManagerPrivateView"]):
    def __init__(
        self,
        *,
        handler: InteractionHandler,
        placeholder: str,
        options: list[discord.SelectOption],
        row: int,
    ) -> None:
        super().__init__(
            placeholder=placeholder,
            min_values=1,
            max_values=1,
            options=options,
            row=row,
        )
        self.handler = handler

    async def callback(self, interaction: discord.Interaction) -> None:
        await self.handler(interaction)


class _FileManagerButton(discord.ui.Button["FileManagerPrivateView"]):
    def __init__(
        self,
        *,
        handler: InteractionHandler,
        label: str,
        style: discord.ButtonStyle,
        disabled: bool = False,
        row: int,
    ) -> None:
        super().__init__(label=label, style=style, disabled=disabled, row=row)
        self.handler = handler

    async def callback(self, interaction: discord.Interaction) -> None:
        await self.handler(interaction)


class FileManagerPrivateView(discord.ui.View):
    """Ephemeral My / Task / Shared selector with typed file actions."""

    page_size = 20

    def __init__(
        self,
        *,
        requester_id: int,
        callbacks: FileManagerCallbacks,
        catalog: WorkspaceManagedFileCatalog,
        task_available: bool,
        timeout: float,
    ) -> None:
        super().__init__(timeout=timeout)
        self.requester_id = requester_id
        self.callbacks = callbacks
        self.catalog = catalog
        self.task_available = task_available
        self.section = self._initial_section()
        self.page = 0
        self.selected_ref: str | None = None
        self.status: str | None = None
        self._rebuild_items()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "Only the requester can use this private file manager.",
            ephemeral=True,
        )
        return False

    async def on_error(
        self,
        interaction: discord.Interaction,
        error: Exception,
        item: discord.ui.Item[discord.ui.View],
    ) -> None:
        del item
        log.exception("Discord private file manager action failed", exc_info=error)
        message = "The file action could not be completed. Refresh and try again."
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)

    def _initial_section(self) -> str:
        if self.catalog.task_count:
            return "task"
        if self.catalog.my_count:
            return "my"
        return "shared"

    def _section_files(self) -> tuple[WorkspaceManagedFile, ...]:
        return tuple(item for item in self.catalog.files if item.section == self.section)

    def _page_files(self) -> tuple[WorkspaceManagedFile, ...]:
        files = self._section_files()
        start = self.page * self.page_size
        return files[start : start + self.page_size]

    def _selected_file(self) -> WorkspaceManagedFile | None:
        if self.selected_ref is None:
            return None
        return next(
            (item for item in self.catalog.files if item.file_ref == self.selected_ref),
            None,
        )

    def _rebuild_items(self) -> None:
        self.clear_items()
        section_select = _FileManagerSelect(
            handler=self._select_section,
            placeholder="Choose My, Task, or Shared",
            options=[
                discord.SelectOption(
                    label=f"My ({self.catalog.my_count})",
                    value="my",
                    default=self.section == "my",
                ),
                discord.SelectOption(
                    label=f"Task ({self.catalog.task_count})",
                    value="task",
                    default=self.section == "task",
                ),
                discord.SelectOption(
                    label=f"Shared ({self.catalog.shared_count})",
                    value="shared",
                    default=self.section == "shared",
                ),
            ],
            row=0,
        )
        self.add_item(section_select)

        page_files = self._page_files()
        if page_files:
            file_select = _FileManagerSelect(
                handler=self._select_file,
                placeholder="Choose a file (no path required)",
                options=[
                    discord.SelectOption(
                        label=_bounded(item.filename, 100),
                        value=item.file_ref,
                        description=_bounded(
                            f"{_format_bytes(item.size_bytes)} · "
                            f"{item.sensitivity} · {item.share_state}",
                            100,
                        ),
                        default=item.file_ref == self.selected_ref,
                    )
                    for item in page_files
                ],
                row=1,
            )
            self.add_item(file_select)

        section_files = self._section_files()
        max_page = max(0, (len(section_files) - 1) // self.page_size)
        previous = _FileManagerButton(
            handler=self._previous_page,
            label="Previous",
            style=discord.ButtonStyle.secondary,
            disabled=self.page == 0,
            row=2,
        )
        self.add_item(previous)
        next_page = _FileManagerButton(
            handler=self._next_page,
            label="Next",
            style=discord.ButtonStyle.secondary,
            disabled=self.page >= max_page,
            row=2,
        )
        self.add_item(next_page)
        refresh = _FileManagerButton(
            handler=self._refresh_catalog,
            label="Refresh",
            style=discord.ButtonStyle.secondary,
            row=2,
        )
        self.add_item(refresh)

        selected = self._selected_file()
        disabled = selected is None
        copy_button = _FileManagerButton(
            handler=self._copy_to_task,
            label="Copy to task",
            style=discord.ButtonStyle.secondary,
            disabled=(
                disabled
                or not self.task_available
                or (selected is not None and selected.section == "shared")
            ),
            row=3,
        )
        self.add_item(copy_button)
        publish_button = _FileManagerButton(
            handler=self._inspect_publish,
            label="Publish copy…",
            style=discord.ButtonStyle.primary,
            disabled=disabled or (selected is not None and selected.section == "shared"),
            row=3,
        )
        self.add_item(publish_button)
        send_button = _FileManagerButton(
            handler=self._send,
            label="Send here",
            style=discord.ButtonStyle.primary,
            disabled=(
                disabled
                or (
                    selected is not None
                    and selected.section == "shared"
                    and selected.share_state != "active"
                )
            ),
            row=3,
        )
        self.add_item(send_button)
        delete_button = _FileManagerButton(
            handler=self._confirm_delete,
            label="Revoke sharing…"
            if selected is not None and selected.section == "shared"
            else "Delete…",
            style=discord.ButtonStyle.danger,
            disabled=disabled,
            row=4,
        )
        self.add_item(delete_button)
        history_button = _FileManagerButton(
            handler=self._history,
            label="History",
            style=discord.ButtonStyle.secondary,
            disabled=disabled,
            row=4,
        )
        self.add_item(history_button)

    def render_embed(self) -> discord.Embed:
        selected = self._selected_file()
        embed = discord.Embed(
            title="Private file manager",
            description=(
                f"My {self.catalog.my_count} · Task {self.catalog.task_count} · "
                f"Shared {self.catalog.shared_count}\n"
                "Only you can see this panel. Select a file without entering a path."
            ),
            colour=discord.Colour.blurple(),
        )
        if selected is None:
            embed.add_field(
                name=self.section.title(),
                value=(
                    "Select a file above."
                    if self._section_files()
                    else "No files in this section."
                ),
                inline=False,
            )
        else:
            embed.add_field(name="File", value=_bounded(selected.filename, 1_024))
            embed.add_field(name="Owner", value=selected.owner)
            embed.add_field(name="Origin", value=selected.origin)
            embed.add_field(name="Sensitivity", value=selected.sensitivity)
            embed.add_field(name="Size", value=_format_bytes(selected.size_bytes))
            embed.add_field(name="Created task", value=selected.created_task)
            share_value: str = selected.share_state
            if selected.target_display_name is not None:
                share_value += f" · {selected.target_display_name}"
            embed.add_field(name="Share state", value=_bounded(share_value, 1_024))
            embed.add_field(
                name="Updated",
                value=_display_time(selected.updated_at),
            )
        if self.status:
            embed.set_footer(text=_bounded(self.status, 2_048))
        return embed

    async def _select_section(self, interaction: discord.Interaction) -> None:
        select = interaction.data.get("values") if interaction.data else None
        value = select[0] if isinstance(select, list) and select else None
        if value not in {"my", "task", "shared"}:
            return
        self.section = value
        self.page = 0
        self.selected_ref = None
        self.status = None
        self._rebuild_items()
        await interaction.response.edit_message(embed=self.render_embed(), view=self)

    async def _select_file(self, interaction: discord.Interaction) -> None:
        select = interaction.data.get("values") if interaction.data else None
        value = select[0] if isinstance(select, list) and select else None
        if not isinstance(value, str) or not any(
            item.file_ref == value for item in self._page_files()
        ):
            return
        self.selected_ref = value
        self.status = None
        self._rebuild_items()
        await interaction.response.edit_message(embed=self.render_embed(), view=self)

    async def _previous_page(self, interaction: discord.Interaction) -> None:
        self.page = max(0, self.page - 1)
        self.selected_ref = None
        self._rebuild_items()
        await interaction.response.edit_message(embed=self.render_embed(), view=self)

    async def _next_page(self, interaction: discord.Interaction) -> None:
        max_page = max(0, (len(self._section_files()) - 1) // self.page_size)
        self.page = min(max_page, self.page + 1)
        self.selected_ref = None
        self._rebuild_items()
        await interaction.response.edit_message(embed=self.render_embed(), view=self)

    async def _refresh_catalog(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer()
        await self._reload(interaction, status="Refreshed current file state.")

    async def _copy_to_task(self, interaction: discord.Interaction) -> None:
        selected = self._selected_file()
        if selected is None:
            return
        await interaction.response.defer()
        status = await self.callbacks.copy_to_task(selected, interaction)
        await self._reload(interaction, status=status)

    async def _inspect_publish(self, interaction: discord.Interaction) -> None:
        selected = self._selected_file()
        if selected is None:
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        review = await self.callbacks.inspect_publish(selected, interaction)
        view = FileManagerPublishConfirmationView(
            requester_id=self.requester_id,
            callbacks=self.callbacks,
            file=selected,
            review=review,
            timeout=float(self.timeout or 900),
        )
        embed = discord.Embed(
            title="Confirm publication copy",
            description=(
                "This creates a revocable copy bound to the exact target audience. "
                "It does not send the file yet."
            ),
            colour=discord.Colour.orange(),
        )
        embed.add_field(name="File", value=_bounded(selected.filename, 1_024))
        embed.add_field(name="Target", value=_bounded(review.target_display_name, 1_024))
        embed.add_field(name="New readers", value=str(review.new_reader_count))
        embed.add_field(name="Expires", value=_display_time(review.expires_at_iso))
        await interaction.edit_original_response(
            embed=embed,
            view=view,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    async def _send(self, interaction: discord.Interaction) -> None:
        selected = self._selected_file()
        if selected is None:
            return
        await interaction.response.defer()
        self.status = await self.callbacks.send(selected, interaction)
        self._rebuild_items()
        await interaction.edit_original_response(embed=self.render_embed(), view=self)

    async def _confirm_delete(self, interaction: discord.Interaction) -> None:
        selected = self._selected_file()
        if selected is None:
            return
        action = (
            "Revoke this publication copy?"
            if selected.section == "shared"
            else "Permanently delete this private file?"
        )
        embed = discord.Embed(
            title="Confirm file action",
            description=action,
            colour=discord.Colour.red(),
        )
        embed.add_field(name="File", value=_bounded(selected.filename, 1_024))
        view = FileManagerDeleteConfirmationView(
            requester_id=self.requester_id,
            callback=self.callbacks.delete_or_revoke,
            file=selected,
            timeout=float(self.timeout or 900),
        )
        await interaction.response.send_message(
            embed=embed,
            view=view,
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    async def _history(self, interaction: discord.Interaction) -> None:
        selected = self._selected_file()
        if selected is None:
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        actions = await self.callbacks.history(selected, interaction)
        embed = discord.Embed(
            title=f"History · {_bounded(selected.filename, 200)}",
            colour=discord.Colour.blurple(),
        )
        if not actions:
            embed.description = "No copy, publish, send, delete, or revoke actions yet."
        else:
            embed.description = "\n".join(
                f"• {_display_time(item.occurred_at)} — {item.summary}"
                for item in actions[:20]
            )
        await interaction.edit_original_response(
            embed=embed,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    async def _reload(self, interaction: discord.Interaction, *, status: str) -> None:
        self.catalog = await self.callbacks.catalog(interaction)
        if self.selected_ref is not None and not any(
            item.file_ref == self.selected_ref for item in self.catalog.files
        ):
            self.selected_ref = None
        self.status = status
        section_files = self._section_files()
        self.page = min(
            self.page,
            max(0, (len(section_files) - 1) // self.page_size),
        )
        self._rebuild_items()
        await interaction.edit_original_response(embed=self.render_embed(), view=self)


class FileManagerPublishConfirmationView(discord.ui.View):
    def __init__(
        self,
        *,
        requester_id: int,
        callbacks: FileManagerCallbacks,
        file: WorkspaceManagedFile,
        review: FileManagerPublishReview,
        timeout: float,
    ) -> None:
        super().__init__(timeout=timeout)
        self.requester_id = requester_id
        self.callbacks = callbacks
        self.file = file
        self.review = review

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id == self.requester_id

    @discord.ui.button(label="Publish exact copy", style=discord.ButtonStyle.danger)
    async def confirm(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[FileManagerPublishConfirmationView],
    ) -> None:
        await interaction.response.defer()
        status = await self.callbacks.publish(self.file, self.review, interaction)
        self.stop()
        await interaction.edit_original_response(content=status, embed=None, view=None)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[FileManagerPublishConfirmationView],
    ) -> None:
        self.stop()
        await interaction.response.edit_message(
            content="Publication cancelled.",
            embed=None,
            view=None,
        )


class FileManagerDeleteConfirmationView(discord.ui.View):
    def __init__(
        self,
        *,
        requester_id: int,
        callback: FileActionCallback,
        file: WorkspaceManagedFile,
        timeout: float,
    ) -> None:
        super().__init__(timeout=timeout)
        self.requester_id = requester_id
        self.callback = callback
        self.file = file

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id == self.requester_id

    @discord.ui.button(label="Confirm", style=discord.ButtonStyle.danger)
    async def confirm(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[FileManagerDeleteConfirmationView],
    ) -> None:
        await interaction.response.defer()
        status = await self.callback(self.file, interaction)
        self.stop()
        await interaction.edit_original_response(content=status, embed=None, view=None)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[FileManagerDeleteConfirmationView],
    ) -> None:
        self.stop()
        await interaction.response.edit_message(
            content="No file action was taken.",
            embed=None,
            view=None,
        )


def _bounded(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    return f"{value[: max(1, maximum - 1)]}…"


def _format_bytes(value: int) -> str:
    if value < 1_024:
        return f"{value} B"
    if value < 1_024**2:
        return f"{value / 1_024:.1f} KiB"
    return f"{value / (1_024**2):.1f} MiB"


def _display_time(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "Unknown"
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return "Unknown"
    return discord.utils.format_dt(parsed, style="R")
