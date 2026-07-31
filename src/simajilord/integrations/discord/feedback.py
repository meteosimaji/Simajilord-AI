"""Private Discord Modal entry point for the local feedback inbox."""

from __future__ import annotations

from typing import cast

import discord
from discord import app_commands
from discord.ext import commands

from simajilord.agent import AGENT_FEEDBACK_GRANT
from simajilord.capabilities.feedback import (
    FeedbackCreateRequest,
    FeedbackCreateResponse,
)
from simajilord.core import InvocationContext
from simajilord.core.errors import UserError
from simajilord.runtime import SimajilordRuntime

from .presenter import EmbedTone, command_embed


class FeedbackModal(discord.ui.Modal, title="Send feedback"):
    """A private, requester-bound form with no user-controlled triage fields."""

    title_input: discord.ui.TextInput[FeedbackModal] = discord.ui.TextInput(
        label="Title (optional)",
        placeholder="A short summary",
        required=False,
        max_length=160,
    )
    details_input: discord.ui.TextInput[FeedbackModal] = discord.ui.TextInput(
        label="What happened or what would help?",
        placeholder="Describe the problem, idea, or experience.",
        style=discord.TextStyle.paragraph,
        required=True,
        max_length=4_000,
    )
    expected_input: discord.ui.TextInput[FeedbackModal] = discord.ui.TextInput(
        label="Expected behaviour (optional)",
        placeholder="What did you expect instead?",
        style=discord.TextStyle.paragraph,
        required=False,
        max_length=2_000,
    )

    def __init__(
        self,
        runtime: SimajilordRuntime,
        *,
        requester_id: int,
    ) -> None:
        super().__init__(timeout=10 * 60)
        self.runtime = runtime
        self.requester_id = requester_id

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            raise UserError("feedback.requester_mismatch")
        await interaction.response.defer(ephemeral=True, thinking=True)
        response = cast(
            FeedbackCreateResponse,
            await self.runtime.registry.invoke(
                "feedback.create",
                FeedbackCreateRequest(
                    title=str(self.title_input),
                    details=str(self.details_input),
                    expected=str(self.expected_input),
                ),
                InvocationContext(
                    actor_id=str(interaction.user.id),
                    workspace_id=(
                        str(interaction.guild_id)
                        if interaction.guild_id is not None
                        else None
                    ),
                    transport="discord",
                    request_id=str(interaction.id),
                    origin_resource_id=(
                        str(interaction.channel_id)
                        if interaction.channel_id is not None
                        else None
                    ),
                    grants=frozenset({AGENT_FEEDBACK_GRANT}),
                    approvals=frozenset({"feedback.create"}),
                ),
            ),
        )
        await interaction.edit_original_response(
            embed=command_embed(
                "Feedback saved" if response.created else "Feedback already saved",
                description=(
                    "The report is in the local administrator inbox.\n"
                    f"Report ID: `{response.report_id}`"
                ),
                tone=EmbedTone.SUCCESS,
            ),
            view=None,
        )

    async def on_error(  # type: ignore[override]
        self,
        interaction: discord.Interaction,
        error: Exception,
    ) -> None:
        # Imported lazily so cogs.py can register this module without a cycle.
        from .cogs import handle_interaction_error

        await handle_interaction_error(interaction, error)


class FeedbackCog(commands.Cog):
    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    @app_commands.command(
        name="feedback",
        description="Open a private form for a bug report, idea, or other feedback.",
    )
    async def feedback(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(
            FeedbackModal(
                self.runtime,
                requester_id=interaction.user.id,
            )
        )


__all__ = ["FeedbackCog", "FeedbackModal"]
