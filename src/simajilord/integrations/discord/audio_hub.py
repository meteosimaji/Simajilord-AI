"""Small, contextual audio entry point; configuration stays off the start path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import discord

from simajilord.capabilities.speech import SpeechSpeakRequest
from simajilord.core.errors import UserError

from .audio_navigation import AudioNavigateRequest, AudioNavigateResponse
from .cogs import (
    _MUSIC_DASHBOARD_ATTRIBUTE,
    MusicAddModal,
    MusicDashboardManager,
    SafeView,
    _send_read_aloud_setup,
    invocation_context,
    send_error,
)
from .presenter import command_embed

if TYPE_CHECKING:
    from simajilord.runtime import SimajilordRuntime


def audio_hub_embed(
    runtime: SimajilordRuntime, workspace: str, destination: str | None
) -> discord.Embed:
    session = runtime.audio.find(workspace)
    route = runtime.read_aloud.resume_route(workspace, destination) if destination else None
    route = route or runtime.read_aloud.get(workspace)
    lines = [
        "Join a voice channel, then choose what to start."
        if destination is None
        else f"Your voice channel: <#{destination}>"
    ]
    if session is not None and session.output.connected:
        lines.append(
            f"BOT: <#{session.destination_id}> · "
            + ("Read aloud only" if session.speech_only else "Music & read aloud")
        )
    else:
        lines.append("The bot is idle. Start with your saved settings.")
    if route is not None:
        lines.append(
            "Read messages from: " + ", ".join(f"<#{value}>" for value in route.text_channel_ids)
        )
    else:
        lines.append("First start reads messages from this conversation.")
    if runtime.audio.follow_actors.get(workspace):
        lines.append(
            f"Following: <@{runtime.audio.follow_actors[workspace]}> · this connection only"
        )
    lines.extend(
        (
            "",
            "**Read aloud here** starts reading and pauses music.",
            "**Resume music** also plays your saved queue.",
            "**Move here** keeps your current audio mode.",
            "**Follow me** moves audio with you when the old channel is empty.",
        )
    )
    return command_embed("Music & read aloud", description="\n".join(lines))


class AudioHubView(SafeView):
    def __init__(
        self,
        runtime: SimajilordRuntime,
        *,
        requester_id: int,
        workspace: str,
        destination: str | None,
        source_id: str | None,
    ) -> None:
        super().__init__(timeout=300)
        self.runtime = runtime
        self.requester_id = requester_id
        self.workspace = workspace
        self.destination = destination
        self.source_id = source_id
        session = runtime.audio.find(workspace)
        self.expected = session.destination_id if session else None
        self.move.disabled = destination is None or self.expected in {None, destination}
        self.speech.disabled = destination is None
        self.both.disabled = destination is None
        self.preview.disabled = destination is None
        self.follow.disabled = (
            destination is None
            or session is None
            or not session.output.connected
            or self.expected != destination
        )
        if runtime.audio.follow_actors.get(workspace) == str(requester_id):
            self.follow.label = "Stop following"
        self.add_item(AudioSettingsSelect(runtime))
        if destination is None:
            self.add_item(AudioHubOpenButton(runtime, label="Refresh after joining"))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "Open your own /audio panel to use these controls.", ephemeral=True
        )
        return False

    async def run(
        self,
        interaction: discord.Interaction,
        mode: Literal["preserve", "speech", "both", "follow", "unfollow"],
        *,
        confirm: bool = False,
    ) -> None:
        if self.destination is None:
            await interaction.response.send_message("Join a voice channel first.", ephemeral=True)
            return
        try:
            await interaction.response.defer()
            result = cast(
                AudioNavigateResponse,
                await self.runtime.registry.invoke(
                    "discord.navigate_audio",
                    AudioNavigateRequest(
                        self.destination,
                        self.expected,
                        mode,
                        self.source_id,
                        confirm,
                    ),
                    invocation_context(interaction),
                ),
            )
            self.expected = result.destination_id
            if mode in {"preserve", "speech", "both"}:
                dashboard = getattr(interaction.client, _MUSIC_DASHBOARD_ATTRIBUTE, None)
                if isinstance(dashboard, MusicDashboardManager):
                    dashboard.bind(interaction.guild_id, int(result.destination_id))
                    await dashboard.publish(self.runtime.audio.require(self.workspace), force=True)
            await interaction.edit_original_response(
                embed=audio_hub_embed(self.runtime, self.workspace, self.destination),
                view=AudioHubView(
                    self.runtime,
                    requester_id=self.requester_id,
                    workspace=self.workspace,
                    destination=self.destination,
                    source_id=self.source_id,
                ),
            )
        except UserError as exc:
            if exc.code == "audio.move_confirmation_required":
                await interaction.edit_original_response(
                    embed=command_embed(
                        "People are still in the other voice channel",
                        description=(
                            "Move the music queue and read aloud to your voice channel.\n"
                            "Audio will stop in the other channel. "
                            "You need Move Members or Manage Server permission."
                        ),
                    ),
                    view=AudioMoveConfirmation(self, mode),
                )
            else:
                await send_error(interaction, exc)
        except Exception as exc:
            await send_error(interaction, exc)

    @discord.ui.button(label="Read aloud here", style=discord.ButtonStyle.primary, row=0)
    async def speech(
        self, interaction: discord.Interaction, _: discord.ui.Button[AudioHubView]
    ) -> None:
        await self.run(interaction, "speech")

    @discord.ui.button(label="Resume music", style=discord.ButtonStyle.success, row=0)
    async def both(
        self, interaction: discord.Interaction, _: discord.ui.Button[AudioHubView]
    ) -> None:
        await self.run(interaction, "both")

    @discord.ui.button(label="Move here", row=0)
    async def move(
        self, interaction: discord.Interaction, _: discord.ui.Button[AudioHubView]
    ) -> None:
        await self.run(interaction, "preserve")

    @discord.ui.button(label="Follow me", row=1)
    async def follow(
        self, interaction: discord.Interaction, _: discord.ui.Button[AudioHubView]
    ) -> None:
        mode = (
            "unfollow"
            if self.runtime.audio.follow_actors.get(self.workspace) == str(self.requester_id)
            else "follow"
        )
        await self.run(interaction, "unfollow" if mode == "unfollow" else "follow")

    @discord.ui.button(label="Add music", row=1)
    async def add_music(
        self, interaction: discord.Interaction, _: discord.ui.Button[AudioHubView]
    ) -> None:
        dashboard = getattr(interaction.client, _MUSIC_DASHBOARD_ATTRIBUTE, None)
        await interaction.response.send_modal(
            MusicAddModal(
                self.runtime, dashboard if isinstance(dashboard, MusicDashboardManager) else None
            )
        )

    @discord.ui.button(label="Test my voice", row=1)
    async def preview(
        self, interaction: discord.Interaction, _: discord.ui.Button[AudioHubView]
    ) -> None:
        try:
            await interaction.response.defer()
            tuning = self.runtime.read_aloud.voice_tuning_for(
                workspace_id=self.workspace, user_id=str(self.requester_id)
            )
            await self.runtime.registry.invoke(
                "discord.speak",
                SpeechSpeakRequest(
                    text="こんにちは。この声と速さで読み上げます。",
                    title="Voice preview",
                    voice_preset=self.runtime.read_aloud.voice_preset_for(
                        workspace_id=self.workspace, user_id=str(self.requester_id)
                    ).value,
                    speed_scale=tuning.speed_scale,
                    pitch_scale=tuning.pitch_scale,
                ),
                invocation_context(interaction),
            )
            await interaction.followup.send("Voice preview added to the queue.", ephemeral=True)
        except Exception as exc:
            await send_error(interaction, exc)


class AudioMoveConfirmation(SafeView):
    def __init__(
        self, hub: AudioHubView, mode: Literal["preserve", "speech", "both", "follow", "unfollow"]
    ) -> None:
        super().__init__(timeout=60)
        self.hub = hub
        self.mode = mode

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return await self.hub.interaction_check(interaction)

    @discord.ui.button(label="Move audio here", style=discord.ButtonStyle.primary)
    async def confirm(
        self, interaction: discord.Interaction, _: discord.ui.Button[AudioMoveConfirmation]
    ) -> None:
        await self.hub.run(interaction, self.mode, confirm=True)
        self.stop()

    @discord.ui.button(label="Cancel")
    async def cancel(
        self, interaction: discord.Interaction, _: discord.ui.Button[AudioMoveConfirmation]
    ) -> None:
        await interaction.response.edit_message(
            embed=audio_hub_embed(self.hub.runtime, self.hub.workspace, self.hub.destination),
            view=self.hub,
        )
        self.stop()


class AudioHubOpenButton(discord.ui.Button[discord.ui.View]):
    def __init__(self, runtime: SimajilordRuntime, *, label: str = "Back to audio") -> None:
        super().__init__(label=label, row=4)
        self.runtime = runtime

    async def callback(self, interaction: discord.Interaction) -> None:
        await send_audio_hub(interaction, self.runtime)


class AudioSettingsSelect(discord.ui.Select[discord.ui.View]):
    def __init__(self, runtime: SimajilordRuntime) -> None:
        super().__init__(
            placeholder="Settings…",
            row=3,
            options=[
                discord.SelectOption(label="My voice & speed", value="personal"),
                discord.SelectOption(label="Reading channels", value="sources"),
                discord.SelectOption(label="Server reading settings", value="shared"),
            ],
        )
        self.runtime = runtime

    async def callback(self, interaction: discord.Interaction) -> None:
        await _send_read_aloud_setup(interaction, self.runtime, section=self.values[0])


async def send_audio_hub(interaction: discord.Interaction, runtime: SimajilordRuntime) -> None:
    if interaction.guild_id is None:
        await interaction.response.send_message("Open this panel in a server.", ephemeral=True)
        return
    member = interaction.user
    destination = (
        str(member.voice.channel.id)
        if isinstance(member, discord.Member) and member.voice and member.voice.channel
        else None
    )
    workspace = str(interaction.guild_id)
    await interaction.response.send_message(
        embed=audio_hub_embed(runtime, workspace, destination),
        view=AudioHubView(
            runtime,
            requester_id=member.id,
            workspace=workspace,
            destination=destination,
            source_id=str(interaction.channel_id) if interaction.channel_id else None,
        ),
        ephemeral=True,
    )
