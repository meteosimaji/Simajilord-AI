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
        "VCに参加して、使いたい音声を選んでください。"
        if destination is None
        else f"あなたのVC:<#{destination}>"
    ]
    if session is not None and session.output.connected:
        lines.append(
            f"BOT:<#{session.destination_id}> · "
            + ("読み上げのみ" if session.speech_only else "音楽・読み上げ")
        )
    else:
        lines.append("BOTは待機中です。保存した設定で再開できます。")
    if route is not None:
        lines.append("読み上げ元:" + "・".join(f"<#{value}>" for value in route.text_channel_ids))
    else:
        lines.append("初めての読み上げは、この会話チャンネルから始めます。")
    if runtime.audio.follow_actors.get(workspace):
        lines.append(f"VC追従:<@{runtime.audio.follow_actors[workspace]}> · 今回の接続中のみ")
    return command_embed("音楽・読み上げ", description="\n".join(lines))


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
            self.follow.label = "追従を解除"
        self.add_item(AudioSettingsSelect(runtime))
        if destination is None:
            self.add_item(AudioHubOpenButton(runtime, label="VC参加後に更新"))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "自分の /audio パネルを開いてください。", ephemeral=True
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
            await interaction.response.send_message("先にVCへ参加してください。", ephemeral=True)
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
                        "元のVCにまだ人がいます",
                        description=(
                            "音楽のキューと読み上げ先をこちらへ移動します。\n"
                            "移動権限を持つ人だけが実行できます。元のVCでは音声が止まります。"
                        ),
                    ),
                    view=AudioMoveConfirmation(self, mode),
                )
            else:
                await send_error(interaction, exc)
        except Exception as exc:
            await send_error(interaction, exc)

    @discord.ui.button(label="ここで読み上げ", style=discord.ButtonStyle.primary, row=0)
    async def speech(
        self, interaction: discord.Interaction, _: discord.ui.Button[AudioHubView]
    ) -> None:
        await self.run(interaction, "speech")

    @discord.ui.button(label="音楽も再開", style=discord.ButtonStyle.success, row=0)
    async def both(
        self, interaction: discord.Interaction, _: discord.ui.Button[AudioHubView]
    ) -> None:
        await self.run(interaction, "both")

    @discord.ui.button(label="このVCへ移動", row=0)
    async def move(
        self, interaction: discord.Interaction, _: discord.ui.Button[AudioHubView]
    ) -> None:
        await self.run(interaction, "preserve")

    @discord.ui.button(label="自分のVC移動に追従", row=1)
    async def follow(
        self, interaction: discord.Interaction, _: discord.ui.Button[AudioHubView]
    ) -> None:
        mode = (
            "unfollow"
            if self.runtime.audio.follow_actors.get(self.workspace) == str(self.requester_id)
            else "follow"
        )
        await self.run(interaction, "unfollow" if mode == "unfollow" else "follow")

    @discord.ui.button(label="曲を追加", row=1)
    async def add_music(
        self, interaction: discord.Interaction, _: discord.ui.Button[AudioHubView]
    ) -> None:
        dashboard = getattr(interaction.client, _MUSIC_DASHBOARD_ATTRIBUTE, None)
        await interaction.response.send_modal(
            MusicAddModal(
                self.runtime, dashboard if isinstance(dashboard, MusicDashboardManager) else None
            )
        )

    @discord.ui.button(label="声を試す", row=1)
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
                    title="読み上げの試聴",
                    voice_preset=self.runtime.read_aloud.voice_preset_for(
                        workspace_id=self.workspace, user_id=str(self.requester_id)
                    ).value,
                    speed_scale=tuning.speed_scale,
                    pitch_scale=tuning.pitch_scale,
                ),
                invocation_context(interaction),
            )
            await interaction.followup.send("試聴を追加しました。", ephemeral=True)
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

    @discord.ui.button(label="このVCへ移動する", style=discord.ButtonStyle.primary)
    async def confirm(
        self, interaction: discord.Interaction, _: discord.ui.Button[AudioMoveConfirmation]
    ) -> None:
        await self.hub.run(interaction, self.mode, confirm=True)
        self.stop()

    @discord.ui.button(label="移動しない")
    async def cancel(
        self, interaction: discord.Interaction, _: discord.ui.Button[AudioMoveConfirmation]
    ) -> None:
        await interaction.response.edit_message(
            embed=audio_hub_embed(self.hub.runtime, self.hub.workspace, self.hub.destination),
            view=self.hub,
        )
        self.stop()


class AudioHubOpenButton(discord.ui.Button[discord.ui.View]):
    def __init__(self, runtime: SimajilordRuntime, *, label: str = "音声パネルへ") -> None:
        super().__init__(label=label, row=2)
        self.runtime = runtime

    async def callback(self, interaction: discord.Interaction) -> None:
        await send_audio_hub(interaction, self.runtime)


class AudioSettingsSelect(discord.ui.Select[discord.ui.View]):
    def __init__(self, runtime: SimajilordRuntime) -> None:
        super().__init__(
            placeholder="設定を変更…",
            row=3,
            options=[
                discord.SelectOption(label="自分の声・速度", value="personal"),
                discord.SelectOption(label="読み上げるチャンネル", value="sources"),
                discord.SelectOption(label="共通の読み方・辞書", value="shared"),
            ],
        )
        self.runtime = runtime

    async def callback(self, interaction: discord.Interaction) -> None:
        await _send_read_aloud_setup(interaction, self.runtime, section=self.values[0])


async def send_audio_hub(interaction: discord.Interaction, runtime: SimajilordRuntime) -> None:
    if interaction.guild_id is None:
        await interaction.response.send_message("サーバー内で開いてください。", ephemeral=True)
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
