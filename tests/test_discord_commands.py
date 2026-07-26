from __future__ import annotations

from typing import cast

import discord
import pytest
from discord import app_commands

from simajilord.integrations.discord.cogs import MusicCog, MusicControlsView
from simajilord.runtime import SimajilordRuntime


def test_common_music_actions_have_short_top_level_commands() -> None:
    commands = {
        command.name: command
        for command in MusicCog.__cog_app_commands__
        if isinstance(command, app_commands.Command)
    }
    assert {"play", "queue", "history"} <= commands.keys()
    assert commands["play"].description == (
        "Add a YouTube/TikTok URL or search to the music queue."
    )
    assert commands["queue"].description == "Show what is playing and what comes next."
    assert commands["history"].description == (
        "Show recently played tracks and who requested them."
    )


def test_advanced_music_group_keeps_compatible_and_power_commands() -> None:
    group = next(
        command
        for command in MusicCog.__cog_app_commands__
        if isinstance(command, app_commands.Group) and command.name == "music"
    )
    names = {command.name for command in group.commands}
    assert names == {
        "play",
        "queue",
        "history",
        "pause",
        "resume",
        "skip",
        "stop",
        "leave",
        "loop",
        "remove",
        "autoleave",
        "shuffle",
        "seek",
        "tune",
    }


@pytest.mark.asyncio
async def test_music_buttons_are_concise_grouped_and_uniquely_addressable() -> None:
    view = MusicControlsView(cast(SimajilordRuntime, object()))
    buttons = [
        child for child in view.children if isinstance(child, discord.ui.Button)
    ]
    assert [button.label for button in buttons] == [
        "Start in VC",
        "Pause",
        "Resume",
        "Skip",
        "Loop",
        "Leave",
    ]
    assert sum(button.row == 0 for button in buttons) == 5
    assert sum(button.row == 1 for button in buttons) == 1
    custom_ids = [button.custom_id for button in buttons]
    assert None not in custom_ids
    assert len(custom_ids) == len(set(custom_ids))
