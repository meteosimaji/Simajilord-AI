from __future__ import annotations

from simajilord.integrations.discord.presenter import (
    EmbedField,
    EmbedTone,
    command_embed,
)


def test_command_embed_keeps_useful_timestamp_without_meta_footer() -> None:
    embed = command_embed(
        "Platform status",
        fields=(EmbedField("Status", "ok"),),
        tone=EmbedTone.SUCCESS,
    )
    assert embed.title == "Platform status"
    assert embed.timestamp is not None
    assert embed.footer.text is None
    assert embed.fields[0].name == "Status"
    assert embed.fields[0].value == "ok"
