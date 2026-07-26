from __future__ import annotations

import json

from simajilord.providers.image.ideogram_mlx import _canonical_caption


def test_canonical_caption_repairs_illustration_and_element_key_order() -> None:
    caption = json.dumps(
        {
            "high_level_description": "A cat.",
            "style_description": {
                "aesthetics": "coherent",
                "lighting": "soft",
                "art_style": "editorial",
                "medium": "digital illustration",
            },
            "compositional_deconstruction": {
                "elements": [{"desc": "cat", "type": "obj"}],
                "background": "room",
            },
        }
    )

    repaired = json.loads(_canonical_caption(caption))

    assert tuple(repaired["style_description"]) == (
        "aesthetics",
        "lighting",
        "medium",
        "art_style",
    )
    assert tuple(repaired["compositional_deconstruction"]) == (
        "background",
        "elements",
    )
    assert tuple(repaired["compositional_deconstruction"]["elements"][0]) == (
        "type",
        "desc",
    )
