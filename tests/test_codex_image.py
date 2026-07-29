from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from simajilord.core.errors import ProviderError
from simajilord.providers.codex_features import codex_feature_arguments
from simajilord.providers.image.codex import (
    CodexImageProvider,
    _image_prompt,
)


def _provider(tmp_path: Path) -> CodexImageProvider:
    return CodexImageProvider(
        executable="codex",
        model="gpt-5.6-terra",
        workspace_dir=tmp_path / "workspace",
        timeout_seconds=60,
    )


def _png_bytes(*, width: int = 13, height: int = 7) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(
        stream,
        format="PNG",
    )
    return stream.getvalue()


def test_codex_image_feature_is_enabled_without_enabling_unrelated_features() -> None:
    arguments = codex_feature_arguments(allow_image_generation=True)
    pairs = tuple(zip(arguments[::2], arguments[1::2], strict=True))

    assert ("--enable", "image_generation") in pairs
    assert ("--disable", "image_generation") not in pairs
    assert ("--disable", "shell_tool") in pairs
    assert ("--disable", "plugins") in pairs


def test_codex_image_prompt_preserves_brief_and_requested_shape() -> None:
    prompt = _image_prompt(
        '{"subject":"one cat","avoid":"text"}',
        width=768,
        height=512,
    )

    assert "exactly one image" in prompt
    assert "landscape (768:512 target ratio)" in prompt
    assert '"subject":"one cat"' in prompt
    assert '"avoid":"text"' in prompt


def test_codex_image_imports_only_from_generated_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codex_home = tmp_path / "codex-home"
    source = codex_home / "generated_images" / "thread" / "result.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(_png_bytes())
    destination = tmp_path / "simajilord" / "job.png"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    dimensions = _provider(tmp_path)._import_image(
        {
            "type": "imageGeneration",
            "status": "completed",
            "result": "",
            "savedPath": str(source),
        },
        destination,
    )

    assert dimensions == (13, 7)
    assert destination.read_bytes() == _png_bytes()
    assert not source.exists()


def test_codex_image_rejects_saved_path_outside_codex_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codex_home = tmp_path / "codex-home"
    outside = tmp_path / "outside.png"
    outside.write_bytes(_png_bytes())
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    with pytest.raises(ProviderError, match="outside its generated directory"):
        _provider(tmp_path)._import_image(
            {
                "type": "imageGeneration",
                "status": "completed",
                "result": "",
                "savedPath": str(outside),
            },
            tmp_path / "destination.png",
        )


def test_codex_image_imports_base64_fallback_and_validates_png(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "destination.png"
    result = base64.b64encode(_png_bytes(width=5, height=9)).decode()

    dimensions = _provider(tmp_path)._import_image(
        {
            "type": "imageGeneration",
            "status": "completed",
            "result": result,
        },
        destination,
    )

    assert dimensions == (5, 9)
    with Image.open(destination) as image:
        assert image.format == "PNG"


@pytest.mark.asyncio
async def test_codex_image_waits_for_completed_turn_after_image_notification(
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path)
    image_item = {
        "type": "imageGeneration",
        "id": "image-1",
        "status": "completed",
        "result": "encoded",
    }
    await provider._notifications.put(
        (
            "item/completed",
            {"threadId": "thread", "turnId": "turn", "item": image_item},
        )
    )
    await provider._notifications.put(
        (
            "turn/completed",
            {
                "threadId": "thread",
                "turnId": "turn",
                "turn": {"id": "turn", "status": "completed", "items": []},
            },
        )
    )

    received = await asyncio.wait_for(
        provider._await_image("thread", "turn", on_progress=None),
        timeout=1,
    )

    assert received is image_item


@pytest.mark.asyncio
async def test_codex_image_reports_turn_without_generated_file(
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path)
    await provider._notifications.put(
        (
            "turn/completed",
            {
                "threadId": "thread",
                "turnId": "turn",
                "turn": {"id": "turn", "status": "completed", "items": []},
            },
        )
    )

    with pytest.raises(ProviderError, match="without generating an image file"):
        await provider._await_image("thread", "turn", on_progress=None)
