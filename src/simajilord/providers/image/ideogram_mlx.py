"""Local Ideogram 4 generation through the MLX-native mflux runner."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path

from simajilord.core.errors import ProviderError

from .base import ImageProgressCallback, ImageProviderResult

_GENERATION_SECONDS = re.compile(r'"generation_time_seconds"\s*:\s*([0-9.]+)')
_PERCENTAGE = re.compile(rb"(?<!\d)(\d{1,3})%\|")
_TOTAL_STEPS = 12


class IdeogramMlxProvider:
    """Run one bounded local generation without exposing host paths to the model."""

    def __init__(
        self,
        *,
        model_path: Path,
        mflux_source: str,
        timeout_seconds: float,
        mlx_cache_limit_gb: int,
    ) -> None:
        if not model_path.is_dir():
            raise ProviderError("The configured Ideogram 4 model directory is unavailable.")
        executable = shutil.which("uvx")
        if executable is None:
            raise ProviderError("uvx is required for local Ideogram 4 generation.")
        self.executable = executable
        self.model_path = model_path
        self.mflux_source = mflux_source
        self.timeout_seconds = timeout_seconds
        self.mlx_cache_limit_gb = mlx_cache_limit_gb

    async def generate(
        self,
        *,
        caption_json: str,
        destination: Path,
        width: int,
        height: int,
        seed: int,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageProviderResult:
        caption_json = _canonical_caption(caption_json)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        metadata = destination.with_suffix(".metadata.json")
        metadata.unlink(missing_ok=True)
        process = await asyncio.create_subprocess_exec(
            self.executable,
            "--from",
            self.mflux_source,
            "mflux-generate-ideogram4",
            "--model",
            str(self.model_path),
            "--prompt",
            caption_json,
            "--width",
            str(width),
            "--height",
            str(height),
            "--preset",
            "V4_TURBO_12",
            "--seed",
            str(seed),
            "--mlx-cache-limit-gb",
            str(self.mlx_cache_limit_gb),
            "--strict-caption-validation",
            "--metadata",
            "--output",
            str(destination),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        chunks: list[bytes] = []
        last_reported_step = 0
        try:
            async with asyncio.timeout(self.timeout_seconds):
                assert process.stdout is not None
                while chunk := await process.stdout.read(4_096):
                    chunks.append(chunk)
                    if sum(map(len, chunks)) > 32_000:
                        chunks = [b"".join(chunks)[-16_000:]]
                    if on_progress is None:
                        continue
                    percentages = [
                        int(match.group(1))
                        for match in _PERCENTAGE.finditer(chunk)
                        if int(match.group(1)) <= 100
                    ]
                    if not percentages:
                        continue
                    step = min(
                        _TOTAL_STEPS,
                        round(max(percentages) * _TOTAL_STEPS / 100),
                    )
                    milestone = max(
                        value
                        for value in (0, 3, 6, 9, 12)
                        if value <= step
                    )
                    if milestone > last_reported_step:
                        last_reported_step = milestone
                        await on_progress(milestone, _TOTAL_STEPS)
                await process.wait()
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            destination.unlink(missing_ok=True)
            metadata.unlink(missing_ok=True)
            raise
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            destination.unlink(missing_ok=True)
            metadata.unlink(missing_ok=True)
            raise ProviderError("Local image generation timed out.") from exc
        output = b"".join(chunks).decode(errors="replace")
        if process.returncode != 0 or not destination.is_file():
            destination.unlink(missing_ok=True)
            metadata.unlink(missing_ok=True)
            detail = output[-1_000:].strip()
            raise ProviderError(f"Local image generation failed: {detail or 'unknown error'}")
        with destination.open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                destination.unlink(missing_ok=True)
                metadata.unlink(missing_ok=True)
                raise ProviderError("Local image generation returned an invalid PNG file.")
        destination.chmod(0o600)
        generation_seconds = _metadata_generation_seconds(metadata, output)
        if metadata.exists():
            metadata.chmod(0o600)
        return ImageProviderResult(
            generation_seconds=generation_seconds,
            model="Ideogram 4 MLX Q8",
        )


def _canonical_caption(caption_json: str) -> str:
    """Normalize the ordered JSON schema before invoking strict mflux validation."""

    try:
        raw = json.loads(caption_json)
    except json.JSONDecodeError as exc:
        raise ProviderError("The Ideogram 4 caption is not valid JSON.") from exc
    if not isinstance(raw, dict):
        raise ProviderError("The Ideogram 4 caption must be a JSON object.")

    payload: dict[str, object] = {}
    for key in (
        "high_level_description",
        "style_description",
        "compositional_deconstruction",
    ):
        if key in raw:
            payload[key] = raw[key]

    style = payload.get("style_description")
    if isinstance(style, dict):
        style_order = (
            ("aesthetics", "lighting", "photo", "medium", "color_palette")
            if "photo" in style
            else ("aesthetics", "lighting", "medium", "art_style", "color_palette")
        )
        payload["style_description"] = {
            key: style[key] for key in style_order if key in style
        }

    composition = payload.get("compositional_deconstruction")
    if isinstance(composition, dict):
        normalized_composition: dict[str, object] = {
            key: composition[key]
            for key in ("background", "elements")
            if key in composition
        }
        elements = normalized_composition.get("elements")
        if isinstance(elements, list):
            normalized_composition["elements"] = [
                _canonical_element(item) for item in elements
            ]
        payload["compositional_deconstruction"] = normalized_composition

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _canonical_element(value: object) -> object:
    if not isinstance(value, dict):
        return value
    order = (
        ("type", "bbox", "text", "desc", "color_palette")
        if value.get("type") == "text"
        else ("type", "bbox", "desc", "color_palette")
    )
    return {key: value[key] for key in order if key in value}


def _metadata_generation_seconds(metadata: Path, output: str) -> float:
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        match = _GENERATION_SECONDS.search(output)
        return float(match.group(1)) if match else 0.0
    value = payload.get("generation_time_seconds")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0
