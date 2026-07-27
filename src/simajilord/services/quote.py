"""Purely local rendering for Discord quote images."""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageFont,
    ImageOps,
    UnidentifiedImageError,
)

_LANDSCAPE_SIZE = (1200, 630)
_PORTRAIT_SIZE = (768, 960)
_CUSTOM_EMOJI_PATTERN = re.compile(
    r"<(?P<animated>a?):(?P<name>[A-Za-z0-9_]{2,32}):(?P<id>[0-9]{15,22})>"
)
_MAX_TEXT_CHARACTERS = 4_000
_MAX_AVATAR_BYTES = 8_000_000
_MAX_CUSTOM_EMOJI_BYTES = 2_000_000
_MAX_CUSTOM_EMOJIS = 25
_MAX_STICKERS = 3
_MAX_ANIMATION_FRAMES = 20
_ANIMATION_FRAME_MS = 100
_TSUKUSHI_A_ROUNDED = Path(
    "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/"
    "a9507b2dd1e57ecf7dcbeff43b7e70c9d42bcd2f.asset/"
    "AssetData/TsukushiAMaruGothic.ttc"
)
_NORMAL_FONT_CANDIDATES = (
    (_TSUKUSHI_A_ROUNDED, 0),
    (Path("/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc"), 0),
    (Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"), 0),
    (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), 0),
    (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), 0),
)
_BOLD_FONT_CANDIDATES = (
    (_TSUKUSHI_A_ROUNDED, 1),
    (Path("/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc"), 0),
    (Path("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"), 0),
    (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"), 0),
    (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"), 0),
)
_EMOJI_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Apple Color Emoji.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"),
)
_QUOTE_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Georgia Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
)


@dataclass(frozen=True, slots=True)
class QuoteCustomEmojiAsset:
    """One Discord-controlled custom emoji used by the source message."""

    emoji_id: str
    name: str
    content: bytes


@dataclass(frozen=True, slots=True)
class QuoteStickerAsset:
    """One Discord sticker displayed by the source message."""

    sticker_id: str
    name: str
    content: bytes


@dataclass(frozen=True, slots=True)
class QuoteRenderRequest:
    """Transport-neutral data needed to render one quote."""

    text: str
    display_name: str
    username: str
    avatar: bytes | None
    custom_emojis: tuple[QuoteCustomEmojiAsset, ...] = ()
    stickers: tuple[QuoteStickerAsset, ...] = ()
    color: bool = False
    light: bool = False
    flip: bool = False
    bold: bool = False
    vertical: bool = False
    animate: bool = False


@dataclass(frozen=True, slots=True)
class QuoteRenderResult:
    """A complete local PNG and a small amount of render metadata."""

    content: bytes
    width: int
    height: int
    rendered_custom_emojis: int
    rendered_stickers: int
    text_truncated: bool
    animated: bool


@dataclass(frozen=True, slots=True)
class _Atom:
    kind: Literal["text", "space", "emoji", "custom", "newline"]
    value: str
    emoji_id: str | None = None
    emoji_name: str | None = None


@dataclass(frozen=True, slots=True)
class _MeasuredAtom:
    atom: _Atom
    width: float


@dataclass(frozen=True, slots=True)
class _TextLayout:
    lines: tuple[tuple[_MeasuredAtom, ...], ...]
    font: ImageFont.FreeTypeFont
    emoji_font: ImageFont.FreeTypeFont | None
    font_size: int
    line_height: int
    bold: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class _DecodedAnimation:
    frames: tuple[bytes, ...]
    durations_ms: tuple[int, ...]

    @property
    def total_duration_ms(self) -> int:
        return sum(self.durations_ms)

    def frame_at(self, timestamp_ms: int) -> bytes:
        offset = timestamp_ms % self.total_duration_ms
        elapsed = 0
        for frame, duration in zip(self.frames, self.durations_ms, strict=True):
            elapsed += duration
            if offset < elapsed:
                return frame
        return self.frames[-1]


class QuoteImageService:
    """Render 1200x630 quote cards without an external rendering API."""

    def render(self, request: QuoteRenderRequest) -> QuoteRenderResult:
        if request.animate:
            animated = self._render_animation(request)
            if animated is not None:
                return animated
        return self._render_static(request)

    def _render_animation(
        self,
        request: QuoteRenderRequest,
    ) -> QuoteRenderResult | None:
        variants = _animation_variants(request)
        if len(variants) < 2:
            return None
        results = tuple(self._render_static(variant) for variant in variants)
        rgb_frames: list[Image.Image] = []
        for result in results:
            with Image.open(io.BytesIO(result.content)) as source_frame:
                rgb_frames.append(source_frame.convert("RGB"))
        frames = [
            rgb_frame.quantize(
                colors=256,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.FLOYDSTEINBERG,
            )
            for rgb_frame in rgb_frames
        ]
        output = io.BytesIO()
        frames[0].save(
            output,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=_ANIMATION_FRAME_MS,
            loop=0,
            optimize=False,
            disposal=1,
            include_color_table=True,
        )
        first = results[0]
        return QuoteRenderResult(
            content=output.getvalue(),
            width=first.width,
            height=first.height,
            rendered_custom_emojis=first.rendered_custom_emojis,
            rendered_stickers=first.rendered_stickers,
            text_truncated=first.text_truncated,
            animated=True,
        )

    def _render_static(self, request: QuoteRenderRequest) -> QuoteRenderResult:
        if request.text.strip() or not request.stickers:
            text, input_truncated = _bounded_text(request.text)
        else:
            text, input_truncated = "", False
        custom_images = _custom_emoji_images(request.custom_emojis)
        sticker_images = _sticker_images(request.stickers)
        canvas_size = _PORTRAIT_SIZE if request.vertical else _LANDSCAPE_SIZE
        canvas = Image.new(
            "RGBA",
            canvas_size,
            (247, 247, 247, 255) if request.light else (0, 0, 0, 255),
        )
        _draw_avatar(
            canvas,
            request.avatar,
            color=request.color,
            light=request.light,
            flip=request.flip,
            vertical=request.vertical,
            display_name=request.display_name,
        )
        foreground = (24, 24, 27, 255) if request.light else (245, 245, 245, 255)
        muted = (92, 92, 100, 255) if request.light else (104, 104, 112, 255)
        if request.vertical:
            text_left, text_right = (70, 698)
        else:
            text_left, text_right = (70, 560) if request.flip else (640, 1130)
        text_width = text_right - text_left
        sticker_size = 0
        if sticker_images:
            sticker_size = 180 if not text else 72
            if request.vertical:
                sticker_size = 190 if not text else 72
        sticker_gap = 18 if sticker_size and text else 0
        total_body_budget = 230 if request.vertical else 400
        text_budget = max(1, total_body_budget - sticker_size - sticker_gap)
        layout = (
            _fit_text(
                text,
                maximum_width=text_width,
                maximum_height=text_budget,
                maximum_lines=4,
                custom_images=custom_images,
                bold=request.bold,
            )
            if text
            else None
        )
        name_font = _font(30, bold=False)
        username_font = _font(20, bold=False)
        body_height = sticker_size + sticker_gap
        if layout is not None:
            body_height += len(layout.lines) * layout.line_height
        if request.vertical:
            # Keep the author block anchored near the bottom like Make it a
            # Quote, while allowing longer quotations to grow upward.
            top = 790 - body_height
        else:
            author_height = 30 + 8 + (26 if request.username else 0)
            group_height = body_height + 32 + author_height
            top = max(42, (canvas_size[1] - group_height) // 2)
        draw = ImageDraw.Draw(canvas)
        if request.vertical:
            quote_font = _quote_font(140)
            quote_text = "“ ”"
            quote_width = draw.textlength(quote_text, font=quote_font)
            draw.text(
                ((canvas_size[0] - quote_width) / 2, 480),
                quote_text,
                font=quote_font,
                fill=foreground,
            )
        custom_render_count = 0
        y = top
        rendered_stickers = _draw_sticker_row(
            canvas,
            sticker_images,
            left=text_left,
            width=text_width,
            y=y,
            size=sticker_size,
        )
        if sticker_size:
            y += sticker_size + sticker_gap
        if layout is not None:
            for line in layout.lines:
                line_width = sum(item.width for item in line)
                x = text_left + max(0.0, (text_width - line_width) / 2)
                for measured in line:
                    rendered = _draw_atom(
                        canvas,
                        draw,
                        measured,
                        x=x,
                        y=y,
                        line_height=layout.line_height,
                        font=layout.font,
                        emoji_font=layout.emoji_font,
                        custom_images=custom_images,
                        foreground=foreground,
                        bold=layout.bold,
                    )
                    custom_render_count += int(rendered)
                    x += measured.width
                y += layout.line_height
        if request.vertical:
            y = 865
            rule_width = min(220, text_width // 2)
            rule_left = (canvas_size[0] - rule_width) / 2
            draw.line(
                (rule_left, y - 16, rule_left + rule_width, y - 16),
                fill=foreground,
                width=2,
            )
            author = request.display_name.strip() or "Unknown"
        else:
            y += 26
            author = f"- {request.display_name.strip() or 'Unknown'}"
        author_width = draw.textlength(author, font=name_font)
        draw.text(
            (text_left + max(0.0, (text_width - author_width) / 2), y),
            author,
            font=name_font,
            fill=foreground,
            stroke_width=int(request.bold),
            stroke_fill=foreground,
        )
        if request.username:
            y += 38
            username = f"@{request.username.lstrip('@')}"
            username_width = draw.textlength(username, font=username_font)
            draw.text(
                (
                    text_left + max(0.0, (text_width - username_width) / 2),
                    y,
                ),
                username,
                font=username_font,
                fill=muted,
            )
        output = io.BytesIO()
        canvas.convert("RGB").save(output, format="PNG", optimize=True)
        return QuoteRenderResult(
            content=output.getvalue(),
            width=canvas_size[0],
            height=canvas_size[1],
            rendered_custom_emojis=custom_render_count,
            rendered_stickers=rendered_stickers,
            text_truncated=input_truncated or bool(layout and layout.truncated),
            animated=False,
        )


def _bounded_text(value: str) -> tuple[str, bool]:
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return "…", False
    if len(text) <= _MAX_TEXT_CHARACTERS:
        return text, False
    return f"{text[: _MAX_TEXT_CHARACTERS - 1]}…", True


def _custom_emoji_images(
    assets: tuple[QuoteCustomEmojiAsset, ...],
) -> dict[str, Image.Image]:
    images: dict[str, Image.Image] = {}
    total = 0
    for asset in assets[:_MAX_CUSTOM_EMOJIS]:
        if (
            asset.emoji_id in images
            or not asset.content
            or len(asset.content) > _MAX_CUSTOM_EMOJI_BYTES
        ):
            continue
        total += len(asset.content)
        if total > _MAX_AVATAR_BYTES:
            break
        try:
            with Image.open(io.BytesIO(asset.content)) as source:
                source.seek(0)
                images[asset.emoji_id] = source.convert("RGBA")
        except (UnidentifiedImageError, OSError):
            continue
    return images


def _sticker_images(
    assets: tuple[QuoteStickerAsset, ...],
) -> tuple[Image.Image, ...]:
    images: list[Image.Image] = []
    total = 0
    for asset in assets[:_MAX_STICKERS]:
        if not asset.content or len(asset.content) > _MAX_CUSTOM_EMOJI_BYTES:
            continue
        total += len(asset.content)
        if total > _MAX_AVATAR_BYTES:
            break
        try:
            with Image.open(io.BytesIO(asset.content)) as source:
                source.seek(0)
                images.append(source.convert("RGBA"))
        except (UnidentifiedImageError, OSError):
            continue
    return tuple(images)


def _draw_sticker_row(
    canvas: Image.Image,
    images: tuple[Image.Image, ...],
    *,
    left: int,
    width: int,
    y: int,
    size: int,
) -> int:
    if not images or size <= 0:
        return 0
    gap = 12
    item_size = min(size, max(1, (width - gap * (len(images) - 1)) // len(images)))
    prepared = tuple(
        ImageOps.contain(
            image,
            (item_size, item_size),
            method=Image.Resampling.LANCZOS,
        )
        for image in images
    )
    row_width = sum(image.width for image in prepared) + gap * (len(prepared) - 1)
    x = left + max(0, (width - row_width) // 2)
    for image in prepared:
        canvas.alpha_composite(
            image,
            (
                x,
                y + max(0, (size - image.height) // 2),
            ),
        )
        x += image.width + gap
    return len(prepared)


def _animation_variants(
    request: QuoteRenderRequest,
) -> tuple[QuoteRenderRequest, ...]:
    custom_animations = tuple(_decode_animation(asset.content) for asset in request.custom_emojis)
    sticker_animations = tuple(_decode_animation(asset.content) for asset in request.stickers)
    animations = tuple(
        item for item in (*custom_animations, *sticker_animations) if item is not None
    )
    if not animations:
        return ()
    duration_ms = min(
        _MAX_ANIMATION_FRAMES * _ANIMATION_FRAME_MS,
        max(item.total_duration_ms for item in animations),
    )
    frame_count = min(
        _MAX_ANIMATION_FRAMES,
        max(2, (duration_ms + _ANIMATION_FRAME_MS - 1) // _ANIMATION_FRAME_MS),
    )
    variants: list[QuoteRenderRequest] = []
    for frame_index in range(frame_count):
        timestamp_ms = frame_index * _ANIMATION_FRAME_MS
        custom_emojis = tuple(
            replace(
                asset,
                content=(
                    animation.frame_at(timestamp_ms) if animation is not None else asset.content
                ),
            )
            for asset, animation in zip(
                request.custom_emojis,
                custom_animations,
                strict=True,
            )
        )
        stickers = tuple(
            replace(
                asset,
                content=(
                    animation.frame_at(timestamp_ms) if animation is not None else asset.content
                ),
            )
            for asset, animation in zip(
                request.stickers,
                sticker_animations,
                strict=True,
            )
        )
        variants.append(
            replace(
                request,
                custom_emojis=custom_emojis,
                stickers=stickers,
                animate=False,
            )
        )
    return tuple(variants)


def _decode_animation(content: bytes) -> _DecodedAnimation | None:
    if not content:
        return None
    try:
        with Image.open(io.BytesIO(content)) as source:
            frame_count = min(
                int(getattr(source, "n_frames", 1)),
                _MAX_ANIMATION_FRAMES,
            )
            if frame_count < 2:
                return None
            frames: list[bytes] = []
            durations: list[int] = []
            for frame_index in range(frame_count):
                source.seek(frame_index)
                frame = source.convert("RGBA")
                output = io.BytesIO()
                frame.save(output, format="PNG", optimize=True)
                frames.append(output.getvalue())
                duration = int(source.info.get("duration", _ANIMATION_FRAME_MS))
                durations.append(max(40, min(1_000, duration)))
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    return _DecodedAnimation(
        frames=tuple(frames),
        durations_ms=tuple(durations),
    )


def _draw_avatar(
    canvas: Image.Image,
    content: bytes | None,
    *,
    color: bool,
    light: bool,
    flip: bool,
    vertical: bool,
    display_name: str,
) -> None:
    avatar_size = canvas.width if vertical else canvas.height
    avatar: Image.Image | None = None
    if content and len(content) <= _MAX_AVATAR_BYTES:
        try:
            with Image.open(io.BytesIO(content)) as source:
                source.seek(0)
                avatar = ImageOps.fit(
                    source.convert("RGBA"),
                    (avatar_size, avatar_size),
                    method=Image.Resampling.LANCZOS,
                )
        except (UnidentifiedImageError, OSError):
            avatar = None
    if avatar is None:
        avatar = _placeholder_avatar(display_name, light=light, size=avatar_size)
    if not color:
        alpha = avatar.getchannel("A")
        avatar = ImageOps.grayscale(avatar).convert("RGBA")
        avatar.putalpha(alpha)
    if flip:
        avatar = ImageOps.mirror(avatar)
    fade = Image.new("L", avatar.size)
    fade_pixels = fade.load()
    assert fade_pixels is not None
    if vertical:
        full_opacity_until = round(avatar_size * 0.42)
        fade_length = avatar_size - full_opacity_until
        for y in range(avatar_size):
            opacity = (
                255
                if y <= full_opacity_until
                else round(255 * max(0.0, (avatar_size - y) / fade_length))
            )
            for x in range(avatar_size):
                fade_pixels[x, y] = opacity
    else:
        for x in range(avatar_size):
            source_x = avatar_size - 1 - x if flip else x
            opacity = (
                255 if source_x <= 290 else round(255 * max(0.0, (avatar_size - source_x) / 340))
            )
            for y in range(avatar_size):
                fade_pixels[x, y] = opacity
    mask = ImageChops.multiply(avatar.getchannel("A"), fade)
    destination_x = canvas.width - avatar_size if flip and not vertical else 0
    canvas.paste(avatar, (destination_x, 0), mask)


def _placeholder_avatar(display_name: str, *, light: bool, size: int) -> Image.Image:
    image = Image.new(
        "RGBA",
        (size, size),
        (224, 224, 228, 255) if light else (24, 24, 27, 255),
    )
    draw = ImageDraw.Draw(image)
    radius = 180
    center = (size // 2 - 30, size // 2)
    fill = (161, 161, 170, 255) if light else (63, 63, 70, 255)
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        fill=fill,
    )
    initial = (display_name.strip() or "?")[0]
    font = _font(180, bold=True)
    bounds = draw.textbbox((0, 0), initial, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (
            center[0] - width / 2,
            center[1] - height / 2 - bounds[1],
        ),
        initial,
        font=font,
        fill=(245, 245, 245, 255) if not light else (39, 39, 42, 255),
    )
    return image


def _quote_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in _QUOTE_FONT_CANDIDATES:
        if not candidate.exists():
            continue
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            continue
    return _font(size, bold=True)


def _fit_text(
    text: str,
    *,
    maximum_width: int,
    maximum_height: int,
    maximum_lines: int,
    custom_images: dict[str, Image.Image],
    bold: bool,
) -> _TextLayout:
    atoms = _tokenize(text)
    for font_size in range(58, 23, -2):
        font = _font(font_size, bold=bold)
        emoji_font = _emoji_font(font_size)
        line_height = round(font_size * 1.3)
        lines = _wrap_atoms(
            atoms,
            maximum_width=maximum_width,
            font=font,
            emoji_font=emoji_font,
            font_size=font_size,
            custom_images=custom_images,
        )
        if len(lines) <= maximum_lines and len(lines) * line_height <= maximum_height:
            return _TextLayout(
                lines=lines,
                font=font,
                emoji_font=emoji_font,
                font_size=font_size,
                line_height=line_height,
                bold=bold,
                truncated=False,
            )
    font_size = 24
    font = _font(font_size, bold=bold)
    emoji_font = _emoji_font(font_size)
    line_height = round(font_size * 1.3)
    fallback_lines = list(
        _wrap_atoms(
            atoms,
            maximum_width=maximum_width,
            font=font,
            emoji_font=emoji_font,
            font_size=font_size,
            custom_images=custom_images,
        )
    )
    visible_lines = min(maximum_lines, max(1, maximum_height // line_height))
    truncated = len(fallback_lines) > visible_lines
    if truncated:
        fallback_lines = fallback_lines[:visible_lines]
        ellipsis = _MeasuredAtom(
            atom=_Atom(kind="text", value="…"),
            width=_measure_text("…", font),
        )
        last = list(fallback_lines[-1])
        while last and sum(item.width for item in (*last, ellipsis)) > maximum_width:
            last.pop()
        last.append(ellipsis)
        fallback_lines[-1] = tuple(last)
    return _TextLayout(
        lines=tuple(fallback_lines),
        font=font,
        emoji_font=emoji_font,
        font_size=font_size,
        line_height=line_height,
        bold=bold,
        truncated=truncated,
    )


def _wrap_atoms(
    atoms: tuple[_Atom, ...],
    *,
    maximum_width: int,
    font: ImageFont.FreeTypeFont,
    emoji_font: ImageFont.FreeTypeFont | None,
    font_size: int,
    custom_images: dict[str, Image.Image],
) -> tuple[tuple[_MeasuredAtom, ...], ...]:
    lines: list[list[_MeasuredAtom]] = [[]]
    current_width = 0.0
    for atom in atoms:
        if atom.kind == "newline":
            lines.append([])
            current_width = 0.0
            continue
        measured = _measure_atom(
            atom,
            font=font,
            emoji_font=emoji_font,
            font_size=font_size,
            custom_images=custom_images,
        )
        if atom.kind == "space" and not lines[-1]:
            continue
        if lines[-1] and current_width + measured.width > maximum_width:
            lines.append([])
            current_width = 0.0
            if atom.kind == "space":
                continue
        if measured.width > maximum_width and atom.kind == "text" and len(atom.value) > 1:
            for character in atom.value:
                nested = _measure_atom(
                    _Atom(kind="text", value=character),
                    font=font,
                    emoji_font=emoji_font,
                    font_size=font_size,
                    custom_images=custom_images,
                )
                if lines[-1] and current_width + nested.width > maximum_width:
                    lines.append([])
                    current_width = 0.0
                lines[-1].append(nested)
                current_width += nested.width
            continue
        lines[-1].append(measured)
        current_width += measured.width
    while len(lines) > 1 and not lines[-1]:
        lines.pop()
    return tuple(tuple(line) for line in lines)


def _measure_atom(
    atom: _Atom,
    *,
    font: ImageFont.FreeTypeFont,
    emoji_font: ImageFont.FreeTypeFont | None,
    font_size: int,
    custom_images: dict[str, Image.Image],
) -> _MeasuredAtom:
    if atom.kind == "custom" and atom.emoji_id in custom_images:
        width = float(round(font_size * 1.08) + 4)
    elif atom.kind == "custom":
        width = _measure_text(f":{atom.emoji_name or 'emoji'}:", font)
    elif atom.kind == "emoji" and emoji_font is not None:
        width = float(round(font_size * 1.08) + 4)
    else:
        width = _measure_text(atom.value, font)
    return _MeasuredAtom(atom=atom, width=width)


def _draw_atom(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    measured: _MeasuredAtom,
    *,
    x: float,
    y: float,
    line_height: int,
    font: ImageFont.FreeTypeFont,
    emoji_font: ImageFont.FreeTypeFont | None,
    custom_images: dict[str, Image.Image],
    foreground: tuple[int, int, int, int],
    bold: bool,
) -> bool:
    atom = measured.atom
    if atom.kind == "custom" and atom.emoji_id in custom_images:
        size = max(1, round(font.size * 1.08))
        source = custom_images[atom.emoji_id]
        icon = ImageOps.contain(
            source,
            (size, size),
            method=Image.Resampling.LANCZOS,
        )
        canvas.alpha_composite(
            icon,
            (
                round(x),
                round(y + max(0, (line_height - icon.height) / 2)),
            ),
        )
        return True
    if atom.kind == "custom":
        draw.text(
            (x, y + max(0, (line_height - font.size) / 2)),
            f":{atom.emoji_name or 'emoji'}:",
            font=font,
            fill=foreground,
            stroke_width=int(bold),
            stroke_fill=foreground,
        )
        return False
    if atom.kind == "emoji" and emoji_font is not None:
        size = max(1, round(font.size * 1.08))
        unicode_icon = _unicode_emoji_image(atom.value, emoji_font, size=size)
        if unicode_icon is not None:
            canvas.alpha_composite(
                unicode_icon,
                (
                    round(x),
                    round(y + max(0, (line_height - unicode_icon.height) / 2)),
                ),
            )
            return False
    draw.text(
        (x, y + max(0, (line_height - font.size) / 2)),
        atom.value,
        font=font,
        fill=foreground,
        stroke_width=int(bold),
        stroke_fill=foreground,
    )
    return False


def _measure_text(value: str, font: ImageFont.FreeTypeFont) -> float:
    image = Image.new("L", (1, 1))
    return ImageDraw.Draw(image).textlength(value, font=font)


def _unicode_emoji_image(
    value: str,
    font: ImageFont.FreeTypeFont,
    *,
    size: int,
) -> Image.Image | None:
    """Rasterize a fixed-strike color font, then constrain it to one text line."""

    probe = Image.new("RGBA", (1, 1))
    probe_draw = ImageDraw.Draw(probe)
    try:
        bounds = probe_draw.textbbox(
            (0, 0),
            value,
            font=font,
            embedded_color=True,
        )
    except (OSError, ValueError):
        return None
    width = max(1, round(bounds[2] - bounds[0]))
    height = max(1, round(bounds[3] - bounds[1]))
    padding = max(4, round(font.size / 8))
    source = Image.new(
        "RGBA",
        (width + padding * 2, height + padding * 2),
        (0, 0, 0, 0),
    )
    try:
        ImageDraw.Draw(source).text(
            (padding - bounds[0], padding - bounds[1]),
            value,
            font=font,
            embedded_color=True,
        )
    except (OSError, ValueError):
        return None
    visible = source.getchannel("A").getbbox()
    if visible is None:
        return None
    return ImageOps.contain(
        source.crop(visible),
        (size, size),
        method=Image.Resampling.LANCZOS,
    )


def _tokenize(text: str) -> tuple[_Atom, ...]:
    atoms: list[_Atom] = []
    index = 0
    while index < len(text):
        custom = _CUSTOM_EMOJI_PATTERN.match(text, index)
        if custom is not None:
            atoms.append(
                _Atom(
                    kind="custom",
                    value=custom.group(0),
                    emoji_id=custom.group("id"),
                    emoji_name=custom.group("name"),
                )
            )
            index = custom.end()
            continue
        character = text[index]
        if character == "\n":
            atoms.append(_Atom(kind="newline", value=character))
            index += 1
            continue
        if character.isspace():
            while index < len(text) and text[index].isspace() and text[index] != "\n":
                index += 1
            atoms.append(_Atom(kind="space", value=" "))
            continue
        cluster, next_index = _grapheme_cluster(text, index)
        if _is_emoji_cluster(cluster):
            atoms.append(_Atom(kind="emoji", value=cluster))
            index = next_index
            continue
        if character.isascii() and (character.isalnum() or character in "'_-"):
            end = index + 1
            while (
                end < len(text)
                and text[end].isascii()
                and (text[end].isalnum() or text[end] in "'_-")
            ):
                end += 1
            atoms.append(_Atom(kind="text", value=text[index:end]))
            index = end
            continue
        atoms.append(_Atom(kind="text", value=cluster))
        index = next_index
    return tuple(atoms)


def _grapheme_cluster(text: str, start: int) -> tuple[str, int]:
    index = start + 1
    first = ord(text[start])
    if (
        _is_regional_indicator(first)
        and index < len(text)
        and _is_regional_indicator(ord(text[index]))
    ):
        index += 1
    while index < len(text) and _is_cluster_suffix(text[index]):
        index += 1
    while index < len(text) and text[index] == "\u200d":
        if index + 1 >= len(text):
            index += 1
            break
        index += 2
        while index < len(text) and _is_cluster_suffix(text[index]):
            index += 1
    return text[start:index], index


def _is_cluster_suffix(character: str) -> bool:
    codepoint = ord(character)
    return (
        unicodedata.combining(character) != 0
        or codepoint in {0xFE0E, 0xFE0F, 0x20E3}
        or 0x1F3FB <= codepoint <= 0x1F3FF
    )


def _is_emoji_cluster(value: str) -> bool:
    return any(
        0x1F000 <= ord(character) <= 0x1FAFF
        or 0x2600 <= ord(character) <= 0x27BF
        or _is_regional_indicator(ord(character))
        or ord(character) in {0x00A9, 0x00AE, 0x2122, 0x20E3}
        for character in value
    )


def _is_regional_indicator(codepoint: int) -> bool:
    return 0x1F1E6 <= codepoint <= 0x1F1FF


def _font(size: int, *, bold: bool) -> ImageFont.FreeTypeFont:
    candidates = _BOLD_FONT_CANDIDATES if bold else _NORMAL_FONT_CANDIDATES
    for path, index in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size, index=index)
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(filename, size)


def _emoji_font(size: int) -> ImageFont.FreeTypeFont | None:
    for path in _EMOJI_FONT_CANDIDATES:
        if not path.exists():
            continue
        for candidate_size in (size, 64, 48, 32, 96, 160):
            try:
                return ImageFont.truetype(str(path), candidate_size)
            except OSError:
                continue
    return None
