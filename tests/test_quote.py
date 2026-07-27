from __future__ import annotations

import io

from PIL import Image, ImageChops, ImageStat

from simajilord.services.quote import (
    QuoteCustomEmojiAsset,
    QuoteImageService,
    QuoteRenderRequest,
    QuoteStickerAsset,
    _draw_avatar,
    _emoji_font,
    _fit_text,
    _unicode_emoji_image,
)


def _png(color: str, *, size: tuple[int, int] = (256, 256)) -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", size, color).save(output, format="PNG")
    return output.getvalue()


def _gif() -> bytes:
    output = io.BytesIO()
    first = Image.new("RGBA", (32, 32), "red")
    second = Image.new("RGBA", (32, 32), "blue")
    first.save(
        output,
        format="GIF",
        save_all=True,
        append_images=(second,),
        duration=(100, 100),
        loop=0,
    )
    return output.getvalue()


def test_quote_renderer_outputs_local_png_with_japanese_and_emojis() -> None:
    result = QuoteImageService().render(
        QuoteRenderRequest(
            text=("日本語の引用 😀 <a:dance:123456789012345678> をローカルで描画します。"),
            display_name="しまじろーど",
            username="simajilord",
            avatar=_png("#cc8844"),
            custom_emojis=(
                QuoteCustomEmojiAsset(
                    emoji_id="123456789012345678",
                    name="dance",
                    content=_gif(),
                ),
            ),
            color=True,
        )
    )

    assert result.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert (result.width, result.height) == (1200, 630)
    assert result.rendered_custom_emojis == 1
    assert result.text_truncated is False
    with Image.open(io.BytesIO(result.content)) as image:
        assert image.size == (1200, 630)
        assert image.mode == "RGB"


def test_quote_renderer_bounds_long_text_without_rejecting_it() -> None:
    result = QuoteImageService().render(
        QuoteRenderRequest(
            text="長文" * 5_000,
            display_name="Test",
            username="test",
            avatar=None,
        )
    )

    assert result.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert result.text_truncated is True


def test_quote_flip_moves_avatar_energy_to_the_right() -> None:
    service = QuoteImageService()
    common = {
        "text": "Flip test",
        "display_name": "Test",
        "username": "test",
        "avatar": _png("white"),
        "color": True,
    }
    left = service.render(QuoteRenderRequest(**common, flip=False))
    right = service.render(QuoteRenderRequest(**common, flip=True))

    with Image.open(io.BytesIO(left.content)) as left_image:
        left_luminance = ImageStat.Stat(left_image.crop((0, 0, 300, 630))).mean[0]
        left_opposite = ImageStat.Stat(left_image.crop((900, 0, 1200, 630))).mean[0]
    with Image.open(io.BytesIO(right.content)) as right_image:
        right_luminance = ImageStat.Stat(right_image.crop((900, 0, 1200, 630))).mean[0]
        right_opposite = ImageStat.Stat(right_image.crop((0, 0, 300, 630))).mean[0]

    assert left_luminance > left_opposite
    assert right_luminance > right_opposite


def test_quote_avatar_keeps_a_square_render_region() -> None:
    canvas = Image.new("RGBA", (1200, 630), "black")

    _draw_avatar(
        canvas,
        _png("white"),
        color=True,
        light=False,
        flip=False,
        vertical=False,
        display_name="Test",
    )

    bounds = canvas.convert("L").getbbox()
    assert bounds is not None
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    assert abs(width - height) <= 2


def test_vertical_quote_uses_a_dedicated_four_by_five_layout() -> None:
    result = QuoteImageService().render(
        QuoteRenderRequest(
            text="縦の引用",
            display_name="しまじろーど",
            username="simajilord",
            avatar=_png("white"),
            vertical=True,
            bold=True,
        )
    )

    assert (result.width, result.height) == (768, 960)
    with Image.open(io.BytesIO(result.content)) as image:
        assert image.size == (768, 960)


def test_quote_text_shrinks_to_fit_the_visible_line_budget() -> None:
    layout = _fit_text(
        (
            "日本語・English・123 の実生成テスト\n"
            "絵文字も同じ行に描画し、長文でも自動的に文字サイズを調整します。"
        ),
        maximum_width=490,
        maximum_height=400,
        maximum_lines=4,
        custom_images={},
        bold=False,
    )

    assert layout.font_size < 58
    assert len(layout.lines) <= 4
    assert layout.truncated is False


def test_fixed_strike_unicode_emoji_is_scaled_inside_the_text_line() -> None:
    emoji_font = _emoji_font(36)
    assert emoji_font is not None

    image = _unicode_emoji_image("😀", emoji_font, size=39)

    assert image is not None
    assert image.width <= 39
    assert image.height <= 39


def test_sticker_only_quote_renders_the_sticker_instead_of_its_name() -> None:
    result = QuoteImageService().render(
        QuoteRenderRequest(
            text="",
            display_name="しまじろーど",
            username="simajilord",
            avatar=_png("#cc8844"),
            stickers=(
                QuoteStickerAsset(
                    sticker_id="333333333333333333",
                    name="animeteo",
                    content=_png("purple", size=(320, 180)),
                ),
            ),
            color=True,
        )
    )

    assert result.rendered_stickers == 1
    assert result.text_truncated is False
    assert result.animated is False


def test_animated_custom_emoji_quote_is_encoded_as_a_gif_on_request() -> None:
    result = QuoteImageService().render(
        QuoteRenderRequest(
            text="<a:dance:123456789012345678>",
            display_name="しまじろーど",
            username="simajilord",
            avatar=_png("#cc8844"),
            custom_emojis=(
                QuoteCustomEmojiAsset(
                    emoji_id="123456789012345678",
                    name="dance",
                    content=_gif(),
                ),
            ),
            animate=True,
        )
    )

    assert result.content.startswith(b"GIF8")
    assert result.animated is True
    with Image.open(io.BytesIO(result.content)) as image:
        assert image.n_frames >= 2


def test_animated_quote_preserves_static_detail_and_later_frame_colors() -> None:
    gradient = Image.linear_gradient("L").resize((630, 630))
    avatar = io.BytesIO()
    Image.merge("RGB", (gradient, gradient, gradient)).save(avatar, format="PNG")
    service = QuoteImageService()
    common = {
        "text": "",
        "display_name": "しまじろーど",
        "username": "simajilord",
        "avatar": avatar.getvalue(),
        "stickers": (
            QuoteStickerAsset(
                sticker_id="333333333333333333",
                name="animeteo",
                content=_gif(),
            ),
        ),
    }

    static = service.render(QuoteRenderRequest(**common))
    animated = service.render(QuoteRenderRequest(**common, animate=True))

    with (
        Image.open(io.BytesIO(static.content)) as static_image,
        Image.open(io.BytesIO(animated.content)) as animated_image,
    ):
        assert animated_image.n_frames == 2
        first_frame = animated_image.convert("RGB")
        difference = ImageChops.difference(
            static_image.convert("RGB"),
            first_frame,
        )
        assert max(ImageStat.Stat(difference).mean) < 3
        animated_image.seek(1)
        later_frame = animated_image.convert("RGB")
        red, green, blue = later_frame.getpixel((850, 260))
        assert blue > red * 2
        assert blue > green * 2


def test_invalid_custom_emoji_falls_back_to_its_readable_name() -> None:
    result = QuoteImageService().render(
        QuoteRenderRequest(
            text="<:broken:123456789012345678>",
            display_name="Test",
            username="test",
            avatar=None,
            custom_emojis=(
                QuoteCustomEmojiAsset(
                    emoji_id="123456789012345678",
                    name="broken",
                    content=b"not-an-image",
                ),
            ),
        )
    )

    assert result.rendered_custom_emojis == 0
    assert result.content.startswith(b"\x89PNG\r\n\x1a\n")
