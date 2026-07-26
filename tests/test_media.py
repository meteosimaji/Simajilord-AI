from __future__ import annotations

import pytest

from simajilord.core.errors import MediaError, UserError
from simajilord.media.providers.yt_dlp import classify_yt_dlp_error
from simajilord.media.security import normalize_media_reference, validate_media_url


@pytest.mark.parametrize(
    "url",
    (
        "https://youtube.com/watch?v=abc",
        "https://www.youtube.com/watch?v=abc",
        "https://youtu.be/abc",
        "https://www.tiktok.com/@user/video/1",
    ),
)
def test_supported_media_urls(url: str) -> None:
    assert validate_media_url(url) == url


@pytest.mark.parametrize(
    "url",
    (
        "http://youtube.com/watch?v=abc",
        "https://youtube.com:444/watch?v=abc",
        "https://user:pass@youtube.com/watch?v=abc",
        "https://127.0.0.1/video",
        "https://youtube.com.example/video",
    ),
)
def test_unsafe_media_urls_are_rejected(url: str) -> None:
    with pytest.raises(UserError) as captured:
        validate_media_url(url)
    assert captured.value.code == "media.url_unsupported"


def test_plain_query_becomes_bounded_youtube_search() -> None:
    assert normalize_media_reference("  example song  ") == "ytsearch1:example song"


@pytest.mark.parametrize(
    ("detail", "category"),
    (
        ("Sign in to confirm your age; cookies required", "cookie_required"),
        ("HTTP Error 429: Too Many Requests", "rate_limited"),
        ("Unsupported URL", "unsupported"),
        ("Private video", "unavailable"),
        ("larger than max-filesize", "too_large"),
    ),
)
def test_provider_errors_have_stable_categories(detail: str, category: str) -> None:
    error: MediaError = classify_yt_dlp_error(detail)
    assert error.category == category
