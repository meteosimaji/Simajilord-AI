from __future__ import annotations

import pytest

from simajilord.core.errors import MediaError, UserError
from simajilord.media.providers.yt_dlp import YtDlpProvider, classify_yt_dlp_error
from simajilord.media.security import (
    normalize_media_query,
    normalize_media_reference,
    validate_media_url,
    validate_public_media_url,
)


@pytest.mark.parametrize(
    "url",
    (
        "https://youtube.com/watch?v=abc",
        "https://www.youtube.com/watch?v=abc",
        "https://youtu.be/abc",
        "https://www.tiktok.com/@user/video/1",
        "https://vimeo.com/123",
        "https://x.com/user/status/1",
        "https://media.example.org/watch/1",
        "https://youtube.com.example/video",
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
        "https://[::1]/video",
        "https://localhost/video",
        "https://intranet/video",
    ),
)
def test_unsafe_media_urls_are_rejected(url: str) -> None:
    with pytest.raises(UserError) as captured:
        validate_media_url(url)
    assert captured.value.code == "media.url_unsupported"


def test_plain_query_becomes_bounded_youtube_search() -> None:
    assert normalize_media_reference("  example song  ") == "ytsearch1:example song"


def test_search_query_rejects_urls() -> None:
    assert normalize_media_query("  example song  ") == "example song"
    with pytest.raises(UserError) as captured:
        normalize_media_query("https://example.com/watch")
    assert captured.value.code == "media.query_url_not_allowed"


@pytest.mark.asyncio
async def test_public_url_dns_boundary_rejects_mixed_private_answers(
    monkeypatch,
) -> None:
    class FakeLoop:
        async def getaddrinfo(
            self,
            host: str,
            port: int,
            *,
            type: int,
        ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
            assert host == "media.example.org"
            assert port == 443
            assert type
            return [
                (2, 1, 6, "", ("203.0.113.10", 443)),
                (2, 1, 6, "", ("10.0.0.4", 443)),
            ]

    monkeypatch.setattr(
        "simajilord.media.security.asyncio.get_running_loop",
        lambda: FakeLoop(),
    )
    with pytest.raises(UserError) as captured:
        await validate_public_media_url("https://media.example.org/watch")
    assert captured.value.code == "media.url_private"


@pytest.mark.asyncio
async def test_provider_search_returns_bounded_transport_neutral_candidates(
    monkeypatch,
) -> None:
    captured_options: dict[str, object] = {}

    class FakeYoutubeDL:
        def __init__(self, options: dict[str, object]) -> None:
            captured_options.update(options)

        def __enter__(self) -> FakeYoutubeDL:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def extract_info(self, reference: str, *, download: bool) -> dict[str, object]:
            assert reference == "ytsearch2:example song"
            assert download is False
            return {
                "entries": [
                    {
                        "id": "first",
                        "title": "Artist - Example Song",
                        "duration": 120,
                        "uploader": "Artist",
                        "thumbnail": "https://img.example.org/first.jpg",
                    },
                    {
                        "id": "second",
                        "title": "Another - Example Song",
                        "duration": 121,
                        "uploader": "Another",
                    },
                ]
            }

    monkeypatch.setattr(
        "simajilord.media.providers.yt_dlp.yt_dlp.YoutubeDL",
        FakeYoutubeDL,
    )
    provider = YtDlpProvider(cookie_file=None, download_timeout_seconds=30)
    results = await provider.search_audio("example song", limit=2)
    assert [item.title for item in results] == [
        "Artist - Example Song",
        "Another - Example Song",
    ]
    assert results[0].reference == "https://www.youtube.com/watch?v=first"
    assert results[0].uploader == "Artist"
    assert captured_options["allowed_extractors"] == ["default", "-generic"]


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
