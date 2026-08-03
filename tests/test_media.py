from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path

import pytest

from simajilord.agent.tools import AgentToolCatalog
from simajilord.capabilities.file_scope import file_workspace_id
from simajilord.capabilities.media import (
    MediaSaveRequest,
    MediaSaveResponse,
    build_media_save_endpoint,
)
from simajilord.core import InvocationContext
from simajilord.core.capabilities import CapabilityRegistry
from simajilord.core.errors import MediaError, UserError
from simajilord.domain.audio import AudioItem
from simajilord.domain.media import (
    DownloadArtifact,
    DownloadBatch,
    DownloadFormat,
    MediaCandidate,
)
from simajilord.media.providers.yt_dlp import YtDlpProvider, classify_yt_dlp_error
from simajilord.media.security import (
    normalize_media_query,
    normalize_media_reference,
    validate_media_url,
    validate_public_media_url,
)
from simajilord.services.files import AgentFileSandbox
from simajilord.services.media import MediaPriority, MediaService


class TrackingMediaProvider:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.active_by_workspace: dict[str, int] = defaultdict(int)
        self.max_by_workspace: dict[str, int] = defaultdict(int)
        self.started: list[str] = []
        self.block_started = asyncio.Event()
        self.release_block = asyncio.Event()

    async def search_audio(
        self,
        query: str,
        *,
        limit: int,
    ) -> tuple[MediaCandidate, ...]:
        workspace_id = query.split(":", maxsplit=1)[0]
        self.active += 1
        self.active_by_workspace[workspace_id] += 1
        self.max_active = max(self.max_active, self.active)
        self.max_by_workspace[workspace_id] = max(
            self.max_by_workspace[workspace_id],
            self.active_by_workspace[workspace_id],
        )
        self.started.append(query)
        try:
            if query == "block:first":
                self.block_started.set()
                await self.release_block.wait()
            else:
                await asyncio.sleep(0.005)
            return (
                MediaCandidate(
                    reference=f"https://example.test/{query}",
                    title=query,
                    duration_seconds=float(limit),
                ),
            )
        finally:
            self.active -= 1
            self.active_by_workspace[workspace_id] -= 1

    async def resolve_audio(self, reference: str) -> AudioItem:
        raise NotImplementedError

    async def mix_audio(
        self,
        seed_references: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[MediaCandidate, ...]:
        raise NotImplementedError

    async def download(
        self,
        url: str,
        media_type: DownloadFormat,
        destination: Path,
        *,
        max_bytes: int,
    ) -> DownloadArtifact:
        raise NotImplementedError


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
            **options: int,
        ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
            assert host == "media.example.org"
            assert port == 443
            assert options["type"]
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


@pytest.mark.asyncio
async def test_provider_combines_multiple_youtube_mix_seeds_without_stream_resolution(
    monkeypatch,
) -> None:
    captured: list[tuple[str, dict[str, object]]] = []

    class FakeYoutubeDL:
        def __init__(self, options: dict[str, object]) -> None:
            self.options = options

        def __enter__(self) -> FakeYoutubeDL:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def extract_info(self, reference: str, *, download: bool) -> dict[str, object]:
            captured.append((reference, self.options))
            assert download is False
            seed = "seed-a" if "RDseed-a" in reference else "seed-b"
            other = "seed-b" if seed == "seed-a" else "seed-a"
            return {
                "entries": [
                    {"id": seed, "title": f"Seed {seed}"},
                    {"id": f"{seed}-one", "title": f"{seed} one"},
                    {"id": "shared", "title": "Shared candidate"},
                    {"id": f"{other}-one", "title": f"{other} one"},
                ]
            }

    monkeypatch.setattr(
        "simajilord.media.providers.yt_dlp.yt_dlp.YoutubeDL",
        FakeYoutubeDL,
    )
    provider = YtDlpProvider(cookie_file=None, download_timeout_seconds=30)
    results = await provider.mix_audio(
        (
            "https://www.youtube.com/watch?v=seed-a",
            "https://youtu.be/seed-b",
        ),
        limit=5,
    )

    assert [item.reference for item in results] == [
        "https://www.youtube.com/watch?v=seed-a-one",
        "https://www.youtube.com/watch?v=seed-b-one",
        "https://www.youtube.com/watch?v=shared",
    ]
    assert len(captured) == 2
    assert all(options["extract_flat"] == "in_playlist" for _, options in captured)
    assert all(options["skip_download"] is True for _, options in captured)
    assert all("format" not in options for _, options in captured)


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


@pytest.mark.asyncio
async def test_provider_downloads_multiple_post_media_with_one_generic_extractor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = YtDlpProvider(cookie_file=None, download_timeout_seconds=30)
    captured_commands: list[list[str]] = []

    async def public_url(value: str) -> str:
        return value

    async def run(
        command: list[str],
        *,
        environment: dict[str, str],
    ) -> tuple[int, str]:
        del environment
        captured_commands.append(command)
        destination = Path(command[command.index("--paths") + 1])
        (destination / "post_01_[one].mp4").write_bytes(b"first")
        (destination / "post_02_[two].mp4").write_bytes(b"second")
        return 0, ""

    monkeypatch.setattr(
        "simajilord.media.providers.yt_dlp.validate_public_media_url",
        public_url,
    )
    monkeypatch.setattr(provider, "_run_download_process", run)

    batch = await provider.download_many(
        "https://x.com/example/status/1",
        DownloadFormat.VIDEO,
        tmp_path / "download",
        max_bytes=1_000,
        max_items=4,
    )

    assert [item.path.name for item in batch.artifacts] == [
        "post_01_[one].mp4",
        "post_02_[two].mp4",
    ]
    assert batch.partial is False
    command = captured_commands[0]
    assert command[command.index("--playlist-end") + 1] == "4"
    assert "--no-playlist" not in command
    assert "--ignore-config" in command
    assert "--no-cache-dir" in command
    selector = command[command.index("--format") + 1]
    assert "filesize_approx<=1000" in selector
    assert "height<=720" in selector
    assert command[-1] == "https://x.com/example/status/1"


@pytest.mark.asyncio
async def test_provider_bounds_unexpected_extra_artifacts_and_scrubs_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = YtDlpProvider(cookie_file=None, download_timeout_seconds=30)
    captured_environment: dict[str, str] = {}

    async def public_url(value: str) -> str:
        return value

    async def run(
        command: list[str],
        *,
        environment: dict[str, str],
    ) -> tuple[int, str]:
        captured_environment.update(environment)
        destination = Path(command[command.index("--paths") + 1])
        for index in range(3):
            (destination / f"post_{index:02d}.mp4").write_bytes(
                str(index).encode()
            )
        return 0, ""

    monkeypatch.setenv("DISCORD_TOKEN", "must-not-reach-downloader")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-downloader")
    monkeypatch.setattr(
        "simajilord.media.providers.yt_dlp.validate_public_media_url",
        public_url,
    )
    monkeypatch.setattr(provider, "_run_download_process", run)

    batch = await provider.download_many(
        "https://example.test/post/1",
        DownloadFormat.VIDEO,
        tmp_path / "download",
        max_bytes=1_000,
        max_items=2,
    )

    assert len(batch.artifacts) == 2
    assert batch.skipped_items == 1
    assert batch.partial is True
    assert "DISCORD_TOKEN" not in captured_environment
    assert "OPENAI_API_KEY" not in captured_environment


@pytest.mark.asyncio
async def test_provider_retries_one_transient_extractor_challenge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = YtDlpProvider(cookie_file=None, download_timeout_seconds=30)
    attempts = 0

    async def public_url(value: str) -> str:
        return value

    async def run(
        command: list[str],
        *,
        environment: dict[str, str],
    ) -> tuple[int, str]:
        nonlocal attempts
        del environment
        attempts += 1
        if attempts == 1:
            return 1, "Unable to extract universal data for rehydration"
        destination = Path(command[command.index("--paths") + 1])
        (destination / "clip_00_[id].mp4").write_bytes(b"usable")
        return 0, ""

    monkeypatch.setattr(
        "simajilord.media.providers.yt_dlp.validate_public_media_url",
        public_url,
    )
    monkeypatch.setattr(provider, "_run_download_process", run)

    batch = await provider.download_many(
        "https://www.tiktok.com/@example/video/1",
        DownloadFormat.VIDEO,
        tmp_path / "download",
        max_bytes=1_000,
        max_items=1,
    )

    assert attempts == 2
    assert batch.artifacts[0].path.read_bytes() == b"usable"


@pytest.mark.asyncio
async def test_provider_bounds_repeated_transient_extractor_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = YtDlpProvider(cookie_file=None, download_timeout_seconds=30)
    attempts = 0

    async def public_url(value: str) -> str:
        return value

    async def run(
        command: list[str],
        *,
        environment: dict[str, str],
    ) -> tuple[int, str]:
        nonlocal attempts
        del command, environment
        attempts += 1
        return 1, "Unable to extract universal data for rehydration"

    monkeypatch.setattr(
        "simajilord.media.providers.yt_dlp.validate_public_media_url",
        public_url,
    )
    monkeypatch.setattr(provider, "_run_download_process", run)

    with pytest.raises(MediaError) as error:
        await provider.download_many(
            "https://www.tiktok.com/@example/video/1",
            DownloadFormat.VIDEO,
            tmp_path / "download",
            max_bytes=1_000,
            max_items=1,
        )

    assert attempts == 3
    assert error.value.category == "extractor_challenge"


@pytest.mark.asyncio
async def test_media_save_creates_content_addressed_workspace_files(
    tmp_path: Path,
) -> None:
    class FakeMedia:
        async def download_many(
            self,
            url: str,
            media_type: DownloadFormat,
            destination: Path,
            *,
            max_bytes: int,
            max_items: int,
            workspace_id: str,
            priority: MediaPriority,
        ) -> DownloadBatch:
            assert max_bytes == 100
            assert max_items == 4
            assert workspace_id == "guild"
            assert priority is MediaPriority.NORMAL
            first = destination / "first.mp4"
            second = destination / "second.mp4"
            first.write_bytes(b"first media")
            second.write_bytes(b"second media")
            return DownloadBatch(
                artifacts=(
                    DownloadArtifact(
                        path=first,
                        title="first",
                        media_type=media_type,
                        source_url=url,
                        size_bytes=first.stat().st_size,
                    ),
                    DownloadArtifact(
                        path=second,
                        title="second",
                        media_type=media_type,
                        source_url=url,
                        size_bytes=second.stat().st_size,
                    ),
                )
            )

    files = AgentFileSandbox(tmp_path / "files", max_file_bytes=100)
    capability = build_media_save_endpoint(FakeMedia(), files)  # type: ignore[arg-type]
    context = InvocationContext("actor", "guild", "agent", "event")
    result = await capability.invoke(
        MediaSaveRequest("https://video.example/post"),
        context,
    )

    assert isinstance(result, MediaSaveResponse)
    assert len(result.files) == 2
    assert all(item.path.startswith("media/") for item in result.files)
    assert [item.size_bytes for item in result.files] == [11, 12]
    assert files.path_for_delivery(
        file_workspace_id(context), result.files[0].path
    ).read_bytes() == b"first media"


@pytest.mark.asyncio
async def test_media_save_is_discoverable_without_platform_specific_tools(
    tmp_path: Path,
) -> None:
    files = AgentFileSandbox(tmp_path / "files")
    registry = CapabilityRegistry()
    registry.register(
        build_media_save_endpoint(
            object(),  # type: ignore[arg-type]
            files,
        )
    )
    catalog = AgentToolCatalog(
        registry,
        ("media.save",),
        required_grants={"media.save": "media"},
        eager_capabilities=(),
        write_capabilities=("media.save",),
    )

    output = await catalog.invoke(
        namespace="simajilord",
        tool_name="capability_search",
        arguments={"query": "SNSの動画を保存したい", "limit": 3},
        context=InvocationContext(
            "actor",
            "guild",
            "agent",
            "event",
            grants=frozenset({"media"}),
        ),
        max_output_characters=4_000,
    )

    assert '"name":"media.save"' in output.text
    assert "media.save_x" not in output.text
    assert "media.save_tiktok" not in output.text


@pytest.mark.asyncio
async def test_media_scheduler_bounds_eight_guild_load_fairly() -> None:
    provider = TrackingMediaProvider()
    media = MediaService(
        provider,
        max_concurrent=3,
        max_per_workspace=1,
    )
    try:
        results = await asyncio.gather(
            *(
                media.search_audio(
                    f"guild-{guild}:{round_index}",
                    limit=1,
                    workspace_id=f"guild-{guild}",
                    priority=MediaPriority.NORMAL,
                )
                for round_index in range(2)
                for guild in range(8)
            )
        )
    finally:
        await media.close()

    assert len(results) == 16
    assert provider.max_active == 3
    assert set(provider.max_by_workspace) == {
        f"guild-{guild}" for guild in range(8)
    }
    assert set(provider.max_by_workspace.values()) == {1}


@pytest.mark.asyncio
async def test_interactive_media_work_overtakes_queued_background_work() -> None:
    provider = TrackingMediaProvider()
    media = MediaService(
        provider,
        max_concurrent=1,
        max_per_workspace=1,
    )
    blocker = asyncio.create_task(
        media.search_audio(
            "block:first",
            limit=1,
            workspace_id="block",
            priority=MediaPriority.BACKGROUND,
        )
    )
    await provider.block_started.wait()
    background = asyncio.create_task(
        media.search_audio(
            "background:second",
            limit=1,
            workspace_id="background",
            priority=MediaPriority.BACKGROUND,
        )
    )
    interactive = asyncio.create_task(
        media.search_audio(
            "interactive:first",
            limit=1,
            workspace_id="interactive",
            priority=MediaPriority.INTERACTIVE,
        )
    )
    await asyncio.sleep(0)
    provider.release_block.set()
    try:
        await asyncio.gather(blocker, background, interactive)
    finally:
        await media.close()

    assert provider.started == [
        "block:first",
        "interactive:first",
        "background:second",
    ]
