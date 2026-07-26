from __future__ import annotations

from dataclasses import replace

import pytest

from simajilord.capabilities.web import (
    WebFetchRequest,
    WebFetchResponse,
    WebFindRequest,
    WebFindResponse,
    WebSearchRequest,
    WebSearchResponse,
    WebStatusRequest,
    WebStatusResponse,
    build_web_endpoints,
)
from simajilord.core import CapabilityRegistry, InvocationContext
from simajilord.core.errors import WebError
from simajilord.domain.web import (
    FetchedWebResource,
    SearchDepth,
    WebSearchOptions,
    WebSearchResult,
    WebSource,
)
from simajilord.providers.web import AiohttpPublicWebFetcher, normalize_public_web_url
from simajilord.providers.web import searxng as searxng_module
from simajilord.providers.web.searxng import SearxngSearchProvider
from simajilord.services.web import WebService


class FakeSearchProvider:
    name = "fake-search"

    def __init__(self) -> None:
        self.calls: list[tuple[str, WebSearchOptions]] = []
        self.closed = False

    async def search(
        self,
        query: str,
        options: WebSearchOptions,
    ) -> WebSearchResult:
        self.calls.append((query, options))
        return WebSearchResult(
            query=query,
            backend=self.name,
            sources=(
                WebSource(
                    source_id="S1",
                    title="Result",
                    url="https://example.com/result",
                    host="example.com",
                    snippet="A useful result.",
                    category="general",
                ),
            ),
            raw_candidate_count=1,
            candidate_count=1,
            maybe_more=False,
            warnings=(),
            searched_queries=(query,),
            searched_pages=(1,),
        )

    async def status(self) -> tuple[bool, str]:
        return True, "ready"

    async def close(self) -> None:
        self.closed = True


class FakePageFetcher:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.closed = False

    async def fetch(
        self,
        url: str,
        *,
        max_bytes: int,
    ) -> FetchedWebResource:
        assert max_bytes == 2_000_000
        self.calls.append(url)
        repeated = "Useful phrase and readable context. " * 20
        return FetchedWebResource(
            final_url="https://example.com/final",
            content_type="text/html",
            charset="utf-8",
            body=(
                "<html><head><title>Example page</title>"
                "<script>ignore this phrase</script></head>"
                f"<body><p>{repeated}</p>"
                '<a href="/next">Next page</a></body></html>'
            ).encode(),
        )

    async def close(self) -> None:
        self.closed = True


def _service() -> tuple[WebService, FakeSearchProvider, FakePageFetcher]:
    search_provider = FakeSearchProvider()
    page_fetcher = FakePageFetcher()
    return (
        WebService(
            search_provider=search_provider,
            page_fetcher=page_fetcher,
            max_fetch_bytes=2_000_000,
        ),
        search_provider,
        page_fetcher,
    )


@pytest.mark.asyncio
async def test_web_search_infers_japanese_and_reuses_short_cache() -> None:
    service, search_provider, _ = _service()
    options = service.search_options(depth=SearchDepth.QUICK)

    first = await service.search("こんにちは", options)
    second = await service.search("こんにちは", options)

    assert first == second
    assert len(search_provider.calls) == 1
    assert search_provider.calls[0][1].language == "ja"
    await service.close()


@pytest.mark.asyncio
async def test_fetch_chunks_and_find_reuse_one_safe_page_fetch() -> None:
    service, _, page_fetcher = _service()
    registry = CapabilityRegistry()
    for capability in build_web_endpoints(service):
        registry.register(capability)
    context = InvocationContext("actor", "workspace", "test", "request")

    fetched = await registry.invoke(
        "web.fetch",
        WebFetchRequest(
            url="https://example.com/start",
            max_characters=200,
            include_links=True,
        ),
        context,
    )
    assert isinstance(fetched, WebFetchResponse)
    assert fetched.title == "Example page"
    assert "Example page" not in fetched.text
    assert "ignore this phrase" not in fetched.text
    assert fetched.next_offset == 200
    assert fetched.links == ("https://example.com/next",)

    found = await registry.invoke(
        "web.find",
        WebFindRequest(
            url="https://example.com/start",
            pattern="Useful phrase",
            max_matches=3,
        ),
        context,
    )
    assert isinstance(found, WebFindResponse)
    assert found.total_matches == 20
    assert len(found.matches) == 3
    assert page_fetcher.calls == ["https://example.com/start"]
    await service.close()


@pytest.mark.asyncio
async def test_web_capability_endpoints_share_one_typed_service_boundary() -> None:
    service, search_provider, _ = _service()
    registry = CapabilityRegistry()
    for capability in build_web_endpoints(service):
        registry.register(capability)
    context = InvocationContext("actor", "workspace", "test", "request")

    searched = await registry.invoke(
        "web.search",
        WebSearchRequest(query="reusable search", depth=SearchDepth.QUICK),
        context,
    )
    status = await registry.invoke(
        "web.status",
        WebStatusRequest(),
        context,
    )

    assert isinstance(searched, WebSearchResponse)
    assert searched.sources[0].title == "Result"
    assert search_provider.calls[0][0] == "reusable search"
    assert isinstance(status, WebStatusResponse)
    assert status == WebStatusResponse(
        ready=True,
        backend="fake-search",
        detail="ready",
    )
    assert {item.descriptor.name for item in registry.all()} == {
        "web.search",
        "web.fetch",
        "web.find",
        "web.status",
    }
    await service.close()


@pytest.mark.parametrize(
    "url",
    (
        "http://127.0.0.1/",
        "https://[::1]/",
        "http://10.0.0.8/",
        "https://localhost/",
        "https://service.internal/",
        "https://user:password@example.com/",
        "https://example.com:8443/",
        "file:///etc/passwd",
    ),
)
def test_public_web_url_boundary_rejects_host_network_and_credentials(url: str) -> None:
    with pytest.raises(WebError):
        normalize_public_web_url(url)


def test_public_web_url_boundary_keeps_public_url_and_removes_fragment() -> None:
    assert normalize_public_web_url("https://example.com/page?q=1#private") == (
        "https://example.com/page?q=1"
    )


def test_public_fetcher_can_be_composed_before_an_event_loop_starts() -> None:
    fetcher = AiohttpPublicWebFetcher(timeout_seconds=5)
    assert fetcher.timeout_seconds == 5


@pytest.mark.asyncio
async def test_searxng_provider_deduplicates_tracking_urls_and_diversifies_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SearxngSearchProvider(
        base_url="http://127.0.0.1:8888",
        timeout_seconds=5,
    )
    options = WebSearchOptions(
        allowed_domains=(),
        blocked_domains=(),
        candidate_limit=10,
        categories=("general",),
        display_limit=5,
        engines=(),
        file_types=(),
        language="en",
        pages=1,
        per_host_limit=1,
        prefer_recent=False,
        safesearch=1,
        start_page=1,
        time_range=None,
    )
    sources = (
        WebSource(
            "S1",
            "First",
            "https://example.com/page?utm_source=test",
            "example.com",
            "one",
            "general",
        ),
        WebSource(
            "S2",
            "Duplicate",
            "https://example.com/page",
            "example.com",
            "two",
            "general",
        ),
        WebSource(
            "S3",
            "Same host",
            "https://example.com/other",
            "example.com",
            "three",
            "general",
        ),
        WebSource(
            "S4",
            "Other host",
            "https://other.example.org/page",
            "other.example.org",
            "four",
            "general",
        ),
    )

    async def fake_fetch_page(**_: object) -> object:
        return searxng_module._SearchPage(sources=sources, warnings=("engine: slow",))

    monkeypatch.setattr(provider, "_fetch_page", fake_fetch_page)
    result = await provider.search("topic", options)

    assert result.candidate_count == 3
    assert [source.host for source in result.sources] == [
        "example.com",
        "other.example.org",
    ]
    assert result.warnings == ("engine: slow",)
    assert result.searched_pages == (1,)
    await provider.close()


@pytest.mark.asyncio
async def test_search_request_budget_rejects_deep_two_domain_recent_search() -> None:
    service, _, _ = _service()
    options = service.search_options(
        depth=SearchDepth.DEEP,
        allowed_domains=("example.com", "example.org"),
    )
    provider = SearxngSearchProvider(
        base_url="http://127.0.0.1:8888",
        timeout_seconds=5,
    )
    with pytest.raises(WebError) as captured:
        await provider.search("topic", options)
    assert captured.value.category == "request_too_broad"
    await provider.close()
    await service.close()


def test_search_options_are_immutable_and_hashable_for_cache_keys() -> None:
    service, _, _ = _service()
    options = service.search_options(depth=SearchDepth.STANDARD)
    assert hash(options)
    assert replace(options, language="en").language == "en"
