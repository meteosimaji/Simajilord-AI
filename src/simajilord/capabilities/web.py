"""Transport-neutral web Search, Fetch, Find, and status endpoints."""

from __future__ import annotations

from dataclasses import dataclass

from simajilord.core.capabilities import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.domain.web import SearchDepth, WebSource, WebTextMatch
from simajilord.services.web import WebService


@dataclass(frozen=True, slots=True)
class WebSearchRequest:
    query: str
    depth: SearchDepth = SearchDepth.STANDARD
    categories: tuple[str, ...] = ("general",)
    engines: tuple[str, ...] = ()
    language: str | None = None
    time_range: str | None = None
    file_types: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    prefer_recent: bool = True
    safesearch: int = 1


@dataclass(frozen=True, slots=True)
class WebSearchResponse:
    query: str
    backend: str
    sources: tuple[WebSource, ...]
    raw_candidate_count: int
    candidate_count: int
    maybe_more: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WebFetchRequest:
    url: str
    offset: int = 0
    max_characters: int = 1_600
    include_links: bool = False


@dataclass(frozen=True, slots=True)
class WebFetchResponse:
    title: str
    url: str
    content_type: str
    text: str
    offset: int
    total_characters: int
    next_offset: int | None
    links: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WebFindRequest:
    url: str
    pattern: str
    max_matches: int = 5
    context_characters: int = 120


@dataclass(frozen=True, slots=True)
class WebFindResponse:
    title: str
    url: str
    pattern: str
    matches: tuple[WebTextMatch, ...]
    total_matches: int


@dataclass(frozen=True, slots=True)
class WebStatusRequest:
    pass


@dataclass(frozen=True, slots=True)
class WebStatusResponse:
    ready: bool
    backend: str
    detail: str


def build_web_endpoints(web: WebService) -> tuple[CapabilityEndpoint, ...]:
    async def search(
        request: WebSearchRequest,
        _: InvocationContext,
    ) -> WebSearchResponse:
        options = web.search_options(
            depth=request.depth,
            categories=request.categories,
            engines=request.engines,
            language=request.language,
            time_range=request.time_range,
            file_types=request.file_types,
            allowed_domains=request.allowed_domains,
            blocked_domains=request.blocked_domains,
            prefer_recent=request.prefer_recent,
            safesearch=request.safesearch,
        )
        result = await web.search(request.query, options)
        return WebSearchResponse(
            query=result.query,
            backend=result.backend,
            sources=result.sources,
            raw_candidate_count=result.raw_candidate_count,
            candidate_count=result.candidate_count,
            maybe_more=result.maybe_more,
            warnings=result.warnings,
        )

    async def fetch(
        request: WebFetchRequest,
        _: InvocationContext,
    ) -> WebFetchResponse:
        if request.offset < 0:
            raise UserError("web.offset_invalid")
        if not 200 <= request.max_characters <= 6_000:
            raise UserError("web.chunk_limit_invalid")
        page = await web.page(request.url)
        if request.offset > len(page.text):
            raise UserError("web.offset_invalid")
        end = min(len(page.text), request.offset + request.max_characters)
        return WebFetchResponse(
            title=page.title,
            url=page.final_url,
            content_type=page.content_type,
            text=page.text[request.offset:end],
            offset=request.offset,
            total_characters=len(page.text),
            next_offset=end if end < len(page.text) else None,
            links=page.links[:20] if request.include_links else (),
        )

    async def find(
        request: WebFindRequest,
        _: InvocationContext,
    ) -> WebFindResponse:
        page, matches, total_matches = await web.find(
            url=request.url,
            pattern=request.pattern,
            max_matches=request.max_matches,
            context_characters=request.context_characters,
        )
        return WebFindResponse(
            title=page.title,
            url=page.final_url,
            pattern=" ".join(request.pattern.split()).strip(),
            matches=matches,
            total_matches=total_matches,
        )

    async def status(
        _: WebStatusRequest,
        __: InvocationContext,
    ) -> WebStatusResponse:
        ready, backend, detail = await web.status()
        return WebStatusResponse(ready=ready, backend=backend, detail=detail)

    return (
        endpoint(
            CapabilityDescriptor(
                name="web.search",
                summary=(
                    "Search the web through the local metasearch provider with bounded "
                    "candidate collection, source diversity, domain filters, and recency."
                ),
                risk=RiskLevel.EXTERNAL,
                keywords=(
                    "web",
                    "search",
                    "sources",
                    "news",
                    "research",
                    "recent",
                ),
                side_effects=("Sends a query to configured search engines.",),
            ),
            WebSearchRequest,
            WebSearchResponse,
            search,
        ),
        endpoint(
            CapabilityDescriptor(
                name="web.fetch",
                summary=(
                    "Open one public URL and return a bounded readable text chunk with "
                    "an offset for continuation."
                ),
                risk=RiskLevel.EXTERNAL,
                keywords=("web", "fetch", "open", "read", "page", "pdf", "url"),
                side_effects=("Fetches one public HTTP or HTTPS resource.",),
            ),
            WebFetchRequest,
            WebFetchResponse,
            fetch,
        ),
        endpoint(
            CapabilityDescriptor(
                name="web.find",
                summary=(
                    "Find a phrase inside one public page and return bounded surrounding "
                    "passages."
                ),
                risk=RiskLevel.EXTERNAL,
                keywords=("web", "find", "page", "phrase", "match", "context"),
                side_effects=("Fetches one public HTTP or HTTPS resource when uncached.",),
            ),
            WebFindRequest,
            WebFindResponse,
            find,
        ),
        endpoint(
            CapabilityDescriptor(
                name="web.status",
                summary="Check whether the local metasearch provider is reachable.",
                risk=RiskLevel.READ,
                keywords=("web", "search", "status", "health", "provider"),
            ),
            WebStatusRequest,
            WebStatusResponse,
            status,
        ),
    )
