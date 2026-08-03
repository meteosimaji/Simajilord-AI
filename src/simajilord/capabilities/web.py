"""Transport-neutral web Search, Fetch, Find, and status endpoints."""

from __future__ import annotations

from dataclasses import dataclass

from simajilord.core.capabilities import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    DisclosureClass,
    EgressDescriptor,
    EgressFieldKind,
    EgressSinkAudience,
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
    complete: bool
    source_truncated: bool


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
    source_truncated: bool


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
        next_offset = end if end < len(page.text) else None
        return WebFetchResponse(
            title=page.title,
            url=page.final_url,
            content_type=page.content_type,
            text=page.text[request.offset:end],
            offset=request.offset,
            total_characters=len(page.text),
            next_offset=next_offset,
            links=page.links[:20] if request.include_links else (),
            complete=next_offset is None and not page.source_truncated,
            source_truncated=page.source_truncated,
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
            source_truncated=page.source_truncated,
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
                    "Search the web through the local metasearch service with "
                    "bounded sources, domains, and time ranges."
                ),
                risk=RiskLevel.EXTERNAL,
                disclosure_class=DisclosureClass.EXTERNAL_PUBLIC,
                keywords=(
                    "web",
                    "search",
                    "sources",
                    "news",
                    "research",
                    "recent",
                ),
                side_effects=("Sends the query to configured search engines.",),
                audit_payload="metadata",
                egress=EgressDescriptor(
                    provider="configured_web_search",
                    field_kinds=(EgressFieldKind.QUERY,),
                    request_fields=(
                        "query",
                        "depth",
                        "categories",
                        "engines",
                        "language",
                        "time_range",
                        "file_types",
                        "allowed_domains",
                        "blocked_domains",
                        "prefer_recent",
                        "safesearch",
                    ),
                    sink_audience=EgressSinkAudience.EXTERNAL_PUBLIC,
                ),
            ),
            WebSearchRequest,
            WebSearchResponse,
            search,
        ),
        endpoint(
            CapabilityDescriptor(
                name="web.fetch",
                summary=(
                    "Fetch readable text from one public URL in bounded chunks. "
                    "Use next_offset to continue. If source_truncated is true, import "
                    "the URL with files.download_url and use paginated files.read."
                ),
                risk=RiskLevel.EXTERNAL,
                disclosure_class=DisclosureClass.EXTERNAL_PUBLIC,
                keywords=("web", "fetch", "open", "read", "page", "pdf", "url"),
                side_effects=("Fetches one public HTTP or HTTPS resource.",),
                audit_payload="metadata",
                egress=EgressDescriptor(
                    provider="public_web",
                    field_kinds=(EgressFieldKind.URL,),
                    request_fields=("url",),
                    sink_audience=EgressSinkAudience.EXTERNAL_PUBLIC,
                ),
            ),
            WebFetchRequest,
            WebFetchResponse,
            fetch,
        ),
        endpoint(
            CapabilityDescriptor(
                name="web.find",
                summary=(
                    "Find a phrase in the extracted portion of a public page and return "
                    "bounded context. source_truncated reports an incomplete source."
                ),
                risk=RiskLevel.EXTERNAL,
                disclosure_class=DisclosureClass.EXTERNAL_PUBLIC,
                keywords=("web", "find", "page", "phrase", "match", "context"),
                side_effects=(
                    "Fetches one public HTTP or HTTPS resource when not cached.",
                ),
                audit_payload="metadata",
                egress=EgressDescriptor(
                    provider="public_web",
                    field_kinds=(EgressFieldKind.URL,),
                    request_fields=("url",),
                    sink_audience=EgressSinkAudience.EXTERNAL_PUBLIC,
                ),
            ),
            WebFindRequest,
            WebFindResponse,
            find,
        ),
        endpoint(
            CapabilityDescriptor(
                name="web.status",
                summary="Check readiness of the local metasearch service.",
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.NO_USER_CONTENT,
                keywords=("web", "search", "status", "health", "provider"),
            ),
            WebStatusRequest,
            WebStatusResponse,
            status,
        ),
    )
