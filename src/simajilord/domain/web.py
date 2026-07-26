"""Transport-neutral models for bounded web search and reading."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SearchDepth(StrEnum):
    """How broadly the local search provider should collect candidates."""

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


@dataclass(frozen=True, slots=True)
class WebSource:
    """One normalized source returned by a search or page operation."""

    source_id: str
    title: str
    url: str
    host: str
    snippet: str
    category: str
    engine: str | None = None
    published_at: str | None = None


@dataclass(frozen=True, slots=True)
class WebSearchOptions:
    """Normalized provider options with bounded fan-out."""

    allowed_domains: tuple[str, ...]
    blocked_domains: tuple[str, ...]
    candidate_limit: int
    categories: tuple[str, ...]
    display_limit: int
    engines: tuple[str, ...]
    file_types: tuple[str, ...]
    language: str | None
    pages: int
    per_host_limit: int
    prefer_recent: bool
    safesearch: int
    start_page: int
    time_range: str | None


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    """Search result plus enough diagnostics to avoid blind retries."""

    query: str
    backend: str
    sources: tuple[WebSource, ...]
    raw_candidate_count: int
    candidate_count: int
    maybe_more: bool
    warnings: tuple[str, ...]
    searched_queries: tuple[str, ...]
    searched_pages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FetchedWebResource:
    """Bounded response bytes returned by the public-network provider."""

    final_url: str
    content_type: str
    charset: str | None
    body: bytes


@dataclass(frozen=True, slots=True)
class ReadableWebPage:
    """Extracted readable page cached inside the platform service."""

    final_url: str
    title: str
    content_type: str
    text: str
    links: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WebTextMatch:
    """One bounded, contextual match inside a fetched page."""

    before: str
    match: str
    after: str
