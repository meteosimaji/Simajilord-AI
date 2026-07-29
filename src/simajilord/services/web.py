"""Reusable Search, Fetch, and Find orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import io
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import TypeVar
from urllib.parse import urljoin, urlsplit

from pypdf import PdfReader

from simajilord.core.errors import UserError, WebError
from simajilord.domain.web import (
    FetchedWebResource,
    ReadableWebPage,
    SearchDepth,
    WebSearchOptions,
    WebSearchResult,
    WebTextMatch,
)
from simajilord.providers.web.base import PublicWebFetcher, WebSearchProvider
from simajilord.providers.web.http import normalize_public_web_url

_SEARCH_PRESETS = {
    SearchDepth.QUICK: (30, 8, 1),
    SearchDepth.STANDARD: (80, 10, 4),
    SearchDepth.DEEP: (160, 20, 8),
}
_MAX_PAGE_TEXT_CHARACTERS = 200_000
_MAX_PAGE_LINKS = 40
_MAX_WEB_PDF_PAGES = 200
_MAX_CACHE_ENTRIES = 32
_CACHE_TTL_SECONDS = 300.0
_BLOCK_ELEMENTS = frozenset(
    {
        "article",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }
)
_SKIPPED_ELEMENTS = frozenset({"script", "style", "noscript", "svg", "template"})
_CacheKey = TypeVar("_CacheKey")
_CacheValue = TypeVar("_CacheValue")


@dataclass(frozen=True, slots=True)
class _CachedPage:
    loaded_at: float
    page: ReadableWebPage


@dataclass(frozen=True, slots=True)
class _CachedSearch:
    loaded_at: float
    result: WebSearchResult


class WebService:
    """One platform authority for search, page reading, and page-local find."""

    def __init__(
        self,
        *,
        search_provider: WebSearchProvider,
        page_fetcher: PublicWebFetcher,
        max_fetch_bytes: int,
    ) -> None:
        self.search_provider = search_provider
        self.page_fetcher = page_fetcher
        self.max_fetch_bytes = max_fetch_bytes
        self._search_cache: dict[tuple[str, WebSearchOptions], _CachedSearch] = {}
        self._page_cache: dict[str, _CachedPage] = {}
        self._cache_lock = asyncio.Lock()
        self._search_semaphore = asyncio.Semaphore(2)
        self._fetch_semaphore = asyncio.Semaphore(4)

    def search_options(
        self,
        *,
        depth: SearchDepth,
        categories: tuple[str, ...] = ("general",),
        engines: tuple[str, ...] = (),
        language: str | None = None,
        time_range: str | None = None,
        file_types: tuple[str, ...] = (),
        allowed_domains: tuple[str, ...] = (),
        blocked_domains: tuple[str, ...] = (),
        prefer_recent: bool = True,
        safesearch: int = 1,
    ) -> WebSearchOptions:
        candidate_limit, display_limit, pages = _SEARCH_PRESETS[depth]
        normalized_language = _normalize_search_token(language or "", maximum=16)
        normalized_time_range = _normalize_search_token(
            time_range or "",
            maximum=16,
        ).lower()
        if normalized_time_range not in {"", "day", "month", "year"}:
            raise UserError("web.time_range_invalid")
        if not 0 <= safesearch <= 2:
            raise UserError("web.safesearch_invalid")
        normalized_categories = _normalize_tokens(categories, maximum=6, length=32)
        return WebSearchOptions(
            allowed_domains=_normalize_domains(allowed_domains, maximum=2),
            blocked_domains=_normalize_domains(blocked_domains, maximum=12),
            candidate_limit=candidate_limit,
            categories=normalized_categories or ("general",),
            display_limit=display_limit,
            engines=_normalize_tokens(engines, maximum=8, length=48),
            file_types=tuple(
                token.lower()
                for token in _normalize_tokens(file_types, maximum=4, length=16)
            ),
            language=normalized_language or None,
            pages=pages,
            per_host_limit=3,
            prefer_recent=prefer_recent,
            safesearch=safesearch,
            start_page=1,
            time_range=normalized_time_range or None,
        )

    async def search(
        self,
        query: str,
        options: WebSearchOptions,
    ) -> WebSearchResult:
        normalized_query = _normalize_query(query)
        if options.language is None:
            inferred_language = _infer_language(normalized_query)
            if inferred_language is not None:
                options = WebSearchOptions(
                    allowed_domains=options.allowed_domains,
                    blocked_domains=options.blocked_domains,
                    candidate_limit=options.candidate_limit,
                    categories=options.categories,
                    display_limit=options.display_limit,
                    engines=options.engines,
                    file_types=options.file_types,
                    language=inferred_language,
                    pages=options.pages,
                    per_host_limit=options.per_host_limit,
                    prefer_recent=options.prefer_recent,
                    safesearch=options.safesearch,
                    start_page=options.start_page,
                    time_range=options.time_range,
                )
        cache_key = (normalized_query, options)
        cached = await self._cached_search(cache_key)
        if cached is not None:
            return cached
        async with self._search_semaphore:
            cached = await self._cached_search(cache_key)
            if cached is not None:
                return cached
            result = await self.search_provider.search(normalized_query, options)
            async with self._cache_lock:
                self._search_cache[cache_key] = _CachedSearch(
                    loaded_at=time.monotonic(),
                    result=result,
                )
                _trim_cache(self._search_cache)
            return result

    async def page(self, url: str) -> ReadableWebPage:
        normalized_url = normalize_public_web_url(url)
        cached = await self._cached_page(normalized_url)
        if cached is not None:
            return cached
        async with self._fetch_semaphore:
            cached = await self._cached_page(normalized_url)
            if cached is not None:
                return cached
            resource = await self.page_fetcher.fetch(
                normalized_url,
                max_bytes=self.max_fetch_bytes,
            )
            page = await _readable_page(resource)
            async with self._cache_lock:
                cached_page = _CachedPage(loaded_at=time.monotonic(), page=page)
                self._page_cache[normalized_url] = cached_page
                self._page_cache[page.final_url] = cached_page
                _trim_cache(self._page_cache)
            return page

    async def find(
        self,
        *,
        url: str,
        pattern: str,
        max_matches: int,
        context_characters: int,
    ) -> tuple[ReadableWebPage, tuple[WebTextMatch, ...], int]:
        normalized_pattern = " ".join(pattern.split()).strip()
        if not normalized_pattern:
            raise UserError("web.pattern_required")
        if len(normalized_pattern) > 300:
            raise UserError("web.pattern_too_long")
        if not 1 <= max_matches <= 10:
            raise UserError("web.match_limit_invalid")
        if not 40 <= context_characters <= 300:
            raise UserError("web.context_limit_invalid")
        page = await self.page(url)
        all_matches = _find_text_matches(
            page.text,
            normalized_pattern,
            context_characters=context_characters,
        )
        return page, all_matches[:max_matches], len(all_matches)

    async def status(self) -> tuple[bool, str, str]:
        ready, detail = await self.search_provider.status()
        return ready, self.search_provider.name, detail

    async def close(self) -> None:
        await self.page_fetcher.close()
        await self.search_provider.close()

    async def _cached_search(
        self,
        key: tuple[str, WebSearchOptions],
    ) -> WebSearchResult | None:
        async with self._cache_lock:
            cached = self._search_cache.get(key)
            if cached is None:
                return None
            if time.monotonic() - cached.loaded_at > _CACHE_TTL_SECONDS:
                self._search_cache.pop(key, None)
                return None
            return cached.result

    async def _cached_page(self, url: str) -> ReadableWebPage | None:
        async with self._cache_lock:
            cached = self._page_cache.get(url)
            if cached is None:
                return None
            if time.monotonic() - cached.loaded_at > _CACHE_TTL_SECONDS:
                self._page_cache.pop(url, None)
                return None
            return cached.page


class _ReadableHTMLParser(HTMLParser):
    def __init__(self, *, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.links: list[str] = []
        self._seen_links: set[str] = set()
        self._skipped_depth = 0
        self._title_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        lowered = tag.lower()
        if lowered in _SKIPPED_ELEMENTS:
            self._skipped_depth += 1
            return
        if self._skipped_depth:
            return
        if lowered == "title":
            self._title_depth += 1
        if lowered in _BLOCK_ELEMENTS:
            self.parts.append("\n")
        if lowered == "a" and len(self.links) < _MAX_PAGE_LINKS:
            href = next((value for name, value in attrs if name.lower() == "href"), None)
            if href:
                candidate = urljoin(self.base_url, href)
                try:
                    parsed = urlsplit(candidate)
                except ValueError:
                    return
                if (
                    parsed.scheme in {"http", "https"}
                    and candidate not in self._seen_links
                ):
                    self._seen_links.add(candidate)
                    self.links.append(candidate)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in _SKIPPED_ELEMENTS:
            self._skipped_depth = max(0, self._skipped_depth - 1)
            return
        if self._skipped_depth:
            return
        if lowered == "title":
            self._title_depth = max(0, self._title_depth - 1)
        if lowered in _BLOCK_ELEMENTS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skipped_depth or not data.strip():
            return
        if self._title_depth:
            self.title_parts.append(data)
            return
        self.parts.append(data)


async def _readable_page(resource: FetchedWebResource) -> ReadableWebPage:
    body = resource.body
    content_type = resource.content_type.lower()
    final_url = resource.final_url
    extension = urlsplit(final_url).path.rsplit(".", maxsplit=1)[-1].lower()
    if content_type == "application/pdf" or body.startswith(b"%PDF-") or extension == "pdf":
        text, title, source_truncated = await asyncio.to_thread(
            _pdf_text,
            body,
            final_url,
        )
        return ReadableWebPage(
            final_url=final_url,
            title=title,
            content_type="application/pdf",
            text=text[:_MAX_PAGE_TEXT_CHARACTERS],
            links=(),
            source_truncated=source_truncated,
        )
    if not _textual_content(content_type, extension):
        raise WebError(
            "content_type_unsupported",
            f"Unsupported content type: {content_type or 'unknown'}.",
        )
    decoded = _decode_body(body, resource.charset)
    if _html_content(content_type, extension):
        parser = _ReadableHTMLParser(base_url=final_url)
        try:
            parser.feed(decoded)
            parser.close()
        except (AssertionError, ValueError) as exc:
            raise WebError("content_invalid", "The page HTML could not be parsed.") from exc
        text = _normalize_readable_text(" ".join(parser.parts))
        title = _normalize_line(" ".join(parser.title_parts))[:180]
        return ReadableWebPage(
            final_url=final_url,
            title=title or _fallback_title(final_url),
            content_type=content_type,
            text=text[:_MAX_PAGE_TEXT_CHARACTERS],
            links=tuple(parser.links),
            source_truncated=len(text) > _MAX_PAGE_TEXT_CHARACTERS,
        )
    text = _normalize_readable_text(decoded)
    return ReadableWebPage(
        final_url=final_url,
        title=_fallback_title(final_url),
        content_type=content_type,
        text=text[:_MAX_PAGE_TEXT_CHARACTERS],
        links=(),
        source_truncated=len(text) > _MAX_PAGE_TEXT_CHARACTERS,
    )


def _pdf_text(body: bytes, final_url: str) -> tuple[str, str, bool]:
    try:
        reader = PdfReader(io.BytesIO(body), strict=False)
        if reader.is_encrypted:
            with contextlib.suppress(Exception):
                reader.decrypt("")
        parts: list[str] = []
        extracted_characters = 0
        source_truncated = False
        total_pages = len(reader.pages)
        page_limit = min(total_pages, _MAX_WEB_PDF_PAGES)
        for index in range(page_limit):
            page = reader.pages[index]
            page_text = page.extract_text() or ""
            parts.append(page_text)
            extracted_characters += len(page_text)
            if extracted_characters >= _MAX_PAGE_TEXT_CHARACTERS:
                source_truncated = index + 1 < total_pages
                break
        else:
            source_truncated = page_limit < total_pages
        metadata_title = (
            str(reader.metadata.title).strip()
            if reader.metadata is not None and reader.metadata.title
            else ""
        )
    except Exception as exc:
        raise WebError("content_invalid", "The PDF could not be read.") from exc
    text = _normalize_readable_text("\n".join(parts))
    if not text:
        raise WebError("content_empty", "The PDF has no extractable text.")
    source_truncated = source_truncated or len(text) > _MAX_PAGE_TEXT_CHARACTERS
    return (
        text,
        metadata_title[:180] or _fallback_title(final_url),
        source_truncated,
    )


def _decode_body(body: bytes, declared_charset: str | None) -> str:
    candidates = [declared_charset] if declared_charset else []
    header = body[:4_096].decode("ascii", errors="ignore")
    match = re.search(r"charset\s*=\s*[\"']?([a-zA-Z0-9._-]+)", header, re.I)
    if match:
        candidates.append(match.group(1))
    candidates.extend(("utf-8", "cp932", "shift_jis", "iso2022_jp"))
    for charset in candidates:
        if not charset:
            continue
        try:
            return body.decode(charset)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", errors="replace")


def _textual_content(content_type: str, extension: str) -> bool:
    return (
        content_type.startswith("text/")
        or content_type
        in {
            "application/json",
            "application/ld+json",
            "application/xhtml+xml",
            "application/xml",
        }
        or extension in {"", "htm", "html", "json", "txt", "xml"}
    )


def _html_content(content_type: str, extension: str) -> bool:
    return (
        "html" in content_type
        or "xml" in content_type
        or extension in {"", "htm", "html", "xml"}
    )


def _find_text_matches(
    text: str,
    pattern: str,
    *,
    context_characters: int,
) -> tuple[WebTextMatch, ...]:
    lowered_text = text.casefold()
    lowered_pattern = pattern.casefold()
    output: list[WebTextMatch] = []
    cursor = 0
    while cursor < len(text):
        index = lowered_text.find(lowered_pattern, cursor)
        if index < 0:
            break
        match_end = index + len(pattern)
        output.append(
            WebTextMatch(
                before=text[max(0, index - context_characters) : index],
                match=text[index:match_end],
                after=text[match_end : match_end + context_characters],
            )
        )
        cursor = max(match_end, index + 1)
    return tuple(output)


def _normalize_query(value: str) -> str:
    normalized = " ".join(value.split()).strip()
    if not normalized:
        raise UserError("web.query_required")
    if len(normalized) > 500:
        raise UserError("web.query_too_long")
    return normalized


def _normalize_tokens(
    values: tuple[str, ...],
    *,
    maximum: int,
    length: int,
) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_search_token(value, maximum=length)
        lowered = normalized.lower()
        if not normalized or lowered in seen:
            continue
        seen.add(lowered)
        output.append(normalized)
        if len(output) >= maximum:
            break
    return tuple(output)


def _normalize_search_token(value: str, *, maximum: int) -> str:
    return "".join(
        character
        for character in value.strip()
        if character.isalnum() or character in "._:-"
    )[:maximum]


def _normalize_domains(values: tuple[str, ...], *, maximum: int) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        candidate = raw_value.strip().lower()
        if "://" not in candidate:
            candidate = f"https://{candidate}"
        try:
            parsed = urlsplit(candidate)
        except ValueError:
            continue
        host = (parsed.hostname or "").rstrip(".")
        if not host or "." not in host or host in seen:
            continue
        try:
            ascii_host = host.encode("idna").decode("ascii")
        except UnicodeError:
            continue
        seen.add(ascii_host)
        output.append(ascii_host)
        if len(output) >= maximum:
            break
    return tuple(output)


def _infer_language(query: str) -> str | None:
    if any("\u3040" <= character <= "\u30ff" for character in query):
        return "ja"
    if any("\uac00" <= character <= "\ud7a3" for character in query):
        return "ko"
    return None


def _normalize_readable_text(value: str) -> str:
    lines = tuple(
        line
        for raw_line in value.replace("\r", "\n").split("\n")
        if (line := _normalize_line(raw_line))
    )
    return "\n".join(lines)


def _normalize_line(value: str) -> str:
    return " ".join(value.split()).strip()


def _fallback_title(url: str) -> str:
    parsed = urlsplit(url)
    tail = parsed.path.rstrip("/").rsplit("/", maxsplit=1)[-1]
    return tail or (parsed.hostname or "Web page")


def _trim_cache(cache: dict[_CacheKey, _CacheValue]) -> None:
    while len(cache) > _MAX_CACHE_ENTRIES:
        cache.pop(next(iter(cache)))
