"""Local-first SearXNG provider with bounded candidate collection."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import aiohttp

from simajilord.core.errors import WebError
from simajilord.domain.web import WebSearchOptions, WebSearchResult, WebSource

_MAX_SEARCH_RESPONSE_BYTES = 512_000
_MAX_BACKEND_REQUESTS = 16
_TRACKING_PARAMETERS = frozenset(
    {"fbclid", "gclid", "yclid", "mc_cid", "mc_eid"}
)


@dataclass(frozen=True, slots=True)
class _SearchPage:
    sources: tuple[WebSource, ...]
    warnings: tuple[str, ...]


class SearxngSearchProvider:
    """Search a private or loopback SearXNG JSON endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        shared_secret: str | None = None,
    ) -> None:
        self.base_url = _normalize_backend_url(base_url)
        self.timeout_seconds = timeout_seconds
        self.shared_secret = shared_secret.strip() if shared_secret else None
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return "searxng"

    async def search(
        self,
        query: str,
        options: WebSearchOptions,
    ) -> WebSearchResult:
        searched_queries = _planned_queries(query, options)
        time_ranges = _effective_time_ranges(options)
        request_count = len(searched_queries) * len(time_ranges) * options.pages
        if request_count > _MAX_BACKEND_REQUESTS:
            raise WebError(
                "request_too_broad",
                "Search request is too broad. Reduce domains, pages, or recency breadth.",
            )

        collected: list[WebSource] = []
        warnings: list[str] = []
        searched_pages: list[int] = []
        successful_requests = 0
        try:
            async with asyncio.timeout(
                min(30.0, max(5.0, self.timeout_seconds * 2))
            ):
                for planned_query in searched_queries:
                    for time_range in time_ranges:
                        for page in range(
                            options.start_page,
                            options.start_page + options.pages,
                        ):
                            try:
                                payload = await self._fetch_page(
                                    query=planned_query,
                                    page=page,
                                    time_range=time_range,
                                    options=options,
                                )
                            except WebError as exc:
                                warnings.append(_bounded_warning(exc.technical_detail))
                                continue
                            successful_requests += 1
                            searched_pages.append(page)
                            collected.extend(payload.sources)
                            warnings.extend(payload.warnings)
        except TimeoutError as exc:
            if not collected:
                raise WebError("timeout", "Search timed out.") from exc
            warnings.append("Search collection stopped at the time limit.")

        if successful_requests == 0 and not collected:
            raise WebError("search_unavailable", "The local search backend is unavailable.")

        filtered = tuple(
            source for source in collected if _source_allowed(source, options)
        )
        deduplicated = _deduplicate_sources(filtered)[: options.candidate_limit]
        displayed = _diversify_sources(
            deduplicated,
            per_host_limit=options.per_host_limit,
        )[: options.display_limit]
        host_counts = Counter(source.host for source in deduplicated)
        maybe_more = _maybe_more_results(
            raw_count=len(collected),
            candidate_count=len(deduplicated),
            displayed_count=len(displayed),
            host_counts=host_counts,
        )
        return WebSearchResult(
            query=query,
            backend=self.name,
            sources=displayed,
            raw_candidate_count=len(collected),
            candidate_count=len(deduplicated),
            maybe_more=maybe_more,
            warnings=_unique_strings(warnings)[:8],
            searched_queries=searched_queries,
            searched_pages=tuple(dict.fromkeys(searched_pages)),
        )

    async def status(self) -> tuple[bool, str]:
        endpoint = _backend_health_endpoint(self.base_url)
        try:
            async with self._client().get(
                endpoint,
                allow_redirects=False,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=min(2.0, self.timeout_seconds)),
            ) as response:
                if response.status == 200:
                    return True, "ready"
                return False, f"HTTP {response.status}"
        except (TimeoutError, aiohttp.ClientError):
            return False, "unreachable"

    async def close(self) -> None:
        session = self._session
        self._session = None
        if session is not None:
            await session.close()

    async def _fetch_page(
        self,
        *,
        query: str,
        page: int,
        time_range: str | None,
        options: WebSearchOptions,
    ) -> _SearchPage:
        parameters: list[tuple[str, str]] = [
            ("q", query),
            ("format", "json"),
            ("pageno", str(page)),
            ("safesearch", str(options.safesearch)),
        ]
        if options.categories:
            parameters.append(("categories", ",".join(options.categories)))
        if options.engines:
            parameters.append(("engines", ",".join(options.engines)))
        if options.language:
            parameters.append(("language", options.language))
        if time_range:
            parameters.append(("time_range", time_range))
        try:
            async with self._client().get(
                _backend_search_endpoint(self.base_url),
                params=parameters,
                allow_redirects=False,
                headers=self._headers(),
            ) as response:
                if response.status != 200:
                    raise WebError(
                        "search_backend_error",
                        f"SearXNG returned HTTP {response.status}.",
                    )
                content_length = _nonnegative_integer(
                    response.headers.get("Content-Length")
                )
                if (
                    content_length is not None
                    and content_length > _MAX_SEARCH_RESPONSE_BYTES
                ):
                    raise WebError(
                        "search_response_too_large",
                        "SearXNG response is too large.",
                    )
                body = await _read_bounded(
                    response,
                    max_bytes=_MAX_SEARCH_RESPONSE_BYTES,
                )
        except WebError:
            raise
        except TimeoutError as exc:
            raise WebError("timeout", "SearXNG request timed out.") from exc
        except aiohttp.ClientError as exc:
            raise WebError(
                "search_unavailable",
                "SearXNG could not be reached.",
            ) from exc
        try:
            payload: object = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebError(
                "search_invalid_response",
                "SearXNG returned invalid JSON.",
            ) from exc
        if not isinstance(payload, dict):
            raise WebError(
                "search_invalid_response",
                "SearXNG returned invalid JSON.",
            )
        raw_results = payload.get("results")
        results = raw_results if isinstance(raw_results, list) else []
        sources = tuple(
            source
            for index, raw_result in enumerate(results)
            if (source := _source_from_result(raw_result, f"S{page}-{index + 1}"))
            is not None
        )
        raw_warnings = payload.get("unresponsive_engines")
        warnings = (
            tuple(
                warning
                for raw_warning in raw_warnings
                if (warning := _engine_warning(raw_warning)) is not None
            )
            if isinstance(raw_warnings, list)
            else ()
        )
        return _SearchPage(sources=sources, warnings=warnings)

    def _client(self) -> aiohttp.ClientSession:
        session = self._session
        if session is not None and not session.closed:
            return session
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
            connector=aiohttp.TCPConnector(limit=4),
            auto_decompress=True,
        )
        self._session = session
        return session

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "SimajilordSearch/0.1 (+https://simajilord.com)",
        }
        if self.shared_secret:
            headers["Authorization"] = f"Bearer {self.shared_secret}"
            headers["X-Simaji-Search-Key"] = self.shared_secret
        return headers


def _normalize_backend_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    try:
        parsed = urlsplit(normalized)
    except ValueError as exc:
        raise ValueError("WEB_SEARCH_BASE_URL is invalid.") from exc
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not host:
        raise ValueError("WEB_SEARCH_BASE_URL must be an HTTP(S) URL.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("WEB_SEARCH_BASE_URL must not contain credentials.")
    if parsed.scheme == "http" and host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("Remote WEB_SEARCH_BASE_URL values must use HTTPS.")
    return normalized


def _backend_search_endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if not path.endswith("/search"):
        path = f"{path}/search"
    return urlunsplit(parsed._replace(path=path, query="", fragment=""))


def _backend_health_endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url)
    return urlunsplit(parsed._replace(path="/healthz", query="", fragment=""))


def _planned_queries(query: str, options: WebSearchOptions) -> tuple[str, ...]:
    file_parts = tuple(f"filetype:{file_type}" for file_type in options.file_types)
    suffix = " ".join(file_parts)
    if options.allowed_domains:
        return _unique_strings(
            tuple(
                " ".join(part for part in (query, f"site:{domain}", suffix) if part)
                for domain in options.allowed_domains
            )
        )
    return (" ".join((query, *file_parts)).strip(),)


def _effective_time_ranges(options: WebSearchOptions) -> tuple[str | None, ...]:
    if options.time_range:
        return (options.time_range,)
    return ("year", None) if options.prefer_recent else (None,)


def _source_from_result(raw_result: object, source_id: str) -> WebSource | None:
    if not isinstance(raw_result, dict):
        return None
    raw_url = raw_result.get("url")
    if not isinstance(raw_url, str):
        return None
    try:
        parsed = urlsplit(raw_url)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    title = _bounded_text(raw_result.get("title"), 180) or raw_url
    snippet = _bounded_text(raw_result.get("content"), 600)
    category = _bounded_text(raw_result.get("category"), 40) or "web"
    engine = _bounded_text(raw_result.get("engine"), 48) or None
    published_at = (
        _bounded_text(raw_result.get("publishedDate"), 80)
        or _bounded_text(raw_result.get("published_date"), 80)
        or None
    )
    return WebSource(
        source_id=source_id[:24],
        title=title,
        url=raw_url[:1_200],
        host=parsed.hostname.lower().removeprefix("www.")[:120],
        snippet=snippet,
        category=category,
        engine=engine,
        published_at=published_at,
    )


def _engine_warning(raw_warning: object) -> str | None:
    if isinstance(raw_warning, list) and raw_warning:
        engine = _bounded_text(raw_warning[0], 48)
        reason = _bounded_text(raw_warning[1] if len(raw_warning) > 1 else None, 120)
        return f"{engine}: {reason or 'unresponsive'}" if engine else None
    if isinstance(raw_warning, dict):
        engine = _bounded_text(raw_warning.get("engine"), 48)
        reason = _bounded_text(raw_warning.get("error_type"), 120)
        return f"{engine}: {reason or 'unresponsive'}" if engine else None
    return None


def _source_allowed(source: WebSource, options: WebSearchOptions) -> bool:
    host = source.host.lower()
    if options.allowed_domains and not any(
        _domain_matches(host, domain) for domain in options.allowed_domains
    ):
        return False
    return not any(_domain_matches(host, domain) for domain in options.blocked_domains)


def _domain_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def _deduplicate_sources(sources: tuple[WebSource, ...]) -> tuple[WebSource, ...]:
    output: list[WebSource] = []
    seen: set[str] = set()
    for source in sources:
        key = _canonical_source_key(source.url)
        if key in seen:
            continue
        seen.add(key)
        output.append(source)
    return tuple(output)


def _diversify_sources(
    sources: tuple[WebSource, ...],
    *,
    per_host_limit: int,
) -> tuple[WebSource, ...]:
    counts: Counter[str] = Counter()
    output: list[WebSource] = []
    for source in sources:
        if counts[source.host] >= per_host_limit:
            continue
        counts[source.host] += 1
        output.append(source)
    return tuple(output)


def _canonical_source_key(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value.lower()
    query = tuple(
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in _TRACKING_PARAMETERS
    )
    normalized = urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            urlencode(query, doseq=True),
            "",
        )
    )
    return normalized.lower()


def _maybe_more_results(
    *,
    raw_count: int,
    candidate_count: int,
    displayed_count: int,
    host_counts: Counter[str],
) -> bool:
    safe_displayed_count = max(1, displayed_count)
    return (
        raw_count >= safe_displayed_count * 5
        or candidate_count >= safe_displayed_count * 3
        or any(
            count >= max(3, safe_displayed_count // 2)
            for count in host_counts.values()
        )
    )


async def _read_bounded(
    response: aiohttp.ClientResponse,
    *,
    max_bytes: int,
) -> str:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.content.iter_chunked(64 * 1_024):
        total += len(chunk)
        if total > max_bytes:
            response.close()
            raise WebError(
                "search_response_too_large",
                "SearXNG response is too large.",
            )
        chunks.append(bytes(chunk))
    return b"".join(chunks).decode("utf-8")


def _unique_strings(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = value.strip()
        lowered = normalized.lower()
        if not normalized or lowered in seen:
            continue
        seen.add(lowered)
        output.append(normalized)
    return tuple(output)


def _bounded_text(value: object, maximum: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:maximum]


def _bounded_warning(value: str) -> str:
    return " ".join(value.split())[:200] or "Search backend request failed."


def _nonnegative_integer(value: str | None) -> int | None:
    if value is None:
        return None
    with suppress(ValueError):
        number = int(value)
        return number if number >= 0 else None
    return None
