"""Provider ports for search engines and public page retrieval."""

from __future__ import annotations

from typing import Protocol

from simajilord.domain.web import FetchedWebResource, WebSearchOptions, WebSearchResult


class WebSearchProvider(Protocol):
    """Search backend selected by the composition root."""

    @property
    def name(self) -> str: ...

    async def search(
        self,
        query: str,
        options: WebSearchOptions,
    ) -> WebSearchResult: ...

    async def status(self) -> tuple[bool, str]: ...

    async def close(self) -> None: ...


class PublicWebFetcher(Protocol):
    """Fetch public HTTP resources without exposing the host network."""

    async def fetch(
        self,
        url: str,
        *,
        max_bytes: int,
    ) -> FetchedWebResource: ...

    async def close(self) -> None: ...
