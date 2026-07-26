"""Provider boundary for synthetic-media analysis."""

from __future__ import annotations

from typing import Protocol

from simajilord.domain.moderation import SyntheticMediaProviderResult


class SyntheticMediaProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def analyze(
        self,
        *,
        content: bytes,
        filename: str,
        content_type: str,
        threshold: float,
    ) -> SyntheticMediaProviderResult: ...

    async def close(self) -> None: ...
