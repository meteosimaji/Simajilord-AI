"""Errors that cross Simajilord service and integration boundaries."""


class ConfigurationError(RuntimeError):
    """The local runtime configuration is invalid."""


class UserError(RuntimeError):
    """A structured, expected failure that a transport presenter can localize."""

    def __init__(self, code: str, **details: object) -> None:
        super().__init__(code)
        self.code = code
        self.details = details


class CapabilityError(RuntimeError):
    """A capability could not be registered or invoked."""


class ProviderError(RuntimeError):
    """An external or local provider failed."""


class EarlyPlaybackEnd(ProviderError):
    """A duration-known audio stream ended substantially before its expected end."""

    def __init__(self, *, elapsed_seconds: float, expected_seconds: float) -> None:
        super().__init__(
            f"Audio ended after {elapsed_seconds:.2f}s; expected {expected_seconds:.2f}s."
        )
        self.elapsed_seconds = elapsed_seconds
        self.expected_seconds = expected_seconds


class MediaError(ProviderError):
    """A media operation failed with a stable provider-neutral category."""

    def __init__(self, category: str, technical_detail: str = "") -> None:
        super().__init__(technical_detail or category)
        self.category = category
        self.technical_detail = technical_detail


class WebError(ProviderError):
    """A web capability failed with a stable provider-neutral category."""

    def __init__(self, category: str, technical_detail: str = "") -> None:
        super().__init__(technical_detail or category)
        self.category = category
        self.technical_detail = technical_detail


class ModerationError(ProviderError):
    """A media-analysis provider failed with a stable provider-neutral category."""

    def __init__(
        self,
        category: str,
        technical_detail: str = "",
        *,
        http_status: int | None = None,
    ) -> None:
        super().__init__(technical_detail or category)
        self.category = category
        self.technical_detail = technical_detail
        self.http_status = http_status
