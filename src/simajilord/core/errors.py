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


class MediaError(ProviderError):
    """A media operation failed with a stable provider-neutral category."""

    def __init__(self, category: str, technical_detail: str = "") -> None:
        super().__init__(technical_detail or category)
        self.category = category
        self.technical_detail = technical_detail
