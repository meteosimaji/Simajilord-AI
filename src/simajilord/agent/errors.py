"""Stable agent error categories."""

from __future__ import annotations


class AgentError(RuntimeError):
    """Base class for failures safe to translate at a transport boundary."""


class AgentUnavailableError(AgentError):
    """The configured provider or its authentication is unavailable."""


class AgentBusyError(AgentError):
    """A bounded concurrency gate rejected an additional request."""


class AgentRateLimitError(AgentError):
    """A local cost or spam budget rejected the request before model use."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class AgentProviderError(AgentError):
    """The provider failed while creating or running a turn."""


class AgentProviderLimitError(AgentProviderError):
    """The upstream provider rejected a turn because its usage limit was reached."""


class AgentThreadError(AgentProviderError):
    """A saved provider thread could not be resumed."""


class AgentToolError(AgentError):
    """A dynamic tool request was invalid, denied, or exceeded its budget."""
