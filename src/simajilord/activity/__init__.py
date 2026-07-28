"""Official Discord Activity transport for read-only audio state."""

from .server import ActivityServer, build_activity_snapshot

__all__ = ["ActivityServer", "build_activity_snapshot"]
