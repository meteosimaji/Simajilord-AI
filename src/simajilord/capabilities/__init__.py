"""Independent application endpoints available to every adapter."""

from .audio import (
    AudioControlRequest,
    AudioControlResponse,
    AudioPlayRequest,
    AudioPlayResponse,
    AudioQueueRequest,
    AudioQueueResponse,
    build_audio_endpoints,
)
from .media import DownloadRequest, DownloadResponse, build_download_endpoint
from .read_aloud import (
    ReadAloudRequest,
    ReadAloudResponse,
    build_read_aloud_endpoint,
)
from .system import (
    CapabilitySearchRequest,
    CapabilitySearchResponse,
    PingRequest,
    PingResponse,
    UptimeRequest,
    UptimeResponse,
    build_system_endpoints,
)
from .utility import (
    ChooseRequest,
    ChooseResponse,
    RollRequest,
    RollResponse,
    build_utility_endpoints,
)

__all__ = [
    "AudioControlRequest",
    "AudioControlResponse",
    "AudioPlayRequest",
    "AudioPlayResponse",
    "AudioQueueRequest",
    "AudioQueueResponse",
    "CapabilitySearchRequest",
    "CapabilitySearchResponse",
    "ChooseRequest",
    "ChooseResponse",
    "DownloadRequest",
    "DownloadResponse",
    "PingRequest",
    "PingResponse",
    "ReadAloudRequest",
    "ReadAloudResponse",
    "RollRequest",
    "RollResponse",
    "UptimeRequest",
    "UptimeResponse",
    "build_audio_endpoints",
    "build_download_endpoint",
    "build_read_aloud_endpoint",
    "build_system_endpoints",
    "build_utility_endpoints",
]
