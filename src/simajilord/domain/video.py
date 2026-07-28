"""Transport-neutral models for encoded real-time video."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class VideoCodec(StrEnum):
    """Codecs supported by the first Simajilord video transport."""

    H264 = "h264"
    VP8 = "vp8"


@dataclass(frozen=True, slots=True)
class VideoProfile:
    """Bounded encoding and RTP parameters for one video sender."""

    codec: VideoCodec = VideoCodec.H264
    width: int = 854
    height: int = 480
    frame_rate: int = 15
    bitrate_kbps: int = 1_200
    keyframe_interval_seconds: int = 2
    payload_type: int = 101
    clock_rate: int = 90_000
    mtu: int = 1_200

    def __post_init__(self) -> None:
        if not 16 <= self.width <= 3_840 or not 16 <= self.height <= 2_160:
            raise ValueError("Video dimensions must be between 16px and 3840x2160.")
        if not 1 <= self.frame_rate <= 60:
            raise ValueError("Video frame rate must be between 1 and 60.")
        if not 64 <= self.bitrate_kbps <= 20_000:
            raise ValueError("Video bitrate must be between 64 and 20000 kbps.")
        if not 1 <= self.keyframe_interval_seconds <= 10:
            raise ValueError("Keyframe interval must be between 1 and 10 seconds.")
        if not 96 <= self.payload_type <= 127:
            raise ValueError("Video RTP payload type must be dynamic (96-127).")
        if self.clock_rate != 90_000:
            raise ValueError("Video RTP uses a 90000 Hz clock.")
        if not 576 <= self.mtu <= 1_500:
            raise ValueError("Video MTU must be between 576 and 1500 bytes.")

    @property
    def timestamp_step(self) -> int:
        """Return the nominal 90 kHz RTP timestamp advance per frame."""

        return round(self.clock_rate / self.frame_rate)


@dataclass(frozen=True, slots=True)
class EncodedVideoFrame:
    """One complete encoded frame before DAVE transformation and RTP packetization."""

    data: bytes
    timestamp: int
    keyframe: bool
    codec: VideoCodec

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("Encoded video frames cannot be empty.")
        if not 0 <= self.timestamp <= 0xFFFFFFFF:
            raise ValueError("RTP timestamps must fit in an unsigned 32-bit integer.")
