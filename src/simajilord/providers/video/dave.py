"""DAVE encoded-frame transform backed by the installed davey/libdave binding."""

from __future__ import annotations

import davey

from simajilord.core.errors import ProviderError
from simajilord.domain.video import EncodedVideoFrame, VideoCodec

_DAVE_CODEC = {
    VideoCodec.H264: davey.Codec.h264,
    VideoCodec.VP8: davey.Codec.vp8,
}


class DaveVideoEncryptor:
    """Encrypt complete encoded frames before codec packetization."""

    def __init__(self, session: davey.DaveSession) -> None:
        self._session = session

    def encrypt(self, frame: EncodedVideoFrame) -> EncodedVideoFrame:
        """Apply the codec-aware DAVE transform to one complete video frame."""

        if not self._session.ready:
            raise ProviderError("The DAVE session is not ready to encrypt video.")
        try:
            encrypted = self._session.encrypt(
                davey.MediaType.video,
                _DAVE_CODEC[frame.codec],
                frame.data,
            )
        except (RuntimeError, ValueError) as exc:
            raise ProviderError("DAVE could not encrypt the encoded video frame.") from exc
        if not encrypted:
            raise ProviderError("DAVE returned an empty encrypted video frame.")
        return EncodedVideoFrame(
            data=encrypted,
            timestamp=frame.timestamp,
            keyframe=frame.keyframe,
            codec=frame.codec,
        )
