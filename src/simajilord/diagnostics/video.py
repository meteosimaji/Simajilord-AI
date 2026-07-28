"""Offline FFmpeg and RTP video pipeline diagnostic."""

from __future__ import annotations

import asyncio
import os
from contextlib import aclosing

import davey

from simajilord.domain.video import VideoCodec, VideoProfile
from simajilord.media.video import (
    H264RtpPacketizer,
    RtpSequence,
    Vp8RtpPacketizer,
    XChaCha20Poly1305RtpEncryptor,
)
from simajilord.providers.video.ffmpeg import FfmpegEncodedVideoSource


async def run_video_doctor() -> tuple[str, ...]:
    """Encode and packetize a local synthetic frame for each supported codec."""

    results: list[str] = []
    for codec in VideoCodec:
        profile = VideoProfile(
            codec=codec,
            width=320,
            height=180,
            frame_rate=5,
            bitrate_kbps=256,
        )
        source = FfmpegEncodedVideoSource(
            "testsrc2=size=320x180:rate=5",
            profile,
            input_format="lavfi",
            realtime=False,
        )
        try:
            async with aclosing(source.frames()) as frames:
                frame = await asyncio.wait_for(anext(frames), timeout=15.0)
                sequence = RtpSequence(65_530)
                if codec is VideoCodec.H264:
                    packets = H264RtpPacketizer(
                        profile,
                        ssrc=1,
                        sequence=sequence,
                    ).packetize(frame)
                else:
                    packets = Vp8RtpPacketizer(
                        profile,
                        ssrc=1,
                        sequence=sequence,
                    ).packetize(frame)
        finally:
            await source.close()
        if not packets or not packets[-1].marker:
            raise RuntimeError(f"{codec.value} RTP packetization produced no complete frame.")
        encrypted = XChaCha20Poly1305RtpEncryptor(os.urandom(32)).encrypt(
            packets[0].to_bytes()
        )
        if len(encrypted) <= len(packets[0].to_bytes()):
            raise RuntimeError(f"{codec.value} transport encryption produced no auth data.")
        results.append(
            f"{codec.value}: frame={len(frame.data)} bytes rtp={len(packets)} packets "
            "transport-aead=ok"
        )
    results.append(f"DAVE binding: {davey.__version__}")
    results.append(
        "Discord live video signaling: unavailable in the documented Bot Voice API"
    )
    return tuple(results)


def main() -> None:
    """Run the diagnostic as a console script."""

    for result in asyncio.run(run_video_doctor()):
        print(result)


if __name__ == "__main__":
    main()
