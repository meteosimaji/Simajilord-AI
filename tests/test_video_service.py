from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncIterator

from simajilord.domain.video import EncodedVideoFrame, VideoCodec, VideoProfile
from simajilord.services.video import IdentityVideoEncryptor, VideoSession


class FakeVideoSource:
    def __init__(self) -> None:
        self.closed = asyncio.Event()
        self.keyframe_requested = asyncio.Event()

    async def frames(self) -> AsyncIterator[EncodedVideoFrame]:
        yield EncodedVideoFrame(
            data=b"\x00\x00\x00\x01\x65frame",
            timestamp=90_000,
            keyframe=True,
            codec=VideoCodec.H264,
        )
        await self.closed.wait()

    async def request_keyframe(self) -> None:
        self.keyframe_requested.set()

    async def close(self) -> None:
        self.closed.set()


class FakeVideoTransport:
    def __init__(self) -> None:
        self.video_ssrc = 77
        self.connected = False
        self.rtp_packets: list[bytes] = []
        self.rtcp_packets: list[bytes] = []
        self.feedback: asyncio.Queue[bytes] = asyncio.Queue()
        self.packet_sent = asyncio.Event()

    async def connect(self) -> None:
        self.connected = True

    async def send_rtp(self, packet: bytes) -> None:
        self.rtp_packets.append(packet)
        self.packet_sent.set()

    async def send_rtcp(self, packet: bytes) -> None:
        self.rtcp_packets.append(packet)

    async def receive_rtcp(self) -> bytes:
        return await self.feedback.get()

    async def disconnect(self) -> None:
        self.connected = False


async def test_video_session_handles_pli_and_nack_then_stops_cleanly() -> None:
    source = FakeVideoSource()
    transport = FakeVideoTransport()
    session = VideoSession(
        VideoProfile(codec=VideoCodec.H264),
        source,
        IdentityVideoEncryptor(),
        transport,
        initial_sequence=123,
    )

    await session.start()
    await asyncio.wait_for(transport.packet_sent.wait(), timeout=1)
    media_ssrc = transport.video_ssrc
    await transport.feedback.put(struct.pack(">BBHII", 0x81, 206, 2, 1, media_ssrc))
    await asyncio.wait_for(source.keyframe_requested.wait(), timeout=1)

    first_packet = transport.rtp_packets[0]
    sequence = struct.unpack_from(">H", first_packet, 2)[0]
    await transport.feedback.put(
        struct.pack(">BBHIIHH", 0x81, 205, 3, 1, media_ssrc, sequence, 0)
    )
    for _ in range(100):
        if len(transport.rtp_packets) >= 2:
            break
        await asyncio.sleep(0.01)
    assert transport.rtp_packets == [first_packet, first_packet]

    await session.stop()
    assert not transport.connected
    assert session.failure is None
