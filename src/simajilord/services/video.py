"""Lifecycle, feedback, retransmission, and reporting for real-time video."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import suppress
from enum import StrEnum
from typing import Protocol

from simajilord.core.errors import ProviderError
from simajilord.domain.video import EncodedVideoFrame, VideoCodec, VideoProfile
from simajilord.media.video import (
    FullIntraRequest,
    GenericNack,
    H264RtpPacketizer,
    PictureLossIndication,
    RtpPacket,
    RtpSequence,
    Vp8RtpPacketizer,
    build_sender_report,
    parse_rtcp_feedback,
)

_RETRANSMISSION_CACHE_PACKETS = 512
_RECONNECT_DELAYS = (0.25, 0.5, 1.0, 2.0)
_SENDER_REPORT_INTERVAL_SECONDS = 1.0


class VideoSessionState(StrEnum):
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    RECONNECTING = "reconnecting"
    STOPPING = "stopping"
    FAILED = "failed"


class EncodedVideoSource(Protocol):
    def frames(self) -> AsyncIterator[EncodedVideoFrame]: ...

    async def request_keyframe(self) -> None: ...

    async def close(self) -> None: ...


class VideoFrameEncryptor(Protocol):
    def encrypt(self, frame: EncodedVideoFrame) -> EncodedVideoFrame: ...


class VideoTransport(Protocol):
    """Discord-specific signaling and AEAD transport lives behind this boundary."""

    @property
    def video_ssrc(self) -> int: ...

    async def connect(self) -> None: ...

    async def send_rtp(self, packet: bytes) -> None: ...

    async def send_rtcp(self, packet: bytes) -> None: ...

    async def receive_rtcp(self) -> bytes: ...

    async def disconnect(self) -> None: ...


class IdentityVideoEncryptor:
    """Test/offline transform that intentionally does not claim DAVE protection."""

    def encrypt(self, frame: EncodedVideoFrame) -> EncodedVideoFrame:
        return frame


class VideoSession:
    """Drive an encoded source over one transport with feedback and recovery."""

    def __init__(
        self,
        profile: VideoProfile,
        source: EncodedVideoSource,
        encryptor: VideoFrameEncryptor,
        transport: VideoTransport,
        *,
        initial_sequence: int = 0,
    ) -> None:
        self.profile = profile
        self.source = source
        self.encryptor = encryptor
        self.transport = transport
        self.state = VideoSessionState.IDLE
        self.failure: Exception | None = None
        self._sequence = RtpSequence(initial_sequence)
        self._worker: asyncio.Task[None] | None = None
        self._feedback_worker: asyncio.Task[None] | None = None
        self._report_worker: asyncio.Task[None] | None = None
        self._stopping = False
        self._packet_count = 0
        self._octet_count = 0
        self._last_timestamp = 0
        self._sent_packets: OrderedDict[int, bytes] = OrderedDict()

    async def start(self) -> None:
        if self._worker is not None and not self._worker.done():
            raise ProviderError("The video session is already running.")
        self.failure = None
        self._stopping = False
        self.state = VideoSessionState.CONNECTING
        await self.transport.connect()
        self.state = VideoSessionState.STREAMING
        self._feedback_worker = asyncio.create_task(
            self._feedback_loop(),
            name="simajilord-video-feedback",
        )
        self._report_worker = asyncio.create_task(
            self._sender_report_loop(),
            name="simajilord-video-sender-reports",
        )
        self._worker = asyncio.create_task(
            self._stream_loop(),
            name="simajilord-video-stream",
        )

    async def wait(self) -> None:
        worker = self._worker
        if worker is not None:
            await worker

    async def stop(self) -> None:
        if self.state is VideoSessionState.IDLE:
            return
        self._stopping = True
        self.state = VideoSessionState.STOPPING
        await self.source.close()
        for task in (self._worker, self._feedback_worker, self._report_worker):
            if task is not None and task is not asyncio.current_task():
                task.cancel()
        for task in (self._worker, self._feedback_worker, self._report_worker):
            if task is not None and task is not asyncio.current_task():
                with suppress(asyncio.CancelledError):
                    await task
        await self.transport.disconnect()
        self._worker = None
        self._feedback_worker = None
        self._report_worker = None
        self.state = VideoSessionState.IDLE

    async def _stream_loop(self) -> None:
        try:
            async for clear_frame in self.source.frames():
                transformed = self.encryptor.encrypt(clear_frame)
                if transformed.codec is not self.profile.codec:
                    raise ProviderError("The encrypted video codec changed unexpectedly.")
                for packet in self._packetize(transformed):
                    serialized = packet.to_bytes()
                    await self._send_with_reconnect(serialized)
                    self._packet_count += 1
                    self._octet_count += len(packet.payload)
                    self._last_timestamp = packet.timestamp
                    self._sent_packets[packet.sequence] = serialized
                    self._sent_packets.move_to_end(packet.sequence)
                    while len(self._sent_packets) > _RETRANSMISSION_CACHE_PACKETS:
                        self._sent_packets.popitem(last=False)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure = exc
            self.state = VideoSessionState.FAILED
            raise
        finally:
            if not self._stopping:
                for task in (self._feedback_worker, self._report_worker):
                    if task is not None:
                        task.cancel()
                await self.transport.disconnect()
                if self.state is not VideoSessionState.FAILED:
                    self.state = VideoSessionState.IDLE

    def _packetize(self, frame: EncodedVideoFrame) -> tuple[RtpPacket, ...]:
        if self.profile.codec is VideoCodec.H264:
            return H264RtpPacketizer(
                self.profile,
                ssrc=self.transport.video_ssrc,
                sequence=self._sequence,
            ).packetize(frame)
        return Vp8RtpPacketizer(
            self.profile,
            ssrc=self.transport.video_ssrc,
            sequence=self._sequence,
        ).packetize(frame)

    async def _send_with_reconnect(self, packet: bytes) -> None:
        try:
            await self.transport.send_rtp(packet)
            return
        except (ConnectionError, OSError):
            pass
        self.state = VideoSessionState.RECONNECTING
        last_error: Exception | None = None
        for delay in _RECONNECT_DELAYS:
            if self._stopping:
                raise asyncio.CancelledError
            await asyncio.sleep(delay)
            try:
                await self.transport.disconnect()
                await self.transport.connect()
                await self.transport.send_rtp(packet)
                self.state = VideoSessionState.STREAMING
                await self.source.request_keyframe()
                return
            except (ConnectionError, OSError) as exc:
                last_error = exc
        raise ProviderError("Video transport reconnection failed.") from last_error

    async def _feedback_loop(self) -> None:
        while not self._stopping:
            packet = await self.transport.receive_rtcp()
            for feedback in parse_rtcp_feedback(packet):
                if isinstance(feedback, (PictureLossIndication, FullIntraRequest)):
                    if feedback.media_ssrc in {0, self.transport.video_ssrc}:
                        await self.source.request_keyframe()
                elif isinstance(feedback, GenericNack):
                    if feedback.media_ssrc not in {0, self.transport.video_ssrc}:
                        continue
                    for sequence in feedback.lost_sequences:
                        cached = self._sent_packets.get(sequence)
                        if cached is not None:
                            await self.transport.send_rtp(cached)

    async def _sender_report_loop(self) -> None:
        while not self._stopping:
            await asyncio.sleep(_SENDER_REPORT_INTERVAL_SECONDS)
            report = build_sender_report(
                ssrc=self.transport.video_ssrc,
                rtp_timestamp=self._last_timestamp,
                packet_count=self._packet_count,
                octet_count=self._octet_count,
            )
            await self.transport.send_rtcp(report)
