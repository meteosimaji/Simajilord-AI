"""RFC 3550, RFC 6184, and RFC 7741 video RTP packetization."""

from __future__ import annotations

import struct
from dataclasses import dataclass

from simajilord.domain.video import EncodedVideoFrame, VideoCodec, VideoProfile

from .h264 import annex_b_nal_units

_RTP_HEADER_BYTES = 12


@dataclass(frozen=True, slots=True)
class RtpPacket:
    """One RTP packet before Discord transport encryption."""

    payload_type: int
    sequence: int
    timestamp: int
    ssrc: int
    marker: bool
    payload: bytes

    def __post_init__(self) -> None:
        if not 0 <= self.payload_type <= 127:
            raise ValueError("RTP payload type must fit in seven bits.")
        if not 0 <= self.sequence <= 0xFFFF:
            raise ValueError("RTP sequence must fit in an unsigned 16-bit integer.")
        if not 0 <= self.timestamp <= 0xFFFFFFFF:
            raise ValueError("RTP timestamp must fit in an unsigned 32-bit integer.")
        if not 0 <= self.ssrc <= 0xFFFFFFFF:
            raise ValueError("RTP SSRC must fit in an unsigned 32-bit integer.")
        if not self.payload:
            raise ValueError("RTP packets cannot have an empty payload.")

    def to_bytes(self) -> bytes:
        """Serialize the fixed RTP v2 header and payload."""

        marker_payload = self.payload_type | (0x80 if self.marker else 0)
        return struct.pack(
            ">BBHII",
            0x80,
            marker_payload,
            self.sequence,
            self.timestamp,
            self.ssrc,
        ) + self.payload


class RtpSequence:
    """Wrap an RTP sequence counter without leaking mutable integers."""

    def __init__(self, initial: int = 0) -> None:
        if not 0 <= initial <= 0xFFFF:
            raise ValueError("Initial RTP sequence must fit in 16 bits.")
        self._value = initial

    def take(self) -> int:
        value = self._value
        self._value = (self._value + 1) & 0xFFFF
        return value


class H264RtpPacketizer:
    """Packetize Annex-B H.264 using single NAL units and FU-A fragments."""

    def __init__(self, profile: VideoProfile, *, ssrc: int, sequence: RtpSequence) -> None:
        if profile.codec is not VideoCodec.H264:
            raise ValueError("H264RtpPacketizer requires an H.264 profile.")
        self.profile = profile
        self.ssrc = ssrc
        self.sequence = sequence

    def packetize(self, frame: EncodedVideoFrame) -> tuple[RtpPacket, ...]:
        if frame.codec is not VideoCodec.H264:
            raise ValueError("The frame codec does not match the H.264 packetizer.")
        units = annex_b_nal_units(frame.data)
        if not units:
            raise ValueError("The H.264 frame contains no NAL units.")
        maximum_payload = self.profile.mtu - _RTP_HEADER_BYTES
        packets: list[RtpPacket] = []
        for unit_index, unit in enumerate(units):
            is_last_unit = unit_index == len(units) - 1
            if len(unit) <= maximum_payload:
                packets.append(
                    self._packet(
                        frame,
                        unit,
                        marker=is_last_unit,
                    )
                )
                continue
            if len(unit) < 2:
                raise ValueError("A fragmented H.264 NAL unit must contain a payload.")
            nal_header = unit[0]
            fu_indicator = (nal_header & 0xE0) | 28
            nal_type = nal_header & 0x1F
            chunk_size = maximum_payload - 2
            payload = unit[1:]
            for offset in range(0, len(payload), chunk_size):
                chunk = payload[offset : offset + chunk_size]
                start = offset == 0
                end = offset + len(chunk) == len(payload)
                fu_header = nal_type | (0x80 if start else 0) | (0x40 if end else 0)
                packets.append(
                    self._packet(
                        frame,
                        bytes((fu_indicator, fu_header)) + chunk,
                        marker=is_last_unit and end,
                    )
                )
        return tuple(packets)

    def _packet(
        self,
        frame: EncodedVideoFrame,
        payload: bytes,
        *,
        marker: bool,
    ) -> RtpPacket:
        return RtpPacket(
            payload_type=self.profile.payload_type,
            sequence=self.sequence.take(),
            timestamp=frame.timestamp,
            ssrc=self.ssrc,
            marker=marker,
            payload=payload,
        )


class Vp8RtpPacketizer:
    """Packetize one-partition VP8 frames using the minimal RFC 7741 descriptor."""

    def __init__(self, profile: VideoProfile, *, ssrc: int, sequence: RtpSequence) -> None:
        if profile.codec is not VideoCodec.VP8:
            raise ValueError("Vp8RtpPacketizer requires a VP8 profile.")
        self.profile = profile
        self.ssrc = ssrc
        self.sequence = sequence

    def packetize(self, frame: EncodedVideoFrame) -> tuple[RtpPacket, ...]:
        if frame.codec is not VideoCodec.VP8:
            raise ValueError("The frame codec does not match the VP8 packetizer.")
        maximum_chunk = self.profile.mtu - _RTP_HEADER_BYTES - 1
        packets: list[RtpPacket] = []
        for offset in range(0, len(frame.data), maximum_chunk):
            chunk = frame.data[offset : offset + maximum_chunk]
            start = offset == 0
            end = offset + len(chunk) == len(frame.data)
            # X=0, R=0, N=0, S=start, PartID=0.
            descriptor = 0x10 if start else 0x00
            packets.append(
                RtpPacket(
                    payload_type=self.profile.payload_type,
                    sequence=self.sequence.take(),
                    timestamp=frame.timestamp,
                    ssrc=self.ssrc,
                    marker=end,
                    payload=bytes((descriptor,)) + chunk,
                )
            )
        return tuple(packets)
