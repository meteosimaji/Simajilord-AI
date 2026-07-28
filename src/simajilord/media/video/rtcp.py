"""Minimal RTCP feedback parsing and sender-report generation."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from time import time
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class PictureLossIndication:
    sender_ssrc: int
    media_ssrc: int


@dataclass(frozen=True, slots=True)
class FullIntraRequest:
    sender_ssrc: int
    media_ssrc: int
    target_ssrc: int
    sequence_number: int


@dataclass(frozen=True, slots=True)
class GenericNack:
    sender_ssrc: int
    media_ssrc: int
    lost_sequences: tuple[int, ...]


RtcpFeedback: TypeAlias = PictureLossIndication | FullIntraRequest | GenericNack


def parse_rtcp_feedback(packet: bytes) -> tuple[RtcpFeedback, ...]:
    """Parse compound RTCP PLI, FIR, and generic NACK feedback packets."""

    feedback: list[RtcpFeedback] = []
    offset = 0
    while offset + 4 <= len(packet):
        first, packet_type, length_words = struct.unpack_from(">BBH", packet, offset)
        version = first >> 6
        fmt = first & 0x1F
        packet_size = (length_words + 1) * 4
        end = offset + packet_size
        if version != 2 or packet_size < 4 or end > len(packet):
            raise ValueError("Malformed RTCP compound packet.")
        body = packet[offset + 4 : end]
        if packet_type in {205, 206} and len(body) >= 8:
            sender_ssrc, media_ssrc = struct.unpack_from(">II", body)
            fci = body[8:]
            if packet_type == 206 and fmt == 1:
                feedback.append(PictureLossIndication(sender_ssrc, media_ssrc))
            elif packet_type == 206 and fmt == 4:
                for fci_offset in range(0, len(fci) - 7, 8):
                    target_ssrc, sequence_number = struct.unpack_from(
                        ">IB",
                        fci,
                        fci_offset,
                    )
                    feedback.append(
                        FullIntraRequest(
                            sender_ssrc,
                            media_ssrc,
                            target_ssrc,
                            sequence_number,
                        )
                    )
            elif packet_type == 205 and fmt == 1:
                lost: list[int] = []
                for fci_offset in range(0, len(fci) - 3, 4):
                    packet_id, bitmask = struct.unpack_from(">HH", fci, fci_offset)
                    lost.append(packet_id)
                    lost.extend(
                        (packet_id + bit + 1) & 0xFFFF
                        for bit in range(16)
                        if bitmask & (1 << bit)
                    )
                feedback.append(GenericNack(sender_ssrc, media_ssrc, tuple(lost)))
        offset = end
    if offset != len(packet):
        raise ValueError("RTCP packet has trailing bytes.")
    return tuple(feedback)


def build_sender_report(
    *,
    ssrc: int,
    rtp_timestamp: int,
    packet_count: int,
    octet_count: int,
    wall_time: float | None = None,
) -> bytes:
    """Build an RFC 3550 sender report without reception report blocks."""

    now = time() if wall_time is None else wall_time
    ntp = now + 2_208_988_800
    ntp_seconds = int(ntp) & 0xFFFFFFFF
    ntp_fraction = int((ntp - int(ntp)) * (1 << 32)) & 0xFFFFFFFF
    return struct.pack(
        ">BBHIIIIII",
        0x80,
        200,
        6,
        ssrc & 0xFFFFFFFF,
        ntp_seconds,
        ntp_fraction,
        rtp_timestamp & 0xFFFFFFFF,
        packet_count & 0xFFFFFFFF,
        octet_count & 0xFFFFFFFF,
    )
