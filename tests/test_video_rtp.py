from __future__ import annotations

import struct

import pytest
from nacl.secret import Aead

from simajilord.domain.video import EncodedVideoFrame, VideoCodec, VideoProfile
from simajilord.media.video import (
    FullIntraRequest,
    GenericNack,
    H264RtpPacketizer,
    PictureLossIndication,
    RtpSequence,
    Vp8RtpPacketizer,
    XChaCha20Poly1305RtpEncryptor,
    build_sender_report,
    parse_rtcp_feedback,
    rtp_header_size,
)
from simajilord.media.video.h264 import annex_b_nal_units, h264_frame_is_keyframe


def test_video_profile_validates_bounded_rtp_settings() -> None:
    profile = VideoProfile(frame_rate=30)
    assert profile.timestamp_step == 3_000
    with pytest.raises(ValueError, match="dynamic"):
        VideoProfile(payload_type=95)


def test_h264_packetizer_preserves_access_unit_and_fragments_fu_a() -> None:
    large_idr = bytes((0x65,)) + bytes(range(256)) * 8 + b"\x00"
    access_unit = b"\x00\x00\x00\x01\x67\x42\x00\x1f" + b"\x00\x00\x01" + large_idr
    frame = EncodedVideoFrame(
        data=access_unit,
        timestamp=90_000,
        keyframe=True,
        codec=VideoCodec.H264,
    )
    profile = VideoProfile(codec=VideoCodec.H264, mtu=600)
    packets = H264RtpPacketizer(
        profile,
        ssrc=42,
        sequence=RtpSequence(65_534),
    ).packetize(frame)

    assert annex_b_nal_units(access_unit) == (b"\x67\x42\x00\x1f", large_idr)
    assert h264_frame_is_keyframe(access_unit)
    assert packets[0].payload == b"\x67\x42\x00\x1f"
    assert [packet.sequence for packet in packets[:4]] == [65_534, 65_535, 0, 1]
    assert not any(packet.marker for packet in packets[:-1])
    assert packets[-1].marker

    fragments = packets[1:]
    nal_header = (fragments[0].payload[0] & 0xE0) | (
        fragments[0].payload[1] & 0x1F
    )
    rebuilt = bytes((nal_header,)) + b"".join(packet.payload[2:] for packet in fragments)
    assert rebuilt == large_idr


def test_vp8_packetizer_sets_start_descriptor_and_final_marker() -> None:
    frame = EncodedVideoFrame(
        data=b"\x10" * 1_500,
        timestamp=123,
        keyframe=True,
        codec=VideoCodec.VP8,
    )
    packets = Vp8RtpPacketizer(
        VideoProfile(codec=VideoCodec.VP8, mtu=600),
        ssrc=7,
        sequence=RtpSequence(),
    ).packetize(frame)

    assert packets[0].payload[0] == 0x10
    assert all(packet.payload[0] == 0 for packet in packets[1:])
    assert not any(packet.marker for packet in packets[:-1])
    assert packets[-1].marker
    assert b"".join(packet.payload[1:] for packet in packets) == frame.data


def test_parse_rtcp_pli_fir_and_nack_and_build_sender_report() -> None:
    pli = struct.pack(">BBHII", 0x81, 206, 2, 11, 22)
    fir = (
        struct.pack(">BBHII", 0x84, 206, 4, 33, 0)
        + struct.pack(">IB3x", 44, 9)
    )
    nack = struct.pack(">BBHIIHH", 0x81, 205, 3, 55, 66, 100, 0b101)

    feedback = parse_rtcp_feedback(pli + fir + nack)
    assert feedback == (
        PictureLossIndication(11, 22),
        FullIntraRequest(33, 0, 44, 9),
        GenericNack(55, 66, (100, 101, 103)),
    )

    report = build_sender_report(
        ssrc=77,
        rtp_timestamp=88,
        packet_count=99,
        octet_count=111,
        wall_time=0,
    )
    assert len(report) == 28
    assert struct.unpack_from(">BBH", report) == (0x80, 200, 6)
    assert struct.unpack_from(">I", report, 4)[0] == 77


def test_rtcp_rejects_trailing_or_truncated_data() -> None:
    with pytest.raises(ValueError, match="trailing"):
        parse_rtcp_feedback(b"\x00")
    with pytest.raises(ValueError, match="Malformed"):
        parse_rtcp_feedback(struct.pack(">BBH", 0x81, 206, 8))


def test_discord_transport_aead_round_trip_and_nonce_increment() -> None:
    key = bytes(range(Aead.KEY_SIZE))
    packet = (
        struct.pack(">BBHII", 0x80, 0xE5, 12, 34, 56)
        + b"first video payload"
    )
    encryptor = XChaCha20Poly1305RtpEncryptor(key, initial_nonce=41)

    encrypted = encryptor.encrypt(packet)
    header = encrypted[:12]
    suffix = encrypted[-4:]
    nonce = suffix + (b"\x00" * (Aead.NONCE_SIZE - 4))
    clear = Aead(key).decrypt(encrypted[12:-4], header, nonce)

    assert header == packet[:12]
    assert suffix == struct.pack(">I", 41)
    assert clear == packet[12:]
    assert encryptor.encrypt(packet)[-4:] == struct.pack(">I", 42)


def test_rtp_header_size_handles_csrc_and_extension() -> None:
    packet = (
        bytes((0x92, 101))
        + struct.pack(">HII", 1, 2, 3)
        + struct.pack(">II", 4, 5)
        + struct.pack(">HH", 0xBEDE, 2)
        + b"12345678"
        + b"payload"
    )
    assert rtp_header_size(packet) == 32

    with pytest.raises(ValueError, match="version"):
        rtp_header_size(bytes(12))
    with pytest.raises(ValueError, match="extension"):
        rtp_header_size(packet[:30])


def test_discord_transport_aead_rejects_nonce_reuse_after_wrap() -> None:
    packet = struct.pack(">BBHII", 0x80, 101, 1, 2, 3) + b"payload"
    encryptor = XChaCha20Poly1305RtpEncryptor(
        b"x" * Aead.KEY_SIZE,
        initial_nonce=0xFFFFFFFF,
    )
    encryptor.encrypt(packet)
    with pytest.raises(OverflowError, match="exhausted"):
        encryptor.encrypt(packet)
