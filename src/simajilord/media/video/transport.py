"""Discord voice transport encryption for complete RTP packets."""

from __future__ import annotations

import struct

from nacl.secret import Aead

_FIXED_RTP_HEADER_BYTES = 12
_MAX_NONCE = 0xFFFFFFFF


def rtp_header_size(packet: bytes) -> int:
    """Return the authenticated RTP header length, including CSRCs/extensions."""

    if len(packet) < _FIXED_RTP_HEADER_BYTES:
        raise ValueError("The RTP packet is shorter than its fixed header.")
    if packet[0] >> 6 != 2:
        raise ValueError("Only RTP version 2 packets are supported.")

    csrc_count = packet[0] & 0x0F
    header_size = _FIXED_RTP_HEADER_BYTES + (csrc_count * 4)
    if len(packet) < header_size:
        raise ValueError("The RTP packet has a truncated CSRC list.")

    has_extension = bool(packet[0] & 0x10)
    if has_extension:
        if len(packet) < header_size + 4:
            raise ValueError("The RTP packet has a truncated extension header.")
        extension_words = struct.unpack_from(">H", packet, header_size + 2)[0]
        header_size += 4 + (extension_words * 4)
        if len(packet) < header_size:
            raise ValueError("The RTP packet has truncated header extension data.")

    if len(packet) == header_size:
        raise ValueError("The RTP packet has no media payload.")
    return header_size


class XChaCha20Poly1305RtpEncryptor:
    """Apply Discord's ``aead_xchacha20_poly1305_rtpsize`` packet format."""

    def __init__(self, secret_key: bytes, *, initial_nonce: int = 0) -> None:
        if len(secret_key) != Aead.KEY_SIZE:
            raise ValueError(f"The Discord voice secret key must be {Aead.KEY_SIZE} bytes.")
        if not 0 <= initial_nonce <= _MAX_NONCE:
            raise ValueError("The initial transport nonce must fit in 32 bits.")
        self._box = Aead(secret_key)
        self._nonce = initial_nonce

    def encrypt(self, packet: bytes) -> bytes:
        """Encrypt an RTP payload while authenticating its clear RTP header."""

        if self._nonce > _MAX_NONCE:
            raise OverflowError("The Discord voice transport nonce is exhausted.")
        header_size = rtp_header_size(packet)
        header = packet[:header_size]
        payload = packet[header_size:]
        nonce_suffix = struct.pack(">I", self._nonce)
        nonce = nonce_suffix + (b"\x00" * (Aead.NONCE_SIZE - len(nonce_suffix)))
        ciphertext = self._box.encrypt(payload, header, nonce).ciphertext
        self._nonce += 1
        return header + ciphertext + nonce_suffix
