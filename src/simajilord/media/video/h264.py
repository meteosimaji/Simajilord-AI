"""Small Annex-B helpers for H.264 encoded-frame transforms."""

from __future__ import annotations


def annex_b_nal_units(frame: bytes) -> tuple[bytes, ...]:
    """Split one Annex-B access unit while removing three/four-byte start codes."""

    starts: list[tuple[int, int]] = []
    index = 0
    size = len(frame)
    while index + 3 <= size:
        if frame[index : index + 4] == b"\x00\x00\x00\x01":
            starts.append((index, 4))
            index += 4
            continue
        if frame[index : index + 3] == b"\x00\x00\x01":
            starts.append((index, 3))
            index += 3
            continue
        index += 1
    if not starts:
        return (frame,) if frame else ()

    units: list[bytes] = []
    for position, (start, prefix_size) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else size
        # A zero byte can be valid RBSP data. The packetizer must preserve the
        # encoded access unit byte-for-byte instead of guessing which trailing
        # zeroes are Annex-B padding.
        unit = frame[start + prefix_size : end]
        if unit:
            units.append(unit)
    return tuple(units)


def h264_frame_is_keyframe(frame: bytes) -> bool:
    """Return whether an Annex-B frame contains an IDR slice."""

    return any((unit[0] & 0x1F) == 5 for unit in annex_b_nal_units(frame))
