from __future__ import annotations

import shutil

import pytest

from simajilord.diagnostics.video import run_video_doctor


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is unavailable")
async def test_video_doctor_encodes_and_packetizes_both_codecs() -> None:
    results = await run_video_doctor()
    assert any(
        result.startswith("h264: frame=") and "transport-aead=ok" in result
        for result in results
    )
    assert any(
        result.startswith("vp8: frame=") and "transport-aead=ok" in result
        for result in results
    )
    assert any(result.startswith("DAVE binding:") for result in results)
