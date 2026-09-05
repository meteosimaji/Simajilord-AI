"""A replayable, bounded speech spool shared by async synthesis and audio threads."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from threading import Condition


class SpeechStream:
    """Expose a growing WAV without mistaking a temporary lack of data for EOF.

    A disk spool bounds memory and allows output preflight/retries to open an
    independent reader. The owner must close the spool when its AudioItem ends.
    """

    def __init__(self, path: Path, *, maximum_bytes: int = 50_000_000) -> None:
        self.path = path
        self.maximum_bytes = maximum_bytes
        self._condition = Condition()
        self._writer = path.open("wb", buffering=0)
        path.chmod(0o600)
        self._size = 0
        self._finished = False
        self._closed = False
        self._error: BaseException | None = None
        self.cancel_production: Callable[[], None] | None = None

    def append(self, data: bytes) -> None:
        with self._condition:
            if self._closed:
                raise OSError("Speech stream is closed.")
            if self._size + len(data) > self.maximum_bytes:
                raise OSError("Speech stream exceeds its size limit.")
            self._writer.write(data)
            self._size += len(data)
            self._condition.notify_all()

    def finish(self, error: BaseException | None = None) -> None:
        with self._condition:
            self._error = error
            self._finished = True
            self._writer.close()
            self._condition.notify_all()

    def wait_finished(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: self._finished or self._closed)
            if self._error is not None:
                raise OSError("Speech synthesis failed during streaming.") from self._error
            if self._closed:
                raise OSError("Speech stream was cancelled.")

    def check_error(self) -> None:
        with self._condition:
            if self._error is not None:
                raise OSError("Speech synthesis failed during streaming.") from self._error

    def open_reader(self) -> SpeechStreamReader:
        return SpeechStreamReader(self)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._writer.close()
            self._condition.notify_all()
        if self.cancel_production is not None:
            self.cancel_production()


class SpeechStreamReader(io.BufferedIOBase):
    """A blocking input for FFmpeg's existing pipe writer thread."""

    def __init__(self, stream: SpeechStream) -> None:
        super().__init__()
        self._stream = stream
        self._position = 0
        self._file = stream.path.open("rb", buffering=0)

    def readable(self) -> bool:
        return True

    def read(self, size: int | None = -1) -> bytes:
        if size == 0:
            return b""
        stream = self._stream
        with stream._condition:
            stream._condition.wait_for(
                lambda: (
                    self.closed
                    or stream._closed
                    or stream._finished
                    or stream._size > self._position
                )
            )
            if self.closed or stream._closed:
                return b""
            if stream._error is not None:
                # End the FFmpeg pipe; the playback adapter checks the stored
                # error after completion rather than stranding its writer.
                return b""
            available = stream._size - self._position
            count = available if size is None or size < 0 else min(size, available)
            data = self._file.read(count)
            self._position += len(data)
            return data

    def close(self) -> None:
        with self._stream._condition:
            self._file.close()
            super().close()
            self._stream._condition.notify_all()
