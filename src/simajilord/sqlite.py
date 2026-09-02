"""SQLite connections whose transaction context also releases the file handle."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Literal


class _ClosingConnection(sqlite3.Connection):
    """Commit or roll back like sqlite3.Connection, then always close."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def closing_connect(
    database: str | Path,
    *,
    timeout: float = 5.0,
) -> sqlite3.Connection:
    """Open a transaction-scoped connection that closes when its ``with`` exits."""

    return sqlite3.connect(
        database,
        timeout=timeout,
        factory=_ClosingConnection,
    )
