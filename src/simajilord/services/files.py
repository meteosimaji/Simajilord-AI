"""Workspace-scoped files for the Discord agent, isolated from host files."""

from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
import threading
import zipfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pypdf import PdfReader

from simajilord.core.errors import UserError

_SAFE_COMPONENT = re.compile(r"^[^/\x00]{1,180}$")
_TEXT_SUFFIXES = {
    ".css",
    ".csv",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".ts",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
_MEDIA_KINDS = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".oga": "audio/ogg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "video/webm",
}


@dataclass(frozen=True, slots=True)
class WorkspaceFileRecord:
    path: str
    size_bytes: int
    sha256: str
    kind: str


@dataclass(frozen=True, slots=True)
class WorkspaceReadResult:
    path: str
    kind: str
    content: str
    size_bytes: int
    sha256: str
    offset: int
    next_offset: int | None
    complete: bool
    page_start: int | None = None
    next_page: int | None = None
    total_pages: int | None = None


class AgentFileSandbox:
    """A per-Discord-workspace authority with no path to project or personal files."""

    def __init__(
        self,
        root: Path,
        *,
        max_file_bytes: int = 25 * 1024 * 1024,
        max_workspace_bytes: int = 500 * 1024 * 1024,
        max_files: int = 1_000,
    ) -> None:
        self.root = root.resolve()
        self.max_file_bytes = max_file_bytes
        self.max_workspace_bytes = max_workspace_bytes
        self.max_files = max_files
        self._lock_guard = threading.Lock()
        self._workspace_locks: dict[str, threading.RLock] = {}
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)

    def list(self, workspace_id: str) -> tuple[WorkspaceFileRecord, ...]:
        with self.locked_workspace(workspace_id):
            return self._list_unlocked(workspace_id)

    def _list_unlocked(
        self,
        workspace_id: str,
    ) -> tuple[WorkspaceFileRecord, ...]:
        scope = self._scope(workspace_id)
        records: list[WorkspaceFileRecord] = []
        for path in sorted(scope.rglob("*")):
            if path.is_symlink():
                raise UserError("files.symlink_forbidden")
            if not path.is_file():
                continue
            records.append(self._record(scope, path))
        return tuple(records)

    def import_bytes(
        self,
        workspace_id: str,
        relative_path: str,
        content: bytes,
    ) -> WorkspaceFileRecord:
        with self.locked_workspace(workspace_id):
            return self._import_bytes_unlocked(
                workspace_id,
                relative_path,
                content,
            )

    def import_batch(
        self,
        workspace_id: str,
        files: Sequence[tuple[str, bytes]],
    ) -> tuple[WorkspaceFileRecord, ...]:
        """Atomically validate and import a bounded group of generated files."""

        if not files:
            return ()
        paths = [path for path, _content in files]
        if len(set(paths)) != len(paths):
            raise UserError("files.path_conflict")
        with self.locked_workspace(workspace_id):
            scope = self._scope(workspace_id)
            current = {
                record.path: record
                for record in self._list_unlocked(workspace_id)
            }
            prospective = dict(current)
            for relative_path, content in files:
                if len(content) > self.max_file_bytes:
                    raise UserError("files.file_too_large")
                destination = self._path(scope, relative_path)
                prospective[relative_path] = WorkspaceFileRecord(
                    path=relative_path,
                    size_bytes=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                    kind=workspace_file_kind(destination, content),
                )
            if len(prospective) > self.max_files:
                raise UserError("files.file_count_limit")
            if (
                sum(record.size_bytes for record in prospective.values())
                > self.max_workspace_bytes
            ):
                raise UserError("files.workspace_quota")

            previous = {
                relative_path: (
                    self._path(scope, relative_path).read_bytes()
                    if relative_path in current
                    else None
                )
                for relative_path in paths
            }
            committed: list[WorkspaceFileRecord] = []
            try:
                for relative_path, content in files:
                    committed.append(
                        self._replace_bytes_unlocked(
                            scope,
                            relative_path,
                            content,
                        )
                    )
            except Exception:
                for relative_path in reversed(paths):
                    prior_content = previous[relative_path]
                    destination = self._path(scope, relative_path)
                    if prior_content is None:
                        destination.unlink(missing_ok=True)
                    else:
                        self._replace_bytes_unlocked(
                            scope,
                            relative_path,
                            prior_content,
                        )
                raise
            return tuple(committed)

    def _import_bytes_unlocked(
        self,
        workspace_id: str,
        relative_path: str,
        content: bytes,
    ) -> WorkspaceFileRecord:
        if len(content) > self.max_file_bytes:
            raise UserError("files.file_too_large")
        scope = self._scope(workspace_id)
        destination = self._path(scope, relative_path)
        existing_size = destination.stat().st_size if destination.exists() else 0
        files = self._list_unlocked(workspace_id)
        if not destination.exists() and len(files) >= self.max_files:
            raise UserError("files.file_count_limit")
        used = sum(item.size_bytes for item in files)
        if used - existing_size + len(content) > self.max_workspace_bytes:
            raise UserError("files.workspace_quota")
        return self._replace_bytes_unlocked(scope, relative_path, content)

    def _replace_bytes_unlocked(
        self,
        scope: Path,
        relative_path: str,
        content: bytes,
    ) -> WorkspaceFileRecord:
        destination = self._path(scope, relative_path)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._assert_no_symlinks(scope, destination)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=".simajilord-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        try:
            temporary.chmod(0o600)
            os.replace(temporary, destination)
            destination.chmod(0o600)
        finally:
            temporary.unlink(missing_ok=True)
        return self._record(scope, destination)

    def write_text(
        self,
        workspace_id: str,
        relative_path: str,
        content: str,
        *,
        expected_sha256: str | None = None,
    ) -> WorkspaceFileRecord:
        if "\x00" in content:
            raise UserError("files.text_invalid")
        with self.locked_workspace(workspace_id):
            scope = self._scope(workspace_id)
            destination = self._path(scope, relative_path)
            self._check_expected_hash(destination, expected_sha256)
            return self._import_bytes_unlocked(
                workspace_id,
                relative_path,
                content.encode("utf-8"),
            )

    def replace_text(
        self,
        workspace_id: str,
        relative_path: str,
        old: str,
        new: str,
        *,
        expected_sha256: str,
    ) -> WorkspaceFileRecord:
        if not old:
            raise UserError("files.replace_old_empty")
        with self.locked_workspace(workspace_id):
            scope = self._scope(workspace_id)
            path = self._path(scope, relative_path)
            self._assert_regular_file(path)
            data = path.read_bytes()
            if len(data) > 200_000:
                raise UserError("files.text_too_large_to_edit")
            try:
                content = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise UserError("files.text_encoding_unsupported") from exc
            current_sha256 = hashlib.sha256(data).hexdigest()
            if current_sha256 != expected_sha256:
                raise UserError("files.hash_conflict")
            occurrences = content.count(old)
            if occurrences != 1:
                raise UserError(
                    "files.replace_not_unique",
                    occurrences=occurrences,
                )
            return self.write_text(
                workspace_id,
                relative_path,
                content.replace(old, new, 1),
                expected_sha256=expected_sha256,
            )

    @contextmanager
    def locked_workspace(self, workspace_id: str) -> Iterator[None]:
        """Serialize quota, CAS, and commit decisions for one workspace."""

        lock = self._workspace_lock(workspace_id)
        with lock:
            yield

    def _workspace_lock(self, workspace_id: str) -> threading.RLock:
        if not workspace_id or len(workspace_id) > 200:
            raise UserError("files.workspace_invalid")
        with self._lock_guard:
            return self._workspace_locks.setdefault(
                workspace_id,
                threading.RLock(),
            )

    def read(
        self,
        workspace_id: str,
        relative_path: str,
        *,
        offset: int,
        max_characters: int,
        page_start: int = 1,
        page_count: int = 5,
    ) -> WorkspaceReadResult:
        if offset < 0 or not 1 <= max_characters <= 20_000:
            raise UserError("files.read_range_invalid")
        if page_start < 1 or not 1 <= page_count <= 20:
            raise UserError("files.page_range_invalid")
        scope = self._scope(workspace_id)
        path = self._path(scope, relative_path)
        self._assert_regular_file(path)
        data = path.read_bytes()
        kind = workspace_file_kind(path, data)
        total_pages: int | None = None
        selected_page_start: int | None = None
        selected_page_end: int | None = None
        if kind == "pdf":
            (
                content,
                total_pages,
                selected_page_start,
                selected_page_end,
            ) = _pdf_text(
                data,
                page_start=page_start,
                page_count=page_count,
            )
        else:
            if page_start != 1 or page_count != 5:
                raise UserError("files.page_range_unsupported")
            content = _inspect_content(path, data, kind)
        if offset > len(content):
            raise UserError("files.read_range_invalid")
        end = min(len(content), offset + max_characters)
        next_offset = end if end < len(content) else None
        next_page = (
            selected_page_end + 1
            if (
                next_offset is None
                and selected_page_end is not None
                and total_pages is not None
                and selected_page_end < total_pages
            )
            else None
        )
        return WorkspaceReadResult(
            path=relative_path,
            kind=kind,
            content=content[offset:end],
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            offset=offset,
            next_offset=next_offset,
            complete=next_offset is None and next_page is None,
            page_start=selected_page_start,
            next_page=next_page,
            total_pages=total_pages,
        )

    def path_for_delivery(self, workspace_id: str, relative_path: str) -> Path:
        scope = self._scope(workspace_id)
        path = self._path(scope, relative_path)
        self._assert_regular_file(path)
        return path

    def snapshot_for_delivery(
        self,
        workspace_id: str,
        relative_path: str,
    ) -> tuple[str, bytes]:
        """Read one immutable send snapshot while workspace writers are excluded."""

        with self.locked_workspace(workspace_id):
            path = self.path_for_delivery(workspace_id, relative_path)
            return path.name, path.read_bytes()

    def validate_path(self, workspace_id: str, relative_path: str) -> None:
        """Validate a future workspace path without creating the target file."""

        scope = self._scope(workspace_id)
        self._path(scope, relative_path)

    def _scope(self, workspace_id: str) -> Path:
        if not workspace_id or len(workspace_id) > 200:
            raise UserError("files.workspace_invalid")
        digest = hashlib.sha256(workspace_id.encode("utf-8")).hexdigest()[:24]
        scope = self.root / digest
        scope.mkdir(mode=0o700, parents=True, exist_ok=True)
        return scope

    def _path(self, scope: Path, relative_path: str) -> Path:
        pure = PurePosixPath(relative_path)
        if (
            not relative_path
            or relative_path.startswith("./")
            or pure.is_absolute()
            or len(pure.parts) > 12
            or any(part in {"", ".", ".."} for part in pure.parts)
            or any(not _SAFE_COMPONENT.fullmatch(part) for part in pure.parts)
        ):
            raise UserError("files.path_invalid")
        candidate = scope.joinpath(*pure.parts)
        self._assert_no_symlinks(scope, candidate)
        try:
            candidate.resolve(strict=False).relative_to(scope)
        except ValueError as exc:
            raise UserError("files.path_forbidden") from exc
        return candidate

    @staticmethod
    def _assert_no_symlinks(scope: Path, candidate: Path) -> None:
        current = candidate
        while current != scope:
            if current.is_symlink():
                raise UserError("files.symlink_forbidden")
            current = current.parent

    @staticmethod
    def _assert_regular_file(path: Path) -> None:
        if path.is_symlink():
            raise UserError("files.symlink_forbidden")
        if not path.is_file():
            raise UserError("files.not_found")

    @staticmethod
    def _check_expected_hash(path: Path, expected: str | None) -> None:
        if expected is None:
            return
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise UserError("files.hash_invalid")
        if not path.is_file():
            raise UserError("files.hash_conflict")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise UserError("files.hash_conflict")

    @staticmethod
    def _record(scope: Path, path: Path) -> WorkspaceFileRecord:
        data = path.read_bytes()
        return WorkspaceFileRecord(
            path=path.relative_to(scope).as_posix(),
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            kind=workspace_file_kind(path, data),
        )


def workspace_file_kind(path: Path, data: bytes) -> str:
    """Classify workspace bytes without opening or executing their contents."""

    suffix = path.suffix.lower()
    if data.startswith(b"%PDF-") or suffix == ".pdf":
        return "pdf"
    if data.startswith(b"PK\x03\x04") or suffix == ".zip":
        return "zip"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in {b"GIF87a", b"GIF89a"}:
        return "image/gif"
    if suffix in _MEDIA_KINDS:
        return _MEDIA_KINDS[suffix]
    if suffix in _TEXT_SUFFIXES:
        return "text"
    try:
        data[:8_192].decode("utf-8")
    except UnicodeDecodeError:
        return "binary"
    return "text"


def _inspect_content(path: Path, data: bytes, kind: str) -> str:
    if kind == "text":
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UserError("files.text_encoding_unsupported") from exc
    if kind == "zip":
        return _zip_listing(data)
    if kind.startswith("image/"):
        return f"Image file: {path.name} ({kind}, {len(data)} bytes)."
    return f"Binary file: {path.name} ({len(data)} bytes)."


def _pdf_text(
    data: bytes,
    *,
    page_start: int,
    page_count: int,
) -> tuple[str, int, int, int]:
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise UserError("files.pdf_encrypted")
        pages = reader.pages
        total_pages = len(pages)
        if total_pages < 1 or page_start > total_pages:
            raise UserError(
                "files.page_range_invalid",
                total_pages=total_pages,
            )
        page_end = min(total_pages, page_start + page_count - 1)
        lines = [
            (
                f"PDF pages: {total_pages}; selected pages: "
                f"{page_start}-{page_end}"
            )
        ]
        for index in range(page_start, page_end + 1):
            page = pages[index - 1]
            text = (page.extract_text() or "").strip()
            lines.append(f"\n--- Page {index} ---\n{text}")
        if page_end < total_pages:
            lines.append(f"\n[Continue with page_start={page_end + 1}.]")
        return "\n".join(lines), total_pages, page_start, page_end
    except UserError:
        raise
    except Exception as exc:
        raise UserError("files.pdf_invalid") from exc


def _zip_listing(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > 1_000:
                raise UserError("files.zip_entry_limit")
            total = sum(item.file_size for item in entries)
            if total > 200 * 1024 * 1024:
                raise UserError("files.zip_expansion_limit")
            for item in entries:
                if item.flag_bits & 0x1:
                    raise UserError("files.zip_encrypted")
                if (
                    item.file_size > 0
                    and item.compress_size > 0
                    and item.file_size / item.compress_size > 100
                ):
                    raise UserError("files.zip_ratio_limit")
                member = PurePosixPath(item.filename)
                if member.is_absolute() or ".." in member.parts:
                    raise UserError("files.zip_path_invalid")
            shown = entries[:200]
            lines = [
                f"ZIP entries: {len(entries)}; uncompressed bytes: {total}",
                *[
                    f"{item.filename}\t{item.file_size} bytes"
                    for item in shown
                ],
            ]
            if len(entries) > len(shown):
                lines.append(f"[{len(entries) - len(shown)} more entries not shown.]")
            return "\n".join(lines)
    except UserError:
        raise
    except (zipfile.BadZipFile, OSError) as exc:
        raise UserError("files.zip_invalid") from exc
