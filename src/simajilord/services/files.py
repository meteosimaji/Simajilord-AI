"""Workspace-scoped files for the Discord agent, isolated from host files."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import tempfile
import threading
import uuid
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal, TypeAlias, cast

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
WorkspaceVisibility: TypeAlias = Literal[
    "guild_public",
    "restricted",
    "uncertain",
    "actor_private",
]
WorkspaceSourceVisibility: TypeAlias = Literal[
    "guild_public",
    "restricted",
    "uncertain",
]
WorkspacePublicationTargetKind: TypeAlias = Literal["discord_channel"]


@dataclass(frozen=True, slots=True)
class WorkspaceFilePublication:
    """Target-bound, revocable authority for one immutable file copy."""

    publication_id: str
    source_workspace_id: str
    source_path: str
    source_sha256: str
    copy_workspace_id: str
    copy_path: str
    target_kind: WorkspacePublicationTargetKind
    target_workspace_id: str
    target_resource_id: str
    target_display_name: str
    target_audience_revision: str
    published_by_actor_id: str
    reason: str
    published_at: str
    expires_at: str
    revision: int = 1
    revoked_at: str | None = None
    revoked_by: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"pub_[0-9a-f]{32}", self.publication_id):
            raise ValueError("invalid file publication ID")
        if self.target_kind != "discord_channel":
            raise ValueError("unsupported file publication target")
        scalar_values = (
            self.source_workspace_id,
            self.copy_workspace_id,
            self.target_workspace_id,
            self.target_resource_id,
            self.target_display_name,
            self.published_by_actor_id,
            self.reason,
            self.published_at,
            self.expires_at,
            self.revoked_at,
            self.revoked_by,
        )
        if any(
            value is not None and (not value or len(value) > 500)
            for value in scalar_values
        ):
            raise ValueError("file publication values must be bounded")
        if not self.source_path or len(self.source_path) > 2_400:
            raise ValueError("file publication source path must be bounded")
        if not self.copy_path or len(self.copy_path) > 2_400:
            raise ValueError("file publication copy path must be bounded")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("invalid file publication source hash")
        if not re.fullmatch(r"[0-9a-f]{64}", self.target_audience_revision):
            raise ValueError("invalid file publication audience revision")
        if self.revision < 1:
            raise ValueError("file publication revision must be positive")
        published_at = _publication_datetime(self.published_at)
        expires_at = _publication_datetime(self.expires_at)
        if expires_at <= published_at:
            raise ValueError("file publication expiry must follow publication")
        if (self.revoked_at is None) != (self.revoked_by is None):
            raise ValueError("file publication revocation fields must be paired")
        if self.revoked_at is not None:
            _publication_datetime(self.revoked_at)

    @property
    def active(self) -> bool:
        """Return whether this authority is usable at the current instant."""

        return (
            self.revoked_at is None
            and datetime.now(UTC) < _publication_datetime(self.expires_at)
        )


@dataclass(frozen=True, slots=True)
class WorkspaceFileProvenance:
    """Bounded source label retained independently from file bytes."""

    owner_actor_ids: tuple[str, ...]
    origin_guild_id: str | None = None
    origin_channel_id: str | None = None
    origin_message_id: str | None = None
    origin_visibility: WorkspaceVisibility = "actor_private"
    created_task_id: str | None = None
    sensitivity: WorkspaceVisibility = "actor_private"
    source_resources: tuple[
        tuple[str, str, WorkspaceSourceVisibility], ...
    ] = ()
    unlabelled_input: bool = False
    sources_truncated: bool = False
    declassified_at: str | None = None
    declassified_by: str | None = None
    publication_id: str | None = None

    def __post_init__(self) -> None:
        normalized_owners = tuple(sorted(set(self.owner_actor_ids)))
        object.__setattr__(self, "owner_actor_ids", normalized_owners)
        scalar_values = (
            self.origin_guild_id,
            self.origin_channel_id,
            self.origin_message_id,
            self.created_task_id,
            self.declassified_at,
            self.declassified_by,
            self.publication_id,
        )
        if any(
            value is not None and len(value) > 200 for value in scalar_values
        ):
            raise ValueError("file provenance values must be bounded")
        if len(normalized_owners) > 32 or any(
            not value or len(value) > 200 for value in normalized_owners
        ):
            raise ValueError("file provenance owners must be bounded")
        if (
            not normalized_owners
            and not self.unlabelled_input
            and self.sensitivity != "uncertain"
        ):
            raise ValueError("file provenance must retain an owner or uncertainty")
        if len(self.source_resources) > 32 or any(
            len(item) != 3 or any(not value or len(value) > 200 for value in item)
            for item in self.source_resources
        ):
            raise ValueError("file provenance source resources must be bounded")
        if self.publication_id is not None and not re.fullmatch(
            r"pub_[0-9a-f]{32}",
            self.publication_id,
        ):
            raise ValueError("invalid file provenance publication ID")
        if (
            len(normalized_owners) != 1
            or self.unlabelled_input
            or self.sources_truncated
        ):
            object.__setattr__(self, "sensitivity", "uncertain")


def unlabelled_file_provenance() -> WorkspaceFileProvenance:
    """Return the fail-closed label for bytes without durable provenance."""

    return WorkspaceFileProvenance(
        owner_actor_ids=(),
        origin_visibility="uncertain",
        sensitivity="uncertain",
        unlabelled_input=True,
    )


def file_provenance_is_owned_by(
    provenance: WorkspaceFileProvenance | None,
    actor_id: str,
) -> bool:
    """Allow private file authority only when one exact actor owns the bytes."""

    return (
        provenance is not None
        and not provenance.unlabelled_input
        and provenance.owner_actor_ids == (actor_id,)
    )


@dataclass(frozen=True, slots=True)
class WorkspaceFileRecord:
    path: str
    size_bytes: int
    sha256: str
    kind: str
    provenance: WorkspaceFileProvenance | None = None


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
    provenance: WorkspaceFileProvenance | None = None


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
        self._provenance_lock = threading.RLock()
        self._workspace_locks: dict[str, threading.RLock] = {}
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self._provenance_path = self.root.with_name(
            f"{self.root.name}.provenance.sqlite3"
        )
        self._initialize_provenance()

    def list(self, workspace_id: str) -> tuple[WorkspaceFileRecord, ...]:
        with self.locked_workspace(workspace_id):
            return self._list_unlocked(workspace_id)

    def list_for_actor(
        self,
        workspace_id: str,
        actor_id: str,
    ) -> tuple[WorkspaceFileRecord, ...]:
        """List only files whose private authority belongs to one exact actor."""

        with self.locked_workspace(workspace_id):
            return tuple(
                record
                for record in self._list_unlocked(workspace_id)
                if file_provenance_is_owned_by(record.provenance, actor_id)
            )

    def _list_unlocked(
        self,
        workspace_id: str,
    ) -> tuple[WorkspaceFileRecord, ...]:
        scope = self._scope_path(workspace_id)
        records: list[WorkspaceFileRecord] = []
        for path in sorted(scope.rglob("*")):
            if path.is_symlink():
                raise UserError("files.symlink_forbidden")
            if not path.is_file():
                continue
            records.append(self._record(workspace_id, scope, path))
        return tuple(records)

    def import_bytes(
        self,
        workspace_id: str,
        relative_path: str,
        content: bytes,
        *,
        provenance: WorkspaceFileProvenance | None = None,
    ) -> WorkspaceFileRecord:
        with self.locked_workspace(workspace_id):
            return self._import_bytes_unlocked(
                workspace_id,
                relative_path,
                content,
                provenance=provenance,
            )

    def validate_import_bytes(
        self,
        workspace_id: str,
        relative_path: str,
        content: bytes,
        *,
        provenance: WorkspaceFileProvenance | None = None,
    ) -> None:
        """Validate an import without mutating bytes or provenance."""

        with self.locked_workspace(workspace_id):
            self._validate_import_bytes_unlocked(
                workspace_id,
                relative_path,
                content,
                provenance=provenance,
            )

    def import_batch(
        self,
        workspace_id: str,
        files: Sequence[tuple[str, bytes]],
        *,
        provenance: WorkspaceFileProvenance | None = None,
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
                current_record = current.get(relative_path)
                prospective[relative_path] = WorkspaceFileRecord(
                    path=relative_path,
                    size_bytes=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                    kind=workspace_file_kind(destination, content),
                    provenance=(
                        provenance
                        if provenance is not None
                        else current_record.provenance
                        if current_record is not None
                        else None
                    ),
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
            previous_provenance = {
                relative_path: (
                    current[relative_path].provenance
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
                            workspace_id,
                            relative_path,
                            content,
                            provenance=provenance,
                        )
                    )
            except Exception:
                for relative_path in reversed(paths):
                    prior_content = previous[relative_path]
                    destination = self._path(scope, relative_path)
                    if prior_content is None:
                        destination.unlink(missing_ok=True)
                        self._delete_provenance(workspace_id, relative_path)
                    else:
                        self._replace_bytes_unlocked(
                            scope,
                            workspace_id,
                            relative_path,
                            prior_content,
                            provenance=previous_provenance[relative_path],
                            replace_provenance=True,
                        )
                raise
            return tuple(committed)

    def _import_bytes_unlocked(
        self,
        workspace_id: str,
        relative_path: str,
        content: bytes,
        *,
        provenance: WorkspaceFileProvenance | None = None,
    ) -> WorkspaceFileRecord:
        self._validate_import_bytes_unlocked(
            workspace_id,
            relative_path,
            content,
            provenance=provenance,
        )
        scope = self._scope(workspace_id)
        return self._replace_bytes_unlocked(
            scope,
            workspace_id,
            relative_path,
            content,
            provenance=provenance,
        )

    def _validate_import_bytes_unlocked(
        self,
        workspace_id: str,
        relative_path: str,
        content: bytes,
        *,
        provenance: WorkspaceFileProvenance | None = None,
    ) -> None:
        if len(content) > self.max_file_bytes:
            raise UserError("files.file_too_large")
        scope = self._scope_path(workspace_id)
        destination = self._path(scope, relative_path)
        existing_size = destination.stat().st_size if destination.exists() else 0
        files = self._list_unlocked(workspace_id)
        if not destination.exists() and len(files) >= self.max_files:
            raise UserError("files.file_count_limit")
        used = sum(item.size_bytes for item in files)
        if used - existing_size + len(content) > self.max_workspace_bytes:
            raise UserError("files.workspace_quota")
        previous_provenance = self._load_provenance(
            workspace_id,
            relative_path,
        )
        if destination.exists() and previous_provenance is None:
            previous_provenance = unlabelled_file_provenance()
        if (
            destination.exists()
            and provenance is not None
            and (
                previous_provenance is None
                or previous_provenance.unlabelled_input
                or previous_provenance.owner_actor_ids
                != provenance.owner_actor_ids
            )
        ):
            raise UserError("files.path_conflict")

    def _replace_bytes_unlocked(
        self,
        scope: Path,
        workspace_id: str,
        relative_path: str,
        content: bytes,
        *,
        provenance: WorkspaceFileProvenance | None = None,
        replace_provenance: bool = False,
    ) -> WorkspaceFileRecord:
        destination = self._path(scope, relative_path)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._assert_no_symlinks(scope, destination)
        previous_provenance = self._load_provenance(workspace_id, relative_path)
        if destination.exists() and previous_provenance is None:
            previous_provenance = unlabelled_file_provenance()
        if (
            destination.exists()
            and provenance is not None
            and not replace_provenance
            and (
                previous_provenance is None
                or previous_provenance.unlabelled_input
                or previous_provenance.owner_actor_ids
                != provenance.owner_actor_ids
            )
        ):
            raise UserError("files.path_conflict")
        next_provenance = previous_provenance
        should_store_provenance = replace_provenance or provenance is not None
        if replace_provenance:
            next_provenance = provenance
        elif provenance is not None:
            next_provenance = (
                provenance
                if previous_provenance is None
                else merge_file_provenances(
                    (previous_provenance, provenance)
                )
            )
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=".simajilord-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        try:
            temporary.chmod(0o600)
            # Persist the label first. A failed byte replacement can leave an
            # over-restrictive label, while the inverse could expose unlabeled
            # restricted bytes after a provenance database failure.
            if should_store_provenance:
                self._store_provenance(
                    workspace_id,
                    relative_path,
                    next_provenance,
                )
            try:
                os.replace(temporary, destination)
                destination.chmod(0o600)
            except Exception:
                if should_store_provenance:
                    # The retained label is at least as restrictive as the
                    # attempted write if restoring the old label also fails.
                    with suppress(Exception):
                        self._store_provenance(
                            workspace_id,
                            relative_path,
                            previous_provenance,
                        )
                raise
        finally:
            temporary.unlink(missing_ok=True)
        return self._record(workspace_id, scope, destination)

    def write_text(
        self,
        workspace_id: str,
        relative_path: str,
        content: str,
        *,
        expected_sha256: str | None = None,
        provenance: WorkspaceFileProvenance | None = None,
    ) -> WorkspaceFileRecord:
        if "\x00" in content:
            raise UserError("files.text_invalid")
        with self.locked_workspace(workspace_id):
            scope = self._scope_path(workspace_id)
            destination = self._path(scope, relative_path)
            self._check_expected_hash(destination, expected_sha256)
            return self._import_bytes_unlocked(
                workspace_id,
                relative_path,
                content.encode("utf-8"),
                provenance=provenance,
            )

    def write_text_for_actor(
        self,
        workspace_id: str,
        actor_id: str,
        relative_path: str,
        content: str,
        *,
        expected_sha256: str | None = None,
        provenance: WorkspaceFileProvenance,
    ) -> WorkspaceFileRecord:
        """Write without allowing a guessed path to overwrite another actor."""

        if not file_provenance_is_owned_by(provenance, actor_id):
            raise UserError("files.provenance_invalid")
        with self.locked_workspace(workspace_id):
            scope = self._scope(workspace_id)
            destination = self._path(scope, relative_path)
            if destination.exists() and not file_provenance_is_owned_by(
                self._load_provenance(workspace_id, relative_path),
                actor_id,
            ):
                raise UserError("files.path_conflict")
            return self.write_text(
                workspace_id,
                relative_path,
                content,
                expected_sha256=expected_sha256,
                provenance=provenance,
            )

    def validate_write_text_for_actor(
        self,
        workspace_id: str,
        actor_id: str,
        relative_path: str,
        content: str,
        *,
        expected_sha256: str | None = None,
        provenance: WorkspaceFileProvenance,
    ) -> None:
        """Preflight an actor-owned text write without changing file bytes or labels."""

        if "\x00" in content:
            raise UserError("files.text_invalid")
        if not file_provenance_is_owned_by(provenance, actor_id):
            raise UserError("files.provenance_invalid")
        encoded = content.encode("utf-8")
        with self.locked_workspace(workspace_id):
            scope = self._scope_path(workspace_id)
            destination = self._path(scope, relative_path)
            if destination.exists() and not file_provenance_is_owned_by(
                self._load_provenance(workspace_id, relative_path),
                actor_id,
            ):
                raise UserError("files.path_conflict")
            self._check_expected_hash(destination, expected_sha256)
            self._validate_import_bytes_unlocked(
                workspace_id,
                relative_path,
                encoded,
                provenance=provenance,
            )

    def replace_text(
        self,
        workspace_id: str,
        relative_path: str,
        old: str,
        new: str,
        *,
        expected_sha256: str,
        provenance: WorkspaceFileProvenance | None = None,
    ) -> WorkspaceFileRecord:
        if not old:
            raise UserError("files.replace_old_empty")
        with self.locked_workspace(workspace_id):
            scope = self._scope_path(workspace_id)
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
                provenance=provenance,
            )

    def replace_text_for_actor(
        self,
        workspace_id: str,
        actor_id: str,
        relative_path: str,
        old: str,
        new: str,
        *,
        expected_sha256: str,
        provenance: WorkspaceFileProvenance,
    ) -> WorkspaceFileRecord:
        """Replace only a file exclusively owned by the current actor."""

        if not file_provenance_is_owned_by(provenance, actor_id):
            raise UserError("files.provenance_invalid")
        with self.locked_workspace(workspace_id):
            existing = self._load_provenance(workspace_id, relative_path)
            if not file_provenance_is_owned_by(existing, actor_id):
                raise UserError("files.not_found")
            return self.replace_text(
                workspace_id,
                relative_path,
                old,
                new,
                expected_sha256=expected_sha256,
                provenance=provenance,
            )

    def validate_replace_text_for_actor(
        self,
        workspace_id: str,
        actor_id: str,
        relative_path: str,
        old: str,
        new: str,
        *,
        expected_sha256: str,
        provenance: WorkspaceFileProvenance,
    ) -> None:
        """Preflight an actor-owned replacement before recording dispatch."""

        if not old:
            raise UserError("files.replace_old_empty")
        if not file_provenance_is_owned_by(provenance, actor_id):
            raise UserError("files.provenance_invalid")
        with self.locked_workspace(workspace_id):
            existing = self._load_provenance(workspace_id, relative_path)
            if not file_provenance_is_owned_by(existing, actor_id):
                raise UserError("files.not_found")
            scope = self._scope_path(workspace_id)
            path = self._path(scope, relative_path)
            self._assert_regular_file(path)
            data = path.read_bytes()
            if len(data) > 200_000:
                raise UserError("files.text_too_large_to_edit")
            try:
                content = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise UserError("files.text_encoding_unsupported") from exc
            if hashlib.sha256(data).hexdigest() != expected_sha256:
                raise UserError("files.hash_conflict")
            occurrences = content.count(old)
            if occurrences != 1:
                raise UserError(
                    "files.replace_not_unique",
                    occurrences=occurrences,
                )
            replacement = content.replace(old, new, 1)
            if "\x00" in replacement:
                raise UserError("files.text_invalid")
            self._validate_import_bytes_unlocked(
                workspace_id,
                relative_path,
                replacement.encode("utf-8"),
                provenance=provenance,
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
            provenance=(
                self._load_provenance(workspace_id, relative_path)
                or unlabelled_file_provenance()
            ),
        )

    def read_for_actor(
        self,
        workspace_id: str,
        actor_id: str,
        relative_path: str,
        *,
        offset: int,
        max_characters: int,
        page_start: int = 1,
        page_count: int = 5,
    ) -> WorkspaceReadResult:
        """Read only after atomically validating the current actor authority."""

        with self.locked_workspace(workspace_id):
            scope = self._scope(workspace_id)
            path = self._path(scope, relative_path)
            self._assert_regular_file(path)
            if not file_provenance_is_owned_by(
                self._load_provenance(workspace_id, relative_path),
                actor_id,
            ):
                raise UserError("files.not_found")
            return self.read(
                workspace_id,
                relative_path,
                offset=offset,
                max_characters=max_characters,
                page_start=page_start,
                page_count=page_count,
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

    def snapshot_for_delivery_with_provenance(
        self,
        workspace_id: str,
        relative_path: str,
    ) -> tuple[str, bytes, WorkspaceFileProvenance | None]:
        """Read immutable delivery bytes and the corresponding source label."""

        with self.locked_workspace(workspace_id):
            path = self.path_for_delivery(workspace_id, relative_path)
            return (
                path.name,
                path.read_bytes(),
                self._load_provenance(workspace_id, relative_path)
                or unlabelled_file_provenance(),
            )

    def snapshot_for_actor_delivery_with_provenance(
        self,
        workspace_id: str,
        actor_id: str,
        relative_path: str,
    ) -> tuple[str, bytes, WorkspaceFileProvenance]:
        """Snapshot only bytes exclusively owned by the requesting actor."""

        with self.locked_workspace(workspace_id):
            path = self.path_for_delivery(workspace_id, relative_path)
            provenance = self._load_provenance(workspace_id, relative_path)
            if not file_provenance_is_owned_by(provenance, actor_id):
                raise UserError("files.not_found")
            assert provenance is not None
            return path.name, path.read_bytes(), provenance

    def publish_copy_for_actor(
        self,
        source_workspace_id: str,
        actor_id: str,
        source_path: str,
        *,
        expected_sha256: str,
        target_workspace_id: str,
        target_resource_id: str,
        target_display_name: str,
        target_audience_revision: str,
        reason: str,
        expires_at: str,
    ) -> WorkspaceFilePublication:
        """Create an immutable copy whose authority is limited to one target."""

        if not reason.strip() or len(reason) > 500:
            raise UserError("files.publication_reason_required")
        expiry = _publication_datetime_or_error(expires_at)
        now = datetime.now(UTC)
        if expiry <= now:
            raise UserError("files.publication_expired")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise UserError("files.hash_invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", target_audience_revision):
            raise UserError("files.publication_audience_invalid")
        filename, content, source_provenance = (
            self.snapshot_for_actor_delivery_with_provenance(
                source_workspace_id,
                actor_id,
                source_path,
            )
        )
        actual_sha256 = hashlib.sha256(content).hexdigest()
        if actual_sha256 != expected_sha256:
            raise UserError("files.hash_conflict")

        publication_id = f"pub_{uuid.uuid4().hex}"
        copy_workspace_id = f"publication:{target_workspace_id}"
        copy_path = f"{publication_id}/{filename}"
        publication = WorkspaceFilePublication(
            publication_id=publication_id,
            source_workspace_id=source_workspace_id,
            source_path=source_path,
            source_sha256=actual_sha256,
            copy_workspace_id=copy_workspace_id,
            copy_path=copy_path,
            target_kind="discord_channel",
            target_workspace_id=target_workspace_id,
            target_resource_id=target_resource_id,
            target_display_name=target_display_name,
            target_audience_revision=target_audience_revision,
            published_by_actor_id=actor_id,
            reason=reason.strip(),
            published_at=now.isoformat(),
            expires_at=expiry.isoformat(),
        )
        copy_provenance = replace(
            source_provenance,
            publication_id=publication_id,
            declassified_at=None,
            declassified_by=None,
        )
        with self.locked_workspace(copy_workspace_id):
            self._validate_import_bytes_unlocked(
                copy_workspace_id,
                copy_path,
                content,
                provenance=copy_provenance,
            )
            self._import_bytes_unlocked(
                copy_workspace_id,
                copy_path,
                content,
                provenance=copy_provenance,
            )
            try:
                self._store_publication(publication)
            except Exception:
                copy_scope = self._scope_path(copy_workspace_id)
                self._path(copy_scope, copy_path).unlink(missing_ok=True)
                self._delete_provenance(copy_workspace_id, copy_path)
                raise
        return publication

    def get_publication_for_actor(
        self,
        publication_id: str,
        actor_id: str,
    ) -> WorkspaceFilePublication:
        """Resolve publication metadata without exposing another actor's record."""

        publication = self._load_publication(publication_id)
        if publication is None or publication.published_by_actor_id != actor_id:
            raise UserError("files.publication_not_found")
        return publication

    def snapshot_publication_for_actor(
        self,
        publication_id: str,
        actor_id: str,
        *,
        target_workspace_id: str,
        target_resource_id: str,
        expected_revision: int,
    ) -> tuple[str, bytes, WorkspaceFileProvenance, WorkspaceFilePublication]:
        """Read a copy only while its exact target authority remains active."""

        if not isinstance(expected_revision, int) or isinstance(
            expected_revision,
            bool,
        ):
            raise UserError("files.publication_revision_conflict")
        publication = self.get_publication_for_actor(publication_id, actor_id)
        if (
            publication.target_workspace_id != target_workspace_id
            or publication.target_resource_id != target_resource_id
        ):
            raise UserError("files.publication_target_mismatch")
        if publication.revision != expected_revision:
            raise UserError("files.publication_revision_conflict")
        if publication.revoked_at is not None:
            raise UserError("files.publication_revoked")
        if datetime.now(UTC) >= _publication_datetime(publication.expires_at):
            raise UserError("files.publication_expired")
        with self.locked_workspace(publication.copy_workspace_id):
            latest = self.get_publication_for_actor(publication_id, actor_id)
            if latest != publication:
                raise UserError("files.publication_revision_conflict")
            path = self.path_for_delivery(
                publication.copy_workspace_id,
                publication.copy_path,
            )
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != publication.source_sha256:
                raise UserError("files.publication_copy_changed")
            provenance = self._load_provenance(
                publication.copy_workspace_id,
                publication.copy_path,
            )
            if provenance is None or provenance.publication_id != publication_id:
                raise UserError("files.provenance_invalid")
            return path.name, content, provenance, publication

    def revoke_publication_for_actor(
        self,
        publication_id: str,
        actor_id: str,
        *,
        expected_revision: int,
    ) -> tuple[WorkspaceFilePublication, bool]:
        """Revoke one publication with compare-and-set, preserving its audit row."""

        if not isinstance(expected_revision, int) or isinstance(
            expected_revision,
            bool,
        ):
            raise UserError("files.publication_revision_conflict")
        with self._provenance_lock, sqlite3.connect(
            self._provenance_path,
            timeout=5.0,
        ) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM file_publications WHERE publication_id = ?",
                (publication_id,),
            ).fetchone()
            publication = _publication_from_row(row) if row is not None else None
            if publication is None or publication.published_by_actor_id != actor_id:
                raise UserError("files.publication_not_found")
            if publication.revoked_at is not None:
                return publication, False
            if publication.revision != expected_revision:
                raise UserError("files.publication_revision_conflict")
            revoked_at = datetime.now(UTC).isoformat()
            next_revision = publication.revision + 1
            connection.execute(
                """
                UPDATE file_publications
                SET revision = ?, revoked_at = ?, revoked_by = ?
                WHERE publication_id = ? AND revision = ? AND revoked_at IS NULL
                """,
                (
                    next_revision,
                    revoked_at,
                    actor_id,
                    publication_id,
                    expected_revision,
                ),
            )
            updated = replace(
                publication,
                revision=next_revision,
                revoked_at=revoked_at,
                revoked_by=actor_id,
            )
        return updated, True

    def set_provenance(
        self,
        workspace_id: str,
        relative_path: str,
        provenance: WorkspaceFileProvenance,
    ) -> WorkspaceFileRecord:
        """Attach a source label to one existing regular file."""

        with self.locked_workspace(workspace_id):
            scope = self._scope(workspace_id)
            path = self._path(scope, relative_path)
            self._assert_regular_file(path)
            self._store_provenance(workspace_id, relative_path, provenance)
            return self._record(workspace_id, scope, path)

    def validate_path(self, workspace_id: str, relative_path: str) -> None:
        """Validate a future workspace path without creating the target file."""

        scope = self._scope_path(workspace_id)
        self._path(scope, relative_path)

    def _scope(self, workspace_id: str) -> Path:
        scope = self._scope_path(workspace_id)
        scope.mkdir(mode=0o700, parents=True, exist_ok=True)
        return scope

    def _scope_path(self, workspace_id: str) -> Path:
        """Derive one workspace path without creating it during preflight."""

        if not workspace_id or len(workspace_id) > 200:
            raise UserError("files.workspace_invalid")
        digest = hashlib.sha256(workspace_id.encode("utf-8")).hexdigest()[:24]
        return self.root / digest

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

    def _record(
        self,
        workspace_id: str,
        scope: Path,
        path: Path,
    ) -> WorkspaceFileRecord:
        data = path.read_bytes()
        relative_path = path.relative_to(scope).as_posix()
        return WorkspaceFileRecord(
            path=relative_path,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            kind=workspace_file_kind(path, data),
            provenance=(
                self._load_provenance(workspace_id, relative_path)
                or unlabelled_file_provenance()
            ),
        )

    def _initialize_provenance(self) -> None:
        with self._provenance_lock, sqlite3.connect(
            self._provenance_path,
            timeout=5.0,
        ) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS file_provenance (
                    workspace_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    owner_actor_id TEXT NOT NULL,
                    owner_actor_ids_json TEXT NOT NULL DEFAULT '[]',
                    origin_guild_id TEXT,
                    origin_channel_id TEXT,
                    origin_message_id TEXT,
                    origin_visibility TEXT NOT NULL,
                    created_task_id TEXT,
                    sensitivity TEXT NOT NULL,
                    source_resources_json TEXT NOT NULL,
                    unlabelled_input INTEGER NOT NULL DEFAULT 0,
                    sources_truncated INTEGER NOT NULL DEFAULT 0,
                    declassified_at TEXT,
                    declassified_by TEXT,
                    publication_id TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (workspace_id, relative_path)
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(file_provenance)"
                ).fetchall()
            }
            additions = {
                "owner_actor_ids_json": "TEXT NOT NULL DEFAULT '[]'",
                "unlabelled_input": "INTEGER NOT NULL DEFAULT 0",
                "sources_truncated": "INTEGER NOT NULL DEFAULT 0",
                "publication_id": "TEXT",
            }
            for name, declaration in additions.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE file_provenance ADD COLUMN {name} {declaration}"
                    )
            legacy_rows = connection.execute(
                """
                SELECT workspace_id, relative_path, owner_actor_id,
                       owner_actor_ids_json
                FROM file_provenance
                """
            ).fetchall()
            for workspace_id, relative_path, owner_actor_id, owners_json in legacy_rows:
                try:
                    owners = json.loads(str(owners_json))
                except json.JSONDecodeError:
                    owners = []
                if isinstance(owners, list) and owners:
                    continue
                if str(owner_actor_id) == "unknown":
                    connection.execute(
                        """
                        UPDATE file_provenance
                        SET owner_actor_ids_json = '[]',
                            unlabelled_input = 1,
                            sensitivity = 'uncertain'
                        WHERE workspace_id = ? AND relative_path = ?
                        """,
                        (str(workspace_id), str(relative_path)),
                    )
                    continue
                connection.execute(
                    """
                    UPDATE file_provenance
                    SET owner_actor_ids_json = ?
                    WHERE workspace_id = ? AND relative_path = ?
                    """,
                    (
                        json.dumps(
                            [str(owner_actor_id)],
                            ensure_ascii=True,
                            separators=(",", ":"),
                        ),
                        str(workspace_id),
                        str(relative_path),
                    ),
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS file_publications (
                    publication_id TEXT PRIMARY KEY,
                    source_workspace_id TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    copy_workspace_id TEXT NOT NULL,
                    copy_path TEXT NOT NULL,
                    target_kind TEXT NOT NULL,
                    target_workspace_id TEXT NOT NULL,
                    target_resource_id TEXT NOT NULL,
                    target_display_name TEXT NOT NULL,
                    target_audience_revision TEXT NOT NULL,
                    published_by_actor_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    revoked_at TEXT,
                    revoked_by TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS file_publications_actor_target
                ON file_publications (
                    published_by_actor_id,
                    target_workspace_id,
                    target_resource_id,
                    published_at
                )
                """
            )
        self._provenance_path.chmod(0o600)

    def _load_provenance(
        self,
        workspace_id: str,
        relative_path: str,
    ) -> WorkspaceFileProvenance | None:
        with self._provenance_lock, sqlite3.connect(
            self._provenance_path,
            timeout=5.0,
        ) as connection:
            row = connection.execute(
                """
                SELECT owner_actor_id, owner_actor_ids_json, origin_guild_id,
                       origin_channel_id, origin_message_id, origin_visibility,
                       created_task_id, sensitivity, source_resources_json,
                       unlabelled_input, sources_truncated, declassified_at,
                       declassified_by, publication_id
                FROM file_provenance
                WHERE workspace_id = ? AND relative_path = ?
                """,
                (workspace_id, relative_path),
            ).fetchone()
        if row is None:
            return None
        try:
            raw_owners = json.loads(str(row[1]))
            if not isinstance(raw_owners, list) or any(
                not isinstance(item, str) for item in raw_owners
            ):
                raise ValueError("invalid workspace provenance owners")
            owner_actor_ids = tuple(str(item) for item in raw_owners)
            if not owner_actor_ids and str(row[0]) != "unknown":
                owner_actor_ids = (str(row[0]),)
            raw_resources = json.loads(str(row[8]))
            source_resources = tuple(
                (
                    str(item[0]),
                    str(item[1]),
                    _source_visibility(str(item[2])),
                )
                for item in raw_resources
            )
            return WorkspaceFileProvenance(
                owner_actor_ids=owner_actor_ids,
                origin_guild_id=str(row[2]) if row[2] is not None else None,
                origin_channel_id=(
                    str(row[3]) if row[3] is not None else None
                ),
                origin_message_id=(
                    str(row[4]) if row[4] is not None else None
                ),
                origin_visibility=_provenance_visibility(str(row[5])),
                created_task_id=str(row[6]) if row[6] is not None else None,
                sensitivity=_provenance_visibility(str(row[7])),
                source_resources=source_resources,
                unlabelled_input=bool(row[9]),
                sources_truncated=bool(row[10]),
                declassified_at=str(row[11]) if row[11] is not None else None,
                declassified_by=str(row[12]) if row[12] is not None else None,
                publication_id=str(row[13]) if row[13] is not None else None,
            )
        except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise UserError("files.provenance_invalid") from exc

    def _store_provenance(
        self,
        workspace_id: str,
        relative_path: str,
        provenance: WorkspaceFileProvenance | None,
    ) -> None:
        if provenance is None:
            self._delete_provenance(workspace_id, relative_path)
            return
        payload = asdict(provenance)
        with self._provenance_lock, sqlite3.connect(
            self._provenance_path,
            timeout=5.0,
        ) as connection:
            connection.execute(
                """
                INSERT INTO file_provenance (
                    workspace_id, relative_path, owner_actor_id,
                    owner_actor_ids_json,
                    origin_guild_id, origin_channel_id, origin_message_id,
                    origin_visibility, created_task_id, sensitivity,
                    source_resources_json, unlabelled_input,
                    sources_truncated, declassified_at, declassified_by,
                    publication_id,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id, relative_path) DO UPDATE SET
                    owner_actor_id = excluded.owner_actor_id,
                    owner_actor_ids_json = excluded.owner_actor_ids_json,
                    origin_guild_id = excluded.origin_guild_id,
                    origin_channel_id = excluded.origin_channel_id,
                    origin_message_id = excluded.origin_message_id,
                    origin_visibility = excluded.origin_visibility,
                    created_task_id = excluded.created_task_id,
                    sensitivity = excluded.sensitivity,
                    source_resources_json = excluded.source_resources_json,
                    unlabelled_input = excluded.unlabelled_input,
                    sources_truncated = excluded.sources_truncated,
                    declassified_at = excluded.declassified_at,
                    declassified_by = excluded.declassified_by,
                    publication_id = excluded.publication_id,
                    updated_at = excluded.updated_at
                """,
                (
                    workspace_id,
                    relative_path,
                    (
                        payload["owner_actor_ids"][0]
                        if payload["owner_actor_ids"]
                        else "unknown"
                    ),
                    json.dumps(
                        payload["owner_actor_ids"],
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ),
                    payload["origin_guild_id"],
                    payload["origin_channel_id"],
                    payload["origin_message_id"],
                    payload["origin_visibility"],
                    payload["created_task_id"],
                    payload["sensitivity"],
                    json.dumps(
                        payload["source_resources"],
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ),
                    int(payload["unlabelled_input"]),
                    int(payload["sources_truncated"]),
                    payload["declassified_at"],
                    payload["declassified_by"],
                    payload["publication_id"],
                    datetime.now(UTC).isoformat(),
                ),
            )

    def _store_publication(self, publication: WorkspaceFilePublication) -> None:
        with self._provenance_lock, sqlite3.connect(
            self._provenance_path,
            timeout=5.0,
        ) as connection:
            connection.execute(
                """
                INSERT INTO file_publications (
                    publication_id, source_workspace_id, source_path,
                    source_sha256, copy_workspace_id, copy_path, target_kind,
                    target_workspace_id, target_resource_id, target_display_name,
                    target_audience_revision, published_by_actor_id, reason,
                    published_at, expires_at, revision, revoked_at, revoked_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    publication.publication_id,
                    publication.source_workspace_id,
                    publication.source_path,
                    publication.source_sha256,
                    publication.copy_workspace_id,
                    publication.copy_path,
                    publication.target_kind,
                    publication.target_workspace_id,
                    publication.target_resource_id,
                    publication.target_display_name,
                    publication.target_audience_revision,
                    publication.published_by_actor_id,
                    publication.reason,
                    publication.published_at,
                    publication.expires_at,
                    publication.revision,
                    publication.revoked_at,
                    publication.revoked_by,
                ),
            )

    def _load_publication(
        self,
        publication_id: str,
    ) -> WorkspaceFilePublication | None:
        if not re.fullmatch(r"pub_[0-9a-f]{32}", publication_id):
            raise UserError("files.publication_not_found")
        with self._provenance_lock, sqlite3.connect(
            self._provenance_path,
            timeout=5.0,
        ) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM file_publications WHERE publication_id = ?",
                (publication_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            return _publication_from_row(row)
        except (TypeError, ValueError) as exc:
            raise UserError("files.publication_invalid") from exc

    def _delete_provenance(
        self,
        workspace_id: str,
        relative_path: str,
    ) -> None:
        with self._provenance_lock, sqlite3.connect(
            self._provenance_path,
            timeout=5.0,
        ) as connection:
            connection.execute(
                "DELETE FROM file_provenance WHERE workspace_id = ? AND relative_path = ?",
                (workspace_id, relative_path),
            )


def _publication_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("file publication time must include a timezone")
    return parsed.astimezone(UTC)


def _publication_datetime_or_error(value: str) -> datetime:
    try:
        return _publication_datetime(value)
    except (TypeError, ValueError) as exc:
        raise UserError("files.publication_expiry_invalid") from exc


def _publication_from_row(row: sqlite3.Row) -> WorkspaceFilePublication:
    return WorkspaceFilePublication(
        publication_id=str(row["publication_id"]),
        source_workspace_id=str(row["source_workspace_id"]),
        source_path=str(row["source_path"]),
        source_sha256=str(row["source_sha256"]),
        copy_workspace_id=str(row["copy_workspace_id"]),
        copy_path=str(row["copy_path"]),
        target_kind=cast(
            WorkspacePublicationTargetKind,
            str(row["target_kind"]),
        ),
        target_workspace_id=str(row["target_workspace_id"]),
        target_resource_id=str(row["target_resource_id"]),
        target_display_name=str(row["target_display_name"]),
        target_audience_revision=str(row["target_audience_revision"]),
        published_by_actor_id=str(row["published_by_actor_id"]),
        reason=str(row["reason"]),
        published_at=str(row["published_at"]),
        expires_at=str(row["expires_at"]),
        revision=int(row["revision"]),
        revoked_at=(
            str(row["revoked_at"])
            if row["revoked_at"] is not None
            else None
        ),
        revoked_by=(
            str(row["revoked_by"])
            if row["revoked_by"] is not None
            else None
        ),
    )


def _provenance_visibility(value: str) -> WorkspaceVisibility:
    if value not in {"guild_public", "restricted", "uncertain", "actor_private"}:
        raise ValueError("invalid workspace provenance visibility")
    return cast(WorkspaceVisibility, value)


def _source_visibility(value: str) -> WorkspaceSourceVisibility:
    if value not in {"guild_public", "restricted", "uncertain"}:
        raise ValueError("invalid workspace provenance source visibility")
    return cast(WorkspaceSourceVisibility, value)


def _merge_file_provenance(
    existing: WorkspaceFileProvenance | None,
    incoming: WorkspaceFileProvenance,
) -> WorkspaceFileProvenance:
    if existing is None:
        return incoming
    all_owners = tuple(
        sorted(set((*existing.owner_actor_ids, *incoming.owner_actor_ids)))
    )
    owners_truncated = len(all_owners) > 32
    owner_actor_ids = all_owners[:32]
    all_source_resources = tuple(
        dict.fromkeys((*existing.source_resources, *incoming.source_resources))
    )
    sources_truncated = (
        existing.sources_truncated
        or incoming.sources_truncated
        or owners_truncated
        or len(all_source_resources) > 32
    )
    source_resources = all_source_resources[:32]
    unlabelled_input = (
        existing.unlabelled_input or incoming.unlabelled_input
    )
    sensitivities = {existing.sensitivity, incoming.sensitivity}
    origins_disagree = any(
        left is not None and right is not None and left != right
        for left, right in (
            (existing.origin_guild_id, incoming.origin_guild_id),
            (existing.origin_channel_id, incoming.origin_channel_id),
            (existing.origin_message_id, incoming.origin_message_id),
        )
    )
    private_origin_disagree = origins_disagree and not all_source_resources
    if (
        "uncertain" in sensitivities
        or len(owner_actor_ids) != 1
        or unlabelled_input
        or sources_truncated
        or private_origin_disagree
    ):
        sensitivity: WorkspaceVisibility = "uncertain"
    elif "actor_private" in sensitivities:
        sensitivity = "actor_private"
    elif "restricted" in sensitivities:
        sensitivity = "restricted"
    else:
        sensitivity = "guild_public"
    return WorkspaceFileProvenance(
        owner_actor_ids=owner_actor_ids,
        origin_guild_id=existing.origin_guild_id or incoming.origin_guild_id,
        origin_channel_id=(
            existing.origin_channel_id or incoming.origin_channel_id
        ),
        origin_message_id=(
            existing.origin_message_id or incoming.origin_message_id
        ),
        origin_visibility=existing.origin_visibility,
        created_task_id=existing.created_task_id or incoming.created_task_id,
        sensitivity=sensitivity,
        source_resources=source_resources,
        unlabelled_input=unlabelled_input,
        sources_truncated=sources_truncated,
    )


def merge_file_provenances(
    provenances: Iterable[WorkspaceFileProvenance | None],
) -> WorkspaceFileProvenance | None:
    """Conservatively combine every file source an arbitrary transform can read."""

    merged: WorkspaceFileProvenance | None = None
    saw_unlabelled = False
    for provenance in provenances:
        if provenance is None:
            saw_unlabelled = True
            continue
        merged = (
            provenance
            if merged is None
            else _merge_file_provenance(merged, provenance)
        )
    if saw_unlabelled:
        unknown = unlabelled_file_provenance()
        merged = (
            unknown
            if merged is None
            else _merge_file_provenance(merged, unknown)
        )
    return merged


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
