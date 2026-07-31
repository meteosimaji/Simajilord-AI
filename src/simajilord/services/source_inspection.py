"""Bounded, read-only inspection of Simajilord's own non-secret source files."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from simajilord.core.errors import UserError

_SOURCE_SUFFIXES = frozenset(
    {".css", ".html", ".js", ".json", ".md", ".mjs", ".py", ".swift", ".toml", ".yml", ".yaml"}
)
_SOURCE_EXACT_FILES = frozenset(
    {
        ".env.example",
        "README.md",
        "pyproject.toml",
        "activity/index.html",
        "activity/package.json",
        "activity/vite.config.js",
        "native/macos/TranslationHelper/Package.swift",
    }
)
_SOURCE_PREFIXES = (
    "src/simajilord/",
    "tests/",
    ".github/workflows/",
    "activity/src/",
    "activity/tests/",
    "native/macos/TranslationHelper/Sources/",
    "scripts/",
)
_SOURCE_EXCLUDED_PREFIXES = (
    "src/simajilord/activity/static/",
    "vendor/",
)
_MAX_SOURCE_FILE_BYTES = 1_000_000
_MAX_SEARCH_QUERY_CHARACTERS = 160
_MAX_SEARCH_LINE_CHARACTERS = 360


@dataclass(frozen=True, slots=True)
class SourceMatch:
    path: str
    line: int
    text: str


@dataclass(frozen=True, slots=True)
class SourceSearchResult:
    matches: tuple[SourceMatch, ...]
    searched_files: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class SourceReadResult:
    path: str
    start_line: int
    end_line: int
    total_lines: int
    content: str
    next_line: int | None
    sha256: str


class SourceInspectionService:
    """Expose only tracked-style code/docs roots, never runtime data or secrets."""

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root.resolve()

    @classmethod
    def for_runtime_file(cls, runtime_file: Path) -> SourceInspectionService:
        resolved = runtime_file.resolve()
        checkout_root = resolved.parents[2]
        if (
            (checkout_root / "pyproject.toml").is_file()
            and (checkout_root / "src" / "simajilord").is_dir()
        ):
            return cls(checkout_root)
        package_root = resolved.parent
        return cls(package_root)

    async def search(
        self,
        query: str,
        *,
        path_prefix: str | None = None,
        limit: int = 12,
    ) -> SourceSearchResult:
        normalized_query = query.strip()
        if (
            not normalized_query
            or len(normalized_query) > _MAX_SEARCH_QUERY_CHARACTERS
        ):
            raise UserError("source.query_invalid")
        if not 1 <= limit <= 30:
            raise UserError("source.limit_invalid")
        normalized_prefix = (
            _normalize_prefix(path_prefix) if path_prefix is not None else None
        )
        return await asyncio.to_thread(
            self._search,
            normalized_query,
            normalized_prefix,
            limit,
        )

    async def read(
        self,
        path: str,
        *,
        start_line: int = 1,
        max_lines: int = 120,
        max_characters: int = 8_000,
    ) -> SourceReadResult:
        if start_line < 1:
            raise UserError("source.start_line_invalid")
        if not 1 <= max_lines <= 200:
            raise UserError("source.max_lines_invalid")
        if not 200 <= max_characters <= 12_000:
            raise UserError("source.max_characters_invalid")
        selected = self._resolve_allowed(path)
        return await asyncio.to_thread(
            self._read,
            selected,
            start_line,
            max_lines,
            max_characters,
        )

    def _search(
        self,
        query: str,
        path_prefix: str | None,
        limit: int,
    ) -> SourceSearchResult:
        needle = query.casefold()
        matches: list[SourceMatch] = []
        searched_files = 0
        truncated = False
        for path in self._source_files():
            relative = path.relative_to(self.repository_root).as_posix()
            if path_prefix is not None and not relative.startswith(path_prefix):
                continue
            searched_files += 1
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if needle not in line.casefold():
                    continue
                matches.append(
                    SourceMatch(
                        path=relative,
                        line=line_number,
                        text=line.strip()[:_MAX_SEARCH_LINE_CHARACTERS],
                    )
                )
                if len(matches) >= limit:
                    truncated = True
                    return SourceSearchResult(
                        matches=tuple(matches),
                        searched_files=searched_files,
                        truncated=truncated,
                    )
        return SourceSearchResult(
            matches=tuple(matches),
            searched_files=searched_files,
            truncated=truncated,
        )

    def _read(
        self,
        path: Path,
        start_line: int,
        max_lines: int,
        max_characters: int,
    ) -> SourceReadResult:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise UserError("source.read_failed") from exc
        if len(raw) > _MAX_SOURCE_FILE_BYTES:
            raise UserError("source.file_too_large")
        try:
            text = raw.decode("utf-8")
        except UnicodeError as exc:
            raise UserError("source.encoding_unsupported") from exc
        lines = text.splitlines()
        start_index = start_line - 1
        if start_index > len(lines):
            raise UserError("source.start_line_invalid")
        selected: list[str] = []
        characters = 0
        for line in lines[start_index : start_index + max_lines]:
            added = len(line) + (1 if selected else 0)
            if selected and characters + added > max_characters:
                break
            if not selected and len(line) > max_characters:
                selected.append(line[:max_characters])
                characters = max_characters
                break
            selected.append(line)
            characters += added
        end_line = start_index + len(selected)
        next_line = end_line + 1 if end_line < len(lines) else None
        return SourceReadResult(
            path=path.relative_to(self.repository_root).as_posix(),
            start_line=start_line,
            end_line=end_line,
            total_lines=len(lines),
            content="\n".join(selected),
            next_line=next_line,
            sha256=hashlib.sha256(raw).hexdigest(),
        )

    def _source_files(self) -> tuple[Path, ...]:
        if self._checkout_layout:
            candidates = (
                *(self.repository_root / "src" / "simajilord").rglob("*"),
                *(self.repository_root / "tests").rglob("*"),
                *(self.repository_root / ".github" / "workflows").glob("*"),
                *(self.repository_root / "activity" / "src").rglob("*"),
                *(self.repository_root / "activity" / "tests").rglob("*"),
                *(
                    self.repository_root
                    / "native"
                    / "macos"
                    / "TranslationHelper"
                    / "Sources"
                ).rglob("*"),
                *(self.repository_root / "scripts").rglob("*"),
                *(self.repository_root / name for name in _SOURCE_EXACT_FILES),
            )
        else:
            candidates = tuple(self.repository_root.rglob("*"))
        allowed = {
            path.resolve()
            for path in candidates
            if path.is_file()
            and not path.is_symlink()
            and self._is_allowed(path.resolve())
            and path.stat().st_size <= _MAX_SOURCE_FILE_BYTES
        }
        return tuple(sorted(allowed))

    def _resolve_allowed(self, value: str) -> Path:
        normalized = _normalize_relative_path(value)
        candidate = (self.repository_root / normalized).resolve()
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or not self._is_allowed(candidate)
        ):
            raise UserError("source.path_forbidden")
        try:
            if candidate.stat().st_size > _MAX_SOURCE_FILE_BYTES:
                raise UserError("source.file_too_large")
        except OSError as exc:
            raise UserError("source.read_failed") from exc
        return candidate

    def _is_allowed(self, path: Path) -> bool:
        if not path.is_relative_to(self.repository_root):
            return False
        relative = path.relative_to(self.repository_root).as_posix()
        if not self._checkout_layout:
            return (
                path.suffix.casefold() in _SOURCE_SUFFIXES
                and not relative.startswith("activity/static/")
            )
        if relative in _SOURCE_EXACT_FILES:
            return True
        return (
            path.suffix.casefold() in _SOURCE_SUFFIXES
            and any(relative.startswith(prefix) for prefix in _SOURCE_PREFIXES)
            and not any(
                relative.startswith(prefix)
                for prefix in _SOURCE_EXCLUDED_PREFIXES
            )
        )

    @property
    def _checkout_layout(self) -> bool:
        return (self.repository_root / "src" / "simajilord").is_dir()


def _normalize_relative_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    pure = PurePosixPath(normalized)
    if (
        not normalized
        or pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
    ):
        raise UserError("source.path_forbidden")
    return pure.as_posix()


def _normalize_prefix(value: str) -> str:
    normalized = _normalize_relative_path(value)
    return normalized if normalized.endswith("/") else f"{normalized}/"
