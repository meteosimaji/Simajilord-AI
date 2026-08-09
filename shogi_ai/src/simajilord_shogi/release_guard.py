"""Fail-closed checks for public Meteo source and checkpoint releases.

The public source package and a public Meteo checkpoint have deliberately
different policies.  Source archives may not contain any model/data artifact.
A checkpoint candidate may contain exactly Meteo's own inference weights, but
it must carry auditable lineage and may not expose local teacher material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import tarfile
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Any, cast
from zipfile import ZipFile, ZipInfo

from .rights_lineage import (
    DERIVED_TEACHER_SOURCE_IDS,
    legacy_ancestor_restriction_id,
    lineage_has_teacher_evidence,
    validate_rights_restriction_summary,
)

RECEIPT_SCHEMA = "meteo-rights-holder-publication-receipt-v1"
RECEIPT_SCOPE = "publish-derived-meteo-checkpoint"

# These include the pre-existing package denylist plus formats used by the
# newly supplied engine, evaluation, and opening-book artifacts.
FORBIDDEN_ARTIFACT_SUFFIXES = frozenset(
    {
        ".7z",
        ".bin",
        ".ckpt",
        ".engine",
        ".hcpe",
        ".nn",
        ".npz",
        ".onnx",
        ".pb",
        ".psv",
        ".pt",
        ".pth",
        ".safetensors",
        ".tflite",
        ".ybb",
    }
)
FORBIDDEN_DATA_SUFFIXES = frozenset({".jsonl", ".csa", ".kif", ".ki2", ".psn"})
FORBIDDEN_PUBLIC_PARTS = frozenset(
    {
        "artifacts",
        "datasets",
        "sealed",
        "sealed_final_test",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
    }
)
LOCAL_ONLY_PARTS = frozenset({"dist", "build"})
TEXT_SUFFIXES = frozenset(
    {
        ".cfg",
        ".csv",
        ".html",
        ".ini",
        ".json",
        ".jsonl",
        ".log",
        ".md",
        ".out",
        ".py",
        ".rst",
        ".toml",
        ".tsv",
        ".txt",
        ".yaml",
        ".yml",
    }
)
DATA_BEARING_TEXT_SUFFIXES = TEXT_SUFFIXES - {".py"}
MAX_INSPECTED_TEXT_BYTES = 4 * 1024 * 1024
MAX_PUBLIC_SOURCE_FILE_BYTES = 5 * 1024 * 1024
MAX_PUBLIC_SOURCE_FILES = 200
MAX_PUBLIC_ARCHIVE_UNPACKED_BYTES = 20 * 1024 * 1024
TEXT_FILENAMES = frozenset(
    {"license", "notice", "thanks", "acknowledgements", "acknowledgments", "credits"}
)

_SHA256_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_URL_RE = re.compile(r"(?i)(?:https?://|www\.)\S+")
_WINDOWS_ABSOLUTE_RE = re.compile(r"(?i)(?:^|[\s'\"(])[a-z]:[\\/]")
_THANKS_HEADING_RE = re.compile(
    r"(?im)^(?P<marks>#{1,6})\s*(?:thanks?|acknowledg(?:e)?ments?|credits?|謝辞)　*\s*$"
)
_ANY_HEADING_RE = re.compile(r"(?m)^(?P<marks>#{1,6})\s+.*$")
_RESTRICTED_MODES = frozenset(
    {
        "limited_local",
        "limited-local",
        "user_authorized_local_analysis_only",
        "blocked_pending_rights_holder_permission",
    }
)
_PUBLICATION_DECISION_KEYS = frozenset(
    {
        "output_only_meteo_publication",
        "derived_checkpoint_redistribution",
        "derived_checkpoint_redistribution_allowed",
    }
)
_NON_PUBLIC_DECISIONS = frozenset(
    {
        "conditional_gpl",
        "excluded_paid",
        "limited",
        "not_applicable",
        "not_approved",
    }
)


@dataclass(frozen=True, slots=True)
class ReleaseFinding:
    """One deterministic reason a candidate cannot be public."""

    path: str
    rule: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LineageRestriction:
    """A non-public ancestor marker that requires rights-holder permission."""

    restriction_id: str
    code: str
    pointer: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReleaseAudit:
    """A successful, machine-readable checkpoint release decision."""

    checkpoint_sha256: str
    files: int
    bytes: int
    restrictions: tuple[LineageRestriction, ...]
    rights_holder_receipt_used: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "files": self.files,
            "bytes": self.bytes,
            "restrictions": [item.to_dict() for item in self.restrictions],
            "rights_holder_receipt_used": self.rights_holder_receipt_used,
        }


class ReleaseGuardError(ValueError):
    """Raised with all findings instead of stopping at the first leak."""

    def __init__(self, findings: Sequence[ReleaseFinding]) -> None:
        ordered = tuple(sorted(findings, key=lambda item: (item.path, item.rule, item.detail)))
        if not ordered:
            raise ValueError("ReleaseGuardError requires at least one finding")
        self.findings = ordered
        summary = "; ".join(f"{item.path}: {item.rule} ({item.detail})" for item in ordered)
        super().__init__(f"public release guard rejected the candidate: {summary}")


def _parts(path: PurePosixPath) -> tuple[str, ...]:
    return tuple(part.casefold() for part in path.parts if part not in {"", "."})


def _is_text_path(path: PurePosixPath) -> bool:
    return path.suffix.casefold() in TEXT_SUFFIXES or path.name.casefold() in TEXT_FILENAMES


def _normalized_bundle_path(path: PurePosixPath) -> str:
    return unicodedata.normalize("NFC", path.as_posix()).casefold()


def _forbidden_suffix(path: PurePosixPath, *, allow_meteo_weights: bool) -> str | None:
    suffix = path.suffix.casefold()
    if allow_meteo_weights and path.name == "weights.safetensors":
        return None
    if suffix in FORBIDDEN_ARTIFACT_SUFFIXES:
        return suffix
    if suffix in FORBIDDEN_DATA_SUFFIXES:
        return suffix
    return None


def _path_findings(path: PurePosixPath, *, allow_meteo_weights: bool) -> list[ReleaseFinding]:
    findings: list[ReleaseFinding] = []
    if path.is_absolute() or ".." in path.parts or "\\" in path.as_posix():
        findings.append(
            ReleaseFinding(
                path.as_posix(),
                "unsafe-path",
                "public bundle path must be a normalized relative POSIX path",
            )
        )
    normalized_parts = _parts(path)
    forbidden_parts = sorted(FORBIDDEN_PUBLIC_PARTS.intersection(normalized_parts))
    if forbidden_parts:
        findings.append(
            ReleaseFinding(
                path.as_posix(),
                "generated-or-sealed-path",
                f"forbidden path component(s): {', '.join(forbidden_parts)}",
            )
        )
    suffix = _forbidden_suffix(path, allow_meteo_weights=allow_meteo_weights)
    if suffix is not None:
        findings.append(
            ReleaseFinding(
                path.as_posix(),
                "model-or-data-artifact",
                f"forbidden public suffix: {suffix}",
            )
        )
    return findings


def _private_location_findings(path: str, text: str) -> list[ReleaseFinding]:
    findings: list[ReleaseFinding] = []
    private_tmp = "/" + "private" + "/tmp"
    user_home = "/" + "Users" + "/"
    if private_tmp in text:
        findings.append(
            ReleaseFinding(path, "local-absolute-path", "contains a private temporary path")
        )
    if user_home in text or _WINDOWS_ABSOLUTE_RE.search(text):
        findings.append(
            ReleaseFinding(path, "local-absolute-path", "contains a user or drive absolute path")
        )
    return findings


def _restricted_distribution_findings(path: str, text: str) -> list[ReleaseFinding]:
    restricted_hosts = (
        "storage" + ".yaneu" + ".com",
        "fanbox" + ".cc",
    )
    if not any(host in text.casefold() for host in restricted_hosts):
        return []
    return [
        ReleaseFinding(
            path,
            "restricted-download-link",
            "contains a direct restricted-distribution storage link",
        )
    ]


def _looks_like_suisho_raw_record(path: PurePosixPath, text: str) -> bool:
    lowered_path = path.as_posix().casefold()
    lowered = text.casefold()
    teacher_marker = "suisho11plus-wcsc36-20260525-local"
    if teacher_marker not in lowered:
        return False
    if path.suffix.casefold() == ".jsonl":
        return True
    record_path_markers = ("startup", "provenance", "replay", "transcript")
    if any(marker in lowered_path for marker in record_path_markers):
        return True
    raw_field_markers = (
        '"startup_provenance"',
        '"startup_transcript"',
        '"stderr_lines"',
        '"stdout_lines"',
        '"engine_working_directory"',
        '"output_replay"',
        '"source_replay"',
    )
    return any(marker in lowered for marker in raw_field_markers)


def _looks_like_generated_private_record(path: PurePosixPath, text: str) -> bool:
    if path.suffix.casefold() not in DATA_BEARING_TEXT_SUFFIXES:
        return False
    lowered_path = path.as_posix().casefold()
    generated_path_markers = (
        ".startup.",
        ".provenance.",
        "startup-provenance",
        "startup_transcript",
        "teacher-replay",
    )
    if any(marker in lowered_path for marker in generated_path_markers):
        return True
    lowered = text.casefold()
    if "meteo-usi-startup-provenance-v1" in lowered:
        return True
    return "suisho11plus-wcsc36-20260525-local" in lowered and any(
        marker in lowered
        for marker in (
            '"artifacts"',
            '"engine_options"',
            '"resolved_executable"',
            '"samples"',
        )
    )


def _thanks_sections(text: str) -> tuple[str, ...]:
    sections: list[str] = []
    for match in _THANKS_HEADING_RE.finditer(text):
        level = len(match.group("marks"))
        end = len(text)
        for heading in _ANY_HEADING_RE.finditer(text, match.end()):
            if len(heading.group("marks")) <= level:
                end = heading.start()
                break
        sections.append(text[match.end() : end])
    return tuple(sections)


def _thanks_findings(path: PurePosixPath, text: str) -> list[ReleaseFinding]:
    is_thanks_file = any(
        marker in path.stem.casefold() for marker in ("thank", "acknowledg", "credit", "謝辞")
    )
    sections = (text,) if is_thanks_file else _thanks_sections(text)
    findings: list[ReleaseFinding] = []
    for section in sections:
        if _URL_RE.search(section):
            findings.append(
                ReleaseFinding(
                    path.as_posix(),
                    "thanks-metadata",
                    "Thanks/acknowledgements may name teachers and authors, not URLs",
                )
            )
        if (
            _SHA256_RE.search(section)
            or "sha256" in section.casefold()
            or "sha-256" in section.casefold()
        ):
            findings.append(
                ReleaseFinding(
                    path.as_posix(),
                    "thanks-metadata",
                    "Thanks/acknowledgements may not contain artifact hashes",
                )
            )
        findings.extend(_private_location_findings(path.as_posix(), section))
        lowered = section.casefold()
        if any(suffix in lowered for suffix in FORBIDDEN_ARTIFACT_SUFFIXES):
            findings.append(
                ReleaseFinding(
                    path.as_posix(),
                    "thanks-metadata",
                    "Thanks/acknowledgements may not name model/archive files",
                )
            )
    return findings


def _text_findings(path: PurePosixPath, raw: bytes) -> list[ReleaseFinding]:
    if len(raw) > MAX_INSPECTED_TEXT_BYTES:
        return [
            ReleaseFinding(
                path.as_posix(),
                "oversized-public-text",
                f"cannot safely inspect {len(raw)} bytes",
            )
        ]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return [
            ReleaseFinding(
                path.as_posix(),
                "non-utf8-public-text",
                "text-like release file is not UTF-8",
            )
        ]
    findings = _restricted_distribution_findings(path.as_posix(), text)
    findings.extend(_private_location_findings(path.as_posix(), text))
    if path.suffix.casefold() in DATA_BEARING_TEXT_SUFFIXES:
        if _looks_like_suisho_raw_record(path, text):
            findings.append(
                ReleaseFinding(
                    path.as_posix(),
                    "restricted-teacher-record",
                    "contains raw Suisho11Plus startup/replay/provenance data",
                )
            )
        elif _looks_like_generated_private_record(path, text):
            findings.append(
                ReleaseFinding(
                    path.as_posix(),
                    "generated-private-record",
                    "contains startup/replay/provenance data that is not a public artifact",
                )
            )
    if path.suffix.casefold() in {".md", ".rst", ".txt"} or path.name.casefold() in TEXT_FILENAMES:
        findings.extend(_thanks_findings(path, text))
    return findings


def _raise_findings(findings: Sequence[ReleaseFinding]) -> None:
    if findings:
        raise ReleaseGuardError(findings)


def scan_source_tree(root: Path) -> int:
    """Scan public source inputs while excluding generated local-only roots."""

    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"source root must be a non-symlink directory: {root}")
    findings: list[ReleaseFinding] = []
    scanned = 0
    seen_paths: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        normalized_parts = _parts(relative)
        if set(normalized_parts).intersection(FORBIDDEN_PUBLIC_PARTS | LOCAL_ONLY_PARTS):
            continue
        if path.is_symlink():
            findings.append(
                ReleaseFinding(
                    relative.as_posix(),
                    "symlink",
                    "source releases cannot contain symlinks",
                )
            )
            continue
        if not path.is_file():
            continue
        scanned += 1
        normalized = _normalized_bundle_path(relative)
        previous = seen_paths.setdefault(normalized, relative.as_posix())
        if previous != relative.as_posix():
            findings.append(
                ReleaseFinding(
                    relative.as_posix(),
                    "path-collision",
                    f"case/Unicode-normalized path collides with {previous}",
                )
            )
        if path.stat().st_size > MAX_PUBLIC_SOURCE_FILE_BYTES:
            findings.append(
                ReleaseFinding(
                    relative.as_posix(),
                    "oversized-source-file",
                    f"public source file is {path.stat().st_size} bytes",
                )
            )
        findings.extend(_path_findings(relative, allow_meteo_weights=False))
        if _is_text_path(relative):
            findings.extend(_text_findings(relative, path.read_bytes()))
    if scanned > MAX_PUBLIC_SOURCE_FILES:
        findings.append(
            ReleaseFinding(
                ".",
                "source-scope-expanded",
                f"public source contains {scanned} files; limit is {MAX_PUBLIC_SOURCE_FILES}",
            )
        )
    _raise_findings(findings)
    return scanned


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    """Run one read-only Git query and fail closed on an unreadable index."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise ValueError("Git is required to audit the repository index") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git repository index query failed: {detail or 'unknown error'}")
    return completed.stdout


def _repository_relative_path(
    raw_path: bytes,
    *,
    source_prefix: PurePosixPath,
) -> tuple[PurePosixPath | None, ReleaseFinding | None]:
    try:
        decoded = raw_path.decode("utf-8")
    except UnicodeDecodeError:
        return None, ReleaseFinding(
            repr(raw_path),
            "non-utf8-path",
            "tracked public paths must be valid UTF-8",
        )
    repository_path = PurePosixPath(decoded)
    try:
        relative = (
            repository_path
            if source_prefix == PurePosixPath(".")
            else repository_path.relative_to(source_prefix)
        )
    except ValueError:
        return None, ReleaseFinding(
            repository_path.as_posix(),
            "index-path-outside-source",
            "Git returned a tracked path outside the audited source root",
        )
    if relative == PurePosixPath("."):
        return None, ReleaseFinding(
            repository_path.as_posix(),
            "invalid-index-path",
            "tracked entry resolves to the source directory rather than a file",
        )
    return relative, None


def _repository_path_findings(path: PurePosixPath) -> list[ReleaseFinding]:
    """Apply source-release path policy with one documentation-only exception."""

    if path.as_posix() == "datasets/README.md":
        findings = _path_findings(path, allow_meteo_weights=False)
        return [finding for finding in findings if finding.rule != "generated-or-sealed-path"]
    findings = _path_findings(path, allow_meteo_weights=False)
    local_parts = sorted(LOCAL_ONLY_PARTS.intersection(_parts(path)))
    if local_parts:
        findings.append(
            ReleaseFinding(
                path.as_posix(),
                "local-build-path",
                f"tracked local build component(s): {', '.join(local_parts)}",
            )
        )
    return findings


def _indexed_blob(
    repository: Path,
    *,
    object_id: str,
    path: PurePosixPath,
) -> tuple[int | None, bytes | None, list[ReleaseFinding]]:
    findings: list[ReleaseFinding] = []
    try:
        raw_size = _git_bytes(repository, "cat-file", "-s", object_id)
        size = int(raw_size.strip())
    except (ValueError, UnicodeError):
        return None, None, [
            ReleaseFinding(
                path.as_posix(),
                "unreadable-index-blob",
                "tracked Git object size is unavailable or invalid",
            )
        ]
    if size < 0:
        return None, None, [
            ReleaseFinding(
                path.as_posix(),
                "unreadable-index-blob",
                "tracked Git object has an invalid negative size",
            )
        ]
    if size > MAX_PUBLIC_SOURCE_FILE_BYTES:
        findings.append(
            ReleaseFinding(
                path.as_posix(),
                "oversized-source-file",
                f"tracked public source blob is {size} bytes",
            )
        )
    if not _is_text_path(path):
        return size, None, findings
    if size > MAX_INSPECTED_TEXT_BYTES:
        findings.append(
            ReleaseFinding(
                path.as_posix(),
                "oversized-public-text",
                f"cannot safely inspect {size} indexed bytes",
            )
        )
        return size, None, findings
    try:
        raw = _git_bytes(repository, "cat-file", "blob", object_id)
    except ValueError:
        findings.append(
            ReleaseFinding(
                path.as_posix(),
                "unreadable-index-blob",
                "tracked Git blob contents are unavailable",
            )
        )
        return size, None, findings
    if len(raw) != size:
        findings.append(
            ReleaseFinding(
                path.as_posix(),
                "unreadable-index-blob",
                "tracked Git blob size changed while auditing",
            )
        )
        return size, None, findings
    return size, raw, findings


def scan_repository_index(source_root: Path) -> int:
    """Audit the exact Git-index blobs that would be published by a commit.

    ``scan_source_tree`` deliberately skips local generated roots so developers
    can keep private training artifacts beside the code.  This complementary
    mode closes the ``git add -f`` boundary: ignored material, staged-only
    content, symlinks, and gitlinks are inspected from the index rather than
    trusted from the current working tree.
    """

    requested_root = source_root.expanduser()
    if requested_root.is_symlink() or not requested_root.is_dir():
        raise ValueError(f"source root must be a non-symlink directory: {requested_root}")
    source_root = requested_root.resolve()
    repository_output = _git_bytes(source_root, "rev-parse", "--show-toplevel")
    try:
        repository = Path(repository_output.rstrip(b"\n").decode("utf-8")).resolve()
    except UnicodeDecodeError as error:
        raise ValueError("Git repository root must be valid UTF-8") from error
    if not repository.is_dir() or not source_root.is_relative_to(repository):
        raise ValueError("source root must be inside the resolved Git repository")
    relative_root = source_root.relative_to(repository)
    source_prefix = PurePosixPath(relative_root.as_posix())
    pathspec = "." if relative_root == Path(".") else relative_root.as_posix()
    raw_index = _git_bytes(repository, "ls-files", "--stage", "-z", "--", pathspec)

    findings: list[ReleaseFinding] = []
    seen_paths: dict[str, str] = {}
    scanned = 0
    for raw_entry in raw_index.split(b"\0"):
        if not raw_entry:
            continue
        metadata, separator, raw_path = raw_entry.partition(b"\t")
        if not separator:
            findings.append(
                ReleaseFinding(
                    repr(raw_entry[:120]),
                    "invalid-index-entry",
                    "tracked entry is missing the NUL-safe stage/path separator",
                )
            )
            continue
        try:
            mode_bytes, object_bytes, stage_bytes = metadata.split(b" ")
            mode = mode_bytes.decode("ascii")
            object_id = object_bytes.decode("ascii")
            stage = int(stage_bytes)
        except (UnicodeDecodeError, ValueError):
            findings.append(
                ReleaseFinding(
                    repr(raw_path),
                    "invalid-index-entry",
                    "tracked mode, object ID, or merge stage is malformed",
                )
            )
            continue
        relative, path_finding = _repository_relative_path(
            raw_path,
            source_prefix=source_prefix,
        )
        if path_finding is not None:
            findings.append(path_finding)
            continue
        assert relative is not None
        scanned += 1
        normalized = _normalized_bundle_path(relative)
        previous = seen_paths.setdefault(normalized, relative.as_posix())
        if previous != relative.as_posix():
            findings.append(
                ReleaseFinding(
                    relative.as_posix(),
                    "path-collision",
                    f"case/Unicode-normalized tracked path collides with {previous}",
                )
            )
        if stage != 0:
            findings.append(
                ReleaseFinding(
                    relative.as_posix(),
                    "unmerged-index-entry",
                    f"tracked path is present at unresolved merge stage {stage}",
                )
            )
        findings.extend(_repository_path_findings(relative))
        if stage != 0:
            continue
        if mode == "120000":
            findings.append(
                ReleaseFinding(
                    relative.as_posix(),
                    "symlink",
                    "tracked source releases cannot contain symlinks",
                )
            )
            continue
        if mode == "160000":
            findings.append(
                ReleaseFinding(
                    relative.as_posix(),
                    "gitlink",
                    "tracked source releases cannot contain gitlinks/submodules",
                )
            )
            continue
        if mode not in {"100644", "100755"}:
            findings.append(
                ReleaseFinding(
                    relative.as_posix(),
                    "unsupported-index-mode",
                    f"tracked source has unsupported Git mode {mode!r}",
                )
            )
            continue
        if _SHA256_RE.fullmatch(object_id) is None and not re.fullmatch(
            r"[0-9a-fA-F]{40}", object_id
        ):
            findings.append(
                ReleaseFinding(
                    relative.as_posix(),
                    "invalid-index-object",
                    "tracked source object ID is not a Git SHA-1/SHA-256 identifier",
                )
            )
            continue
        _size, raw, blob_findings = _indexed_blob(
            repository,
            object_id=object_id,
            path=relative,
        )
        findings.extend(blob_findings)
        if raw is not None:
            findings.extend(_text_findings(relative, raw))
    if scanned > MAX_PUBLIC_SOURCE_FILES:
        findings.append(
            ReleaseFinding(
                ".",
                "source-scope-expanded",
                (
                    f"tracked public source contains {scanned} files; limit is "
                    f"{MAX_PUBLIC_SOURCE_FILES}"
                ),
            )
        )
    if _git_bytes(repository, "ls-files", "--stage", "-z", "--", pathspec) != raw_index:
        findings.append(
            ReleaseFinding(
                ".",
                "index-changed-during-audit",
                "Git index changed while tracked public blobs were being inspected",
            )
        )
    _raise_findings(findings)
    return scanned


def _zip_member_is_symlink(member: ZipInfo) -> bool:
    mode = (member.external_attr >> 16) & 0o177777
    return stat.S_ISLNK(mode)


def _inspect_archive_member(
    member_path: PurePosixPath,
    *,
    size: int,
    stream: IO[bytes] | None,
) -> list[ReleaseFinding]:
    findings = _path_findings(member_path, allow_meteo_weights=False)
    if size > MAX_PUBLIC_SOURCE_FILE_BYTES:
        findings.append(
            ReleaseFinding(
                member_path.as_posix(),
                "oversized-archive-member",
                f"public build member is {size} bytes",
            )
        )
    if stream is not None and _is_text_path(member_path):
        if size > MAX_INSPECTED_TEXT_BYTES:
            findings.append(
                ReleaseFinding(
                    member_path.as_posix(),
                    "oversized-public-text",
                    f"cannot safely inspect {size} bytes",
                )
            )
        else:
            findings.extend(_text_findings(member_path, stream.read(MAX_INSPECTED_TEXT_BYTES + 1)))
    return findings


def scan_build_archive(archive: Path) -> int:
    """Inspect a wheel/zip or tar source archive without extracting it."""

    archive = archive.expanduser().resolve()
    if not archive.is_file() or archive.is_symlink():
        raise ValueError(f"build archive must be a non-symlink regular file: {archive}")
    findings: list[ReleaseFinding] = []
    scanned = 0
    unpacked_bytes = 0
    if archive.suffix.casefold() in {".whl", ".zip"}:
        with ZipFile(archive) as bundle:
            zip_names: dict[str, str] = {}
            for zip_member in bundle.infolist():
                path = PurePosixPath(zip_member.filename)
                normalized = _normalized_bundle_path(path)
                previous = zip_names.get(normalized)
                if previous is not None:
                    findings.append(
                        ReleaseFinding(
                            path.as_posix(),
                            "duplicate-member",
                            f"duplicate or normalized-colliding archive path: {previous}",
                        )
                    )
                else:
                    zip_names[normalized] = path.as_posix()
                if zip_member.is_dir():
                    continue
                scanned += 1
                unpacked_bytes += zip_member.file_size
                if _zip_member_is_symlink(zip_member):
                    findings.append(
                        ReleaseFinding(path.as_posix(), "symlink", "archive member is a symlink")
                    )
                    continue
                with bundle.open(zip_member) as zip_stream:
                    findings.extend(
                        _inspect_archive_member(path, size=zip_member.file_size, stream=zip_stream)
                    )
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as bundle:
            tar_names: dict[str, str] = {}
            for tar_member in bundle.getmembers():
                path = PurePosixPath(tar_member.name)
                normalized = _normalized_bundle_path(path)
                previous = tar_names.get(normalized)
                if previous is not None:
                    findings.append(
                        ReleaseFinding(
                            path.as_posix(),
                            "duplicate-member",
                            f"duplicate or normalized-colliding archive path: {previous}",
                        )
                    )
                else:
                    tar_names[normalized] = path.as_posix()
                if tar_member.issym() or tar_member.islnk():
                    findings.append(
                        ReleaseFinding(path.as_posix(), "symlink", "archive member is a link")
                    )
                    continue
                if not tar_member.isfile():
                    continue
                scanned += 1
                unpacked_bytes += tar_member.size
                tar_stream = bundle.extractfile(tar_member)
                if tar_stream is None:
                    findings.append(
                        ReleaseFinding(
                            path.as_posix(),
                            "unreadable-member",
                            "archive file has no stream",
                        )
                    )
                    continue
                with tar_stream:
                    findings.extend(
                        _inspect_archive_member(path, size=tar_member.size, stream=tar_stream)
                    )
    else:
        raise ValueError(f"unsupported build archive format: {archive}")
    if scanned > MAX_PUBLIC_SOURCE_FILES:
        findings.append(
            ReleaseFinding(
                archive.name,
                "archive-scope-expanded",
                f"archive contains {scanned} files; limit is {MAX_PUBLIC_SOURCE_FILES}",
            )
        )
    if unpacked_bytes > MAX_PUBLIC_ARCHIVE_UNPACKED_BYTES:
        findings.append(
            ReleaseFinding(
                archive.name,
                "archive-scope-expanded",
                (
                    f"archive expands to {unpacked_bytes} bytes; limit is "
                    f"{MAX_PUBLIC_ARCHIVE_UNPACKED_BYTES}"
                ),
            )
        )
    _raise_findings(findings)
    return scanned


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _restriction(*, code: str, pointer: str, detail: str) -> LineageRestriction:
    payload = f"{code}\0{pointer}\0{detail}".encode()
    return LineageRestriction(
        restriction_id=hashlib.sha256(payload).hexdigest(),
        code=code,
        pointer=pointer,
        detail=detail,
    )


def _teacher_rights_restriction(source: str, *, pointer: str) -> LineageRestriction | None:
    from .model_rights import RightsDecision, model_rights

    try:
        rights = model_rights(source)
    except ValueError:
        return _restriction(
            code="unknown-teacher-rights",
            pointer=pointer,
            detail=f"teacher source {source!r} has no reviewed publication decision",
        )
    if rights.output_only_meteo_publication == RightsDecision.ALLOWED:
        return None
    return _restriction(
        code="teacher-publication-not-approved",
        pointer=pointer,
        detail=(
            f"teacher {source!r} publication decision is "
            f"{rights.output_only_meteo_publication.value}"
        ),
    )


def find_lineage_restrictions(lineage: Mapping[str, Any]) -> tuple[LineageRestriction, ...]:
    """Find non-public markers recursively, including nested checkpoint ancestors."""

    restrictions: dict[str, LineageRestriction] = {}

    def add(item: LineageRestriction) -> None:
        restrictions.setdefault(item.restriction_id, item)

    if lineage_has_teacher_evidence(lineage) and not isinstance(
        lineage.get("rights_restriction_summary"), dict
    ):
        add(
            LineageRestriction(
                restriction_id=legacy_ancestor_restriction_id(),
                code="legacy-teacher-lineage-summary-missing",
                pointer="/lineage/rights_restriction_summary",
                detail=(
                    "teacher-bearing training lineage predates the privacy-safe rights "
                    "summary and must be migrated or covered by an exact receipt"
                ),
            )
        )

    def add_summary_restrictions(
        raw_summary: Mapping[str, object], *, pointer: str
    ) -> bool:
        summary = validate_rights_restriction_summary(raw_summary)
        source_details: dict[str, tuple[str, str]] = {}
        for raw_source in cast(list[dict[str, object]], summary["sources"]):
            rights_id = cast(str, raw_source["rights_id"])
            decision = cast(str, raw_source["decision"])
            for restriction_id in cast(list[str], raw_source["restriction_ids"]):
                source_details[restriction_id] = (rights_id, decision)
        inherited = set(cast(list[str], summary["inherited_restriction_ids"]))
        unattributed = set(cast(list[str], summary["unattributed_restriction_ids"]))
        for index, restriction_id in enumerate(
            cast(list[str], summary["restriction_ids"])
        ):
            restriction_pointer = f"{pointer}/restriction_ids/{index}"
            if restriction_id in source_details:
                rights_id, decision = source_details[restriction_id]
                add(
                    LineageRestriction(
                        restriction_id=restriction_id,
                        code="summarized-teacher-publication-restricted",
                        pointer=restriction_pointer,
                        detail=(
                            f"teacher {rights_id!r} publication decision is {decision}"
                        ),
                    )
                )
            elif restriction_id in inherited:
                add(
                    LineageRestriction(
                        restriction_id=restriction_id,
                        code="inherited-teacher-publication-restriction",
                        pointer=restriction_pointer,
                        detail="checkpoint inherits a teacher-publication restriction",
                    )
                )
            elif restriction_id in unattributed:
                add(
                    LineageRestriction(
                        restriction_id=restriction_id,
                        code="unattributed-sidecar-publication-block",
                        pointer=restriction_pointer,
                        detail="teacher sidecar blocks publication without a rights ID",
                    )
                )
            else:  # pragma: no cover - canonical validation makes this unreachable.
                raise ValueError("rights summary contains an unexplained restriction ID")
        return bool(cast(list[dict[str, object]], summary["sources"]))

    def visit(
        value: object,
        pointer: str,
        key: str | None = None,
        *,
        derived_source_resolved: bool = False,
    ) -> None:
        normalized_key = key.casefold() if key is not None else None
        if normalized_key == "rights_restriction_summary":
            if not isinstance(value, dict):
                raise ValueError("checkpoint rights restriction summary must be an object")
            add_summary_restrictions(value, pointer=pointer)
            return
        if normalized_key == "publication_allowed" and value is False:
            add(
                _restriction(
                    code="publication-explicitly-blocked",
                    pointer=pointer,
                    detail="ancestor sets publication_allowed=false",
                )
            )
        if normalized_key is not None and "local_only" in normalized_key and value is True:
            add(
                _restriction(
                    code="limited-local-ancestor",
                    pointer=pointer,
                    detail=f"ancestor sets {key}=true",
                )
            )
        if normalized_key in _PUBLICATION_DECISION_KEYS:
            if isinstance(value, bool) and not value:
                add(
                    _restriction(
                        code="publication-decision-not-approved",
                        pointer=pointer,
                        detail=f"ancestor sets {key}=false",
                    )
                )
            elif isinstance(value, str) and value.casefold() in _NON_PUBLIC_DECISIONS:
                add(
                    _restriction(
                        code="publication-decision-not-approved",
                        pointer=pointer,
                        detail=f"ancestor sets {key}={value}",
                    )
                )
        if isinstance(value, str) and value.casefold() in _RESTRICTED_MODES:
            add(
                _restriction(
                    code="limited-local-ancestor",
                    pointer=pointer,
                    detail=f"ancestor contains restricted mode {value!r}",
                )
            )
        if normalized_key == "teacher_source" and isinstance(value, str):
            if value in DERIVED_TEACHER_SOURCE_IDS and derived_source_resolved:
                return
            restriction = _teacher_rights_restriction(value, pointer=pointer)
            if restriction is not None:
                add(restriction)
        if normalized_key == "teacher_sources" and isinstance(value, list):
            for index, item in enumerate(value):
                if not isinstance(item, str):
                    add(
                        _restriction(
                            code="invalid-teacher-rights-lineage",
                            pointer=f"{pointer}/{index}",
                            detail="teacher source is not a string",
                        )
                    )
                    continue
                if item in DERIVED_TEACHER_SOURCE_IDS and derived_source_resolved:
                    continue
                restriction = _teacher_rights_restriction(item, pointer=f"{pointer}/{index}")
                if restriction is not None:
                    add(restriction)
        if isinstance(value, dict):
            local_derived_source_resolved = derived_source_resolved
            raw_summary = value.get("rights_restriction_summary")
            if isinstance(raw_summary, dict):
                canonical_summary = validate_rights_restriction_summary(raw_summary)
                local_derived_source_resolved = bool(canonical_summary["sources"])
            for child_key in sorted(value):
                if not isinstance(child_key, str):
                    add(
                        _restriction(
                            code="invalid-lineage-key",
                            pointer=pointer,
                            detail="lineage object contains a non-string key",
                        )
                    )
                    continue
                visit(
                    value[child_key],
                    f"{pointer}/{_json_pointer_token(child_key)}",
                    child_key,
                    derived_source_resolved=local_derived_source_resolved,
                )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(
                    item,
                    f"{pointer}/{index}",
                    derived_source_resolved=derived_source_resolved,
                )

    visit(dict(lineage), "/lineage")
    return tuple(sorted(restrictions.values(), key=lambda item: item.restriction_id))


def checkpoint_release_identity(directory: Path) -> tuple[str, int, int]:
    """Hash the exact metadata and inference-weight bytes bound by a receipt."""

    digest = hashlib.sha256()
    files = 0
    byte_count = 0
    for name in ("metadata.json", "weights.safetensors"):
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"checkpoint release requires a non-symlink {name}")
        relative = name.encode("utf-8")
        contents_size = path.stat().st_size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(contents_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files += 1
        byte_count += contents_size
    return digest.hexdigest(), files, byte_count


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable UTF-8 JSON: {path}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    return cast(dict[str, Any], value)


def _validate_permission_receipt(
    receipt_path: Path,
    *,
    checkpoint_sha256: str,
    restrictions: Sequence[LineageRestriction],
) -> None:
    receipt_path = receipt_path.expanduser().resolve()
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError("rights-holder receipt must be a non-symlink regular file")
    receipt = _read_json_object(receipt_path, label="rights-holder receipt")
    expected_keys = {
        "schema",
        "scope",
        "checkpoint_sha256",
        "rights_holder",
        "rights_holder_permission",
        "evidence_sha256",
        "restriction_ids",
        "issued_at",
    }
    if set(receipt) != expected_keys:
        raise ValueError(
            "rights-holder receipt fields must exactly match the publication receipt schema"
        )
    if receipt["schema"] != RECEIPT_SCHEMA or receipt["scope"] != RECEIPT_SCOPE:
        raise ValueError("rights-holder receipt schema or scope is invalid")
    if receipt["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("rights-holder receipt does not identify this exact checkpoint")
    if receipt["rights_holder_permission"] is not True:
        raise ValueError("rights-holder receipt does not grant publication permission")
    for field in ("rights_holder", "issued_at"):
        if not isinstance(receipt[field], str) or not receipt[field].strip():
            raise ValueError(f"rights-holder receipt {field} must be a non-empty string")
    evidence = receipt["evidence_sha256"]
    if not isinstance(evidence, str) or _SHA256_RE.fullmatch(evidence) is None:
        raise ValueError("rights-holder receipt evidence_sha256 must be SHA-256")
    receipt_restrictions = receipt["restriction_ids"]
    if not isinstance(receipt_restrictions, list) or not all(
        isinstance(item, str) and _SHA256_RE.fullmatch(item) is not None
        for item in receipt_restrictions
    ):
        raise ValueError("rights-holder receipt restriction_ids must be SHA-256 strings")
    if len(set(receipt_restrictions)) != len(receipt_restrictions):
        raise ValueError("rights-holder receipt contains duplicate restriction IDs")
    required = {item.restriction_id for item in restrictions}
    if set(receipt_restrictions) != required:
        raise ValueError("rights-holder receipt does not cover exactly all lineage restrictions")


def audit_checkpoint_release(
    checkpoint: Path,
    *,
    rights_holder_receipt: Path | None = None,
) -> ReleaseAudit:
    """Approve only a sanitized Meteo inference checkpoint with publishable lineage."""

    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise ValueError("checkpoint release candidate must be a non-symlink directory")
    findings: list[ReleaseFinding] = []
    allowed_artifacts = {"metadata.json", "weights.safetensors"}
    allowed_ancillary_names = {
        "license",
        "notice",
        "thanks",
        "acknowledgements",
        "acknowledgments",
        "credits",
        "readme.md",
        "thanks.md",
        "acknowledgements.md",
        "acknowledgments.md",
        "credits.md",
    }
    for path in sorted(
        checkpoint.rglob("*"),
        key=lambda item: item.relative_to(checkpoint).as_posix(),
    ):
        relative = PurePosixPath(path.relative_to(checkpoint).as_posix())
        if path.is_symlink():
            findings.append(
                ReleaseFinding(
                    relative.as_posix(),
                    "symlink",
                    "checkpoint release contains a symlink",
                )
            )
            continue
        if not path.is_file():
            continue
        if relative.parent == PurePosixPath(".") and relative.name in allowed_artifacts:
            if relative.name == "metadata.json":
                findings.extend(_text_findings(relative, path.read_bytes()))
            continue
        if (
            relative.parent == PurePosixPath(".")
            and relative.name.casefold() in allowed_ancillary_names
        ):
            findings.extend(_text_findings(relative, path.read_bytes()))
            continue
        findings.extend(_path_findings(relative, allow_meteo_weights=False))
        if _is_text_path(relative):
            findings.extend(_text_findings(relative, path.read_bytes()))
        else:
            findings.append(
                ReleaseFinding(
                    relative.as_posix(),
                    "unexpected-checkpoint-file",
                    "public checkpoint contains an unreviewed ancillary file",
                )
            )
    _raise_findings(findings)

    metadata = _read_json_object(checkpoint / "metadata.json", label="checkpoint metadata")
    engine = metadata.get("engine")
    if not isinstance(engine, dict) or engine.get("romanized_name") != "Meteo":
        raise ValueError("release candidate is not identified as a Meteo checkpoint")
    lineage = metadata.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("release candidate requires complete JSON lineage")
    checkpoint_sha256, file_count, byte_count = checkpoint_release_identity(checkpoint)
    restrictions = find_lineage_restrictions(cast(dict[str, Any], lineage))
    if restrictions and rights_holder_receipt is None:
        findings = [
            ReleaseFinding(
                restriction.pointer,
                "rights-holder-permission-required",
                restriction.detail,
            )
            for restriction in restrictions
        ]
        _raise_findings(findings)
    if rights_holder_receipt is not None:
        if not restrictions:
            raise ValueError("rights-holder receipt supplied for an unrestricted checkpoint")
        _validate_permission_receipt(
            rights_holder_receipt,
            checkpoint_sha256=checkpoint_sha256,
            restrictions=restrictions,
        )
    return ReleaseAudit(
        checkpoint_sha256=checkpoint_sha256,
        files=file_count,
        bytes=byte_count,
        restrictions=restrictions,
        rights_holder_receipt_used=rights_holder_receipt is not None,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("source", help="scan the public source tree")
    source.add_argument("root", type=Path)
    repository = subparsers.add_parser(
        "repository", help="scan exact tracked Git-index source blobs"
    )
    repository.add_argument("root", type=Path)
    archive = subparsers.add_parser("archive", help="scan built wheel/sdist archives")
    archive.add_argument("paths", nargs="+", type=Path)
    checkpoint = subparsers.add_parser("checkpoint", help="audit a Meteo checkpoint release")
    checkpoint.add_argument("path", type=Path)
    checkpoint.add_argument("--rights-holder-receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "source":
        print(json.dumps({"source_files_scanned": scan_source_tree(args.root)}))
        return 0
    if args.command == "repository":
        print(json.dumps({"tracked_source_files_scanned": scan_repository_index(args.root)}))
        return 0
    if args.command == "archive":
        results = [
            {"path": str(path), "files_scanned": scan_build_archive(path)} for path in args.paths
        ]
        print(json.dumps({"archives": results}, indent=2))
        return 0
    if args.command == "checkpoint":
        audit = audit_checkpoint_release(
            args.path, rights_holder_receipt=args.rights_holder_receipt
        )
        print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled release guard command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
