from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from pypdf import PdfWriter

from simajilord.core.errors import UserError
from simajilord.services.files import AgentFileSandbox


def test_file_sandbox_write_read_replace_and_hash_conflict(tmp_path: Path) -> None:
    sandbox = AgentFileSandbox(tmp_path / "files")
    created = sandbox.write_text("guild-a", "notes/plan.md", "hello")
    assert created.path == "notes/plan.md"
    assert len(created.sha256) == 64

    read = sandbox.read(
        "guild-a",
        "notes/plan.md",
        offset=0,
        max_characters=100,
    )
    assert read.content == "hello"
    assert read.complete

    changed = sandbox.replace_text(
        "guild-a",
        "notes/plan.md",
        "hello",
        "hello world",
        expected_sha256=created.sha256,
    )
    assert changed.sha256 != created.sha256
    with pytest.raises(UserError, match=r"files\.hash_conflict"):
        sandbox.write_text(
            "guild-a",
            "notes/plan.md",
            "stale",
            expected_sha256=created.sha256,
        )


@pytest.mark.parametrize(
    "path",
    ("../secret.txt", "/tmp/secret.txt", "a/../../secret.txt", "./secret.txt"),
)
def test_file_sandbox_rejects_path_escape(tmp_path: Path, path: str) -> None:
    sandbox = AgentFileSandbox(tmp_path / "files")
    with pytest.raises(UserError, match=r"files\.path_invalid"):
        sandbox.write_text("guild", path, "no")


def test_file_sandbox_rejects_symlink_parent(tmp_path: Path) -> None:
    sandbox = AgentFileSandbox(tmp_path / "files")
    sandbox.write_text("guild", "safe.txt", "safe")
    scope = next((tmp_path / "files").iterdir())
    (scope / "link").symlink_to(tmp_path)
    with pytest.raises(UserError, match=r"files\.symlink_forbidden"):
        sandbox.write_text("guild", "link/escape.txt", "no")


def test_file_sandbox_inspects_zip_without_extracting(tmp_path: Path) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("docs/readme.txt", "hello")
    sandbox = AgentFileSandbox(tmp_path / "files")
    sandbox.import_bytes("guild", "upload.zip", payload.getvalue())
    result = sandbox.read("guild", "upload.zip", offset=0, max_characters=2_000)
    assert result.kind == "zip"
    assert "docs/readme.txt" in result.content
    scope = next((tmp_path / "files").iterdir())
    assert not (scope / "docs").exists()


def test_file_sandbox_reads_bounded_pdf_summary(tmp_path: Path) -> None:
    payload = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(payload)
    sandbox = AgentFileSandbox(tmp_path / "files")
    sandbox.import_bytes("guild", "document.pdf", payload.getvalue())
    result = sandbox.read("guild", "document.pdf", offset=0, max_characters=2_000)
    assert result.kind == "pdf"
    assert "PDF pages: 1" in result.content
    assert "Page 1" in result.content
