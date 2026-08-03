from __future__ import annotations

import io
import json
import sqlite3
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest
from pypdf import PdfWriter

from simajilord.capabilities.file_scope import file_provenance, file_workspace_id
from simajilord.capabilities.files import (
    FileReadRequest,
    FileWriteTextRequest,
    build_file_endpoints,
)
from simajilord.core import DisclosureObservation, InvocationContext
from simajilord.core.errors import UserError
from simajilord.services.files import (
    AgentFileSandbox,
    WorkspaceFileProvenance,
    merge_file_provenances,
)


def _agent_file_context(
    *,
    actor_id: str = "actor-a",
    task_id: str = "task-a",
    mode: Literal["actor_task", "actor", "guild_shared"] = "actor_task",
) -> InvocationContext:
    return InvocationContext(
        actor_id=actor_id,
        workspace_id="guild",
        transport="agent",
        request_id="event",
        origin_resource_id="staff-channel",
        active_message_id="message",
        agent_task_id=task_id,
        file_workspace_mode=mode,
    )


def test_agent_file_workspace_mode_isolates_actor_and_task() -> None:
    context = _agent_file_context()
    actor_scope = file_workspace_id(replace(context, file_workspace_mode="actor"))
    task_scope = file_workspace_id(context)

    assert task_scope != file_workspace_id(
        _agent_file_context(actor_id="actor-b")
    )
    assert task_scope != file_workspace_id(
        _agent_file_context(task_id="task-b")
    )
    assert actor_scope == file_workspace_id(
        _agent_file_context(task_id="task-b", mode="actor")
    )
    assert file_workspace_id(
        replace(context, file_workspace_mode="guild_shared")
    ) == "guild"


@pytest.mark.asyncio
async def test_actor_workspace_reuses_owned_files_across_tasks(
    tmp_path: Path,
) -> None:
    sandbox = AgentFileSandbox(tmp_path / "files")
    endpoints = {
        item.descriptor.name: item for item in build_file_endpoints(sandbox)
    }
    first_task = _agent_file_context(task_id="task-a", mode="actor")
    second_task = _agent_file_context(task_id="task-b", mode="actor")

    await endpoints["files.write_text"].invoke(
        FileWriteTextRequest(path="notes.txt", content="shared by actor"),
        first_task,
    )
    result = await endpoints["files.read"].invoke(
        FileReadRequest(path="notes.txt"),
        second_task,
    )

    assert result.content == "shared by actor"
    assert result.provenance is not None
    assert result.provenance.owner_actor_ids == ("actor-a",)


def test_guild_shared_workspace_enforces_exact_actor_file_authority(
    tmp_path: Path,
) -> None:
    sandbox = AgentFileSandbox(tmp_path / "files")
    actor_a = WorkspaceFileProvenance(owner_actor_ids=("actor-a",))
    actor_b = WorkspaceFileProvenance(owner_actor_ids=("actor-b",))
    sandbox.import_bytes("guild", "a.txt", b"a", provenance=actor_a)
    sandbox.import_bytes("guild", "b.txt", b"b", provenance=actor_b)

    assert tuple(
        record.path for record in sandbox.list_for_actor("guild", "actor-a")
    ) == ("a.txt",)
    assert sandbox.read_for_actor(
        "guild",
        "actor-a",
        "a.txt",
        offset=0,
        max_characters=100,
    ).content == "a"
    with pytest.raises(UserError, match=r"files\.not_found"):
        sandbox.read_for_actor(
            "guild",
            "actor-a",
            "b.txt",
            offset=0,
            max_characters=100,
        )
    with pytest.raises(UserError, match=r"files\.not_found"):
        sandbox.snapshot_for_actor_delivery_with_provenance(
            "guild",
            "actor-a",
            "b.txt",
        )
    with pytest.raises(UserError, match=r"files\.path_conflict"):
        sandbox.write_text_for_actor(
            "guild",
            "actor-a",
            "b.txt",
            "overwrite",
            provenance=actor_a,
        )
    with pytest.raises(UserError, match=r"files\.path_conflict"):
        sandbox.import_bytes(
            "guild",
            "b.txt",
            b"overwrite",
            provenance=actor_a,
        )

    assert sandbox.read_for_actor(
        "guild",
        "actor-b",
        "b.txt",
        offset=0,
        max_characters=100,
    ).content == "b"


def test_unlabelled_legacy_file_is_hidden_until_ownership_is_assigned(
    tmp_path: Path,
) -> None:
    sandbox = AgentFileSandbox(tmp_path / "files")
    record = sandbox.import_bytes("guild", "legacy.txt", b"legacy")

    assert record.provenance is not None
    assert record.provenance.unlabelled_input
    assert record.provenance.sensitivity == "uncertain"
    assert sandbox.list_for_actor("guild", "actor") == ()
    with pytest.raises(UserError, match=r"files\.not_found"):
        sandbox.read_for_actor(
            "guild",
            "actor",
            "legacy.txt",
            offset=0,
            max_characters=100,
        )


def test_provenance_merge_retains_all_owners_and_taints_missing_or_overflow() -> None:
    actor_a = WorkspaceFileProvenance(owner_actor_ids=("actor-a",))
    actor_b = WorkspaceFileProvenance(owner_actor_ids=("actor-b",))
    multi_owner = merge_file_provenances((actor_a, actor_b))

    assert multi_owner is not None
    assert multi_owner.owner_actor_ids == ("actor-a", "actor-b")
    assert multi_owner.sensitivity == "uncertain"

    unlabelled = merge_file_provenances((actor_a, None))
    assert unlabelled is not None
    assert unlabelled.owner_actor_ids == ("actor-a",)
    assert unlabelled.unlabelled_input
    assert unlabelled.sensitivity == "uncertain"

    public_sources = tuple(
        ("guild", f"public-{index}", "guild_public")
        for index in range(20)
    )
    restricted_sources = tuple(
        ("guild", f"staff-{index}", "restricted")
        for index in range(20)
    )
    overflow = merge_file_provenances(
        (
            WorkspaceFileProvenance(
                owner_actor_ids=("actor-a",),
                sensitivity="guild_public",
                source_resources=public_sources,
            ),
            WorkspaceFileProvenance(
                owner_actor_ids=("actor-a",),
                sensitivity="restricted",
                source_resources=restricted_sources,
            ),
        )
    )
    assert overflow is not None
    assert len(overflow.source_resources) == 32
    assert overflow.sources_truncated
    assert overflow.sensitivity == "uncertain"


def test_provenance_schema_migrates_legacy_single_owner_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "files"
    database = root.with_name("files.provenance.sqlite3")
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE file_provenance (
                workspace_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                owner_actor_id TEXT NOT NULL,
                origin_guild_id TEXT,
                origin_channel_id TEXT,
                origin_message_id TEXT,
                origin_visibility TEXT NOT NULL,
                created_task_id TEXT,
                sensitivity TEXT NOT NULL,
                source_resources_json TEXT NOT NULL,
                declassified_at TEXT,
                declassified_by TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (workspace_id, relative_path)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO file_provenance VALUES (
                ?, ?, ?, NULL, NULL, NULL, ?, NULL, ?, ?, NULL, NULL, ?
            )
            """,
            (
                "guild",
                "legacy.txt",
                "actor-a",
                "actor_private",
                "actor_private",
                json.dumps([]),
                "2026-08-03T00:00:00+00:00",
            ),
        )

    sandbox = AgentFileSandbox(root)
    provenance = sandbox._load_provenance("guild", "legacy.txt")

    assert provenance is not None
    assert provenance.owner_actor_ids == ("actor-a",)
    assert not provenance.unlabelled_input
    with sqlite3.connect(database) as connection:
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(file_provenance)"
            )
        }
    assert {
        "owner_actor_ids_json",
        "unlabelled_input",
        "sources_truncated",
    } <= columns


def test_uncertain_and_multi_owner_provenance_survives_restart(
    tmp_path: Path,
) -> None:
    root = tmp_path / "files"
    sandbox = AgentFileSandbox(root)
    sandbox.import_bytes("guild", "unlabelled.txt", b"unlabelled")
    sandbox.import_bytes(
        "guild",
        "multi.txt",
        b"multi",
        provenance=WorkspaceFileProvenance(
            owner_actor_ids=("actor-a", "actor-b"),
        ),
    )

    restarted = AgentFileSandbox(root)
    records = {record.path: record for record in restarted.list("guild")}

    unlabelled = records["unlabelled.txt"].provenance
    assert unlabelled is not None
    assert unlabelled.owner_actor_ids == ()
    assert unlabelled.unlabelled_input
    assert unlabelled.sensitivity == "uncertain"
    multi = records["multi.txt"].provenance
    assert multi is not None
    assert multi.owner_actor_ids == ("actor-a", "actor-b")
    assert multi.sensitivity == "uncertain"


def test_target_bound_publication_preserves_source_and_rejects_other_targets(
    tmp_path: Path,
) -> None:
    sandbox = AgentFileSandbox(tmp_path / "files")
    source = WorkspaceFileProvenance(
        owner_actor_ids=("7",),
        origin_guild_id="10",
        origin_channel_id="20",
        origin_visibility="restricted",
        sensitivity="restricted",
        source_resources=(("10", "20", "restricted"),),
    )
    original = sandbox.import_bytes(
        "10",
        "report.txt",
        b"private report",
        provenance=source,
    )
    expires_at = (datetime.now(UTC) + timedelta(hours=1)).isoformat()

    publication = sandbox.publish_copy_for_actor(
        "10",
        "7",
        "report.txt",
        expected_sha256=original.sha256,
        target_workspace_id="10",
        target_resource_id="21",
        target_display_name="#review",
        target_audience_revision="a" * 64,
        reason="Share the reviewed report",
        expires_at=expires_at,
    )

    unchanged = sandbox.list_for_actor("10", "7")[0]
    assert unchanged.provenance == source
    assert unchanged.sha256 == original.sha256
    assert publication.source_path == "report.txt"
    assert publication.target_resource_id == "21"
    filename, content, copied_provenance, loaded = (
        sandbox.snapshot_publication_for_actor(
            publication.publication_id,
            "7",
            target_workspace_id="10",
            target_resource_id="21",
            expected_revision=1,
        )
    )
    assert filename == "report.txt"
    assert content == b"private report"
    assert copied_provenance.publication_id == publication.publication_id
    assert loaded == publication

    with pytest.raises(UserError, match=r"files\.publication_target_mismatch"):
        sandbox.snapshot_publication_for_actor(
            publication.publication_id,
            "7",
            target_workspace_id="10",
            target_resource_id="22",
            expected_revision=1,
        )

    revoked, changed = sandbox.revoke_publication_for_actor(
        publication.publication_id,
        "7",
        expected_revision=1,
    )
    assert changed
    assert revoked.revision == 2
    assert revoked.revoked_at is not None
    with pytest.raises(UserError, match=r"files\.publication_revoked"):
        sandbox.snapshot_publication_for_actor(
            publication.publication_id,
            "7",
            target_workspace_id="10",
            target_resource_id="21",
            expected_revision=2,
        )


def test_expired_publication_copy_cannot_be_reused(tmp_path: Path) -> None:
    sandbox = AgentFileSandbox(tmp_path / "files")
    provenance = WorkspaceFileProvenance(owner_actor_ids=("7",))
    source = sandbox.import_bytes(
        "10",
        "brief.txt",
        b"brief",
        provenance=provenance,
    )
    publication = sandbox.publish_copy_for_actor(
        "10",
        "7",
        "brief.txt",
        expected_sha256=source.sha256,
        target_workspace_id="10",
        target_resource_id="21",
        target_display_name="#review",
        target_audience_revision="b" * 64,
        reason="Temporary review",
        expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
    )
    with sqlite3.connect(sandbox._provenance_path) as connection:
        connection.execute(
            """
            UPDATE file_publications
            SET published_at = ?, expires_at = ?
            WHERE publication_id = ?
            """,
            (
                (datetime.now(UTC) - timedelta(minutes=2)).isoformat(),
                (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
                publication.publication_id,
            ),
        )

    with pytest.raises(UserError, match=r"files\.publication_expired"):
        sandbox.snapshot_publication_for_actor(
            publication.publication_id,
            "7",
            target_workspace_id="10",
            target_resource_id="21",
            expected_revision=1,
        )


def test_file_provenance_persists_and_cannot_be_downgraded(
    tmp_path: Path,
) -> None:
    context = replace(
        _agent_file_context(),
        disclosure_observations=(
            DisclosureObservation(
                source_workspace_id="guild",
                source_resource_id="staff-channel",
                visibility="restricted",
                relation_to_origin="same_or_narrower",
            ),
        ),
    )
    workspace_id = file_workspace_id(context)
    sandbox = AgentFileSandbox(tmp_path / "files")
    restricted = file_provenance(context)
    sandbox.import_bytes(
        workspace_id,
        "notes.txt",
        b"staff",
        provenance=restricted,
    )
    public = WorkspaceFileProvenance(
        owner_actor_ids=(context.actor_id,),
        origin_guild_id="guild",
        origin_channel_id="public-channel",
        origin_visibility="guild_public",
        created_task_id=context.agent_task_id,
        sensitivity="guild_public",
        source_resources=(("guild", "public-channel", "guild_public"),),
    )
    sandbox.import_bytes(
        workspace_id,
        "notes.txt",
        b"rewritten",
        provenance=public,
    )

    restarted = AgentFileSandbox(tmp_path / "files")
    record = restarted.list(workspace_id)[0]
    assert record.provenance is not None
    assert record.provenance.sensitivity == "restricted"
    assert ("guild", "staff-channel", "restricted") in (
        record.provenance.source_resources
    )
    assert ("guild", "public-channel", "guild_public") in (
        record.provenance.source_resources
    )


def test_provenance_failure_does_not_commit_new_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sandbox = AgentFileSandbox(tmp_path / "files")
    provenance = WorkspaceFileProvenance(
        owner_actor_ids=("actor",),
        sensitivity="restricted",
    )
    sandbox.import_bytes(
        "scope",
        "notes.txt",
        b"old",
        provenance=provenance,
    )

    def fail_store(*_args: object, **_kwargs: object) -> None:
        raise OSError("provenance disk unavailable")

    monkeypatch.setattr(sandbox, "_store_provenance", fail_store)
    with pytest.raises(OSError, match="provenance disk unavailable"):
        sandbox.import_bytes(
            "scope",
            "notes.txt",
            b"restricted",
            provenance=provenance,
        )

    assert sandbox.path_for_delivery("scope", "notes.txt").read_bytes() == b"old"


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


def test_file_sandbox_delivery_snapshot_is_detached_from_later_writes(
    tmp_path: Path,
) -> None:
    sandbox = AgentFileSandbox(tmp_path / "files")
    sandbox.import_bytes("guild", "exports/result.bin", b"first")

    filename, content = sandbox.snapshot_for_delivery(
        "guild",
        "exports/result.bin",
    )
    sandbox.import_bytes("guild", "exports/result.bin", b"second")

    assert filename == "result.bin"
    assert content == b"first"


def test_file_sandbox_expected_hash_is_atomic_across_writers(
    tmp_path: Path,
) -> None:
    sandbox = AgentFileSandbox(tmp_path / "files")
    original = sandbox.write_text("guild", "shared.txt", "original")
    barrier = threading.Barrier(3)

    def write(value: str) -> str:
        barrier.wait()
        try:
            return sandbox.write_text(
                "guild",
                "shared.txt",
                value,
                expected_sha256=original.sha256,
            ).sha256
        except UserError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(write, "first")
        second = executor.submit(write, "second")
        barrier.wait()
        results = (first.result(), second.result())

    assert results.count("files.hash_conflict") == 1
    assert sum(len(value) == 64 for value in results) == 1


def test_file_sandbox_quota_is_atomic_across_importers(tmp_path: Path) -> None:
    sandbox = AgentFileSandbox(
        tmp_path / "files",
        max_files=1,
        max_workspace_bytes=10,
    )
    barrier = threading.Barrier(3)

    def write(path: str) -> str:
        barrier.wait()
        try:
            return sandbox.import_bytes("guild", path, b"123456").path
        except UserError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(write, "first.bin")
        second = executor.submit(write, "second.bin")
        barrier.wait()
        results = (first.result(), second.result())

    assert len(sandbox.list("guild")) == 1
    assert (
        sum(
            value in {"files.file_count_limit", "files.workspace_quota"}
            for value in results
        )
        == 1
    )


def test_file_sandbox_batch_rejects_without_partial_commit(tmp_path: Path) -> None:
    sandbox = AgentFileSandbox(
        tmp_path / "files",
        max_workspace_bytes=5,
    )

    with pytest.raises(UserError, match=r"files\.workspace_quota"):
        sandbox.import_batch(
            "guild",
            (("first.bin", b"123"), ("second.bin", b"456")),
        )

    assert sandbox.list("guild") == ()


@pytest.mark.parametrize(
    "path",
    ("../secret.txt", "/tmp/secret.txt", "a/../../secret.txt", "./secret.txt"),
)
def test_file_sandbox_rejects_path_escape(tmp_path: Path, path: str) -> None:
    sandbox = AgentFileSandbox(tmp_path / "files")
    with pytest.raises(UserError, match=r"files\.path_invalid"):
        sandbox.write_text("guild", path, "no")
    assert list((tmp_path / "files").iterdir()) == []


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
    assert result.page_start == 1
    assert result.total_pages == 1
    assert result.next_page is None
    assert result.complete


def test_file_sandbox_pages_through_pdf_beyond_old_twenty_page_limit(
    tmp_path: Path,
) -> None:
    payload = io.BytesIO()
    writer = PdfWriter()
    for _ in range(23):
        writer.add_blank_page(width=100, height=100)
    writer.write(payload)
    sandbox = AgentFileSandbox(tmp_path / "files")
    sandbox.import_bytes("guild", "long.pdf", payload.getvalue())

    first = sandbox.read(
        "guild",
        "long.pdf",
        offset=0,
        max_characters=20_000,
        page_start=1,
        page_count=5,
    )
    assert first.page_start == 1
    assert first.next_page == 6
    assert first.total_pages == 23
    assert not first.complete
    assert "Page 5" in first.content
    assert "Page 6" not in first.content

    final = sandbox.read(
        "guild",
        "long.pdf",
        offset=0,
        max_characters=20_000,
        page_start=21,
        page_count=5,
    )
    assert final.page_start == 21
    assert final.next_page is None
    assert final.total_pages == 23
    assert final.complete
    assert "Page 21" in final.content
    assert "Page 23" in final.content


def test_file_sandbox_rejects_pdf_page_controls_for_non_pdf(
    tmp_path: Path,
) -> None:
    sandbox = AgentFileSandbox(tmp_path / "files")
    sandbox.write_text("guild", "notes.txt", "hello")

    with pytest.raises(UserError, match=r"files\.page_range_unsupported"):
        sandbox.read(
            "guild",
            "notes.txt",
            offset=0,
            max_characters=100,
            page_start=2,
        )


@pytest.mark.parametrize(
    ("filename", "kind"),
    (
        ("clip.mp4", "video/mp4"),
        ("clip.webm", "video/webm"),
        ("sound.m4a", "audio/mp4"),
        ("sound.mp3", "audio/mpeg"),
    ),
)
def test_file_sandbox_preserves_media_kind_for_later_delivery(
    tmp_path: Path,
    filename: str,
    kind: str,
) -> None:
    sandbox = AgentFileSandbox(tmp_path / "files")
    record = sandbox.import_bytes("guild", f"media/{filename}", b"\x00media")
    assert record.kind == kind
    inspected = sandbox.read(
        "guild",
        f"media/{filename}",
        offset=0,
        max_characters=2_000,
    )
    assert inspected.kind == kind
    assert "Binary file:" in inspected.content
