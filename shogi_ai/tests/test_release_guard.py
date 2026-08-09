from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest

from simajilord_shogi.release_guard import (
    RECEIPT_SCHEMA,
    RECEIPT_SCOPE,
    ReleaseGuardError,
    audit_checkpoint_release,
    checkpoint_release_identity,
    find_lineage_restrictions,
    scan_build_archive,
    scan_repository_index,
    scan_source_tree,
)
from simajilord_shogi.rights_lineage import (
    merge_rights_restriction_summaries,
    summarize_teacher_sidecar,
)


def _git(repository: Path, *arguments: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        input=stdin,
        check=True,
        capture_output=True,
    ).stdout


def _repository_fixture(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Meteo release test")
    _git(repository, "config", "user.email", "meteo-release@example.invalid")
    source = repository / "shogi_ai"
    (source / "src").mkdir(parents=True)
    (source / "src" / "engine.py").write_text("answer = 42\n", encoding="utf-8")
    (source / ".gitignore").write_text(
        "/artifacts/\n/datasets/*\n!/datasets/README.md\n*.7z\n",
        encoding="utf-8",
    )
    _git(repository, "add", "--", "shogi_ai/.gitignore", "shogi_ai/src/engine.py")
    return repository, source


def _metadata(lineage: dict[str, object]) -> dict[str, object]:
    return {
        "format_version": 1,
        "step": 10,
        "backend": "mlx",
        "engine": {
            "display_name": "めてお",
            "romanized_name": "Meteo",
            "author": "meteosimaji",
        },
        "model": {"name": "fixture"},
        "lineage": lineage,
    }


def _checkpoint(root: Path, lineage: dict[str, object]) -> Path:
    root.mkdir()
    (root / "metadata.json").write_text(
        json.dumps(_metadata(lineage), sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "weights.safetensors").write_bytes(b"independent Meteo fixture weights")
    return root


def _receipt(
    path: Path,
    *,
    checkpoint: Path,
    lineage: dict[str, object],
) -> Path:
    checkpoint_sha256, _files, _bytes = checkpoint_release_identity(checkpoint)
    restrictions = find_lineage_restrictions(lineage)
    path.write_text(
        json.dumps(
            {
                "schema": RECEIPT_SCHEMA,
                "scope": RECEIPT_SCOPE,
                "checkpoint_sha256": checkpoint_sha256,
                "rights_holder": "fixture rights holder",
                "rights_holder_permission": True,
                "evidence_sha256": "e" * 64,
                "restriction_ids": [item.restriction_id for item in restrictions],
                "issued_at": "2026-08-09T00:00:00+00:00",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_source_tree_accepts_code_and_teacher_author_only_thanks(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "engine.py").write_text("answer = 42\n", encoding="utf-8")
    (tmp_path / "THANKS.md").write_text(
        "# Thanks\n\nNAGISA and Suisho by their respective authors.\n", encoding="utf-8"
    )

    assert scan_source_tree(tmp_path) == 2


def test_repository_index_is_nul_safe_and_allows_only_dataset_readme(
    tmp_path: Path,
) -> None:
    repository, source = _repository_fixture(tmp_path)
    unusual = source / "src" / "line\nbreak.py"
    unusual.write_text("safe = True\n", encoding="utf-8")
    dataset_readme = source / "datasets" / "README.md"
    dataset_readme.parent.mkdir()
    dataset_readme.write_text("Generated datasets stay local.\n", encoding="utf-8")
    _git(repository, "add", "--", str(unusual.relative_to(repository)))
    _git(repository, "add", "-f", "--", str(dataset_readme.relative_to(repository)))

    assert scan_repository_index(source) == 4


@pytest.mark.parametrize(
    ("relative", "rule"),
    [
        ("artifacts/private-teacher.txt", "generated-or-sealed-path"),
        ("src/private-teacher.7z", "model-or-data-artifact"),
        ("datasets/private-replay.jsonl", "generated-or-sealed-path"),
    ],
)
def test_repository_index_rejects_git_add_force_private_artifacts(
    tmp_path: Path,
    relative: str,
    rule: str,
) -> None:
    repository, source = _repository_fixture(tmp_path)
    private = source / relative
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_bytes(b"private supporter artifact")
    _git(repository, "add", "-f", "--", str(private.relative_to(repository)))

    with pytest.raises(ReleaseGuardError) as captured:
        scan_repository_index(source)

    assert rule in {finding.rule for finding in captured.value.findings}


def test_repository_index_reads_staged_blob_not_cleaned_worktree(tmp_path: Path) -> None:
    repository, source = _repository_fixture(tmp_path)
    readme = source / "README.md"
    restricted_host = "storage" + ".yaneu" + ".com"
    readme.write_text(f"https://{restricted_host}/private/model.7z\n", encoding="utf-8")
    _git(repository, "add", "--", "shogi_ai/README.md")
    readme.write_text("The worktree was cleaned after staging.\n", encoding="utf-8")

    with pytest.raises(ReleaseGuardError) as captured:
        scan_repository_index(source)

    assert "restricted-download-link" in {
        finding.rule for finding in captured.value.findings
    }


def test_repository_index_rejects_symlink_gitlink_and_normalized_collision(
    tmp_path: Path,
) -> None:
    repository, source = _repository_fixture(tmp_path)
    link = source / "src" / "engine-link.py"
    link.symlink_to("engine.py")
    _git(repository, "add", "--", "shogi_ai/src/engine-link.py")
    _git(repository, "commit", "--quiet", "-m", "fixture")
    commit = _git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    blob = _git(repository, "rev-parse", "HEAD:shogi_ai/src/engine.py").decode("ascii").strip()
    _git(
        repository,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{commit},shogi_ai/vendor/private-teacher",
    )
    _git(
        repository,
        "update-index",
        "--add",
        "--cacheinfo",
        f"100644,{blob},shogi_ai/src/Engine.py",
    )

    with pytest.raises(ReleaseGuardError) as captured:
        scan_repository_index(source)

    rules = {finding.rule for finding in captured.value.findings}
    assert {"symlink", "gitlink", "path-collision"} <= rules


def test_extensionless_thanks_rejects_download_metadata(tmp_path: Path) -> None:
    (tmp_path / "THANKS").write_text(
        "Suisho author - https://example.invalid/download/model.bin\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseGuardError) as captured:
        scan_source_tree(tmp_path)
    assert "thanks-metadata" in {finding.rule for finding in captured.value.findings}


@pytest.mark.parametrize(
    ("name", "contents", "rule"),
    [
        ("teacher.7z", b"paid bytes", "model-or-data-artifact"),
        ("book.ybb", b"book bytes", "model-or-data-artifact"),
        ("sealed_final_test/data.txt", b"hidden", "generated-or-sealed-path"),
    ],
)
def test_build_archive_rejects_model_book_and_sealed_artifacts(
    tmp_path: Path, name: str, contents: bytes, rule: str
) -> None:
    archive = tmp_path / "candidate.whl"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("package/__init__.py", "")
        bundle.writestr(name, contents)

    with pytest.raises(ReleaseGuardError) as captured:
        scan_build_archive(archive)
    assert rule in {finding.rule for finding in captured.value.findings}


def test_source_tree_rejects_restricted_link_private_path_and_thanks_metadata(
    tmp_path: Path,
) -> None:
    restricted_host = "storage" + ".yaneu" + ".com"
    (tmp_path / "README.md").write_text(
        "\n".join(
            (
                "# Project",
                f"download: https://{restricted_host}/paid/model.7z",
                "/" + "Users" + "/person/Downloads/model",
                "## Thanks",
                "Suisho author",
                "archive hash: " + "a" * 64,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseGuardError) as captured:
        scan_source_tree(tmp_path)
    rules = {finding.rule for finding in captured.value.findings}
    assert "restricted-download-link" in rules
    assert "local-absolute-path" in rules
    assert "thanks-metadata" in rules


def test_source_and_repository_reject_supporter_platform_links(tmp_path: Path) -> None:
    repository, source = _repository_fixture(tmp_path)
    supporter_host = "fanbox" + ".cc"
    rights = source / "MODEL_RIGHTS.md"
    rights.write_text(
        f"supporter artifact: https://creator.{supporter_host}/posts/private-fixture\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseGuardError, match="restricted-download-link"):
        scan_source_tree(source)

    _git(repository, "add", "--", "shogi_ai/MODEL_RIGHTS.md")
    with pytest.raises(ReleaseGuardError, match="restricted-download-link"):
        scan_repository_index(source)


def test_archive_rejects_raw_suisho_startup_provenance(tmp_path: Path) -> None:
    archive = tmp_path / "candidate.tar.gz"
    payload = json.dumps(
        {
            "rights_id": "suisho11plus-wcsc36-20260525-local",
            "startup_provenance": {"stdout_lines": ["raw USI output"]},
        }
    ).encode()
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("release/teacher.startup.json")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))

    with pytest.raises(ReleaseGuardError) as captured:
        scan_build_archive(archive)
    assert "restricted-teacher-record" in {finding.rule for finding in captured.value.findings}


def test_archive_rejects_renamed_usi_startup_and_normalized_path_collision(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.whl"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "release/runtime.json",
            json.dumps({"schema": "meteo-usi-startup-provenance-v1"}),
        )
        bundle.writestr("release/README.md", "one")
        bundle.writestr("release/readme.md", "two")

    with pytest.raises(ReleaseGuardError) as captured:
        scan_build_archive(archive)
    rules = {finding.rule for finding in captured.value.findings}
    assert "generated-private-record" in rules
    assert "duplicate-member" in rules


def test_build_archive_accepts_sanitized_source(tmp_path: Path) -> None:
    archive = tmp_path / "candidate.tar.gz"
    payload = b"# public source\n"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("simajilord_shogi/README.md")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))

    assert scan_build_archive(archive) == 1


def test_checkpoint_release_accepts_meteo_weights_with_public_teacher_lineage(
    tmp_path: Path,
) -> None:
    rights_summary = merge_rights_restriction_summaries(
        [], teacher_sources=["nagisa-v3.1"]
    )
    checkpoint = _checkpoint(
        tmp_path / "checkpoint",
        {
            "schema": "meteo-training-lineage-v2",
            "teacher_sources": ["nagisa-v3.1"],
            "rights_restriction_summary": rights_summary,
        },
    )

    audit = audit_checkpoint_release(checkpoint)

    assert audit.files == 2
    assert audit.restrictions == ()
    assert not audit.rights_holder_receipt_used


def test_checkpoint_release_rejects_limited_teacher_without_author_receipt(
    tmp_path: Path,
) -> None:
    rights_summary = summarize_teacher_sidecar(
        {
            "schema": "meteo-teacher-ensemble-v1",
            "inputs": [
                {
                    "provenance": {
                        "rights": {
                            "rights_id": "nagisa-v3.1",
                            "output_only_meteo_publication": "allowed",
                        }
                    }
                },
                {
                    "provenance": {
                        "rights_mode": "limited_local",
                        "publication_allowed": False,
                        "rights": {
                            "rights_id": (
                                "suisho11plus-wcsc36-20260525-local"
                            ),
                            "output_only_meteo_publication": "not_approved",
                        },
                    }
                },
            ],
        },
        sidecar_sha256="a" * 64,
    )
    lineage: dict[str, object] = {
        "schema": "meteo-training-lineage-v2",
        "teacher_sources": ["meteo-teacher-ensemble"],
        # An inference-only export intentionally has no local sidecar or path;
        # the privacy-safe, SHA-bound summary alone must retain the block.
        "rights_restriction_summary": rights_summary,
    }
    checkpoint = _checkpoint(tmp_path / "checkpoint", lineage)
    summarized_restrictions = find_lineage_restrictions(lineage)
    assert [item.restriction_id for item in summarized_restrictions] == (
        rights_summary["restriction_ids"]
    )

    with pytest.raises(ReleaseGuardError, match="suisho11plus"):
        audit_checkpoint_release(checkpoint)

    receipt = _receipt(
        tmp_path / "receipt.json", checkpoint=checkpoint, lineage=lineage
    )
    audit = audit_checkpoint_release(checkpoint, rights_holder_receipt=receipt)
    assert audit.rights_holder_receipt_used
    assert [item.restriction_id for item in audit.restrictions] == (
        rights_summary["restriction_ids"]
    )


def test_checkpoint_release_fails_closed_for_legacy_teacher_lineage_without_summary(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(
        tmp_path / "checkpoint",
        {
            "schema": "meteo-training-lineage-v2",
            "teacher_sources": ["nagisa-v3.1"],
        },
    )

    with pytest.raises(ReleaseGuardError, match="must be migrated"):
        audit_checkpoint_release(checkpoint)


def test_exact_author_receipt_allows_limited_ancestor_but_not_teacher_artifacts(
    tmp_path: Path,
) -> None:
    lineage: dict[str, object] = {
        "schema": "meteo-training-lineage-v2",
        "teacher_sources": ["suisho11plus-wcsc36-20260525-local"],
        "parent_checkpoint": {
            "lineage": {
                "rights_mode": "limited_local",
                "publication_allowed": False,
            }
        },
    }
    checkpoint = _checkpoint(tmp_path / "checkpoint", lineage)
    receipt = _receipt(tmp_path / "receipt.json", checkpoint=checkpoint, lineage=lineage)

    audit = audit_checkpoint_release(checkpoint, rights_holder_receipt=receipt)
    assert audit.rights_holder_receipt_used
    assert len(audit.restrictions) >= 3

    (checkpoint / "teacher.nn").write_bytes(b"restricted teacher")
    with pytest.raises(ReleaseGuardError, match="model-or-data-artifact"):
        audit_checkpoint_release(checkpoint, rights_holder_receipt=receipt)


def test_receipt_is_bound_to_exact_checkpoint_bytes(tmp_path: Path) -> None:
    lineage: dict[str, object] = {
        "schema": "meteo-training-lineage-v2",
        "publication_allowed": False,
    }
    checkpoint = _checkpoint(tmp_path / "checkpoint", lineage)
    receipt = _receipt(tmp_path / "receipt.json", checkpoint=checkpoint, lineage=lineage)
    (checkpoint / "weights.safetensors").write_bytes(b"changed Meteo weights")

    with pytest.raises(ValueError, match="exact checkpoint"):
        audit_checkpoint_release(checkpoint, rights_holder_receipt=receipt)


def test_receipt_cannot_override_private_paths_in_metadata(tmp_path: Path) -> None:
    local_path = "/" + "private" + "/tmp/teacher/eval"
    lineage: dict[str, object] = {
        "schema": "meteo-training-lineage-v2",
        "publication_allowed": False,
        "training_inputs": [{"path": local_path}],
    }
    checkpoint = _checkpoint(tmp_path / "checkpoint", lineage)
    receipt = _receipt(tmp_path / "receipt.json", checkpoint=checkpoint, lineage=lineage)

    with pytest.raises(ReleaseGuardError, match="local-absolute-path"):
        audit_checkpoint_release(checkpoint, rights_holder_receipt=receipt)


def test_checkpoint_rejects_optimizer_state_and_unknown_teacher(tmp_path: Path) -> None:
    checkpoint = _checkpoint(
        tmp_path / "checkpoint",
        {
            "schema": "meteo-training-lineage-v2",
            "teacher_sources": ["unreviewed-private-teacher"],
        },
    )
    with pytest.raises(ReleaseGuardError, match="no reviewed publication decision"):
        audit_checkpoint_release(checkpoint)

    (checkpoint / "optimizer.safetensors").write_bytes(b"training state")
    with pytest.raises(ReleaseGuardError, match="model-or-data-artifact"):
        audit_checkpoint_release(checkpoint)
