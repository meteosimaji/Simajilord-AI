"""NAGISA-style value-only NNUE training orchestration for Meteo.

This module is deliberately independent from the legacy MLX Policy+Value
stack.  It prepares and supervises a pinned Tatara LayerStack run whose only
learning target is the scalar score stored in PackedSfenValue records.  It
does not manufacture a policy target from ``Move16`` and it never copies a
teacher evaluation network into the student.

The production corpus is streamed one immutable Hugging Face LFS object at a
time.  A completed shard is removed only after Tatara has written and the
supervisor has hashed a raw resume checkpoint.  This bounds local storage
while preserving exact optimizer resume state.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import quote

import numpy as np
import psutil
from rsshogi.core import Board
from rsshogi.numpy import PackedSfenValue

from .teacher_data import PSV_RECORD_BYTES
from .teacher_lineage import (
    CorpusSeedSamplingScope,
    PsvMoveFieldContract,
    corpus_lineage,
)

NAGISA_NNUE_PLAN_SCHEMA = "meteo-nagisa-nnue-plan-v3"
NAGISA_NNUE_STATE_SCHEMA = "meteo-nagisa-nnue-state-v1"
NAGISA_NNUE_SHARD_RECEIPT_SCHEMA = "meteo-nagisa-nnue-shard-receipt-v1"
NAGISA_NNUE_STATUS_SCHEMA = "meteo-nagisa-nnue-status-v1"
TATARA_REPOSITORY_URL = "https://github.com/nodchip/tatara.git"
TATARA_PINNED_COMMIT = "34c7ea511816186d8cdc7ec4aa791e2eac04ce9c"
SOUJOU_DATASETS_1_CORPUS_ID = "soujou-team-datasets-1"
TARGET_PRESENTATIONS = 100_000_000_000
PROGRESS_KPABS_WEIGHTS = 81 * 1_548
PROGRESS_KPABS_BYTES = PROGRESS_KPABS_WEIGHTS * 8
DEFAULT_HELDOUT_POSITIONS = 1_000_000
DEFAULT_BATCH_SIZE = 65_536
DEFAULT_NET_ID = "meteo-nagisa-v1"
DEFAULT_SCORE_SCALE = 600.0
NNUE_QA = 127
NNUE_QB = 64
NAGISA_YANEURAOU_FV_SCALE = 16
NAGISA_WRM_NNUE2SCORE = float(NNUE_QA * NNUE_QB) / NAGISA_YANEURAOU_FV_SCALE
_USER_AGENT = "Simajilord-Meteo-NNUE/1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TATARA_LOSS_LINE = re.compile(
    r"\[train\] superbatch (?P<superbatch>\d+)/(?P<end>\d+) \| "
    r"loss (?P<loss>[0-9.eE+-]+) \| (?P<positions_per_second>[0-9.eE+-]+) pos/s"
)
_TATARA_TEST_LOSS = re.compile(r"\| test_loss (?P<test_loss>[0-9.eE+-]+)(?: \||$)")
_TATARA_EVAL_ONLY_LINE = re.compile(
    r"\[eval-only\] test_loss=(?P<test_loss>[0-9.eE+-]+) "
    r"test_accuracy=(?P<test_accuracy>[0-9.eE+-]+) "
    r"n_positions=(?P<n_positions>\d+) n_counted=(?P<n_counted>\d+)"
)


class IncompleteShardDownloadError(OSError):
    """A resumable public shard transfer ended before its declared size.

    The verified partial file is intentionally retained.  This is distinct
    from immutable-source integrity failures such as a bad SHA-256 or an
    invalid Content-Range, which remain non-retriable ``ValueError`` cases.
    """


def nagisa_wrm_contract(score_scale: float = DEFAULT_SCORE_SCALE) -> dict[str, object]:
    """Return the scale-aligned WRM contract used by NAGISA-style Meteo.

    The target remains exactly ``sigmoid(score / score_scale)``.  The network
    itself learns ``score / 508`` because YaneuraOu divides the quantised raw
    output (whose gain is ``QA * QB == 8128``) by ``FV_SCALE == 16``.  Keeping
    these two scales separate avoids the systematic 3.2% inference shrinkage
    caused by rounding ``8128 / 600`` to an integer FV_SCALE of 14.
    """

    if not math.isfinite(score_scale) or score_scale <= 0:
        raise ValueError("score_scale must be finite and positive")
    effective_cp_per_output = float(NNUE_QA * NNUE_QB) / NAGISA_YANEURAOU_FV_SCALE
    if effective_cp_per_output != NAGISA_WRM_NNUE2SCORE:
        raise AssertionError("NAGISA WRM scale constants are inconsistent")
    return {
        "kind": "wrm",
        "wdl_lambda": 0.0,
        "nnue2score": NAGISA_WRM_NNUE2SCORE,
        "in_scaling": float(score_scale),
        "in_offset": 0.0,
        "target_scaling": float(score_scale),
        "target_offset": 0.0,
        "loss_pow_exp": 2.0,
        "loss_qp_asymmetry": 0.0,
        "loss_weight_boost_w1": 0.0,
        "loss_weight_boost_w2": 0.5,
        "target_identity": f"sigmoid(score/{float(score_scale)})",
        "network_optimum": f"score/{NAGISA_WRM_NNUE2SCORE}",
        "quantization_gain": NNUE_QA * NNUE_QB,
        "yaneuraou_fv_scale": NAGISA_YANEURAOU_FV_SCALE,
        "effective_cp_per_float_output": effective_cp_per_output,
        "systematic_export_scale_ratio": 1.0,
    }


def _dataloader_threads() -> int:
    configured = os.environ.get("METEO_NNUE_THREADS")
    if configured is not None:
        try:
            value = int(configured)
        except ValueError as error:
            raise ValueError("METEO_NNUE_THREADS must be a positive integer") from error
        if not 1 <= value <= 256:
            raise ValueError("METEO_NNUE_THREADS must be in [1, 256]")
        return value
    physical = psutil.cpu_count(logical=False)
    available = physical or os.cpu_count() or 16
    return max(1, min(64, available))


@contextmanager
def _exclusive_run_lock(root: Path):  # type: ignore[no-untyped-def]
    lock_path = root / "RUNNING.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.chmod(lock_path, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another NNUE supervisor holds {lock_path}") from error
        yield
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class NagisaNnueArchitecture:
    """The public NAGISA V3.1-compatible SFNN topology used by Meteo."""

    feature_set: str = "halfka-hm-merged"
    yaneuraou_feature: str = "HalfKA_hm2"
    feature_dimensions: int = 73_305
    maximum_active_features: int = 40
    ft_out: int = 1_024
    l1_out: int = 16
    l2_out: int = 64
    bucket_mode: str = "progress8kpabs"
    num_buckets: int = 9

    @property
    def yaneuraou_header(self) -> str:
        return (
            "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)"
            "[73305->1024x2],Network=SFNN_HALFKAHM2_1024_15_64_K3K3"
            "{LayerStack=9}"
        )

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "yaneuraou_header": self.yaneuraou_header}


@dataclass(frozen=True, slots=True)
class PublicPsvShard:
    filename: str
    byte_size: int
    records: int
    sha256: str
    resolve_url: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.filename)
        if path.is_absolute() or ".." in path.parts or path.name != self.filename:
            raise ValueError(f"unsafe public PSV shard name: {self.filename!r}")
        if self.byte_size < PSV_RECORD_BYTES or self.byte_size % PSV_RECORD_BYTES:
            raise ValueError(f"PSV shard is not record aligned: {self.filename}")
        if self.records != self.byte_size // PSV_RECORD_BYTES:
            raise ValueError(f"PSV shard record count mismatch: {self.filename}")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError(f"PSV shard has no strict SHA-256: {self.filename}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingSegment:
    superbatch: int
    corpus_pass: int
    shard_index: int
    filename: str
    batches: int
    presentations: int
    heldout_tail_excluded: int

    def __post_init__(self) -> None:
        if (
            min(
                self.superbatch,
                self.corpus_pass,
                self.shard_index + 1,
                self.batches,
                self.presentations,
            )
            < 1
        ):
            raise ValueError("training segment counters must be positive")
        if self.presentations % DEFAULT_BATCH_SIZE:
            raise ValueError("training segment presentations must be batch aligned")
        if self.heldout_tail_excluded < 0:
            raise ValueError("heldout tail exclusion must not be negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _reject_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    normalized_keys: dict[str, str] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        normalized = key.casefold()
        if normalized in normalized_keys:
            raise ValueError(
                "case-insensitive JSON key collision: "
                f"{normalized_keys[normalized]!r} and {key!r}"
            )
        normalized_keys[normalized] = key
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} root must be a JSON object")
    return cast(dict[str, Any], value)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_stage = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    stage = Path(raw_stage)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage, path)
        _fsync_directory(path.parent)
    except BaseException:
        stage.unlink(missing_ok=True)
        raise


def _fetch_url(url: str, *, timeout_seconds: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return cast(bytes, response.read())


def _validate_local_only_corpus(corpus_id: str, *, acknowledged: bool) -> tuple[str, str]:
    lineage = corpus_lineage(corpus_id)
    if lineage.seed_sampling_scope is not CorpusSeedSamplingScope.USER_ATTESTED_LOCAL_ONLY:
        raise PermissionError(
            "NAGISA-style direct value training requires a reviewed local-only corpus"
        )
    if lineage.move_field_contract is not PsvMoveFieldContract.ZERO_VALUE_ONLY_REQUIRED:
        raise ValueError("direct value training requires the reviewed Move16=0 PSV contract")
    if not acknowledged:
        raise PermissionError(
            "direct corpus use requires --allow-user-attested-local-only acknowledgement"
        )
    if lineage.revision is None:
        raise ValueError("direct value corpus must pin an immutable revision")
    repository_path = lineage.repository_url.removeprefix("https://huggingface.co/datasets/")
    if repository_path == lineage.repository_url or not repository_path:
        raise ValueError("direct value corpus is not a canonical Hugging Face dataset URL")
    return repository_path, lineage.revision


def index_public_value_corpus(
    corpus_id: str = SOUJOU_DATASETS_1_CORPUS_ID,
    *,
    timeout_seconds: float = 60.0,
    allow_user_attested_local_only: bool = False,
) -> dict[str, object]:
    """Bind every PSV shard to a pinned revision, byte count, and LFS SHA-256."""

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    repository_path, revision = _validate_local_only_corpus(
        corpus_id, acknowledged=allow_user_attested_local_only
    )
    lineage = corpus_lineage(corpus_id)
    api_url = (
        f"https://huggingface.co/api/datasets/{repository_path}/revision/{revision}?blobs=true"
    )
    raw = _fetch_url(api_url, timeout_seconds=timeout_seconds)
    payload = _strict_json(raw, label="public corpus API response")
    if payload.get("sha") != revision:
        raise ValueError(
            f"public corpus revision mismatch: expected={revision} observed={payload.get('sha')}"
        )
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        raise ValueError("public corpus siblings must be an array")
    filename_pattern = lineage.sample_filename_pattern
    if filename_pattern is None:
        raise ValueError("direct value corpus has no reviewed shard filename pattern")
    shards: list[PublicPsvShard] = []
    for item in siblings:
        if not isinstance(item, dict) or not isinstance(item.get("rfilename"), str):
            raise ValueError("public corpus sibling lacks a string rfilename")
        filename = cast(str, item["rfilename"])
        if re.fullmatch(filename_pattern, filename) is None:
            continue
        lfs = item.get("lfs")
        if not isinstance(lfs, dict):
            raise ValueError(f"PSV shard has no LFS identity: {filename}")
        byte_size = lfs.get("size")
        sha256 = lfs.get("sha256")
        if not isinstance(byte_size, int) or isinstance(byte_size, bool):
            raise ValueError(f"PSV shard has no integer LFS size: {filename}")
        if not isinstance(sha256, str):
            raise ValueError(f"PSV shard has no LFS SHA-256: {filename}")
        shards.append(
            PublicPsvShard(
                filename=filename,
                byte_size=byte_size,
                records=byte_size // PSV_RECORD_BYTES,
                sha256=sha256,
                resolve_url=(
                    f"https://huggingface.co/datasets/{repository_path}/resolve/{revision}/"
                    f"{quote(filename, safe='/')}"
                ),
            )
        )
    shards.sort(key=lambda shard: shard.filename)
    if not shards or len({shard.filename for shard in shards}) != len(shards):
        raise ValueError("public value corpus has no unique reviewed PSV shards")
    total_bytes = sum(shard.byte_size for shard in shards)
    total_records = sum(shard.records for shard in shards)
    if lineage.data_bytes is not None and total_bytes != lineage.data_bytes:
        raise ValueError(
            "public corpus byte total changed: "
            f"expected={lineage.data_bytes} observed={total_bytes}"
        )
    if lineage.position_records is not None and total_records != lineage.position_records:
        raise ValueError(
            "public corpus record total changed: "
            f"expected={lineage.position_records} observed={total_records}"
        )
    return {
        "schema": "meteo-public-value-psv-index-v1",
        "corpus_id": corpus_id,
        "repository": repository_path,
        "revision": revision,
        "api_url": api_url,
        "api_sha256": hashlib.sha256(raw).hexdigest(),
        "shards": [shard.to_dict() for shard in shards],
        "totals": {
            "shards": len(shards),
            "bytes": total_bytes,
            "records": total_records,
        },
        "rights": {
            "scope": CorpusSeedSamplingScope.USER_ATTESTED_LOCAL_ONLY.value,
            "operator_acknowledged": True,
            "source_and_derived_checkpoints_must_remain_local": True,
            "public_redistribution_allowed": False,
        },
        "label_contract": {
            "move16": PsvMoveFieldContract.ZERO_VALUE_ONLY_REQUIRED.value,
            "scalar_score_is_training_target": True,
            "policy_target_available": False,
            "teacher_reanalysis_required_before_value_training": False,
        },
    }


def _shards_from_index(index: Mapping[str, object]) -> tuple[PublicPsvShard, ...]:
    raw_shards = index.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("corpus index has no shard array")
    shards: list[PublicPsvShard] = []
    for raw in raw_shards:
        if not isinstance(raw, dict):
            raise ValueError("corpus index shard must be an object")
        shards.append(
            PublicPsvShard(
                filename=cast(str, raw.get("filename")),
                byte_size=cast(int, raw.get("byte_size")),
                records=cast(int, raw.get("records")),
                sha256=cast(str, raw.get("sha256")),
                resolve_url=cast(str, raw.get("resolve_url")),
            )
        )
    return tuple(shards)


def extract_nagisa_progress(archive: Path, destination: Path) -> dict[str, object]:
    """Extract only NAGISA's routing model; NNUE student weights stay random."""

    source = archive.expanduser().resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError("NAGISA archive must be a regular non-symlink file")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite progress coefficient: {destination}")
    with zipfile.ZipFile(source) as bundle:
        candidates = []
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("NAGISA archive contains an unsafe member path")
            if not member.is_dir() and path.name == "progress.bin" and "eval" in path.parts:
                candidates.append(member)
        if len(candidates) != 1:
            raise ValueError(
                "NAGISA archive must contain exactly one eval/progress.bin; "
                f"found {len(candidates)}"
            )
        member = candidates[0]
        if member.file_size != PROGRESS_KPABS_BYTES:
            raise ValueError(
                f"progress.bin size mismatch: expected={PROGRESS_KPABS_BYTES} "
                f"observed={member.file_size}"
            )
        payload = bundle.read(member)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_new(destination, payload, mode=0o600)
    return {
        "source_archive": str(source),
        "source_archive_sha256": _sha256_file(source),
        "member": member.filename,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "copied_teacher_nnue_weights": False,
        "role": "fixed_progress_bucket_router_only",
        "rights_scope": "local_only",
    }


def probe_value_only_psv(path: Path, *, board_samples: int = 4_096) -> dict[str, object]:
    """Validate value-only PSV fields without pretending Move16 is policy."""

    source = path.expanduser().resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError("PSV input must be a regular non-symlink file")
    byte_size = source.stat().st_size
    if byte_size < PSV_RECORD_BYTES or byte_size % PSV_RECORD_BYTES:
        raise ValueError("PSV input must contain positive whole 40-byte records")
    if isinstance(board_samples, bool) or board_samples < 1:
        raise ValueError("board_samples must be a positive integer")
    records: Any = np.memmap(source, dtype=PackedSfenValue, mode="r")
    count = len(records)
    score_min = 32_767
    score_max = -32_768
    mate_stamps = 0
    result_counts = {"loss": 0, "draw": 0, "win": 0}
    chunk_records = 1_000_000
    for start in range(0, count, chunk_records):
        end = min(count, start + chunk_records)
        chunk = records[start:end]
        moves = np.asarray(chunk["move"])
        nonzero = np.flatnonzero(moves)
        if nonzero.size:
            first = start + int(nonzero[0])
            raise ValueError(
                f"value-only PSV requires Move16=0; record {first} "
                f"has {int(records[first]['move'])}"
            )
        results = np.asarray(chunk["game_result"])
        invalid_results = np.flatnonzero(~np.isin(results, (-1, 0, 1)))
        if invalid_results.size:
            first = start + int(invalid_results[0])
            raise ValueError(
                f"PSV record {first} has invalid game_result={int(records[first]['game_result'])}"
            )
        result_counts["loss"] += int(np.count_nonzero(results == -1))
        result_counts["draw"] += int(np.count_nonzero(results == 0))
        result_counts["win"] += int(np.count_nonzero(results == 1))
        scores = np.asarray(chunk["score"])
        score_min = min(score_min, int(scores.min()))
        score_max = max(score_max, int(scores.max()))
        mate_stamps += int(np.count_nonzero(np.abs(scores.astype(np.int32)) >= 32_000))
    sample_count = min(count, board_samples)
    indices = np.linspace(0, count - 1, num=sample_count, dtype=np.int64)
    for index in indices:
        board = Board()
        board.set_packed_sfen(records[int(index)]["sfen"].tobytes())
        if not board.is_valid():
            raise ValueError(f"PSV record {int(index)} does not decode to a valid board")
    return {
        "path": str(source),
        "bytes": byte_size,
        "sha256": _sha256_file(source),
        "records": count,
        "move16_zero_records": count,
        "policy_targets": 0,
        "score": {
            "minimum": score_min,
            "maximum": score_max,
            "mate_stamp_abs_32000_or_more": mate_stamps,
        },
        "game_result_counts": result_counts,
        "boards_validated": sample_count,
        "optimizer_contract": "value_only_scalar_score",
    }


def _training_schedule(
    shards: Sequence[PublicPsvShard],
    *,
    target_presentations: int,
    batch_size: int,
    heldout_filename: str,
    heldout_positions: int,
) -> tuple[TrainingSegment, ...]:
    if target_presentations < 1 or batch_size < 1 or heldout_positions < 1:
        raise ValueError("target, batch size, and heldout size must be positive")
    segments: list[TrainingSegment] = []
    remaining = target_presentations
    corpus_pass = 1
    superbatch = 1
    while remaining > 0:
        made_progress = False
        for shard_index, shard in enumerate(shards):
            excluded = heldout_positions if shard.filename == heldout_filename else 0
            available = shard.records - excluded
            if available < batch_size:
                raise ValueError(
                    f"PSV shard is too small after heldout exclusion: {shard.filename}"
                )
            desired = min(available, remaining)
            batches = math.ceil(desired / batch_size)
            if batches * batch_size > available:
                batches = available // batch_size
            presentations = batches * batch_size
            if presentations < 1:
                continue
            segments.append(
                TrainingSegment(
                    superbatch=superbatch,
                    corpus_pass=corpus_pass,
                    shard_index=shard_index,
                    filename=shard.filename,
                    batches=batches,
                    presentations=presentations,
                    heldout_tail_excluded=excluded,
                )
            )
            remaining -= presentations
            superbatch += 1
            made_progress = True
            if remaining <= 0:
                break
        if not made_progress:
            raise ValueError("training schedule cannot make progress")
        corpus_pass += 1
    return tuple(segments)


def _run_git_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise RuntimeError(
            f"Git command failed with status {completed.returncode}: {command!r}: {detail}"
        )
    return completed


def _clone_pinned_tatara(destination: Path, *, patch: Path | None) -> dict[str, object]:
    _run_git_checked(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            TATARA_REPOSITORY_URL,
            str(destination),
        ]
    )
    _run_git_checked(["git", "-C", str(destination), "checkout", "--detach", TATARA_PINNED_COMMIT])
    patch_identity: dict[str, object] | None = None
    if patch is not None:
        patch_path = patch.expanduser().resolve(strict=True)
        _run_git_checked(["git", "-C", str(destination), "apply", "--check", str(patch_path)])
        _run_git_checked(["git", "-C", str(destination), "apply", str(patch_path)])
        patch_identity = {
            "path": str(patch_path),
            "sha256": _sha256_file(patch_path),
            "bytes": patch_path.stat().st_size,
        }
    observed = _run_git_checked(["git", "-C", str(destination), "rev-parse", "HEAD"]).stdout.strip()
    if observed != TATARA_PINNED_COMMIT:
        raise ValueError(
            f"Tatara checkout mismatch: expected={TATARA_PINNED_COMMIT} observed={observed}"
        )
    return {
        "repository": TATARA_REPOSITORY_URL,
        "commit": observed,
        "license": "MIT",
        "patch": patch_identity,
    }


def prepare_nagisa_nnue_run(
    output: Path,
    *,
    nagisa_archive: Path,
    tatara_patch: Path | None,
    target_presentations: int = TARGET_PRESENTATIONS,
    heldout_positions: int = DEFAULT_HELDOUT_POSITIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    score_scale: float = DEFAULT_SCORE_SCALE,
    corpus_id: str = SOUJOU_DATASETS_1_CORPUS_ID,
    timeout_seconds: float = 60.0,
    allow_user_attested_local_only: bool = False,
) -> dict[str, object]:
    """Create a portable, immutable production run plan from random weights."""

    if target_presentations < 1:
        raise ValueError("target_presentations must be positive")
    if batch_size != DEFAULT_BATCH_SIZE:
        raise ValueError(f"NAGISA production batch_size is pinned to {DEFAULT_BATCH_SIZE}")
    wrm = nagisa_wrm_contract(score_scale)
    yaneuraou_fv_scale = int(cast(int, wrm["yaneuraou_fv_scale"]))
    destination = output.expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite NNUE run: {destination}")
    parent = destination.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
    try:
        for relative in ("cache", "checkpoints", "exports", "logs", "private", "receipts"):
            (stage / relative).mkdir(mode=0o700)
        corpus_index = index_public_value_corpus(
            corpus_id,
            timeout_seconds=timeout_seconds,
            allow_user_attested_local_only=allow_user_attested_local_only,
        )
        shards = _shards_from_index(corpus_index)
        heldout_shard = shards[-1]
        if heldout_positions >= heldout_shard.records:
            raise ValueError("heldout tail must be smaller than its source shard")
        progress = extract_nagisa_progress(nagisa_archive, stage / "private" / "progress.bin")
        tatara = _clone_pinned_tatara(stage / "dependencies" / "tatara", patch=tatara_patch)
        schedule = _training_schedule(
            shards,
            target_presentations=target_presentations,
            batch_size=batch_size,
            heldout_filename=heldout_shard.filename,
            heldout_positions=heldout_positions,
        )
        architecture = NagisaNnueArchitecture()
        plan: dict[str, object] = {
            "schema": NAGISA_NNUE_PLAN_SCHEMA,
            "created_unix": time.time(),
            "mode": "value_only_nnue_sfnn",
            "legacy_policy_value_mcts_used": False,
            "architecture": architecture.to_dict(),
            "initialization": {
                "student_weights": "random",
                "teacher_nnue_weights_copied": False,
                "optimizer_state": "new_on_superbatch_1_exact_resume_afterward",
                "progress_router": progress,
            },
            "tatara": tatara,
            "corpus": corpus_index,
            "training": {
                "net_id": DEFAULT_NET_ID,
                "target_presentations": target_presentations,
                "planned_presentations": sum(item.presentations for item in schedule),
                "batch_size": batch_size,
                "superbatches": len(schedule),
                "score_scale": score_scale,
                "yaneuraou_fv_scale": yaneuraou_fv_scale,
                "loss": wrm,
                "score_target": "PackedSfenValue.score",
                "policy_loss": 0.0,
                "wdl_lambda": 0.0,
                "score_drop_abs": None,
                "extreme_score_policy": "keep_including_mate_stamps",
                "learning_rate": 8.75e-4,
                "learning_rate_schedule": "cosine",
                "learning_rate_final": 1e-5,
                "precision": "fp32",
                "ft_factorization": True,
                "checkpoint_generations": 2,
                "quantized_generations": 2,
                "cache_shards": 1,
                "dataloader_threads": "auto_physical_cores_cap_64",
                "dataloader_threads_override": "METEO_NNUE_THREADS_1_to_256",
                "checkpoint_reload_validation": "optimizer_state_plus_heldout_full_pass",
            },
            "heldout": {
                "source_filename": heldout_shard.filename,
                "tail_positions": heldout_positions,
                "byte_start": heldout_shard.byte_size - heldout_positions * PSV_RECORD_BYTES,
                "byte_end_inclusive": heldout_shard.byte_size - 1,
                "test_positions_per_superbatch": heldout_positions,
                "never_used_for_gradient": True,
            },
            "schedule": [item.to_dict() for item in schedule],
            "storage_bound": {
                "one_remote_shard_max_bytes": max(shard.byte_size for shard in shards),
                "raw_checkpoints_kept": 2,
                "tatara_bins_kept": 2,
                "yaneuraou_exports_kept": 2,
                "completed_cache_shard_deleted": True,
            },
            "distribution": {
                "local_training_and_derived_checkpoint_use_only": True,
                "public_checkpoint_release_allowed": False,
            },
        }
        _write_new(stage / "corpus-index.json", _json_bytes(corpus_index))
        _write_new(stage / "plan.json", _json_bytes(plan))
        state = {
            "schema": NAGISA_NNUE_STATE_SCHEMA,
            "status": "prepared",
            "completed_superbatch": 0,
            "presentations": 0,
            "latest_raw_checkpoint": None,
            "latest_tatara_bin": None,
            "latest_yaneuraou_export": None,
            "last_metrics": None,
            "failure": None,
            "updated_unix": time.time(),
        }
        _write_new(stage / "state.json", _json_bytes(state), mode=0o600)
        _write_new(
            stage / "README.txt",
            (
                b"Meteo NAGISA-style value-only NNUE private run.\n"
                b"Source and derived checkpoints are local-only.\n"
                b"Run with: simajilord-nnue run <this-directory>\n"
                b"Stop safely between shards with: touch STOP\n"
            ),
        )
        _fsync_directory(stage)
        os.rename(stage, destination)
        _fsync_directory(parent)
        return plan
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def load_nagisa_plan(run_directory: Path) -> dict[str, Any]:
    path = run_directory.expanduser().resolve(strict=True) / "plan.json"
    value = _strict_json(path.read_bytes(), label="NNUE plan")
    if value.get("schema") != NAGISA_NNUE_PLAN_SCHEMA:
        raise ValueError("NNUE plan schema mismatch")
    return value


def load_nagisa_state(run_directory: Path) -> dict[str, Any]:
    path = run_directory.expanduser().resolve(strict=True) / "state.json"
    value = _strict_json(path.read_bytes(), label="NNUE state")
    if value.get("schema") != NAGISA_NNUE_STATE_SCHEMA:
        raise ValueError("NNUE state schema mismatch")
    return value


def _tatara_wrm_arguments(training: Mapping[str, object]) -> list[str]:
    loss = training.get("loss")
    if not isinstance(loss, dict) or loss.get("kind") != "wrm":
        raise ValueError("NAGISA-style production requires an explicit WRM loss contract")
    expected = nagisa_wrm_contract(float(cast(float, training["score_scale"])))
    if loss != expected:
        raise ValueError("stored WRM loss contract differs from the scale-aligned NAGISA contract")
    return [
        "--win-rate-model",
        "--wrm-nnue2score",
        str(float(cast(float, loss["nnue2score"]))),
        "--wrm-in-scaling",
        str(float(cast(float, loss["in_scaling"]))),
        "--wrm-in-offset",
        str(float(cast(float, loss["in_offset"]))),
        "--wrm-target-scaling",
        str(float(cast(float, loss["target_scaling"]))),
        "--wrm-target-offset",
        str(float(cast(float, loss["target_offset"]))),
        "--loss-pow-exp",
        str(float(cast(float, loss["loss_pow_exp"]))),
        "--loss-qp-asymmetry",
        str(float(cast(float, loss["loss_qp_asymmetry"]))),
        "--loss-weight-boost-w1",
        str(float(cast(float, loss["loss_weight_boost_w1"]))),
        "--loss-weight-boost-w2",
        str(float(cast(float, loss["loss_weight_boost_w2"]))),
    ]


def tatara_command_for_segment(
    run_directory: Path,
    segment: Mapping[str, object],
    *,
    tatara_binary: Path,
    cached_psv: Path,
    resume_checkpoint: Path | None,
) -> list[str]:
    """Build one exact, one-shard Tatara command; no policy option exists."""

    root = run_directory.expanduser().resolve(strict=True)
    plan = load_nagisa_plan(root)
    training = cast(dict[str, Any], plan["training"])
    architecture = cast(dict[str, Any], plan["architecture"])
    heldout = cast(dict[str, Any], plan["heldout"])
    superbatch = int(cast(int, segment["superbatch"]))
    command = [
        str(tatara_binary.expanduser().resolve(strict=True)),
        "--data",
        str(cached_psv.expanduser().resolve(strict=True)),
        "--output",
        str((root / "checkpoints").resolve()),
        "--output-format",
        "tatara",
        "--net-id",
        cast(str, training["net_id"]),
        "--feature-set",
        cast(str, architecture["feature_set"]),
        "--superbatches",
        str(superbatch),
        "--start-superbatch",
        str(superbatch),
        "--batches-per-superbatch",
        str(int(cast(int, segment["batches"]))),
        "--batch-size",
        str(int(cast(int, training["batch_size"]))),
        "--lr",
        str(float(cast(float, training["learning_rate"]))),
        "--lr-schedule",
        cast(str, training["learning_rate_schedule"]),
        "--lr-final",
        str(float(cast(float, training["learning_rate_final"]))),
        "--lr-final-superbatch",
        str(int(cast(int, training["superbatches"]))),
        "--wdl",
        str(float(cast(float, training["wdl_lambda"]))),
        "--scale",
        str(float(cast(float, training["score_scale"]))),
        "--save-rate",
        "1",
        "--keep-checkpoints",
        "2",
        "--threads",
        str(_dataloader_threads()),
    ]
    command.extend(_tatara_wrm_arguments(training))
    score_drop_abs = training.get("score_drop_abs")
    if score_drop_abs is not None:
        command.extend(["--score-drop-abs", str(int(cast(int, score_drop_abs)))])
    if cast(str, segment["filename"]) == cast(str, heldout["source_filename"]):
        command.extend(["--test-tail-positions", str(int(cast(int, heldout["tail_positions"])))])
    else:
        heldout_path = root / "private" / "heldout.psv"
        if not heldout_path.is_file():
            raise FileNotFoundError(f"heldout PSV is missing: {heldout_path}")
        command.extend(["--test-data", str(heldout_path.resolve())])
    command.extend(
        [
            "--test-positions",
            str(int(cast(int, heldout["test_positions_per_superbatch"]))),
        ]
    )
    if resume_checkpoint is not None:
        command.extend(["--resume", str(resume_checkpoint.expanduser().resolve(strict=True))])
    command.extend(
        [
            "layerstack",
            "--progress-coeff",
            str((root / "private" / "progress.bin").resolve()),
            "--bucket-mode",
            cast(str, architecture["bucket_mode"]),
            "--num-buckets",
            str(int(cast(int, architecture["num_buckets"]))),
            "--fv-scale",
            str(int(cast(int, training["yaneuraou_fv_scale"]))),
            "--ft-out",
            str(int(cast(int, architecture["ft_out"]))),
            "--l1",
            str(int(cast(int, architecture["l1_out"]))),
            "--l2",
            str(int(cast(int, architecture["l2_out"]))),
        ]
    )
    if superbatch == 1 and any(flag in command for flag in ("--resume", "--init-from")):
        raise AssertionError("first NNUE segment must start from random weights")
    if superbatch > 1 and "--resume" not in command:
        raise ValueError("NNUE segment after superbatch 1 requires exact optimizer resume")
    return command


def parse_tatara_metrics(lines: Sequence[str]) -> dict[str, object] | None:
    latest: dict[str, object] | None = None
    for line in lines:
        match = _TATARA_LOSS_LINE.search(line)
        if match is None:
            continue
        test_match = _TATARA_TEST_LOSS.search(line)
        latest = {
            "superbatch": int(match.group("superbatch")),
            "loss": float(match.group("loss")),
            "positions_per_second": float(match.group("positions_per_second")),
            "test_loss": (None if test_match is None else float(test_match.group("test_loss"))),
        }
    return latest


def parse_tatara_reload_metrics(lines: Sequence[str]) -> dict[str, object] | None:
    latest: dict[str, object] | None = None
    for line in lines:
        match = _TATARA_EVAL_ONLY_LINE.search(line)
        if match is None:
            continue
        latest = {
            "test_loss": float(match.group("test_loss")),
            "test_accuracy": float(match.group("test_accuracy")),
            "n_positions": int(match.group("n_positions")),
            "n_counted": int(match.group("n_counted")),
        }
    return latest


def tatara_reload_command(
    run_directory: Path,
    *,
    tatara_binary: Path,
    checkpoint: Path,
) -> list[str]:
    """Load weights and every resume-state group, then evaluate held-out data."""

    root = run_directory.expanduser().resolve(strict=True)
    plan = load_nagisa_plan(root)
    training = cast(dict[str, Any], plan["training"])
    architecture = cast(dict[str, Any], plan["architecture"])
    heldout = cast(dict[str, Any], plan["heldout"])
    heldout_path = root / "private" / "heldout.psv"
    command = [
        str(tatara_binary.expanduser().resolve(strict=True)),
        "--eval-only",
        "--resume",
        str(checkpoint.expanduser().resolve(strict=True)),
        "--test-data",
        str(heldout_path.resolve(strict=True)),
        "--test-positions",
        str(int(cast(int, heldout["test_positions_per_superbatch"]))),
        "--output",
        str((root / "checkpoints").resolve()),
        "--net-id",
        cast(str, training["net_id"]),
        "--feature-set",
        cast(str, architecture["feature_set"]),
        "--superbatches",
        str(int(cast(int, training["superbatches"]))),
        "--batch-size",
        str(int(cast(int, training["batch_size"]))),
        "--wdl",
        str(float(cast(float, training["wdl_lambda"]))),
        "--scale",
        str(float(cast(float, training["score_scale"]))),
        "--threads",
        str(_dataloader_threads()),
    ]
    command.extend(_tatara_wrm_arguments(training))
    score_drop_abs = training.get("score_drop_abs")
    if score_drop_abs is not None:
        command.extend(["--score-drop-abs", str(int(cast(int, score_drop_abs)))])
    command.extend(
        [
            "layerstack",
            "--progress-coeff",
            str((root / "private" / "progress.bin").resolve(strict=True)),
            "--bucket-mode",
            cast(str, architecture["bucket_mode"]),
            "--num-buckets",
            str(int(cast(int, architecture["num_buckets"]))),
            "--fv-scale",
            str(int(cast(int, training["yaneuraou_fv_scale"]))),
            "--ft-out",
            str(int(cast(int, architecture["ft_out"]))),
            "--l1",
            str(int(cast(int, architecture["l1_out"]))),
            "--l2",
            str(int(cast(int, architecture["l2_out"]))),
        ]
    )
    return command


def _range_header(value: str | None) -> tuple[int, int, int]:
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value or "")
    if match is None:
        raise ValueError("HTTP range response has no strict Content-Range")
    return tuple(map(int, match.groups()))  # type: ignore[return-value]


def ensure_nagisa_heldout(
    run_directory: Path,
    *,
    timeout_seconds: float = 120.0,
) -> dict[str, object]:
    """Fetch the immutable held-out tail without downloading its 20 GB shard."""

    root = run_directory.expanduser().resolve(strict=True)
    plan = load_nagisa_plan(root)
    heldout = cast(dict[str, Any], plan["heldout"])
    shards = _shards_from_index(cast(dict[str, Any], plan["corpus"]))
    shard = next(item for item in shards if item.filename == cast(str, heldout["source_filename"]))
    destination = root / "private" / "heldout.psv"
    receipt_path = root / "receipts" / "heldout.json"
    if destination.is_file() and receipt_path.is_file():
        existing_receipt = _strict_json(receipt_path.read_bytes(), label="heldout receipt")
        output = existing_receipt.get("output")
        if not isinstance(output, dict):
            raise ValueError("heldout receipt has no output identity")
        if output.get("bytes") != destination.stat().st_size or output.get(
            "sha256"
        ) != _sha256_file(destination):
            raise ValueError("heldout PSV no longer matches its receipt")
        return cast(dict[str, object], existing_receipt)
    if destination.exists() or destination.is_symlink() or receipt_path.exists():
        raise FileExistsError("partial heldout artifacts require explicit operator inspection")
    start = int(cast(int, heldout["byte_start"]))
    end = int(cast(int, heldout["byte_end_inclusive"]))
    request = urllib.request.Request(
        shard.resolve_url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": _USER_AGENT},
    )
    stage = destination.with_name(f".{destination.name}.download")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", response.getcode())
            if status != 206:
                raise ValueError(f"heldout server ignored byte range with HTTP {status}")
            observed = _range_header(response.headers.get("Content-Range"))
            if observed != (start, end, shard.byte_size):
                raise ValueError(
                    f"heldout Content-Range mismatch: expected={(start, end, shard.byte_size)} "
                    f"observed={observed}"
                )
            descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                while chunk := response.read(8 * 1024 * 1024):
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        expected_bytes = end - start + 1
        if stage.stat().st_size != expected_bytes:
            raise ValueError("heldout byte count mismatch after download")
        os.replace(stage, destination)
        _fsync_directory(destination.parent)
        probe = probe_value_only_psv(destination)
        receipt: dict[str, object] = {
            "schema": "meteo-nagisa-heldout-receipt-v1",
            "source": {
                "filename": shard.filename,
                "revision": cast(dict[str, Any], plan["corpus"])["revision"],
                "source_sha256": shard.sha256,
                "byte_range": [start, end],
            },
            "output": probe,
            "contract": {
                "gradient_updates_allowed": False,
                "heldout_only": True,
                "policy_target_available": False,
            },
        }
        _write_new(receipt_path, _json_bytes(receipt), mode=0o600)
        _fsync_directory(receipt_path.parent)
        return receipt
    except BaseException:
        stage.unlink(missing_ok=True)
        raise


def download_nagisa_shard(
    run_directory: Path,
    shard: PublicPsvShard,
    *,
    timeout_seconds: float = 300.0,
) -> Path:
    """Resume-download one LFS object and verify its publisher SHA-256."""

    root = run_directory.expanduser().resolve(strict=True)
    cache = (root / "cache").resolve(strict=True)
    destination = cache / shard.filename
    part = cache / f"{shard.filename}.part"
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("cached PSV shard must be a regular non-symlink")
        if destination.stat().st_size != shard.byte_size:
            raise ValueError("cached PSV shard size differs from the immutable index")
        if _sha256_file(destination) != shard.sha256:
            raise ValueError("cached PSV shard hash differs from the immutable index")
        return destination
    existing = part.stat().st_size if part.is_file() and not part.is_symlink() else 0
    if part.exists() and (part.is_symlink() or not part.is_file()):
        raise ValueError("partial PSV shard must be a regular non-symlink")
    if existing > shard.byte_size:
        raise ValueError("partial PSV shard is larger than the immutable source")
    required = shard.byte_size - existing
    safety = 8 * 1024**3
    free = shutil.disk_usage(cache).free
    if free < required + safety:
        raise OSError(
            f"insufficient free storage for one bounded shard: free={free} "
            f"required_with_safety={required + safety}"
        )
    headers = {"User-Agent": _USER_AGENT}
    if existing:
        headers["Range"] = f"bytes={existing}-{shard.byte_size - 1}"
    request = urllib.request.Request(shard.resolve_url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = getattr(response, "status", response.getcode())
        if existing:
            if status != 206:
                raise ValueError(
                    "public shard server did not honor a resumable Range request; "
                    "the verified partial file was left in place"
                )
            observed = _range_header(response.headers.get("Content-Range"))
            if observed != (existing, shard.byte_size - 1, shard.byte_size):
                raise ValueError("public shard resume Content-Range mismatch")
        elif status not in {200, 206}:
            raise ValueError(f"public shard download failed with HTTP {status}")
        mode = "ab" if existing else "xb"
        with part.open(mode) as stream:
            while chunk := response.read(8 * 1024 * 1024):
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    if part.stat().st_size != shard.byte_size:
        raise IncompleteShardDownloadError(
            f"downloaded shard is incomplete: expected={shard.byte_size} "
            f"observed={part.stat().st_size}"
        )
    observed_sha256 = _sha256_file(part)
    if observed_sha256 != shard.sha256:
        raise ValueError(
            f"downloaded shard SHA-256 mismatch: expected={shard.sha256} observed={observed_sha256}"
        )
    os.replace(part, destination)
    _fsync_directory(cache)
    return destination


def nagisa_hardware_preflight(run_directory: Path) -> dict[str, object]:
    root = run_directory.expanduser().resolve(strict=True)
    plan = load_nagisa_plan(root)
    training_binary = root / "dependencies" / "tatara" / "target" / "release" / "nnue-train"
    converter_binary = root / "dependencies" / "tatara" / "target" / "release" / "net_to_yo"
    reasons: list[str] = []
    if platform.system() != "Linux":
        reasons.append("Tatara production training requires Linux")
    nvidia_smi = shutil.which("nvidia-smi")
    gpu_output: str | None = None
    if nvidia_smi is None:
        reasons.append("nvidia-smi is unavailable; Tatara supports NVIDIA CUDA only")
    else:
        result = subprocess.run(
            [nvidia_smi, "-L"], check=False, capture_output=True, text=True, timeout=15
        )
        gpu_output = result.stdout.strip() or result.stderr.strip()
        if result.returncode != 0 or not result.stdout.strip():
            reasons.append("no usable NVIDIA GPU was reported by nvidia-smi")
    for label, binary in (
        ("Tatara trainer", training_binary),
        ("progress-aware YaneuraOu converter", converter_binary),
    ):
        if not binary.is_file() or binary.is_symlink() or not os.access(binary, os.X_OK):
            reasons.append(f"{label} binary is not built: {binary}")
    progress = root / "private" / "progress.bin"
    progress_plan = cast(
        dict[str, Any],
        cast(dict[str, Any], plan["initialization"])["progress_router"],
    )
    if (
        not progress.is_file()
        or progress.is_symlink()
        or progress.stat().st_size != PROGRESS_KPABS_BYTES
        or _sha256_file(progress) != progress_plan.get("sha256")
    ):
        reasons.append("progress.bin is missing or differs from the immutable plan")
    return {
        "ready": not reasons,
        "reasons": reasons,
        "platform": platform.platform(),
        "nvidia_smi": gpu_output,
        "training_binary": str(training_binary),
        "converter_binary": str(converter_binary),
        "tatara_commit": cast(dict[str, Any], plan["tatara"])["commit"],
    }


def _yaneuraou_architecture(path: Path) -> str:
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError("YaneuraOu export has a truncated header")
        architecture_bytes = int.from_bytes(header[8:12], "little")
        if not 1 <= architecture_bytes <= 16_384:
            raise ValueError("YaneuraOu export has an invalid architecture length")
        raw = stream.read(architecture_bytes)
        if len(raw) != architecture_bytes:
            raise ValueError("YaneuraOu export has a truncated architecture string")
    return raw.decode("utf-8")


def _export_yaneuraou_generation(
    run_directory: Path,
    *,
    superbatch: int,
    tatara_bin: Path,
    converter_binary: Path,
) -> tuple[Path, dict[str, object]]:
    root = run_directory.resolve(strict=True)
    destination = root / "exports" / f"{DEFAULT_NET_ID}-{superbatch}"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite NNUE export: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        nn_bin = stage / "nn.bin"
        progress = root / "private" / "progress.bin"
        plan = load_nagisa_plan(root)
        training = cast(dict[str, Any], plan["training"])
        fv_scale = int(cast(int, training["yaneuraou_fv_scale"]))
        result = subprocess.run(
            [
                str(converter_binary),
                "--input",
                str(tatara_bin),
                "--output",
                str(nn_bin),
                "--assume-progress8kpabs",
                "--progress-coeff",
                str(progress),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "progress-aware YaneuraOu conversion failed: "
                f"stdout={result.stdout!r} stderr={result.stderr!r}"
            )
        os.chmod(nn_bin, 0o600)
        _fsync_file(nn_bin)
        shutil.copyfile(progress, stage / "progress.bin")
        os.chmod(stage / "progress.bin", 0o600)
        _fsync_file(stage / "progress.bin")
        _write_new(
            stage / "eval_options.txt",
            (
                "LS_BUCKET_MODE progress8kpabs\n"
                "LS_PROGRESS_COEFF progress.bin\n"
                f"FV_SCALE {fv_scale}\n"
            ).encode(),
            mode=0o600,
        )
        architecture = _yaneuraou_architecture(nn_bin)
        expected = NagisaNnueArchitecture().yaneuraou_header
        if architecture != expected:
            raise ValueError(
                f"YaneuraOu architecture mismatch: expected={expected!r} observed={architecture!r}"
            )
        receipt: dict[str, object] = {
            "schema": "meteo-nagisa-yaneuraou-export-v2",
            "superbatch": superbatch,
            "architecture": architecture,
            "nn_bin": {
                "bytes": nn_bin.stat().st_size,
                "sha256": _sha256_file(nn_bin),
            },
            "progress_bin": {
                "bytes": (stage / "progress.bin").stat().st_size,
                "sha256": _sha256_file(stage / "progress.bin"),
            },
            "routing": "progress8kpabs",
            "layer_stacks": 9,
            "score_scale": float(cast(float, training["score_scale"])),
            "yaneuraou_fv_scale": fv_scale,
            "extreme_score_policy": training["extreme_score_policy"],
            "local_only": True,
        }
        _write_new(stage / "receipt.json", _json_bytes(receipt), mode=0o600)
        _fsync_directory(stage)
        os.rename(stage, destination)
        _fsync_directory(destination.parent)
        return destination, receipt
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _prune_export_directories(directory: Path, *, keep: int = 2) -> tuple[Path, ...]:
    pattern = re.compile(rf"^{re.escape(DEFAULT_NET_ID)}-(\d+)$")
    numbered: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"NNUE export must be a regular non-symlink directory: {path}")
        numbered.append((int(match.group(1)), path))
    numbered.sort(reverse=True)
    for _, path in numbered[keep:]:
        shutil.rmtree(path)
    _fsync_directory(directory)
    return tuple(path for _, path in numbered[:keep])


def _run_logged(command: Sequence[str], log_path: Path) -> tuple[int, list[str]]:
    lines: list[str] = []
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line.encode())
                log.flush()
                lines.append(line.rstrip("\n"))
            return process.wait(), lines
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=30)
            raise


def run_nagisa_nnue(
    run_directory: Path,
    *,
    timeout_seconds: float = 300.0,
) -> dict[str, object]:
    """Run or resume the production schedule until STOP or 100B presentations."""

    root = run_directory.expanduser().resolve(strict=True)
    plan = load_nagisa_plan(root)
    state = load_nagisa_state(root)
    preflight = nagisa_hardware_preflight(root)
    if not preflight["ready"]:
        state.update(
            {
                "status": "waiting_for_nvidia",
                "failure": {"kind": "hardware_preflight", "details": preflight},
                "updated_unix": time.time(),
            }
        )
        _write_json_atomic(root / "state.json", state)
        reasons = "; ".join(cast(list[str], preflight["reasons"]))
        raise RuntimeError(f"NAGISA NNUE hardware preflight failed: {reasons}")
    ensure_nagisa_heldout(root, timeout_seconds=timeout_seconds)
    schedule = cast(list[dict[str, object]], plan["schedule"])
    shards = {
        shard.filename: shard for shard in _shards_from_index(cast(dict[str, Any], plan["corpus"]))
    }
    completed = int(state.get("completed_superbatch", 0))
    trainer = Path(cast(str, preflight["training_binary"]))
    converter = Path(cast(str, preflight["converter_binary"]))
    state.update({"status": "running", "failure": None, "updated_unix": time.time()})
    _write_json_atomic(root / "state.json", state)
    for segment in schedule[completed:]:
        superbatch = int(cast(int, segment["superbatch"]))
        if (root / "STOP").exists():
            state.update({"status": "stopped_by_marker", "updated_unix": time.time()})
            _write_json_atomic(root / "state.json", state)
            return nagisa_run_status(root)
        shard = shards[cast(str, segment["filename"])]
        cached = download_nagisa_shard(root, shard, timeout_seconds=timeout_seconds)
        source_receipt = root / "receipts" / f"source-{shard.sha256}.json"
        if not source_receipt.exists():
            probe = probe_value_only_psv(cached)
            _write_new(
                source_receipt,
                _json_bytes(
                    {
                        "schema": "meteo-nagisa-source-probe-v1",
                        "source": shard.to_dict(),
                        "probe": probe,
                    }
                ),
            )
        resume = None
        if superbatch > 1:
            previous = root / "checkpoints" / f"{DEFAULT_NET_ID}-{superbatch - 1}.ckpt"
            if not previous.is_file() or previous.is_symlink():
                raise FileNotFoundError(f"exact resume checkpoint is missing: {previous}")
            resume = previous
        command = tatara_command_for_segment(
            root,
            segment,
            tatara_binary=trainer,
            cached_psv=cached,
            resume_checkpoint=resume,
        )
        log_path = root / "logs" / f"tatara-{superbatch:04d}.log"
        try:
            returncode, lines = _run_logged(command, log_path)
        except KeyboardInterrupt:
            state.update({"status": "stopped_by_signal", "updated_unix": time.time()})
            _write_json_atomic(root / "state.json", state)
            raise
        if returncode != 0:
            state.update(
                {
                    "status": "failed",
                    "failure": {
                        "kind": "tatara_exit",
                        "superbatch": superbatch,
                        "returncode": returncode,
                        "log": str(log_path),
                    },
                    "updated_unix": time.time(),
                }
            )
            _write_json_atomic(root / "state.json", state)
            raise RuntimeError(f"Tatara exited with status {returncode}; see {log_path}")
        metrics = parse_tatara_metrics(lines)
        if metrics is None or metrics["superbatch"] != superbatch:
            raise ValueError("Tatara completed without a parseable matching loss/speed record")
        checkpoint = root / "checkpoints" / f"{DEFAULT_NET_ID}-{superbatch}.ckpt"
        tatara_bin = root / "checkpoints" / f"{DEFAULT_NET_ID}-{superbatch}.bin"
        for artifact in (checkpoint, tatara_bin):
            if not artifact.is_file() or artifact.is_symlink() or artifact.stat().st_size < 1:
                raise ValueError(
                    f"Tatara did not atomically publish the expected artifact: {artifact}"
                )
        export, export_receipt = _export_yaneuraou_generation(
            root,
            superbatch=superbatch,
            tatara_bin=tatara_bin,
            converter_binary=converter,
        )
        receipt = {
            "schema": NAGISA_NNUE_SHARD_RECEIPT_SCHEMA,
            "superbatch": superbatch,
            "segment": segment,
            "source": shard.to_dict(),
            "command": command,
            "metrics": metrics,
            "raw_checkpoint": {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256": _sha256_file(checkpoint),
            },
            "tatara_bin": {
                "path": str(tatara_bin),
                "bytes": tatara_bin.stat().st_size,
                "sha256": _sha256_file(tatara_bin),
            },
            "yaneuraou_export": {"path": str(export), "receipt": export_receipt},
        }
        _write_new(
            root / "receipts" / f"superbatch-{superbatch:04d}.json",
            _json_bytes(receipt),
        )
        presentations = int(state.get("presentations", 0)) + int(
            cast(int, segment["presentations"])
        )
        state.update(
            {
                "status": "running",
                "completed_superbatch": superbatch,
                "presentations": presentations,
                "latest_raw_checkpoint": str(checkpoint),
                "latest_tatara_bin": str(tatara_bin),
                "latest_yaneuraou_export": str(export),
                "last_metrics": metrics,
                "failure": None,
                "updated_unix": time.time(),
            }
        )
        _write_json_atomic(root / "state.json", state)
        prune_numbered_artifacts(root / "checkpoints", net_id=DEFAULT_NET_ID, suffix=".ckpt")
        prune_numbered_artifacts(root / "checkpoints", net_id=DEFAULT_NET_ID, suffix=".bin")
        _prune_export_directories(root / "exports")
        cached.unlink()
        _fsync_directory(cached.parent)
    state.update({"status": "complete", "updated_unix": time.time()})
    _write_json_atomic(root / "state.json", state)
    return nagisa_run_status(root)


def prune_numbered_artifacts(
    directory: Path,
    *,
    net_id: str,
    suffix: str,
    keep: int = 2,
) -> tuple[Path, ...]:
    """Keep latest and previous numbered files; never follow symlinks."""

    if keep < 1 or not suffix.startswith("."):
        raise ValueError("artifact keep count and suffix are invalid")
    root = directory.expanduser().resolve(strict=True)
    pattern = re.compile(rf"^{re.escape(net_id)}-(\d+){re.escape(suffix)}$")
    numbered: list[tuple[int, Path]] = []
    for path in root.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"numbered artifact must be a regular non-symlink: {path}")
        numbered.append((int(match.group(1)), path))
    numbered.sort(reverse=True)
    for _, path in numbered[keep:]:
        path.unlink()
    _fsync_directory(root)
    return tuple(path for _, path in numbered[:keep])


def nagisa_run_status(run_directory: Path) -> dict[str, object]:
    root = run_directory.expanduser().resolve(strict=True)
    plan = load_nagisa_plan(root)
    state = load_nagisa_state(root)
    target = int(cast(dict[str, Any], plan["training"])["target_presentations"])
    presentations = int(state.get("presentations", 0))
    metrics = state.get("last_metrics")
    eta_seconds: float | None = None
    if isinstance(metrics, dict):
        speed = metrics.get("positions_per_second")
        if isinstance(speed, (int, float)) and not isinstance(speed, bool) and speed > 0:
            eta_seconds = max(0, target - presentations) / float(speed)
    return {
        "schema": NAGISA_NNUE_STATUS_SCHEMA,
        "run_directory": str(root),
        "status": state.get("status"),
        "completed_superbatch": state.get("completed_superbatch"),
        "planned_superbatches": cast(dict[str, Any], plan["training"])["superbatches"],
        "presentations": presentations,
        "target_presentations": target,
        "completion_fraction": presentations / target,
        "last_metrics": metrics,
        "compute_eta_seconds_excluding_download": eta_seconds,
        "stop_requested": (root / "STOP").exists(),
        "platform": platform.platform(),
        "failure": state.get("failure"),
    }
