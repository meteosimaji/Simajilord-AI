"""Receipt-bound qsearch-PV-leaf conversion for scalar NNUE labels.

YaneuraOu's ``qsearch_psv`` command moves each PSV record to the quiet leaf
reached by that engine's qsearch PV.  It correctly flips score/WDL when the PV
has odd length, but it deliberately does *not* recompute the score.  Therefore
this module exposes conversion as a separate intermediate artifact and marks
the result ineligible for training until the selected anchor scorer has
re-evaluated the converted leaf.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import shutil
import stat
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO

from .artifact_provenance import sha256_file
from .nnue_training import probe_value_only_psv

QSEARCH_LEAF_RECEIPT_SCHEMA = "meteo-qsearch-leaf-conversion-v1"
_DONE_PREFIX = "info string qsearch_psv done:"
_DONE_PATTERN = re.compile(
    r"^info string qsearch_psv done: records=(?P<records>\d+) "
    r"replaced=(?P<replaced>\d+) decode_errors=(?P<decode_errors>\d+) "
    r"illegal_pv=(?P<illegal_pv>\d+) max_leaf_ply=(?P<max_leaf_ply>\d+) "
    r"workers=(?P<workers>\d+)$"
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _read_line(
    process: subprocess.Popen[bytes],
    selector: selectors.BaseSelector,
    buffer: bytearray,
    *,
    deadline: float,
) -> str:
    assert process.stdout is not None
    while True:
        newline = buffer.find(b"\n")
        if newline >= 0:
            raw = bytes(buffer[:newline])
            del buffer[: newline + 1]
            try:
                return raw.rstrip(b"\r").decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("qsearch engine emitted non-UTF-8 stdout") from error
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("qsearch leaf conversion timed out")
        events = selector.select(timeout=remaining)
        if not events:
            raise TimeoutError("qsearch leaf conversion timed out")
        chunk = os.read(process.stdout.fileno(), 65_536)
        if not chunk:
            raise RuntimeError(
                f"qsearch engine exited before completion: returncode={process.poll()}"
            )
        buffer.extend(chunk)


def _send(stream: IO[bytes], command: str) -> None:
    stream.write(command.encode("utf-8") + b"\n")
    stream.flush()


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_symlink_components(path: Path, *, label: str) -> Path:
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink: {current}")
    return absolute


def _regular_file(path: Path, *, label: str) -> Path:
    absolute = _reject_symlink_components(path, label=label)
    try:
        metadata = absolute.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise FileNotFoundError(absolute) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {absolute}")
    return absolute


def _regular_directory(path: Path, *, label: str) -> Path:
    absolute = _reject_symlink_components(path, label=label)
    try:
        metadata = absolute.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise FileNotFoundError(absolute) from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink directory: {absolute}")
    return absolute


@dataclass(frozen=True, slots=True)
class QsearchLeafEngineProfile:
    """Pinned options for a qsearch-capable YaneuraOu executable."""

    threads: int = 1
    hash_mb: int = 64
    eval_dir: str = "eval"
    fv_scale: int = 16
    ls_bucket_mode: str = "progress8kpabs"
    ls_progress_coeff: str = "progress.bin"

    def __post_init__(self) -> None:
        if self.threads < 1 or self.hash_mb < 1 or self.fv_scale < 1:
            raise ValueError("qsearch engine numeric options must be positive")
        for value in (self.eval_dir, self.ls_bucket_mode, self.ls_progress_coeff):
            if not value or "\n" in value or "\r" in value:
                raise ValueError("qsearch engine option values must be non-empty single lines")

    def options(self) -> tuple[tuple[str, str], ...]:
        return (
            ("Threads", str(self.threads)),
            ("USI_Hash", str(self.hash_mb)),
            ("USI_OwnBook", "false"),
            ("BookFile", "no_book"),
            ("EvalDir", self.eval_dir),
            ("FV_SCALE", str(self.fv_scale)),
            ("LS_BUCKET_MODE", self.ls_bucket_mode),
            ("LS_PROGRESS_COEFF", self.ls_progress_coeff),
        )


def convert_qsearch_leaves(
    source_psv: Path,
    *,
    engine: Path,
    working_directory: Path,
    output_directory: Path,
    profile: QsearchLeafEngineProfile | None = None,
    timeout_seconds: float = 300.0,
) -> dict[str, object]:
    """Convert a value-only PSV to qsearch leaves, without claiming rescoring."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if profile is None:
        profile = QsearchLeafEngineProfile()
    source = _regular_file(source_psv, label="source PSV")
    executable = _regular_file(engine, label="engine")
    cwd = _regular_directory(working_directory, label="working_directory")
    destination = _reject_symlink_components(output_directory, label="qsearch leaf artifact")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite qsearch leaf artifact: {destination}")
    source_probe = probe_value_only_psv(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent, label="qsearch leaf artifact parent")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    output = stage / "leaves.psv"
    transport = Path(tempfile.mkdtemp(prefix="meteo-qsearch-"))
    if any(character.isspace() for character in str(transport)):
        transport.rmdir()
        raise RuntimeError("qsearch command transport path unexpectedly contains whitespace")
    transport_input = transport / "input.psv"
    transport_output = transport / "output.psv"
    try:
        os.link(source, transport_input)
    except OSError:
        shutil.copyfile(source, transport_input)
    process = subprocess.Popen(
        [str(executable)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        bufsize=0,
    )
    assert process.stdin is not None and process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    buffer = bytearray()
    transcript: list[str] = []
    deadline = time.monotonic() + timeout_seconds
    try:
        _send(process.stdin, "usi")
        declarations: set[str] = set()
        while True:
            line = _read_line(process, selector, buffer, deadline=deadline)
            transcript.append(line)
            if line.startswith("option name "):
                remainder = line.removeprefix("option name ")
                name, separator, _option_type = remainder.partition(" type ")
                if separator:
                    declarations.add(name.casefold())
            if line == "usiok":
                break
        required_names = {name.casefold() for name, _value in profile.options()}
        missing = sorted(required_names - declarations)
        if missing:
            raise ValueError(f"qsearch engine omitted required options: {missing}")
        for name, value in profile.options():
            _send(process.stdin, f"setoption name {name} value {value}")
        _send(process.stdin, "isready")
        while True:
            line = _read_line(process, selector, buffer, deadline=deadline)
            transcript.append(line)
            if line == "readyok":
                break
        _send(
            process.stdin,
            f"qsearch_psv {transport_input} {transport_output} {profile.threads}",
        )
        done_line: str | None = None
        failed = False
        while done_line is None:
            line = _read_line(process, selector, buffer, deadline=deadline)
            transcript.append(line)
            if line == "info string qsearch_psv failed":
                failed = True
            if line.startswith(_DONE_PREFIX):
                done_line = line
        done_match = _DONE_PATTERN.fullmatch(done_line)
        if done_match is None:
            raise RuntimeError(f"qsearch_psv emitted a malformed completion line: {done_line}")
        done_fields = {name: int(value) for name, value in done_match.groupdict().items()}
        _send(process.stdin, "isready")
        while True:
            line = _read_line(process, selector, buffer, deadline=deadline)
            transcript.append(line)
            if line == "info string qsearch_psv failed":
                failed = True
            if line == "readyok":
                break
        if (
            failed
            or done_fields["decode_errors"] != 0
            or done_fields["illegal_pv"] != 0
            or done_fields["records"] != source_probe["records"]
            or done_fields["workers"] != profile.threads
        ):
            raise RuntimeError(f"qsearch_psv failed: {done_line}")
        _send(process.stdin, "quit")
        process.wait(timeout=max(deadline - time.monotonic(), 0.1))
        if process.returncode != 0:
            raise RuntimeError(f"qsearch engine exited with {process.returncode}")
        if not transport_output.is_file() or transport_output.is_symlink():
            raise RuntimeError("qsearch engine did not produce a regular output PSV")
        shutil.copyfile(transport_output, output)
        output_probe = probe_value_only_psv(output)
        if output_probe["records"] != source_probe["records"]:
            raise ValueError("qsearch conversion changed the PSV record count")
        if output.stat().st_size != source.stat().st_size:
            raise ValueError("qsearch conversion changed the PSV byte length")
        transcript_bytes = "\n".join(transcript).encode("utf-8") + b"\n"
        _write_new(stage / "transcript.txt", transcript_bytes)
        receipt: dict[str, object] = {
            "schema": QSEARCH_LEAF_RECEIPT_SCHEMA,
            "source": {
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
                "records": source_probe["records"],
            },
            "engine": {
                "sha256": sha256_file(executable),
                "profile": asdict(profile),
            },
            "output": {
                "file": "leaves.psv",
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
                "records": output_probe["records"],
            },
            "transcript": {
                "file": "transcript.txt",
                "sha256": hashlib.sha256(transcript_bytes).hexdigest(),
                "done_line": done_line,
                "done_fields": done_fields,
            },
            "score_was_recomputed_at_leaf": False,
            "eligible_for_value_training": False,
            "required_next_stage": "single_anchor_rescore_every_leaf",
            "complete": True,
        }
        _write_new(stage / "receipt.json", _json_bytes(receipt))
        descriptor = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.rename(stage, destination)
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return receipt
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        for child in stage.iterdir():
            child.unlink(missing_ok=True)
        stage.rmdir()
        raise
    finally:
        selector.close()
        for child in transport.iterdir():
            child.unlink(missing_ok=True)
        transport.rmdir()
