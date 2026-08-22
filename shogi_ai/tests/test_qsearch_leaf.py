from __future__ import annotations

import json
import stat
from pathlib import Path

import numpy as np
import pytest
from rsshogi.core import Board
from rsshogi.numpy import PackedSfenValue

from simajilord_shogi.qsearch_leaf import (
    QSEARCH_LEAF_RECEIPT_SCHEMA,
    convert_qsearch_leaves,
)


def _psv(path: Path) -> None:
    record = np.zeros(1, dtype=PackedSfenValue)
    record[0]["sfen"] = np.frombuffer(Board().to_packed_sfen(), dtype=np.uint8)
    record[0]["score"] = 123
    path.write_bytes(record.tobytes())


def _engine(path: Path, *, fail: bool = False) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import shutil
import sys

OPTIONS = ['Threads', 'USI_Hash', 'USI_OwnBook', 'BookFile', 'EvalDir',
           'FV_SCALE', 'LS_BUCKET_MODE', 'LS_PROGRESS_COEFF']
for raw in sys.stdin:
    line = raw.strip()
    if line == 'usi':
        print('id name Fake Qsearch')
        for name in OPTIONS:
            print(f'option name {name} type string default x')
        print('usiok', flush=True)
    elif line == 'isready':
        print('readyok', flush=True)
    elif line.startswith('qsearch_psv '):
        _, source, output, workers = line.split()
        shutil.copyfile(source, output)
        print('info string qsearch_psv done: records=1 replaced=0 '
              'decode_errors=0 illegal_pv=0 max_leaf_ply=0 workers=' + workers,
              flush=True)
        if FAIL:
            print('info string qsearch_psv failed', flush=True)
    elif line == 'quit':
        break
""".replace("FAIL", "True" if fail else "False"),
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_qsearch_leaf_conversion_is_receipted_but_not_training_eligible(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.psv"
    engine = tmp_path / "engine.py"
    _psv(source)
    _engine(engine)

    receipt = convert_qsearch_leaves(
        source,
        engine=engine,
        working_directory=tmp_path,
        output_directory=tmp_path / "leaves",
    )

    assert receipt["schema"] == QSEARCH_LEAF_RECEIPT_SCHEMA
    assert receipt["score_was_recomputed_at_leaf"] is False
    assert receipt["eligible_for_value_training"] is False
    assert receipt["required_next_stage"] == "single_anchor_rescore_every_leaf"
    persisted = json.loads((tmp_path / "leaves" / "receipt.json").read_text())
    assert persisted == receipt


def test_qsearch_leaf_conversion_is_create_only(tmp_path: Path) -> None:
    source = tmp_path / "input.psv"
    engine = tmp_path / "engine.py"
    _psv(source)
    _engine(engine)
    output = tmp_path / "leaves"
    convert_qsearch_leaves(
        source,
        engine=engine,
        working_directory=tmp_path,
        output_directory=output,
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        convert_qsearch_leaves(
            source,
            engine=engine,
            working_directory=tmp_path,
            output_directory=output,
        )


def test_qsearch_leaf_conversion_supports_workspace_paths_with_spaces(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace with spaces"
    workspace.mkdir()
    source = workspace / "input with spaces.psv"
    engine = workspace / "engine with spaces.py"
    _psv(source)
    _engine(engine)

    receipt = convert_qsearch_leaves(
        source,
        engine=engine,
        working_directory=workspace,
        output_directory=workspace / "leaf output",
    )

    assert receipt["output"]["records"] == 1


def test_qsearch_leaf_conversion_drains_post_completion_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.psv"
    engine = tmp_path / "engine.py"
    output = tmp_path / "leaves"
    _psv(source)
    _engine(engine, fail=True)

    with pytest.raises(RuntimeError, match="qsearch_psv failed"):
        convert_qsearch_leaves(
            source,
            engine=engine,
            working_directory=tmp_path,
            output_directory=output,
        )
    assert not output.exists()
