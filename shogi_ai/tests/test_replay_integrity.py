from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
from rsshogi.core import Board

from simajilord_shogi.cli import main
from simajilord_shogi.domain import GameRecord, PositionSample, Termination
from simajilord_shogi.replay import append_games, load_games
from simajilord_shogi.replay_integrity import sanitize_replay_rules

INFERIOR_CYCLE_SFEN = (
    "ln1g1gsnl/1r1s1k1b1/2p1pp2p/p2p2pp1/1p6P/2P1P4/"
    "PPGP1PPP1/1B2RK3/LNS2GSNL b - 17"
)
INFERIOR_CYCLE_MOVES = (
    "9g9f",
    "9a9c",
    "7f7e",
    "3a3b",
    "9i9g",
    "3b3c",
    "8g8f",
    "4c4d",
    "4g4f",
    "8e8f",
    "6g6f",
    "6b6c",
    "5f5e",
    "6c5d",
    "4h5i",
    "8f8g+",
    "7g8g",
    "P*8f",
    "8g7g",
)
DRAW_SFEN = "4k4/9/9/9/9/9/9/9/4K4 b - 1"
DRAW_MOVES = ("5i5h", "5a5b", "5h5i", "5b5a") * 3


def _sample(*, sfen: str, ply: int) -> PositionSample:
    return PositionSample(
        sfen=sfen,
        ply=ply,
        turn=ply % 2,
        policy={"5i5h": 1.0},
        root_value=0.0,
        teacher_policy={"5i5h": 1.0},
        teacher_value=0.0,
        teacher_source="nagisa-v3.1",
    )


def test_rules_sanitizer_drops_nonterminal_dominance_cycle_and_keeps_draw(
    tmp_path: Path,
) -> None:
    source = tmp_path / "teacher.jsonl"
    output = tmp_path / "clean" / "teacher.rules-clean.jsonl"
    invalid = GameRecord(
        initial_sfen=INFERIOR_CYCLE_SFEN,
        moves=INFERIOR_CYCLE_MOVES,
        samples=(_sample(sfen=INFERIOR_CYCLE_SFEN, ply=0),) * 3,
        winner=1,
        termination=Termination.REPETITION,
    )
    valid = GameRecord(
        initial_sfen=DRAW_SFEN,
        moves=DRAW_MOVES,
        samples=(_sample(sfen=DRAW_SFEN, ply=0),),
        winner=None,
        termination=Termination.REPETITION,
    )
    append_games(source, (invalid, valid))
    # A non-rights dedup receipt must not hide the direct NAGISA teacher ID.
    source.with_suffix(source.suffix + ".dedup.json").write_text(
        json.dumps({"schema": "fixture-dedup-v1"}), encoding="utf-8"
    )

    payload = sanitize_replay_rules(source, output)
    source_receipt = cast(dict[str, object], payload["source"])
    output_receipt = cast(dict[str, object], payload["output"])
    dropped_games = cast(list[dict[str, object]], payload["dropped_games"])

    assert load_games(output) == [valid]
    assert payload["dropped_game_count"] == 1
    assert payload["dropped_sample_count"] == 3
    assert source_receipt["games"] == 2
    assert output_receipt["games"] == 1
    assert payload["teacher_rights"] == [
        {
            "rights_id": "nagisa-v3.1",
            "output_only_meteo_publication": "allowed",
            "publication_allowed": True,
        }
    ]
    assert payload["publication_allowed"] is True
    dropped = dropped_games[0]
    assert dropped["observed_repetition_state"] == "rep_inf"
    assert dropped["reason"] == "recorded_repetition_is_not_rules_terminal"

    provenance = output.with_suffix(output.suffix + ".provenance.json")
    assert json.loads(provenance.read_text(encoding="utf-8")) == payload
    assert output_receipt["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_rules_sanitizer_refuses_symlinks_and_overwrites(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    append_games(
        source,
        [
            GameRecord(
                initial_sfen=Board().to_sfen(),
                moves=(),
                samples=(),
                winner=None,
                termination=Termination.AGREED_DRAW,
            )
        ],
    )
    link = tmp_path / "source-link.jsonl"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="non-symlink"):
        sanitize_replay_rules(link, tmp_path / "link-output.jsonl")

    output = tmp_path / "output.jsonl"
    sanitize_replay_rules(source, output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        sanitize_replay_rules(source, output)


def test_sanitize_replay_rules_cli_prints_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    append_games(
        source,
        [
            GameRecord(
                initial_sfen=DRAW_SFEN,
                moves=DRAW_MOVES,
                samples=(),
                winner=None,
                termination=Termination.REPETITION,
            )
        ],
    )

    assert main(["sanitize-replay-rules", str(source), str(output)]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["dropped_game_count"] == 0
    assert printed["output"]["games"] == 1


def test_init_seed_is_reproducible_and_recorded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    different = tmp_path / "different"

    assert main(["init", str(first), "--profile", "smoke", "--seed", "73"]) == 0
    capsys.readouterr()
    assert main(["init", str(second), "--profile", "smoke", "--seed", "73"]) == 0
    capsys.readouterr()
    assert main(["init", str(different), "--profile", "smoke", "--seed", "74"]) == 0
    capsys.readouterr()

    assert (first / "weights.safetensors").read_bytes() == (
        second / "weights.safetensors"
    ).read_bytes()
    assert (first / "weights.safetensors").read_bytes() != (
        different / "weights.safetensors"
    ).read_bytes()
    metadata = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["step"] == 0
    assert metadata["model"]["name"] == "smoke_2x32"
    assert metadata["lineage"]["schema"] == "meteo-random-initialization-v1"
    assert metadata["lineage"]["seed"] == 73
    assert metadata["lineage"]["teacher_weights_copied"] is False

    with pytest.raises(ValueError, match="non-negative"):
        main(["init", str(tmp_path / "negative"), "--seed", "-1"])


def test_init_can_create_a_clean_canonical_v2_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "canonical-v2"

    assert (
        main(
            [
                "init",
                str(output),
                "--profile",
                "smoke",
                "--seed",
                "20260811",
                "--canonical-v2",
            ]
        )
        == 0
    )

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["step"] == 0
    assert metadata["model"]["history_input_version"] == 2
    assert metadata["model"]["canonical_head_version"] == 2
    assert metadata["model"]["canonical_teacher_count"] == 3
    assert metadata["lineage"]["canonical_target_version"] == 2
    assert metadata["lineage"]["history_input_version"] == 2
    assert metadata["lineage"]["canonical_head_version"] == 2
    assert metadata["lineage"]["canonical_teacher_count"] == 3
    assert metadata["lineage"]["teacher_weights_copied"] is False
