from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from rsshogi.core import Board

from simajilord_shogi.cli import build_parser, main
from simajilord_shogi.domain import (
    GameRecord,
    PositionSample,
    TeacherScoreBound,
    TeacherScoreKind,
    TeacherVariation,
    Termination,
)
from simajilord_shogi.replay import append_games, load_games
from simajilord_shogi.value_scale_ablation import (
    DLSHOGI_COEFFICIENT_756,
    PONANZA_COEFFICIENT_600,
    ValueScaleAblationConfig,
    ponanza_signed_value,
    ponanza_win_probability,
    prepare_value_scale_ablation,
    rescale_cp_between_ponanza_coefficients,
)


def _unique_positions(count: int) -> list[str]:
    root = Board()
    frontier = [root.to_sfen()]
    seen = {" ".join(root.to_sfen().split()[:3])}
    positions: list[str] = []
    while frontier and len(positions) < count:
        sfen = frontier.pop(0)
        board = Board(sfen)
        if board.legal_moves():
            positions.append(sfen)
        for move in sorted(board.legal_moves(), key=lambda candidate: candidate.to_usi()):
            child = Board(sfen)
            child.apply_move(move)
            child_sfen = child.to_sfen()
            normalized = " ".join(child_sfen.split()[:3])
            if normalized not in seen:
                seen.add(normalized)
                frontier.append(child_sfen)
            if len(frontier) + len(positions) >= count * 3:
                break
    if len(positions) < count:
        raise AssertionError("fixture failed to generate enough unique shogi positions")
    return positions[:count]


def _labelled_games(
    positions: list[str],
    *,
    true_coefficient: float,
    repeats_per_cp: int,
) -> list[GameRecord]:
    cps = (-900, -600, -350, -150, 150, 350, 600, 900)
    if len(positions) != len(cps) * repeats_per_cp:
        raise AssertionError("fixture position count does not match cp grid")
    games: list[GameRecord] = []
    position_index = 0
    for cp in cps:
        probability = ponanza_win_probability(cp, true_coefficient)
        for repeat in range(repeats_per_cp):
            board = Board(positions[position_index])
            position_index += 1
            moves = sorted(board.legal_moves(), key=lambda move: move.to_usi())
            best = moves[0].to_usi()
            alternative = moves[1].to_usi() if len(moves) > 1 else best
            # Deterministic stratification approximates the requested logistic
            # probability without making the test dependent on an RNG version.
            label = float((repeat + 0.5) / repeats_per_cp < probability)
            winner = board.turn.value if label == 1.0 else board.turn.opponent().value
            sample = PositionSample(
                sfen=board.to_sfen(),
                ply=0,
                turn=board.turn.value,
                policy={best: 1.0},
                root_value=0.0,
                value_target=1.0 if label == 1.0 else -1.0,
                teacher_policy={best: 0.8, alternative: 0.2},
                teacher_value=0.0,
                teacher_best_move=best,
                teacher_source="fixture-teacher",
                teacher_variations=(
                    TeacherVariation(
                        rank=1,
                        move=best,
                        score_kind=TeacherScoreKind.CENTIPAWN,
                        bound=TeacherScoreBound.EXACT,
                        pv=(best,),
                        score_cp=cp,
                    ),
                ),
                teacher_policy_temperature=173.0,
                teacher_value_scale=999.0,
            )
            games.append(
                GameRecord(
                    initial_sfen=board.to_sfen(),
                    moves=(),
                    samples=(sample,),
                    winner=winner,
                    termination=Termination.RESIGNATION,
                )
            )
    return games


def _parent_checkpoint(path: Path) -> None:
    path.mkdir()
    (path / "metadata.json").write_text('{"step": 7}\n', encoding="utf-8")
    (path / "weights.safetensors").write_bytes(b"fixed-parent-weights")


def test_ponanza_coefficient_is_probability_denominator_not_tanh_denominator() -> None:
    probability = ponanza_win_probability(600, PONANZA_COEFFICIENT_600)
    signed = ponanza_signed_value(600, PONANZA_COEFFICIENT_600)

    assert probability == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    assert signed == pytest.approx(2.0 * probability - 1.0)
    assert signed == pytest.approx(math.tanh(0.5))
    assert rescale_cp_between_ponanza_coefficients(
        DLSHOGI_COEFFICIENT_756,
        source_coefficient=DLSHOGI_COEFFICIENT_756,
        target_coefficient=PONANZA_COEFFICIENT_600,
    ) == pytest.approx(600.0)


def test_ablation_emits_equal_data_fixed_arms_and_heldout_gated_fit(tmp_path: Path) -> None:
    repeats = 12
    positions = _unique_positions(16 * repeats)
    train_games = _labelled_games(
        positions[: 8 * repeats], true_coefficient=220.0, repeats_per_cp=repeats
    )
    validation_games = _labelled_games(
        positions[8 * repeats :], true_coefficient=220.0, repeats_per_cp=repeats
    )
    train_replay = tmp_path / "train.jsonl"
    validation_replay = tmp_path / "validation.jsonl"
    checkpoint = tmp_path / "parent"
    output = tmp_path / "ablation"
    append_games(train_replay, train_games)
    append_games(validation_replay, validation_games)
    _parent_checkpoint(checkpoint)

    manifest = prepare_value_scale_ablation(
        train_replay,
        validation_replay,
        output,
        checkpoint,
        train_split="fixture-train",
        validation_split="fixture-validation",
        config=ValueScaleAblationConfig(
            training_seed=37,
            steps=11,
            batch_size=8,
            learning_rate=2e-4,
            bootstrap_resamples=40,
            minimum_fit_games=40,
            minimum_fit_samples=40,
            minimum_validation_games=40,
            minimum_validation_samples=40,
        ),
    )

    assert manifest["three_arm_training_ready"] is True
    persisted = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["schema"] == "meteo-value-scale-ablation-v1"
    assert persisted["splits"]["normalized_position_overlap"] == 0
    assert persisted["formula_contract"]["win_probability"] == "p = sigmoid(cp / C)"
    assert persisted["formula_contract"]["stored_teacher_value_scale"].startswith("2C")
    fit_validation = persisted["fit_gate"]["teacher_reports"]["fixture-teacher"][
        "validation"
    ]
    assert (
        fit_validation["fitted_improvement_cluster_bootstrap"]["improvement_p2_5"] > 0
    )

    arm_names = ("ponanza-600", "dlshogi-756.086496", "teacher-fit")
    train_outputs = [load_games(output / "train" / f"{name}.jsonl") for name in arm_names]
    validation_outputs = [
        load_games(output / "validation" / f"{name}.jsonl") for name in arm_names
    ]
    assert {sum(len(game.samples) for game in games) for games in train_outputs} == {
        len(train_games)
    }
    assert {sum(len(game.samples) for game in games) for games in validation_outputs} == {
        len(validation_games)
    }
    fixed_600 = train_outputs[0][0].samples[0]
    fixed_756 = train_outputs[1][0].samples[0]
    assert fixed_600.teacher_value_scale == 1200.0
    assert fixed_756.teacher_value_scale == pytest.approx(2.0 * DLSHOGI_COEFFICIENT_756)
    assert fixed_600.teacher_policy == fixed_756.teacher_policy
    assert fixed_600.teacher_policy_temperature == 173.0
    assert fixed_600.teacher_value != fixed_756.teacher_value
    assert {
        persisted["arms"][name]["train_replay"]["position_fingerprint"]
        for name in arm_names
    } == {persisted["data_contract"]["train"]["position_fingerprint"]}
    assert {
        persisted["arms"][name]["train_replay"]["policy_fingerprint"]
        for name in arm_names
    } == {persisted["data_contract"]["train"]["policy_fingerprint"]}
    for name in arm_names:
        command = persisted["arms"][name]["training_command"]
        assert command[command.index("--seed") + 1] == "37"
        assert command[command.index("--steps") + 1] == "11"


def test_small_fit_is_blocked_without_emitting_a_teacher_fit_replay(tmp_path: Path) -> None:
    positions = _unique_positions(16)
    train_replay = tmp_path / "train.jsonl"
    validation_replay = tmp_path / "validation.jsonl"
    checkpoint = tmp_path / "parent"
    output = tmp_path / "ablation"
    append_games(
        train_replay,
        _labelled_games(positions[:8], true_coefficient=220.0, repeats_per_cp=1),
    )
    append_games(
        validation_replay,
        _labelled_games(positions[8:], true_coefficient=220.0, repeats_per_cp=1),
    )
    _parent_checkpoint(checkpoint)

    manifest = prepare_value_scale_ablation(
        train_replay,
        validation_replay,
        output,
        checkpoint,
        config=ValueScaleAblationConfig(bootstrap_resamples=10),
    )

    assert manifest["three_arm_training_ready"] is False
    assert (output / "train" / "ponanza-600.jsonl").is_file()
    assert (output / "train" / "dlshogi-756.086496.jsonl").is_file()
    assert not (output / "train" / "teacher-fit.jsonl").exists()
    teacher_report = manifest["fit_gate"]["teacher_reports"]["fixture-teacher"]
    assert teacher_report["status"] == "blocked"
    assert any(
        reason.startswith("training_fit:small_")
        for reason in teacher_report["blocking_reasons"]
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_value_scale_ablation(
            train_replay,
            validation_replay,
            output,
            checkpoint,
            config=ValueScaleAblationConfig(bootstrap_resamples=10),
        )


def test_split_overlap_fails_before_creating_output(tmp_path: Path) -> None:
    positions = _unique_positions(8)
    games = _labelled_games(positions, true_coefficient=220.0, repeats_per_cp=1)
    train_replay = tmp_path / "train.jsonl"
    validation_replay = tmp_path / "validation.jsonl"
    checkpoint = tmp_path / "parent"
    output = tmp_path / "ablation"
    append_games(train_replay, games)
    append_games(validation_replay, games)
    _parent_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="normalized-position overlap"):
        prepare_value_scale_ablation(
            train_replay,
            validation_replay,
            output,
            checkpoint,
            config=ValueScaleAblationConfig(bootstrap_resamples=10),
        )

    assert not output.exists()


def test_cli_exposes_one_shared_training_seed_and_writes_manifest(tmp_path: Path) -> None:
    positions = _unique_positions(16)
    train_replay = tmp_path / "train.jsonl"
    validation_replay = tmp_path / "validation.jsonl"
    checkpoint = tmp_path / "parent"
    output = tmp_path / "ablation"
    append_games(
        train_replay,
        _labelled_games(positions[:8], true_coefficient=220.0, repeats_per_cp=1),
    )
    append_games(
        validation_replay,
        _labelled_games(positions[8:], true_coefficient=220.0, repeats_per_cp=1),
    )
    _parent_checkpoint(checkpoint)
    argv = [
        "prepare-value-scale-ablation",
        str(train_replay),
        str(validation_replay),
        str(output),
        "--parent-checkpoint",
        str(checkpoint),
        "--training-seed",
        "91",
        "--bootstrap-resamples",
        "10",
    ]

    parsed = build_parser().parse_args(argv)
    assert parsed.command == "prepare-value-scale-ablation"
    assert parsed.training_seed == 91
    assert main(argv) == 0

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_contract"]["training_seed"] == 91
    for name in ("ponanza-600", "dlshogi-756.086496"):
        command = manifest["arms"][name]["training_command"]
        assert command[command.index("--seed") + 1] == "91"
