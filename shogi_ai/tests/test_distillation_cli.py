from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from rsshogi.core import Board

from simajilord_shogi.checkpoint import load_checkpoint_with_training_state, save_checkpoint
from simajilord_shogi.cli import (
    _checkpoint_sha256,
    _training_input_provenance,
    _verify_training_input_provenance_unchanged,
    build_parser,
    main,
)
from simajilord_shogi.config import model_profile
from simajilord_shogi.domain import GameRecord, PositionSample, Termination
from simajilord_shogi.model import PolicyValueResNet
from simajilord_shogi.replay import append_games

from .test_distillation_targets import _payload, _replay, _write_sidecar
from .test_mcts_game import MATE_IN_ONE_SFEN


def _teacher_replay(
    path: Path, *, teacher_source: str = "held-out-teacher"
) -> None:
    sample = PositionSample(
        sfen=MATE_IN_ONE_SFEN,
        ply=0,
        turn=0,
        policy={"G*5b": 1.0},
        root_value=0.0,
        value_target=1.0,
        chosen_move="G*5b",
        actor_best_move="G*5b",
        teacher_policy={"G*5b": 0.75, "6c7d": 0.25},
        teacher_value=1.0,
        teacher_source=teacher_source,
        teacher_context="forced-mate",
    )
    duplicate = replace(sample, sfen=MATE_IN_ONE_SFEN.rsplit(" ", 1)[0] + " 12")
    append_games(
        path,
        [
            GameRecord(
                initial_sfen=Board().to_sfen(),
                moves=(),
                samples=(sample, duplicate),
                winner=None,
                termination=Termination.REPETITION,
            )
        ],
    )


def test_evaluate_distillation_cli_writes_hashed_wrapper_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / "checkpoint"
    replay = tmp_path / "held-out.jsonl"
    output = tmp_path / "reports" / "alignment.json"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), checkpoint, step=17)
    _teacher_replay(replay)

    assert (
        main(
            [
                "evaluate-distillation",
                str(checkpoint),
                str(replay),
                "--batch-size",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert printed == payload
    assert payload["schema"] == "meteo-distillation-evaluation-v1"
    assert payload["output"] == str(output.resolve())
    assert payload["batch_size"] == 1
    assert payload["checkpoint"]["path"] == str(checkpoint.resolve())
    assert payload["checkpoint"]["sha256"] == _checkpoint_sha256(checkpoint.resolve())
    assert payload["checkpoint"]["step"] == 17
    assert payload["checkpoint"]["model_profile"] == "smoke_2x32"
    assert len(payload["checkpoint"]["weights"]["sha256"]) == 64
    assert payload["replay"]["path"] == str(replay.resolve())
    assert payload["replay"]["sha256"] == hashlib.sha256(replay.read_bytes()).hexdigest()
    assert payload["replay"]["raw_samples"] == 2
    assert payload["replay"]["eligible_samples"] == 2
    assert payload["replay"]["teacher_labelled_samples"] == 2
    assert payload["replay"]["teacher_source_position_pairs"] == 1
    assert payload["replay"]["teacher_unique_positions"] == 1
    assert payload["replay"]["evaluated_samples"] == 1
    assert payload["replay"]["evaluated_unique_positions"] == 1
    assert payload["metrics"]["overall"]["samples"] == 1
    assert payload["metrics"]["by_teacher"]["held-out-teacher"]["samples"] == 1
    assert payload["metrics"]["by_phase"]["opening"]["samples"] == 1
    for metric in (
        "policy_cross_entropy",
        "policy_js_divergence",
        "teacher_mass_at_1",
        "teacher_mass_at_3",
        "teacher_mass_at_5",
        "teacher_best_top_1",
        "teacher_best_top_3",
        "teacher_best_top_5",
        "value_mse",
        "value_brier",
    ):
        assert metric in payload["metrics"]["overall"]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(
            [
                "evaluate-distillation",
                str(checkpoint),
                str(replay),
                "--output",
                str(output),
            ]
        )


def test_evaluate_distillation_parser_defaults_to_batch_size_32() -> None:
    args = build_parser().parse_args(
        ["evaluate-distillation", "checkpoint", "held-out.jsonl"]
    )

    assert args.batch_size == 32
    assert args.output is None


def test_train_cli_persists_and_then_restores_exact_adamw_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = tmp_path / "base"
    replay = tmp_path / "teacher.jsonl"
    first_output = tmp_path / "generation-1"
    second_output = tmp_path / "generation-2"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), base, step=3)
    _teacher_replay(replay)
    shared_arguments = [
        str(replay),
        "--steps",
        "1",
        "--batch-size",
        "1",
        "--learning-rate",
        "0.0001",
        "--seed",
        "23",
        "--telemetry-interval",
        "1",
        "--max-probe-loss-ratio",
        "100",
    ]

    assert (
        main(
            ["train", str(base), shared_arguments[0], str(first_output), *shared_arguments[1:]]
        )
        == 0
    )
    first_report = json.loads(capsys.readouterr().out)
    first_model, first_step, first_state, first_trace = (
        load_checkpoint_with_training_state(first_output)
    )
    assert first_model is not None
    assert first_step == 4
    assert first_state is not None
    assert first_state.optimizer_step == 1
    assert len(first_trace) == 1
    assert first_report["exact_resume_from_parent"] is False
    assert first_report["output_contains_exact_resume_state"] is True

    assert (
        main(
            [
                "train",
                str(first_output),
                shared_arguments[0],
                str(second_output),
                *shared_arguments[1:],
            ]
        )
        == 0
    )
    second_report = json.loads(capsys.readouterr().out)
    _second_model, second_step, second_state, second_trace = (
        load_checkpoint_with_training_state(second_output)
    )
    assert second_step == 5
    assert second_state is not None
    assert second_state.optimizer_step == 2
    assert len(second_trace) == 1
    assert second_report["exact_resume_from_parent"] is True
    metadata = json.loads((second_output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["lineage"]["exact_resume_from_parent"] is True
    assert metadata["lineage"]["optimizer"]["state_restored"] is True
    first_metadata = json.loads(
        (first_output / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["lineage"]["rights_restriction_summary"] == (
        first_metadata["lineage"]["rights_restriction_summary"]
    )

    with pytest.raises(ValueError, match="new checkpoint"):
        main(["train", str(second_output), str(replay), str(second_output), "--steps", "1"])


def test_train_cli_flattens_ensemble_rights_without_private_sidecar_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = tmp_path / "base"
    replay = tmp_path / "ensemble.jsonl"
    output = tmp_path / "candidate"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), base, step=0)
    _teacher_replay(replay, teacher_source="meteo-teacher-ensemble")
    sidecar = replay.with_suffix(replay.suffix + ".ensemble.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema": "meteo-teacher-ensemble-v1",
                "inputs": [
                    {
                        "provenance": {
                            "engine_working_directory": (
                                "/" + "Users" + "/private/nagisa"
                            ),
                            "engine_sha256": "e" * 64,
                            "rights": {
                                "rights_id": "nagisa-v3.1",
                                "output_only_meteo_publication": "allowed",
                                "sources": ["https://example.invalid/private-nagisa"],
                            },
                        }
                    },
                    {
                        "provenance": {
                            "rights_mode": "limited_local",
                            "publication_allowed": False,
                            "local_only_root": "/" + "private" + "/tmp/suisho",
                            "startup_provenance_sha256": "f" * 64,
                            "rights": {
                                "rights_id": (
                                    "suisho11plus-wcsc36-20260525-local"
                                ),
                                "output_only_meteo_publication": "not_approved",
                                "sources": ["https://example.invalid/paid-suisho"],
                            },
                        }
                    },
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "train",
                str(base),
                str(replay),
                str(output),
                "--steps",
                "1",
                "--batch-size",
                "1",
                "--learning-rate",
                "0.0001",
                "--seed",
                "29",
                "--max-probe-loss-ratio",
                "100",
            ]
        )
        == 0
    )
    capsys.readouterr()
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    summary = metadata["lineage"]["rights_restriction_summary"]
    assert summary["publication_allowed"] is False
    assert [source["rights_id"] for source in summary["sources"]] == [
        "nagisa-v3.1",
        "suisho11plus-wcsc36-20260525-local",
    ]
    assert len(summary["sidecar_sha256s"]) == 1
    assert len(summary["restriction_ids"]) == 1
    encoded_summary = json.dumps(summary, sort_keys=True)
    for private_token in (
        "/" + "Users",
        "/" + "private" + "/tmp",
        "example.invalid",
        "e" * 64,
        "f" * 64,
    ):
        assert private_token not in encoded_summary


def test_training_input_snapshot_rejects_sidecar_change_before_checkpoint(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "teacher.jsonl"
    _teacher_replay(replay, teacher_source="nagisa-v3.1")
    sidecar = replay.with_suffix(replay.suffix + ".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "rights": {
                    "rights_id": "nagisa-v3.1",
                    "output_only_meteo_publication": "allowed",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    expected = _training_input_provenance([replay])
    sidecar.write_text(
        json.dumps(
            {
                "publication_allowed": False,
                "rights": {
                    "rights_id": "nagisa-v3.1",
                    "output_only_meteo_publication": "allowed",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="changed during training"):
        _verify_training_input_provenance_unchanged([replay], expected)


def test_canonical_cli_requires_explicit_v2_reset_then_resumes_exactly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = tmp_path / "base"
    replay = tmp_path / "actor.jsonl"
    legacy = tmp_path / "legacy-v1"
    canonical = tmp_path / "canonical-v2"
    resumed = tmp_path / "canonical-v2-resumed"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), base, step=0)
    game = _replay(replay)

    assert (
        main(
            [
                "train",
                str(base),
                str(replay),
                str(legacy),
                "--steps",
                "1",
                "--batch-size",
                "1",
                "--learning-rate",
                "0.0001",
                "--allow-actor-targets",
                "--max-probe-loss-ratio",
                "100",
            ]
        )
        == 0
    )
    capsys.readouterr()
    sidecar = replay.with_suffix(replay.suffix + ".distillation-targets.json")
    _write_sidecar(sidecar, _payload(replay, game))

    shared = [
        str(replay),
        "--steps",
        "1",
        "--batch-size",
        "1",
        "--learning-rate",
        "0.0001",
        "--canonical-teacher-only",
        "--value-loss-weight",
        "0.25",
        "--best-union-top5-weight",
        "0.1",
        "--max-probe-loss-ratio",
        "100",
    ]
    with pytest.raises(ValueError, match="dataset fingerprint changed"):
        main(["train", str(legacy), shared[0], str(canonical), *shared[1:]])
    assert not canonical.exists()

    assert (
        main(
            [
                "train",
                str(legacy),
                shared[0],
                str(canonical),
                *shared[1:],
                "--reset-optimizer-state",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["canonical_teacher_only"] is True
    assert report["actor_policy_target_mass"] == 0.0
    assert report["game_outcome_value_contribution"] == 0.0
    assert report["value_loss_weight"] == 0.25
    assert report["exact_resume_from_parent"] is False
    assert report["optimizer_state_reset_requested"] is True

    metadata_text = (canonical / "metadata.json").read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)
    hyperparameters = metadata["lineage"]["hyperparameters"]
    assert hyperparameters["canonical_target_contract"]["scorer_ids"] == [
        "nagisa-v3.1",
        "suisho11plus-wcsc36-20260525-local",
    ]
    assert hyperparameters["canonical_target_contract"][
        "sidecar_path_or_raw_hash_recorded"
    ] is False
    assert str(sidecar.resolve()) not in metadata_text
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest() not in metadata_text

    assert (
        main(
            [
                "train",
                str(canonical),
                shared[0],
                str(resumed),
                *shared[1:],
            ]
        )
        == 0
    )
    resumed_report = json.loads(capsys.readouterr().out)
    assert resumed_report["exact_resume_from_parent"] is True
    _model, step, state, _trace = load_checkpoint_with_training_state(resumed)
    assert step == 3
    assert state is not None
    assert state.optimizer_step == 2
