from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from simajilord_shogi.cli import (
    _artifact_provenance,
    _depth_pass_inputs,
    _disagreement_teacher_inputs,
    _engine_options,
    _require_limited_local_destinations,
    _require_suisho11plus_teacher_options,
    _resolve_ponanza_value_scale,
    build_parser,
)
from simajilord_shogi.trainer import TrainingInterlockConfig


def test_engine_option_names_cannot_collide_case_insensitively() -> None:
    with pytest.raises(ValueError, match="duplicate engine option"):
        _engine_options(["Threads=2", "threads=4"])


def test_artifact_provenance_hashes_once_and_rejects_duplicate_paths(tmp_path: Path) -> None:
    artifact = tmp_path / "nn.bin"
    artifact.write_bytes(b"reviewed teacher artifact")

    records = _artifact_provenance([artifact])

    assert records == [
        {
            "path": str(artifact.resolve()),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "bytes": artifact.stat().st_size,
        }
    ]
    with pytest.raises(ValueError, match="duplicate teacher artifact"):
        _artifact_provenance([artifact, artifact])


def test_benchmark_uses_a_reviewed_rights_profile() -> None:
    args = build_parser().parse_args(
        [
            "benchmark-usi",
            "checkpoint",
            "output",
            "--engine",
            "engine",
            "--rights-profile",
            "gikou2-v2.0.2",
            "--legacy-single-opening-debug",
        ]
    )

    assert args.rights_profile == "gikou2-v2.0.2"


def test_external_reanalysis_exposes_teacher_specific_score_calibration() -> None:
    args = build_parser().parse_args(
        [
            "reanalyse-usi",
            "replay.jsonl",
            "teacher.jsonl",
            "--engine",
            "engine",
            "--rights-profile",
            "nagisa-v3.1",
            "--teacher-policy-temperature",
            "175",
            "--teacher-value-scale",
            "250",
        ]
    )

    assert args.teacher_policy_temperature == 175.0
    assert args.teacher_value_scale == 250.0


def test_external_reanalysis_exposes_suisho11plus_local_only_gate() -> None:
    args = build_parser().parse_args(
        [
            "reanalyse-usi",
            "replay.jsonl",
            "teacher.jsonl",
            "--engine",
            "engine",
            "--rights-profile",
            "suisho11plus-wcsc36-20260525-local",
            "--local-only-user-authorized",
            "--local-only-root",
            "/tmp/meteo-local-only-fixture",
        ]
    )

    assert args.local_only_user_authorized
    assert args.local_only_root == Path("/tmp/meteo-local-only-fixture")


def test_suisho11plus_options_require_no_book_fv40_and_yaneuraou_hash(
    tmp_path: Path,
) -> None:
    eval_directory = tmp_path / "eval"
    eval_directory.mkdir()
    options = {
        "EvalDir": str(eval_directory),
        "FV_SCALE": "40",
        "Threads": "1",
        "USI_Hash": "64",
        "USI_OwnBook": "false",
        "BookFile": "no_book",
        "PvInterval": "0",
    }

    _require_suisho11plus_teacher_options(options, multipv=8)
    with pytest.raises(ValueError, match=r"FV_SCALE|fv_scale"):
        _require_suisho11plus_teacher_options({**options, "FV_SCALE": "16"}, multipv=8)
    with pytest.raises(ValueError, match=r"USI_Hash|usi_hash"):
        without_usi_hash = {name: value for name, value in options.items() if name != "USI_Hash"}
        _require_suisho11plus_teacher_options({**without_usi_hash, "Hash": "64"}, multipv=8)
    with pytest.raises(ValueError, match="MultiPV"):
        _require_suisho11plus_teacher_options(options, multipv=4)


def test_limited_local_outputs_must_stay_below_explicit_root(tmp_path: Path) -> None:
    local_root = tmp_path / "private"
    local_root.mkdir()
    output = local_root / "teacher.jsonl"
    provenance = local_root / "teacher.jsonl.provenance.json"

    assert (
        _require_limited_local_destinations(local_root, (output, provenance))
        == local_root.resolve()
    )
    with pytest.raises(ValueError, match="child of local-only root"):
        _require_limited_local_destinations(local_root, (tmp_path / "escaped.jsonl",))


def test_ponanza_coefficient_resolution_preserves_explicit_legacy_denominators() -> None:
    coefficient, denominator, convention = _resolve_ponanza_value_scale(
        ponanza_coefficient=None,
        legacy_tanh_denominator=None,
    )
    assert coefficient == 600.0
    assert denominator == 1_200.0
    assert convention == "default_ponanza_coefficient"

    coefficient, denominator, convention = _resolve_ponanza_value_scale(
        ponanza_coefficient=756.0864962951762,
        legacy_tanh_denominator=None,
    )
    assert coefficient == 756.0864962951762
    assert denominator == pytest.approx(1_512.1729925903524)
    assert convention == "explicit_ponanza_coefficient"

    coefficient, denominator, convention = _resolve_ponanza_value_scale(
        ponanza_coefficient=None,
        legacy_tanh_denominator=600.0,
    )
    assert coefficient == 300.0
    assert denominator == 600.0
    assert convention == "explicit_legacy_tanh_denominator"

    with pytest.raises(ValueError, match="mutually exclusive"):
        _resolve_ponanza_value_scale(
            ponanza_coefficient=600.0,
            legacy_tanh_denominator=1_200.0,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "reanalyse-usi",
            "replay.jsonl",
            "teacher.jsonl",
            "--engine",
            "engine",
            "--rights-profile",
            "nagisa-v3.1",
            "--teacher-ponanza-coefficient",
            "600",
            "--teacher-value-scale",
            "1200",
        ],
        [
            "train-psv",
            "checkpoint",
            "teacher.psv",
            "output",
            "--source-name",
            "fixture",
            "--score-ponanza-coefficient",
            "600",
            "--score-scale",
            "1200",
        ],
        [
            "fit-teacher-value-scale",
            "teacher.jsonl",
            "--output",
            "fit.json",
            "--baseline-ponanza-coefficient",
            "600",
            "--baseline-scale",
            "1200",
        ],
    ],
)
def test_value_coefficient_and_legacy_denominator_flags_are_mutually_exclusive(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)


def test_nps_benchmark_estimates_one_hundred_million_nodes_per_root() -> None:
    args = build_parser().parse_args(["bench-nps"])

    assert args.target_nodes_per_root == 100_000_000


@pytest.mark.parametrize(
    "arguments",
    [
        ["selfplay", "checkpoint", "replay.jsonl"],
        ["train", "checkpoint", "replay.jsonl", "output"],
        ["train-psv", "checkpoint", "teacher.psv", "output", "--source-name", "x"],
        ["reanalyse", "checkpoint", "replay.jsonl", "output"],
        ["improve", "checkpoint", "workdir"],
        [
            "benchmark-usi",
            "checkpoint",
            "output",
            "--engine",
            "engine",
            "--rights-profile",
            "gikou2-v2.0.2",
            "--legacy-single-opening-debug",
        ],
    ],
)
def test_training_commands_expose_one_shared_human_play_interlock(
    arguments: list[str], tmp_path: Path
) -> None:
    state_root = tmp_path / "state"
    args = build_parser().parse_args(
        [
            *arguments,
            "--human-play-state-root",
            str(state_root),
            "--human-play-wait-timeout-seconds",
            "12",
            "--human-play-poll-interval-seconds",
            "0.1",
            "--training-lease-ttl-seconds",
            "30",
            "--training-lease-heartbeat-seconds",
            "2",
        ]
    )

    assert args.human_play_state_root == state_root
    assert args.human_play_wait_timeout_seconds == 12
    assert args.human_play_poll_interval_seconds == 0.1
    assert args.training_lease_ttl_seconds == 30
    assert args.training_lease_heartbeat_seconds == 2


def test_training_interlock_public_metadata_never_records_private_path(
    tmp_path: Path,
) -> None:
    private_token = "do-not-publish-human-state"
    config = TrainingInterlockConfig(state_root=tmp_path / private_token)

    assert private_token not in str(config.public_metadata())
    assert config.public_metadata()["state_root_recorded"] is False


def test_train_parser_exposes_fail_closed_canonical_loss_controls() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "checkpoint",
            "replay.jsonl",
            "output",
            "--canonical-teacher-only",
            "--canonical-target-sidecar",
            "targets.json",
            "--value-loss-weight",
            "0.25",
            "--best-union-top5-weight",
            "0.5",
            "--reset-optimizer-state",
        ]
    )

    assert args.canonical_teacher_only is True
    assert args.canonical_target_sidecar == Path("targets.json")
    assert args.value_loss_weight == 0.25
    assert args.best_union_top5_weight == 0.5
    assert args.reset_optimizer_state is True


def test_teacher_dedup_cli_requires_named_split_and_exposes_overlap_guards() -> None:
    args = build_parser().parse_args(
        [
            "deduplicate-teacher-replay",
            "raw.jsonl",
            "train.jsonl",
            "--split",
            "train",
            "--forbid-overlap-with",
            "validation.jsonl",
            "--forbid-overlap-with",
            "sealed.jsonl",
        ]
    )

    assert args.split == "train"
    assert args.forbid_overlap_with == [Path("validation.jsonl"), Path("sealed.jsonl")]


def test_floodgate_corpus_cli_requires_a_byte_pinned_rating_source() -> None:
    args = build_parser().parse_args(
        [
            "prepare-floodgate-corpus",
            "wdoor2026.7z",
            "players-floodgate14-20260809.html",
            "corpus",
            "--rating-url",
            "https://wdoor.c.u-tokyo.ac.jp/shogi/x/rating/players-floodgate14-20260809.html",
            "--min-rating",
            "4000",
            "--min-games",
            "20",
        ]
    )

    assert args.min_rating == 4000
    assert args.min_games == 20
    assert args.limit_members is None
    assert args.maximum_csa_bytes == 2_000_000
    assert args.extraction_batch_bytes == 768 * 1024 * 1024
    assert args.extraction_batch_members == 50_000
    assert args.extraction_timeout_seconds == 600


def test_ybb_plan_cli_requires_local_authorization_and_multiple_teacher_ids() -> None:
    args = build_parser().parse_args(
        [
            "prepare-ybb-reanalysis-plan",
            "book.ybb",
            "plan.json",
            "--source-archive",
            "book.7z",
            "--source-url",
            "https://storage.example/book.7z",
            "--teacher-id",
            "nagisa-v3.1",
            "--teacher-id",
            "suisho11plus-wcsc36-20260525",
            "--local-only-user-authorized",
        ]
    )

    assert args.teacher_id == ["nagisa-v3.1", "suisho11plus-wcsc36-20260525"]
    assert args.local_only_user_authorized is True
    assert args.multipv == 8


def test_deep_disagreement_cli_requires_explicit_correlation_families() -> None:
    inputs = _disagreement_teacher_inputs(
        ["hao=hao.jsonl", "tanuki=tanuki.jsonl"],
        ["hao=tanuki-family", "tanuki=tanuki-family"],
    )

    assert [item.family for item in inputs] == ["tanuki-family", "tanuki-family"]
    with pytest.raises(ValueError, match="exactly match"):
        _disagreement_teacher_inputs(
            ["hao=hao.jsonl", "tanuki=tanuki.jsonl"],
            ["hao=tanuki-family"],
        )


def test_depth_pass_cli_keeps_depth_family_and_validation_scope_separate() -> None:
    passes = _depth_pass_inputs(
        ["shallow=shallow.jsonl", "deep=deep.jsonl"],
        ["shallow=tanuki", "deep=tanuki"],
        ["shallow=20000", "deep=2000000"],
        ["deep=unknown_split"],
    )

    assert [item.depth for item in passes] == [20_000, 2_000_000]
    assert [item.family for item in passes] == ["tanuki", "tanuki"]
    assert [item.validation_scope for item in passes] == ["in_domain", "unknown_split"]


def test_arbitration_parser_exposes_exact_proof_and_posthoc_value_semantics() -> None:
    args = build_parser().parse_args(
        [
            "arbitrate-depth-passes",
            "base.jsonl",
            "output.jsonl",
            "--pass",
            "shallow=shallow.jsonl",
            "--pass",
            "deep=deep.jsonl",
            "--pass-family",
            "shallow=tanuki",
            "--pass-family",
            "deep=tanuki",
            "--pass-depth",
            "shallow=20000",
            "--pass-depth",
            "deep=2000000",
            "--proof-plies",
            "5",
            "--centipawn-value-scale",
            "1512.173",
        ]
    )

    assert args.proof_plies == 5
    assert args.centipawn_value_scale == 1512.173
