from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from rsshogi.core import Board

from simajilord_shogi.arbitration import (
    ARBITRATION_REPORT_SCHEMA,
    OPPONENT_EXPLOIT_EVIDENCE_SCHEMA,
    ArbitrationConfig,
    DepthPassInput,
    build_depth_arbitration,
    write_depth_arbitration,
)
from simajilord_shogi.artifact_provenance import canonical_json_sha256
from simajilord_shogi.cli import main
from simajilord_shogi.domain import (
    GameRecord,
    PositionSample,
    TeacherScoreBound,
    TeacherScoreKind,
    TeacherVariation,
    Termination,
)
from simajilord_shogi.ensemble import normalized_sfen
from simajilord_shogi.replay import append_games, load_games, position_samples
from simajilord_shogi.tsume import TsumeSolver

MULTIPLE_MATE_IN_ONE_SFEN = (
    "3rkr3/3p1p3/9/6B2/9/9/9/9/4K4 b B4G4S4N4L16P 1"
)


def _base_game(sfen: str = Board().to_sfen()) -> GameRecord:
    first_legal = Board(sfen).legal_moves()[0].to_usi()
    return GameRecord(
        initial_sfen=sfen,
        moves=(first_legal,),
        samples=(
            PositionSample(
                sfen=sfen,
                ply=0,
                turn=Board(sfen).turn.value,
                policy={first_legal: 1.0},
                root_value=0.0,
                value_target=0.0,
                actor_best_move=first_legal,
            ),
        ),
        winner=None,
        termination=Termination.REPETITION,
    )


def _write_pass(
    path: Path,
    *,
    sfen: str,
    source: str,
    top_move: str,
    scores: dict[str, int] | None = None,
    score_rows: tuple[tuple[str, int], ...] | None = None,
    reported_mate_move: str | None = None,
    legacy_value: float = 0.99,
) -> None:
    variations: list[TeacherVariation] = []
    if reported_mate_move is not None:
        variations.append(
            TeacherVariation(
                rank=1,
                move=reported_mate_move,
                score_kind=TeacherScoreKind.MATE,
                bound=TeacherScoreBound.EXACT,
                pv=(reported_mate_move,),
                mate_plies=1,
            )
        )
    rows = score_rows if score_rows is not None else tuple((scores or {}).items())
    for rank, (move, score) in enumerate(rows, start=1):
        variations.append(
            TeacherVariation(
                rank=rank,
                move=move,
                score_kind=TeacherScoreKind.CENTIPAWN,
                bound=TeacherScoreBound.EXACT,
                pv=(move,),
                score_cp=score,
            )
        )
    sample = PositionSample(
        sfen=sfen,
        ply=0,
        turn=Board(sfen).turn.value,
        policy={top_move: 1.0},
        root_value=0.0,
        value_target=0.0,
        actor_best_move=top_move,
        teacher_policy={top_move: 1.0},
        teacher_value=legacy_value,
        teacher_best_move=top_move,
        teacher_source=source,
        teacher_depth=12,
        teacher_variations=tuple(variations),
    )
    append_games(
        path,
        [
            GameRecord(
                initial_sfen=sfen,
                moves=(top_move,),
                samples=(sample,),
                winner=None,
                termination=Termination.REPETITION,
            )
        ],
    )


def test_arbitration_keeps_family_union_stability_risks_and_separate_match_axis(
    tmp_path: Path,
) -> None:
    sfen = Board().to_sfen()
    move_a = "7g7f"
    move_b = "2g2f"
    base = tmp_path / "base.jsonl"
    append_games(base, [_base_game(sfen)])

    shallow = tmp_path / "hao-shallow.jsonl"
    deep = tmp_path / "tanuki-deep.jsonl"
    nagisa = tmp_path / "nagisa-deep.jsonl"
    _write_pass(
        shallow,
        sfen=sfen,
        source="hao",
        top_move=move_a,
        scores={move_a: 600, move_b: 500},
    )
    _write_pass(
        deep,
        sfen=sfen,
        source="tanuki-dr4",
        top_move=move_b,
        scores={move_a: -600, move_b: 300},
    )
    _write_pass(
        nagisa,
        sfen=sfen,
        source="nagisa",
        top_move=move_a,
        scores={move_a: 400},
    )
    evidence = tmp_path / "opponent-evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema": OPPONENT_EXPLOIT_EVIDENCE_SCHEMA,
                "records": [
                    {
                        "normalized_sfen": normalized_sfen(sfen),
                        "move": move_a,
                        "opponent_family": "unseen-family",
                        "expected_score_gain": 0.2,
                        "games": 80,
                        "validation_scope": "unknown_split",
                        "unknown_opponent_family": True,
                    },
                    {
                        "normalized_sfen": normalized_sfen(sfen),
                        "move": move_b,
                        "opponent_family": "NAGISA",
                        "expected_score_gain": 0.1,
                        "games": 80,
                        "validation_scope": "heldout_nagisa",
                        "unknown_opponent_family": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    passes = (
        DepthPassInput("hao-shallow", shallow, "tanuki", 1),
        DepthPassInput("tanuki-deep", deep, "tanuki", 2, "independent"),
        DepthPassInput("nagisa-deep", nagisa, "nagisa", 2, "unknown_split"),
    )

    build = build_depth_arbitration(
        base,
        passes,
        config=ArbitrationConfig(proof_max_plies=1),
        opponent_evidence=evidence,
    )
    repeated = build_depth_arbitration(
        base,
        passes,
        config=ArbitrationConfig(proof_max_plies=1),
        opponent_evidence=evidence,
    )

    assert build == repeated
    assert build.report.schema == ARBITRATION_REPORT_SCHEMA
    position = build.report.positions[0]
    assert position.unresolved_top1_disagreement
    assert position.top1_union == (move_b, move_a)
    assert position.effective_family_weights == {"nagisa": 0.5, "tanuki": 0.5}
    output_sample = position_samples(build.games, include_incomplete=True)[0]
    assert output_sample.teacher_policy == {move_b: 0.5, move_a: 0.5}
    assert output_sample.teacher_best_move is None
    expected_value = (math.tanh(300 / 1_200) + math.tanh(400 / 1_200)) / 2
    assert output_sample.teacher_value == pytest.approx(expected_value)
    assert output_sample.teacher_value != pytest.approx(0.99)
    assert "never uses legacy replay teacher_value" in build.report.value_transform_semantics

    risk_codes = {flag.code for flag in position.risk_flags}
    assert "shallow_deep_top1_reversal" in risk_codes
    assert "search_effort_move_collapse" in risk_codes
    assert "deep_exact_cp_drop" in risk_codes
    assert "family_unique_terminal_move" in risk_codes
    assert "cross_teacher_move_fragility" in risk_codes
    assert len(position.strategy_branches) == 2
    tanuki_story = next(
        branch for branch in position.strategy_branches if branch.family == "tanuki"
    )
    assert tanuki_story.top_move == move_b
    assert [story.declared_depth for story in tanuki_story.pass_stories] == [1, 2]
    assert tanuki_story.pass_stories[0].variations
    assert position.trajectory.opening_family.startswith("opening-fingerprint:")

    tradeoffs = {tradeoff.move: tradeoff for tradeoff in position.practical_tradeoffs}
    assert tradeoffs[move_a].objective_deep_regret_cp == pytest.approx(450.0)
    assert tradeoffs[move_a].opponent_exploit_gain == pytest.approx(0.2)
    assert tradeoffs[move_a].usage_classification == "general_practical_strength_candidate"
    assert tradeoffs[move_a].pareto_candidate
    assert tradeoffs[move_b].usage_classification == "targeted_match_strategy_candidate"
    assert tradeoffs[move_b].pareto_candidate
    assert all(tradeoff.core_training_target for tradeoff in tradeoffs.values())

    output = tmp_path / "arbitrated.jsonl"
    payload = write_depth_arbitration(build, output)
    report_path = output.with_suffix(output.suffix + ".arbitration.json")
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    recorded_hash = saved.pop("provenance_sha256")
    assert recorded_hash == canonical_json_sha256(saved)
    assert payload["provenance_sha256"] == recorded_hash
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_depth_arbitration(build, output)


def test_internal_all_mate_proof_overrides_nonmate_and_preserves_every_root(
    tmp_path: Path,
) -> None:
    board = Board(MULTIPLE_MATE_IN_ONE_SFEN)
    internally_proven = set(
        TsumeSolver(node_limit=10_000).solve_all(board, max_plies=1).first_moves
    )
    assert len(internally_proven) > 1
    nonmate = next(
        move.to_usi() for move in board.legal_moves() if move.to_usi() not in internally_proven
    )
    base = tmp_path / "mate-base.jsonl"
    append_games(base, [_base_game(MULTIPLE_MATE_IN_ONE_SFEN)])
    reported = tmp_path / "reported.jsonl"
    ordinary = tmp_path / "ordinary.jsonl"
    _write_pass(
        reported,
        sfen=MULTIPLE_MATE_IN_ONE_SFEN,
        source="usi-reporter",
        top_move=nonmate,
        reported_mate_move=nonmate,
    )
    _write_pass(
        ordinary,
        sfen=MULTIPLE_MATE_IN_ONE_SFEN,
        source="nonmate-teacher",
        top_move=nonmate,
        scores={nonmate: 1_000},
    )

    build = build_depth_arbitration(
        base,
        (
            DepthPassInput("reported", reported, "reporter", 1),
            DepthPassInput("ordinary", ordinary, "ordinary", 1),
        ),
        config=ArbitrationConfig(proof_max_plies=1, proof_node_limit=10_000),
    )

    position = build.report.positions[0]
    assert position.mate_proof.status == "proven"
    assert set(position.mate_proof.proven_moves) == internally_proven
    assert position.reported_winning_mates[0].move == nonmate
    assert nonmate not in position.mate_proof.proven_moves
    sample = position_samples(build.games, include_incomplete=True)[0]
    assert set(sample.teacher_policy or {}) == internally_proven
    assert all(
        probability == pytest.approx(1 / len(internally_proven))
        for probability in (sample.teacher_policy or {}).values()
    )
    assert sample.teacher_value == 1.0
    assert sample.teacher_best_move is None
    assert position.final_target_kind == "internally_proven_mate_uniform"


def test_missing_exact_cp_leaves_legacy_value_unset(tmp_path: Path) -> None:
    sfen = Board().to_sfen()
    base = tmp_path / "base.jsonl"
    append_games(base, [_base_game(sfen)])
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_pass(first, sfen=sfen, source="first", top_move="7g7f", scores=None)
    _write_pass(second, sfen=sfen, source="second", top_move="7g7f", scores=None)

    build = build_depth_arbitration(
        base,
        (
            DepthPassInput("first", first, "first", 1),
            DepthPassInput("second", second, "second", 1),
        ),
        config=ArbitrationConfig(proof_max_plies=1),
    )

    sample = position_samples(build.games, include_incomplete=True)[0]
    assert sample.teacher_policy == {"7g7f": 1.0}
    assert sample.teacher_value is None


def test_phase_family_priors_stay_soft_and_distill_to_one_target(tmp_path: Path) -> None:
    sfen = Board().to_sfen()
    base = tmp_path / "base.jsonl"
    append_games(base, [_base_game(sfen)])
    dl = tmp_path / "dl.jsonl"
    nnue = tmp_path / "nnue.jsonl"
    _write_pass(dl, sfen=sfen, source="dl", top_move="7g7f", scores={"7g7f": 200})
    _write_pass(nnue, sfen=sfen, source="nnue", top_move="2g2f", scores={"2g2f": 100})

    build = build_depth_arbitration(
        base,
        (
            DepthPassInput("dl", dl, "dl-family", 100),
            DepthPassInput("nnue", nnue, "nnue-family", 100),
        ),
        config=ArbitrationConfig(
            proof_max_plies=1,
            phase_family_weights={
                "opening_plies_1_24": {"dl-family": 3.0, "nnue-family": 1.0}
            },
        ),
    )

    position = build.report.positions[0]
    assert position.trajectory.phase == "opening_plies_1_24"
    assert position.effective_family_weights == {
        "dl-family": pytest.approx(0.75),
        "nnue-family": pytest.approx(0.25),
    }
    sample = position_samples(build.games, include_incomplete=True)[0]
    assert sample.teacher_source == "meteo-adaptive-deep-arbitration"
    assert sample.teacher_policy == {
        "2g2f": pytest.approx(0.25),
        "7g7f": pytest.approx(0.75),
    }
    assert set(sample.teacher_policy or {}) == {"2g2f", "7g7f"}
    assert "one Meteo model" in build.report.family_aggregation


def test_phase_family_prior_rejects_unknown_phase_and_family(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown phase"):
        ArbitrationConfig(phase_family_weights={"opening": {"dl": 1.0}})

    sfen = Board().to_sfen()
    base = tmp_path / "base.jsonl"
    append_games(base, [_base_game(sfen)])
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_pass(first, sfen=sfen, source="first", top_move="7g7f")
    _write_pass(second, sfen=sfen, source="second", top_move="2g2f")
    with pytest.raises(ValueError, match="unavailable families"):
        build_depth_arbitration(
            base,
            (
                DepthPassInput("first", first, "first", 1),
                DepthPassInput("second", second, "second", 1),
            ),
            config=ArbitrationConfig(
                proof_max_plies=1,
                phase_family_weights={
                    "opening_plies_1_24": {"misspelled-family": 2.0}
                },
            ),
        )


def test_arbitration_cli_writes_create_only_output(tmp_path: Path) -> None:
    sfen = Board().to_sfen()
    base = tmp_path / "base.jsonl"
    append_games(base, [_base_game(sfen)])
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "output.jsonl"
    _write_pass(first, sfen=sfen, source="first", top_move="7g7f", scores={"7g7f": 100})
    _write_pass(second, sfen=sfen, source="second", top_move="2g2f", scores={"2g2f": 200})

    status = main(
        [
            "arbitrate-depth-passes",
            str(base),
            str(output),
            "--pass",
            f"first={first}",
            "--pass",
            f"second={second}",
            "--pass-family",
            "first=family-a",
            "--pass-family",
            "second=family-b",
            "--pass-depth",
            "first=100",
            "--pass-depth",
            "second=200",
            "--proof-plies",
            "1",
            "--phase-family-weight",
            "opening_plies_1_24:family-a=3",
        ]
    )

    assert status == 0
    assert output.is_file()
    assert output.with_suffix(output.suffix + ".arbitration.json").is_file()
    output_sample = position_samples(tuple(load_games(output)), include_incomplete=True)[0]
    assert output_sample.teacher_policy == {
        "2g2f": pytest.approx(0.25),
        "7g7f": pytest.approx(0.75),
    }


def test_repeated_multipv_root_uses_smallest_rank_without_losing_pvs(
    tmp_path: Path,
) -> None:
    sfen = Board().to_sfen()
    base = tmp_path / "base.jsonl"
    append_games(base, [_base_game(sfen)])
    repeated = tmp_path / "repeated.jsonl"
    independent = tmp_path / "independent.jsonl"
    _write_pass(
        repeated,
        sfen=sfen,
        source="repeated",
        top_move="7g7f",
        score_rows=(("7g7f", 100), ("7g7f", 900)),
    )
    _write_pass(
        independent,
        sfen=sfen,
        source="independent",
        top_move="7g7f",
        scores={"7g7f": 100},
    )

    build = build_depth_arbitration(
        base,
        (
            DepthPassInput("repeated", repeated, "repeated", 1),
            DepthPassInput("independent", independent, "independent", 1),
        ),
        config=ArbitrationConfig(proof_max_plies=1),
    )

    sample = position_samples(build.games, include_incomplete=True)[0]
    assert sample.teacher_value == pytest.approx(math.tanh(100 / 1_200))
    branch = next(
        item for item in build.report.positions[0].strategy_branches if item.family == "repeated"
    )
    assert len(branch.pass_stories[0].variations) == 2
