from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest
from rsshogi.core import Board

from simajilord_shogi.improvement_loop import (
    BenchmarkEvidence,
    BoundedExplorationConfig,
    CandidateArtifact,
    ContinuousLoopConfig,
    DatasetSplit,
    ExplorationCandidate,
    GameArtifact,
    GenerationStage,
    OpponentRole,
    Participant,
    PromotionDecision,
    TeacherSource,
    append_loop_state,
    append_stage_receipt,
    build_generation_plan,
    build_loop_state,
    choose_bounded_exploration_move,
    deterministic_retry_seed,
    evaluate_promotion,
    freeze_dataset_growth,
    load_latest_loop_state,
    trajectory_sha256,
    validate_stage_chain,
    write_generation_plan,
)
from simajilord_shogi.opening_suite import OpeningPosition


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _openings() -> tuple[str, tuple[str, ...]]:
    """Return flat start plus 500 distinct, legal, non-flat positions."""

    flat = Board().to_sfen()
    queue = deque([Board()])
    seen = {OpeningPosition.from_sfen(flat).normalized_key}
    evaluation: list[str] = []
    while queue and len(evaluation) < 500:
        board = queue.popleft()
        for move in board.legal_moves():
            child = board.copy()
            child.apply_move(move)
            position = OpeningPosition.from_sfen(child.to_sfen())
            if position.normalized_key in seen:
                continue
            seen.add(position.normalized_key)
            evaluation.append(position.sfen)
            queue.append(child)
            if len(evaluation) == 500:
                break
    assert len(evaluation) == 500
    return flat, tuple(evaluation)


def _participants() -> tuple[Participant, ...]:
    return (
        Participant("champion-v3", OpponentRole.CHAMPION, _digest("champion-v3")),
        Participant("meteo-v1", OpponentRole.HISTORICAL_METEO, _digest("meteo-v1")),
        Participant("nagisa", OpponentRole.EXTERNAL_TEACHER, _digest("nagisa")),
        Participant("takewarabe", OpponentRole.WEAK, _digest("takewarabe")),
        Participant("uniform-random", OpponentRole.RANDOM, _digest("uniform-random")),
    )


def _teachers() -> tuple[TeacherSource, ...]:
    return (
        TeacherSource(
            "nagisa-multipv",
            "nagisa",
            _digest("nagisa-eval"),
            nodes=1_000_000,
            multipv=8,
            training_outputs_allowed=True,
        ),
        TeacherSource(
            "suisho-multipv",
            "suisho",
            _digest("suisho-eval"),
            nodes=1_000_000,
            multipv=8,
            training_outputs_allowed=True,
        ),
    )


def _plan() -> object:
    flat, evaluation = _openings()
    participants = _participants()
    return build_generation_plan(
        generation=1,
        lineage_id="meteo-continuous-test",
        dataset_parent_sha256=_digest("dataset-v0"),
        champion=participants[0],
        participants=participants,
        teachers=_teachers(),
        training_opening_sfens=(flat,),
        evaluation_opening_sfens=evaluation,
        config=ContinuousLoopConfig(
            training_games_per_generation=10,
            bootstrap_iterations=100,
        ),
        seed=42,
    )


def _game_digests(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(_digest(f"{prefix}-{index}") for index in range(count))


def _passing_evidence(plan: object) -> tuple[BenchmarkEvidence, ...]:
    keys = tuple(opening.normalized_key for opening in plan.evaluation_openings)
    wins = (1.0,) * len(keys)
    reports: list[BenchmarkEvidence] = []
    for participant in plan.participants:
        reference = participant.role is OpponentRole.EXTERNAL_TEACHER
        reports.append(
            BenchmarkEvidence(
                opponent_id=participant.participant_id,
                opponent_role=participant.role,
                opening_keys=keys,
                candidate_black_points=wins,
                candidate_white_points=wins,
                game_sha256=_game_digests(participant.participant_id, 2 * len(keys)),
                reference_black_points=wins if reference else None,
                reference_white_points=wins if reference else None,
                reference_game_sha256=(
                    _game_digests(f"reference-{participant.participant_id}", 2 * len(keys))
                    if reference
                    else None
                ),
            )
        )
    return tuple(reports)


def _training_entries(plan: object) -> tuple[GameArtifact, ...]:
    teacher_ids = tuple(teacher.teacher_id for teacher in plan.teachers)
    return tuple(
        GameArtifact(
            job_id=job.job_id,
            trajectory_sha256=_digest(f"trajectory-{index}"),
            artifact_sha256=_digest(f"artifact-{index}"),
            split=DatasetSplit.TRAIN,
            opening_key=job.opening_key,
            producer_id=job.subject_id,
            execution_seed=job.seed,
            retry=0,
            complete=True,
            eligible_for_training=True,
            teacher_ids=teacher_ids,
        )
        for index, job in enumerate(plan.training_jobs[:2])
    )


def test_generation_plan_is_teacher_assisted_balanced_and_uses_1000_game_gates() -> None:
    plan = _plan()

    flat_key = OpeningPosition.from_sfen(Board().to_sfen()).normalized_key
    assert [job.opening_key for job in plan.training_jobs[:2]] == [flat_key, flat_key]
    assert [job.subject_color for job in plan.training_jobs[:2]] == ["black", "white"]
    assert all(job.exploration_enabled for job in plan.training_jobs)
    assert all(job.temperature > 0 for job in plan.training_jobs)
    assert len(plan.evaluation_openings) == 500
    for participant in plan.participants:
        jobs = [
            job for job in plan.evaluation_jobs if job.opponent_id == participant.participant_id
        ]
        assert len(jobs) == 1000
        assert sum(job.subject_color == "black" for job in jobs) == 500
        assert sum(job.subject_color == "white" for job in jobs) == 500
        assert all(not job.exploration_enabled for job in jobs)
        assert all(job.temperature == 0 and job.dirichlet_fraction == 0 for job in jobs)
    assert len(plan.reference_evaluation_jobs) == 1000
    all_jobs = (*plan.training_jobs, *plan.evaluation_jobs, *plan.reference_evaluation_jobs)
    assert len({job.job_id for job in all_jobs}) == len(all_jobs)
    assert len({job.seed for job in all_jobs}) == len(all_jobs)


def test_plan_requires_every_role_flat_start_and_500_unique_heldout_pairs() -> None:
    flat, evaluation = _openings()
    participants = _participants()
    common = {
        "generation": 1,
        "lineage_id": "strict-plan",
        "dataset_parent_sha256": _digest("parent"),
        "champion": participants[0],
        "teachers": _teachers(),
        "config": ContinuousLoopConfig(training_games_per_generation=2, bootstrap_iterations=100),
    }

    with pytest.raises(ValueError, match="missing required opponent roles"):
        build_generation_plan(
            **common,
            participants=participants[:-1],
            training_opening_sfens=(flat,),
            evaluation_opening_sfens=evaluation,
        )
    with pytest.raises(ValueError, match="flat starting position"):
        build_generation_plan(
            **common,
            participants=participants,
            training_opening_sfens=(evaluation[0],),
            evaluation_opening_sfens=(*evaluation[1:], flat),
        )
    with pytest.raises(ValueError, match="at least 500 unique openings"):
        build_generation_plan(
            **common,
            participants=participants,
            training_opening_sfens=(flat,),
            evaluation_opening_sfens=evaluation[:499],
        )
    with pytest.raises(ValueError, match="opening splits overlap"):
        build_generation_plan(
            **common,
            participants=participants,
            training_opening_sfens=(flat, evaluation[0]),
            evaluation_opening_sfens=evaluation,
        )


def test_next_plan_requires_only_the_retained_previous_weight_not_every_ancestor() -> None:
    flat, evaluation = _openings()
    participants = _participants()
    previous = {
        "schema": "meteo-continuous-improvement-state-v2",
        "generation": 1,
        "champion_after": {
            "candidate_id": "champion-v3",
            "artifact_sha256": _digest("champion-v3"),
        },
        "model_history": [
            {
                "candidate_id": "old-pruned-generation",
                "artifact_sha256": _digest("old-pruned-generation"),
                "weights_retained": False,
            },
            {
                "candidate_id": "meteo-generation-1",
                "artifact_sha256": _digest("meteo-generation-1"),
                "weights_retained": True,
            },
        ],
        "playable_model_window": [
            {
                "candidate_id": "champion-v3",
                "artifact_sha256": _digest("champion-v3"),
            },
            {
                "candidate_id": "meteo-generation-1",
                "artifact_sha256": _digest("meteo-generation-1"),
            },
        ],
    }
    arguments = {
        "generation": 2,
        "lineage_id": "history-complete",
        "dataset_parent_sha256": _digest("dataset-generation-1"),
        "champion": participants[0],
        "teachers": _teachers(),
        "training_opening_sfens": (flat,),
        "evaluation_opening_sfens": evaluation,
        "config": ContinuousLoopConfig(
            training_games_per_generation=2,
            bootstrap_iterations=100,
        ),
        "previous_state": previous,
    }

    with pytest.raises(ValueError, match="exactly match the retained playable window"):
        build_generation_plan(**arguments, participants=participants)

    historical = Participant(
        "meteo-generation-1",
        OpponentRole.HISTORICAL_METEO,
        _digest("meteo-generation-1"),
    )
    plan = build_generation_plan(
        **arguments,
        participants=(participants[0], historical, *participants[2:]),
    )
    assert plan.previous_state_sha256 is not None
    assert plan.required_historical_artifact_sha256 == (_digest("meteo-generation-1"),)
    assert _digest("old-pruned-generation") not in plan.required_historical_artifact_sha256

    stale_historical = Participant(
        "old-pruned-generation",
        OpponentRole.HISTORICAL_METEO,
        _digest("old-pruned-generation"),
    )
    with pytest.raises(ValueError, match="at most one retained previous"):
        build_generation_plan(
            **arguments,
            participants=(participants[0], historical, stale_historical, *participants[2:]),
        )


def test_temperature_can_only_choose_deep_bounded_regret_opening_alternatives() -> None:
    config = BoundedExplorationConfig(
        max_ply_exclusive=12,
        temperature=1.0,
        dirichlet_fraction=0.10,
        max_regret_upper_bound=0.02,
        minimum_nodes=1000,
        minimum_multipv=3,
        minimum_teacher_agreement=2,
    )
    candidates = (
        ExplorationCandidate("7g7f", 1, 1.0, 0.0, 10_000, 3, 2),
        ExplorationCandidate("2g2f", 2, 100.0, 0.01, 10_000, 3, 2),
        ExplorationCandidate("9g9f", 3, 1000.0, 0.20, 10_000, 3, 2),
    )

    decisions = [
        choose_bounded_exploration_move(candidates, ply=3, config=config, seed=seed)
        for seed in range(20)
    ]
    assert {decision.move for decision in decisions} <= {"7g7f", "2g2f"}
    assert any(decision.move == "2g2f" for decision in decisions)
    assert all("9g9f" not in decision.eligible_moves for decision in decisions)
    assert all(decision.regret_upper_bound <= 0.02 for decision in decisions)
    assert (
        choose_bounded_exploration_move(
            candidates, ply=3, config=config, seed=0, evaluation=True
        ).move
        == "7g7f"
    )
    assert (
        choose_bounded_exploration_move(candidates, ply=12, config=config, seed=0).reason
        == "after_opening_window"
    )

    mate_candidates = (
        replace(candidates[0], proves_winning_mate=True),
        candidates[1],
    )
    mate = choose_bounded_exploration_move(mate_candidates, ply=1, config=config, seed=4)
    assert mate.move == "7g7f"
    assert mate.reason == "no_safe_alternative"


def test_dataset_growth_deduplicates_trajectories_and_seals_arena_games() -> None:
    plan = _plan()
    train = _training_entries(plan)
    arena_job = plan.evaluation_jobs[0]
    arena = GameArtifact(
        job_id=arena_job.job_id,
        trajectory_sha256=_digest("arena-trajectory"),
        artifact_sha256=_digest("arena-artifact"),
        split=DatasetSplit.ARENA,
        opening_key=arena_job.opening_key,
        producer_id="candidate",
        execution_seed=arena_job.seed,
        retry=0,
        complete=True,
        eligible_for_training=False,
    )

    dataset = freeze_dataset_growth(plan, (*train, arena), parent_train_trajectories=())
    assert dataset.payload["new_unique_train_games"] == 2
    assert dataset.payload["split_counts"] == {
        "train": 2,
        "validation": 0,
        "sealed_test": 0,
        "arena": 1,
    }
    assert not dataset.payload["arena_training_allowed"]

    duplicate = replace(train[1], trajectory_sha256=train[0].trajectory_sha256)
    with pytest.raises(ValueError, match="duplicate trajectories"):
        freeze_dataset_growth(plan, (train[0], duplicate), parent_train_trajectories=())
    with pytest.raises(ValueError, match="duplicate the parent dataset"):
        freeze_dataset_growth(
            plan,
            train,
            parent_train_trajectories=(train[0].trajectory_sha256,),
        )
    with pytest.raises(ValueError, match="permanently ineligible"):
        freeze_dataset_growth(
            plan,
            (*train, replace(arena, eligible_for_training=True)),
            parent_train_trajectories=(),
        )
    with pytest.raises(ValueError, match="execution seed"):
        freeze_dataset_growth(
            plan,
            (replace(train[0], execution_seed=train[0].execution_seed + 1), train[1]),
            parent_train_trajectories=(),
        )


def test_role_gates_require_both_colors_and_do_not_hide_teacher_regression() -> None:
    plan = _plan()
    reports = list(_passing_evidence(plan))
    decision = evaluate_promotion(plan, reports)
    assert decision.promoted
    assert all(result.games == 1000 for result in decision.opponent_results)
    teacher_result = next(
        result
        for result in decision.opponent_results
        if result.role is OpponentRole.EXTERNAL_TEACHER
    )
    assert teacher_result.external_target_beaten
    assert teacher_result.reference_delta_black is not None
    assert teacher_result.reference_delta_black.score == 0.0

    champion_index = next(
        index
        for index, report in enumerate(reports)
        if report.opponent_role is OpponentRole.CHAMPION
    )
    reports[champion_index] = replace(
        reports[champion_index],
        candidate_white_points=(0.5,) * 500,
    )
    unstable = evaluate_promotion(plan, reports)
    assert not unstable.promoted
    assert "champion-v3:white_superiority_not_proven" in unstable.blockers

    reports = list(_passing_evidence(plan))
    teacher_index = next(
        index
        for index, report in enumerate(reports)
        if report.opponent_role is OpponentRole.EXTERNAL_TEACHER
    )
    reports[teacher_index] = replace(
        reports[teacher_index],
        candidate_black_points=(0.0,) * 500,
    )
    regression = evaluate_promotion(plan, reports)
    assert not regression.promoted
    assert "nagisa:black_external_teacher_regression" in regression.blockers


def test_external_target_can_require_a_version_specific_confidence_reserve() -> None:
    base = _plan()
    reserved_teacher = replace(
        next(
            participant
            for participant in base.participants
            if participant.participant_id == "nagisa"
        ),
        minimum_score_lower_95=0.55,
    )
    plan = replace(
        base,
        participants=tuple(
            reserved_teacher if participant.participant_id == "nagisa" else participant
            for participant in base.participants
        ),
    )
    reports = list(_passing_evidence(plan))
    teacher_index = next(
        index for index, report in enumerate(reports) if report.opponent_id == "nagisa"
    )
    proxy_score = (1.0,) * 50 + (0.5,) * 450
    reports[teacher_index] = replace(
        reports[teacher_index],
        candidate_black_points=proxy_score,
        candidate_white_points=proxy_score,
        reference_black_points=proxy_score,
        reference_white_points=proxy_score,
    )

    decision = evaluate_promotion(plan, reports)
    assert not decision.promoted
    assert "nagisa:black_external_target_superiority_not_proven" in decision.blockers
    assert "nagisa:white_external_target_superiority_not_proven" in decision.blockers
    assert "nagisa:paired_external_target_superiority_not_proven" in decision.blockers
    manifest_participant = next(
        participant
        for participant in plan.to_manifest()["participants"]
        if participant["participant_id"] == "nagisa"
    )
    assert manifest_participant["minimum_score_lower_95"] == 0.55

    with pytest.raises(ValueError, match="valid only for an external teacher"):
        Participant(
            "not-external",
            OpponentRole.CHAMPION,
            _digest("not-external"),
            minimum_score_lower_95=0.55,
        )


def test_random_gate_means_every_formal_game_and_game_artifacts_are_unique() -> None:
    plan = _plan()
    reports = list(_passing_evidence(plan))
    random_index = next(
        index for index, report in enumerate(reports) if report.opponent_role is OpponentRole.RANDOM
    )
    one_loss = list(reports[random_index].candidate_white_points)
    one_loss[0] = 0.0
    reports[random_index] = replace(
        reports[random_index],
        candidate_white_points=tuple(one_loss),
    )
    decision = evaluate_promotion(plan, reports)
    assert not decision.promoted
    assert "uniform-random:white_random_floor_missed" in decision.blockers

    reports = list(_passing_evidence(plan))
    reports[1] = replace(reports[1], game_sha256=reports[0].game_sha256)
    with pytest.raises(ValueError, match="reused across opponents"):
        evaluate_promotion(plan, reports)


def test_append_only_generation_receipts_reject_reordering_and_mutation(tmp_path: Path) -> None:
    plan = _plan()
    generation_dir = tmp_path / "generation-000001"
    plan_digest = write_generation_plan(generation_dir, plan)
    assert write_generation_plan(generation_dir, plan) == plan_digest
    with pytest.raises(FileExistsError, match="different create-only artifact"):
        changed = replace(plan, seed=plan.seed + 1)
        write_generation_plan(generation_dir, changed)

    first_digest = append_stage_receipt(
        generation_dir,
        GenerationStage.GAMES_COLLECTED,
        {"unique_trajectories": 10},
    )
    assert (
        append_stage_receipt(
            generation_dir,
            GenerationStage.GAMES_COLLECTED,
            {"unique_trajectories": 10},
        )
        == first_digest
    )
    with pytest.raises(FileExistsError, match="different create-only artifact"):
        append_stage_receipt(
            generation_dir,
            GenerationStage.GAMES_COLLECTED,
            {"unique_trajectories": 9},
        )
    with pytest.raises(ValueError, match="next stage"):
        append_stage_receipt(
            generation_dir,
            GenerationStage.DATASET_FROZEN,
            {"dataset": _digest("dataset")},
        )
    append_stage_receipt(
        generation_dir,
        GenerationStage.TEACHER_LABELLED,
        {"teachers": ["nagisa-multipv", "suisho-multipv"]},
    )
    receipt = generation_dir / "receipt-01-games_collected.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["evidence"]["unique_trajectories"] = 11
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="predecessor digest mismatch"):
        validate_stage_chain(generation_dir)


def test_rejected_generation_keeps_champion_but_advances_data_and_history(tmp_path: Path) -> None:
    plan = _plan()
    generation_dir = tmp_path / "generation-000001"
    write_generation_plan(generation_dir, plan)
    final_digest = ""
    for stage in GenerationStage:
        final_digest = append_stage_receipt(generation_dir, stage, {"stage": stage.value})
    dataset = freeze_dataset_growth(plan, _training_entries(plan), parent_train_trajectories=())
    decision = PromotionDecision(
        plan_sha256=plan.sha256,
        promoted=False,
        blockers=("candidate:not_good_enough",),
        opponent_results=(),
    )
    candidate = CandidateArtifact("meteo-generation-1", _digest("candidate-v1"))
    state = build_loop_state(
        plan,
        decision,
        dataset,
        candidate,
        final_receipt_sha256=final_digest,
    )

    assert state["champion_after"] == state["champion_before"]
    assert state["dataset_after_sha256"] == dataset.snapshot_sha256
    assert state["dataset_after_sha256"] != state["dataset_before_sha256"]
    assert state["model_history"][-1]["promotion_status"] == "rejected"
    assert state["model_history"][-1]["eligible_as_historical_opponent"]
    assert state["model_history"][-1]["weights_retained"]
    assert len(state["playable_model_window"]) == 2
    assert state["playable_model_window_limit"] == 2
    assert state["old_generation_metadata_retained"]
    assert not state["old_generation_weights_retained"]
    state_digest = append_loop_state(tmp_path, state)
    assert append_loop_state(tmp_path, state) == state_digest
    loaded = load_latest_loop_state(tmp_path)
    assert loaded is not None
    assert loaded[0] == state
    assert loaded[1] == state_digest


def test_next_generation_prunes_old_weight_but_keeps_its_metadata() -> None:
    first_plan = _plan()
    first_dataset = freeze_dataset_growth(
        first_plan,
        _training_entries(first_plan),
        parent_train_trajectories=(),
    )
    first_candidate = CandidateArtifact("meteo-generation-1", _digest("candidate-v1"))
    first_state = build_loop_state(
        first_plan,
        PromotionDecision(
            plan_sha256=first_plan.sha256,
            promoted=False,
            blockers=("candidate:not_good_enough",),
            opponent_results=(),
        ),
        first_dataset,
        first_candidate,
        final_receipt_sha256=_digest("generation-1-final-receipt"),
    )

    flat, evaluation = _openings()
    participants = _participants()
    retained_candidate = Participant(
        first_candidate.candidate_id,
        OpponentRole.HISTORICAL_METEO,
        first_candidate.artifact_sha256,
    )
    second_plan = build_generation_plan(
        generation=2,
        lineage_id="meteo-continuous-test",
        dataset_parent_sha256=first_dataset.snapshot_sha256,
        champion=participants[0],
        participants=(participants[0], retained_candidate, *participants[2:]),
        teachers=_teachers(),
        training_opening_sfens=(flat,),
        evaluation_opening_sfens=evaluation,
        config=ContinuousLoopConfig(
            training_games_per_generation=2,
            bootstrap_iterations=100,
        ),
        seed=43,
        previous_state=first_state,
    )
    second_dataset = freeze_dataset_growth(
        second_plan,
        _training_entries(second_plan),
        parent_train_trajectories=(),
    )
    second_candidate = CandidateArtifact("meteo-generation-2", _digest("candidate-v2"))
    second_state = build_loop_state(
        second_plan,
        PromotionDecision(
            plan_sha256=second_plan.sha256,
            promoted=True,
            blockers=(),
            opponent_results=(),
        ),
        second_dataset,
        second_candidate,
        previous_state=first_state,
        final_receipt_sha256=_digest("generation-2-final-receipt"),
    )

    first_history = second_state["model_history"][0]
    assert first_history["artifact_sha256"] == first_candidate.artifact_sha256
    assert first_history["weights_retained"] is False
    assert first_history["eligible_as_historical_opponent"] is False
    assert len(second_state["model_history"]) == 2
    assert len(second_state["playable_model_window"]) == 2


def test_trajectory_and_retry_identity_are_deterministic_without_counting_duplicates() -> None:
    plan = _plan()
    job = plan.training_jobs[0]
    assert trajectory_sha256(Board().to_sfen(), ("7g7f",)) == trajectory_sha256(
        Board().to_sfen(),
        ("7g7f",),
    )
    assert deterministic_retry_seed(job, retry=1) == deterministic_retry_seed(job, retry=1)
    assert deterministic_retry_seed(job, retry=1) != deterministic_retry_seed(job, retry=2)
    with pytest.raises(ValueError, match="positive"):
        deterministic_retry_seed(job, retry=0)
