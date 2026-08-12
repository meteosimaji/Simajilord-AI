from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from rsshogi.core import Board

from simajilord_shogi.distillation_targets import (
    CANONICAL_POSITION_IDENTITY,
    CANONICAL_SCORER_IDS,
    CANONICAL_TARGET_MODE,
    CANONICAL_TARGET_SIDECAR_SCHEMA,
    canonical_history_context_sha256,
    load_canonical_target_sidecar,
    resolve_canonical_target_sidecar,
)
from simajilord_shogi.domain import GameRecord, PositionSample, Termination
from simajilord_shogi.replay import append_games


def _replay(path: Path, *, termination: Termination = Termination.RESIGNATION) -> GameRecord:
    sample = PositionSample(
        sfen=Board().to_sfen(),
        ply=0,
        turn=0,
        policy={"8g8f": 1.0},
        root_value=0.25,
        value_target=1.0,
        chosen_move="7g7f",
        actor_best_move="8g8f",
        actor_source="meteo-actor",
    )
    game = GameRecord(
        initial_sfen=Board().to_sfen(),
        moves=("7g7f",),
        samples=(sample,),
        winner=0,
        termination=termination,
    )
    append_games(path, [game])
    return game


def _payload(replay: Path, game: GameRecord) -> dict[str, object]:
    policy_nagisa = {"7g7f": 0.7, "2g2f": 0.2, "8g8f": 0.1}
    policy_suisho = {"2g2f": 0.55, "7g7f": 0.35, "8g8f": 0.1}
    policy_soujou = {"2g2f": 0.6, "7g7f": 0.3, "8g8f": 0.1}
    return {
        "schema": CANONICAL_TARGET_SIDECAR_SCHEMA,
        "replay": {
            "sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
            "bytes": replay.stat().st_size,
        },
        "position_identity": CANONICAL_POSITION_IDENTITY,
        "target_mode": CANONICAL_TARGET_MODE,
        "canonical_scorers": [
            {
                "scorer_id": CANONICAL_SCORER_IDS[0],
                "family": "nagisa",
                "role": "canonical_scorer",
            },
            {
                "scorer_id": CANONICAL_SCORER_IDS[1],
                "family": "suisho",
                "role": "canonical_scorer",
            },
            {
                "scorer_id": CANONICAL_SCORER_IDS[2],
                "family": "soujou",
                "role": "canonical_scorer",
            },
        ],
        "positions": [
            {
                "game_index": 0,
                "sample_index": 0,
                "normalized_sfen": " ".join(Board().to_sfen().split()[:3]),
                "history_context_sha256": canonical_history_context_sha256(
                    game, game.samples[0]
                ),
                "scorers": [
                    {
                        "scorer_id": CANONICAL_SCORER_IDS[0],
                        "best_move": "7g7f",
                        "policy": policy_nagisa,
                        "value": -0.4,
                    },
                    {
                        "scorer_id": CANONICAL_SCORER_IDS[1],
                        "best_move": "2g2f",
                        "policy": policy_suisho,
                        "value": -0.2,
                    },
                    {
                        "scorer_id": CANONICAL_SCORER_IDS[2],
                        "best_move": "2g2f",
                        "policy": policy_soujou,
                        "value": -0.1,
                    },
                ],
                "canonical_best_union": ["2g2f", "7g7f"],
                "equivalence_groups": [
                    {
                        "group_id": 0,
                        "kind": "near_optimal",
                        "moves": ["2g2f", "7g7f"],
                        "target_mass": 1.0,
                        "worst_teacher_regret": 0.03,
                        "teacher_depth_dispersion": 0.01,
                        "reply_dispersion": 0.02,
                    }
                ],
                "proven_winning_mate_set": {"status": "none", "moves": []},
                "canonical_values": {
                    CANONICAL_SCORER_IDS[0]: -0.4,
                    CANONICAL_SCORER_IDS[1]: -0.2,
                    CANONICAL_SCORER_IDS[2]: -0.1,
                },
                "history_training_weight": 1.0,
            }
        ],
    }


def _write_sidecar(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_canonical_sidecar_binds_exact_replay_history_and_dual_scorers(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "actor.jsonl"
    game = _replay(replay)
    sidecar = replay.with_suffix(replay.suffix + ".distillation-targets.json")
    _write_sidecar(sidecar, _payload(replay, game))

    loaded = load_canonical_target_sidecar(replay, sidecar, games=[game])

    assert loaded.schema == CANONICAL_TARGET_SIDECAR_SCHEMA
    assert tuple(item.scorer_id for item in loaded.canonical_scorers) == (
        CANONICAL_SCORER_IDS
    )
    assert loaded.positions[0].canonical_best_union == ("2g2f", "7g7f")
    assert loaded.positions[0].canonical_values == {
        CANONICAL_SCORER_IDS[0]: -0.4,
        CANONICAL_SCORER_IDS[1]: -0.2,
        CANONICAL_SCORER_IDS[2]: -0.1,
    }
    assert resolve_canonical_target_sidecar(replay) == sidecar.resolve()


def test_anonymous_ensemble_or_incomplete_scorer_pair_is_rejected(tmp_path: Path) -> None:
    replay = tmp_path / "actor.jsonl"
    game = _replay(replay)
    sidecar = tmp_path / "targets.json"
    payload = _payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list)
    position = positions[0]
    assert isinstance(position, dict)
    scorers = position["scorers"]
    assert isinstance(scorers, list)
    scorers[0]["scorer_id"] = "meteo-teacher-ensemble"
    _write_sidecar(sidecar, payload)
    with pytest.raises(ValueError, match="exactly NAGISA, Suisho11Plus, and Soujou"):
        load_canonical_target_sidecar(replay, sidecar, games=[game])


def test_canonical_scorer_best_move_must_be_legal_and_present_in_policy(tmp_path: Path) -> None:
    replay = tmp_path / "actor.jsonl"
    game = _replay(replay)
    sidecar = tmp_path / "targets.json"
    payload = _payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list)
    position = positions[0]
    assert isinstance(position, dict)
    scorers = position["scorers"]
    assert isinstance(scorers, list)
    scorer = scorers[0]
    assert isinstance(scorer, dict)
    scorer["best_move"] = "9a9b"
    _write_sidecar(sidecar, payload)
    with pytest.raises(ValueError, match="illegal canonical scorer"):
        load_canonical_target_sidecar(replay, sidecar, games=[game])

    scorer["best_move"] = "1g1f"
    _write_sidecar(sidecar, payload)
    with pytest.raises(ValueError, match="best_move is absent from its policy"):
        load_canonical_target_sidecar(replay, sidecar, games=[game])

    scorer["best_move"] = "2g2f"
    scorer["policy"] = {"2g2f": 0.1, "7g7f": 0.9}
    _write_sidecar(sidecar, payload)
    with pytest.raises(ValueError, match="not a maximum-probability move"):
        load_canonical_target_sidecar(replay, sidecar, games=[game])

    payload = _payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list)
    position = positions[0]
    assert isinstance(position, dict)
    position["scorers"] = position["scorers"][:1]
    _write_sidecar(sidecar, payload)
    with pytest.raises(ValueError, match="exactly NAGISA, Suisho11Plus, and Soujou"):
        load_canonical_target_sidecar(replay, sidecar, games=[game])


def test_sidecar_rejects_history_drift_group_overlap_and_unproven_mate_moves(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "actor.jsonl"
    game = _replay(replay)
    sidecar = tmp_path / "targets.json"
    payload = _payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list)
    position = positions[0]
    assert isinstance(position, dict)
    position["history_context_sha256"] = "0" * 64
    _write_sidecar(sidecar, payload)
    with pytest.raises(ValueError, match="history identity"):
        load_canonical_target_sidecar(replay, sidecar, games=[game])

    payload = _payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list)
    position = positions[0]
    assert isinstance(position, dict)
    groups = position["equivalence_groups"]
    assert isinstance(groups, list)
    groups.append(
        {
            "group_id": 1,
            "kind": "near_optimal",
            "moves": ["7g7f"],
            "target_mass": 0.2,
            "worst_teacher_regret": 0.0,
            "teacher_depth_dispersion": 0.0,
            "reply_dispersion": 0.0,
        }
    )
    groups[0]["target_mass"] = 0.8
    _write_sidecar(sidecar, payload)
    with pytest.raises(ValueError, match="overlap"):
        load_canonical_target_sidecar(replay, sidecar, games=[game])

    payload = _payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list)
    position = positions[0]
    assert isinstance(position, dict)
    position["proven_winning_mate_set"] = {"status": "none", "moves": ["7g7f"]}
    _write_sidecar(sidecar, payload)
    with pytest.raises(ValueError, match="only a proven mate set"):
        load_canonical_target_sidecar(replay, sidecar, games=[game])


def test_sidecar_resolution_rejects_distinct_explicit_and_adjacent_candidates(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "actor.jsonl"
    _replay(replay)
    adjacent = replay.with_suffix(replay.suffix + ".distillation-targets.json")
    explicit = tmp_path / "explicit.json"
    adjacent.write_text("{}\n", encoding="utf-8")
    explicit.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="multiple canonical target sidecars"):
        resolve_canonical_target_sidecar(replay, explicit=explicit)

    assert resolve_canonical_target_sidecar(replay, explicit=adjacent) == adjacent.resolve()


def test_zero_mass_equivalence_group_is_valid_but_negative_mass_is_rejected(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "actor.jsonl"
    game = _replay(replay)
    sidecar = tmp_path / "targets.json"
    payload = _payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list)
    position = positions[0]
    assert isinstance(position, dict)
    groups = position["equivalence_groups"]
    assert isinstance(groups, list)
    first_group = groups[0]
    assert isinstance(first_group, dict)
    first_group["target_mass"] = 0.0
    groups.append(
        {
            "group_id": 1,
            "kind": "near_optimal",
            "moves": ["8g8f"],
            "target_mass": 1.0,
            "worst_teacher_regret": 0.0,
            "teacher_depth_dispersion": 0.0,
            "reply_dispersion": 0.0,
        }
    )
    _write_sidecar(sidecar, payload)
    loaded = load_canonical_target_sidecar(replay, sidecar, games=[game])
    assert loaded.positions[0].equivalence_groups[0].target_mass == 0.0

    second_group = groups[1]
    assert isinstance(second_group, dict)
    first_group["target_mass"] = -0.1
    second_group["target_mass"] = 1.1
    _write_sidecar(sidecar, payload)
    with pytest.raises(ValueError, match="target_mass must be non-negative"):
        load_canonical_target_sidecar(replay, sidecar, games=[game])


def test_self_declared_proven_mate_is_rejected_until_proof_artifact_is_verified(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "actor.jsonl"
    game = _replay(replay)
    sidecar = tmp_path / "targets.json"
    payload = _payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list)
    position = positions[0]
    assert isinstance(position, dict)
    position["proven_winning_mate_set"] = {"status": "proven", "moves": ["8g8f"]}
    _write_sidecar(sidecar, payload)

    with pytest.raises(ValueError, match="rejects self-declared proven mates"):
        load_canonical_target_sidecar(replay, sidecar, games=[game])


def test_equivalence_group_moves_must_be_scored_by_both_canonical_teachers(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "actor.jsonl"
    game = _replay(replay)
    sidecar = tmp_path / "targets.json"
    payload = _payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list)
    position = positions[0]
    assert isinstance(position, dict)
    groups = position["equivalence_groups"]
    assert isinstance(groups, list)
    group = groups[0]
    assert isinstance(group, dict)
    group["moves"] = ["2g2f", "5i6h", "7g7f"]
    _write_sidecar(sidecar, payload)

    with pytest.raises(ValueError, match="scored by every canonical teacher"):
        load_canonical_target_sidecar(replay, sidecar, games=[game])


def test_canonical_sidecar_rejects_incomplete_game_even_with_complete_targets(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "actor.jsonl"
    game = _replay(replay, termination=Termination.MAX_PLIES)
    sidecar = tmp_path / "targets.json"
    _write_sidecar(sidecar, _payload(replay, game))

    with pytest.raises(ValueError, match="rejects incomplete max_plies"):
        load_canonical_target_sidecar(replay, sidecar, games=[game])
