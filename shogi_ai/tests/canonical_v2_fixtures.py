from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from rsshogi.core import Board, Move

from simajilord_shogi.distillation_targets import (
    CANONICAL_POSITION_IDENTITY,
    CANONICAL_SCORER_IDS,
    canonical_history_context_sha256,
)
from simajilord_shogi.distillation_targets_v2 import (
    CANONICAL_CALIBRATION_FORMULA,
    CANONICAL_CALIBRATION_SCHEMA,
    CANONICAL_POLICY_CALIBRATION_FORMULA,
    CANONICAL_Q_VALUE_PERSPECTIVE,
    CANONICAL_RAW_SCORE_PERSPECTIVE,
    CANONICAL_TARGET_V2_MODE,
    CANONICAL_TARGET_V2_SCHEMA,
)
from simajilord_shogi.domain import GameRecord, PositionSample, Termination
from simajilord_shogi.model_rights import model_rights
from simajilord_shogi.replay import append_games
from simajilord_shogi.research_context import game_phase
from simajilord_shogi.rights_lineage import (
    merge_rights_restriction_summaries,
    summarize_teacher_sidecar,
)


def write_canonical_v2_replay(
    path: Path,
    *,
    initial_sfen: str | None = None,
) -> GameRecord:
    initial_board = Board() if initial_sfen is None else Board(initial_sfen)
    sample = PositionSample(
        sfen=initial_board.to_sfen(),
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
        initial_sfen=initial_board.to_sfen(),
        moves=("7g7f",),
        samples=(sample,),
        winner=0,
        termination=Termination.RESIGNATION,
    )
    append_games(path, [game])
    return game


def canonical_v2_rights_summary() -> dict[str, object]:
    summaries = [
        summarize_teacher_sidecar(
            {
                "publication_allowed": index == 0,
                "rights": model_rights(scorer_id).to_dict(),
            },
            sidecar_sha256=marker * 64,
        )
        for index, (scorer_id, marker) in enumerate(
            zip(CANONICAL_SCORER_IDS, ("a", "b", "c"), strict=True)
        )
    ]
    return merge_rights_restriction_summaries(summaries)


def calibrated_cp_and_q(desired_root_q: float, *, root_player_sign: int) -> tuple[int, float]:
    raw_q = desired_root_q / root_player_sign
    score_cp = round(1200.0 * math.atanh(raw_q))
    return score_cp, root_player_sign * math.tanh(score_cp / 1200.0)


def _candidate_score(
    move_usi: str,
    *,
    q_value: float,
    depth: int,
    requested_nodes: int,
    replies: list[str],
) -> dict[str, object]:
    reply = replies[0]
    score_cp, calibrated_q = calibrated_cp_and_q(q_value, root_player_sign=1)
    reply_cp, reply_q = calibrated_cp_and_q(q_value - 0.05, root_player_sign=-1)
    return {
        "move": move_usi,
        "q_value": calibrated_q,
        "raw_score_perspective": CANONICAL_RAW_SCORE_PERSPECTIVE,
        "root_player_sign": 1,
        "score_kind": "cp",
        "bound": "exact",
        "score_cp": score_cp,
        "mate_plies": None,
        "mate_unknown_sign": None,
        "reported_nodes": requested_nodes,
        "depth": depth,
        "time_ms": 10,
        "pv": [move_usi, reply],
        "principal_replies": [
            {
                "move": reply_move,
                "q_value": reply_q,
                "raw_score_perspective": CANONICAL_RAW_SCORE_PERSPECTIVE,
                "root_player_sign": -1,
                "score_kind": "cp",
                "bound": "exact",
                "score_cp": reply_cp,
                "mate_plies": None,
                "mate_unknown_sign": None,
                "reported_nodes": requested_nodes,
                "depth": depth,
                "time_ms": 8,
                "pv": [reply_move],
            }
            for reply_move in replies
        ],
    }


def _budget_score(
    candidates: list[str],
    values: tuple[float, float],
    *,
    budget: int,
    budget_index: int,
    reply_candidates: list[dict[str, object]],
) -> dict[str, object]:
    scored_candidates: list[dict[str, object]] = []
    for move, q_value, reply_candidate in zip(
        candidates, values, reply_candidates, strict=True
    ):
        raw_reply_moves = reply_candidate["reply_moves"]
        assert isinstance(raw_reply_moves, list) and all(
            isinstance(reply_move, str) for reply_move in raw_reply_moves
        )
        scored_candidates.append(
            _candidate_score(
                move,
                q_value=q_value,
                depth=10 + budget_index,
                requested_nodes=budget,
                replies=list(raw_reply_moves),
            )
        )
    reported_nodes = 0
    for candidate in scored_candidates:
        candidate_nodes = candidate["reported_nodes"]
        candidate_replies = candidate["principal_replies"]
        assert isinstance(candidate_nodes, int)
        assert isinstance(candidate_replies, list)
        reported_nodes += candidate_nodes
        for reply in candidate_replies:
            assert isinstance(reply, dict)
            reply_nodes = reply["reported_nodes"]
            assert isinstance(reply_nodes, int)
            reported_nodes += reply_nodes
    return {
        "requested_nodes": budget,
        "reported_nodes": reported_nodes,
        "candidates": scored_candidates,
    }


def build_canonical_v2_payload(
    replay: Path,
    game: GameRecord,
    *,
    coverage_complete: bool = True,
    unanimous: bool = True,
) -> dict[str, object]:
    candidates = ["2g2f", "7g7f"]
    budgets = [100, 200]
    scorer_values = (
        (
            ((0.26, 0.18), (0.29, 0.20))
            if unanimous
            else ((-0.36, -0.31), (-0.35, -0.30))
        ),
        ((0.38, 0.28), (0.40, 0.30)),
        ((0.36, 0.29), (0.39, 0.31)),
    )
    reply_proposals: list[dict[str, object]] = []
    reply_candidates_by_budget: list[list[dict[str, object]]] = []
    for budget_index, budget in enumerate(budgets):
        reply_candidates: list[dict[str, object]] = []
        for candidate_index, candidate in enumerate(candidates):
            child = Board()
            child.apply_move(Move.from_usi(candidate))
            legal_replies = sorted(move.to_usi() for move in child.legal_moves())
            proposed = (
                legal_replies[0],
                legal_replies[len(legal_replies) // 2],
                legal_replies[-1],
            )
            union = sorted(set(proposed))
            reply_candidates.append(
                {
                    "move": candidate,
                    "proposals": [
                        {
                            "scorer_id": scorer_id,
                            "provenance_sha256": hashlib.sha256(
                                f"{budget_index}:{candidate_index}:{scorer_index}".encode()
                            ).hexdigest(),
                            "moves": [proposed[scorer_index]],
                            "reported_nodes": budget,
                            "depth": 9 + budget_index,
                            "time_ms": 7,
                        }
                        for scorer_index, scorer_id in enumerate(CANONICAL_SCORER_IDS)
                    ],
                    "reply_moves": union,
                }
            )
        reply_proposals.append(
            {
                "requested_nodes": budget,
                "candidates": reply_candidates,
            }
        )
        reply_candidates_by_budget.append(reply_candidates)
    descriptors = []
    for index, (scorer_id, family) in enumerate(
        zip(CANONICAL_SCORER_IDS, ("nagisa", "suisho", "soujou"), strict=True)
    ):
        marker = str(index + 1)
        descriptors.append(
            {
                "scorer_id": scorer_id,
                "family": family,
                "role": "canonical_scorer",
                "engine_sha256": marker * 64,
                "evaluation_artifacts_sha256": str(index + 3) * 64,
                "startup_provenance_sha256": str(index + 5) * 64,
                "option_fingerprint_sha256": str(index + 7) * 64,
                "history_mode": "game_prefix",
                "book_enabled": False,
                "threads": 1,
                "hash_mb": 256,
                "multipv": 8,
            }
        )
    matrices = []
    for scorer_id, budget_values in zip(
        CANONICAL_SCORER_IDS, scorer_values, strict=True
    ):
        matrices.append(
            {
                "scorer_id": scorer_id,
                "q_value_perspective": CANONICAL_Q_VALUE_PERSPECTIVE,
                "calibration": {
                    "schema": CANONICAL_CALIBRATION_SCHEMA,
                    "phase": game_phase(0).value,
                    "formula": CANONICAL_CALIBRATION_FORMULA,
                    "policy_formula": CANONICAL_POLICY_CALIBRATION_FORMULA,
                    "policy_temperature": 600.0,
                    "ponanza_coefficient": 600.0,
                    "tanh_denominator": 1200.0,
                    "artifact_sha256": "c" * 64,
                    "independent_validation_passed": True,
                },
                "budgets": [
                    _budget_score(
                        candidates,
                        values,
                        budget=budget,
                        budget_index=budget_index,
                        reply_candidates=reply_candidates_by_budget[budget_index],
                    )
                    for budget_index, (budget, values) in enumerate(
                        zip(budgets, budget_values, strict=True)
                    )
                ],
            }
        )
    scorer_targets: list[dict[str, object]] = []
    for scorer_id, matrix in zip(CANONICAL_SCORER_IDS, matrices, strict=True):
        calibration = matrix["calibration"]
        matrix_budgets = matrix["budgets"]
        assert isinstance(calibration, dict) and isinstance(matrix_budgets, list)
        deepest = matrix_budgets[-1]
        assert isinstance(deepest, dict)
        scored_candidates = deepest["candidates"]
        assert isinstance(scored_candidates, list)
        maximum_cp = max(int(candidate["score_cp"]) for candidate in scored_candidates)
        temperature = float(calibration["policy_temperature"])
        weights = {
            str(candidate["move"]): math.exp(
                (int(candidate["score_cp"]) - maximum_cp) / temperature
            )
            for candidate in scored_candidates
        }
        total = math.fsum(weights.values())
        policy = {move: weight / total for move, weight in weights.items()}
        best_move = max(policy, key=policy.__getitem__)
        best_candidate = next(
            candidate for candidate in scored_candidates if candidate["move"] == best_move
        )
        scorer_targets.append(
            {
                "scorer_id": scorer_id,
                "best_move": best_move,
                "policy": policy,
                "value": best_candidate["q_value"],
            }
        )
    play = (
        {
            "kind": "unanimous_consensus",
            "train_play": True,
            "additional_search_required": False,
        }
        if coverage_complete and unanimous
        else {
            "kind": "unresolved",
            "train_play": False,
            "additional_search_required": True,
        }
    )
    return {
        "schema": CANONICAL_TARGET_V2_SCHEMA,
        "replay": {
            "sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
            "bytes": replay.stat().st_size,
        },
        "position_identity": CANONICAL_POSITION_IDENTITY,
        "target_mode": CANONICAL_TARGET_V2_MODE,
        "canonical_scorers": descriptors,
        "required_candidate_families": [
            "meteo",
            "nagisa",
            "soujou",
            "suisho",
            "tactical",
        ],
        "rights_restriction_summary": canonical_v2_rights_summary(),
        "positions": [
            {
                "game_index": 0,
                "sample_index": 0,
                "normalized_sfen": " ".join(Board().to_sfen().split()[:3]),
                "history_context_sha256": canonical_history_context_sha256(
                    game, game.samples[0]
                ),
                "scorers": scorer_targets,
                "candidate_proposals": [
                    {
                        "family": family,
                        "producer": f"fixture-{family}-candidate-proposer-v1",
                        "provenance_sha256": str(index + 1) * 64,
                        "moves": candidates,
                    }
                    for index, family in enumerate(
                        ("meteo", "nagisa", "soujou", "suisho", "tactical")
                        if coverage_complete
                        else ("nagisa", "soujou", "suisho")
                    )
                ],
                "candidate_family_coverage_complete": coverage_complete,
                "reply_coverage_complete": True,
                "candidate_moves": candidates,
                "budgets": budgets,
                "reply_proposals": reply_proposals,
                "score_matrix": matrices,
                "thresholds": {
                    "regret": 0.05,
                    "depth_dispersion": 0.05,
                    "reply_dispersion": 0.05,
                    "policy_temperature": 0.025,
                },
                "proof": {
                    "state": "not_applicable",
                    "max_plies": None,
                    "node_limit": None,
                    "moves": [],
                },
                "play": play,
            }
        ],
    }


def write_canonical_v2_sidecar(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
