from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from rsshogi.core import Board

from simajilord_shogi.distillation_targets import CANONICAL_SCORER_IDS
from simajilord_shogi.domain import TeacherScoreBound, TeacherScoreKind
from simajilord_shogi.external_usi import (
    ExternalTeacherPolicy,
    ExternalUsiTeacher,
    UsiPositionHistory,
)
from simajilord_shogi.production_score_matrix import (
    ACTUAL_PLAYED_CANDIDATE_SOURCE_ID,
    METEO_CANDIDATE_SOURCE_ID,
    SEARCH_STATE_RESET_COMMAND,
    BranchSearchEvidence,
    CandidateProposalEvidence,
    CandidateSourceKind,
    CanonicalProposalIdentity,
    CanonicalScorerIdentity,
    ExternalUsiProductionScorer,
    MatrixResolutionKind,
    MatrixUnresolvedReason,
    ProductionScoreMatrixBuilder,
    RawSearchScore,
    history_context_sha256,
    tactical_legal_moves,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _transcript_hash(sent_commands: tuple[str, ...], transcript_lines: tuple[str, ...]) -> str:
    encoded = json.dumps(
        {
            "sent_commands": sent_commands,
            "transcript_lines": transcript_lines,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _history() -> UsiPositionHistory:
    sfen = Board().to_sfen()
    return UsiPositionHistory(initial_sfen=sfen, moves=(), target_sfen=sfen)


def _proposal(
    history: UsiPositionHistory,
    source_id: str,
    moves: tuple[str, ...],
    *,
    source_kind: CandidateSourceKind = CandidateSourceKind.PROPOSAL_ONLY,
) -> CandidateProposalEvidence:
    sent_commands = (
        *((SEARCH_STATE_RESET_COMMAND,) if source_id in CANONICAL_SCORER_IDS else ()),
        history.position_command(),
        f"collect proposal {source_id}",
    )
    transcript_lines = (f"proposal {source_id} {' '.join(sorted(moves))}",)
    producer_identity_sha256 = (
        _proposal_identity(source_id).identity_sha256
        if source_id in CANONICAL_SCORER_IDS
        else _sha(f"identity:{source_id}")
    )
    return CandidateProposalEvidence(
        source_id=source_id,
        producer=f"fake-{source_id}",
        source_kind=source_kind,
        target_sfen=history.target_sfen,
        history_context_sha256=history_context_sha256(history),
        moves=tuple(sorted(moves)),
        requested_nodes=5_000,
        reported_nodes=5_000,
        multipv=(
            _proposal_identity(source_id).multipv
            if source_id in CANONICAL_SCORER_IDS
            else max(1, len(moves))
        ),
        sent_commands=sent_commands,
        transcript_lines=transcript_lines,
        transcript_sha256=_transcript_hash(sent_commands, transcript_lines),
        producer_identity_sha256=producer_identity_sha256,
        producer_code_sha256=_sha(f"producer:{source_id}"),
    )


@dataclass
class FakeScorer:
    identity: CanonicalScorerIdentity
    root_values: dict[str, float]
    child_bounds: dict[str, TeacherScoreBound]
    mate_roots: frozenset[str] = frozenset()
    underreport: bool = False

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self.proposal_calls: list[tuple[str, int]] = []

    @property
    def proposal_identity(self) -> CanonicalProposalIdentity:
        return _proposal_identity(self.identity.scorer_id)

    @property
    def proposal_process_separate(self) -> bool:
        return True

    def propose_replies(
        self,
        history: UsiPositionHistory,
        *,
        legal_moves: tuple[str, ...],
        requested_nodes: int,
    ) -> CandidateProposalEvidence:
        self.proposal_calls.append((history.target_sfen, requested_nodes))
        index = CANONICAL_SCORER_IDS.index(self.identity.scorer_id)
        move = legal_moves[index % len(legal_moves)]
        sent_commands = (
            SEARCH_STATE_RESET_COMMAND,
            history.position_command(),
            f"go nodes {requested_nodes} searchmoves {' '.join(legal_moves)}",
        )
        transcript_lines = (f"info nodes {requested_nodes} pv {move}", f"bestmove {move}")
        return CandidateProposalEvidence(
            source_id=self.identity.scorer_id,
            producer=f"fake-reply-{self.identity.scorer_id}",
            source_kind=CandidateSourceKind.PRINCIPAL_REPLY_PROPOSAL,
            target_sfen=history.target_sfen,
            history_context_sha256=history_context_sha256(history),
            moves=(move,),
            requested_nodes=requested_nodes,
            reported_nodes=requested_nodes,
            multipv=self.proposal_identity.multipv,
            sent_commands=sent_commands,
            transcript_lines=transcript_lines,
            transcript_sha256=_transcript_hash(sent_commands, transcript_lines),
            producer_identity_sha256=self.proposal_identity.identity_sha256,
            producer_code_sha256=self.proposal_identity.proposer_code_sha256,
        )

    def score_searchmove(
        self,
        history: UsiPositionHistory,
        *,
        move: str,
        requested_nodes: int,
    ) -> BranchSearchEvidence:
        self.calls.append((history.target_sfen, move, requested_nodes))
        is_root = not history.moves
        root_move = move if is_root else history.moves[-1]
        root_q = self.root_values[root_move]
        if is_root:
            q_value = root_q
            bound = TeacherScoreBound.EXACT
            score = RawSearchScore(
                q_value=q_value,
                score_kind=TeacherScoreKind.CENTIPAWN,
                bound=bound,
                score_cp=round(q_value * 1_000),
            )
        elif root_move in self.mate_roots:
            score = RawSearchScore(
                q_value=-1.0,
                score_kind=TeacherScoreKind.MATE,
                bound=TeacherScoreBound.EXACT,
                mate_plies=-3,
            )
        else:
            q_value = -root_q
            bound = self.child_bounds.get(root_move, TeacherScoreBound.EXACT)
            score = RawSearchScore(
                q_value=q_value,
                score_kind=TeacherScoreKind.CENTIPAWN,
                bound=bound,
                score_cp=round(q_value * 1_000),
            )
        sent_commands = (
            SEARCH_STATE_RESET_COMMAND,
            history.position_command(),
            f"go nodes {requested_nodes} searchmoves {move}",
        )
        transcript_lines = (
            f"info nodes {requested_nodes} score {score.score_kind.value} pv {move}",
            f"bestmove {move}",
        )
        return BranchSearchEvidence(
            scorer_id=self.identity.scorer_id,
            target_sfen=history.target_sfen,
            history_context_sha256=history_context_sha256(history),
            searchmove=move,
            requested_nodes=requested_nodes,
            reported_nodes=requested_nodes - 1 if self.underreport else requested_nodes,
            multipv=1,
            raw_score_perspective="search_position_side_to_move",
            score=score,
            depth=12,
            seldepth=18,
            time_ms=25,
            nps=200_000,
            pv=(move,),
            sent_commands=sent_commands,
            transcript_lines=transcript_lines,
            transcript_sha256=_transcript_hash(sent_commands, transcript_lines),
        )


def _identity(scorer_id: str) -> CanonicalScorerIdentity:
    return CanonicalScorerIdentity(
        scorer_id=scorer_id,
        engine_sha256=_sha(f"engine:{scorer_id}"),
        evaluation_artifacts_sha256=_sha(f"eval:{scorer_id}"),
        options=(("Hash", "256"), ("MultiPV", "1"), ("Threads", "1")),
        startup_transcript_sha256=_sha(f"startup:{scorer_id}"),
        parser_sha256=_sha("parser-v1"),
        scorer_code_sha256=_sha("fake-scorer-v1"),
        calibration_sha256=_sha(f"calibration:{scorer_id}"),
        threads=1,
        hash_mb=256,
        multipv=1,
        book_enabled=False,
    )


def _proposal_identity(scorer_id: str) -> CanonicalProposalIdentity:
    return CanonicalProposalIdentity(
        scorer_id=scorer_id,
        engine_sha256=_sha(f"engine:{scorer_id}"),
        evaluation_artifacts_sha256=_sha(f"eval:{scorer_id}"),
        options=(("Hash", "256"), ("MultiPV", "4"), ("Threads", "1")),
        startup_transcript_sha256=_sha(f"proposal-startup:{scorer_id}"),
        parser_sha256=_sha("parser-v1"),
        proposer_code_sha256=_sha("fake-scorer-v1"),
        threads=1,
        hash_mb=256,
        multipv=4,
        book_enabled=False,
    )


def _scorers(
    *,
    values: dict[str, float] | None = None,
    bounds: dict[str, TeacherScoreBound] | None = None,
    mate_roots: frozenset[str] = frozenset(),
    underreport: bool = False,
) -> tuple[FakeScorer, ...]:
    return tuple(
        FakeScorer(
            identity=_identity(scorer_id),
            root_values=values or {"2g2f": 0.2, "5g5f": -0.1, "7g7f": 0.6},
            child_bounds=bounds or {},
            mate_roots=mate_roots,
            underreport=underreport,
        )
        for scorer_id in CANONICAL_SCORER_IDS
    )


def _root_proposals(history: UsiPositionHistory) -> tuple[CandidateProposalEvidence, ...]:
    return (
        _proposal(history, CANONICAL_SCORER_IDS[0], ("7g7f",)),
        _proposal(history, CANONICAL_SCORER_IDS[1], ("2g2f",)),
        _proposal(history, CANONICAL_SCORER_IDS[2], ("7g7f",)),
        _proposal(history, METEO_CANDIDATE_SOURCE_ID, ("5g5f",)),
        _proposal(
            history,
            ACTUAL_PLAYED_CANDIDATE_SOURCE_ID,
            ("5g5f",),
            source_kind=CandidateSourceKind.ACTUAL_PLAYED_MOVE,
        ),
    )


def test_builder_cross_scores_every_candidate_and_actual_move_is_not_truth(
    tmp_path: Path,
) -> None:
    history = _history()
    scorers = _scorers()
    receipt = ProductionScoreMatrixBuilder(
        scorers,
        requested_nodes=10_000,
        reply_proposal_nodes=2_000,
    ).build(history, _root_proposals(history))

    assert tuple(candidate.move for candidate in receipt.candidates) == (
        "2g2f",
        "5g5f",
        "7g7f",
    )
    actual = next(candidate for candidate in receipt.candidates if candidate.move == "5g5f")
    assert ACTUAL_PLAYED_CANDIDATE_SOURCE_ID in actual.source_ids
    assert receipt.proposal_membership_is_truth is False
    assert receipt.qsearch_leaf_re_evaluated is False
    assert receipt.position_mode == "exact_history_root_with_one_ply_reply_reanalysis"
    assert receipt.cross_teacher_value_average is False
    assert receipt.majority_vote is False
    assert all(
        matrix.resolution.kind is MatrixResolutionKind.EXACT_Q_DISTRIBUTION
        and matrix.resolution.chosen_move == "7g7f"
        for matrix in receipt.scorer_matrices
    )
    assert all(
        branch.root_search.requested_nodes == 10_000 and branch.root_search.multipv == 1
        for matrix in receipt.scorer_matrices
        for branch in matrix.candidates
    )
    assert all(
        set(search.searchmove for search in branch.reply_searches)
        == set(
            next(row for row in receipt.reply_unions if row.root_move == branch.move).reply_moves
        )
        for matrix in receipt.scorer_matrices
        for branch in matrix.candidates
    )
    assert scorers[0].calls
    assert all(scorer.proposal_calls for scorer in scorers)

    output = tmp_path / "matrix.json"
    receipt.write_create_only(output)
    payload = json.loads(output.read_text())
    assert payload["receipt_sha256"] == receipt.receipt_sha256
    assert payload["history"]["target_sfen"] == history.target_sfen
    assert payload["scorer_matrices"][0]["scorer"]["engine_sha256"]
    assert payload["scorer_matrices"][0]["scorer"]["evaluation_artifacts_sha256"]
    assert payload["scorer_matrices"][0]["scorer"]["options"]
    assert payload["scorer_matrices"][0]["candidates"][0]["root_search"]["transcript_sha256"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        receipt.write_create_only(output)


def test_interval_dominance_uses_bounds_without_midpoint_averaging() -> None:
    history = _history()
    scorers = _scorers(
        values={"2g2f": 0.5, "5g5f": 0.1, "7g7f": 0.7},
        bounds={
            "2g2f": TeacherScoreBound.LOWER,
            "5g5f": TeacherScoreBound.LOWER,
            "7g7f": TeacherScoreBound.UPPER,
        },
    )
    receipt = ProductionScoreMatrixBuilder(
        scorers,
        requested_nodes=10_000,
        reply_proposal_nodes=2_000,
        interval_dominance_margin=0.05,
    ).build(history, _root_proposals(history))

    for matrix in receipt.scorer_matrices:
        resolution = matrix.resolution
        assert resolution.kind is MatrixResolutionKind.INTERVAL_DOMINANCE
        assert resolution.chosen_move == "7g7f"
        assert resolution.value_interval is not None
        assert resolution.value_interval.lower == pytest.approx(0.7)
        assert resolution.value_interval.upper == pytest.approx(1.0)
        rows = {candidate.move: candidate for candidate in matrix.candidates}
        assert rows["7g7f"].reply_searches[0].score.bound is TeacherScoreBound.UPPER
        assert rows["2g2f"].reply_searches[0].score.bound is TeacherScoreBound.LOWER


def test_overlapping_bounds_and_unproven_mate_remain_unresolved() -> None:
    history = _history()
    overlapping = _scorers(
        values={"2g2f": 0.5, "5g5f": 0.4, "7g7f": 0.6},
        bounds={
            "2g2f": TeacherScoreBound.UPPER,
            "5g5f": TeacherScoreBound.UPPER,
            "7g7f": TeacherScoreBound.LOWER,
        },
    )
    receipt = ProductionScoreMatrixBuilder(
        overlapping,
        requested_nodes=10_000,
        reply_proposal_nodes=2_000,
    ).build(history, _root_proposals(history))
    assert all(matrix.resolution.unresolved for matrix in receipt.scorer_matrices)
    assert all(
        MatrixUnresolvedReason.OVERLAPPING_INTERVALS in matrix.resolution.unresolved_reasons
        for matrix in receipt.scorer_matrices
    )

    mates = _scorers(mate_roots=frozenset({"7g7f"}))
    mate_receipt = ProductionScoreMatrixBuilder(
        mates,
        requested_nodes=10_000,
        reply_proposal_nodes=2_000,
    ).build(history, _root_proposals(history))
    for matrix in mate_receipt.scorer_matrices:
        assert matrix.resolution.unresolved is True
        assert MatrixUnresolvedReason.REPORTED_MATE_UNPROVEN in (
            matrix.resolution.unresolved_reasons
        )
        mate_candidate = next(
            candidate for candidate in matrix.candidates if candidate.move == "7g7f"
        )
        assert mate_candidate.reply_searches[0].score.score_kind is TeacherScoreKind.MATE
        assert mate_candidate.reply_searches[0].score.mate_plies == -3


def test_underreported_requested_nodes_lower_confidence_and_require_more_search() -> None:
    history = _history()
    receipt = ProductionScoreMatrixBuilder(
        _scorers(underreport=True),
        requested_nodes=10_000,
        reply_proposal_nodes=2_000,
    ).build(history, _root_proposals(history))

    assert all(matrix.resolution.unresolved for matrix in receipt.scorer_matrices)
    assert all(
        MatrixUnresolvedReason.REQUESTED_BUDGET_NOT_FULFILLED
        in matrix.resolution.unresolved_reasons
        for matrix in receipt.scorer_matrices
    )


def test_tactical_enumeration_adds_only_legal_forcing_moves() -> None:
    board = Board("4k4/9/5R3/9/9/9/9/9/4K4 b G 1")
    tactical = tactical_legal_moves(board)

    legal = {move.to_usi() for move in board.legal_moves()}
    assert tactical
    assert set(tactical).issubset(legal)
    assert "4c5c" in tactical


def test_builder_merges_internal_tactical_candidates_before_cross_scoring() -> None:
    sfen = "4k4/9/5R3/9/9/9/9/9/4K4 b G 1"
    history = UsiPositionHistory(initial_sfen=sfen, moves=(), target_sfen=sfen)
    forcing = tactical_legal_moves(history.target_board())
    seed_move = forcing[0]
    proposals = tuple(
        _proposal(history, source_id, (seed_move,))
        for source_id in (*CANONICAL_SCORER_IDS, METEO_CANDIDATE_SOURCE_ID)
    )
    receipt = ProductionScoreMatrixBuilder(
        _scorers(values={move: 0.0 for move in forcing}),
        requested_nodes=10_000,
        reply_proposal_nodes=2_000,
    ).build(history, proposals)

    assert tuple(candidate.move for candidate in receipt.candidates) == forcing
    assert all(
        "tactical" in candidate.source_ids
        for candidate in receipt.candidates
        if candidate.move != seed_move
    )


def test_candidate_transcript_hash_is_verified_not_trusted() -> None:
    history = _history()
    proposal = _proposal(history, METEO_CANDIDATE_SOURCE_ID, ("7g7f",))

    with pytest.raises(ValueError, match="does not match"):
        replace(proposal, transcript_sha256="0" * 64)


def test_external_usi_adapter_preserves_real_searchmove_transcript(tmp_path: Path) -> None:
    engine_path = tmp_path / "fake_usi.py"
    engine_path.write_text(
        """#!/usr/bin/env python3
import sys
for raw in sys.stdin:
    command = raw.strip()
    if command == 'usi':
        print('id name Matrix Fake')
        print('option name MultiPV type spin default 1 min 1 max 16')
        print('option name Threads type spin default 1 min 1 max 16')
        print('option name Hash type spin default 256 min 1 max 4096')
        print('usiok', flush=True)
    elif command == 'isready':
        print('readyok', flush=True)
    elif command.startswith('go nodes '):
        fields = command.split()
        nodes = int(fields[2])
        choices = fields[4:] if 'searchmoves' in fields else ['7g7f']
        move = choices[0]
        print(f'info depth 4 seldepth 6 nodes {nodes} time 1 nps {nodes} score cp 42 pv {move}')
        print(f'bestmove {move}', flush=True)
    elif command == 'quit':
        break
""",
        encoding="utf-8",
    )
    engine_path.chmod(engine_path.stat().st_mode | stat.S_IXUSR)
    scorer_id = CANONICAL_SCORER_IDS[0]
    policy = ExternalTeacherPolicy(
        policy_id=scorer_id,
        name="matrix fake",
        source="unit test",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=False,
    )
    history = _history()
    with ExternalUsiTeacher(
        [str(engine_path)],
        policy,
        nodes=1_000,
        multipv=1,
        options={"Hash": 256, "Threads": 1},
        training_use=True,
        working_directory=tmp_path,
    ) as engine:
        adapter = ExternalUsiProductionScorer(engine, _identity(scorer_id))
        proposal = adapter.propose_replies(
            history,
            legal_moves=("2g2f", "7g7f"),
            requested_nodes=321,
        )
        branch = adapter.score_searchmove(history, move="7g7f", requested_nodes=654)

    assert proposal.moves == ("2g2f",)
    assert proposal.reported_nodes == 321
    assert "go nodes 321 searchmoves 2g2f 7g7f" in proposal.sent_commands
    assert proposal.sent_commands[0] == SEARCH_STATE_RESET_COMMAND
    assert branch.reported_nodes == 654
    assert branch.depth == 4
    assert branch.seldepth == 6
    assert branch.score.score_cp == 42
    assert "go nodes 654 searchmoves 7g7f" in branch.sent_commands
    assert branch.sent_commands[0] == SEARCH_STATE_RESET_COMMAND


def test_builder_fails_closed_without_every_required_root_proposer() -> None:
    history = _history()
    proposals = _root_proposals(history)

    with pytest.raises(ValueError, match="missing required root candidate sources"):
        ProductionScoreMatrixBuilder(
            _scorers(),
            requested_nodes=10_000,
            reply_proposal_nodes=2_000,
        ).build(
            history,
            tuple(
                proposal
                for proposal in proposals
                if proposal.source_id != METEO_CANDIDATE_SOURCE_ID
            ),
        )
