"""Production receipts for equal-budget, individually constrained USI scores.

One or three canonical teachers, Meteo, tactical enumeration, and—only in the
separate B9 arm—the other six registered models may *propose* moves.  Proposal
membership is never a correctness label.  A single pinned anchor instead
receives every root candidate through an independent ``MultiPV=1``
``searchmoves <move>`` search with the same requested node budget.  Canonical
proposal engines run as separate ``MultiPV=K`` processes.  Opponent replies
are unioned and then scored one at a time by the same anchor before a
negamax/min backup.

This module intentionally produces an evidence receipt rather than a training
sidecar.  It preserves centipawn, mate, lower-bound, and upper-bound domains;
it never averages values between scorers or replaces a bound with a midpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from rsshogi.core import Board, Move

from .adjudication import adjudicate_board
from .distillation_targets import CANONICAL_SCORER_IDS
from .domain import TeacherScoreBound, TeacherScoreKind, TeacherVariation
from .external_usi import ExternalUsiTeacher, UsiAnalysis, UsiPositionHistory

PRODUCTION_SCORE_MATRIX_SCHEMA = "meteo-production-score-matrix-receipt-v2"
PRODUCTION_SCORE_MATRIX_BUILDER = "multi-proposer-single-anchor-equal-node-negamax-v2"
METEO_CANDIDATE_SOURCE_ID = "meteo"
TACTICAL_CANDIDATE_SOURCE_ID = "tactical"
ACTUAL_PLAYED_CANDIDATE_SOURCE_ID = "actual_played_move"
RAW_SCORE_PERSPECTIVE = "search_position_side_to_move"
SEARCH_STATE_RESET_COMMAND = "usinewgame"
SEARCH_STATE_RESET_MODE = "usinewgame_before_every_independent_searchmoves"
PROPOSAL_ROLE = "top_k_candidate_generation_only_not_policy_or_value_truth"
SCORING_ROLE = "single_anchor_equal_node_multipv1_searchmoves"

EXPERIMENT_ARM_PROPOSAL_SOURCES: dict[str, tuple[str, ...]] = {
    "A-N": (CANONICAL_SCORER_IDS[0],),
    "A-W": (CANONICAL_SCORER_IDS[1],),
    "A-S": (CANONICAL_SCORER_IDS[2],),
    "B": CANONICAL_SCORER_IDS,
    "B9": CANONICAL_SCORER_IDS,
}
AUXILIARY_PROPOSER_IDS = (
    "aobannue-v1.1",
    "suisho5",
    "shinden3-2025-02-21",
    "gikou2-v2.0.2",
    "hao-2023-05-08",
    "tanuki-dr4-2023-12-03",
)
EXPERIMENT_ARM_AUXILIARY_SOURCES: dict[str, tuple[str, ...]] = {
    "A-N": (),
    "A-W": (),
    "A-S": (),
    "B": (),
    "B9": AUXILIARY_PROPOSER_IDS,
}

_SHA256_HEXDIGITS = frozenset("0123456789abcdef")


def _require_sha256(value: str, *, label: str) -> None:
    if len(value) != 64 or any(character not in _SHA256_HEXDIGITS for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _transcript_sha256(sent_commands: tuple[str, ...], transcript_lines: tuple[str, ...]) -> str:
    return _json_sha256(
        {
            "sent_commands": sent_commands,
            "transcript_lines": transcript_lines,
        }
    )


def _validate_transcript(
    sent_commands: tuple[str, ...],
    transcript_lines: tuple[str, ...],
    transcript_sha256: str,
    *,
    label: str,
) -> None:
    if not sent_commands:
        raise ValueError(f"{label} must record at least one sent command")
    if not transcript_lines:
        raise ValueError(f"{label} must record at least one response line")
    if any("\n" in line or "\r" in line for line in (*sent_commands, *transcript_lines)):
        raise ValueError(f"{label} entries must each be one line")
    _require_sha256(transcript_sha256, label=f"{label} hash")
    expected = _transcript_sha256(sent_commands, transcript_lines)
    if transcript_sha256 != expected:
        raise ValueError(f"{label} hash does not match the recorded transcript")


def history_context_sha256(history: UsiPositionHistory) -> str:
    """Hash the exact game prefix, not merely the board-only SFEN."""

    encoded = (
        history.initial_sfen.encode("utf-8")
        + b"\0"
        + " ".join(history.moves).encode("ascii")
        + b"\0"
        + history.target_sfen.encode("utf-8")
    )
    return hashlib.sha256(encoded).hexdigest()


class CandidateSourceKind(StrEnum):
    """All source kinds are proposal-only and carry no correctness privilege."""

    PROPOSAL_ONLY = "proposal_only"
    TACTICAL_ENUMERATION = "tactical_enumeration"
    ACTUAL_PLAYED_MOVE = "actual_played_move_proposal_only"
    PRINCIPAL_REPLY_PROPOSAL = "principal_reply_proposal_only"


class MatrixResolutionKind(StrEnum):
    EXACT_Q_DISTRIBUTION = "exact_q_distribution"
    INTERVAL_DOMINANCE = "interval_dominance"
    UNRESOLVED = "unresolved"


class MatrixConfidenceKind(StrEnum):
    EXACT = "exact"
    INTERVAL_PROVEN = "interval_proven"
    UNRESOLVED = "unresolved"


class MatrixUnresolvedReason(StrEnum):
    OVERLAPPING_INTERVALS = "overlapping_intervals"
    REPORTED_MATE_UNPROVEN = "reported_mate_requires_independent_proof"
    REQUESTED_BUDGET_NOT_FULFILLED = "requested_node_budget_not_fulfilled"


@dataclass(frozen=True, slots=True)
class ScoreInterval:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.lower)
            or not math.isfinite(self.upper)
            or not -1.0 <= self.lower <= self.upper <= 1.0
        ):
            raise ValueError("score interval must be finite and ordered inside [-1, 1]")

    def negated(self) -> ScoreInterval:
        return ScoreInterval(-self.upper, -self.lower)


@dataclass(frozen=True, slots=True)
class CanonicalScorerIdentity:
    """Pinned scorer/runtime identity required to interpret one matrix row."""

    scorer_id: str
    engine_sha256: str
    evaluation_artifacts_sha256: str
    options: tuple[tuple[str, str], ...]
    startup_transcript_sha256: str
    parser_sha256: str
    scorer_code_sha256: str
    calibration_sha256: str
    threads: int
    hash_mb: int
    multipv: int
    book_enabled: bool
    hash_option_name: str = "Hash"

    def __post_init__(self) -> None:
        if self.scorer_id not in CANONICAL_SCORER_IDS:
            raise ValueError(f"non-canonical scorer identity: {self.scorer_id!r}")
        for label, value in (
            ("engine", self.engine_sha256),
            ("evaluation artifacts", self.evaluation_artifacts_sha256),
            ("startup transcript", self.startup_transcript_sha256),
            ("parser", self.parser_sha256),
            ("scorer code", self.scorer_code_sha256),
            ("calibration", self.calibration_sha256),
        ):
            _require_sha256(value, label=f"{label} hash")
        if self.threads != 1:
            raise ValueError("production canonical scoring requires exactly one thread")
        if self.hash_mb < 1:
            raise ValueError("production canonical scoring hash size must be positive")
        if self.multipv != 1:
            raise ValueError("production canonical scoring requires MultiPV=1")
        if self.book_enabled:
            raise ValueError("production canonical scoring must disable the opening book")
        if self.hash_option_name not in {"Hash", "USI_Hash"}:
            raise ValueError("hash option name must be exactly Hash or USI_Hash")
        normalized_names: set[str] = set()
        normalized_options: dict[str, tuple[str, str]] = {}
        for name, value in self.options:
            if not name or name.strip() != name or "\n" in name or "\r" in name:
                raise ValueError("scorer option names must be non-empty single-line strings")
            if "\n" in value or "\r" in value:
                raise ValueError("scorer option values must be single-line strings")
            normalized = name.casefold()
            if normalized in normalized_names:
                raise ValueError(f"duplicate scorer option ignoring case: {name!r}")
            normalized_names.add(normalized)
            normalized_options[normalized] = (name, value)
        if self.options != tuple(sorted(self.options, key=lambda item: item[0].casefold())):
            raise ValueError("scorer options must be sorted by case-insensitive name")
        hash_options = normalized_names & {"hash", "usi_hash"}
        if hash_options != {self.hash_option_name.casefold()}:
            raise ValueError("scorer options must contain exactly the declared hash option")
        raw_hash_name, raw_hash_value = normalized_options[self.hash_option_name.casefold()]
        if raw_hash_name != self.hash_option_name:
            raise ValueError("scorer hash option spelling must exactly match hash_option_name")
        try:
            recorded_hash_mb = int(raw_hash_value)
        except ValueError as error:
            raise ValueError("scorer hash option must be an integer") from error
        if recorded_hash_mb != self.hash_mb:
            raise ValueError("scorer hash option does not match hash_mb")
        if normalized_options.get("threads") != ("Threads", str(self.threads)):
            raise ValueError("scorer options must pin the canonical Threads spelling and value")
        if normalized_options.get("multipv") != ("MultiPV", "1"):
            raise ValueError("scorer options must pin MultiPV=1")

    @property
    def identity_sha256(self) -> str:
        return _json_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class CanonicalProposalIdentity:
    """Pinned identity of a proposal-only MultiPV process.

    This identity is deliberately distinct from :class:`CanonicalScorerIdentity`:
    its process runs ``MultiPV=K`` only to discover candidates.  Its scores and
    ranks are never copied into the final value target.
    """

    scorer_id: str
    engine_sha256: str
    evaluation_artifacts_sha256: str
    options: tuple[tuple[str, str], ...]
    startup_transcript_sha256: str
    parser_sha256: str
    proposer_code_sha256: str
    threads: int
    hash_mb: int
    multipv: int
    book_enabled: bool
    hash_option_name: str = "Hash"
    role: str = PROPOSAL_ROLE

    def __post_init__(self) -> None:
        if self.scorer_id not in CANONICAL_SCORER_IDS:
            raise ValueError(f"non-canonical proposal identity: {self.scorer_id!r}")
        for label, value in (
            ("engine", self.engine_sha256),
            ("evaluation artifacts", self.evaluation_artifacts_sha256),
            ("startup transcript", self.startup_transcript_sha256),
            ("parser", self.parser_sha256),
            ("proposer code", self.proposer_code_sha256),
        ):
            _require_sha256(value, label=f"{label} hash")
        if self.threads != 1:
            raise ValueError("canonical proposal generation requires exactly one thread")
        if self.hash_mb < 1:
            raise ValueError("canonical proposal hash size must be positive")
        if self.multipv < 1:
            raise ValueError("canonical proposal MultiPV must be positive")
        if self.book_enabled:
            raise ValueError("canonical proposal generation must disable the opening book")
        if self.hash_option_name not in {"Hash", "USI_Hash"}:
            raise ValueError("hash option name must be exactly Hash or USI_Hash")
        if self.role != PROPOSAL_ROLE:
            raise ValueError("canonical proposal identity has the wrong non-truth role")
        normalized_options: dict[str, tuple[str, str]] = {}
        for name, value in self.options:
            if not name or name.strip() != name or "\n" in name or "\r" in name:
                raise ValueError("proposal option names must be non-empty single-line strings")
            if "\n" in value or "\r" in value:
                raise ValueError("proposal option values must be single-line strings")
            normalized = name.casefold()
            if normalized in normalized_options:
                raise ValueError(f"duplicate proposal option ignoring case: {name!r}")
            normalized_options[normalized] = (name, value)
        if self.options != tuple(sorted(self.options, key=lambda item: item[0].casefold())):
            raise ValueError("proposal options must be sorted by case-insensitive name")
        hash_options = set(normalized_options) & {"hash", "usi_hash"}
        if hash_options != {self.hash_option_name.casefold()}:
            raise ValueError("proposal options must contain exactly the declared hash option")
        raw_hash_name, raw_hash_value = normalized_options[self.hash_option_name.casefold()]
        if raw_hash_name != self.hash_option_name:
            raise ValueError("proposal hash option spelling must exactly match hash_option_name")
        try:
            recorded_hash_mb = int(raw_hash_value)
        except ValueError as error:
            raise ValueError("proposal hash option must be an integer") from error
        if recorded_hash_mb != self.hash_mb:
            raise ValueError("proposal hash option does not match hash_mb")
        if normalized_options.get("threads") != ("Threads", str(self.threads)):
            raise ValueError("proposal options must pin the canonical Threads spelling and value")
        if normalized_options.get("multipv") != ("MultiPV", str(self.multipv)):
            raise ValueError("proposal options must pin the declared MultiPV value")

    @property
    def identity_sha256(self) -> str:
        return _json_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class CandidateProposalEvidence:
    """Lossless provenance for one proposal-only candidate source."""

    source_id: str
    producer: str
    source_kind: CandidateSourceKind
    target_sfen: str
    history_context_sha256: str
    moves: tuple[str, ...]
    requested_nodes: int
    reported_nodes: int
    multipv: int
    sent_commands: tuple[str, ...]
    transcript_lines: tuple[str, ...]
    transcript_sha256: str
    producer_identity_sha256: str
    producer_code_sha256: str

    def __post_init__(self) -> None:
        if not self.source_id or self.source_id.strip() != self.source_id:
            raise ValueError("candidate source ID must be non-empty and trimmed")
        if not self.producer or self.producer.strip() != self.producer:
            raise ValueError("candidate producer must be non-empty and trimmed")
        if not self.target_sfen:
            raise ValueError("candidate proposal target SFEN must not be empty")
        _require_sha256(self.history_context_sha256, label="candidate history hash")
        _validate_transcript(
            self.sent_commands,
            self.transcript_lines,
            self.transcript_sha256,
            label="candidate transcript",
        )
        _require_sha256(
            self.producer_identity_sha256,
            label="candidate producer identity hash",
        )
        _require_sha256(self.producer_code_sha256, label="candidate producer code hash")
        if self.moves != tuple(sorted(set(self.moves))):
            raise ValueError("candidate proposal moves must be unique and sorted")
        if self.requested_nodes < 0 or self.reported_nodes < 0 or self.multipv < 0:
            raise ValueError("candidate proposal node and MultiPV counts must be non-negative")


@dataclass(frozen=True, slots=True)
class AuxiliaryProposalAuthorization:
    """Rights/provenance binding for one optional auxiliary proposer."""

    source_id: str
    rights_id: str
    rights_reviewed_at: str
    distillation_scope: str
    rights_registry_identity_sha256: str
    proposal_identity_sha256: str

    def __post_init__(self) -> None:
        if self.source_id not in AUXILIARY_PROPOSER_IDS or self.rights_id != self.source_id:
            raise ValueError("auxiliary proposal source must equal one registered rights ID")
        if (
            not self.rights_reviewed_at
            or self.rights_reviewed_at.strip() != self.rights_reviewed_at
        ):
            raise ValueError("auxiliary rights review date must be non-empty and trimmed")
        if self.distillation_scope not in {
            "public_release_allowed",
            "local_authorized_only",
        }:
            raise ValueError("auxiliary proposal rights do not authorize distillation")
        _require_sha256(
            self.rights_registry_identity_sha256,
            label="auxiliary rights registry identity",
        )
        _require_sha256(
            self.proposal_identity_sha256,
            label="auxiliary proposal identity",
        )


@dataclass(frozen=True, slots=True)
class RawSearchScore:
    """One calibrated scalar while retaining its original USI score domain."""

    q_value: float
    score_kind: TeacherScoreKind
    bound: TeacherScoreBound
    score_cp: int | None = None
    mate_plies: int | None = None
    mate_unknown_sign: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.q_value) or not -1.0 <= self.q_value <= 1.0:
            raise ValueError("search Q must be finite in [-1, 1]")
        if self.score_kind is TeacherScoreKind.CENTIPAWN:
            if (
                self.score_cp is None
                or self.mate_plies is not None
                or self.mate_unknown_sign is not None
            ):
                raise ValueError("centipawn search score fields are inconsistent")
            return
        if self.score_cp is not None or (self.mate_plies is None) == (
            self.mate_unknown_sign is None
        ):
            raise ValueError("mate search score fields are inconsistent")
        if self.mate_unknown_sign not in {None, -1, 1}:
            raise ValueError("mate_unknown_sign must be -1 or 1")
        raw_sign = (
            self.mate_unknown_sign
            if self.mate_unknown_sign is not None
            else 1
            if self.mate_plies is not None and self.mate_plies >= 0
            else -1
        )
        if self.q_value != float(raw_sign):
            raise ValueError("mate Q must retain the raw mate sign without point averaging")

    @property
    def interval(self) -> ScoreInterval:
        if self.bound is TeacherScoreBound.EXACT:
            return ScoreInterval(self.q_value, self.q_value)
        if self.bound is TeacherScoreBound.LOWER:
            return ScoreInterval(self.q_value, 1.0)
        if self.bound is TeacherScoreBound.UPPER:
            return ScoreInterval(-1.0, self.q_value)
        raise AssertionError(f"unsupported score bound: {self.bound!r}")


@dataclass(frozen=True, slots=True)
class BranchSearchEvidence:
    """One and only one constrained root branch at a declared node budget."""

    scorer_id: str
    target_sfen: str
    history_context_sha256: str
    searchmove: str
    requested_nodes: int
    reported_nodes: int
    multipv: int
    raw_score_perspective: str
    score: RawSearchScore
    depth: int
    seldepth: int
    time_ms: int
    nps: int
    pv: tuple[str, ...]
    sent_commands: tuple[str, ...]
    transcript_lines: tuple[str, ...]
    transcript_sha256: str

    def __post_init__(self) -> None:
        if self.scorer_id not in CANONICAL_SCORER_IDS:
            raise ValueError("branch evidence has a non-canonical scorer ID")
        if not self.target_sfen or not self.searchmove:
            raise ValueError("branch evidence requires target SFEN and searchmove")
        _require_sha256(self.history_context_sha256, label="branch history hash")
        _validate_transcript(
            self.sent_commands,
            self.transcript_lines,
            self.transcript_sha256,
            label="branch transcript",
        )
        if self.requested_nodes < 1 or self.reported_nodes < 0:
            raise ValueError("branch node counts are invalid")
        if self.multipv != 1:
            raise ValueError("every production branch search must use MultiPV=1")
        if self.raw_score_perspective != RAW_SCORE_PERSPECTIVE:
            raise ValueError("branch raw score perspective must be side-to-move")
        if min(self.depth, self.seldepth, self.time_ms, self.nps) < 0:
            raise ValueError("branch search metrics must be non-negative")
        if not self.pv or self.pv[0] != self.searchmove:
            raise ValueError("branch PV must begin with its constrained searchmove")
        if not self.sent_commands or self.sent_commands[0] != SEARCH_STATE_RESET_COMMAND:
            raise ValueError(
                "every independent branch search must record a preceding usinewgame reset"
            )
        if self.sent_commands.count(SEARCH_STATE_RESET_COMMAND) != 1:
            raise ValueError("branch evidence must record exactly one search-state reset")

    @property
    def requested_budget_fulfilled(self) -> bool:
        return self.reported_nodes >= self.requested_nodes


@runtime_checkable
class ProductionCanonicalScorer(Protocol):
    """Adapter boundary for real external-USI or deterministic fake scorers."""

    @property
    def identity(self) -> CanonicalScorerIdentity: ...

    @property
    def proposal_identity(self) -> CanonicalProposalIdentity: ...

    @property
    def proposal_process_separate(self) -> bool: ...

    def propose_replies(
        self,
        history: UsiPositionHistory,
        *,
        legal_moves: tuple[str, ...],
        requested_nodes: int,
    ) -> CandidateProposalEvidence: ...

    def score_searchmove(
        self,
        history: UsiPositionHistory,
        *,
        move: str,
        requested_nodes: int,
    ) -> BranchSearchEvidence: ...


class ExternalUsiProductionScorer:
    """Adapt one rights-gated external USI engine to production matrix evidence.

    The scoring engine must be configured with ``MultiPV=1``.  Production
    candidate generation uses a distinct engine instance configured with
    ``MultiPV=K``.  The proposal process can widen the candidate union, but only
    the single selected anchor's equal-node ``MultiPV=1`` searches produce
    final values.
    """

    def __init__(
        self,
        engine: ExternalUsiTeacher,
        identity: CanonicalScorerIdentity,
        *,
        proposal_engine: ExternalUsiTeacher | None = None,
        proposal_identity: CanonicalProposalIdentity | None = None,
    ) -> None:
        if engine.policy.policy_id != identity.scorer_id:
            raise ValueError("external scorer policy ID does not match canonical identity")
        if engine.multipv != 1 or identity.multipv != 1:
            raise ValueError("production external scorer requires MultiPV=1")
        if not engine.training_use:
            raise PermissionError("production external scorer must be rights-opened for training")
        configured_options = {name.casefold(): str(value) for name, value in engine.options.items()}
        configured_options["multipv"] = str(engine.multipv)
        declared_options = {name.casefold(): value for name, value in identity.options}
        if configured_options != declared_options:
            raise ValueError(
                "canonical identity options do not exactly match the USI scorer configuration"
            )
        if configured_options.get("threads") != str(identity.threads):
            raise ValueError("production scorer must explicitly configure its recorded Threads")
        if configured_options.get(identity.hash_option_name.casefold()) != str(identity.hash_mb):
            raise ValueError("production scorer must explicitly configure its recorded hash option")
        self.engine = engine
        self._identity = identity
        if (proposal_engine is None) != (proposal_identity is None):
            raise ValueError("proposal engine and proposal identity must be supplied together")
        self.proposal_engine = engine if proposal_engine is None else proposal_engine
        if proposal_identity is None:
            proposal_identity = CanonicalProposalIdentity(
                scorer_id=identity.scorer_id,
                engine_sha256=identity.engine_sha256,
                evaluation_artifacts_sha256=identity.evaluation_artifacts_sha256,
                options=identity.options,
                startup_transcript_sha256=identity.startup_transcript_sha256,
                parser_sha256=identity.parser_sha256,
                proposer_code_sha256=identity.scorer_code_sha256,
                threads=identity.threads,
                hash_mb=identity.hash_mb,
                multipv=identity.multipv,
                book_enabled=identity.book_enabled,
                hash_option_name=identity.hash_option_name,
            )
        if self.proposal_engine.policy.policy_id != proposal_identity.scorer_id:
            raise ValueError("proposal engine policy ID does not match proposal identity")
        if proposal_identity.scorer_id != identity.scorer_id:
            raise ValueError("proposal and scoring identities must refer to the same teacher")
        if (
            proposal_identity.engine_sha256 != identity.engine_sha256
            or proposal_identity.evaluation_artifacts_sha256 != identity.evaluation_artifacts_sha256
            or proposal_identity.parser_sha256 != identity.parser_sha256
        ):
            raise ValueError(
                "proposal and scoring processes must pin the same engine, evaluation, and parser"
            )
        if self.proposal_engine.multipv != proposal_identity.multipv:
            raise ValueError("proposal engine MultiPV does not match its pinned identity")
        if not self.proposal_engine.training_use:
            raise PermissionError("proposal engine must be rights-opened for training")
        proposal_options = {
            name.casefold(): str(value) for name, value in self.proposal_engine.options.items()
        }
        proposal_options["multipv"] = str(self.proposal_engine.multipv)
        declared_proposal_options = {
            name.casefold(): value for name, value in proposal_identity.options
        }
        if proposal_options != declared_proposal_options:
            raise ValueError("proposal identity options do not exactly match the proposal process")
        if proposal_options.get("threads") != str(proposal_identity.threads):
            raise ValueError("proposal process must explicitly configure its recorded Threads")
        if proposal_options.get(proposal_identity.hash_option_name.casefold()) != str(
            proposal_identity.hash_mb
        ):
            raise ValueError("proposal process must explicitly configure its recorded hash option")
        if proposal_identity.hash_mb != identity.hash_mb:
            raise ValueError("proposal and scoring processes must use the same hash size")
        if proposal_identity.hash_option_name != identity.hash_option_name:
            raise ValueError("proposal and scoring processes must pin the same hash option name")
        self._proposal_identity = proposal_identity

    @property
    def identity(self) -> CanonicalScorerIdentity:
        return self._identity

    @property
    def proposal_identity(self) -> CanonicalProposalIdentity:
        return self._proposal_identity

    @property
    def proposal_process_separate(self) -> bool:
        return self.proposal_engine is not self.engine

    def close(self) -> None:
        """Close both role-separated USI processes without double-closing."""

        if self.proposal_engine is not self.engine:
            self.proposal_engine.close()
        self.engine.close()

    @staticmethod
    def _transcript(analysis: UsiAnalysis) -> tuple[tuple[str, ...], tuple[str, ...], str]:
        sent_commands = (SEARCH_STATE_RESET_COMMAND, *analysis.sent_commands)
        transcript_lines = analysis.transcript_lines
        if not sent_commands or not transcript_lines:
            raise ValueError("USI analysis omitted the lossless search transcript")
        digest = _transcript_sha256(sent_commands, transcript_lines)
        return sent_commands, transcript_lines, digest

    @staticmethod
    def _raw_score(variation: TeacherVariation, *, value_scale: float) -> RawSearchScore:
        score_kind = variation.score_kind
        bound = variation.bound
        score_cp = variation.score_cp
        mate_plies = variation.mate_plies
        mate_unknown_sign = variation.mate_unknown_sign
        if score_kind is TeacherScoreKind.CENTIPAWN:
            assert isinstance(score_cp, int)
            q_value = math.tanh(score_cp / value_scale)
        else:
            sign = (
                mate_unknown_sign
                if mate_unknown_sign is not None
                else 1
                if mate_plies is not None and mate_plies >= 0
                else -1
            )
            q_value = float(sign)
        return RawSearchScore(
            q_value=q_value,
            score_kind=score_kind,
            bound=bound,
            score_cp=score_cp,
            mate_plies=mate_plies,
            mate_unknown_sign=mate_unknown_sign,
        )

    def propose_replies(
        self,
        history: UsiPositionHistory,
        *,
        legal_moves: tuple[str, ...],
        requested_nodes: int,
    ) -> CandidateProposalEvidence:
        legal = set(legal_moves)
        if not legal:
            raise ValueError("reply proposal requires at least one legal move")
        self.proposal_engine.new_game()
        analysis = self.proposal_engine.analyse_with_history_searchmoves(
            history, legal_moves, nodes=requested_nodes
        )
        proposed = tuple(
            sorted({variation.move for variation in analysis.candidates if variation.move in legal})
        )
        if not proposed and analysis.bestmove in legal:
            proposed = (analysis.bestmove,)
        sent, lines, digest = self._transcript(analysis)
        return CandidateProposalEvidence(
            source_id=self.identity.scorer_id,
            producer="simajilord_shogi.production_score_matrix.ExternalUsiProductionScorer",
            source_kind=CandidateSourceKind.PRINCIPAL_REPLY_PROPOSAL,
            target_sfen=history.target_sfen,
            history_context_sha256=history_context_sha256(history),
            moves=proposed,
            requested_nodes=requested_nodes,
            reported_nodes=analysis.nodes or 0,
            multipv=self.proposal_identity.multipv,
            sent_commands=sent,
            transcript_lines=lines,
            transcript_sha256=digest,
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
        self.engine.new_game()
        analysis = self.engine.analyse_with_history_searchmoves(
            history, (move,), nodes=requested_nodes
        )
        matching = tuple(variation for variation in analysis.candidates if variation.move == move)
        if len(matching) != 1:
            raise RuntimeError("constrained USI search did not return one scored branch")
        variation = matching[0]
        sent, lines, digest = self._transcript(analysis)
        return BranchSearchEvidence(
            scorer_id=self.identity.scorer_id,
            target_sfen=history.target_sfen,
            history_context_sha256=history_context_sha256(history),
            searchmove=move,
            requested_nodes=requested_nodes,
            reported_nodes=analysis.nodes or 0,
            multipv=1,
            raw_score_perspective=RAW_SCORE_PERSPECTIVE,
            score=self._raw_score(variation, value_scale=self.engine.value_scale),
            depth=analysis.depth or 0,
            seldepth=analysis.seldepth or 0,
            time_ms=analysis.time_ms or 0,
            nps=analysis.nps or 0,
            pv=variation.pv,
            sent_commands=sent,
            transcript_lines=lines,
            transcript_sha256=digest,
        )


@dataclass(frozen=True, slots=True)
class HistoryReceipt:
    initial_sfen: str
    moves: tuple[str, ...]
    target_sfen: str
    history_context_sha256: str


@dataclass(frozen=True, slots=True)
class RootCandidateEvidence:
    move: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReplyUnionEvidence:
    root_move: str
    child_sfen: str
    child_history_context_sha256: str
    terminal_root_value: float | None
    terminal_reason: str | None
    proposals: tuple[CandidateProposalEvidence, ...]
    reply_moves: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScorerCandidateEvidence:
    move: str
    root_search: BranchSearchEvidence
    direct_root_interval: ScoreInterval
    reply_searches: tuple[BranchSearchEvidence, ...]
    reply_backed_interval: ScoreInterval
    final_interval: ScoreInterval


@dataclass(frozen=True, slots=True)
class MatrixConfidence:
    kind: MatrixConfidenceKind
    candidate_coverage_complete: bool
    reply_coverage_complete: bool
    all_requested_budgets_fulfilled: bool
    all_search_scores_exact: bool
    proof_margin: float | None
    unproven_mate_reported: bool


@dataclass(frozen=True, slots=True)
class ScorerMatrixResolution:
    scorer_id: str
    kind: MatrixResolutionKind
    best_moves: tuple[str, ...]
    chosen_move: str | None
    value_interval: ScoreInterval | None
    unresolved: bool
    additional_search_required: bool
    unresolved_reasons: tuple[MatrixUnresolvedReason, ...]
    confidence: MatrixConfidence


@dataclass(frozen=True, slots=True)
class CanonicalScorerMatrixReceipt:
    scorer: CanonicalScorerIdentity
    candidates: tuple[ScorerCandidateEvidence, ...]
    resolution: ScorerMatrixResolution


@dataclass(frozen=True, slots=True)
class ProductionScoreMatrixReceipt:
    schema: str
    builder: str
    builder_code_sha256: str
    history: HistoryReceipt
    requested_nodes_per_searchmove: int
    reply_proposal_nodes: int
    experiment_arm_id: str
    scoring_anchor_id: str
    proposal_source_ids: tuple[str, ...]
    auxiliary_proposal_source_ids: tuple[str, ...]
    proposal_identities: tuple[CanonicalProposalIdentity, ...]
    auxiliary_proposal_authorizations: tuple[AuxiliaryProposalAuthorization, ...]
    proposal_role: str
    scoring_role: str
    search_state_reset_mode: str
    candidate_sources: tuple[CandidateProposalEvidence, ...]
    candidates: tuple[RootCandidateEvidence, ...]
    reply_unions: tuple[ReplyUnionEvidence, ...]
    scorer_matrices: tuple[CanonicalScorerMatrixReceipt, ...]
    position_mode: str
    qsearch_leaf_re_evaluated: bool
    cross_teacher_value_average: bool
    majority_vote: bool
    proposal_membership_is_truth: bool
    local_only: bool
    publication_allowed: bool

    def __post_init__(self) -> None:
        if self.schema != PRODUCTION_SCORE_MATRIX_SCHEMA:
            raise ValueError("production score-matrix receipt schema mismatch")
        if self.builder != PRODUCTION_SCORE_MATRIX_BUILDER:
            raise ValueError("production score-matrix builder identity mismatch")
        _require_sha256(self.builder_code_sha256, label="builder code hash")
        _require_sha256(
            self.history.history_context_sha256,
            label="receipt history context hash",
        )
        if self.requested_nodes_per_searchmove < 1 or self.reply_proposal_nodes < 1:
            raise ValueError("production receipt node budgets must be positive")
        expected_sources = EXPERIMENT_ARM_PROPOSAL_SOURCES.get(self.experiment_arm_id)
        if expected_sources is None or self.proposal_source_ids != expected_sources:
            raise ValueError("experiment arm does not match its canonical proposal sources")
        expected_auxiliary_sources = EXPERIMENT_ARM_AUXILIARY_SOURCES.get(self.experiment_arm_id)
        if expected_auxiliary_sources != self.auxiliary_proposal_source_ids:
            raise ValueError("experiment arm does not match its auxiliary proposal sources")
        if self.scoring_anchor_id not in CANONICAL_SCORER_IDS:
            raise ValueError("production receipt has a non-canonical scoring anchor")
        if self.proposal_role != PROPOSAL_ROLE or self.scoring_role != SCORING_ROLE:
            raise ValueError("production receipt role separation is not explicit")
        if self.search_state_reset_mode != SEARCH_STATE_RESET_MODE:
            raise ValueError("production receipt has an unsupported search-state reset mode")
        if tuple(identity.scorer_id for identity in self.proposal_identities) != (
            self.proposal_source_ids
        ):
            raise ValueError("proposal identities do not match the fixed proposal-source order")
        if (
            tuple(
                authorization.source_id for authorization in self.auxiliary_proposal_authorizations
            )
            != self.auxiliary_proposal_source_ids
        ):
            raise ValueError(
                "auxiliary proposal authorizations do not match the fixed source order"
            )
        if any(identity.multipv < 2 for identity in self.proposal_identities):
            raise ValueError("production proposal processes require MultiPV >= 2")
        source_ids = tuple(source.source_id for source in self.candidate_sources)
        if source_ids != tuple(sorted(set(source_ids))):
            raise ValueError("production candidate sources must be unique and sorted")
        if len({source_id.casefold() for source_id in source_ids}) != len(source_ids):
            raise ValueError("production candidate sources must be unique ignoring case")
        candidate_moves = tuple(candidate.move for candidate in self.candidates)
        if not candidate_moves or candidate_moves != tuple(sorted(set(candidate_moves))):
            raise ValueError("production receipt candidates must be non-empty and sorted")
        if tuple(reply.root_move for reply in self.reply_unions) != candidate_moves:
            raise ValueError("reply-union order must exactly match the root candidate order")
        if tuple(matrix.scorer.scorer_id for matrix in self.scorer_matrices) != (
            self.scoring_anchor_id,
        ):
            raise ValueError("production receipt must contain exactly one anchor score matrix")
        canonical_candidate_sources = tuple(
            source_id for source_id in CANONICAL_SCORER_IDS if source_id in source_ids
        )
        if canonical_candidate_sources != self.proposal_source_ids:
            raise ValueError("candidate sources do not match the experiment's fixed proposers")
        proposal_by_id = {identity.scorer_id: identity for identity in self.proposal_identities}
        for source in self.candidate_sources:
            proposal_identity = proposal_by_id.get(source.source_id)
            if proposal_identity is None:
                continue
            if source.producer_identity_sha256 != proposal_identity.identity_sha256:
                raise ValueError("canonical proposal is not bound to its proposal-only identity")
            if source.multipv != proposal_identity.multipv:
                raise ValueError("canonical proposal MultiPV differs from its process identity")
        authorization_by_id = {
            authorization.source_id: authorization
            for authorization in self.auxiliary_proposal_authorizations
        }
        observed_auxiliary_sources = tuple(
            source_id for source_id in AUXILIARY_PROPOSER_IDS if source_id in source_ids
        )
        if observed_auxiliary_sources != self.auxiliary_proposal_source_ids:
            raise ValueError("receipt candidate sources omit or add an auxiliary proposer")
        for source in self.candidate_sources:
            authorization = authorization_by_id.get(source.source_id)
            if authorization is not None and (
                source.producer_identity_sha256 != authorization.proposal_identity_sha256
            ):
                raise ValueError(
                    "auxiliary proposal is not bound to its authorized artifact identity"
                )
        if self.position_mode != "exact_history_root_with_one_ply_reply_reanalysis":
            raise ValueError("unsupported production score-matrix position mode")
        if self.qsearch_leaf_re_evaluated:
            raise ValueError("receipt v1 cannot claim unimplemented qsearch-leaf re-evaluation")
        if (
            self.cross_teacher_value_average
            or self.majority_vote
            or self.proposal_membership_is_truth
        ):
            raise ValueError(
                "production receipt cannot upgrade proposal or committee votes to truth"
            )
        if not self.local_only or self.publication_allowed:
            raise ValueError("canonical production score matrices must remain private local-only")

    def _unsigned_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def receipt_sha256(self) -> str:
        return _json_sha256(self._unsigned_dict())

    def to_dict(self) -> dict[str, object]:
        payload = self._unsigned_dict()
        payload["receipt_sha256"] = self.receipt_sha256
        return payload

    def write_create_only(self, path: Path) -> None:
        """Atomically create one immutable JSON receipt without overwriting."""

        destination = Path(os.path.abspath(os.fspath(path.expanduser())))
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialized = (
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary_path, destination)
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite production score-matrix receipt: {destination}"
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def tactical_legal_moves(board: Board) -> tuple[str, ...]:
    """Return legal checks, captures, promotions, or every legal check evasion."""

    legal_moves = tuple(board.legal_moves())
    if board.is_in_check():
        return tuple(sorted(move.to_usi() for move in legal_moves))
    tactical: set[str] = set()
    for move in legal_moves:
        is_capture = move.is_normal() and not board.is_square_empty(move.to_sq)
        if is_capture or move.is_promotion():
            tactical.add(move.to_usi())
            continue
        child = board.copy()
        child.apply_move(move)
        if child.is_in_check():
            tactical.add(move.to_usi())
    return tuple(sorted(tactical))


def _validate_legal_move(board: Board, move_usi: str, *, label: str) -> str:
    try:
        move = Move.from_usi(move_usi)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{label} is not valid USI: {move_usi!r}") from error
    if not board.is_legal_move(move):
        raise ValueError(f"illegal {label} {move_usi!r} at {board.to_sfen()}")
    return move.to_usi()


def _validate_pv(board: Board, evidence: BranchSearchEvidence) -> None:
    pv_board = board.copy()
    for ply, move_usi in enumerate(evidence.pv):
        move = _validate_legal_move(pv_board, move_usi, label=f"PV[{ply}]")
        pv_board.apply_move(Move.from_usi(move))


def _validate_proposal(
    proposal: CandidateProposalEvidence,
    *,
    history: UsiPositionHistory,
    board: Board,
    expected_source_id: str | None = None,
    expected_requested_nodes: int | None = None,
    expected_identity_sha256: str | None = None,
) -> None:
    if expected_source_id is not None and proposal.source_id != expected_source_id:
        raise ValueError(
            f"proposal source mismatch: expected={expected_source_id!r} "
            f"observed={proposal.source_id!r}"
        )
    if proposal.target_sfen != history.target_sfen:
        raise ValueError("candidate proposal SFEN does not match its validated history")
    if proposal.history_context_sha256 != history_context_sha256(history):
        raise ValueError("candidate proposal history hash does not match")
    if (
        expected_requested_nodes is not None
        and proposal.requested_nodes != expected_requested_nodes
    ):
        raise ValueError("reply proposal did not use the requested common node budget")
    if (
        expected_identity_sha256 is not None
        and proposal.producer_identity_sha256 != expected_identity_sha256
    ):
        raise ValueError("candidate proposal is not bound to the expected producer identity")
    if expected_identity_sha256 is not None:
        if not proposal.sent_commands or proposal.sent_commands[0] != SEARCH_STATE_RESET_COMMAND:
            raise ValueError("canonical proposal must record a preceding usinewgame reset")
        if proposal.sent_commands.count(SEARCH_STATE_RESET_COMMAND) != 1:
            raise ValueError("canonical proposal must record exactly one search-state reset")
    for move in proposal.moves:
        _validate_legal_move(board, move, label=f"{proposal.source_id} proposal")


def _validate_branch(
    evidence: BranchSearchEvidence,
    *,
    scorer: CanonicalScorerIdentity,
    history: UsiPositionHistory,
    board: Board,
    move: str,
    requested_nodes: int,
) -> None:
    if evidence.scorer_id != scorer.scorer_id:
        raise ValueError("branch search scorer identity mismatch")
    if evidence.target_sfen != history.target_sfen:
        raise ValueError("branch search SFEN does not match its validated history")
    if evidence.history_context_sha256 != history_context_sha256(history):
        raise ValueError("branch search history hash does not match")
    if evidence.searchmove != move:
        raise ValueError("branch search returned a different searchmove")
    if evidence.requested_nodes != requested_nodes:
        raise ValueError("branch search did not use the common requested node budget")
    if history.position_command() not in evidence.sent_commands:
        raise ValueError("branch transcript omits the exact history position command")
    expected_go = f"go nodes {requested_nodes} searchmoves {move}"
    if expected_go not in evidence.sent_commands:
        raise ValueError("branch transcript omits the exact constrained go command")
    _validate_legal_move(board, move, label="searchmove")
    _validate_pv(board, evidence)


def _terminal_root_value(board: Board, *, root_turn: int) -> tuple[float, str] | None:
    adjudication = adjudicate_board(board)
    if adjudication is None:
        return None
    if adjudication.winner is None:
        return 0.0, adjudication.termination.value
    value = 1.0 if adjudication.winner == root_turn else -1.0
    return value, adjudication.termination.value


def _backup_reply_intervals(
    reply_searches: Sequence[BranchSearchEvidence],
) -> ScoreInterval:
    if not reply_searches:
        raise ValueError("negamax/min reply backup requires at least one scored reply")
    root_intervals = tuple(search.score.interval.negated() for search in reply_searches)
    return ScoreInterval(
        min(interval.lower for interval in root_intervals),
        min(interval.upper for interval in root_intervals),
    )


def _module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class ProductionScoreMatrixBuilder:
    """Build a provenance-complete score matrix without touching a trainer."""

    def __init__(
        self,
        scorers: Sequence[ProductionCanonicalScorer],
        *,
        requested_nodes: int,
        reply_proposal_nodes: int,
        interval_dominance_margin: float = 0.0,
        experiment_arm_id: str = "B",
        scoring_anchor_id: str = CANONICAL_SCORER_IDS[0],
        proposal_source_ids: tuple[str, ...] = CANONICAL_SCORER_IDS,
        auxiliary_proposal_source_ids: tuple[str, ...] = (),
        auxiliary_proposal_authorizations: tuple[AuxiliaryProposalAuthorization, ...] = (),
    ) -> None:
        if requested_nodes < 1 or reply_proposal_nodes < 1:
            raise ValueError("production score-matrix node budgets must be positive")
        if not math.isfinite(interval_dominance_margin) or interval_dominance_margin < 0.0:
            raise ValueError("interval dominance margin must be finite and non-negative")
        expected_proposal_sources = EXPERIMENT_ARM_PROPOSAL_SOURCES.get(experiment_arm_id)
        if expected_proposal_sources is None:
            raise ValueError(f"unsupported production experiment arm: {experiment_arm_id!r}")
        if proposal_source_ids != expected_proposal_sources:
            raise ValueError("proposal_source_ids do not match the declared experiment arm")
        expected_auxiliary_sources = EXPERIMENT_ARM_AUXILIARY_SOURCES[experiment_arm_id]
        if auxiliary_proposal_source_ids != expected_auxiliary_sources:
            raise ValueError(
                "auxiliary_proposal_source_ids do not match the declared experiment arm"
            )
        if (
            tuple(authorization.source_id for authorization in auxiliary_proposal_authorizations)
            != auxiliary_proposal_source_ids
        ):
            raise ValueError(
                "auxiliary proposal authorizations must exactly match the declared sources"
            )
        if scoring_anchor_id not in CANONICAL_SCORER_IDS:
            raise ValueError("scoring anchor must be one canonical teacher")
        by_id = {scorer.identity.scorer_id: scorer for scorer in scorers}
        if len(by_id) != len(scorers):
            raise ValueError("canonical production scorer IDs must be unique")
        required_adapters = set(proposal_source_ids) | {scoring_anchor_id}
        if set(by_id) != required_adapters:
            raise ValueError("production adapters must exactly equal the anchor/proposer union")
        ordered = tuple(
            by_id[scorer_id] for scorer_id in CANONICAL_SCORER_IDS if scorer_id in by_id
        )
        hash_sizes = {scorer.identity.hash_mb for scorer in ordered}
        if len(hash_sizes) != 1:
            raise ValueError("canonical scorers must use the same numeric hash size")
        proposal_scorers = tuple(by_id[scorer_id] for scorer_id in proposal_source_ids)
        if any(not scorer.proposal_process_separate for scorer in proposal_scorers):
            raise ValueError("production proposal generation requires separate USI processes")
        if any(scorer.proposal_identity.multipv < 2 for scorer in proposal_scorers):
            raise ValueError("production proposal processes require MultiPV >= 2")
        if any(
            scorer.proposal_identity.hash_mb != scorer.identity.hash_mb
            for scorer in proposal_scorers
        ):
            raise ValueError("proposal/scoring process hash sizes must match")
        self.scorers = ordered
        self.scorers_by_id = by_id
        self.anchor = by_id[scoring_anchor_id]
        self.proposal_scorers = proposal_scorers
        self.requested_nodes = requested_nodes
        self.reply_proposal_nodes = reply_proposal_nodes
        self.interval_dominance_margin = interval_dominance_margin
        self.experiment_arm_id = experiment_arm_id
        self.scoring_anchor_id = scoring_anchor_id
        self.proposal_source_ids = proposal_source_ids
        self.auxiliary_proposal_source_ids = auxiliary_proposal_source_ids
        self.auxiliary_proposal_authorizations = auxiliary_proposal_authorizations

    def build(
        self,
        history: UsiPositionHistory,
        candidate_proposals: Sequence[CandidateProposalEvidence],
    ) -> ProductionScoreMatrixReceipt:
        board = history.target_board()
        if adjudicate_board(board) is not None:
            raise ValueError("cannot build a score matrix for a terminal root")
        root_history_hash = history_context_sha256(history)
        provided: dict[str, CandidateProposalEvidence] = {}
        for proposal in candidate_proposals:
            if proposal.source_id == TACTICAL_CANDIDATE_SOURCE_ID:
                raise ValueError("tactical candidates are generated internally")
            folded_source_ids = {source_id.casefold(): source_id for source_id in provided}
            if proposal.source_id.casefold() in folded_source_ids:
                raise ValueError(f"duplicate candidate source ID: {proposal.source_id!r}")
            permitted_sources = {
                *self.proposal_source_ids,
                *self.auxiliary_proposal_source_ids,
                METEO_CANDIDATE_SOURCE_ID,
                ACTUAL_PLAYED_CANDIDATE_SOURCE_ID,
            }
            if proposal.source_id not in permitted_sources:
                raise ValueError(
                    f"candidate source is outside the experiment arm: {proposal.source_id!r}"
                )
            matching_scorer = self.scorers_by_id.get(proposal.source_id)
            _validate_proposal(
                proposal,
                history=history,
                board=board,
                expected_identity_sha256=(
                    None
                    if matching_scorer is None
                    else matching_scorer.proposal_identity.identity_sha256
                ),
            )
            if proposal.source_id in self.auxiliary_proposal_source_ids and (
                not proposal.sent_commands
                or proposal.sent_commands[0] != SEARCH_STATE_RESET_COMMAND
                or proposal.sent_commands.count(SEARCH_STATE_RESET_COMMAND) != 1
            ):
                raise ValueError(
                    "auxiliary proposal must record exactly one preceding usinewgame reset"
                )
            required_root_proposer_ids = (*self.proposal_source_ids, METEO_CANDIDATE_SOURCE_ID)
            if proposal.source_id in required_root_proposer_ids and (
                proposal.source_kind is not CandidateSourceKind.PROPOSAL_ONLY
            ):
                raise ValueError("required root sources must remain proposal-only evidence")
            if proposal.source_id == ACTUAL_PLAYED_CANDIDATE_SOURCE_ID and (
                proposal.source_kind is not CandidateSourceKind.ACTUAL_PLAYED_MOVE
            ):
                raise ValueError("the actual played move source must use its proposal-only kind")
            if proposal.source_id != ACTUAL_PLAYED_CANDIDATE_SOURCE_ID and (
                proposal.source_kind is CandidateSourceKind.ACTUAL_PLAYED_MOVE
            ):
                raise ValueError("actual-played proposal kind is reserved for its source ID")
            provided[proposal.source_id] = proposal
        unexpected_canonical = sorted(
            (set(provided) & set(CANONICAL_SCORER_IDS)) - set(self.proposal_source_ids)
        )
        if unexpected_canonical:
            raise ValueError(
                f"candidate sources include teachers outside the experiment arm: "
                f"{unexpected_canonical!r}"
            )
        missing = sorted(
            set(
                (
                    *self.proposal_source_ids,
                    *self.auxiliary_proposal_source_ids,
                    METEO_CANDIDATE_SOURCE_ID,
                )
            )
            - set(provided)
        )
        if missing:
            raise ValueError(f"missing required root candidate sources: {missing!r}")

        builder_hash = _module_sha256()
        tactical_moves = tactical_legal_moves(board)
        tactical_payload = {
            "algorithm": "checks-captures-promotions-evasions-v1",
            "target_sfen": history.target_sfen,
            "history_context_sha256": root_history_hash,
            "moves": tactical_moves,
        }
        tactical_commands = ("enumerate tactical legal moves",)
        tactical_lines = (json.dumps(tactical_payload, sort_keys=True, separators=(",", ":")),)
        tactical = CandidateProposalEvidence(
            source_id=TACTICAL_CANDIDATE_SOURCE_ID,
            producer="simajilord_shogi.production_score_matrix.tactical_legal_moves",
            source_kind=CandidateSourceKind.TACTICAL_ENUMERATION,
            target_sfen=history.target_sfen,
            history_context_sha256=root_history_hash,
            moves=tactical_moves,
            requested_nodes=0,
            reported_nodes=0,
            multipv=0,
            sent_commands=tactical_commands,
            transcript_lines=tactical_lines,
            transcript_sha256=_transcript_sha256(tactical_commands, tactical_lines),
            producer_identity_sha256=builder_hash,
            producer_code_sha256=builder_hash,
        )
        all_sources = tuple(
            sorted((*provided.values(), tactical), key=lambda proposal: proposal.source_id)
        )
        candidate_moves = tuple(
            sorted({move for proposal in all_sources for move in proposal.moves})
        )
        if not candidate_moves:
            raise ValueError("production candidate union must not be empty")
        root_candidates = tuple(
            RootCandidateEvidence(
                move=move,
                source_ids=tuple(
                    proposal.source_id for proposal in all_sources if move in proposal.moves
                ),
            )
            for move in candidate_moves
        )

        root_searches: dict[str, dict[str, BranchSearchEvidence]] = {}
        for scorer in (self.anchor,):
            scorer_searches: dict[str, BranchSearchEvidence] = {}
            for move in candidate_moves:
                search = scorer.score_searchmove(
                    history,
                    move=move,
                    requested_nodes=self.requested_nodes,
                )
                _validate_branch(
                    search,
                    scorer=scorer.identity,
                    history=history,
                    board=board,
                    move=move,
                    requested_nodes=self.requested_nodes,
                )
                scorer_searches[move] = search
            root_searches[scorer.identity.scorer_id] = scorer_searches

        reply_unions: list[ReplyUnionEvidence] = []
        child_histories: dict[str, UsiPositionHistory] = {}
        for root_move in candidate_moves:
            child = board.copy()
            child.apply_move(Move.from_usi(root_move))
            child_history = UsiPositionHistory(
                initial_sfen=history.initial_sfen,
                moves=(*history.moves, root_move),
                target_sfen=child.to_sfen(),
            )
            child_histories[root_move] = child_history
            terminal = _terminal_root_value(child, root_turn=board.turn.value)
            if terminal is not None:
                value, reason = terminal
                reply_unions.append(
                    ReplyUnionEvidence(
                        root_move=root_move,
                        child_sfen=child.to_sfen(),
                        child_history_context_sha256=history_context_sha256(child_history),
                        terminal_root_value=value,
                        terminal_reason=reason,
                        proposals=(),
                        reply_moves=(),
                    )
                )
                continue
            legal_replies = tuple(sorted(move.to_usi() for move in child.legal_moves()))
            if not legal_replies:
                raise AssertionError("nonterminal candidate child has no legal reply")
            proposals: list[CandidateProposalEvidence] = []
            for scorer in self.proposal_scorers:
                proposal = scorer.propose_replies(
                    child_history,
                    legal_moves=legal_replies,
                    requested_nodes=self.reply_proposal_nodes,
                )
                _validate_proposal(
                    proposal,
                    history=child_history,
                    board=child,
                    expected_source_id=scorer.identity.scorer_id,
                    expected_requested_nodes=self.reply_proposal_nodes,
                    expected_identity_sha256=scorer.proposal_identity.identity_sha256,
                )
                if proposal.source_kind is not CandidateSourceKind.PRINCIPAL_REPLY_PROPOSAL:
                    raise ValueError("canonical reply proposal has the wrong source kind")
                if not proposal.moves:
                    raise ValueError("every canonical scorer must propose a principal reply")
                proposals.append(proposal)
            reply_moves = {move for proposal in proposals for move in proposal.moves}
            for scorer in (self.anchor,):
                root_pv = root_searches[scorer.identity.scorer_id][root_move].pv
                if len(root_pv) > 1:
                    reply_moves.add(_validate_legal_move(child, root_pv[1], label="root PV reply"))
            reply_unions.append(
                ReplyUnionEvidence(
                    root_move=root_move,
                    child_sfen=child.to_sfen(),
                    child_history_context_sha256=history_context_sha256(child_history),
                    terminal_root_value=None,
                    terminal_reason=None,
                    proposals=tuple(proposals),
                    reply_moves=tuple(sorted(reply_moves)),
                )
            )

        reply_by_move = {reply.root_move: reply for reply in reply_unions}
        matrices: list[CanonicalScorerMatrixReceipt] = []
        reply_proposal_budgets_fulfilled = all(
            proposal.reported_nodes >= proposal.requested_nodes
            for reply_union in reply_unions
            for proposal in reply_union.proposals
        )
        for scorer in (self.anchor,):
            candidates: list[ScorerCandidateEvidence] = []
            for root_move in candidate_moves:
                root_search = root_searches[scorer.identity.scorer_id][root_move]
                reply_union = reply_by_move[root_move]
                if reply_union.terminal_root_value is not None:
                    backed = ScoreInterval(
                        reply_union.terminal_root_value,
                        reply_union.terminal_root_value,
                    )
                    reply_searches: tuple[BranchSearchEvidence, ...] = ()
                else:
                    child_history = child_histories[root_move]
                    child = child_history.target_board()
                    searches: list[BranchSearchEvidence] = []
                    for reply in reply_union.reply_moves:
                        search = scorer.score_searchmove(
                            child_history,
                            move=reply,
                            requested_nodes=self.requested_nodes,
                        )
                        _validate_branch(
                            search,
                            scorer=scorer.identity,
                            history=child_history,
                            board=child,
                            move=reply,
                            requested_nodes=self.requested_nodes,
                        )
                        searches.append(search)
                    reply_searches = tuple(searches)
                    backed = _backup_reply_intervals(reply_searches)
                candidates.append(
                    ScorerCandidateEvidence(
                        move=root_move,
                        root_search=root_search,
                        direct_root_interval=root_search.score.interval,
                        reply_searches=reply_searches,
                        reply_backed_interval=backed,
                        final_interval=backed,
                    )
                )
            ordered_candidates = tuple(candidates)
            matrices.append(
                CanonicalScorerMatrixReceipt(
                    scorer=scorer.identity,
                    candidates=ordered_candidates,
                    resolution=self._resolve_matrix(
                        scorer.identity.scorer_id,
                        ordered_candidates,
                        reply_proposal_budgets_fulfilled=reply_proposal_budgets_fulfilled,
                    ),
                )
            )

        return ProductionScoreMatrixReceipt(
            schema=PRODUCTION_SCORE_MATRIX_SCHEMA,
            builder=PRODUCTION_SCORE_MATRIX_BUILDER,
            builder_code_sha256=builder_hash,
            history=HistoryReceipt(
                initial_sfen=history.initial_sfen,
                moves=history.moves,
                target_sfen=history.target_sfen,
                history_context_sha256=root_history_hash,
            ),
            requested_nodes_per_searchmove=self.requested_nodes,
            reply_proposal_nodes=self.reply_proposal_nodes,
            experiment_arm_id=self.experiment_arm_id,
            scoring_anchor_id=self.scoring_anchor_id,
            proposal_source_ids=self.proposal_source_ids,
            auxiliary_proposal_source_ids=self.auxiliary_proposal_source_ids,
            proposal_identities=tuple(scorer.proposal_identity for scorer in self.proposal_scorers),
            auxiliary_proposal_authorizations=self.auxiliary_proposal_authorizations,
            proposal_role=PROPOSAL_ROLE,
            scoring_role=SCORING_ROLE,
            search_state_reset_mode=SEARCH_STATE_RESET_MODE,
            candidate_sources=all_sources,
            candidates=root_candidates,
            reply_unions=tuple(reply_unions),
            scorer_matrices=tuple(matrices),
            position_mode="exact_history_root_with_one_ply_reply_reanalysis",
            qsearch_leaf_re_evaluated=False,
            cross_teacher_value_average=False,
            majority_vote=False,
            proposal_membership_is_truth=False,
            local_only=True,
            publication_allowed=False,
        )

    def _resolve_matrix(
        self,
        scorer_id: str,
        candidates: tuple[ScorerCandidateEvidence, ...],
        *,
        reply_proposal_budgets_fulfilled: bool,
    ) -> ScorerMatrixResolution:
        searches = tuple(
            search
            for candidate in candidates
            for search in (candidate.root_search, *candidate.reply_searches)
        )
        all_budget_fulfilled = reply_proposal_budgets_fulfilled and all(
            search.requested_budget_fulfilled for search in searches
        )
        all_exact = all(search.score.bound is TeacherScoreBound.EXACT for search in searches)
        unproven_mate = any(search.score.score_kind is TeacherScoreKind.MATE for search in searches)
        reasons: set[MatrixUnresolvedReason] = set()
        if not all_budget_fulfilled:
            reasons.add(MatrixUnresolvedReason.REQUESTED_BUDGET_NOT_FULFILLED)
        if unproven_mate:
            reasons.add(MatrixUnresolvedReason.REPORTED_MATE_UNPROVEN)

        intervals = {candidate.move: candidate.final_interval for candidate in candidates}
        all_points = all(interval.lower == interval.upper for interval in intervals.values())
        dominant: str | None = None
        proof_margin: float | None = None
        if len(candidates) == 1:
            dominant = candidates[0].move
            proof_margin = 2.0
        else:
            for move, interval in intervals.items():
                other_upper = max(
                    other.upper for other_move, other in intervals.items() if other_move != move
                )
                margin = interval.lower - other_upper
                if margin > self.interval_dominance_margin:
                    if dominant is not None:
                        raise AssertionError("two score intervals unexpectedly both dominate")
                    dominant = move
                    proof_margin = margin

        if not reasons and all_points:
            maximum = max(interval.lower for interval in intervals.values())
            best_moves = tuple(
                move
                for move, interval in intervals.items()
                if math.isclose(interval.lower, maximum, rel_tol=0.0, abs_tol=1e-12)
            )
            if len(intervals) > 1:
                ordered = sorted((interval.lower for interval in intervals.values()), reverse=True)
                proof_margin = ordered[0] - ordered[1]
            confidence = MatrixConfidence(
                kind=MatrixConfidenceKind.EXACT,
                candidate_coverage_complete=True,
                reply_coverage_complete=True,
                all_requested_budgets_fulfilled=True,
                all_search_scores_exact=all_exact,
                proof_margin=proof_margin,
                unproven_mate_reported=False,
            )
            return ScorerMatrixResolution(
                scorer_id=scorer_id,
                kind=MatrixResolutionKind.EXACT_Q_DISTRIBUTION,
                best_moves=best_moves,
                chosen_move=best_moves[0],
                value_interval=ScoreInterval(maximum, maximum),
                unresolved=False,
                additional_search_required=False,
                unresolved_reasons=(),
                confidence=confidence,
            )

        if not reasons and dominant is not None:
            confidence = MatrixConfidence(
                kind=MatrixConfidenceKind.INTERVAL_PROVEN,
                candidate_coverage_complete=True,
                reply_coverage_complete=True,
                all_requested_budgets_fulfilled=True,
                all_search_scores_exact=all_exact,
                proof_margin=proof_margin,
                unproven_mate_reported=False,
            )
            return ScorerMatrixResolution(
                scorer_id=scorer_id,
                kind=MatrixResolutionKind.INTERVAL_DOMINANCE,
                best_moves=(dominant,),
                chosen_move=dominant,
                value_interval=intervals[dominant],
                unresolved=False,
                additional_search_required=False,
                unresolved_reasons=(),
                confidence=confidence,
            )

        if not reasons:
            reasons.add(MatrixUnresolvedReason.OVERLAPPING_INTERVALS)
        value_interval = ScoreInterval(
            max(interval.lower for interval in intervals.values()),
            max(interval.upper for interval in intervals.values()),
        )
        confidence = MatrixConfidence(
            kind=MatrixConfidenceKind.UNRESOLVED,
            candidate_coverage_complete=True,
            reply_coverage_complete=True,
            all_requested_budgets_fulfilled=all_budget_fulfilled,
            all_search_scores_exact=all_exact,
            proof_margin=proof_margin,
            unproven_mate_reported=unproven_mate,
        )
        return ScorerMatrixResolution(
            scorer_id=scorer_id,
            kind=MatrixResolutionKind.UNRESOLVED,
            best_moves=(),
            chosen_move=None,
            value_interval=value_interval,
            unresolved=True,
            additional_search_required=True,
            unresolved_reasons=tuple(sorted(reasons, key=str)),
            confidence=confidence,
        )
