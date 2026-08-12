"""Transport-independent game and training records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class Termination(StrEnum):
    CHECKMATE = "checkmate"
    RESIGNATION = "resignation"
    REPETITION = "repetition"
    DECLARATION = "entering_king_declaration"
    MAX_PLIES = "max_plies"
    INTERRUPTION = "interruption"
    IMPASSE = "impasse"
    TIME_FORFEIT = "time_forfeit"
    ILLEGAL_MOVE = "illegal_move"
    AGREED_DRAW = "agreed_draw"
    NO_MATE = "no_mate"


class TeacherScoreKind(StrEnum):
    """The two score domains defined by the USI protocol."""

    CENTIPAWN = "cp"
    MATE = "mate"


class TeacherScoreBound(StrEnum):
    """Whether a teacher score is exact or only a search bound."""

    EXACT = "exact"
    LOWER = "lowerbound"
    UPPER = "upperbound"


@dataclass(frozen=True, slots=True)
class TeacherVariation:
    """One typed, lossless MultiPV root variation from an external teacher.

    Centipawn and mate scores deliberately occupy separate optional fields so a
    mate-distance encoding can never be mistaken for an ordinary evaluation.
    ``mate_unknown_sign`` preserves the USI extensions ``score mate +`` and
    ``score mate -`` when an engine knows the result but not the distance.
    """

    rank: int
    move: str
    score_kind: TeacherScoreKind
    bound: TeacherScoreBound
    pv: tuple[str, ...]
    score_cp: int | None = None
    mate_plies: int | None = None
    mate_unknown_sign: int | None = None

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("teacher variation rank must be positive")
        if not self.pv or self.pv[0] != self.move:
            raise ValueError("teacher variation PV must start with its root move")
        if self.score_kind is TeacherScoreKind.CENTIPAWN:
            if self.score_cp is None or self.mate_plies is not None:
                raise ValueError("centipawn variation must contain only score_cp")
            if self.mate_unknown_sign is not None:
                raise ValueError("centipawn variation cannot contain a mate sign")
            return
        if self.score_cp is not None:
            raise ValueError("mate variation cannot contain score_cp")
        if (self.mate_plies is None) == (self.mate_unknown_sign is None):
            raise ValueError(
                "mate variation must contain either mate_plies or mate_unknown_sign"
            )
        if self.mate_unknown_sign not in {None, -1, 1}:
            raise ValueError("unknown-distance mate sign must be -1 or 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TeacherVariation:
        return cls(
            rank=int(value["rank"]),
            move=str(value["move"]),
            score_kind=TeacherScoreKind(value["score_kind"]),
            bound=TeacherScoreBound(value["bound"]),
            pv=tuple(str(move) for move in value["pv"]),
            score_cp=(None if value.get("score_cp") is None else int(value["score_cp"])),
            mate_plies=(
                None if value.get("mate_plies") is None else int(value["mate_plies"])
            ),
            mate_unknown_sign=(
                None
                if value.get("mate_unknown_sign") is None
                else int(value["mate_unknown_sign"])
            ),
        )


@dataclass(frozen=True, slots=True)
class PositionSample:
    sfen: str
    ply: int
    turn: int
    policy: dict[str, float]
    root_value: float
    value_target: float = 0.0
    teacher_policy: dict[str, float] | None = None
    teacher_value: float | None = None
    policy_reversal: bool = False
    discovery_simulation: int | None = None
    chosen_move: str | None = None
    actor_best_move: str | None = None
    actor_regret: float | None = None
    actor_simulations: int | None = None
    actor_search_seconds: float | None = None
    actor_nps: float | None = None
    actor_source: str | None = None
    actor_peak_tree_nodes: int | None = None
    actor_tree_recycles: int = 0
    actor_resignation_overridden: bool = False
    # Complete root-search evidence.  The implicit policy is derived only when
    # these maps exactly cover every legal move and every move has a real visit.
    # Proven immediate mates are a set-valued target and intentionally leave
    # ``actor_implicit_policy`` empty instead of inventing a dense distribution.
    actor_move_values: dict[str, float] | None = None
    actor_move_visits: dict[str, int] | None = None
    actor_implicit_policy: dict[str, float] | None = None
    actor_proven_mate_moves: tuple[str, ...] = ()
    teacher_best_move: str | None = None
    teacher_regret: float | None = None
    teacher_regret_is_lower_bound: bool = False
    teacher_nodes: int | None = None
    teacher_depth_ratio: float | None = None
    teacher_time_ms: int | None = None
    teacher_nps: int | None = None
    teacher_depth: int | None = None
    teacher_source: str | None = None
    teacher_context: str | None = None
    teacher_move_values: dict[str, float] | None = None
    teacher_move_visits: dict[str, int] | None = None
    teacher_implicit_policy: dict[str, float] | None = None
    teacher_proven_mate_moves: tuple[str, ...] = ()
    teacher_variations: tuple[TeacherVariation, ...] | None = None
    teacher_policy_temperature: float | None = None
    teacher_value_scale: float | None = None
    # Exact full-budget evaluation of the move actually played.  A normal
    # MultiPV row may omit that move, so this evidence comes from a separate
    # USI ``searchmoves <chosen_move>`` branch search and must not be inferred
    # from the worst visible candidate.
    teacher_played_move_value: float | None = None
    teacher_played_move_nodes: int | None = None
    teacher_played_move_time_ms: int | None = None
    teacher_played_move_nps: int | None = None
    teacher_played_move_depth: int | None = None
    teacher_played_move_exact: bool = False
    teacher_played_move_variation: TeacherVariation | None = None
    # In-memory self-play replay annotation.  A value of one is the actionable
    # position immediately before the mating move; terminal boards themselves
    # have no legal policy target and are never added as samples.
    terminal_checkmate_distance: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PositionSample:
        fields = dict(value)
        for field_name in ("actor_proven_mate_moves", "teacher_proven_mate_moves"):
            moves = fields.get(field_name)
            if moves is not None:
                fields[field_name] = tuple(str(move) for move in moves)
        variations = fields.get("teacher_variations")
        if variations is not None:
            fields["teacher_variations"] = tuple(
                TeacherVariation.from_dict(variation) for variation in variations
            )
        played_variation = fields.get("teacher_played_move_variation")
        if played_variation is not None:
            fields["teacher_played_move_variation"] = TeacherVariation.from_dict(
                played_variation
            )
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class GameRecord:
    initial_sfen: str
    moves: tuple[str, ...]
    samples: tuple[PositionSample, ...]
    winner: int | None
    termination: Termination

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["termination"] = self.termination.value
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GameRecord:
        return cls(
            initial_sfen=str(value["initial_sfen"]),
            moves=tuple(value["moves"]),
            samples=tuple(PositionSample.from_dict(item) for item in value["samples"]),
            winner=value["winner"],
            termination=Termination(value["termination"]),
        )
