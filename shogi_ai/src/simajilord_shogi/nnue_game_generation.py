"""Production full-trajectory generation between external USI NNUE engines.

The generator deliberately does not adjudicate from an evaluation or accept a
resignation as a game result.  Engines are asked to continue from a legal PV
when they emit ``bestmove resign``; an unaccompanied resignation fails closed.
Only a board-rule result may create a strong WDL label.  A safety ``max_plies``
cutoff remains useful operationally, but its positions are position sources
rather than strong game-result targets.

Every worker owns two independent USI processes.  That is required even for
self-play because ``usinewgame``, transposition tables, and history state are
process-local.  Opening prefixes are replayed from their original SFEN so
repetition and perpetual-check adjudication retain the complete game history.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from rsshogi.core import Board, Move
from rsshogi.types import Color

from .adjudication import adjudicate_board, can_declare_win_csa27
from .arena import ExternalUsiPlayer
from .domain import GameRecord, PositionSample, Termination
from .external_usi import (
    ExternalTeacherPolicy,
    ExternalUsiTeacher,
    UsiOptionValueVerification,
    UsiPositionHistory,
)
from .model_rights import RightsDecision, model_rights
from .opening_suite import OpeningPosition

NNUE_GAME_GENERATION_SCHEMA = "meteo-nnue-game-generation-v1"
NNUE_GAME_RECEIPT_SCHEMA = "meteo-nnue-game-trajectory-v1"
_HISTORY_HASH_SEPARATOR = b"\0"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _single_line(value: str, *, label: str) -> str:
    if not value or value != value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be a non-empty trimmed single-line string")
    return value


def _history_context_sha256(initial_sfen: str, moves: tuple[str, ...]) -> str:
    encoded = (
        initial_sfen.encode("utf-8") + _HISTORY_HASH_SEPARATOR + " ".join(moves).encode("ascii")
    )
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    """Identity of an executable, NNUE, progress file, or exact engine input."""

    path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _artifact_receipt(path: Path, *, label: str) -> ArtifactReceipt:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {expanded}")
    try:
        resolved = expanded.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} does not exist or is unreadable: {expanded}") from error
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return ArtifactReceipt(
        path=str(resolved),
        size_bytes=resolved.stat().st_size,
        sha256=_sha256_file(resolved),
    )


@dataclass(frozen=True, slots=True)
class NnueGenerationOpening:
    """A canonical opening target plus the exact legal prefix used to reach it."""

    initial_sfen: str
    moves: tuple[str, ...] = ()
    target_sfen: str = field(init=False)
    normalized_target_key: str = field(init=False)
    history_context_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            board = Board(self.initial_sfen)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid generation-opening initial SFEN") from error
        if not board.is_valid():
            raise ValueError(f"invalid generation-opening initial SFEN: {self.initial_sfen}")
        canonical_initial = board.to_sfen()
        canonical_moves: list[str] = []
        for ply, raw_move in enumerate(self.moves):
            terminal = adjudicate_board(board)
            if terminal is not None:
                raise ValueError(
                    "generation-opening prefix continues after a rules-terminal position "
                    f"at ply {ply}: {terminal.termination.value}"
                )
            try:
                move = Move.from_usi(raw_move)
            except (RuntimeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid generation-opening move syntax at ply {ply}: {raw_move}"
                ) from error
            if not board.is_legal_move(move):
                raise ValueError(
                    f"illegal generation-opening move at ply {ply}: {raw_move} at {board.to_sfen()}"
                )
            canonical_moves.append(move.to_usi())
            board.apply_move(move)
        target = OpeningPosition.from_sfen(board.to_sfen())
        moves = tuple(canonical_moves)
        object.__setattr__(self, "initial_sfen", canonical_initial)
        object.__setattr__(self, "moves", moves)
        object.__setattr__(self, "target_sfen", target.sfen)
        object.__setattr__(self, "normalized_target_key", target.normalized_key)
        object.__setattr__(
            self,
            "history_context_sha256",
            _history_context_sha256(canonical_initial, moves),
        )

    @classmethod
    def from_sfen(cls, sfen: str) -> NnueGenerationOpening:
        return cls(initial_sfen=sfen)

    def board_with_history(self) -> Board:
        board = Board(self.initial_sfen)
        for move_usi in self.moves:
            board.apply_move(Move.from_usi(move_usi))
        if board.to_sfen() != self.target_sfen:
            raise AssertionError("validated generation opening changed during replay")
        return board

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_sfen": self.initial_sfen,
            "moves": list(self.moves),
            "target_sfen": self.target_sfen,
            "normalized_target_key": self.normalized_target_key,
            "history_context_sha256": self.history_context_sha256,
        }


@dataclass(frozen=True, slots=True)
class UsiEngineSpec:
    """Immutable, rights-gated recipe for one independently spawned USI engine."""

    actor_source: str
    command: tuple[str, ...]
    policy: ExternalTeacherPolicy
    nodes: int
    options: tuple[tuple[str, str | int], ...] = ()
    timeout_seconds: float = 300.0
    working_directory: Path | None = None
    policy_temperature: float = 200.0
    value_scale: float = 1_200.0
    option_value_verification: UsiOptionValueVerification = UsiOptionValueVerification.NONE
    expected_fatal_startup_diagnostics: tuple[tuple[str, str], ...] = ()
    artifact_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        _single_line(self.actor_source, label="actor_source")
        if not self.command or any(
            not argument or "\n" in argument or "\r" in argument for argument in self.command
        ):
            raise ValueError("USI command arguments must be non-empty single-line strings")
        if self.nodes < 1:
            raise ValueError("USI game-generation nodes must be positive")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("USI timeout must be finite and positive")
        if not math.isfinite(self.policy_temperature) or self.policy_temperature <= 0:
            raise ValueError("USI policy temperature must be finite and positive")
        if not math.isfinite(self.value_scale) or self.value_scale <= 0:
            raise ValueError("USI value scale must be finite and positive")
        normalized_options: set[str] = set()
        for name, value in self.options:
            _single_line(name, label="USI option name")
            if isinstance(value, bool):
                raise TypeError(f"boolean Python value is ambiguous for USI option {name!r}")
            rendered = str(value)
            if "\n" in rendered or "\r" in rendered:
                raise ValueError(f"USI option {name!r} contains a line break")
            normalized = name.casefold()
            if normalized in normalized_options:
                raise ValueError(f"duplicate USI option name ignoring case: {name}")
            if normalized == "multipv":
                raise ValueError("game generation fixes MultiPV=1; omit it from options")
            normalized_options.add(normalized)
        normalized_artifacts = {
            os.path.abspath(os.fspath(path.expanduser())) for path in self.artifact_paths
        }
        if len(normalized_artifacts) != len(self.artifact_paths):
            raise ValueError("duplicate engine artifact path")

    @classmethod
    def from_rights_profile(
        cls,
        *,
        actor_source: str,
        command: tuple[str, ...],
        rights_id: str,
        nodes: int,
        allow_limited_local: bool = False,
        options: tuple[tuple[str, str | int], ...] = (),
        timeout_seconds: float = 300.0,
        working_directory: Path | None = None,
        policy_temperature: float = 200.0,
        value_scale: float = 1_200.0,
        option_value_verification: UsiOptionValueVerification = (UsiOptionValueVerification.NONE),
        expected_fatal_startup_diagnostics: tuple[tuple[str, str], ...] = (),
        artifact_paths: tuple[Path, ...] = (),
    ) -> Self:
        rights = model_rights(rights_id)
        return cls(
            actor_source=actor_source,
            command=command,
            policy=rights.teacher_policy(allow_limited_local=allow_limited_local),
            nodes=nodes,
            options=options,
            timeout_seconds=timeout_seconds,
            working_directory=working_directory,
            policy_temperature=policy_temperature,
            value_scale=value_scale,
            option_value_verification=option_value_verification,
            expected_fatal_startup_diagnostics=expected_fatal_startup_diagnostics,
            artifact_paths=artifact_paths,
        )

    def open_engine(self) -> ExternalUsiTeacher:
        """Create a new process adapter; callers must never share it across workers."""

        return ExternalUsiTeacher(
            list(self.command),
            self.policy,
            nodes=self.nodes,
            multipv=1,
            options=dict(self.options),
            timeout_seconds=self.timeout_seconds,
            training_use=True,
            working_directory=self.working_directory,
            policy_temperature=self.policy_temperature,
            value_scale=self.value_scale,
            option_value_verification=self.option_value_verification,
            expected_fatal_startup_diagnostics=self.expected_fatal_startup_diagnostics,
        )

    def require_game_generation_permission(self) -> None:
        """Enforce both output-distillation and complete-game rights decisions."""

        self.policy.require_training_permission()
        try:
            rights = model_rights(self.policy.policy_id)
        except ValueError:
            # Meteo exports and test doubles have explicit policies but no
            # third-party registry row.  Production external engines should be
            # constructed with ``from_rights_profile``.
            return
        hard_game_allowed = rights.hard_game_training is RightsDecision.ALLOWED or (
            rights.hard_game_training is RightsDecision.LIMITED
            and self.policy.training_outputs_local_only
        )
        if not hard_game_allowed:
            raise PermissionError(
                f"engine {rights.name!r} is not authorized for complete-game training: "
                f"{rights.hard_game_training.value}"
            )

    def _hard_game_rights_receipt(self) -> dict[str, object]:
        try:
            rights = model_rights(self.policy.policy_id)
        except ValueError:
            return {
                "registered": False,
                "decision": "explicit_unregistered_policy",
            }
        return {
            "registered": True,
            "rights_id": rights.rights_id,
            "decision": rights.hard_game_training.value,
            "limited_local_authorized": self.policy.training_outputs_local_only,
        }

    def to_profile_dict(self, artifact_receipts: tuple[ArtifactReceipt, ...]) -> dict[str, object]:
        return {
            "actor_source": self.actor_source,
            "command": list(self.command),
            "rights_policy": asdict(self.policy),
            "hard_game_training_rights": self._hard_game_rights_receipt(),
            "nodes_per_move": self.nodes,
            "multipv": 1,
            "options": {name: value for name, value in self.options},
            "timeout_seconds": self.timeout_seconds,
            "working_directory": (
                None
                if self.working_directory is None
                else str(self.working_directory.expanduser().resolve())
            ),
            "policy_temperature": self.policy_temperature,
            "value_scale": self.value_scale,
            "option_value_verification": self.option_value_verification.value,
            "expected_fatal_startup_diagnostics": [
                list(diagnostic) for diagnostic in self.expected_fatal_startup_diagnostics
            ],
            "artifacts": [receipt.to_dict() for receipt in artifact_receipts],
        }


@dataclass(frozen=True, slots=True)
class NnueGameGenerationConfig:
    """One reproducible color-swapped generation batch."""

    generation_id: str
    openings: tuple[NnueGenerationOpening | str, ...]
    max_plies: int = 1024
    parallelism: int = 1

    def __post_init__(self) -> None:
        _single_line(self.generation_id, label="generation_id")
        if not self.openings:
            raise ValueError("at least one generation opening is required")
        if self.max_plies < 1:
            raise ValueError("max_plies must be positive")
        if self.parallelism < 1:
            raise ValueError("parallelism must be positive")

    def normalized_openings(self) -> tuple[NnueGenerationOpening, ...]:
        openings = tuple(
            opening
            if isinstance(opening, NnueGenerationOpening)
            else NnueGenerationOpening.from_sfen(opening)
            for opening in self.openings
        )
        identities = [opening.history_context_sha256 for opening in openings]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate exact opening history is not allowed")
        return openings


@dataclass(frozen=True, slots=True)
class PlyTrajectoryReceipt:
    """Search and history evidence for one move actually played by an engine."""

    generated_ply: int
    absolute_ply: int
    turn: int
    actor_source: str
    rights_policy_id: str
    sfen: str
    move: str
    root_value: float
    nodes: int | None
    nps: float | None
    search_seconds: float | None
    history_prefix_length: int
    history_context_sha256: str
    resignation_overridden: bool
    termination: Termination
    rules_terminal_trajectory: bool
    strong_value_target: float | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["termination"] = self.termination.value
        payload["history_prefix"] = {
            "source": "game.full_game_moves",
            "length": self.history_prefix_length,
            "sha256": self.history_context_sha256,
        }
        return payload

    def resolve_history_prefix(
        self,
        *,
        initial_sfen: str,
        full_game_moves: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Resolve and verify the compact reference to the game's shared move list."""

        if not 0 <= self.history_prefix_length <= len(full_game_moves):
            raise ValueError("per-ply history-prefix length is outside the full trajectory")
        prefix = full_game_moves[: self.history_prefix_length]
        if _history_context_sha256(initial_sfen, prefix) != self.history_context_sha256:
            raise ValueError("per-ply history-prefix hash does not match the full trajectory")
        return prefix


@dataclass(frozen=True, slots=True)
class GameTrajectoryReceipt:
    """A complete game path and its WDL-label eligibility."""

    schema: str
    game_id: str
    opening_index: int
    color_swap_leg: int
    initial_sfen: str
    opening_prefix_moves: tuple[str, ...]
    starting_sfen: str
    opening_history_context_sha256: str
    black_actor_source: str
    white_actor_source: str
    generated_moves: tuple[str, ...]
    full_game_moves: tuple[str, ...]
    final_sfen: str
    plies: tuple[PlyTrajectoryReceipt, ...]
    winner: int | None
    termination: Termination
    rules_terminal_trajectory: bool
    safety_truncated: bool
    strong_wdl_label_allowed: bool
    strong_wdl_label_black: float | None
    game_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "game_id": self.game_id,
            "opening_index": self.opening_index,
            "color_swap_leg": self.color_swap_leg,
            "initial_sfen": self.initial_sfen,
            "opening_prefix_moves": list(self.opening_prefix_moves),
            "starting_sfen": self.starting_sfen,
            "opening_history_context_sha256": self.opening_history_context_sha256,
            "black_actor_source": self.black_actor_source,
            "white_actor_source": self.white_actor_source,
            "generated_moves": list(self.generated_moves),
            "full_game_moves": list(self.full_game_moves),
            "final_sfen": self.final_sfen,
            "plies": [ply.to_dict() for ply in self.plies],
            "winner": self.winner,
            "termination": self.termination.value,
            "rules_terminal_trajectory": self.rules_terminal_trajectory,
            "safety_truncated": self.safety_truncated,
            "strong_wdl_label_allowed": self.strong_wdl_label_allowed,
            "strong_wdl_label_black": self.strong_wdl_label_black,
            "game_seconds": self.game_seconds,
        }


@dataclass(frozen=True, slots=True)
class ActorSpeedMetrics:
    actor_source: str
    positions: int
    nodes: int
    search_seconds: float
    nodes_per_search_second: float
    reported_nps_median: float | None
    reported_nps_p95: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GenerationSpeedMetrics:
    requested_parallelism: int
    effective_parallelism: int
    peak_concurrent_games: int
    games: int
    rules_complete_games: int
    incomplete_games: int
    positions: int
    wall_seconds: float
    game_seconds_median: float
    game_seconds_p95: float
    move_search_seconds_median: float
    move_search_seconds_p95: float
    games_per_hour: float
    positions_per_second: float
    actors: tuple[ActorSpeedMetrics, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["actors"] = [actor.to_dict() for actor in self.actors]
        return payload


@dataclass(frozen=True, slots=True)
class EngineStartupReceipt:
    worker_index: int
    engine_slot: str
    actor_source: str
    rights_policy_id: str
    executable: ArtifactReceipt
    startup_provenance: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "worker_index": self.worker_index,
            "engine_slot": self.engine_slot,
            "actor_source": self.actor_source,
            "rights_policy_id": self.rights_policy_id,
            "executable": self.executable.to_dict(),
            "startup_provenance": self.startup_provenance,
        }


@dataclass(frozen=True, slots=True)
class NnueGameGenerationReceipt:
    schema: str
    generation_id: str
    started_at: str
    finished_at: str
    no_resignation_adjudication: bool
    no_evaluation_adjudication: bool
    rules_adjudicator: str
    max_plies_is_not_a_strong_wdl_label: bool
    max_plies: int
    engine_profiles: tuple[dict[str, object], ...]
    engine_startups: tuple[EngineStartupReceipt, ...]
    openings: tuple[NnueGenerationOpening, ...]
    games: tuple[GameTrajectoryReceipt, ...]
    metrics: GenerationSpeedMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "generation_id": self.generation_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "no_resignation_adjudication": self.no_resignation_adjudication,
            "no_evaluation_adjudication": self.no_evaluation_adjudication,
            "rules_adjudicator": self.rules_adjudicator,
            "max_plies_is_not_a_strong_wdl_label": (self.max_plies_is_not_a_strong_wdl_label),
            "max_plies": self.max_plies,
            "engine_profiles": list(self.engine_profiles),
            "engine_startups": [startup.to_dict() for startup in self.engine_startups],
            "openings": [opening.to_dict() for opening in self.openings],
            "games": [game.to_dict() for game in self.games],
            "metrics": self.metrics.to_dict(),
        }

    def write_create_only(self, path: Path) -> None:
        """Atomically publish one self-hashed JSON receipt without replacement."""

        destination = Path(os.path.abspath(os.fspath(path.expanduser())))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite generation receipt: {destination}")
        payload = self.to_dict()
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
        payload["payload_sha256_scope"] = "receipt_without_payload_sha256_fields"
        encoded = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise FileExistsError(
                    f"refusing to overwrite generation receipt: {destination}"
                ) from error
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class GeneratedNnueGame:
    """A position-source record coupled to its mandatory eligibility receipt."""

    record: GameRecord
    receipt: GameTrajectoryReceipt


@dataclass(frozen=True, slots=True)
class NnueGameGenerationResult:
    games: tuple[GeneratedNnueGame, ...]
    receipt: NnueGameGenerationReceipt

    def position_source_records(self) -> tuple[GameRecord, ...]:
        """Return every replay, including safety-truncated position sources."""

        return tuple(game.record for game in self.games)

    def strong_wdl_records(self) -> tuple[GameRecord, ...]:
        """Return only games whose result was established by the board rules."""

        return tuple(game.record for game in self.games if game.receipt.strong_wdl_label_allowed)


@dataclass(frozen=True, slots=True)
class _PendingPly:
    generated_ply: int
    absolute_ply: int
    turn: int
    actor_source: str
    rights_policy_id: str
    sfen: str
    move: str
    root_value: float
    nodes: int | None
    nps: float | None
    search_seconds: float | None
    history_prefix_length: int
    history_context_sha256: str
    resignation_overridden: bool


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    games: tuple[GeneratedNnueGame, ...]
    startups: tuple[EngineStartupReceipt, ...]


class _ConcurrencyGauge:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)

    def leave(self) -> None:
        with self._lock:
            self._active -= 1
            if self._active < 0:
                raise AssertionError("game concurrency gauge underflow")


def _strong_wdl_label_black(winner: int | None) -> float:
    if winner is None:
        return 0.5
    return 1.0 if winner == Color.BLACK.value else 0.0


def _sample_value_target(turn: int, winner: int | None) -> float:
    if winner is None:
        return 0.0
    return 1.0 if turn == winner else -1.0


def _validate_search_telemetry(
    *, value: float, nodes: int | None, nps: float | None, seconds: float | None
) -> None:
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("USI engine produced an invalid calibrated root value")
    if nodes is not None and nodes < 0:
        raise ValueError("USI engine reported negative nodes")
    if nps is not None and (not math.isfinite(nps) or nps < 0):
        raise ValueError("USI engine reported invalid NPS")
    if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
        raise ValueError("USI engine search duration is invalid")


def _final_board(record: GameRecord) -> Board:
    board = Board(record.initial_sfen)
    for ply, move_usi in enumerate(record.moves):
        move = Move.from_usi(move_usi)
        if not board.is_legal_move(move):
            raise ValueError(f"generated replay contains illegal move at ply {ply}: {move_usi}")
        board.apply_move(move)
    return board


def _game_id(
    *,
    generation_id: str,
    opening: NnueGenerationOpening,
    opening_index: int,
    color_swap_leg: int,
    black_source: str,
    white_source: str,
) -> str:
    fields = (
        generation_id,
        opening.history_context_sha256,
        str(opening_index),
        str(color_swap_leg),
        black_source,
        white_source,
    )
    encoded = _HISTORY_HASH_SEPARATOR.join(item.encode() for item in fields)
    return hashlib.sha256(encoded).hexdigest()


def _play_generated_game(
    *,
    generation_id: str,
    opening: NnueGenerationOpening,
    opening_index: int,
    color_swap_leg: int,
    black_engine: ExternalUsiTeacher,
    black_spec: UsiEngineSpec,
    white_engine: ExternalUsiTeacher,
    white_spec: UsiEngineSpec,
    max_plies: int,
    gauge: _ConcurrencyGauge,
) -> GeneratedNnueGame:
    board = opening.board_with_history()
    moves = list(opening.moves)
    generated_moves: list[str] = []
    pending_samples: list[PositionSample] = []
    pending_receipts: list[_PendingPly] = []
    players = {
        Color.BLACK.value: (
            ExternalUsiPlayer(black_engine, continue_after_resign=True),
            black_spec,
        ),
        Color.WHITE.value: (
            ExternalUsiPlayer(white_engine, continue_after_resign=True),
            white_spec,
        ),
    }
    winner: int | None = None
    termination = Termination.MAX_PLIES
    rules_terminal = False
    started = time.monotonic()
    gauge.enter()
    try:
        for generated_ply in range(max_plies):
            adjudication = adjudicate_board(board)
            if adjudication is not None:
                winner = adjudication.winner
                termination = adjudication.termination
                rules_terminal = True
                break
            turn = board.turn.value
            player, spec = players[turn]
            history = UsiPositionHistory(
                initial_sfen=opening.initial_sfen,
                moves=tuple(moves),
                target_sfen=board.to_sfen(),
            )
            decision = player.choose_move_with_history(board, history)
            if decision.move == "resign":
                raise RuntimeError(
                    "no-resignation game generator received an unoverridden resignation"
                )
            if decision.move == "win":
                if not can_declare_win_csa27(board):
                    raise ValueError(
                        "external engine claimed an illegal CSARule27 entering-king win: "
                        f"source={spec.actor_source!r}, sfen={board.to_sfen()}"
                    )
                winner = turn
                termination = Termination.DECLARATION
                rules_terminal = True
                break
            move = Move.from_usi(decision.move)
            if not board.is_legal_move(move):
                raise ValueError(
                    f"engine {spec.actor_source!r} selected illegal move "
                    f"{decision.move} at {board.to_sfen()}"
                )
            _validate_search_telemetry(
                value=decision.value,
                nodes=decision.nodes,
                nps=decision.nps,
                seconds=decision.elapsed_seconds,
            )
            absolute_ply = len(moves)
            pending_samples.append(
                PositionSample(
                    sfen=board.to_sfen(),
                    ply=absolute_ply,
                    turn=turn,
                    policy=decision.policy,
                    root_value=decision.value,
                    chosen_move=decision.move,
                    actor_best_move=decision.move,
                    actor_simulations=decision.nodes,
                    actor_search_seconds=decision.elapsed_seconds,
                    actor_nps=decision.nps,
                    actor_source=spec.actor_source,
                    actor_peak_tree_nodes=decision.peak_tree_nodes,
                    actor_tree_recycles=decision.tree_recycles,
                    actor_resignation_overridden=decision.resignation_overridden,
                )
            )
            pending_receipts.append(
                _PendingPly(
                    generated_ply=generated_ply,
                    absolute_ply=absolute_ply,
                    turn=turn,
                    actor_source=spec.actor_source,
                    rights_policy_id=spec.policy.policy_id,
                    sfen=board.to_sfen(),
                    move=decision.move,
                    root_value=decision.value,
                    nodes=decision.nodes,
                    nps=decision.nps,
                    search_seconds=decision.elapsed_seconds,
                    history_prefix_length=len(history.moves),
                    history_context_sha256=_history_context_sha256(
                        history.initial_sfen, history.moves
                    ),
                    resignation_overridden=decision.resignation_overridden,
                )
            )
            board.apply_move(move)
            moves.append(decision.move)
            generated_moves.append(decision.move)
        else:
            # A mating or repetition move may be the final permitted move.  The
            # shared direct-game helper checks at loop entry, so production
            # generation explicitly re-adjudicates the final board at the cap.
            adjudication = adjudicate_board(board)
            if adjudication is None:
                winner = None
                termination = Termination.MAX_PLIES
                rules_terminal = False
            else:
                winner = adjudication.winner
                termination = adjudication.termination
                rules_terminal = True
    finally:
        gauge.leave()
    elapsed = time.monotonic() - started

    samples = tuple(
        replace(
            sample,
            value_target=(_sample_value_target(sample.turn, winner) if rules_terminal else 0.0),
        )
        for sample in pending_samples
    )
    record = GameRecord(
        initial_sfen=opening.initial_sfen,
        moves=tuple(moves),
        samples=samples,
        winner=winner,
        termination=termination,
    )
    final_board = _final_board(record)
    final_adjudication = adjudicate_board(final_board)
    if rules_terminal:
        if final_adjudication is None:
            raise AssertionError("rules-terminal generated game has a non-terminal final board")
        if final_adjudication.winner != winner or final_adjudication.termination is not termination:
            raise AssertionError("generated terminal receipt disagrees with board adjudication")
    elif final_adjudication is not None:
        raise AssertionError("safety-truncated game ended on an unrecorded rules result")

    strong_wdl = _strong_wdl_label_black(winner) if rules_terminal else None
    ply_receipts = tuple(
        PlyTrajectoryReceipt(
            generated_ply=pending.generated_ply,
            absolute_ply=pending.absolute_ply,
            turn=pending.turn,
            actor_source=pending.actor_source,
            rights_policy_id=pending.rights_policy_id,
            sfen=pending.sfen,
            move=pending.move,
            root_value=pending.root_value,
            nodes=pending.nodes,
            nps=pending.nps,
            search_seconds=pending.search_seconds,
            history_prefix_length=pending.history_prefix_length,
            history_context_sha256=pending.history_context_sha256,
            resignation_overridden=pending.resignation_overridden,
            termination=termination,
            rules_terminal_trajectory=rules_terminal,
            strong_value_target=(
                _sample_value_target(pending.turn, winner) if rules_terminal else None
            ),
        )
        for pending in pending_receipts
    )
    receipt = GameTrajectoryReceipt(
        schema=NNUE_GAME_RECEIPT_SCHEMA,
        game_id=_game_id(
            generation_id=generation_id,
            opening=opening,
            opening_index=opening_index,
            color_swap_leg=color_swap_leg,
            black_source=black_spec.actor_source,
            white_source=white_spec.actor_source,
        ),
        opening_index=opening_index,
        color_swap_leg=color_swap_leg,
        initial_sfen=opening.initial_sfen,
        opening_prefix_moves=opening.moves,
        starting_sfen=opening.target_sfen,
        opening_history_context_sha256=opening.history_context_sha256,
        black_actor_source=black_spec.actor_source,
        white_actor_source=white_spec.actor_source,
        generated_moves=tuple(generated_moves),
        full_game_moves=tuple(moves),
        final_sfen=final_board.to_sfen(),
        plies=ply_receipts,
        winner=winner,
        termination=termination,
        rules_terminal_trajectory=rules_terminal,
        safety_truncated=not rules_terminal,
        strong_wdl_label_allowed=rules_terminal,
        strong_wdl_label_black=strong_wdl,
        game_seconds=elapsed,
    )
    if len(record.samples) != len(receipt.plies):
        raise AssertionError("sample and per-ply receipt counts diverged")
    return GeneratedNnueGame(record=record, receipt=receipt)


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _actor_speed_metrics(
    games: tuple[GeneratedNnueGame, ...],
) -> tuple[ActorSpeedMetrics, ...]:
    grouped: dict[str, list[PlyTrajectoryReceipt]] = {}
    for game in games:
        for ply in game.receipt.plies:
            grouped.setdefault(ply.actor_source, []).append(ply)
    result: list[ActorSpeedMetrics] = []
    for source, plies in sorted(grouped.items()):
        nodes = sum(ply.nodes or 0 for ply in plies)
        search_seconds = sum(ply.search_seconds or 0.0 for ply in plies)
        reported_nps = tuple(ply.nps for ply in plies if ply.nps is not None)
        result.append(
            ActorSpeedMetrics(
                actor_source=source,
                positions=len(plies),
                nodes=nodes,
                search_seconds=search_seconds,
                nodes_per_search_second=(nodes / search_seconds if search_seconds > 0 else 0.0),
                reported_nps_median=(statistics.median(reported_nps) if reported_nps else None),
                reported_nps_p95=(_percentile(reported_nps, 0.95) if reported_nps else None),
            )
        )
    return tuple(result)


def _speed_metrics(
    games: tuple[GeneratedNnueGame, ...],
    *,
    requested_parallelism: int,
    effective_parallelism: int,
    peak_concurrent_games: int,
    wall_seconds: float,
) -> GenerationSpeedMetrics:
    if not games:
        raise AssertionError("generation metrics require at least one game")
    game_seconds = tuple(game.receipt.game_seconds for game in games)
    search_seconds = tuple(
        ply.search_seconds
        for game in games
        for ply in game.receipt.plies
        if ply.search_seconds is not None
    )
    positions = sum(len(game.receipt.plies) for game in games)
    complete = sum(game.receipt.rules_terminal_trajectory for game in games)
    return GenerationSpeedMetrics(
        requested_parallelism=requested_parallelism,
        effective_parallelism=effective_parallelism,
        peak_concurrent_games=peak_concurrent_games,
        games=len(games),
        rules_complete_games=complete,
        incomplete_games=len(games) - complete,
        positions=positions,
        wall_seconds=wall_seconds,
        game_seconds_median=statistics.median(game_seconds),
        game_seconds_p95=_percentile(game_seconds, 0.95),
        move_search_seconds_median=(statistics.median(search_seconds) if search_seconds else 0.0),
        move_search_seconds_p95=_percentile(search_seconds, 0.95),
        games_per_hour=len(games) * 3600.0 / wall_seconds,
        positions_per_second=positions / wall_seconds,
        actors=_actor_speed_metrics(games),
    )


class NnueGameGenerator:
    """Generate paired external-USI games and a create-only production receipt."""

    def __init__(
        self,
        first: UsiEngineSpec,
        second: UsiEngineSpec,
        config: NnueGameGenerationConfig,
    ) -> None:
        if first.actor_source == second.actor_source and first != second:
            raise ValueError(
                "different engine profiles cannot share one actor_source; use exact "
                "versioned actor identities"
            )
        self.first = first
        self.second = second
        self.config = config

    @staticmethod
    def _preflight_artifacts(
        specs: tuple[UsiEngineSpec, ...],
    ) -> dict[str, ArtifactReceipt]:
        receipts: dict[str, ArtifactReceipt] = {}
        for spec in specs:
            spec.require_game_generation_permission()
            for artifact_path in spec.artifact_paths:
                receipt = _artifact_receipt(
                    artifact_path,
                    label=f"engine {spec.actor_source!r} artifact",
                )
                existing = receipts.get(receipt.path)
                if existing is not None and existing != receipt:
                    raise RuntimeError("same artifact path produced conflicting identities")
                receipts[receipt.path] = receipt
        return receipts

    @staticmethod
    def _profile_artifacts(
        spec: UsiEngineSpec, receipts: dict[str, ArtifactReceipt]
    ) -> tuple[ArtifactReceipt, ...]:
        result: list[ArtifactReceipt] = []
        for artifact_path in spec.artifact_paths:
            resolved = str(artifact_path.expanduser().resolve(strict=True))
            result.append(receipts[resolved])
        return tuple(result)

    @staticmethod
    def _assert_artifacts_unchanged(before: dict[str, ArtifactReceipt]) -> None:
        for path, expected in before.items():
            current = _artifact_receipt(Path(path), label="post-generation engine artifact")
            if current != expected:
                raise RuntimeError(
                    "engine artifact changed while generating games: "
                    f"before={expected.to_dict()!r} after={current.to_dict()!r}"
                )

    def _worker(
        self,
        *,
        worker_index: int,
        assignments: tuple[tuple[int, NnueGenerationOpening], ...],
        gauge: _ConcurrencyGauge,
    ) -> _WorkerResult:
        games: list[GeneratedNnueGame] = []
        with self.first.open_engine() as first_engine, self.second.open_engine() as second_engine:
            startups: list[EngineStartupReceipt] = []
            for engine_slot, spec, engine in (
                ("first", self.first, first_engine),
                ("second", self.second, second_engine),
            ):
                provenance = engine.startup_provenance
                startups.append(
                    EngineStartupReceipt(
                        worker_index=worker_index,
                        engine_slot=engine_slot,
                        actor_source=spec.actor_source,
                        rights_policy_id=spec.policy.policy_id,
                        executable=_artifact_receipt(
                            Path(provenance.resolved_executable),
                            label=f"engine {spec.actor_source!r} executable",
                        ),
                        startup_provenance=provenance.to_dict(),
                    )
                )
            for opening_index, opening in assignments:
                first_engine.new_game()
                second_engine.new_game()
                games.append(
                    _play_generated_game(
                        generation_id=self.config.generation_id,
                        opening=opening,
                        opening_index=opening_index,
                        color_swap_leg=0,
                        black_engine=first_engine,
                        black_spec=self.first,
                        white_engine=second_engine,
                        white_spec=self.second,
                        max_plies=self.config.max_plies,
                        gauge=gauge,
                    )
                )
                first_engine.new_game()
                second_engine.new_game()
                games.append(
                    _play_generated_game(
                        generation_id=self.config.generation_id,
                        opening=opening,
                        opening_index=opening_index,
                        color_swap_leg=1,
                        black_engine=second_engine,
                        black_spec=self.second,
                        white_engine=first_engine,
                        white_spec=self.first,
                        max_plies=self.config.max_plies,
                        gauge=gauge,
                    )
                )
        return _WorkerResult(games=tuple(games), startups=tuple(startups))

    def generate(self) -> NnueGameGenerationResult:
        """Run all color-swapped pairs without touching any existing replay or run."""

        openings = self.config.normalized_openings()
        artifact_receipts = self._preflight_artifacts((self.first, self.second))
        effective_parallelism = min(self.config.parallelism, len(openings))
        assignments = tuple(
            tuple(
                (index, opening)
                for index, opening in enumerate(openings)
                if index % effective_parallelism == worker_index
            )
            for worker_index in range(effective_parallelism)
        )
        if any(not worker_assignments for worker_assignments in assignments):
            raise AssertionError("effective worker partition unexpectedly contains no openings")
        gauge = _ConcurrencyGauge()
        started_at = _utc_now()
        started = time.monotonic()
        worker_results: list[_WorkerResult] = []
        with ThreadPoolExecutor(
            max_workers=effective_parallelism,
            thread_name_prefix="meteo-nnue-gamegen",
        ) as executor:
            futures = tuple(
                executor.submit(
                    self._worker,
                    worker_index=worker_index,
                    assignments=worker_assignments,
                    gauge=gauge,
                )
                for worker_index, worker_assignments in enumerate(assignments)
            )
            for future in futures:
                worker_results.append(future.result())
        wall_seconds = time.monotonic() - started
        if not math.isfinite(wall_seconds) or wall_seconds <= 0:
            raise RuntimeError("generation wall-clock duration is invalid")
        self._assert_artifacts_unchanged(artifact_receipts)

        games = tuple(
            sorted(
                (game for result in worker_results for game in result.games),
                key=lambda game: (
                    game.receipt.opening_index,
                    game.receipt.color_swap_leg,
                ),
            )
        )
        expected_games = 2 * len(openings)
        if len(games) != expected_games:
            raise AssertionError(
                f"paired generator produced {len(games)} games, expected {expected_games}"
            )
        for opening_index in range(len(openings)):
            pair = tuple(game for game in games if game.receipt.opening_index == opening_index)
            if tuple(game.receipt.color_swap_leg for game in pair) != (0, 1):
                raise AssertionError("opening pair does not contain exactly both color swaps")

        startups = tuple(
            sorted(
                (startup for result in worker_results for startup in result.startups),
                key=lambda startup: (startup.worker_index, startup.engine_slot),
            )
        )
        metrics = _speed_metrics(
            games,
            requested_parallelism=self.config.parallelism,
            effective_parallelism=effective_parallelism,
            peak_concurrent_games=gauge.peak,
            wall_seconds=wall_seconds,
        )
        engine_profiles = tuple(
            {
                "engine_slot": engine_slot,
                **spec.to_profile_dict(self._profile_artifacts(spec, artifact_receipts)),
            }
            for engine_slot, spec in (
                ("first", self.first),
                ("second", self.second),
            )
        )
        receipt = NnueGameGenerationReceipt(
            schema=NNUE_GAME_GENERATION_SCHEMA,
            generation_id=self.config.generation_id,
            started_at=started_at,
            finished_at=_utc_now(),
            no_resignation_adjudication=True,
            no_evaluation_adjudication=True,
            rules_adjudicator=("simajilord_shogi.adjudication.adjudicate_board+CSARule27"),
            max_plies_is_not_a_strong_wdl_label=True,
            max_plies=self.config.max_plies,
            engine_profiles=engine_profiles,
            engine_startups=startups,
            openings=openings,
            games=tuple(game.receipt for game in games),
            metrics=metrics,
        )
        return NnueGameGenerationResult(games=games, receipt=receipt)
