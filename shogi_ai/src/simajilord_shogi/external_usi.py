"""Out-of-process USI teacher adapter for explicitly reviewed engines."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import select
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from rsshogi.core import Board, Move
from rsshogi.usi import UsiInfo, parse_bestmove, parse_info

from .domain import (
    GameRecord,
    PositionSample,
    TeacherScoreBound,
    TeacherScoreKind,
    TeacherVariation,
)
from .evaluator import Evaluation

_USI_MATE_SCORE = 32_000
_STARTPOS_SFEN = Board().to_sfen()
_YANEURAOU_GETOPTION = re.compile(r"^Options\[(?P<name>.+)]\s*=\s*(?P<value>.*)$")
_WARNING_MARKERS = (
    "warning",
    "error",
    "mismatch",
    "failed",
    "failure",
    "cannot",
    "can't",
    "not found",
)


class UsiOptionValueVerification(StrEnum):
    """Optional engine-specific proof that ``setoption`` actually took effect.

    USI exposes option declarations but has no standard command for reading a
    current value.  YaneuraOu supplies ``getoption NAME``; selecting that
    dialect makes every non-button requested option fail closed unless the
    post-``isready`` value is returned and equals the request.
    """

    NONE = "none"
    YANEURAOU_GETOPTION = "yaneuraou_getoption"


@dataclass(frozen=True, slots=True)
class UsiOptionDeclaration:
    """One option advertised by the engine during the USI handshake."""

    name: str
    option_type: str
    default: str | None
    minimum: int | None
    maximum: int | None
    choices: tuple[str, ...]
    raw: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UsiAppliedOption:
    """Canonicalized request plus optional engine-reported current value."""

    name: str
    option_type: str
    requested_value: str | None
    applied_value: str | None
    verified: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UsiStartupProvenance:
    """Immutable evidence from one successful USI startup handshake."""

    schema: str
    resolved_executable: str
    arguments: tuple[str, ...]
    working_directory: str
    option_value_verification: str
    identity_lines: tuple[str, ...]
    option_declarations: tuple[UsiOptionDeclaration, ...]
    applied_options: tuple[UsiAppliedOption, ...]
    sent_commands: tuple[str, ...]
    stdout_lines: tuple[str, ...]
    stdout_sha256: str
    stdout_bytes: int
    stderr_lines: tuple[str, ...]
    stderr_sha256: str
    stderr_bytes: int
    warnings: tuple[tuple[str, str], ...]
    warnings_sha256: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["transcript_ordering"] = "per_channel; sent commands are separately ordered"
        return payload

    def write_create_only(self, path: Path) -> None:
        """Write a complete sidecar without following or replacing a target."""

        destination = Path(os.path.abspath(os.fspath(path.expanduser())))
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        serialized_without_hash = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        payload["provenance_sha256"] = hashlib.sha256(serialized_without_hash).hexdigest()
        serialized = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        try:
            with destination.open("xb") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite USI startup provenance: {destination}"
            ) from error


def _option_declaration(line: str) -> UsiOptionDeclaration | None:
    """Parse the standard portion of one ``option name ... type ...`` line."""

    tokens = line.strip().split()
    if not tokens or tokens[0] != "option":
        return None
    if len(tokens) < 5 or tokens[1] != "name":
        raise ValueError(f"malformed USI option declaration: {line}")
    try:
        type_index = tokens.index("type", 2)
    except ValueError as error:
        raise ValueError(f"malformed USI option declaration: {line}") from error
    name = " ".join(tokens[2:type_index])
    if not name or type_index + 1 >= len(tokens):
        raise ValueError(f"malformed USI option declaration: {line}")
    option_type = tokens[type_index + 1].casefold()
    if option_type not in {"button", "check", "combo", "spin", "string"}:
        raise ValueError(f"unsupported USI option type {option_type!r} for {name!r}")
    tail = tokens[type_index + 2 :]

    default: str | None = None
    minimum: int | None = None
    maximum: int | None = None
    choices: list[str] = []
    index = 0
    while index < len(tail):
        key = tail[index]
        index += 1
        if key == "default":
            start = index
            while index < len(tail) and tail[index] not in {"min", "max", "var"}:
                index += 1
            default = " ".join(tail[start:index])
        elif key in {"min", "max"}:
            if index >= len(tail):
                raise ValueError(f"malformed USI {key} in option declaration: {line}")
            try:
                bound = int(tail[index])
            except ValueError as error:
                raise ValueError(f"non-integer USI {key} in option declaration: {line}") from error
            index += 1
            if key == "min":
                minimum = bound
            else:
                maximum = bound
        elif key == "var":
            if index >= len(tail):
                raise ValueError(f"empty USI combo choice in option declaration: {line}")
            choices.append(tail[index])
            index += 1
        else:
            raise ValueError(f"unexpected USI option declaration token {key!r}: {line}")

    if option_type == "spin" and (minimum is None or maximum is None):
        raise ValueError(f"USI spin option lacks min/max: {line}")
    if (
        option_type == "spin"
        and minimum is not None
        and maximum is not None
        and minimum > maximum
    ):
        raise ValueError(f"USI spin option has inverted bounds: {line}")
    return UsiOptionDeclaration(
        name=name,
        option_type=option_type,
        default=default,
        minimum=minimum,
        maximum=maximum,
        choices=tuple(choices),
        raw=line,
    )


def _validated_option_value(declaration: UsiOptionDeclaration, value: str | int) -> str | None:
    """Validate a requested setting against the engine's own declaration."""

    if isinstance(value, bool):
        raise TypeError(f"boolean Python value is ambiguous for USI option {declaration.name!r}")
    rendered = str(value)
    if "\n" in rendered or "\r" in rendered:
        raise ValueError(f"USI option {declaration.name!r} contains a line break")
    if declaration.option_type == "button":
        if rendered not in {"", "<empty>"}:
            raise ValueError(f"USI button option {declaration.name!r} cannot have a value")
        return None
    if declaration.option_type == "spin":
        try:
            numeric = int(rendered)
        except ValueError as error:
            raise ValueError(f"USI spin option {declaration.name!r} requires an integer") from error
        assert declaration.minimum is not None
        assert declaration.maximum is not None
        if not declaration.minimum <= numeric <= declaration.maximum:
            raise ValueError(
                f"USI spin option {declaration.name!r} value {numeric} is outside "
                f"{declaration.minimum}..{declaration.maximum}"
            )
    elif declaration.option_type == "check" and rendered.casefold() not in {"true", "false"}:
        raise ValueError(f"USI check option {declaration.name!r} requires true or false")
    elif declaration.option_type == "combo" and rendered.casefold() not in {
        choice.casefold() for choice in declaration.choices
    }:
        raise ValueError(
            f"USI combo option {declaration.name!r} has no declared choice {rendered!r}"
        )
    return rendered


def _option_values_equal(
    declaration: UsiOptionDeclaration, requested: str | None, applied: str
) -> bool:
    if requested is None:
        return True
    if declaration.option_type == "spin":
        try:
            return int(requested) == int(applied)
        except ValueError:
            return False
    if declaration.option_type in {"check", "combo"}:
        return requested.casefold() == applied.casefold()
    if declaration.option_type == "string" and requested == "<empty>":
        return applied == ""
    return requested == applied


class UsiHistoryMode(StrEnum):
    """How a USI position command represents the path to its target board."""

    BOARD_ONLY = "board_only"
    GAME_PREFIX = "game_prefix"


@dataclass(frozen=True, slots=True)
class UsiPositionHistory:
    """A validated game prefix whose replay must equal one target position.

    Some evaluation functions use repetition state or recent moves in addition
    to the current board.  This typed value preserves that story without
    trusting replay metadata: every prefix move is replayed for legality and the
    resulting canonical SFEN must exactly match ``target_sfen``.
    """

    initial_sfen: str
    moves: tuple[str, ...]
    target_sfen: str

    def __post_init__(self) -> None:
        try:
            board = Board(self.initial_sfen)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid history initial SFEN") from error
        if not board.is_valid():
            raise ValueError(f"invalid history initial SFEN: {self.initial_sfen}")
        canonical_initial_sfen = board.to_sfen()
        canonical_moves: list[str] = []
        for ply, move_usi in enumerate(self.moves):
            try:
                move = Move.from_usi(move_usi)
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid history move syntax at ply {ply}: {move_usi}") from error
            if not board.is_legal_move(move):
                raise ValueError(
                    f"illegal history move at ply {ply}: {move_usi} at {board.to_sfen()}"
                )
            canonical_moves.append(move.to_usi())
            board.apply_move(move)

        try:
            target = Board(self.target_sfen)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid history target SFEN") from error
        if not target.is_valid():
            raise ValueError(f"invalid history target SFEN: {self.target_sfen}")
        canonical_target_sfen = target.to_sfen()
        if board.to_sfen() != canonical_target_sfen:
            raise ValueError(
                "replayed history does not match target position: "
                f"replayed={board.to_sfen()} target={canonical_target_sfen}"
            )
        object.__setattr__(self, "initial_sfen", canonical_initial_sfen)
        object.__setattr__(self, "moves", tuple(canonical_moves))
        object.__setattr__(self, "target_sfen", canonical_target_sfen)

    @classmethod
    def from_game(cls, game: GameRecord, sample: PositionSample) -> UsiPositionHistory:
        """Build the exact prefix ending immediately before ``sample``'s move."""

        if sample.ply < 0 or sample.ply > len(game.moves):
            raise ValueError(
                f"sample ply {sample.ply} is outside recorded move range 0..{len(game.moves)}"
            )
        if (
            sample.chosen_move is not None
            and sample.ply < len(game.moves)
            and sample.chosen_move != game.moves[sample.ply]
        ):
            raise ValueError(
                f"sample chosen move {sample.chosen_move} does not match recorded move "
                f"{game.moves[sample.ply]} at ply {sample.ply}"
            )
        history = cls(
            initial_sfen=game.initial_sfen,
            moves=game.moves[: sample.ply],
            target_sfen=sample.sfen,
        )
        target_turn = history.target_board().turn.value
        if target_turn != sample.turn:
            raise ValueError(
                f"history target turn {target_turn} does not match sample turn {sample.turn} "
                f"at ply {sample.ply}"
            )
        return history

    @property
    def mode(self) -> UsiHistoryMode:
        return UsiHistoryMode.GAME_PREFIX

    def target_board(self) -> Board:
        board = Board(self.initial_sfen)
        for move_usi in self.moves:
            board.apply_move(Move.from_usi(move_usi))
        if board.to_sfen() != self.target_sfen:
            raise AssertionError("validated USI history changed before target-board replay")
        return board

    def position_command(self) -> str:
        root = (
            "position startpos"
            if self.initial_sfen == _STARTPOS_SFEN
            else f"position sfen {self.initial_sfen}"
        )
        if not self.moves:
            return root
        return f"{root} moves {' '.join(self.moves)}"


@dataclass(frozen=True, slots=True)
class ExternalTeacherPolicy:
    """Rights gate separating local labels from permission to publish them."""

    policy_id: str
    name: str
    source: str
    analysis_allowed: bool
    training_outputs_allowed: bool
    redistribution_allowed: bool
    requires_explicit_local_authorization: bool = False
    training_outputs_local_only: bool = False

    def __post_init__(self) -> None:
        if self.training_outputs_local_only and not self.training_outputs_allowed:
            raise ValueError("local-only training outputs must first be approved for training")
        if self.training_outputs_local_only and self.redistribution_allowed:
            raise ValueError("local-only training outputs cannot be marked redistributable")

    def require_analysis_permission(self) -> None:
        if self.requires_explicit_local_authorization:
            raise PermissionError(
                f"engine {self.name!r} is outside the public-release-safe teacher set; "
                "use its exact reviewed local profile with an explicit lawful-acquisition "
                "acknowledgement when local use is authorized"
            )
        if not self.analysis_allowed:
            raise PermissionError(f"engine {self.name!r} is not approved for analysis")

    def require_training_permission(self) -> None:
        self.require_analysis_permission()
        if not self.analysis_allowed or not self.training_outputs_allowed:
            raise PermissionError(
                f"teacher {self.name!r} is not approved for training-data generation"
            )


@dataclass(frozen=True, slots=True)
class UsiAnalysis:
    bestmove: str
    candidates: tuple[TeacherVariation, ...]
    nodes: int | None
    time_ms: int | None
    nps: int | None
    depth: int | None
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ExternalTeacherTarget:
    """One rights-approved MultiPV target in the side-to-move perspective."""

    bestmove: str
    policy: dict[str, float]
    value: float
    candidates: tuple[TeacherVariation, ...]
    move_values: tuple[tuple[str, float], ...]
    scores: tuple[tuple[str, int], ...]
    policy_temperature: float
    value_scale: float
    nodes: int | None
    time_ms: int | None
    nps: int | None
    depth: int | None
    elapsed_seconds: float


def _variation_from_info_line(line: str, rank: int, info: UsiInfo) -> TeacherVariation:
    """Recover score syntax that ``UsiScore`` intentionally normalizes to an integer."""

    if not info.pv or info.score is None:
        raise ValueError("teacher variation requires both a score and a PV")
    tokens = line.strip().split()
    try:
        score_index = tokens.index("score")
        score_kind = TeacherScoreKind(tokens[score_index + 1])
        raw_value = tokens[score_index + 2]
    except (IndexError, ValueError) as error:
        raise ValueError("malformed USI score in teacher variation") from error
    bound_text = info.bound.to_string() or TeacherScoreBound.EXACT.value
    bound = TeacherScoreBound(bound_text)
    move = info.pv[0].to_usi()
    pv = tuple(candidate_move.to_usi() for candidate_move in info.pv)
    if score_kind is TeacherScoreKind.CENTIPAWN:
        return TeacherVariation(
            rank=rank,
            move=move,
            score_kind=score_kind,
            bound=bound,
            pv=pv,
            score_cp=int(raw_value),
        )
    if raw_value in {"+", "-"}:
        return TeacherVariation(
            rank=rank,
            move=move,
            score_kind=score_kind,
            bound=bound,
            pv=pv,
            mate_unknown_sign=1 if raw_value == "+" else -1,
        )
    return TeacherVariation(
        rank=rank,
        move=move,
        score_kind=score_kind,
        bound=bound,
        pv=pv,
        mate_plies=int(raw_value),
    )


def _mate_sign(variation: TeacherVariation) -> int:
    if variation.score_kind is not TeacherScoreKind.MATE:
        raise TypeError("centipawn variation has no mate sign")
    if variation.mate_unknown_sign is not None:
        return variation.mate_unknown_sign
    assert variation.mate_plies is not None
    return 1 if variation.mate_plies >= 0 else -1


def _legacy_encoded_score(variation: TeacherVariation) -> int:
    """Retain the old integer score API without using mate values as centipawns."""

    if variation.score_kind is TeacherScoreKind.CENTIPAWN:
        assert variation.score_cp is not None
        return variation.score_cp
    if variation.mate_unknown_sign is not None:
        return variation.mate_unknown_sign * _USI_MATE_SCORE
    assert variation.mate_plies is not None
    if variation.mate_plies >= 0:
        return _USI_MATE_SCORE - variation.mate_plies
    return -_USI_MATE_SCORE - variation.mate_plies


def _variation_preference(variation: TeacherVariation) -> tuple[int, int]:
    """Order score domains without treating a mate encoding as a cp measurement."""

    if variation.score_kind is TeacherScoreKind.MATE:
        domain = 2 if _mate_sign(variation) > 0 else 0
    else:
        domain = 1
    return domain, _legacy_encoded_score(variation)


def _variation_value(variation: TeacherVariation, value_scale: float) -> float:
    if variation.score_kind is TeacherScoreKind.MATE:
        return float(_mate_sign(variation))
    assert variation.score_cp is not None
    return math.tanh(variation.score_cp / value_scale)


def _policy_from_variations(
    variations: dict[str, TeacherVariation], policy_temperature: float
) -> dict[str, float]:
    """Build a policy inside one score domain, never by averaging mate and cp units."""

    winning_mates = {
        move: variation
        for move, variation in variations.items()
        if variation.score_kind is TeacherScoreKind.MATE and _mate_sign(variation) > 0
    }
    if winning_mates:
        probability = 1.0 / len(winning_mates)
        return {move: probability for move in winning_mates}
    centipawns = {
        move: variation
        for move, variation in variations.items()
        if variation.score_kind is TeacherScoreKind.CENTIPAWN
    }
    if centipawns:
        maximum = max(
            variation.score_cp
            for variation in centipawns.values()
            if variation.score_cp is not None
        )
        weights = {
            move: math.exp((variation.score_cp - maximum) / policy_temperature)
            for move, variation in centipawns.items()
            if variation.score_cp is not None
        }
        total = sum(weights.values())
        return {move: weight / total for move, weight in weights.items()}
    probability = 1.0 / len(variations)
    return {move: probability for move in variations}


class ExternalUsiTeacher:
    """Run an explicitly configured engine without linking or copying its weights.

    ``value_scale`` is the raw denominator D in the signed value conversion
    ``tanh(cp / D)``.  Its default 1200 corresponds to Ponanza coefficient
    C=600 because ``2 * sigmoid(cp / C) - 1 == tanh(cp / (2C))``.
    """

    def __init__(
        self,
        command: list[str],
        policy: ExternalTeacherPolicy,
        *,
        nodes: int,
        multipv: int = 32,
        options: dict[str, str | int] | None = None,
        timeout_seconds: float = 300.0,
        training_use: bool = True,
        working_directory: Path | None = None,
        policy_temperature: float = 200.0,
        value_scale: float = 1_200.0,
        option_value_verification: UsiOptionValueVerification = UsiOptionValueVerification.NONE,
        startup_provenance_path: Path | None = None,
    ) -> None:
        if not command or nodes < 1 or multipv < 1 or timeout_seconds <= 0:
            raise ValueError("command, nodes, multipv, and timeout must be valid")
        if not math.isfinite(policy_temperature) or policy_temperature <= 0:
            raise ValueError("policy_temperature must be finite and positive")
        if not math.isfinite(value_scale) or value_scale <= 0:
            raise ValueError("value_scale must be finite and positive")
        if training_use:
            policy.require_training_permission()
        else:
            policy.require_analysis_permission()
        if any("\n" in argument or "\r" in argument for argument in command):
            raise ValueError("USI engine command arguments must not contain line breaks")
        self.command = command
        self.policy = policy
        self.nodes = nodes
        self.multipv = multipv
        self.options = dict(options or {})
        normalized_option_names: set[str] = set()
        for name in self.options:
            if not name or "\n" in name or "\r" in name:
                raise ValueError("USI option names must be non-empty single-line strings")
            normalized_name = name.casefold()
            if normalized_name in normalized_option_names:
                raise ValueError(f"duplicate USI option name ignoring case: {name}")
            normalized_option_names.add(normalized_name)
        if any(name.casefold() == "multipv" for name in self.options):
            raise ValueError("MultiPV is reserved; configure it through the multipv argument")
        self.timeout_seconds = timeout_seconds
        self.training_use = training_use
        self.policy_temperature = policy_temperature
        self.value_scale = value_scale
        self.option_value_verification = UsiOptionValueVerification(option_value_verification)
        self.startup_provenance_path = (
            Path(os.path.abspath(os.fspath(startup_provenance_path.expanduser())))
            if startup_provenance_path is not None
            else None
        )
        self.working_directory = (
            working_directory.expanduser().resolve() if working_directory is not None else None
        )
        if self.working_directory is not None and not self.working_directory.is_dir():
            raise NotADirectoryError(self.working_directory)
        self.process: subprocess.Popen[bytes] | None = None
        self._read_buffer = bytearray()
        self._stderr_open = False
        self._capturing_startup = False
        self._startup_stdout = bytearray()
        self._startup_stderr = bytearray()
        self._startup_commands: list[str] = []
        self._startup_provenance: UsiStartupProvenance | None = None

    def start(self) -> None:
        if self.process is not None:
            return
        requested = Path(self.command[0]).expanduser()
        resolved = str(requested.resolve()) if requested.exists() else shutil.which(self.command[0])
        if resolved is None:
            raise FileNotFoundError(requested)
        provenance_path = self.startup_provenance_path
        if provenance_path is not None and (
            provenance_path.exists() or provenance_path.is_symlink()
        ):
            raise FileExistsError(
                f"refusing to overwrite USI startup provenance: {provenance_path}"
            )
        self._read_buffer.clear()
        self._startup_stdout.clear()
        self._startup_stderr.clear()
        self._startup_commands.clear()
        self._startup_provenance = None
        self._capturing_startup = True
        self.process = subprocess.Popen(
            [resolved, *self.command[1:]],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=str(self.working_directory or Path(resolved).resolve().parent),
        )
        self._stderr_open = True
        try:
            self._send("usi")
            handshake_lines = self._read_until("usiok")
            declarations = self._parse_option_declarations(handshake_lines)
            requested_options = self._validate_requested_options(declarations)
            for declaration, requested_value in requested_options:
                command = f"setoption name {declaration.name}"
                if requested_value is not None:
                    command += f" value {requested_value}"
                self._send(command)
            self._send("isready")
            self._read_until("readyok")

            applied_options: list[UsiAppliedOption] = []
            verify_values = (
                self.option_value_verification
                is UsiOptionValueVerification.YANEURAOU_GETOPTION
            )
            for declaration, requested_value in requested_options:
                if requested_value is None:
                    applied_options.append(
                        UsiAppliedOption(
                            name=declaration.name,
                            option_type=declaration.option_type,
                            requested_value=None,
                            applied_value=None,
                            verified=False,
                        )
                    )
                    continue
                applied_value: str | None = None
                if verify_values:
                    applied_value = self._read_yaneuraou_option(declaration.name)
                    if not _option_values_equal(declaration, requested_value, applied_value):
                        raise RuntimeError(
                            f"USI option {declaration.name!r} was not applied: "
                            f"requested={requested_value!r} current={applied_value!r}"
                        )
                applied_options.append(
                    UsiAppliedOption(
                        name=declaration.name,
                        option_type=declaration.option_type,
                        requested_value=requested_value,
                        applied_value=applied_value,
                        verified=verify_values,
                    )
                )
            self._drain_startup_stderr()
            self._startup_provenance = self._build_startup_provenance(
                resolved=resolved,
                declarations=declarations,
                applied_options=tuple(applied_options),
            )
            self._capturing_startup = False
            if provenance_path is not None:
                self._startup_provenance.write_create_only(provenance_path)
            self.new_game()
        except BaseException:
            self._capturing_startup = False
            self.close()
            raise

    @property
    def startup_provenance(self) -> UsiStartupProvenance:
        """Return evidence only after a complete, verified startup."""

        if self._startup_provenance is None:
            raise RuntimeError("USI startup provenance is unavailable before successful start")
        return self._startup_provenance

    def write_startup_provenance(self, path: Path) -> None:
        """Persist the completed handshake as a create-only JSON sidecar."""

        self.startup_provenance.write_create_only(path)

    @staticmethod
    def _parse_option_declarations(
        handshake_lines: tuple[str, ...],
    ) -> dict[str, UsiOptionDeclaration]:
        declarations: dict[str, UsiOptionDeclaration] = {}
        for line in handshake_lines:
            declaration = _option_declaration(line)
            if declaration is None:
                continue
            normalized = declaration.name.casefold()
            if normalized in declarations:
                raise ValueError(
                    f"duplicate USI option declaration ignoring case: {declaration.name}"
                )
            declarations[normalized] = declaration
        return declarations

    def _validate_requested_options(
        self, declarations: dict[str, UsiOptionDeclaration]
    ) -> tuple[tuple[UsiOptionDeclaration, str | None], ...]:
        requested: list[tuple[str, str | int]] = [("MultiPV", self.multipv)]
        requested.extend(self.options.items())
        validated: list[tuple[UsiOptionDeclaration, str | None]] = []
        for requested_name, requested_value in requested:
            declaration = declarations.get(requested_name.casefold())
            if declaration is None:
                available = ", ".join(
                    sorted((entry.name for entry in declarations.values()), key=str.casefold)
                )
                raise ValueError(
                    f"USI engine did not declare requested option {requested_name!r}; "
                    f"declared options: {available or '<none>'}"
                )
            validated.append(
                (declaration, _validated_option_value(declaration, requested_value))
            )
        return tuple(validated)

    def _read_yaneuraou_option(self, option_name: str) -> str:
        if any(character.isspace() for character in option_name):
            raise ValueError(
                "YaneuraOu getoption verification does not support option names with whitespace: "
                f"{option_name!r}"
            )
        self._send(f"getoption {option_name}")
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            line = self._read_line(deadline)
            match = _YANEURAOU_GETOPTION.fullmatch(line.strip())
            if match is not None:
                reported_name = match.group("name")
                if reported_name.casefold() != option_name.casefold():
                    raise RuntimeError(
                        f"YaneuraOu getoption returned {reported_name!r} while querying "
                        f"{option_name!r}"
                    )
                return match.group("value")
            lowered = line.casefold()
            if "no such option" in lowered or lowered.startswith("error"):
                raise RuntimeError(
                    f"YaneuraOu getoption failed for {option_name!r}: {line}"
                )

    def _build_startup_provenance(
        self,
        *,
        resolved: str,
        declarations: dict[str, UsiOptionDeclaration],
        applied_options: tuple[UsiAppliedOption, ...],
    ) -> UsiStartupProvenance:
        stdout = bytes(self._startup_stdout)
        stderr = bytes(self._startup_stderr)
        stdout_lines = tuple(stdout.decode(errors="replace").splitlines())
        stderr_lines = tuple(stderr.decode(errors="replace").splitlines())
        warnings = tuple(
            [("stderr", line) for line in stderr_lines if line.strip()]
            + [
                ("stdout", line)
                for line in stdout_lines
                if any(marker in line.casefold() for marker in _WARNING_MARKERS)
            ]
        )
        warnings_bytes = json.dumps(
            warnings,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return UsiStartupProvenance(
            schema="meteo-usi-startup-provenance-v1",
            resolved_executable=str(Path(resolved).resolve()),
            arguments=tuple(self.command[1:]),
            working_directory=str(self.working_directory or Path(resolved).resolve().parent),
            option_value_verification=self.option_value_verification.value,
            identity_lines=tuple(
                line for line in stdout_lines if line.startswith("id ")
            ),
            option_declarations=tuple(declarations.values()),
            applied_options=applied_options,
            sent_commands=tuple(self._startup_commands),
            stdout_lines=stdout_lines,
            stdout_sha256=hashlib.sha256(stdout).hexdigest(),
            stdout_bytes=len(stdout),
            stderr_lines=stderr_lines,
            stderr_sha256=hashlib.sha256(stderr).hexdigest(),
            stderr_bytes=len(stderr),
            warnings=warnings,
            warnings_sha256=hashlib.sha256(warnings_bytes).hexdigest(),
        )

    def _drain_startup_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None or not self._stderr_open:
            return
        deadline = time.monotonic() + min(0.05, self.timeout_seconds)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            readable, _, _ = select.select([process.stderr.fileno()], [], [], remaining)
            if not readable:
                return
            chunk = os.read(process.stderr.fileno(), 65536)
            if not chunk:
                self._stderr_open = False
                return
            self._startup_stderr.extend(chunk)

    def new_game(self) -> None:
        """Reset engine-side game state without restarting the process."""

        if self.process is None:
            return
        self._send("usinewgame")

    def evaluate(self, board: Board) -> Evaluation:
        target = self.target_from_analysis(board, self.analyse(board))
        return Evaluation(policy=target.policy, value=target.value)

    def training_target(self, board: Board) -> ExternalTeacherTarget:
        """Return a target only when output use was explicitly approved at construction."""

        if not self.training_use:
            raise PermissionError(
                f"teacher {self.policy.name!r} was opened for analysis only, not training"
            )
        return self.target_from_analysis(board, self.analyse(board))

    def training_target_with_history(self, position: UsiPositionHistory) -> ExternalTeacherTarget:
        """Return a training target while preserving a validated game prefix."""

        if not self.training_use:
            raise PermissionError(
                f"teacher {self.policy.name!r} was opened for analysis only, not training"
            )
        board = position.target_board()
        return self.target_from_analysis(board, self.analyse_with_history(position))

    def analyse(self, board: Board) -> UsiAnalysis:
        """Analyse one board without prior-move history for backward compatibility."""

        return self._analyse_command(f"position sfen {board.to_sfen()}")

    def analyse_with_history(self, position: UsiPositionHistory) -> UsiAnalysis:
        """Analyse a target using its validated initial position and move prefix."""

        return self._analyse_command(position.position_command())

    def _analyse_command(self, position_command: str) -> UsiAnalysis:
        self.start()
        started = time.monotonic()
        self._send(position_command)
        self._send(f"go nodes {self.nodes}")
        candidates_by_rank: dict[int, TeacherVariation] = {}
        last_nodes: int | None = None
        last_time_ms: int | None = None
        last_nps: int | None = None
        last_depth: int | None = None
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            line = self._read_line(deadline)
            if line.startswith("info "):
                try:
                    info = parse_info(line)
                except ValueError:
                    continue
                rank = info.multipv or 1
                if info.pv and info.score is not None:
                    candidates_by_rank[rank] = _variation_from_info_line(line, rank, info)
                if info.nodes is not None:
                    last_nodes = info.nodes
                if info.time is not None:
                    last_time_ms = info.time
                if info.nps is not None:
                    last_nps = info.nps
                if info.depth is not None:
                    last_depth = info.depth
            elif line.startswith("bestmove "):
                best_token = line.split(maxsplit=2)[1]
                candidates = tuple(
                    candidates_by_rank[rank] for rank in sorted(candidates_by_rank)
                )
                if best_token in {"resign", "win"}:
                    return UsiAnalysis(
                        best_token,
                        candidates,
                        last_nodes,
                        last_time_ms,
                        last_nps,
                        last_depth,
                        time.monotonic() - started,
                    )
                best = parse_bestmove(line).bestmove.to_usi()
                return UsiAnalysis(
                    best,
                    candidates,
                    last_nodes,
                    last_time_ms,
                    last_nps,
                    last_depth,
                    time.monotonic() - started,
                )

    def target_from_analysis(
        self, board: Board, analysis: UsiAnalysis
    ) -> ExternalTeacherTarget:
        """Calibrate an already obtained analysis without changing its rights scope.

        This conversion is intentionally available to analysis-only callers such
        as arenas.  It does not run the engine and does not grant permission to
        persist the result as training data; ``training_target*`` remains the
        rights-gated API for that purpose.
        """

        legal = {move.to_usi() for move in board.legal_moves()}
        if not legal:
            raise ValueError("cannot create an external teacher target for a terminal position")
        candidates = tuple(
            candidate for candidate in analysis.candidates if candidate.move in legal
        )
        representatives: dict[str, TeacherVariation] = {}
        for candidate in candidates:
            current = representatives.get(candidate.move)
            if current is None or _variation_preference(candidate) > _variation_preference(current):
                representatives[candidate.move] = candidate
        if not representatives:
            ordered_legal = sorted(legal)
            probability = 1.0 / len(ordered_legal)
            bestmove = analysis.bestmove if analysis.bestmove in legal else ordered_legal[0]
            return ExternalTeacherTarget(
                bestmove=bestmove,
                policy={move: probability for move in ordered_legal},
                value=0.0,
                candidates=(),
                move_values=(),
                scores=(),
                policy_temperature=self.policy_temperature,
                value_scale=self.value_scale,
                nodes=analysis.nodes,
                time_ms=analysis.time_ms,
                nps=analysis.nps,
                depth=analysis.depth,
                elapsed_seconds=analysis.elapsed_seconds,
            )
        policy = _policy_from_variations(representatives, self.policy_temperature)
        bestmove = (
            analysis.bestmove
            if analysis.bestmove in legal
            else max(representatives, key=lambda move: _variation_preference(representatives[move]))
        )
        if bestmove not in policy:
            highest_probability = max(policy.values())
            policy[bestmove] = highest_probability
            normalization = sum(policy.values())
            policy = {move: probability / normalization for move, probability in policy.items()}
        move_values = {
            move: _variation_value(candidate, self.value_scale)
            for move, candidate in representatives.items()
        }
        legacy_scores = {
            move: _legacy_encoded_score(candidate) for move, candidate in representatives.items()
        }
        if bestmove not in move_values:
            best_candidate = max(representatives.values(), key=_variation_preference)
            move_values[bestmove] = _variation_value(best_candidate, self.value_scale)
            legacy_scores[bestmove] = _legacy_encoded_score(best_candidate)
        ordered_scores = tuple(
            sorted(legacy_scores.items(), key=lambda item: item[1], reverse=True)
        )
        return ExternalTeacherTarget(
            bestmove=bestmove,
            policy=policy,
            value=move_values[bestmove],
            candidates=candidates,
            move_values=tuple(sorted(move_values.items())),
            scores=ordered_scores,
            policy_temperature=self.policy_temperature,
            value_scale=self.value_scale,
            nodes=analysis.nodes,
            time_ms=analysis.time_ms,
            nps=analysis.nps,
            depth=analysis.depth,
            elapsed_seconds=analysis.elapsed_seconds,
        )

    def close(self) -> None:
        process = self.process
        self.process = None
        self._read_buffer.clear()
        self._stderr_open = False
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write(b"quit\n")
                    process.stdin.flush()
                process.wait(timeout=2)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                process.wait(timeout=2)

    def __enter__(self) -> ExternalUsiTeacher:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _send(self, command: str) -> None:
        process = self.process
        if process is None or process.stdin is None:
            raise RuntimeError("USI teacher is not running")
        if "\n" in command or "\r" in command:
            raise ValueError("USI command must be exactly one line")
        if self._capturing_startup:
            self._startup_commands.append(command)
        process.stdin.write((command + "\n").encode())
        process.stdin.flush()

    def _read_until(self, expected: str) -> tuple[str, ...]:
        deadline = time.monotonic() + self.timeout_seconds
        lines: list[str] = []
        while True:
            line = self._read_line(deadline)
            lines.append(line)
            if line == expected:
                return tuple(lines)

    def _read_line(self, deadline: float) -> str:
        process = self.process
        if process is None or process.stdout is None:
            raise RuntimeError("USI teacher is not running")
        while True:
            newline = self._read_buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._read_buffer[:newline])
                del self._read_buffer[: newline + 1]
                return line.rstrip(b"\r").decode(errors="replace")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("USI teacher response timed out")
            descriptors = [process.stdout.fileno()]
            if process.stderr is not None and self._stderr_open:
                descriptors.append(process.stderr.fileno())
            readable, _, _ = select.select(descriptors, [], [], remaining)
            if not readable:
                raise TimeoutError("USI teacher response timed out")
            if process.stderr is not None and process.stderr.fileno() in readable:
                stderr_chunk = os.read(process.stderr.fileno(), 65536)
                if stderr_chunk:
                    if self._capturing_startup:
                        self._startup_stderr.extend(stderr_chunk)
                else:
                    self._stderr_open = False
            if process.stdout.fileno() in readable:
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError(f"USI teacher exited with code {process.poll()}")
                if self._capturing_startup:
                    self._startup_stdout.extend(chunk)
                self._read_buffer.extend(chunk)
