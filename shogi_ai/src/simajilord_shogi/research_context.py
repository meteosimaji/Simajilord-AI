"""Deterministic game, opening, phase, and continuous-chunk research context."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum

from .domain import GameRecord
from .ensemble import normalized_sfen


class GamePhase(StrEnum):
    OPENING = "opening_plies_1_24"
    TRANSITION = "transition_plies_25_48"
    MIDDLEGAME = "middlegame_plies_49_96"
    ENDGAME_CONVERSION = "endgame_or_conversion_plies_97_plus"


@dataclass(frozen=True, slots=True)
class TrajectoryContext:
    game_id: str
    opening_family: str
    game_index: int
    sample_index: int
    ply: int
    absolute_ply: int
    phase: str
    chunk_index: int
    chunk_start_ply: int
    chunk_end_ply: int
    previous_normalized_sfen: str | None
    previous_ply: int | None
    previous_is_consecutive: bool
    next_normalized_sfen: str | None
    next_ply: int | None
    next_is_consecutive: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def game_phase(ply: int) -> GamePhase:
    """Classify zero-based sample ply using the documented human-ply ranges."""

    human_ply = ply + 1
    if human_ply <= 24:
        return GamePhase.OPENING
    if human_ply <= 48:
        return GamePhase.TRANSITION
    if human_ply <= 96:
        return GamePhase.MIDDLEGAME
    return GamePhase.ENDGAME_CONVERSION


def _stable_digest(payload: object) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def game_identifier(game: GameRecord) -> str:
    """Identify a complete recorded trajectory without relying on a file path."""

    digest = _stable_digest(
        {
            "initial_sfen": game.initial_sfen,
            "moves": game.moves,
            "winner": game.winner,
            "termination": game.termination.value,
        }
    )
    return f"game:{digest}"


def opening_family_identifier(game: GameRecord) -> str:
    """Return a reproducible opening-root fingerprint, not a guessed joseki name."""

    initial = normalized_sfen(game.initial_sfen)
    digest = _stable_digest(
        {
            "initial_normalized_sfen": initial,
            "first_24_moves": game.moves[:24],
        }
    )
    return f"opening-fingerprint:{digest[:24]}"


def trajectory_contexts(games: tuple[GameRecord, ...]) -> dict[str, TrajectoryContext]:
    """Index every unique normalized position with adjacent and chunk context."""

    contexts: dict[str, TrajectoryContext] = {}
    for game_index, game in enumerate(games):
        game_id = game_identifier(game)
        opening_family = opening_family_identifier(game)
        initial_fields = game.initial_sfen.split()
        if len(initial_fields) != 4:
            raise ValueError(f"game initial SFEN must contain four fields: {game.initial_sfen}")
        initial_move_number = int(initial_fields[3])
        if initial_move_number < 1:
            raise ValueError("SFEN move number must be positive")
        samples = game.samples
        chunk_bounds: dict[int, tuple[int, int, int]] = {}
        chunk_index = 0
        chunk_start = 0
        for sample_index, sample in enumerate(samples):
            if (
                sample_index > 0
                and sample.ply != samples[sample_index - 1].ply + 1
            ):
                for member_index in range(chunk_start, sample_index):
                    chunk_bounds[member_index] = (
                        chunk_index,
                        samples[chunk_start].ply,
                        samples[sample_index - 1].ply,
                    )
                chunk_index += 1
                chunk_start = sample_index
        for member_index in range(chunk_start, len(samples)):
            chunk_bounds[member_index] = (
                chunk_index,
                samples[chunk_start].ply,
                samples[-1].ply,
            )

        for sample_index, sample in enumerate(samples):
            key = normalized_sfen(sample.sfen)
            if key in contexts:
                raise ValueError(f"trajectory repeats normalized position {key}")
            previous = samples[sample_index - 1] if sample_index > 0 else None
            following = samples[sample_index + 1] if sample_index + 1 < len(samples) else None
            current_chunk, chunk_start_ply, chunk_end_ply = chunk_bounds[sample_index]
            absolute_ply = initial_move_number - 1 + sample.ply
            contexts[key] = TrajectoryContext(
                game_id=game_id,
                opening_family=opening_family,
                game_index=game_index,
                sample_index=sample_index,
                ply=sample.ply,
                absolute_ply=absolute_ply,
                phase=game_phase(absolute_ply).value,
                chunk_index=current_chunk,
                chunk_start_ply=chunk_start_ply,
                chunk_end_ply=chunk_end_ply,
                previous_normalized_sfen=(
                    normalized_sfen(previous.sfen) if previous is not None else None
                ),
                previous_ply=previous.ply if previous is not None else None,
                previous_is_consecutive=(
                    previous is not None and previous.ply + 1 == sample.ply
                ),
                next_normalized_sfen=(
                    normalized_sfen(following.sfen) if following is not None else None
                ),
                next_ply=following.ply if following is not None else None,
                next_is_consecutive=(
                    following is not None and sample.ply + 1 == following.ply
                ),
            )
    return contexts
