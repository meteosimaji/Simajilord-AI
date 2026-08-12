"""Create rules-clean replay derivatives with bounded provenance receipts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from rsshogi.core import Board, Move

from .adjudication import terminal_repetition_adjudication
from .domain import GameRecord, Termination
from .model_rights import model_rights
from .replay import append_games, load_games
from .rights_lineage import (
    DERIVED_TEACHER_SOURCE_IDS,
    merge_rights_restriction_summaries,
    summarize_teacher_sidecar,
)

RULES_CLEAN_REPLAY_SCHEMA = "meteo-rules-clean-replay-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _final_board(game: GameRecord) -> Board:
    board = Board(game.initial_sfen)
    for ply, move_usi in enumerate(game.moves):
        move = Move.from_usi(move_usi)
        if not board.is_legal_move(move):
            raise ValueError(f"illegal replay move at ply {ply}: {move_usi}")
        board.apply_move(move)
    return board


@dataclass(frozen=True, slots=True)
class DroppedRepetitionGame:
    game_index: int
    samples: int
    recorded_winner: int | None
    observed_repetition_state: str
    reason: str
    initial_sfen_sha256: str
    moves_sha256: str


def _sidecar_paths(source: Path) -> tuple[Path, ...]:
    candidates = tuple(
        source.with_suffix(source.suffix + suffix)
        for suffix in (".ensemble.json", ".provenance.json", ".dedup.json")
    )
    return tuple(path.resolve(strict=True) for path in candidates if path.is_file())


def _teacher_rights(
    games: list[GameRecord], source_sidecars: tuple[Path, ...]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summaries: list[dict[str, object]] = []
    sidecar_records: list[dict[str, object]] = []
    for sidecar in source_sidecars:
        raw = sidecar.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"replay lineage sidecar must be a JSON object: {sidecar}")
        digest = hashlib.sha256(raw).hexdigest()
        summary = summarize_teacher_sidecar(payload, sidecar_sha256=digest)
        summaries.append(summary)
        sidecar_records.append(
            {"name": sidecar.name, "sha256": digest, "bytes": len(raw)}
        )

    sources: list[dict[str, object]] = []
    if summaries:
        merged = merge_rights_restriction_summaries(summaries)
        sources = list(cast(list[dict[str, object]], merged["sources"]))
    if not sources:
        teacher_ids = sorted(
            {
                sample.teacher_source
                for game in games
                for sample in game.samples
                if sample.teacher_source is not None
                and sample.teacher_source not in DERIVED_TEACHER_SOURCE_IDS
            }
        )
        sources = []
        for rights_id in teacher_ids:
            rights = model_rights(rights_id)
            decision = rights.output_only_meteo_publication.value
            sources.append(
                {
                    "rights_id": rights.rights_id,
                    "decision": decision,
                    "publication_allowed": decision == "allowed",
                }
            )

    rights_records = [
        {
            "rights_id": str(source["rights_id"]),
            "output_only_meteo_publication": str(source["decision"]),
            "publication_allowed": bool(source["publication_allowed"]),
        }
        for source in sources
    ]
    return rights_records, sidecar_records


def sanitize_replay_rules(source: Path, output: Path) -> dict[str, object]:
    """Drop replay games whose recorded repetition is not a legal terminal result."""

    supplied_source = source.expanduser()
    if supplied_source.is_symlink():
        raise ValueError(f"source replay must be a regular non-symlink file: {source}")
    source = supplied_source.resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"source replay must be a regular non-symlink file: {source}")
    output = output.expanduser().resolve()
    provenance = output.with_suffix(output.suffix + ".provenance.json")
    if output == source or output in _sidecar_paths(source):
        raise ValueError("rules-clean output must differ from its source and source sidecars")
    for target in (output, provenance):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to overwrite rules-clean artifact: {target}")
    output.parent.mkdir(parents=True, exist_ok=True)

    games = load_games(source)
    kept: list[GameRecord] = []
    dropped: list[DroppedRepetitionGame] = []
    for game_index, game in enumerate(games):
        if game.termination is not Termination.REPETITION:
            kept.append(game)
            continue
        board = _final_board(game)
        adjudication = terminal_repetition_adjudication(board)
        reason: str | None = None
        if adjudication is None:
            reason = "recorded_repetition_is_not_rules_terminal"
        elif game.winner != adjudication.winner:
            reason = "recorded_repetition_winner_mismatch"
        if reason is None:
            kept.append(game)
            continue
        dropped.append(
            DroppedRepetitionGame(
                game_index=game_index,
                samples=len(game.samples),
                recorded_winner=game.winner,
                observed_repetition_state=str(board.repetition_state()),
                reason=reason,
                initial_sfen_sha256=hashlib.sha256(game.initial_sfen.encode("utf-8")).hexdigest(),
                moves_sha256=hashlib.sha256(" ".join(game.moves).encode("utf-8")).hexdigest(),
            )
        )

    temporary_output = output.with_suffix(output.suffix + ".tmp")
    if temporary_output.exists() or temporary_output.is_symlink():
        raise FileExistsError(f"temporary rules-clean replay already exists: {temporary_output}")
    append_games(temporary_output, kept)
    temporary_output.replace(output)

    source_sidecars = _sidecar_paths(source)
    teacher_rights, sidecar_records = _teacher_rights(games, source_sidecars)
    payload: dict[str, object] = {
        "schema": RULES_CLEAN_REPLAY_SCHEMA,
        "rule_contract": {
            "terminal_repetition_states": ["DRAW", "WIN", "LOSE"],
            "nonterminal_repetition_states": ["SUPERIOR", "INFERIOR", "NONE"],
            "action": "drop_invalid_repetition_games_without_relabeling",
        },
        "source": {
            "name": source.name,
            "sha256": _sha256_file(source),
            "bytes": source.stat().st_size,
            "games": len(games),
            "samples": sum(len(game.samples) for game in games),
            "lineage_sidecars": sidecar_records,
        },
        "output": {
            "name": output.name,
            "sha256": _sha256_file(output),
            "bytes": output.stat().st_size,
            "games": len(kept),
            "samples": sum(len(game.samples) for game in kept),
        },
        "dropped_games": [asdict(record) for record in dropped],
        "dropped_game_count": len(dropped),
        "dropped_sample_count": sum(record.samples for record in dropped),
        "teacher_rights": teacher_rights,
        "publication_allowed": bool(teacher_rights)
        and all(bool(record["publication_allowed"]) for record in teacher_rights),
    }
    _atomic_json(provenance, payload)
    return payload
