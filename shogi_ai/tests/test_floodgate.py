from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from rsshogi.core import Board

from simajilord_shogi import floodgate as floodgate_module
from simajilord_shogi.cli import main
from simajilord_shogi.config import ReanalysisConfig, SearchConfig
from simajilord_shogi.domain import Termination
from simajilord_shogi.ensemble import normalized_sfen
from simajilord_shogi.evaluator import UniformEvaluator
from simajilord_shogi.floodgate import (
    CANDIDATE_SIDECAR_SCHEMA,
    CORPUS_SCHEMA,
    ORIGINAL_CANDIDATE_SOURCE,
    CsaSource,
    EngineIdentityRule,
    FloodgateCorpusConfig,
    RatingSnapshot,
    create_floodgate_corpus,
    list_7z_csa_members,
    parse_csa_game,
    parse_rating_snapshot,
    read_7z_csa_member,
    resolve_engine_identities,
    stream_7z_csa_sources,
)
from simajilord_shogi.reanalysis import reanalyse_game
from simajilord_shogi.replay import load_games


def _rating_html(rows: list[tuple[str, str, int, int, int]]) -> bytes:
    body = []
    for player_id, name, rating, wins, losses in rows:
        games = wins + losses
        win_rate = wins / games if games else 0.0
        body.append(
            "<tr>"
            f'<td class="name"><a href="/shogi/x/2026/player/{player_id}.html">'
            f"{name}</a></td>"
            f'<td class="rate">{rating}</td>'
            f'<td class="wins">{wins}</td>'
            f'<td class="losses">{losses}</td>'
            f'<td class="win_rate">{win_rate:.3f}</td>'
            '<td class="last_modified">2026-08-09 10:30:00</tr>'
        )
    return (
        "<!doctype html><html><body>"
        "<h1>Shogi Server Rating 2026-08-09</h1>"
        '<table class="player-rating"><tbody>'
        + "".join(body)
        + "</tbody></table></body></html>"
    ).encode()


def _snapshot() -> RatingSnapshot:
    return parse_rating_snapshot(
        _rating_html(
            [
                ("Alpha+aaaaaaa", "Alpha", 4300, 90, 10),
                ("Beta+bbbbbbb", "Beta", 4200, 100, 0),
                ("Gamma+ccccccc", "Gamma", 3500, 20, 20),
            ]
        ),
        source_url=(
            "https://wdoor.c.u-tokyo.ac.jp/shogi/x/rating/"
            "players-floodgate14-20260809.html"
        ),
        scope="14_day",
    )


def _csa(
    *,
    black: str = "Alpha",
    white: str = "Beta",
    moves: tuple[str, ...] = ("+7776FU", "-3334FU"),
    terminal: str = "%TORYO",
    position: str = "PI\n+",
    nonce: str = "fixture",
) -> bytes:
    move_rows = "\n".join(f"{move}\nT{index}" for index, move in enumerate(moves))
    return (
        f"V2\nN+{black}\nN-{white}\n"
        f"$EVENT:{nonce}\n$START_TIME:2026/08/09 10:00:00\n"
        f"{position}\n{move_rows}\n{terminal},'* terminal comment\n"
        "$END_TIME:2026/08/09 10:10:00\n"
    ).encode()


def _seven_zip() -> str:
    executable = shutil.which("7z") or shutil.which("7zz")
    if executable is None:
        pytest.skip("7-Zip is required for archive integration coverage")
    return executable


def _create_7z_fixture(tmp_path: Path, members: dict[str, bytes]) -> Path:
    executable = _seven_zip()
    source_directory = tmp_path / "archive-source"
    source_directory.mkdir()
    for member, raw_bytes in members.items():
        path = source_directory / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw_bytes)
    archive = tmp_path / "fixture.7z"
    completed = subprocess.run(
        [executable, "a", "-t7z", "-mx=1", str(archive), "."],
        cwd=source_directory,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        pytest.fail(completed.stderr.decode("utf-8", errors="replace"))
    return archive


def test_7z_batch_stream_is_exact_bounded_and_single_member_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_members = {
        f"2026/game-{index:04d}.csa": _csa(nonce=f"batch-{index:04d}")
        for index in range(128)
    }
    archive = _create_7z_fixture(tmp_path, raw_members)
    members = list_7z_csa_members(archive, seven_zip=_seven_zip())
    extraction_commands = 0
    temporary_roots: list[Path] = []
    original_run = floodgate_module.subprocess.run
    original_mkdtemp = floodgate_module.tempfile.mkdtemp

    def recording_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal extraction_commands
        command = args[0]
        if isinstance(command, list) and len(command) > 1 and command[1] == "x":
            extraction_commands += 1
        return original_run(*args, **kwargs)  # type: ignore[no-any-return]

    def recording_mkdtemp(*args: object, **kwargs: object) -> str:
        created = original_mkdtemp(*args, **kwargs)  # type: ignore[arg-type]
        temporary_roots.append(Path(created))
        return created

    monkeypatch.setattr(floodgate_module.subprocess, "run", recording_run)
    monkeypatch.setattr(floodgate_module.tempfile, "mkdtemp", recording_mkdtemp)
    with stream_7z_csa_sources(
        archive,
        members,
        seven_zip=_seven_zip(),
        maximum_bytes=100_000,
        maximum_batch_bytes=2_000_000,
        maximum_batch_members=1_000,
    ) as sources:
        streamed = [(source.archive_member, source.raw_bytes) for source in sources]

    assert extraction_commands == 1
    assert streamed == [(member, raw_members[member]) for member in members]
    assert temporary_roots and all(not path.exists() for path in temporary_roots)
    single = read_7z_csa_member(
        archive,
        members[17],
        seven_zip=_seven_zip(),
        maximum_bytes=100_000,
    )
    assert single.archive_member == members[17]
    assert single.raw_bytes == raw_members[members[17]]
    assert all(not path.exists() for path in temporary_roots)


def test_prepare_floodgate_cli_consumes_stream_and_creates_corpus(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive = _create_7z_fixture(
        tmp_path,
        {
            "2026/game-a.csa": _csa(nonce="cli-a"),
            "2026/game-b.csa": _csa(nonce="cli-b"),
        },
    )
    rating_html = tmp_path / "players-floodgate-20260809.html"
    rating_html.write_bytes(
        _rating_html(
            [
                ("Alpha+aaaaaaa", "Alpha", 4300, 90, 10),
                ("Beta+bbbbbbb", "Beta", 4200, 90, 10),
            ]
        )
    )
    output = tmp_path / "cli-corpus"

    result = main(
        [
            "prepare-floodgate-corpus",
            str(archive),
            str(rating_html),
            str(output),
            "--rating-url",
            "https://example.test/players-floodgate-20260809.html",
            "--rating-scope",
            "test",
            "--min-rating",
            "0",
            "--min-games",
            "1",
            "--seven-zip",
            _seven_zip(),
            "--extraction-batch-bytes",
            "2000000",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert result == 0
    assert report["archive_members"] == 2
    assert report["inspection"]["sources"] == 2
    assert report["inspection"]["parsed_games"] == 2
    assert (output / "manifest.json").is_file()


def test_7z_stream_rejects_oversize_and_unsafe_member_before_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _create_7z_fixture(
        tmp_path,
        {"2026/large.csa": _csa(nonce="large") + b"x" * 1_000},
    )
    extraction_commands = 0
    original_run = floodgate_module.subprocess.run

    def recording_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal extraction_commands
        command = args[0]
        if isinstance(command, list) and len(command) > 1 and command[1] == "x":
            extraction_commands += 1
        return original_run(*args, **kwargs)  # type: ignore[no-any-return]

    monkeypatch.setattr(floodgate_module.subprocess, "run", recording_run)
    with pytest.raises(ValueError, match="exceeds maximum_bytes"), stream_7z_csa_sources(
        archive,
        ("2026/large.csa",),
        seven_zip=_seven_zip(),
        maximum_bytes=100,
        maximum_batch_bytes=100,
    ) as sources:
        tuple(sources)
    assert extraction_commands == 0

    with pytest.raises(ValueError, match="unsafe archive member"), stream_7z_csa_sources(
        archive,
        ("../escape.csa",),
        seven_zip=_seven_zip(),
    ):
        pass


def test_extracted_batch_audit_fails_closed_on_symbolic_link(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    target = tmp_path / "target.csa"
    target.write_bytes(_csa(nonce="link-target"))
    link = extracted / "linked.csa"
    link.symlink_to(target)
    entry = floodgate_module._SevenZipMember(
        path="linked.csa",
        size=target.stat().st_size,
        is_directory=False,
    )

    with pytest.raises(ValueError, match="non-regular file"):
        floodgate_module._audit_extracted_batch(
            extracted,
            (entry,),
            maximum_bytes=100_000,
            maximum_batch_bytes=100_000,
        )


@pytest.mark.parametrize(
    ("member_section", "message"),
    [
        (
            "Path = ../escape.txt\nSize = 1\nAttributes = A_ -rw-r--r--\n",
            "unsafe archive member",
        ),
        (
            "Path = linked.csa\nSize = 1\nAttributes = A_ lrwxr-xr-x\n"
            "Symbolic Link = target.csa\n",
            "link entry",
        ),
        (
            "Path = Game.csa\nSize = 1\nAttributes = A_ -rw-r--r--\n\n"
            "Path = game.csa\nSize = 1\nAttributes = A_ -rw-r--r--\n",
            "filesystem-colliding",
        ),
        (
            "Path = parent.csa\nSize = 1\nAttributes = A_ -rw-r--r--\n\n"
            "Path = parent.csa/child.csa\nSize = 1\nAttributes = A_ -rw-r--r--\n",
            "file/directory shadow",
        ),
    ],
)
def test_7z_technical_listing_fails_closed_on_unsafe_archive_metadata(
    member_section: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listing = f"archive metadata\n----------\n{member_section}"

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=listing, stderr="")

    monkeypatch.setattr(floodgate_module.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match=message):
        floodgate_module._list_7z_members(
            "/private/input.7z",
            executable="7z",
            timeout_seconds=1,
        )


def test_rating_snapshot_preserves_exact_id_hash_and_zero_loss_uncertainty() -> None:
    raw = _rating_html(
        [
            ("AKATSUKI_Hyohga+517c1da", "AKATSUKI_Hyohga", 4439, 25, 3),
            ("miao4+6d67de3", "miao4", 7398, 50, 0),
        ]
    )
    snapshot = parse_rating_snapshot(
        raw,
        source_url=(
            "https://wdoor.c.u-tokyo.ac.jp/shogi/x/rating/"
            "players-floodgate14-20260809.html"
        ),
        scope="14_day",
    )

    assert snapshot.snapshot_date == "2026-08-09"
    assert snapshot.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert snapshot.players[0].player_id == "AKATSUKI_Hyohga+517c1da"
    assert snapshot.players[0].games == 28
    assert not snapshot.players[0].zero_loss_uncertain
    assert snapshot.players[1].zero_loss_uncertain


def test_rating_snapshot_rejects_date_and_numeric_inconsistency() -> None:
    raw = _rating_html([("Alpha+aaaaaaa", "Alpha", 4300, 9, 1)])
    bad_rate = raw.replace(b">0.900<", b">0.100<")
    with pytest.raises(ValueError, match="inconsistent win rate"):
        parse_rating_snapshot(
            bad_rate,
            source_url="https://example.test/players-20260809.html",
            scope="test",
        )

    with pytest.raises(ValueError, match="URL date"):
        parse_rating_snapshot(
            raw,
            source_url="https://example.test/players-2025-08-09.html",
            scope="test",
        )


def test_csa_standard_position_promotion_times_names_and_candidate_only_contract() -> None:
    source = CsaSource(
        "2026/promotion.csa",
        _csa(moves=("+7776FU", "-3334FU", "+8822UM", "-3122GI")),
    )
    parsed = parse_csa_game(source)

    assert parsed.black_name == "Alpha"
    assert parsed.white_name == "Beta"
    assert parsed.event == "fixture"
    assert parsed.start_time == "2026/08/09 10:00:00"
    assert parsed.end_time == "2026/08/09 10:10:00"
    assert parsed.initial_position_mode == "PI"
    assert parsed.record.moves == ("7g7f", "3c3d", "8h2b+", "3a2b")
    assert parsed.move_times_seconds == (0, 1, 2, 3)
    assert parsed.record.termination is Termination.RESIGNATION
    assert parsed.record.winner == 1
    first = parsed.record.samples[0]
    assert first.policy == {}
    assert first.chosen_move == "7g7f"
    assert first.actor_best_move is None
    assert first.actor_source == ORIGINAL_CANDIDATE_SOURCE
    assert first.value_target == -1.0
    assert parsed.record.samples[1].value_target == 1.0
    assert first.teacher_policy is None
    assert first.teacher_value is None
    assert first.teacher_best_move is None
    assert first.teacher_variations is None

    # Candidate-only records remain directly usable by the reanalysis path:
    # chosen_move is actor evidence, while actor_best_move stays deliberately
    # unset because the public game did not prove it was best.
    revised = reanalyse_game(
        parsed.record,
        UniformEvaluator(),
        SearchConfig(simulations=1, root_min_visits=1, dirichlet_fraction=0),
        ReanalysisConfig(
            teacher_simulation_multiplier=2,
            minimum_teacher_simulations=2,
            reanalyse_fraction=1.0,
        ),
    )
    assert all(sample.teacher_policy is not None for sample in revised.samples)


def test_csa_explicit_position_and_piece_drop_are_legally_replayed() -> None:
    initial = Board("4k4/9/9/9/9/9/9/9/4K4 b P 1")
    parsed = parse_csa_game(
        CsaSource(
            "2026/drop.csa",
            _csa(
                position=initial.to_csa().rstrip(),
                moves=("+0055FU",),
                terminal="%TORYO",
            ),
        )
    )

    assert parsed.initial_position_mode == "explicit-rows"
    assert parsed.record.initial_sfen == initial.to_sfen()
    assert parsed.record.moves == ("P*5e",)
    assert parsed.record.samples[0].chosen_move == "P*5e"


def test_csa_repetition_interrupt_impasse_and_time_forfeit_are_not_collapsed() -> None:
    cycle = ("+5958OU", "-5152OU", "+5859OU", "-5251OU")
    repetition = parse_csa_game(
        CsaSource(
            "2026/repetition.csa",
            _csa(moves=cycle * 3, terminal="%SENNICHITE"),
        )
    )
    assert repetition.record.termination is Termination.REPETITION
    assert repetition.record.winner is None
    assert repetition.terminal_validation == "locally_verified_repetition"

    interrupted = parse_csa_game(
        CsaSource("2026/interrupted.csa", _csa(moves=(), terminal="%CHUDAN"))
    )
    impasse = parse_csa_game(
        CsaSource("2026/impasse.csa", _csa(moves=(), terminal="%JISHOGI"))
    )
    timeout = parse_csa_game(
        CsaSource("2026/timeout.csa", _csa(moves=(), terminal="%TIME_UP"))
    )
    assert interrupted.record.termination is Termination.INTERRUPTION
    assert impasse.record.termination is Termination.IMPASSE
    assert timeout.record.termination is Termination.TIME_FORFEIT
    assert timeout.record.winner == 1

    # Floodgate sometimes records the declaration action, its time, and the
    # same terminal result again.  Preserve that dialect without accepting a
    # conflicting second result.
    declaration_raw = _csa(moves=(), terminal="%KACHI").replace(
        b"%KACHI,'* terminal comment", b"%KACHI\nT0\n%KACHI"
    )
    declaration = parse_csa_game(
        CsaSource("2026/declaration.csa", declaration_raw)
    )
    assert declaration.record.termination is Termination.DECLARATION
    assert declaration.record.winner == 0
    assert declaration.terminal_time_seconds == 0
    assert declaration.terminal_record_count == 2
    assert declaration.terminal_validation == (
        "server_reported_declaration_not_locally_proven"
    )


def test_csa_fails_closed_on_wrong_side_illegal_move_false_repetition_and_unknown_code() -> None:
    with pytest.raises(ValueError, match="wrong side"):
        parse_csa_game(
            CsaSource("2026/wrong-side.csa", _csa(moves=("-3334FU",)))
        )
    with pytest.raises(ValueError, match=r"(?:invalid|illegal) CSA move"):
        parse_csa_game(
            CsaSource("2026/illegal.csa", _csa(moves=("+7775FU",)))
        )
    with pytest.raises(ValueError, match="not a legal repetition"):
        parse_csa_game(
            CsaSource(
                "2026/false-repetition.csa",
                _csa(moves=("+7776FU",), terminal="%SENNICHITE"),
            )
        )
    with pytest.raises(ValueError, match="unknown CSA terminal"):
        parse_csa_game(
            CsaSource("2026/unknown.csa", _csa(moves=(), terminal="%MYSTERY"))
        )


def test_engine_identity_is_unknown_unless_an_exact_nonshadowed_rule_exists() -> None:
    rule = EngineIdentityRule(
        rule_id="nagisa-v3-n150",
        player_names=("NAGISA_V3_N150",),
        family="NAGISA",
        version="V3",
        hardware="N150 as declared in rule source",
    )
    identities = resolve_engine_identities(
        ("NAGISA_V3_N150", "nagisa_v4_p1351"), (rule,)
    )

    assert identities["NAGISA_V3_N150"].family == "NAGISA"
    assert identities["NAGISA_V3_N150"].rule_id == "nagisa-v3-n150"
    assert identities["nagisa_v4_p1351"].family == "unknown"
    assert identities["nagisa_v4_p1351"].version == "unknown"
    assert identities["nagisa_v4_p1351"].hardware == "unknown"
    assert not identities["nagisa_v4_p1351"].inferred

    shadow = EngineIdentityRule(
        rule_id="shadow",
        player_names=("NAGISA_V3_N150",),
        family="other",
        version="other",
        hardware="other",
    )
    with pytest.raises(ValueError, match="shadowed"):
        resolve_engine_identities(("NAGISA_V3_N150",), (rule, shadow))


def _distinct_game(index: int, *, white: str = "Beta") -> CsaSource:
    board = Board()
    black_move = list(board.legal_moves())[index % len(list(board.legal_moves()))]
    black_csa = board.move32_from_move(black_move).to_csa()
    assert black_csa is not None
    board.apply_move(black_move)
    white_moves = list(board.legal_moves())
    white_move = white_moves[(index // 30) % len(white_moves)]
    white_csa = board.move32_from_move(white_move).to_csa()
    assert white_csa is not None
    return CsaSource(
        f"2026/game-{index:03d}.csa",
        _csa(
            white=white,
            moves=(black_csa, white_csa),
            nonce=f"game-{index:03d}",
        ),
    )


def test_corpus_is_create_only_deterministic_strong_on_both_sides_and_leak_free(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "wdoor2026.7z"
    archive.write_bytes(b"byte-identified archive fixture")
    sources = [_distinct_game(index) for index in range(120)]
    sources.append(_distinct_game(999, white="Gamma"))
    sources.append(CsaSource("2026/broken.csa", b"V2\nN+Alpha\nN-Beta\nPI\n+\n"))
    config = FloodgateCorpusConfig(min_rating=3900, min_games=20)
    rules = (
        EngineIdentityRule(
            rule_id="alpha-rule",
            player_names=("Alpha",),
            family="AlphaFamily",
            version="explicit-v1",
            hardware="explicit-hardware",
        ),
    )

    first_output = tmp_path / "corpus-a"
    second_output = tmp_path / "corpus-b"
    create_floodgate_corpus(
        first_output,
        archive=archive,
        csa_sources=sources,
        rating_snapshot=_snapshot(),
        config=config,
        identity_rules=rules,
    )
    create_floodgate_corpus(
        second_output,
        archive=archive,
        csa_sources=(source for source in sources),
        rating_snapshot=_snapshot(),
        config=config,
        identity_rules=rules,
    )

    first_manifest_bytes = (first_output / "manifest.json").read_bytes()
    assert first_manifest_bytes == (second_output / "manifest.json").read_bytes()
    manifest = json.loads(first_manifest_bytes)
    assert manifest["schema"] == CORPUS_SCHEMA
    assert manifest["rights"]["classification"] == "local-analysis-only"
    assert not manifest["rights"]["derived_checkpoint_redistribution_allowed"]
    assert manifest["label_contract"]["reanalysis_required"]
    assert not manifest["label_contract"]["original_move_is_teacher_truth"]
    assert not manifest["label_contract"]["direct_training_allowed"]
    assert manifest["inspection"]["selected_games"] == 120
    assert manifest["inspection"]["parsed_games"] == 121
    assert manifest["inspection"]["parse_failures"] == 1
    assert manifest["inspection"]["excluded_games"] == 2
    excluded = [
        row for row in manifest["inspection"]["rows"] if not row["included"]
    ]
    assert {tuple(row["exclusion_reasons"]) for row in excluded} == {
        ("csa_parse_failed",),
        ("white_rating_below_minimum",),
    }
    assert all(
        row["normalized_sample_positions"] == 0
        for row in manifest["leakage_guard"]["pairwise_intersections"]
    )
    assert manifest["leakage_guard"]["verified_zero_cross_split_overlap"]
    assert any(row["zero_loss_rating_uncertainty"] for row in manifest["games"])
    assert {row["split"] for row in manifest["games"]} == {
        "train",
        "validation",
        "sealed_final_test",
    }
    first_game = manifest["games"][0]
    assert first_game["black_identity"]["family"] == "AlphaFamily"
    assert first_game["white_identity"]["family"] == "unknown"

    positions_by_split: dict[str, set[str]] = {}
    candidate_count = 0
    for split in ("train", "validation", "sealed_final_test"):
        replay = first_output / "candidates" / f"{split}.jsonl"
        games = load_games(replay)
        positions_by_split[split] = {
            normalized_sfen(sample.sfen) for game in games for sample in game.samples
        }
        for game in games:
            for sample in game.samples:
                candidate_count += 1
                assert sample.policy == {}
                assert sample.chosen_move is not None
                assert sample.actor_best_move is None
                assert sample.teacher_policy is None
                assert sample.teacher_value is None
        sidecar = json.loads(
            replay.with_suffix(".jsonl.provenance.json").read_text(encoding="utf-8")
        )
        assert sidecar["schema"] == CANDIDATE_SIDECAR_SCHEMA
        assert sidecar["reanalysis_required"]
        assert not sidecar["direct_training_allowed"]
    assert candidate_count > 0
    split_names = sorted(positions_by_split)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            assert positions_by_split[left].isdisjoint(positions_by_split[right])

    with pytest.raises(FileExistsError):
        create_floodgate_corpus(
            first_output,
            archive=archive,
            csa_sources=sources,
            rating_snapshot=_snapshot(),
            config=config,
            identity_rules=rules,
        )
