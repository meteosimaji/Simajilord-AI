from __future__ import annotations

import hashlib
import http.client
import json
import math
import threading
import time
from pathlib import Path

import pytest
from rsshogi.core import Board

from simajilord_shogi.arena import MoveDecision
from simajilord_shogi.compute_interlock import InterlockedEvaluator, training_step
from simajilord_shogi.config import SearchConfig
from simajilord_shogi.evaluator import Evaluation, UniformEvaluator
from simajilord_shogi.human_gui import (
    ActiveCheckpointUsiEngine,
    CheckpointIdentity,
    CheckpointRegistry,
    ComputeInterlock,
    HumanGameLog,
    HumanGuiService,
    HumanPlayLeaseHandle,
    MeteoGuiHttpServer,
    SessionStore,
    TrainingStatus,
    TrainingStatusStore,
    VersionConflictError,
    _parse_usi_position_command,
)
from simajilord_shogi.trainer import TrainingInterlockConfig


class FirstLegalPlayer:
    def choose_move(self, board: Board) -> MoveDecision:
        move = sorted(candidate.to_usi() for candidate in board.legal_moves())[0]
        return MoveDecision(move=move, policy={move: 1.0}, value=0.0, source="test")


class LeaseInspectingUniformEvaluator:
    def __init__(self, interlock: ComputeInterlock) -> None:
        self.interlock = interlock
        self.observed_active_lease = False

    def evaluate(self, board: Board) -> Evaluation:
        self.observed_active_lease = self.interlock.snapshot().human_active_count == 1
        return UniformEvaluator().evaluate(board)


class NonFinitePlayer:
    def choose_move(self, board: Board) -> MoveDecision:
        move = sorted(candidate.to_usi() for candidate in board.legal_moves())[0]
        return MoveDecision(move=move, policy={move: 1.0}, value=math.nan, source="test")


def _checkpoint(root: Path, name: str, *, step: int) -> Path:
    checkpoint = root / name
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(
        json.dumps({"format_version": 1, "step": step, "model": {}}) + "\n",
        encoding="utf-8",
    )
    (checkpoint / "weights.safetensors").write_bytes(f"weights-{name}".encode())
    return checkpoint


def _service(
    tmp_path: Path,
    *,
    player: FirstLegalPlayer | NonFinitePlayer | None = None,
) -> tuple[HumanGuiService, CheckpointRegistry, Path]:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    first = _checkpoint(checkpoint_root, "generation-1", step=10)
    state_root = tmp_path / "gui-state"
    registry = CheckpointRegistry(checkpoint_root, state_root)
    registry.publish(first, generation=1)
    service = HumanGuiService(
        registry,
        TrainingStatusStore(state_root / "training-status.json"),
        SessionStore(state_root / "sessions"),
        HumanGameLog(tmp_path / "human-games.jsonl"),
        lambda _identity: player or FirstLegalPlayer(),
    )
    return service, registry, checkpoint_root


def _legal_moves(service: HumanGuiService, session_id: str) -> list[str]:
    session = service.get_session(session_id)
    value = service.session_payload(session)["legal_moves"]
    assert isinstance(value, list)
    assert all(isinstance(move, str) for move in value)
    return value


def test_checkpoint_generation_is_pinned_until_a_new_game(tmp_path: Path) -> None:
    service, registry, checkpoint_root = _service(tmp_path)
    first_game = service.new_session(0)
    assert first_game.pinned_checkpoint.generation == 1

    second = _checkpoint(checkpoint_root, "generation-2", step=20)
    registry.publish(second, generation=2)

    resumed = service.get_session(first_game.session_id)
    second_game = service.new_session(0)
    assert resumed.pinned_checkpoint.generation == 1
    assert resumed.pinned_checkpoint.checkpoint_sha256 == (
        first_game.pinned_checkpoint.checkpoint_sha256
    )
    assert second_game.pinned_checkpoint.generation == 2
    assert (
        second_game.pinned_checkpoint.checkpoint_sha256
        != first_game.pinned_checkpoint.checkpoint_sha256
    )


def test_legal_move_and_optimistic_session_version(tmp_path: Path) -> None:
    service, _registry, _root = _service(tmp_path)
    game = service.new_session(0)
    legal = service.session_payload(game)["legal_moves"]
    assert isinstance(legal, list) and legal

    advanced = service.move(game.session_id, str(legal[0]), expected_version=game.version)
    assert advanced.version == game.version + 2
    assert len(advanced.moves) == 2
    with pytest.raises(VersionConflictError):
        service.move(game.session_id, str(legal[0]), expected_version=game.version)
    with pytest.raises(ValueError, match=r"illegal move|malformed"):
        service.move(game.session_id, "9z9z", expected_version=advanced.version)


def test_human_white_receives_ai_opening_move(tmp_path: Path) -> None:
    service, _registry, _root = _service(tmp_path)
    game = service.new_session(1)
    payload = service.session_payload(game)
    assert game.version == 1
    assert len(game.moves) == 1
    assert payload["human_turn"] is True
    assert payload["turn"] == 1


def test_terminal_game_is_candidate_only_hash_chained_and_idempotent(tmp_path: Path) -> None:
    service, _registry, _root = _service(tmp_path)
    game = service.new_session(0)
    finished = service.resign(game.session_id, expected_version=game.version)
    assert finished.raw_recorded is True

    log_path = tmp_path / "human-games.jsonl"
    rows = log_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["candidate_only"] is True
    assert row["teacher_reanalysis_required"] is True
    assert row["deduplication_required"] is True
    assert row["forbidden_destinations"] == ["arena", "validation", "sealed_test"]
    assert row["pinned_checkpoint"]["generation"] == 1

    service.get_session(game.session_id)
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 1


def test_partial_human_game_row_fails_closed(tmp_path: Path) -> None:
    log = HumanGameLog(tmp_path / "human-games.jsonl")
    log.path.write_bytes(b'{"partial":')
    identity = CheckpointIdentity(
        generation=1,
        relative_path="checkpoint",
        checkpoint_sha256="a" * 64,
        metadata_sha256="b" * 64,
        weights_sha256="c" * 64,
        step=1,
        published_unix_seconds=time.time(),
    )
    from simajilord_shogi.human_gui import HumanSession

    board = Board()
    session = HumanSession(
        session_id="0" * 32,
        version=1,
        human_color=0,
        initial_sfen=board.to_sfen(),
        current_sfen=board.to_sfen(),
        moves=(),
        status="complete",
        winner=1,
        termination="resignation",
        pinned_checkpoint=identity,
        created_unix_seconds=time.time(),
        updated_unix_seconds=time.time(),
    )
    with pytest.raises(ValueError, match="partial row"):
        log.append_session(session)


def test_restart_resumes_old_pin_after_new_checkpoint_is_active(tmp_path: Path) -> None:
    service, registry, checkpoint_root = _service(tmp_path)
    original = service.new_session(0)
    second = _checkpoint(checkpoint_root, "generation-2", step=20)
    registry.publish(second, generation=2)

    restarted = HumanGuiService(
        registry,
        TrainingStatusStore(tmp_path / "gui-state" / "training-status.json"),
        SessionStore(tmp_path / "gui-state" / "sessions"),
        HumanGameLog(tmp_path / "human-games.jsonl"),
        lambda _identity: FirstLegalPlayer(),
    )
    resumed = restarted.get_session(original.session_id)
    assert resumed.pinned_checkpoint.generation == 1
    advanced = restarted.move(
        resumed.session_id,
        _legal_moves(restarted, resumed.session_id)[0],
        expected_version=resumed.version,
    )
    assert advanced.pinned_checkpoint.generation == 1


def test_concurrent_same_version_allows_exactly_one_update(tmp_path: Path) -> None:
    service, _registry, _root = _service(tmp_path)
    game = service.new_session(0)
    move = _legal_moves(service, game.session_id)[0]
    barrier = threading.Barrier(3)
    results: list[str] = []

    def run() -> None:
        barrier.wait()
        try:
            service.move(game.session_id, move, expected_version=game.version)
            results.append("ok")
        except VersionConflictError:
            results.append("conflict")

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    assert sorted(results) == ["conflict", "ok"]


def test_registry_rejects_partial_symlink_and_mutated_checkpoints(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    registry = CheckpointRegistry(checkpoint_root, tmp_path / "state")
    partial = checkpoint_root / "partial"
    partial.mkdir()
    (partial / "metadata.json").write_text('{"step":1,"model":{}}\n')
    with pytest.raises(ValueError, match="incomplete"):
        registry.publish(partial, generation=1)

    valid = _checkpoint(checkpoint_root, "valid", step=1)
    identity = registry.publish(valid, generation=1)
    (valid / "weights.safetensors").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed"):
        registry.verify(identity)

    target = _checkpoint(checkpoint_root, "target", step=2)
    link = checkpoint_root / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        registry.identify(link, generation=2)


def test_training_status_is_separate_hash_checked_state(tmp_path: Path) -> None:
    store = TrainingStatusStore(tmp_path / "training-status.json")
    status = TrainingStatus("running", 7, 0.25, "蒸留中", time.time(), 123)
    store.publish(status)
    assert store.read().generation == 7
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["status"]["progress"] = 0.75
    store.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.read()
    with pytest.raises(ValueError, match="progress"):
        TrainingStatus("running", 7, math.nan, "bad", time.time())


def test_nonfinite_ai_decision_fails_without_committing_human_move(tmp_path: Path) -> None:
    service, _registry, _root = _service(tmp_path, player=NonFinitePlayer())
    game = service.new_session(0)
    move = _legal_moves(service, game.session_id)[0]
    with pytest.raises(ValueError, match="finite"):
        service.move(game.session_id, move, expected_version=game.version)
    unchanged = service.get_session(game.session_id)
    assert unchanged.version == game.version
    assert unchanged.moves == ()


def test_http_server_requires_local_secret_header(tmp_path: Path) -> None:
    service, _registry, _root = _service(tmp_path)
    token = "t" * 40
    server = MeteoGuiHttpServer(("127.0.0.1", 0), service, access_token=token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/api/status")
        forbidden = connection.getresponse()
        forbidden.read()
        assert forbidden.status == 403
        connection.request("GET", "/api/status", headers={"X-Meteo-Token": token})
        allowed = connection.getresponse()
        body = json.loads(allowed.read())
        assert allowed.status == 200
        assert body["parallelism"]["mutable_model_state_shared"] is False
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_checkpoint_pointer_publication_hash_is_reproducible(tmp_path: Path) -> None:
    _service_value, registry, _root = _service(tmp_path)
    identity = registry.read_active()
    expected = hashlib.sha256(
        f"{identity.metadata_sha256}:{identity.weights_sha256}".encode()
    ).hexdigest()
    assert identity.checkpoint_sha256 == expected


def test_shogihome_usi_bridge_pins_until_next_usinewgame_and_logs_result(
    tmp_path: Path,
) -> None:
    _service_value, registry, checkpoint_root = _service(tmp_path)
    status_store = TrainingStatusStore(tmp_path / "gui-state" / "training-status.json")
    status_store.publish(
        TrainingStatus("running", 2, 0.5, "水匠11Plusで蒸留中", time.time(), 321)
    )
    log_path = tmp_path / "shogihome-human-games.jsonl"
    loaded: list[str] = []

    def load(identity: CheckpointIdentity) -> UniformEvaluator:
        loaded.append(identity.checkpoint_sha256)
        return UniformEvaluator()

    engine = ActiveCheckpointUsiEngine(
        registry,
        status_store,
        HumanGameLog(log_path),
        SearchConfig(
            simulations=1,
            temperature_moves=0,
            temperature=0.0,
            dirichlet_fraction=0.0,
            resign_threshold=None,
        ),
        evaluator_loader=load,
    )
    first_hash = registry.read_active().checkpoint_sha256
    assert any(response.startswith("id name Meteo g1") for response in engine.handle("usi"))
    engine.handle("usinewgame")
    assert loaded == [first_hash]
    engine.handle("position startpos moves 7g7f")
    first_go = engine.handle("go nodes 1")
    assert any("meteo_pin generation=1" in response for response in first_go)
    assert any("training=running generation=2 progress=50.0%" in response for response in first_go)

    second = _checkpoint(checkpoint_root, "generation-2", step=20)
    second_identity = registry.publish(second, generation=2)
    midgame_go = engine.handle("go nodes 1")
    assert any("meteo_pin generation=1" in response for response in midgame_go)
    assert engine.pinned_checkpoint.checkpoint_sha256 == first_hash

    engine.handle("gameover lose")
    row = json.loads(log_path.read_text(encoding="utf-8"))
    assert row["source"] == "shogihome_usi_bridge"
    assert row["engine_color"] == 1
    assert row["human_color"] == 0
    assert row["winner"] == 0
    assert row["gameover_result_from_engine_perspective"] == "lose"
    assert row["candidate_only"] is True
    assert row["shogihome_record_reconciliation_required"] is True
    assert row["usi_observation_may_omit_terminal_human_move"] is True
    assert row["pinned_checkpoint"]["checkpoint_sha256"] == first_hash

    engine.handle("usinewgame")
    assert engine.pinned_checkpoint.checkpoint_sha256 == second_identity.checkpoint_sha256
    assert loaded == [first_hash, second_identity.checkpoint_sha256]
    assert engine.compute_interlock.snapshot().human_active_count == 0
    engine.handle("quit")
    assert engine.compute_interlock.snapshot().human_active_count == 0


def test_usi_position_parser_legally_replays_history() -> None:
    initial, moves, resulting = _parse_usi_position_command("position startpos moves 7g7f 3c3d")
    assert initial == Board().to_sfen()
    assert moves == ("7g7f", "3c3d")
    assert Board(resulting).turn.value == 0
    with pytest.raises(ValueError, match="illegal USI history"):
        _parse_usi_position_command("position startpos moves 7g7f 7g7f")


def test_usi_gameover_before_engine_turn_is_logged_without_guessing_color(
    tmp_path: Path,
) -> None:
    _service_value, registry, _root = _service(tmp_path)
    log_path = tmp_path / "early-resignation.jsonl"
    engine = ActiveCheckpointUsiEngine(
        registry,
        TrainingStatusStore(tmp_path / "gui-state" / "training-status.json"),
        HumanGameLog(log_path),
        SearchConfig(simulations=1),
        evaluator_loader=lambda _identity: UniformEvaluator(),
    )
    engine.handle("usinewgame")
    engine.handle("gameover win")
    row = json.loads(log_path.read_text(encoding="utf-8"))
    assert row["engine_color"] is None
    assert row["human_color"] is None
    assert row["winner"] is None
    assert row["winner_requires_shogihome_reconciliation"] is True


def test_compute_interlock_allows_multiple_humans_and_heartbeats(
    tmp_path: Path,
) -> None:
    _service_value, registry, _root = _service(tmp_path)
    interlock = ComputeInterlock(registry.state_root)
    identity = registry.read_active()
    first = interlock.acquire_human(
        "1" * 32,
        identity,
        ttl_seconds=3,
        heartbeat_interval_seconds=1,
        wait_timeout_seconds=0,
    )
    second = interlock.acquire_human(
        "2" * 32,
        identity,
        ttl_seconds=3,
        heartbeat_interval_seconds=1,
        wait_timeout_seconds=0,
    )
    try:
        snapshot = interlock.snapshot()
        assert snapshot.active_human_sessions == ("1" * 32, "2" * 32)
        before = first.record.expires_unix_seconds
        assert first.heartbeat().expires_unix_seconds >= before
        observed_expiry = max(
            first.record.expires_unix_seconds, second.record.expires_unix_seconds
        )
        future = interlock.snapshot(now=observed_expiry + 1)
        assert future.human_active_count == 0
        assert future.expired_human_leases == 2
    finally:
        first.release()
        second.release()
    assert interlock.snapshot().human_active_count == 0
    resumed = interlock.acquire_human(
        "1" * 32,
        identity,
        ttl_seconds=3,
        heartbeat_interval_seconds=1,
        wait_timeout_seconds=0,
    )
    resumed.release()


def test_compute_interlock_excludes_human_play_and_training_steps(
    tmp_path: Path,
) -> None:
    _service_value, registry, _root = _service(tmp_path)
    interlock = ComputeInterlock(registry.state_root)
    identity = registry.read_active()
    human = interlock.acquire_human(
        "3" * 32,
        identity,
        ttl_seconds=3,
        heartbeat_interval_seconds=1,
        wait_timeout_seconds=0,
    )
    try:
        with pytest.raises(TimeoutError, match="human play is active"):
            interlock.training_step(generation=1, wait_timeout_seconds=0)
    finally:
        human.release()

    with interlock.training_step(
        generation=1,
        ttl_seconds=3,
        heartbeat_interval_seconds=1,
        wait_timeout_seconds=0,
    ) as training:
        assert interlock.snapshot().training_step_active_count == 1
        training.assert_healthy()
        with pytest.raises(TimeoutError, match="active training steps"):
            interlock.acquire_human(
                "4" * 32,
                identity,
                ttl_seconds=3,
                heartbeat_interval_seconds=1,
                wait_timeout_seconds=0,
            )
    assert interlock.snapshot().training_step_active_count == 0
    assert not any((registry.state_root / "training-step-leases").iterdir())

    with training_step(
        registry.state_root,
        generation=1,
        ttl_seconds=3,
        heartbeat_interval_seconds=1,
    ) as adapter_lease:
        adapter_lease.assert_healthy()
        assert interlock.snapshot().training_step_active_count == 1
    assert interlock.snapshot().training_step_active_count == 0


def test_compute_interlock_waits_outside_lock_then_acquires_human(
    tmp_path: Path,
) -> None:
    _service_value, registry, _root = _service(tmp_path)
    interlock = ComputeInterlock(registry.state_root)
    identity = registry.read_active()
    training = interlock.training_step(
        generation=1,
        ttl_seconds=3,
        heartbeat_interval_seconds=1,
        wait_timeout_seconds=0,
    )
    started = threading.Event()
    acquired: list[HumanPlayLeaseHandle] = []
    errors: list[BaseException] = []

    def acquire() -> None:
        started.set()
        try:
            acquired.append(
                interlock.acquire_human(
                    "7" * 32,
                    identity,
                    ttl_seconds=3,
                    heartbeat_interval_seconds=1,
                    wait_timeout_seconds=2,
                    poll_interval_seconds=0.01,
                )
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=acquire)
    thread.start()
    assert started.wait(timeout=1)
    training.release()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert errors == []
    assert len(acquired) == 1
    acquired[0].release()


def test_waiting_human_has_priority_between_serial_training_steps(
    tmp_path: Path,
) -> None:
    _service_value, registry, _root = _service(tmp_path)
    interlock = ComputeInterlock(registry.state_root)
    identity = registry.read_active()
    first_training = interlock.training_step(
        generation=1,
        ttl_seconds=3,
        heartbeat_interval_seconds=1,
        wait_timeout_seconds=0,
    )
    acquired: list[HumanPlayLeaseHandle] = []
    errors: list[BaseException] = []
    waiting = threading.Event()

    def acquire_human() -> None:
        waiting.set()
        try:
            acquired.append(
                interlock.acquire_human(
                    "8" * 32,
                    identity,
                    ttl_seconds=3,
                    heartbeat_interval_seconds=1,
                    wait_timeout_seconds=2,
                    poll_interval_seconds=0.01,
                )
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=acquire_human)
    thread.start()
    assert waiting.wait(timeout=1)
    time.sleep(0.03)
    first_training.release()
    with pytest.raises(TimeoutError, match="human play is active"):
        interlock.training_step(
            generation=2,
            ttl_seconds=3,
            heartbeat_interval_seconds=1,
            wait_timeout_seconds=0,
        )
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert errors == []
    assert len(acquired) == 1
    acquired[0].release()


def test_training_steps_are_serialized_on_one_metal_state_root(tmp_path: Path) -> None:
    interlock = ComputeInterlock(tmp_path / "state")
    first = interlock.training_step(
        ttl_seconds=3,
        heartbeat_interval_seconds=1,
        wait_timeout_seconds=0,
    )
    try:
        with pytest.raises(TimeoutError, match="another training step is active"):
            interlock.training_step(wait_timeout_seconds=0)
    finally:
        first.release()


def test_background_evaluator_yields_to_human_play_between_batches(
    tmp_path: Path,
) -> None:
    _service_value, registry, _root = _service(tmp_path)
    interlock = ComputeInterlock(registry.state_root)
    human = interlock.acquire_human(
        "9" * 32,
        registry.read_active(),
        ttl_seconds=3,
        heartbeat_interval_seconds=1,
        wait_timeout_seconds=0,
    )
    released = threading.Event()

    def release_human() -> None:
        time.sleep(0.06)
        human.release()
        released.set()

    thread = threading.Thread(target=release_human)
    thread.start()
    evaluator = InterlockedEvaluator(
        UniformEvaluator(),
        TrainingInterlockConfig(
            state_root=registry.state_root,
            ttl_seconds=3,
            heartbeat_interval_seconds=1,
            wait_timeout_seconds=2,
            poll_interval_seconds=0.01,
        ),
    )
    started = time.monotonic()
    evaluations = evaluator.evaluate_batch([Board(), Board()])
    elapsed = time.monotonic() - started
    thread.join(timeout=2)

    assert released.is_set()
    assert elapsed >= 0.04
    assert len(evaluations) == 2
    assert all(sum(result.policy.values()) == pytest.approx(1.0) for result in evaluations)
    assert interlock.snapshot().training_step_active_count == 0


def test_compute_interlock_corruption_and_symlinks_fail_closed(
    tmp_path: Path,
) -> None:
    _service_value, registry, _root = _service(tmp_path)
    interlock = ComputeInterlock(registry.state_root)
    human_root = registry.state_root / "human-play-leases"
    corrupt = human_root / f"{'5' * 32}.json"
    corrupt.write_text('{"partial":', encoding="utf-8")
    with pytest.raises(ValueError):
        interlock.snapshot()
    corrupt.unlink()

    target = tmp_path / "outside.json"
    target.write_text("{}\n", encoding="utf-8")
    symlink = human_root / f"{'6' * 32}.json"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        interlock.training_step(wait_timeout_seconds=0)


def test_registry_and_interlock_reject_symlinked_roots(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    checkpoint_link = tmp_path / "checkpoint-link"
    checkpoint_link.symlink_to(checkpoint_root, target_is_directory=True)
    with pytest.raises(ValueError, match="checkpoint_root"):
        CheckpointRegistry(checkpoint_link, tmp_path / "state")

    state_root = tmp_path / "real-state"
    state_root.mkdir()
    state_link = tmp_path / "state-link"
    state_link.symlink_to(state_root, target_is_directory=True)
    with pytest.raises(ValueError, match="state_root"):
        ComputeInterlock(state_link)


def test_shogihome_keeps_game_pin_but_owns_compute_only_during_go(
    tmp_path: Path,
) -> None:
    _service_value, registry, _root = _service(tmp_path)
    interlock = ComputeInterlock(registry.state_root)
    evaluator = LeaseInspectingUniformEvaluator(interlock)
    engine = ActiveCheckpointUsiEngine(
        registry,
        TrainingStatusStore(registry.state_root / "training-status.json"),
        HumanGameLog(tmp_path / "lease-game.jsonl"),
        SearchConfig(simulations=1),
        evaluator_loader=lambda _identity: evaluator,
        compute_interlock=interlock,
    )
    assert interlock.snapshot().human_active_count == 0
    engine.handle("usinewgame")
    engine.handle("position startpos")
    assert interlock.snapshot().human_active_count == 0
    with interlock.training_step(
        ttl_seconds=3,
        heartbeat_interval_seconds=1,
        wait_timeout_seconds=0,
    ) as between_searches:
        between_searches.assert_healthy()
    responses = engine.handle("go nodes 1")
    assert len(responses) >= 1
    assert evaluator.observed_active_lease is True
    assert interlock.snapshot().human_active_count == 0
    with interlock.training_step(
        ttl_seconds=3,
        heartbeat_interval_seconds=1,
        wait_timeout_seconds=0,
    ) as after_search:
        after_search.assert_healthy()
    engine.handle("gameover draw")
    assert interlock.snapshot().human_active_count == 0


def test_shogihome_usi_rejects_a_branched_position_story(tmp_path: Path) -> None:
    _service_value, registry, _root = _service(tmp_path)
    engine = ActiveCheckpointUsiEngine(
        registry,
        TrainingStatusStore(registry.state_root / "training-status.json"),
        HumanGameLog(tmp_path / "branched-game.jsonl"),
        SearchConfig(simulations=1),
        evaluator_loader=lambda _identity: UniformEvaluator(),
    )
    engine.handle("usinewgame")
    engine.handle("position startpos moves 7g7f")
    with pytest.raises(ValueError, match="branched"):
        engine.handle("position startpos moves 2g2f")
    engine.handle("quit")
