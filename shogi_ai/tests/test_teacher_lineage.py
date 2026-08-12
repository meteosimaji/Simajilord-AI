from __future__ import annotations

import json

import pytest

from simajilord_shogi.cli import main
from simajilord_shogi.teacher_lineage import (
    MAIN_TEACHER_LINEAGES,
    PSV_RECORD_BYTES,
    PUBLIC_CORPORA,
    CorpusReuseDecision,
    CorpusSeedSamplingScope,
    PsvMoveFieldContract,
    corpus_lineage,
    teacher_lineage_contract,
)


def test_main_teacher_lineage_keeps_three_scorers_and_shared_ancestry() -> None:
    by_id = {teacher.teacher_id: teacher for teacher in MAIN_TEACHER_LINEAGES}

    assert set(by_id) == {
        "nagisa-v3.1",
        "suisho11plus-wcsc36-20260525-local",
        "soujou-tsec7-paid",
    }
    assert (
        by_id["nagisa-v3.1"].correlation_family
        == by_id["soujou-tsec7-paid"].correlation_family
    )
    assert by_id["suisho11plus-wcsc36-20260525-local"].ensemble_formula_public is False
    assert by_id["suisho11plus-wcsc36-20260525-local"].reported_position_count is None


def test_dlsuisho_unique_corpus_is_value_only_and_fails_closed_on_rights() -> None:
    corpus = corpus_lineage("dlsuisho15b-unique-public")

    assert corpus.data_bytes == corpus.position_records * PSV_RECORD_BYTES
    assert corpus.position_records == 14_668_949_437
    assert corpus.policy_target_available is False
    assert corpus.license_id is None
    assert (
        corpus.reuse_decision
        is CorpusReuseDecision.VALUE_ONLY_LOADER_AND_PERMISSION_REQUIRED
    )
    assert corpus.local_probe == {
        "records_checked": 100,
        "move16_zero_records": 100,
        "board_decode": "valid",
        "value_and_game_result_fields": "present",
        "conclusion": "sampled records provide value targets but no legal move/policy target",
    }


def test_mit_position_sources_require_fresh_qsearch_and_relabel() -> None:
    for corpus_id in (
        "nodchip-tanuki-nnue-pytorch-2024-07-30.1",
        "nodchip-shogi-hao-depth9",
    ):
        corpus = corpus_lineage(corpus_id)
        assert corpus.license_id == "MIT"
        assert corpus.position_records is not None
        assert corpus.position_records > 8_000_000_000
        assert corpus.data_bytes == corpus.position_records * PSV_RECORD_BYTES
        assert corpus.qsearch_leaf is False
        assert corpus.reuse_decision is CorpusReuseDecision.QSEARCH_RELABEL_AND_STORAGE_REQUIRED


def test_soujou_and_precursor_are_user_attested_local_value_only_sources() -> None:
    expected = {
        "soujou-team-datasets-1": 49_594_855_063,
        "dlsuisho15b-public-precursor": 16_667_054_287,
    }
    for corpus_id, positions in expected.items():
        corpus = corpus_lineage(corpus_id)
        assert corpus.position_records == positions
        assert corpus.data_bytes == positions * PSV_RECORD_BYTES
        assert corpus.policy_target_available is False
        assert corpus.license_id is None
        assert (
            corpus.reuse_decision
            is CorpusReuseDecision.LOCAL_ONLY_VALUE_NNUE_TRAINING_ALLOWED
        )
        assert (
            corpus.seed_sampling_scope
            is CorpusSeedSamplingScope.USER_ATTESTED_LOCAL_ONLY
        )
        assert (
            corpus.move_field_contract
            is PsvMoveFieldContract.ZERO_VALUE_ONLY_REQUIRED
        )
        assert corpus.local_probe is not None
        assert "direct local-only scalar target" in str(corpus.local_probe["conclusion"])


def test_soujou_datasets_2_is_counted_but_blocked_behind_gate_and_probe() -> None:
    corpus = corpus_lineage("soujou-team-datasets-2")

    assert corpus.position_records == 111_945_173_977
    assert corpus.data_bytes == corpus.position_records * PSV_RECORD_BYTES
    assert corpus.repository_bytes == 4_477_806_962_019
    assert corpus.license_id is None
    assert corpus.seed_sampling_scope is None
    assert (
        corpus.reuse_decision
        is CorpusReuseDecision.GATED_ACCESS_PERMISSION_AND_PROBE_REQUIRED
    )
    assert corpus.local_probe is not None
    assert corpus.local_probe["binary_range_http_status_without_authentication"] == 401
    assert corpus.local_probe["records_sampled"] == 0


def test_suisho5_public_sources_keep_normal_and_entering_king_contracts_distinct() -> None:
    validation = corpus_lineage("nodchip-shogi-suisho5-depth9-validation")
    entering = corpus_lineage("nodchip-shogi-suisho5-depth9-entering-king")

    assert validation.license_id == entering.license_id == "MIT"
    assert validation.qsearch_leaf is True
    assert validation.move_field_contract is PsvMoveFieldContract.ZERO_VALUE_ONLY_REQUIRED
    assert validation.position_records == 7_240_673
    assert entering.qsearch_leaf is False
    assert entering.move_field_contract is PsvMoveFieldContract.LEGAL_MOVE_REQUIRED
    assert entering.position_records == 500_006_044


def test_aobazero_games_are_versioned_seeds_not_direct_policy_labels() -> None:
    corpus = corpus_lineage("aobazero-public-selfplay-games")

    assert corpus.game_records == 78_685_549
    assert corpus.position_records is None
    assert corpus.policy_target_available is None
    assert corpus.license_id is None
    assert (
        corpus.reuse_decision
        is CorpusReuseDecision.GAME_REPLAY_RELABEL_AND_PERMISSION_REQUIRED
    )
    assert corpus.local_probe is not None
    assert corpus.local_probe["first_neural_game_number"] == 121_032
    assert corpus.local_probe["all_24_point_rule_generation_from_weight"] == "w4747"


def test_strategy_and_game_corpora_preserve_source_specific_contracts() -> None:
    komaochi = corpus_lineage("aoba-komaochi-public-selfplay-games")
    furibisha = corpus_lineage("aoba-furibisha-public-selfplay-games")
    gct = corpus_lineage("gct-wcsc31-public-training-data")
    qpd = corpus_lineage("qhapaq-pretty-daabi-training-kit")
    denryu = corpus_lineage("denryu-public-game-records")

    assert komaochi.game_records == 13_002_907
    assert furibisha.game_records == 22_028_029
    assert furibisha.policy_target_available is True
    assert gct.record_format.startswith("HCPE and HCPE3")
    assert gct.policy_target_available is True
    assert qpd.position_records == 915_204
    assert qpd.data_bytes == qpd.position_records * PSV_RECORD_BYTES
    assert qpd.local_probe is not None
    assert qpd.local_probe["original_training_repetition_factor"] == 20
    assert denryu.reuse_decision is CorpusReuseDecision.GAME_REPLAY_RELABEL_REQUIRED


def test_public_corpus_ids_are_unique_and_psv_counts_are_byte_aligned() -> None:
    corpus_ids = [corpus.corpus_id for corpus in PUBLIC_CORPORA]
    assert len(corpus_ids) == len(set(corpus_ids))

    for corpus in PUBLIC_CORPORA:
        if corpus.record_format == "YaneuraOu PackedSfenValue (40 bytes)":
            assert corpus.data_bytes is not None
            assert corpus.position_records is not None
            assert corpus.data_bytes == corpus.position_records * PSV_RECORD_BYTES


def test_teacher_lineage_contract_does_not_double_count_nagisa_and_soujou() -> None:
    contract = teacher_lineage_contract()

    assert contract["schema"] == "meteo-teacher-lineage-catalog-v1"
    independence = contract["independence_contract"]
    assert independence["nagisa_and_soujou_independent_corpus_votes"] is False
    assert independence["position_counts_must_be_deduplicated_across_lineage_families"] is True


def test_teacher_lineage_cli_prints_the_same_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["teacher-lineage"]) == 0

    assert json.loads(capsys.readouterr().out) == teacher_lineage_contract()
