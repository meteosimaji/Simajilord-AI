"""Audited teacher ancestry and public-corpus reuse constraints for Meteo.

The catalog separates an author's reported training lineage from a byte-level
identity that Meteo can actually consume.  Public downloadability is not a
license grant, a PackedSfenValue record is not necessarily a policy target,
and a corpus shared by two teachers must not be counted as two independent
sources of evidence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import cast

TEACHER_LINEAGE_SCHEMA = "meteo-teacher-lineage-catalog-v1"
PSV_RECORD_BYTES = 40


class CorpusReuseDecision(StrEnum):
    """Fail-closed result for the next technically and legally valid use."""

    LOCAL_ONLY_VALUE_NNUE_TRAINING_ALLOWED = "local_only_value_nnue_training_allowed"
    VALUE_ONLY_LOADER_AND_PERMISSION_REQUIRED = "value_only_loader_and_permission_required"
    QSEARCH_RELABEL_AND_STORAGE_REQUIRED = "qsearch_relabel_and_storage_required"
    LOCAL_ONLY_POSITION_RELABEL_REQUIRED = "local_only_position_relabel_required"
    GATED_ACCESS_PERMISSION_AND_PROBE_REQUIRED = (
        "gated_access_permission_and_probe_required"
    )
    GAME_REPLAY_RELABEL_REQUIRED = "game_replay_relabel_required"
    GAME_REPLAY_RELABEL_AND_PERMISSION_REQUIRED = (
        "game_replay_relabel_and_permission_required"
    )
    SOURCE_DOCUMENTATION_ONLY = "source_documentation_only"


class CorpusSeedSamplingScope(StrEnum):
    """Rights scope under which immutable PSV ranges may be sampled."""

    PUBLIC_LICENSED = "public_licensed"
    USER_ATTESTED_LOCAL_ONLY = "user_attested_local_only"


class PsvMoveFieldContract(StrEnum):
    """Reviewed meaning of the Move16 field in a directly sampled PSV source."""

    LEGAL_MOVE_REQUIRED = "legal_move_required"
    ZERO_VALUE_ONLY_REQUIRED = "zero_value_only_required"


@dataclass(frozen=True, slots=True)
class PublicCorpusLineage:
    corpus_id: str
    name: str
    repository_url: str
    revision: str | None
    repository_bytes: int | None
    data_bytes: int | None
    position_records: int | None
    record_format: str
    label_capability: str
    policy_target_available: bool | None
    qsearch_leaf: bool | None
    deduplicated: bool | None
    license_id: str | None
    license_evidence_url: str | None
    reuse_decision: CorpusReuseDecision
    required_before_training: tuple[str, ...]
    evidence: tuple[str, ...]
    local_probe: dict[str, object] | None = None
    game_records: int | None = None
    seed_sampling_scope: CorpusSeedSamplingScope | None = None
    move_field_contract: PsvMoveFieldContract | None = None
    sample_filename_pattern: str | None = None

    def __post_init__(self) -> None:
        if not self.corpus_id or not self.name or not self.repository_url:
            raise ValueError("corpus identity fields must not be empty")
        for label, value in (
            ("repository_bytes", self.repository_bytes),
            ("data_bytes", self.data_bytes),
            ("position_records", self.position_records),
            ("game_records", self.game_records),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{label} must be positive when known")
        if (
            self.data_bytes is not None
            and self.position_records is not None
            and self.record_format == "YaneuraOu PackedSfenValue (40 bytes)"
        ):
            if self.data_bytes // PSV_RECORD_BYTES != self.position_records:
                raise ValueError("PSV bytes and record count disagree")
            if self.data_bytes % PSV_RECORD_BYTES:
                raise ValueError("PSV byte count is not record aligned")
        if self.license_id is None and self.license_evidence_url is not None:
            raise ValueError("license evidence requires a reviewed license identifier")
        if (self.seed_sampling_scope is None) != (self.move_field_contract is None):
            raise ValueError("seed sampling scope and Move16 contract must be set together")
        if self.sample_filename_pattern is not None and self.seed_sampling_scope is None:
            raise ValueError("a sample filename pattern requires a seed sampling scope")
        if self.seed_sampling_scope is not None:
            if self.revision is None:
                raise ValueError("a directly sampled corpus must pin a repository revision")
            if "PackedSfenValue" not in self.record_format:
                raise ValueError("direct seed sampling is restricted to PackedSfenValue")
        if (
            self.seed_sampling_scope is CorpusSeedSamplingScope.PUBLIC_LICENSED
            and self.license_id is None
        ):
            raise ValueError("publicly sampled data requires a reviewed license")

    def to_dict(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            json.loads(json.dumps(asdict(self), allow_nan=False, sort_keys=True)),
        )


@dataclass(frozen=True, slots=True)
class TeacherTrainingLineage:
    teacher_id: str
    name: str
    learner: str
    architecture: str
    direct_teacher_models: tuple[str, ...]
    corpus_families: tuple[str, ...]
    reported_position_count: int | None
    reported_position_count_kind: str
    correlation_family: str
    exact_training_bytes_public: bool
    ensemble_formula_public: bool | None
    evidence: tuple[str, ...]
    caveats: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.teacher_id or not self.name or not self.correlation_family:
            raise ValueError("teacher lineage identity fields must not be empty")
        if len(set(self.direct_teacher_models)) != len(self.direct_teacher_models):
            raise ValueError("direct teacher models must be unique")
        if len(set(self.corpus_families)) != len(self.corpus_families):
            raise ValueError("teacher corpus families must be unique")
        if self.reported_position_count is not None and self.reported_position_count < 1:
            raise ValueError("reported position count must be positive when known")

    def to_dict(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            json.loads(json.dumps(asdict(self), allow_nan=False, sort_keys=True)),
        )


_DLSUISHO_UNIQUE_DATA_BYTES = 586_757_977_480
_DLSUISHO_PRECURSOR_RAW_BYTES = 666_682_171_480
_SOUJOU_DATASETS_1_BYTES = 1_983_794_202_520

PUBLIC_CORPORA: tuple[PublicCorpusLineage, ...] = (
    PublicCorpusLineage(
        corpus_id="soujou-team-datasets-1",
        name="Soujou team datasets_1, DL-Suisho relabelled and deduplicated",
        repository_url="https://huggingface.co/datasets/sojoteam/datasets_1",
        revision="4dfad115d4a808ebe20b6f65f6416ad75a69a6e7",
        repository_bytes=1_983_794_205_693,
        data_bytes=_SOUJOU_DATASETS_1_BYTES,
        position_records=49_594_855_063,
        record_format="YaneuraOu PackedSfenValue (40 bytes)",
        label_capability="DL-Suisho scalar value; sampled Move16 fields are zero",
        policy_target_available=False,
        qsearch_leaf=True,
        deduplicated=True,
        license_id=None,
        license_evidence_url=None,
        reuse_decision=CorpusReuseDecision.LOCAL_ONLY_VALUE_NNUE_TRAINING_ALLOWED,
        required_before_training=(
            "require an explicit operator acknowledgement of user-attested local reuse",
            "keep downloaded ranges, raw annotations, and derived checkpoints local-only",
            "pin every streamed shard by repository revision, byte count, and LFS SHA-256",
            "validate Move16=0, board decoding, score, result, and record alignment",
            "reserve an immutable held-out range before the first optimizer update",
            "use the historical scalar only in the value-only NNUE route; it is not policy",
            "obtain a public redistribution grant before publishing derived checkpoints",
        ),
        evidence=(
            "Hugging Face repository API and README at the pinned revision",
            "publisher reports nodchip, AobaZero, and DL-Suisho policy-expanded positions",
            "publisher reports qsearch, DL-Suisho relabel, shuffle, and deduplication",
            "user reports that local reuse was confirmed as acceptable on 2026-08-11",
        ),
        local_probe={
            "records_checked": 256,
            "move16_zero_records": 256,
            "board_decode": "valid",
            "score_range": [-4858, 32000],
            "conclusion": (
                "direct local-only scalar target for NAGISA-style value NNUE; "
                "no policy target"
            ),
        },
        seed_sampling_scope=CorpusSeedSamplingScope.USER_ATTESTED_LOCAL_ONLY,
        move_field_contract=PsvMoveFieldContract.ZERO_VALUE_ONLY_REQUIRED,
        sample_filename_pattern=r"split_\d{3}\.bin",
    ),
    PublicCorpusLineage(
        corpus_id="soujou-team-datasets-2",
        name="Soujou team datasets_2, two-stage DL-Suisho policy expansion",
        repository_url="https://huggingface.co/datasets/sojoteam/datasets_2",
        revision="b2a90707a7afc932ae05f1923219ff20c6b5d138",
        repository_bytes=4_477_806_962_019,
        data_bytes=4_477_806_959_080,
        position_records=111_945_173_977,
        record_format="YaneuraOu PackedSfenValue (40 bytes)",
        label_capability=(
            "datasets_1 plus two successive DL-Suisho policy >=10% expansions; "
            "binary fields are not yet sampled"
        ),
        policy_target_available=None,
        qsearch_leaf=True,
        deduplicated=True,
        license_id=None,
        license_evidence_url=None,
        reuse_decision=(
            CorpusReuseDecision.GATED_ACCESS_PERMISSION_AND_PROBE_REQUIRED
        ),
        required_before_training=(
            "obtain Hugging Face manual-gate access without recording an access token",
            "obtain or record an explicit local-use and derived-checkpoint rights statement",
            "range-probe immutable shards for board decoding, Move16, score, result, and qsearch",
            "deduplicate against datasets_1 and every nodchip ancestor before unique counting",
            "discard historical labels and generate current three-teacher score matrices",
        ),
        evidence=(
            "Hugging Face API and README at the pinned revision",
            "publisher reports datasets_1 plus two-stage DL-Suisho policy expansion, "
            "qsearch, shuffle, and deduplication",
            "complete 226-entry tree totals 111,945,173,977 aligned PSV records",
        ),
        local_probe={
            "metadata_and_readme": "public",
            "binary_range_http_status_without_authentication": 401,
            "binary_access_error": "GatedRepo",
            "records_sampled": 0,
            "conclusion": "scale is verified but content is not an optimizer input",
        },
    ),
    PublicCorpusLineage(
        corpus_id="dlsuisho15b-unique-public",
        name="Knowledge distilled dataset by DL Suisho 15b, deduplicated",
        repository_url=(
            "https://huggingface.co/datasets/"
            "washiun/Knowledge_distilled_dataset_by_DLSuisho15b_unique"
        ),
        revision="5da309f4de4091cfb004eff94da97d49e3268aa2",
        repository_bytes=586_757_979_941,
        data_bytes=_DLSUISHO_UNIQUE_DATA_BYTES,
        position_records=_DLSUISHO_UNIQUE_DATA_BYTES // PSV_RECORD_BYTES,
        record_format="YaneuraOu PackedSfenValue (40 bytes)",
        label_capability="value_only_in_sampled_records",
        policy_target_available=False,
        qsearch_leaf=True,
        deduplicated=True,
        license_id=None,
        license_evidence_url=None,
        reuse_decision=CorpusReuseDecision.VALUE_ONLY_LOADER_AND_PERMISSION_REQUIRED,
        required_before_training=(
            "obtain an explicit dataset reuse/license statement",
            "implement a value-only PSV path with a zero policy-loss mask",
            "stream selected immutable shards from storage larger than the local free space",
            "create board-hash train/calibration/held-out split receipts before optimization",
            "never report these records as policy-labelled positions",
        ),
        evidence=(
            "Hugging Face repository API at the pinned revision",
            "publisher README for the approximately 16-billion-record precursor",
            "Soujou WCSC36 appeal reporting 24B to 14.5B after deduplication",
        ),
        local_probe={
            "records_checked": 100,
            "move16_zero_records": 100,
            "board_decode": "valid",
            "value_and_game_result_fields": "present",
            "conclusion": "sampled records provide value targets but no legal move/policy target",
        },
    ),
    PublicCorpusLineage(
        corpus_id="dlsuisho15b-public-precursor",
        name="Knowledge distilled dataset by DL Suisho 15b",
        repository_url=(
            "https://huggingface.co/datasets/"
            "penguinkumimanu/Knowledge_distilled_dataset_by_DLSuisho15b"
        ),
        revision="aefe43c547f4230f9d5d16dda671c61c7c28b796",
        repository_bytes=855_096_289_587,
        data_bytes=_DLSUISHO_PRECURSOR_RAW_BYTES,
        position_records=_DLSUISHO_PRECURSOR_RAW_BYTES // PSV_RECORD_BYTES,
        record_format="YaneuraOu PackedSfenValue (40 bytes)",
        label_capability="DL-Suisho value labels; sampled Move16 fields are zero",
        policy_target_available=False,
        qsearch_leaf=True,
        deduplicated=False,
        license_id=None,
        license_evidence_url=None,
        reuse_decision=CorpusReuseDecision.LOCAL_ONLY_VALUE_NNUE_TRAINING_ALLOWED,
        required_before_training=(
            "require an explicit operator acknowledgement of user-attested local reuse",
            "keep downloaded ranges, raw annotations, and derived checkpoints local-only",
            "pin each streamed file/range and validate the Move16=0 value-only contract",
            "use the historical scalar only in a value-only NNUE learner, never as policy",
            "do not count the bundled Suisho5 archive as additional independent positions",
            "do not add these records to datasets_1 counts without cross-corpus deduplication",
            "obtain a public redistribution grant before publishing derived checkpoints",
        ),
        evidence=(
            "publisher README: Hao qsearch shuffle, DL Suisho 15b relabel, Eval_Coef=600",
            "pinned file tree: 16,667,054,287 raw PSV records plus source archives",
            "user reports that local reuse was confirmed as acceptable on 2026-08-11",
        ),
        local_probe={
            "records_checked": 512,
            "move16_zero_records": 512,
            "board_decode": "valid",
            "groups_checked": ["hao_depth_9_shuffled", "shuffled"],
            "conclusion": (
                "direct local-only scalar target for value NNUE; overlaps the primary "
                "datasets_1 lineage and is not an additive policy corpus"
            ),
        },
        seed_sampling_scope=CorpusSeedSamplingScope.USER_ATTESTED_LOCAL_ONLY,
        move_field_contract=PsvMoveFieldContract.ZERO_VALUE_ONLY_REQUIRED,
        sample_filename_pattern=r"(?:hao_depth_9_shuffled|shuffled_).+\.bin",
    ),
    PublicCorpusLineage(
        corpus_id="aobazero-public-selfplay-games",
        name="AobaZero public self-play game archives",
        repository_url="http://www.yss-aya.com/aobazero/",
        revision="site-snapshot-2026-08-11T21:25+09:00-weight-w4753",
        repository_bytes=None,
        data_bytes=None,
        position_records=None,
        record_format="AobaZero CSA self-play archives (generation-dependent comments)",
        label_capability=(
            "game result plus generation-dependent search visits, raw policy, and value; "
            "not a current three-teacher score matrix"
        ),
        policy_target_available=None,
        qsearch_leaf=False,
        deduplicated=False,
        license_id=None,
        license_evidence_url=None,
        reuse_decision=(
            CorpusReuseDecision.GAME_REPLAY_RELABEL_AND_PERMISSION_REQUIRED
        ),
        required_before_training=(
            "obtain an explicit reuse statement for the externally hosted game archives",
            "pin every selected archive by URL, byte count, and SHA-256 because the "
            "folders are mutable",
            "exclude games 0 through 121031 from neural-teacher claims because they "
            "used random policy/value",
            "parse each weight generation under its recorded comment and rule contract",
            "split by complete game and source weight before deriving any positions",
            "reanalyse selected positions with the current Meteo three-teacher protocol",
        ),
        evidence=(
            "AobaZero official status and archive page captured 2026-08-11",
            "kobanium/aobazero README: aobaz is GPLv3 and other repository code is public domain; "
            "this does not explicitly license external game archives",
        ),
        local_probe={
            "official_game_count_at_snapshot": 78_685_549,
            "current_weight_at_snapshot": "w4753",
            "average_plies_recent_1m_games": 110.1,
            "archive_unit_approx_compressed_bytes": 120_000_000,
            "first_neural_game_number": 121_032,
            "raw_nn_value_and_policy_comments_from_weight": "w4201",
            "board_history_equivalence_change_from_weight": "w4659",
            "all_24_point_rule_generation_from_weight": "w4747",
        },
        game_records=78_685_549,
    ),
    PublicCorpusLineage(
        corpus_id="aoba-komaochi-public-selfplay-games",
        name="Aoba Komaochi seven-rule self-play game archives",
        repository_url="http://www.yss-aya.com/komaochi/index.html",
        revision="site-snapshot-2026-08-11-weight-w1250",
        repository_bytes=None,
        data_bytes=None,
        position_records=None,
        record_format="Aoba CSA self-play archives with seven handicap classes",
        label_capability="game result and generation-dependent MCTS search comments",
        policy_target_available=None,
        qsearch_leaf=False,
        deduplicated=False,
        license_id=None,
        license_evidence_url=None,
        reuse_decision=(
            CorpusReuseDecision.GAME_REPLAY_RELABEL_AND_PERMISSION_REQUIRED
        ),
        required_before_training=(
            "keep handicap rule classes separate and use even-game records only for the "
            "current standard-shogi Meteo model",
            "exclude games 0 through 500007 from neural-teacher claims",
            "prefer games from w40 and 900k games onward as recommended by the publisher",
            "exclude the deleted w745 through w761 generation range",
            "pin complete games by archive, weight generation, hash, and rule class",
            "reanalyse selected positions with current teachers and rule adjudication",
        ),
        evidence=(
            "official Aoba Komaochi status, archive, and training-history page",
            "publisher reports seven simultaneous rule classes and known bad generations",
        ),
        local_probe={
            "official_game_count_at_snapshot": 13_002_907,
            "current_weight_at_snapshot": "w1250",
            "first_neural_game_number": 500_008,
            "publisher_recommended_minimum_game_number": 900_000,
            "deleted_bad_generation_games": "7,940,000 through 8,100,000",
        },
        game_records=13_002_907,
    ),
    PublicCorpusLineage(
        corpus_id="aoba-furibisha-public-selfplay-games",
        name="Aoba Furibisha strategy-conditioned self-play game archives",
        repository_url="http://www.yss-aya.com/furibisha/",
        revision="site-snapshot-2026-08-11-weight-w2195",
        repository_bytes=None,
        data_bytes=None,
        position_records=None,
        record_format="Aoba CSA self-play archives with rook-file conditioning",
        label_capability="MCTS visit distribution, game result, and strategy-conditioning metadata",
        policy_target_available=True,
        qsearch_leaf=False,
        deduplicated=False,
        license_id=None,
        license_evidence_url=None,
        reuse_decision=(
            CorpusReuseDecision.GAME_REPLAY_RELABEL_AND_PERMISSION_REQUIRED
        ),
        required_before_training=(
            "exclude games 0 through 100028 from neural-teacher claims",
            "preserve requested and realized rook-file conditioning as group metadata",
            "do not use the played move as a best-move label; the publisher says to use "
            "maximum visits when a single best move is required",
            "treat strategy bonuses as sampling metadata rather than objective game value",
            "reanalyse every retained position with current three-teacher labels",
        ),
        evidence=(
            "official Aoba Furibisha status, archive, and training-history page",
            "publisher explicitly distinguishes the played noisy move from maximum visits",
        ),
        local_probe={
            "official_game_count_at_snapshot": 22_028_029,
            "current_weight_at_snapshot": "w2195",
            "first_neural_game_number": 100_029,
            "no_resignation_from_weight": "w978",
            "current_training_playouts": "generation dependent; up to average 3200",
        },
        game_records=22_028_029,
    ),
    PublicCorpusLineage(
        corpus_id="nodchip-tanuki-nnue-pytorch-2024-07-30.1",
        name="tanuki-.nnue-pytorch-2024-07-30.1 training positions",
        repository_url=(
            "https://huggingface.co/datasets/nodchip/"
            "tanuki-.nnue-pytorch-2024-07-30.1"
        ),
        revision="59fd246d3f85d51707a62c89531564a1a8aeb793",
        repository_bytes=320_002_295_568,
        data_bytes=320_002_292_200,
        position_records=8_000_057_305,
        record_format="YaneuraOu PackedSfenValue (40 bytes)",
        label_capability="search move and scalar value PSV source",
        policy_target_available=True,
        qsearch_leaf=False,
        deduplicated=False,
        license_id="MIT",
        license_evidence_url=(
            "https://huggingface.co/datasets/nodchip/"
            "tanuki-.nnue-pytorch-2024-07-30.1"
        ),
        reuse_decision=CorpusReuseDecision.QSEARCH_RELABEL_AND_STORAGE_REQUIRED,
        required_before_training=(
            "stream or attach at least 320 GB of source storage",
            "shuffle and split by board/game lineage",
            "move positions to qsearch leaves",
            "relabel values and candidate moves with the current Meteo teacher protocol",
        ),
        evidence=("Hugging Face MIT dataset card and pinned file tree",),
        seed_sampling_scope=CorpusSeedSamplingScope.PUBLIC_LICENSED,
        move_field_contract=PsvMoveFieldContract.LEGAL_MOVE_REQUIRED,
        sample_filename_pattern=r".+\.bin",
    ),
    PublicCorpusLineage(
        corpus_id="nodchip-shogi-hao-depth9",
        name="Hao depth-9 training positions",
        repository_url="https://huggingface.co/datasets/nodchip/shogi_hao_depth9",
        revision="7bc19a9e880ea307a52c57f57ea6c752301b25bc",
        repository_bytes=320_002_982_670,
        data_bytes=320_002_979_440,
        position_records=8_000_074_486,
        record_format="YaneuraOu PackedSfenValue (40 bytes)",
        label_capability="search move and scalar value PSV source",
        policy_target_available=True,
        qsearch_leaf=False,
        deduplicated=False,
        license_id="MIT",
        license_evidence_url="https://huggingface.co/datasets/nodchip/shogi_hao_depth9",
        reuse_decision=CorpusReuseDecision.QSEARCH_RELABEL_AND_STORAGE_REQUIRED,
        required_before_training=(
            "stream or attach at least 320 GB of source storage",
            "shuffle the publisher-declared unshuffled records",
            "move positions to qsearch leaves",
            "relabel values and candidate moves with the current Meteo teacher protocol",
        ),
        evidence=(
            "Hugging Face MIT dataset card explicitly states unshuffled and not qsearch-leaf",
        ),
        seed_sampling_scope=CorpusSeedSamplingScope.PUBLIC_LICENSED,
        move_field_contract=PsvMoveFieldContract.LEGAL_MOVE_REQUIRED,
        sample_filename_pattern=r".+\.bin",
    ),
    PublicCorpusLineage(
        corpus_id="nodchip-shogi-suisho5-depth9-validation",
        name="Suisho5 depth-9 qsearch-leaf validation positions",
        repository_url="https://huggingface.co/datasets/nodchip/shogi_suisho5_depth9",
        revision="a399f456222756baa2c742a8f90ab7b9edcd6faf",
        repository_bytes=184_925_809_788,
        data_bytes=289_626_920,
        position_records=7_240_673,
        record_format="YaneuraOu PackedSfenValue (40 bytes)",
        label_capability="Suisho5 scalar value; sampled validation Move16 fields are zero",
        policy_target_available=False,
        qsearch_leaf=True,
        deduplicated=False,
        license_id="MIT",
        license_evidence_url=(
            "https://huggingface.co/datasets/nodchip/shogi_suisho5_depth9"
        ),
        reuse_decision=CorpusReuseDecision.QSEARCH_RELABEL_AND_STORAGE_REQUIRED,
        required_before_training=(
            "sample the direct shuffled.bin validation file without downloading the 7z set",
            "discard historical values and regenerate current three-teacher labels",
            "attach enough storage and verify the multipart archive before using the full set",
        ),
        evidence=(
            "Hugging Face MIT dataset card states shuffled and qsearch-PV-leaf converted",
            "pinned repository exposes one 289,626,920-byte direct validation PSV file",
        ),
        local_probe={
            "records_checked": 256,
            "move16_zero_records": 256,
            "board_decode": "valid",
            "conclusion": "direct validation file is value-only in the sample",
        },
        seed_sampling_scope=CorpusSeedSamplingScope.PUBLIC_LICENSED,
        move_field_contract=PsvMoveFieldContract.ZERO_VALUE_ONLY_REQUIRED,
        sample_filename_pattern=r"shuffled\.bin",
    ),
    PublicCorpusLineage(
        corpus_id="nodchip-shogi-suisho5-depth9-entering-king",
        name="Suisho5 depth-9 entering-king positions",
        repository_url=(
            "https://huggingface.co/datasets/nodchip/"
            "shogi_suisho5_depth9_entering_king"
        ),
        revision="441f149296686876853a355eaeac1b482d3b206e",
        repository_bytes=20_000_245_181,
        data_bytes=20_000_241_760,
        position_records=500_006_044,
        record_format="YaneuraOu PackedSfenValue (40 bytes)",
        label_capability="Suisho5 search move and scalar value PSV source",
        policy_target_available=True,
        qsearch_leaf=False,
        deduplicated=False,
        license_id="MIT",
        license_evidence_url=(
            "https://huggingface.co/datasets/nodchip/"
            "shogi_suisho5_depth9_entering_king"
        ),
        reuse_decision=CorpusReuseDecision.QSEARCH_RELABEL_AND_STORAGE_REQUIRED,
        required_before_training=(
            "shuffle and deduplicate records by normalized board",
            "move positions to current-teacher qsearch leaves where applicable",
            "preserve entering-king and rule-history grouping in split receipts",
            "discard historical labels and regenerate current three-teacher labels",
        ),
        evidence=(
            "Hugging Face MIT dataset card: Floodgate 2015-2024 entering-king starts",
            "pinned file tree and deterministic legal-Move16 range probe",
        ),
        local_probe={
            "records_checked": 256,
            "legal_nonzero_move16_records": 256,
            "board_decode": "valid",
            "conclusion": "legal historical move is present but remains a discarded label",
        },
        seed_sampling_scope=CorpusSeedSamplingScope.PUBLIC_LICENSED,
        move_field_contract=PsvMoveFieldContract.LEGAL_MOVE_REQUIRED,
        sample_filename_pattern=r".+\.bin",
    ),
    PublicCorpusLineage(
        corpus_id="suishopsv-150m-public",
        name="Suisho5 2M-nodes-per-move game positions, about 150M PSV",
        repository_url=(
            "https://drive.google.com/file/d/1R9kI3xDKeoIjyFPD0RS6K-1wwcko75fr/view"
        ),
        revision=None,
        repository_bytes=None,
        data_bytes=None,
        position_records=150_000_000,
        record_format="YaneuraOu PackedSfenValue (publisher-reported)",
        label_capability="historical Suisho5 search labels; byte contract not yet probed",
        policy_target_available=None,
        qsearch_leaf=None,
        deduplicated=None,
        license_id=None,
        license_evidence_url=None,
        reuse_decision=CorpusReuseDecision.SOURCE_DOCUMENTATION_ONLY,
        required_before_training=(
            "download once, record exact bytes and SHA-256, and verify archive integrity",
            "probe Move16, score viewpoint, qsearch status, and duplicate rate",
            "use positions only after current three-teacher reanalysis",
        ),
        evidence=(
            "publisher note links the PSV archive and reports about 150M positions",
            "YaneuraOu training wiki reports Suisho5 at 2M nodes and 150M positions",
        ),
    ),
    PublicCorpusLineage(
        corpus_id="suisho10mn-100m-public",
        name="Suisho5 10M-nodes-per-position, about 100M PSV",
        repository_url=(
            "https://drive.google.com/drive/folders/19Al69YMkJ_cXSBhtn8df9yfxhn8QFCYo"
        ),
        revision=None,
        repository_bytes=None,
        data_bytes=None,
        position_records=100_000_000,
        record_format="YaneuraOu PackedSfenValue (publisher-reported)",
        label_capability="historical Suisho5 deep-search labels; byte contract not yet probed",
        policy_target_available=None,
        qsearch_leaf=None,
        deduplicated=None,
        license_id=None,
        license_evidence_url=None,
        reuse_decision=CorpusReuseDecision.SOURCE_DOCUMENTATION_ONLY,
        required_before_training=(
            "resolve and pin the exact Suisho10Mn_psv.7z file rather than a mutable folder",
            "record exact bytes and SHA-256 and verify archive integrity",
            "probe Move16, score viewpoint, qsearch status, and duplicate rate",
            "use positions only after current three-teacher reanalysis",
        ),
        evidence=(
            "WCSC36 collected appeal lists Suisho10Mn_psv.7z and about 100M positions",
            "YaneuraOu training wiki reports Suisho5 at 10M nodes and 100M positions",
        ),
    ),
    PublicCorpusLineage(
        corpus_id="gct-wcsc31-public-training-data",
        name="dlshogi with GCT WCSC31 public HCPE/HCPE3 training corpus",
        repository_url="https://tadaoyamaoka.hatenablog.com/entry/2021/05/06/223701",
        revision="publisher-post-2021-05-06",
        repository_bytes=None,
        data_bytes=None,
        position_records=None,
        record_format="HCPE and HCPE3 xz shards with generation-specific contracts",
        label_capability=(
            "game outcome, scalar value, and for HCPE3 an MCTS visit distribution"
        ),
        policy_target_available=True,
        qsearch_leaf=False,
        deduplicated=None,
        license_id="publisher-public-training-use-statement",
        license_evidence_url=(
            "https://tadaoyamaoka.hatenablog.com/entry/2021/05/06/223701"
        ),
        reuse_decision=CorpusReuseDecision.GAME_REPLAY_RELABEL_REQUIRED,
        required_before_training=(
            "inventory every mutable Drive shard by path, bytes, SHA-256, format, and generation",
            "split by complete source game and generator generation before position extraction",
            "preserve HCPE3 visit counts as historical-policy metadata, not alpha-beta nodes",
            "exclude or separately classify mate-filtered files rather than silently mixing them",
            "reanalyse selected positions with current Meteo teachers before production use",
        ),
        evidence=(
            "publisher documents GCT self-play generations at 1600 to 2400 playouts",
            "publisher documents Floodgate, AobaZero, entering-king, and local-game subsets",
            "publisher states the corpus was released to reduce the barrier for new developers",
        ),
    ),
    PublicCorpusLineage(
        corpus_id="qhapaq-pretty-daabi-training-kit",
        name="Qhapaq Pretty Daabi strategy teacher kit",
        repository_url="https://github.com/qhapaq-49/qhapaq-bin/releases/tag/dataset",
        revision="release-id-53913109-2021-11-23",
        repository_bytes=11_304_930,
        data_bytes=36_608_160,
        position_records=915_204,
        record_format="YaneuraOu PackedSfenValue (40 bytes)",
        label_capability="historical filtered furibisha move, scalar score, and result",
        policy_target_available=True,
        qsearch_leaf=None,
        deduplicated=None,
        license_id="QPD-attribution-terms-2021-11-23",
        license_evidence_url=(
            "https://github.com/qhapaq-49/qhapaq-bin/releases/tag/dataset"
        ),
        reuse_decision=CorpusReuseDecision.QSEARCH_RELABEL_AND_STORAGE_REQUIRED,
        required_before_training=(
            "record QPD use in any competition PR, published evaluation function, or service",
            "count the 915,204 source records once; the original recipe repeated them 20 times",
            "deduplicate and split by normalized board before any training",
            "use as a low-weight strategy/worst-group seed and regenerate current labels",
        ),
        evidence=(
            "official release and publisher post",
            "readme.txt inside the release asset permits free use subject to attribution terms",
            "archive and payload were inspected without retaining a workspace copy",
        ),
        local_probe={
            "release_asset_sha256": (
                "8887ae22cbe234c78042dea503fd097a9b88bf6bce4fdeaeddf91b3868f2a6ae"
            ),
            "archive_payload": "QPD_train/furi_win02_plain.bin",
            "payload_bytes": 36_608_160,
            "records": 915_204,
            "original_training_repetition_factor": 20,
        },
    ),
    PublicCorpusLineage(
        corpus_id="denryu-public-game-records",
        name="World Shogi AI Denryu-sen public tournament game records",
        repository_url="https://denryu-sen.jp/denryusen/dr5_hardware3/dr1_live.php",
        revision="event-pages-reviewed-2026-08-11",
        repository_bytes=None,
        data_bytes=None,
        position_records=None,
        record_format="event-specific CSA/KIF game archives",
        label_capability="complete played games and rule terminal results",
        policy_target_available=False,
        qsearch_leaf=False,
        deduplicated=False,
        license_id="Denryu-game-record-free-use",
        license_evidence_url=(
            "https://denryu-sen.jp/denryusen/dr5_hardware3/dr1_live.php"
        ),
        reuse_decision=CorpusReuseDecision.GAME_REPLAY_RELABEL_REQUIRED,
        required_before_training=(
            "record the exact event page, archive hash, rules, and complete-game identity",
            "split at the tournament/game level and never use evaluation games for training",
            "treat played moves as trajectories, not current best-move labels",
            "reanalyse selected positions with current teachers and terminal rule checks",
        ),
        evidence=(
            "the linked Denryu event page explicitly says game-record use is unrestricted",
            "Denryu-sen fourth-event rules also state that game records may be freely used",
        ),
    ),
)


MAIN_TEACHER_LINEAGES: tuple[TeacherTrainingLineage, ...] = (
    TeacherTrainingLineage(
        teacher_id="nagisa-v3.1",
        name="NAGISA V3.1",
        learner="Tatara",
        architecture="SFNN HalfKA_hm2 1024 with 9-layer progress stack (local header)",
        direct_teacher_models=("Soujou WCSC36 published teacher-position corpus",),
        corpus_families=("soujou-dlsuisho-nodchip-family",),
        reported_position_count=14_500_000_000,
        reported_position_count_kind="inherited approximate corpus scale, not an independent count",
        correlation_family="soujou-dlsuisho-nodchip-family",
        exact_training_bytes_public=False,
        ensemble_formula_public=None,
        evidence=(
            "https://ngs436.booth.pm/items/8639574",
            "local NAGISA V3.1 archive/header inspection",
        ),
        caveats=(
            "NAGISA and Soujou are correlated teachers because NAGISA reports using Soujou data",
            "do not add 14.5B twice when reporting independent source positions",
        ),
    ),
    TeacherTrainingLineage(
        teacher_id="suisho11plus-wcsc36-20260525-local",
        name="Suisho 11 / local Suisho11Plus profile",
        learner="NNUE learner not fully disclosed in the added appeal",
        architecture="SFNNwithoutPSQT HalfKAv2-1024_8_64",
        direct_teacher_models=("Ryfamate latest", "DL Suisho", "AobaZero"),
        corpus_families=("suisho11-three-dl-model-ensemble",),
        reported_position_count=None,
        reported_position_count_kind="not publicly reported",
        correlation_family="suisho11-three-dl-model-ensemble",
        exact_training_bytes_public=False,
        ensemble_formula_public=False,
        evidence=(
            "https://www.apply.computer-shogi.org/wcsc36/appeal/Suisho/appeal2.pdf",
        ),
        caveats=(
            "the appeal discloses the three model names but not weights, calibration, "
            "or averaging space",
            "extreme <=1% and >=99% win-rate positions exceed 30% and removing them "
            "reduced strength",
            "the appeal increased the Ponanza constant to reduce int8 saturation and "
            "adjusted FV_SCALE",
        ),
    ),
    TeacherTrainingLineage(
        teacher_id="soujou-tsec7-paid",
        name="Soujou WCSC36 / TSEC7 family",
        learner="bullet-shogi",
        architecture="progress-conditioned SFNNwoP family; exact TSEC7 runtime is locally pinned",
        direct_teacher_models=("DL Suisho",),
        corpus_families=("soujou-dlsuisho-nodchip-family",),
        reported_position_count=14_500_000_000,
        reported_position_count_kind="WCSC36 appeal: 24B raw to about 14.5B after deduplication",
        correlation_family="soujou-dlsuisho-nodchip-family",
        exact_training_bytes_public=False,
        ensemble_formula_public=None,
        evidence=(
            "https://www.apply.computer-shogi.org/wcsc36/appeal/sojo/sojo_WCSC36_appeal.pdf",
            "https://www.apply.computer-shogi.org/wcsc36/appeal/appeal_final_260504.pdf",
        ),
        caveats=(
            "the public 14,668,949,437-record corpus matches the reported family and scale "
            "but is not byte-hash proof of the exact TSEC7 training set",
            "the public corpus is value-only in the sampled records and has no declared "
            "repository license",
            "the newer 49,594,855,063-position publisher corpus is a stronger fresh "
            "bootstrap candidate but is not proof of the exact TSEC7 training bytes",
        ),
    ),
)


def teacher_lineage_contract() -> dict[str, object]:
    """Return the machine-readable ancestry and reuse decision catalog."""

    return {
        "schema": TEACHER_LINEAGE_SCHEMA,
        "main_teachers": [teacher.to_dict() for teacher in MAIN_TEACHER_LINEAGES],
        "public_corpora": [corpus.to_dict() for corpus in PUBLIC_CORPORA],
        "independence_contract": {
            "nagisa_and_soujou_independent_corpus_votes": False,
            "position_counts_must_be_deduplicated_across_lineage_families": True,
            "suisho11_component_weights_public": False,
        },
        "reuse_priority": [
            "datasets_2 exceeds the 100B lifetime target in nominal records but remains gated, "
            "unprobed, and deeply overlapping; do not count or train it yet",
            "use the user-attested Soujou 49.595B and DL-Suisho 16.667B corpora only "
            "behind an explicit local-only acknowledgement",
            "request explicit reuse terms for the separate deduplicated DL-Suisho corpus",
            "the NAGISA-style route may consume the reviewed Move16=0 scalar values directly "
            "with policy loss absent, immutable heldout, and a random-initialized NNUE",
            "the legacy Policy+Value route may use the same records only as position seeds and "
            "must generate fresh policy/search labels; do not invent policy from Move16=0",
            "treat AobaZero archives as generation-versioned game seeds, not as one "
            "homogeneous policy corpus",
            "use MIT nodchip corpora as position seeds only after qsearch, deduplication, "
            "and relabeling",
            "oversample the MIT Suisho5 entering-king corpus as a worst-group source",
            "use GCT HCPE3, Aoba Furibisha, QPD, and Denryu games as diversity strata only "
            "after source-specific parsing and current-teacher relabeling",
            "keep raw source, unique labelled, retained unique, and cumulative seen "
            "counters separate",
        ],
    }


def corpus_lineage(corpus_id: str) -> PublicCorpusLineage:
    matches = [corpus for corpus in PUBLIC_CORPORA if corpus.corpus_id == corpus_id]
    if len(matches) != 1:
        raise KeyError(f"unknown or duplicate teacher corpus ID: {corpus_id}")
    return matches[0]
