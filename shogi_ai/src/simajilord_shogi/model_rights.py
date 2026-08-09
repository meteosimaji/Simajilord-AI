"""Versioned rights decisions for external shogi engines and model artifacts.

The registry deliberately separates running an engine, learning from its USI
output, copying its weights, and publishing a resulting Meteo checkpoint.  A
GPL engine license does not by itself settle a separately distributed model's
terms, while GPLv3 section 2 also does not automatically place ordinary program
output under the GPL.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

from .external_usi import ExternalTeacherPolicy

RIGHTS_REVIEW_DATE = "2026-08-09"


class RightsDecision(StrEnum):
    """A deliberately small, machine-checkable rights decision vocabulary."""

    ALLOWED = "allowed"
    CONDITIONAL_GPL = "conditional_gpl"
    LIMITED = "limited"
    NOT_APPROVED = "not_approved"
    NOT_APPLICABLE = "not_applicable"
    EXCLUDED_PAID = "excluded_paid"


@dataclass(frozen=True, slots=True)
class ModelRights:
    """One reviewed engine/model combination, pinned to a public version."""

    rights_id: str
    name: str
    version: str
    family: str
    availability: str
    engine_license: str
    model_terms: str
    sources: tuple[str, ...]
    analysis: RightsDecision
    output_distillation: RightsDecision
    hard_game_training: RightsDecision
    direct_weight_use: RightsDecision
    original_artifact_redistribution: RightsDecision
    output_only_meteo_publication: RightsDecision
    notes: tuple[str, ...]
    reviewed_at: str = RIGHTS_REVIEW_DATE

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def teacher_policy(
        self, *, allow_limited_local: bool = False
    ) -> ExternalTeacherPolicy:
        """Create a fail-closed USI policy from this reviewed registry row.

        A LIMITED source is never enabled implicitly.  The caller must make a
        run-specific acknowledgement, and its labels remain non-redistributable.
        """

        analysis_allowed = self.analysis == RightsDecision.ALLOWED or (
            allow_limited_local and self.analysis == RightsDecision.LIMITED
        )
        training_outputs_allowed = self.output_distillation == RightsDecision.ALLOWED or (
            allow_limited_local
            and self.output_distillation == RightsDecision.LIMITED
        )
        local_only = self.output_distillation == RightsDecision.LIMITED

        return ExternalTeacherPolicy(
            policy_id=self.rights_id,
            name=self.name,
            source=self.sources[0],
            analysis_allowed=analysis_allowed,
            training_outputs_allowed=training_outputs_allowed,
            redistribution_allowed=(self.output_only_meteo_publication == RightsDecision.ALLOWED),
            requires_payment=self.availability == "paid" and not allow_limited_local,
            training_outputs_local_only=local_only and allow_limited_local,
        )


_GPL_OUTPUT_NOTE = (
    "USI moves, scores, and PVs are approved for output-only distillation because no "
    "separate output restriction was found; this does not authorize copying weights."
)


MODEL_RIGHTS: tuple[ModelRights, ...] = (
    ModelRights(
        rights_id="aobazero-public-domain",
        name="AobaZero",
        version="original public releases",
        family="DL/MCTS",
        availability="free",
        engine_license="GPL-3.0 for aobaz; other published artifacts declared public domain",
        model_terms="README declares weights and game data outside aobaz public domain",
        sources=("https://github.com/kobanium/aobazero",),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.ALLOWED,
        hard_game_training=RightsDecision.ALLOWED,
        direct_weight_use=RightsDecision.ALLOWED,
        original_artifact_redistribution=RightsDecision.ALLOWED,
        output_only_meteo_publication=RightsDecision.ALLOWED,
        notes=(
            "Best rights-clean bootstrap source among the reviewed neural projects.",
            "Keep source and version provenance even where the author disclaims rights.",
        ),
    ),
    ModelRights(
        rights_id="yaneuraou-rezero",
        name="YaneuraOu ReZero evaluation",
        version="official public evaluation",
        family="classical evaluation",
        availability="free",
        engine_license="GPL-3.0 engine",
        model_terms="project README says no rights are asserted over the ReZero evaluation",
        sources=("https://github.com/yaneurao/YaneuraOu",),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.ALLOWED,
        hard_game_training=RightsDecision.ALLOWED,
        direct_weight_use=RightsDecision.ALLOWED,
        original_artifact_redistribution=RightsDecision.ALLOWED,
        output_only_meteo_publication=RightsDecision.ALLOWED,
        notes=("Rights-clean but much weaker than current top teachers.",),
    ),
    ModelRights(
        rights_id="takewarabe-approx-v7.50-material9",
        name="Takewarabe approximate V7.50 MaterialLv9",
        version="YaneuraOu v7.50-wcsc32 / MATERIAL_LEVEL=9",
        family="handcrafted material/effect evaluation",
        availability="free",
        engine_license="GPL-3.0 YaneuraOu source",
        model_terms="not applicable; the approximate engine has no external evaluation file",
        sources=(
            "https://yaneuraou.yaneu.com/2020/11/18/takewarabe/",
            "https://github.com/yaneurao/YaneuraOu/tree/v7.50-wcsc32",
        ),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.ALLOWED,
        hard_game_training=RightsDecision.ALLOWED,
        direct_weight_use=RightsDecision.NOT_APPLICABLE,
        original_artifact_redistribution=RightsDecision.CONDITIONAL_GPL,
        output_only_meteo_publication=RightsDecision.ALLOWED,
        notes=(
            _GPL_OUTPUT_NOTE,
            "The author states that YaneuraOu V7.50 material evaluation at "
            "MATERIAL_LEVEL=9 is approximately Takewarabe and confirms OSX builds; label the "
            "reproduction approximate rather than claiming it is the original binary.",
            "Use it as an unusual weak/human-style opponent and position generator. Its moves "
            "are not ground-truth labels; strong teachers must re-adjudicate the resulting "
            "positions before Meteo trains on them.",
            "Disable the opening book and vary nodes or clock deliberately when measuring how "
            "Meteo handles off-book, human-like play.",
        ),
    ),
    ModelRights(
        rights_id="aobannue-v1.1",
        name="AobaNNUE",
        version="v1.1",
        family="NNUE",
        availability="free",
        engine_license="GPL-3.0",
        model_terms=(
            "the repository is GPL-3.0, but the linked Drive archive and release text do not "
            "separately license the nn.bin artifact"
        ),
        sources=(
            "https://github.com/yssaya/AobaNNUE",
            "https://github.com/yssaya/AobaNNUE/releases/tag/v1.1",
            "https://github.com/yssaya/AobaNNUE/blob/master/Copying.txt",
        ),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.ALLOWED,
        hard_game_training=RightsDecision.ALLOWED,
        direct_weight_use=RightsDecision.NOT_APPROVED,
        original_artifact_redistribution=RightsDecision.NOT_APPROVED,
        output_only_meteo_publication=RightsDecision.ALLOWED,
        notes=(
            _GPL_OUTPUT_NOTE,
            "Repository-level GPL evidence is not treated as an artifact-specific grant for the "
            "separately hosted nn.bin; use AobaNNUE only through ordinary USI output.",
        ),
    ),
    ModelRights(
        rights_id="dlshogi-aoba-wcsc35",
        name="AobaZero WCSC35 dlshogi_aoba",
        version="v1 / 2025 weight",
        family="DL/MCTS ResNet 30x384",
        availability="free",
        engine_license="GPL-3.0",
        model_terms="GPL-3.0 repository explicitly releases the WCSC35 weight",
        sources=(
            "https://github.com/yssaya/dlshogi_aoba",
            "https://github.com/yssaya/dlshogi_aoba/releases/tag/v1",
        ),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.ALLOWED,
        hard_game_training=RightsDecision.ALLOWED,
        direct_weight_use=RightsDecision.CONDITIONAL_GPL,
        original_artifact_redistribution=RightsDecision.CONDITIONAL_GPL,
        output_only_meteo_publication=RightsDecision.ALLOWED,
        notes=(
            _GPL_OUTPUT_NOTE,
            "Its additional ply, turn, and pawn-count features make the weight incompatible "
            "with stock dlshogi and Meteo competition_v1.",
        ),
    ),
    ModelRights(
        rights_id="nagisa-v3.1",
        name="NAGISA",
        version="v3.1",
        family="NNUE",
        availability="free",
        engine_license="GPL-3.0 engine",
        model_terms="product page prohibits unauthorized redistribution of the evaluation file",
        sources=("https://booth.pm/ja/items/8639574",),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.ALLOWED,
        hard_game_training=RightsDecision.ALLOWED,
        direct_weight_use=RightsDecision.NOT_APPROVED,
        original_artifact_redistribution=RightsDecision.NOT_APPROVED,
        output_only_meteo_publication=RightsDecision.ALLOWED,
        notes=(
            _GPL_OUTPUT_NOTE,
            "The evaluation-file redistribution ban is honored; neither nn.bin nor a conversion "
            "of it may be placed in Meteo.",
        ),
    ),
    ModelRights(
        rights_id="suisho5",
        name="Suisho5",
        version="official public release",
        family="NNUE HalfKP256",
        availability="free",
        engine_license="GPL-3.0 YaneuraOu engine",
        model_terms="official release contains no separate evaluation-file terms",
        sources=(
            "https://github.com/yaneurao/YaneuraOu/releases/tag/suisho5",
            "https://github.com/yaneurao/YaneuraOu",
        ),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.ALLOWED,
        hard_game_training=RightsDecision.ALLOWED,
        direct_weight_use=RightsDecision.NOT_APPROVED,
        original_artifact_redistribution=RightsDecision.NOT_APPROVED,
        output_only_meteo_publication=RightsDecision.ALLOWED,
        notes=(
            _GPL_OUTPUT_NOTE,
            "No separate grant for copying or converting the contributed nn.bin was found.",
        ),
    ),
    ModelRights(
        rights_id="shinden3-2025-02-21",
        name="Shinden3",
        version="2025-02-21 official video/Drive release",
        family="NNUE HalfKP512 / ranging-rook specialist",
        availability="free",
        engine_license="GPL-3.0 YaneuraOu 7.70 source bundled in the archive",
        model_terms=(
            "the official archive bundles eval/nn.bin and GPL-3.0 engine source, but no "
            "separate evaluation-file license or redistribution grant was found"
        ),
        sources=(
            "https://www.youtube.com/watch?v=07PhE_I6c1s",
            "https://drive.google.com/file/d/115yqu8iVEtLF2sQ2QlDn56zhlyhiozw2/view",
        ),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.ALLOWED,
        hard_game_training=RightsDecision.ALLOWED,
        direct_weight_use=RightsDecision.NOT_APPROVED,
        original_artifact_redistribution=RightsDecision.NOT_APPROVED,
        output_only_meteo_publication=RightsDecision.ALLOWED,
        notes=(
            _GPL_OUTPUT_NOTE,
            "The official archive contains Windows binaries and complete YaneuraOu 7.70 "
            "source; an Apple Silicon APPLEM1 build was reproduced locally without modifying "
            "the source.",
            "Use as a ranging-rook specialist opponent and story-preserving teacher branch, "
            "not as an independent generic-vote duplicate of other NNUE engines.",
            "Book must remain disabled for search-label comparisons unless a separately "
            "declared opening-repertoire experiment intentionally enables it.",
        ),
    ),
    ModelRights(
        rights_id="gikou2-v2.0.2",
        name="Gikou 2",
        version="v2.0.2",
        family="alpha-beta / handcrafted learned evaluation",
        availability="free",
        engine_license="GPL-3.0",
        model_terms="GPL-3.0 release archive includes Copying.txt and learned parameter files",
        sources=(
            "https://github.com/gikou-official/Gikou",
            "https://github.com/gikou-official/Gikou/releases/tag/v2.0.2",
        ),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.ALLOWED,
        hard_game_training=RightsDecision.ALLOWED,
        direct_weight_use=RightsDecision.CONDITIONAL_GPL,
        original_artifact_redistribution=RightsDecision.CONDITIONAL_GPL,
        output_only_meteo_publication=RightsDecision.ALLOWED,
        notes=(
            _GPL_OUTPUT_NOTE,
            "Useful as a tactical and opening-style diversity teacher, not as the "
            "strongest anchor.",
        ),
    ),
    ModelRights(
        rights_id="hao-2023-05-08",
        name="Háo",
        version="tanuki-.halfkp_256x2-32-32.2023-05-08",
        family="NNUE HalfKP256",
        availability="free",
        engine_license="GPL-3.0 YaneuraOu/tanuki- engine",
        model_terms="official release archive includes gpl-3.0.txt beside eval/nn.bin",
        sources=(
            "https://github.com/nodchip/tanuki-/releases/tag/"
            "tanuki-.halfkp_256x2-32-32.2023-05-08",
            "https://huggingface.co/datasets/nodchip/shogi_hao_depth9",
        ),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.ALLOWED,
        hard_game_training=RightsDecision.ALLOWED,
        direct_weight_use=RightsDecision.CONDITIONAL_GPL,
        original_artifact_redistribution=RightsDecision.CONDITIONAL_GPL,
        output_only_meteo_publication=RightsDecision.ALLOWED,
        notes=(
            _GPL_OUTPUT_NOTE,
            "The author reports this standard NNUE at approximately the strength of Lí, the "
            "tanuki- WCSC33 evaluation; FV_SCALE=20 is the documented setting.",
            "The separately published depth-9 PackedSfenValue dataset is marked MIT, but it "
            "contains single-line hard targets rather than MultiPV distributions or full PVs.",
            "Treat Háo and tanuki-dr4 as one correlated tanuki-family source when assigning "
            "ensemble weights.",
        ),
    ),
    ModelRights(
        rights_id="tanuki-dr4-2023-12-03",
        name="tanuki- Lí-VENGE",
        version="tanuki-dr4-2023-12-03",
        family="NNUE HalfKP1024 / custom YaneuraOu search",
        availability="free",
        engine_license="GPL-3.0",
        model_terms=(
            "official release archive includes gpl-3.0.txt, engine binaries, eval/nn.bin, "
            "and book databases"
        ),
        sources=(
            "https://github.com/nodchip/tanuki-/releases/tag/tanuki-dr4",
            "https://github.com/nodchip/tanuki-/tree/tanuki-dr4-engine",
        ),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.ALLOWED,
        hard_game_training=RightsDecision.ALLOWED,
        direct_weight_use=RightsDecision.CONDITIONAL_GPL,
        original_artifact_redistribution=RightsDecision.CONDITIONAL_GPL,
        output_only_meteo_publication=RightsDecision.ALLOWED,
        notes=(
            _GPL_OUTPUT_NOTE,
            "The public archive contains Windows binaries only; Apple Silicon use requires a "
            "reproducible build of the linked GPL engine branch.",
            "Disable the bundled book for teacher comparison and distillation so its signal is "
            "search/evaluation behavior rather than opening-book lookup.",
            "Treat this release and Háo as one correlated tanuki-family source unless held-out "
            "measurements demonstrate complementary errors.",
        ),
    ),
    ModelRights(
        rights_id="dlshogi-gct-wcsc31",
        name="dlshogi with GCT",
        version="WCSC31 release",
        family="DL/MCTS",
        availability="free",
        engine_license="GPL-3.0",
        model_terms=(
            "the GPL-3.0 project release bundles models, but the release text and archive do not "
            "state model-specific terms"
        ),
        sources=(
            "https://github.com/TadaoYamaoka/DeepLearningShogi/releases/tag/wcwc31",
            "https://github.com/TadaoYamaoka/DeepLearningShogi",
        ),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.ALLOWED,
        hard_game_training=RightsDecision.ALLOWED,
        direct_weight_use=RightsDecision.NOT_APPROVED,
        original_artifact_redistribution=RightsDecision.NOT_APPROVED,
        output_only_meteo_publication=RightsDecision.ALLOWED,
        notes=(
            _GPL_OUTPUT_NOTE,
            "Bundling by a GPL repository is not treated as an artifact-specific grant for the "
            "ONNX models; use this release only through ordinary USI output.",
        ),
    ),
    ModelRights(
        rights_id="dlshogi-dr2-exhi",
        name="dlshogi Denryu2 exhibition model",
        version="dr2_exhi",
        family="DL/MCTS",
        availability="free-with-separate-terms",
        engine_license="GPL-3.0 code",
        model_terms="separate model terms prohibit extra training, parameter reuse, modification, "
        "reverse engineering, and redistribution",
        sources=(
            "https://github.com/TadaoYamaoka/DeepLearningShogi/releases/tag/dr2_exhi",
            "https://tadaoyamaoka.hatenablog.com/entry/2021/08/17/000710",
        ),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.NOT_APPROVED,
        hard_game_training=RightsDecision.LIMITED,
        direct_weight_use=RightsDecision.NOT_APPROVED,
        original_artifact_redistribution=RightsDecision.NOT_APPROVED,
        output_only_meteo_publication=RightsDecision.NOT_APPROVED,
        notes=(
            "The narrow tournament permission for generating model-learning game records is not "
            "treated as permission for general soft-label distillation or Apache publication.",
        ),
    ),
    ModelRights(
        rights_id="zimetu-2026-01-26",
        name="zimetu",
        version="2026-01-26",
        family="NNUE HalfKP256",
        availability="free",
        engine_license="GPL-3.0 source",
        model_terms="free product page states no separate output restriction",
        sources=(
            "https://booth.pm/ja/items/7916789",
            "https://github.com/nodchip/tanuki-/tree/zimetu.2026-01-26",
        ),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.ALLOWED,
        hard_game_training=RightsDecision.ALLOWED,
        direct_weight_use=RightsDecision.NOT_APPROVED,
        original_artifact_redistribution=RightsDecision.NOT_APPROVED,
        output_only_meteo_publication=RightsDecision.ALLOWED,
        notes=(
            _GPL_OUTPUT_NOTE,
            "A style-specialized diversity teacher; its product page does not separately license "
            "the evaluation file for copying.",
        ),
    ),
    ModelRights(
        rights_id="apery-public",
        name="Apery",
        version="2019-06-17 official evaluation binaries",
        family="alpha-beta / learned evaluation",
        availability="free",
        engine_license="GPL-3.0-or-later",
        model_terms="the separate official evaluation-binaries repository declares MIT",
        sources=(
            "https://github.com/HiraokaTakuya/apery",
            "https://bitbucket.org/hiraoka64/apery-evaluation-binaries-2019-06-17/"
            "src/master/README.md",
            "https://bitbucket.org/hiraoka64/apery-evaluation-binaries-2019-06-17/"
            "src/master/LICENSE-MIT",
        ),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.ALLOWED,
        hard_game_training=RightsDecision.ALLOWED,
        direct_weight_use=RightsDecision.ALLOWED,
        original_artifact_redistribution=RightsDecision.ALLOWED,
        output_only_meteo_publication=RightsDecision.ALLOWED,
        notes=(
            _GPL_OUTPUT_NOTE,
            "Keep the MIT copyright and permission notice when copying or redistributing these "
            "exact evaluation binaries.",
        ),
    ),
    ModelRights(
        rights_id="elmo-wcsc27",
        name="elmo",
        version="WCSC27 public evaluation",
        family="alpha-beta / learned evaluation",
        availability="free",
        engine_license="GPL-3.0 evaluation-generation code",
        model_terms="no separate output restriction found in the official public repository",
        sources=("https://github.com/mk-takizawa/elmo_for_learn",),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.ALLOWED,
        hard_game_training=RightsDecision.ALLOWED,
        direct_weight_use=RightsDecision.CONDITIONAL_GPL,
        original_artifact_redistribution=RightsDecision.CONDITIONAL_GPL,
        output_only_meteo_publication=RightsDecision.ALLOWED,
        notes=(_GPL_OUTPUT_NOTE,),
    ),
    ModelRights(
        rights_id="gpsfish-public",
        name="GPSFish",
        version="public release line",
        family="alpha-beta / classical evaluation",
        availability="free",
        engine_license="GPL-3.0",
        model_terms="official project publishes source and data as free software",
        sources=("https://gps.tanaka.ecc.u-tokyo.ac.jp/gpsshogi/",),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.ALLOWED,
        hard_game_training=RightsDecision.ALLOWED,
        direct_weight_use=RightsDecision.CONDITIONAL_GPL,
        original_artifact_redistribution=RightsDecision.CONDITIONAL_GPL,
        output_only_meteo_publication=RightsDecision.ALLOWED,
        notes=(_GPL_OUTPUT_NOTE,),
    ),
    ModelRights(
        rights_id="sunfish4",
        name="Sunfish 4",
        version="public repository",
        family="alpha-beta / classical evaluation",
        availability="free",
        engine_license="MIT",
        model_terms="MIT repository; no separately restricted neural weight",
        sources=("https://github.com/sunfish-shogi/sunfish4",),
        analysis=RightsDecision.ALLOWED,
        output_distillation=RightsDecision.ALLOWED,
        hard_game_training=RightsDecision.ALLOWED,
        direct_weight_use=RightsDecision.NOT_APPLICABLE,
        original_artifact_redistribution=RightsDecision.ALLOWED,
        output_only_meteo_publication=RightsDecision.ALLOWED,
        notes=("Useful mainly for diversity and regression, not current top strength.",),
    ),
    ModelRights(
        rights_id="hisui-wcsc36-hosted",
        name="Hisui (氷彗)",
        version="WCSC36 champion",
        family="NNUE / custom YaneuraOu search",
        availability="paid",
        engine_license="YaneuraOu-based; downloadable corresponding source/model not published",
        model_terms="available as a hosted Kishin Analytics engine; training data is private",
        sources=(
            "https://www.apply.computer-shogi.org/wcsc36/appeal/hisui/hisui_detail.pdf",
            "https://note.com/kishin_analytics/n/n0effe0c2e5d9",
        ),
        analysis=RightsDecision.EXCLUDED_PAID,
        output_distillation=RightsDecision.EXCLUDED_PAID,
        hard_game_training=RightsDecision.LIMITED,
        direct_weight_use=RightsDecision.NOT_APPROVED,
        original_artifact_redistribution=RightsDecision.NOT_APPROVED,
        output_only_meteo_publication=RightsDecision.NOT_APPROVED,
        notes=(
            "No downloadable WCSC36 binary or weight was found in the official sources.",
            "Public WCSC games may be imported separately as hard behavioral examples; that is "
            "not equivalent to distilling Hisui's evaluation or MultiPV distribution.",
            "The public appeal says Hisui itself used knowledge distillation from dlshogi.",
        ),
    ),
    ModelRights(
        rights_id="suisho11plus-wcsc36-20260525-local",
        name="Suisho11Plus WCSC36 2026-05-25 local teacher",
        version="WCSC36 / 2026-05-25 SFNN",
        family="NNUE/SFNN",
        availability="paid",
        engine_license="GPL-3.0 YaneuraOu V9.70DEV engine",
        model_terms=(
            "lawfully acquired supporter evaluation; archive has no LICENSE; ordinary USI "
            "labels are approved only for user-authorized private local distillation"
        ),
        sources=(
            "https://www.apply.computer-shogi.org/wcsc36/appeal/appeal_round2_260502.pdf",
            "https://yaneuraou.yaneu.com/2026/05/07/wcsc36-petashock-suisho11/",
            "https://www.bunka.go.jp/seisaku/chosakuken/pdf/94097701_01.pdf",
        ),
        analysis=RightsDecision.LIMITED,
        output_distillation=RightsDecision.LIMITED,
        hard_game_training=RightsDecision.LIMITED,
        direct_weight_use=RightsDecision.NOT_APPROVED,
        original_artifact_redistribution=RightsDecision.NOT_APPROVED,
        output_only_meteo_publication=RightsDecision.NOT_APPROVED,
        notes=(
            "The exact supporter post was reviewed from the user's lawfully accessed copy; "
            "its paid download URL is intentionally omitted from the public registry.",
            "Use only through an external USI process; do not copy nn.bin or the supplied "
            "architecture header into Meteo.",
            "Raw labels and derived checkpoints stay local until the rights holder explicitly "
            "clears publication of the distilled Meteo weights.",
        ),
    ),
    ModelRights(
        rights_id="suisho10-11-supporter",
        name="Suisho 10/11 supporter builds",
        version="current supporter distributions",
        family="NNUE/SFNN",
        availability="paid",
        engine_license="GPL-3.0 YaneuraOu engine",
        model_terms="supporter-only evaluation terms were not accepted or downloaded",
        sources=("https://github.com/yaneurao/YaneuraOu",),
        analysis=RightsDecision.EXCLUDED_PAID,
        output_distillation=RightsDecision.EXCLUDED_PAID,
        hard_game_training=RightsDecision.EXCLUDED_PAID,
        direct_weight_use=RightsDecision.EXCLUDED_PAID,
        original_artifact_redistribution=RightsDecision.EXCLUDED_PAID,
        output_only_meteo_publication=RightsDecision.EXCLUDED_PAID,
        notes=(
            "Excluded by Meteo's current free-only policy, irrespective of possible "
            "GPL output use.",
        ),
    ),
    ModelRights(
        rights_id="tanuki-wcsc36-paid",
        name="tanuki- WCSC36 (六角堂狸)",
        version="WCSC36",
        family="SFNN",
        availability="paid",
        engine_license="GPL-3.0 source",
        model_terms="paid evaluation distribution; no purchase terms were accepted",
        sources=("https://booth.pm/ja/items/8303452",),
        analysis=RightsDecision.EXCLUDED_PAID,
        output_distillation=RightsDecision.EXCLUDED_PAID,
        hard_game_training=RightsDecision.EXCLUDED_PAID,
        direct_weight_use=RightsDecision.EXCLUDED_PAID,
        original_artifact_redistribution=RightsDecision.EXCLUDED_PAID,
        output_only_meteo_publication=RightsDecision.EXCLUDED_PAID,
        notes=("Excluded by Meteo's current free-only policy.",),
    ),
    ModelRights(
        rights_id="soujou-tsec7-paid",
        name="奏乗 TSEC7",
        version="TSEC7",
        family="NNUE",
        availability="paid",
        engine_license="GPL-3.0-derived engine source",
        model_terms="paid evaluation distribution; no purchase terms were accepted",
        sources=("https://booth.pm/ja/items/8606196",),
        analysis=RightsDecision.EXCLUDED_PAID,
        output_distillation=RightsDecision.EXCLUDED_PAID,
        hard_game_training=RightsDecision.EXCLUDED_PAID,
        direct_weight_use=RightsDecision.EXCLUDED_PAID,
        original_artifact_redistribution=RightsDecision.EXCLUDED_PAID,
        output_only_meteo_publication=RightsDecision.EXCLUDED_PAID,
        notes=("Excluded by Meteo's current free-only policy.",),
    ),
    ModelRights(
        rights_id="kanade-wcsc35-paid",
        name="Kanade",
        version="WCSC35 commercial model",
        family="DL",
        availability="paid",
        engine_license="external dlshogi-compatible engine",
        model_terms="paid model-only distribution; no purchase terms were accepted",
        sources=("https://booth.pm/ja/items/7108913",),
        analysis=RightsDecision.EXCLUDED_PAID,
        output_distillation=RightsDecision.EXCLUDED_PAID,
        hard_game_training=RightsDecision.EXCLUDED_PAID,
        direct_weight_use=RightsDecision.EXCLUDED_PAID,
        original_artifact_redistribution=RightsDecision.EXCLUDED_PAID,
        output_only_meteo_publication=RightsDecision.EXCLUDED_PAID,
        notes=("Excluded by Meteo's current free-only policy.",),
    ),
)


_RIGHTS_BY_ID = {record.rights_id: record for record in MODEL_RIGHTS}
if len(_RIGHTS_BY_ID) != len(MODEL_RIGHTS):
    raise RuntimeError("duplicate model-rights identifier")
if len({record.name.casefold() for record in MODEL_RIGHTS}) != len(MODEL_RIGHTS):
    raise RuntimeError("case-insensitive model-rights name collision")


def model_rights(rights_id: str) -> ModelRights:
    """Return an exact reviewed profile; unknown or misspelled IDs fail closed."""

    try:
        return _RIGHTS_BY_ID[rights_id]
    except KeyError as error:
        choices = ", ".join(sorted(_RIGHTS_BY_ID))
        raise ValueError(
            f"unknown model-rights profile {rights_id!r}; choose one of: {choices}"
        ) from error


def distillable_rights_ids() -> tuple[str, ...]:
    """IDs approved for free, output-only USI distillation."""

    return tuple(
        record.rights_id
        for record in MODEL_RIGHTS
        if record.availability == "free"
        and record.analysis == RightsDecision.ALLOWED
        and record.output_distillation == RightsDecision.ALLOWED
        and record.output_only_meteo_publication == RightsDecision.ALLOWED
    )


def locally_distillable_rights_ids() -> tuple[str, ...]:
    """IDs usable for USI labels, including explicitly acknowledged local-only sources."""

    eligible = {RightsDecision.ALLOWED, RightsDecision.LIMITED}
    return tuple(
        record.rights_id
        for record in MODEL_RIGHTS
        if record.analysis in eligible and record.output_distillation in eligible
    )


def analysable_rights_ids() -> tuple[str, ...]:
    """IDs approved for local analysis and fair benchmarks under the free-only policy."""

    return tuple(
        record.rights_id
        for record in MODEL_RIGHTS
        if record.availability == "free" and record.analysis == RightsDecision.ALLOWED
    )
