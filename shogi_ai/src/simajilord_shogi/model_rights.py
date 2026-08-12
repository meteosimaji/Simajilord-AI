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

RIGHTS_REVIEW_DATE = "2026-08-12"


class RightsDecision(StrEnum):
    """A deliberately small, machine-checkable rights decision vocabulary."""

    ALLOWED = "allowed"
    CONDITIONAL_GPL = "conditional_gpl"
    LIMITED = "limited"
    NOT_APPROVED = "not_approved"
    NOT_APPLICABLE = "not_applicable"


class DistillationScope(StrEnum):
    """Where labels from one reviewed profile may be consumed.

    This is deliberately independent of the artifact's price.  A lawfully
    acquired local teacher can be approved for private training while still
    being forbidden from a public checkpoint.  Conversely, appearing in the
    public-safe list never means Meteo may redistribute the original model.
    """

    PUBLIC_RELEASE_ALLOWED = "public_release_allowed"
    LOCAL_AUTHORIZED_ONLY = "local_authorized_only"
    NOT_AUTHORIZED = "not_authorized"


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

    @property
    def distillation_scope(self) -> DistillationScope:
        if (
            self.analysis == RightsDecision.ALLOWED
            and self.output_distillation == RightsDecision.ALLOWED
            and self.output_only_meteo_publication == RightsDecision.ALLOWED
        ):
            return DistillationScope.PUBLIC_RELEASE_ALLOWED
        if (
            self.analysis in {RightsDecision.ALLOWED, RightsDecision.LIMITED}
            and self.output_distillation in {RightsDecision.ALLOWED, RightsDecision.LIMITED}
            and (
                self.analysis == RightsDecision.LIMITED
                or self.output_distillation == RightsDecision.LIMITED
            )
        ):
            return DistillationScope.LOCAL_AUTHORIZED_ONLY
        return DistillationScope.NOT_AUTHORIZED

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            {
                "distillation_scope": self.distillation_scope.value,
                "local_label_generation_allowed": self.distillation_scope
                in {
                    DistillationScope.PUBLIC_RELEASE_ALLOWED,
                    DistillationScope.LOCAL_AUTHORIZED_ONLY,
                },
                "public_checkpoint_allowed": (
                    self.distillation_scope is DistillationScope.PUBLIC_RELEASE_ALLOWED
                ),
            }
        )
        return result

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
            requires_explicit_local_authorization=(
                self.distillation_scope is DistillationScope.LOCAL_AUTHORIZED_ONLY
                and not allow_limited_local
            ),
            training_outputs_local_only=local_only and allow_limited_local,
        )


_GPL_OUTPUT_NOTE = (
    "USI moves, scores, and PVs are approved for output-only distillation because no "
    "separate output restriction was found; this does not authorize copying weights."
)


MODEL_RIGHTS: tuple[ModelRights, ...] = (
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
            "The private NAGISA-style NNUE run may extract progress.bin only as a fixed local "
            "LayerStack router; it never copies nn.bin, and every resulting export remains "
            "local-only unless separate redistribution permission is recorded.",
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
        rights_id="soujou-tsec7-paid",
        name="奏乗 TSEC7",
        version="TSEC7 / SOJO_TSEC7 NNUE 9.60 exact local profile",
        family="NNUE / HalfKaHmMerged / layer-stack 9",
        availability="paid",
        engine_license="GPL-3.0 YaneuraOu-derived source",
        model_terms=(
            "lawfully acquired user-supplied evaluation archive; archive has no LICENSE; "
            "ordinary USI labels are approved only for private local distillation"
        ),
        sources=(
            "https://booth.pm/ja/items/8606196",
            "https://github.com/keinoda/YaneuraOu/tree/sojo_tsec7",
        ),
        analysis=RightsDecision.LIMITED,
        output_distillation=RightsDecision.LIMITED,
        hard_game_training=RightsDecision.LIMITED,
        direct_weight_use=RightsDecision.NOT_APPROVED,
        original_artifact_redistribution=RightsDecision.NOT_APPROVED,
        output_only_meteo_publication=RightsDecision.NOT_APPROVED,
        notes=(
            "The user supplied the exact TSEC7 archive for this private local run; its original "
            "download and evaluation artifact must never enter the repository or a release.",
            "Run only through the pinned external USI engine with FV_SCALE=28, "
            "LS_BUCKET_MODE=progress8kpabs, and the supplied progress.bin.",
            "Raw labels and every checkpoint descended from them remain local until the rights "
            "holder explicitly clears publication of distilled Meteo weights.",
        ),
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


def public_distillable_rights_ids() -> tuple[str, ...]:
    """IDs whose output-only labels may flow into a public Meteo checkpoint."""

    return tuple(
        record.rights_id
        for record in MODEL_RIGHTS
        if record.distillation_scope is DistillationScope.PUBLIC_RELEASE_ALLOWED
    )


def distillable_rights_ids() -> tuple[str, ...]:
    """Backward-compatible alias for the public-release-safe teacher set.

    The historical name was ambiguous: it omitted authorized local-only
    teachers such as Suisho11Plus and could therefore be misread as "unused".
    New code and user-facing output should say ``public_distillable`` explicitly.
    """

    return public_distillable_rights_ids()


def locally_distillable_rights_ids() -> tuple[str, ...]:
    """IDs usable for USI labels, including explicitly acknowledged local-only sources."""

    eligible = {RightsDecision.ALLOWED, RightsDecision.LIMITED}
    return tuple(
        record.rights_id
        for record in MODEL_RIGHTS
        if record.analysis in eligible and record.output_distillation in eligible
    )


def local_only_distillable_rights_ids() -> tuple[str, ...]:
    """IDs whose labels are authorized only for an acknowledged local run."""

    return tuple(
        record.rights_id
        for record in MODEL_RIGHTS
        if record.distillation_scope is DistillationScope.LOCAL_AUTHORIZED_ONLY
    )


def not_authorized_rights_ids() -> tuple[str, ...]:
    """IDs that remain unavailable for label generation in this workspace."""

    return tuple(
        record.rights_id
        for record in MODEL_RIGHTS
        if record.distillation_scope is DistillationScope.NOT_AUTHORIZED
    )


def analysable_rights_ids() -> tuple[str, ...]:
    """IDs approved for analysis without a local authorization receipt."""

    return tuple(
        record.rights_id
        for record in MODEL_RIGHTS
        if record.analysis == RightsDecision.ALLOWED
    )
