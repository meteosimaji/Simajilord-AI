"""Typed capabilities for bounded inspection of Simajilord's own source."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from simajilord.core.capabilities import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    DisclosureClass,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.services.source_inspection import SourceInspectionService


@dataclass(frozen=True, slots=True)
class SourceSearchRequest:
    query: str
    path_prefix: str | None = field(
        default=None,
        metadata={
            "description": ("Optional repository-relative directory such as src/simajilord/agent.")
        },
    )
    limit: int = 12


@dataclass(frozen=True, slots=True)
class SourceMatchItem:
    path: str
    line: int
    text: str


@dataclass(frozen=True, slots=True)
class SourceSearchResponse:
    matches: tuple[SourceMatchItem, ...]
    searched_files: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class SourceReadRequest:
    path: str
    start_line: int = 1
    max_lines: int = 120
    max_characters: int = 8_000


@dataclass(frozen=True, slots=True)
class SourceReadResponse:
    path: str
    start_line: int
    end_line: int
    total_lines: int
    content: str
    next_line: int | None
    sha256: str


@dataclass(frozen=True, slots=True)
class EvidencePlanRequest:
    """The model's semantic decision; the host never derives it from user text."""

    execution_model: Literal["primary", "escalation"]
    conversation_context: Literal["required", "not_required"] = field(
        metadata={
            "description": (
                "Required only when earlier Discord messages can change the active request's "
                "meaning or referent. Current live platform state alone does not require history."
            )
        },
    )
    source_inspection: Literal["required", "not_required"] = field(
        metadata={
            "description": (
                "Required only when current Simajilord implementation evidence is needed."
            )
        },
    )
    capability_discovery: Literal["required", "not_required"] = field(
        metadata={
            "description": (
                "Required whenever the answer itself may assert, deny, explain, or give an "
                "opinion about a current Simajilord state, ability, or action. This includes "
                "questions about whether something is possible even when no execution is "
                "requested. The model must then use capability_search's complete index."
            )
        },
    )
    reason: str


@dataclass(frozen=True, slots=True)
class EvidencePlanResponse:
    execution_model: Literal["primary", "escalation"]
    conversation_context: Literal["required", "not_required"]
    source_inspection: Literal["required", "not_required"]
    capability_discovery: Literal["required", "not_required"]
    reason: str
    recorded: bool


def build_source_inspection_endpoints(
    service: SourceInspectionService,
) -> tuple[CapabilityEndpoint, ...]:
    async def evidence_plan(
        request: EvidencePlanRequest,
        _: InvocationContext,
    ) -> EvidencePlanResponse:
        reason = " ".join(request.reason.split())
        if not reason or len(reason) > 400:
            raise UserError("agent.evidence_plan_reason_invalid")
        return EvidencePlanResponse(
            execution_model=request.execution_model,
            conversation_context=request.conversation_context,
            source_inspection=request.source_inspection,
            capability_discovery=request.capability_discovery,
            reason=reason,
            recorded=True,
        )

    async def search(
        request: SourceSearchRequest,
        _: InvocationContext,
    ) -> SourceSearchResponse:
        result = await service.search(
            request.query,
            path_prefix=request.path_prefix,
            limit=request.limit,
        )
        return SourceSearchResponse(
            matches=tuple(
                SourceMatchItem(match.path, match.line, match.text) for match in result.matches
            ),
            searched_files=result.searched_files,
            truncated=result.truncated,
        )

    async def read(
        request: SourceReadRequest,
        _: InvocationContext,
    ) -> SourceReadResponse:
        result = await service.read(
            request.path,
            start_line=request.start_line,
            max_lines=request.max_lines,
            max_characters=request.max_characters,
        )
        return SourceReadResponse(
            path=result.path,
            start_line=result.start_line,
            end_line=result.end_line,
            total_lines=result.total_lines,
            content=result.content,
            next_line=result.next_line,
            sha256=result.sha256,
        )

    return (
        endpoint(
            CapabilityDescriptor(
                name="turn.evidence_plan",
                summary=(
                    "After reading the exact active request, semantically decide the "
                    "required evidence and execution model. Default to the primary model "
                    "and use decomposition, evidence, and tools; length or technicality "
                    "alone is not a reason to escalate. Choose the escalation model only "
                    "for a concrete residual judgment or reliability risk the harness "
                    "cannot adequately resolve. Declare capability discovery required for "
                    "claims about current Simajilord state, ability, or action, not only when "
                    "executing one. The AI decides from meaning, never from a host keyword list."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.NO_USER_CONTENT,
                keywords=(
                    "evidence plan",
                    "source decision",
                    "根拠計画",
                    "ソース確認判断",
                ),
                idempotency="read",
                expected_errors=(
                    "agent.evidence_plan_reason_invalid",
                    "agent.event_message_not_read",
                ),
                timeout_seconds=5,
            ),
            EvidencePlanRequest,
            EvidencePlanResponse,
            evidence_plan,
        ),
        endpoint(
            CapabilityDescriptor(
                name="source.search",
                summary=(
                    "Search bounded lines in this running Simajilord checkout or package "
                    "without exposing runtime data, secrets, or the full repository."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.NO_USER_CONTENT,
                keywords=(
                    "source",
                    "repository",
                    "implementation",
                    "code search",
                    "ソース",
                    "レポジトリ",
                    "実装",
                    "コード検索",
                ),
                idempotency="read",
                expected_errors=(
                    "source.query_invalid",
                    "source.limit_invalid",
                    "source.path_forbidden",
                ),
                timeout_seconds=5,
            ),
            SourceSearchRequest,
            SourceSearchResponse,
            search,
        ),
        endpoint(
            CapabilityDescriptor(
                name="source.read",
                summary=(
                    "Read one bounded line range from an allowlisted Simajilord source, "
                    "test, or documentation file returned by source.search."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.NO_USER_CONTENT,
                keywords=(
                    "source",
                    "repository",
                    "implementation",
                    "read code",
                    "ソース",
                    "レポジトリ",
                    "実装",
                    "コードを読む",
                ),
                idempotency="read",
                expected_errors=(
                    "source.path_forbidden",
                    "source.start_line_invalid",
                    "source.max_lines_invalid",
                    "source.max_characters_invalid",
                    "source.file_too_large",
                    "source.encoding_unsupported",
                    "source.read_failed",
                ),
                timeout_seconds=5,
            ),
            SourceReadRequest,
            SourceReadResponse,
            read,
        ),
    )
