"""Package-owned, tool-only workflows for complex agent tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from simajilord.core import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    DisclosureClass,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.core.search import (
    normalized_substring,
    phrase_match_score,
    search_overlap_score,
)


@dataclass(frozen=True, slots=True)
class CuratedWorkflowSearchRequest:
    """Find a small reusable workflow by goal instead of user command syntax."""

    query: str = field(
        metadata={
            "description": (
                "Concrete goal and object, such as researching a Discord community "
                "or reading a long PDF."
            )
        }
    )
    limit: int = 3


@dataclass(frozen=True, slots=True)
class CuratedWorkflowStep:
    order: int
    capability: str
    instruction: str


@dataclass(frozen=True, slots=True)
class CuratedWorkflow:
    workflow_id: str
    summary: str
    required_grants: tuple[str, ...]
    steps: tuple[CuratedWorkflowStep, ...]
    stop_conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CuratedWorkflowCatalogItem:
    workflow_id: str
    summary: str


@dataclass(frozen=True, slots=True)
class CuratedWorkflowSearchResponse:
    query: str
    workflows: tuple[CuratedWorkflow, ...]
    catalog_index: tuple[CuratedWorkflowCatalogItem, ...]
    catalog_complete: bool


@dataclass(frozen=True, slots=True)
class _WorkflowDefinition:
    workflow: CuratedWorkflow
    keywords: tuple[str, ...]
    required_capabilities: frozenset[str]


def build_curated_workflow_endpoint(
    available_capabilities: frozenset[str],
    *,
    capability_grants: Mapping[str, str] | None = None,
    approval_capabilities: frozenset[str] = frozenset(),
) -> CapabilityEndpoint:
    """Expose only workflows whose Simajilord capabilities actually exist."""

    _validate_workflows(_WORKFLOWS)
    grants_by_capability = dict(capability_grants or {})
    unknown_policies = (
        set(grants_by_capability) | set(approval_capabilities)
    ) - available_capabilities
    if unknown_policies:
        raise ValueError(
            "Workflow policies reference unavailable capabilities: "
            + ", ".join(sorted(unknown_policies))
        )
    workflows = tuple(
        item
        for item in _WORKFLOWS
        if item.required_capabilities <= available_capabilities
    )

    async def search(
        request: CuratedWorkflowSearchRequest,
        context: InvocationContext,
    ) -> CuratedWorkflowSearchResponse:
        query = " ".join(request.query.split())
        if not query or len(query) > 200:
            raise UserError("workflow.query_invalid")
        if not 1 <= request.limit <= 5:
            raise UserError("workflow.limit_invalid")
        available = tuple(
            item
            for item in workflows
            if set(item.workflow.required_grants) <= context.grants
            and all(
                (
                    grants_by_capability.get(capability) is None
                    or grants_by_capability[capability] in context.grants
                )
                and (
                    capability not in approval_capabilities
                    or capability in context.approvals
                )
                for capability in item.required_capabilities
            )
        )
        matches = _search_workflows(query, available, limit=request.limit)
        return CuratedWorkflowSearchResponse(
            query=query,
            workflows=tuple(item.workflow for item in matches),
            catalog_index=tuple(
                CuratedWorkflowCatalogItem(
                    workflow_id=item.workflow.workflow_id,
                    summary=item.workflow.summary,
                )
                for item in available
            ),
            catalog_complete=True,
        )

    return endpoint(
        CapabilityDescriptor(
            name="workflow.search",
            summary=(
                "Find a package-owned multi-step workflow for Discord, web/PDF, "
                "files, media, memory, or reversible actions. Ranked matches are "
                "lexical hints only; always inspect the complete catalog_index "
                "semantically, then query the exact workflow_id when needed."
            ),
            risk=RiskLevel.READ,
            disclosure_class=DisclosureClass.NO_USER_CONTENT,
            keywords=(
                "workflow",
                "multi-step task",
                "community research",
                "long PDF",
                "safe compute",
                "media download",
                "memory",
                "undo",
                "手順",
                "調査",
                "長文",
                "保存",
            ),
            expected_errors=(
                "workflow.query_invalid",
                "workflow.limit_invalid",
            ),
            timeout_seconds=2,
        ),
        CuratedWorkflowSearchRequest,
        CuratedWorkflowSearchResponse,
        search,
    )


def _search_workflows(
    query: str,
    workflows: tuple[_WorkflowDefinition, ...],
    *,
    limit: int,
) -> tuple[_WorkflowDefinition, ...]:
    scored: list[tuple[int, str, _WorkflowDefinition]] = []
    for item in workflows:
        workflow = item.workflow
        searchable = " ".join(
            (
                workflow.workflow_id,
                workflow.summary,
                *item.keywords,
                *(step.instruction for step in workflow.steps),
            )
        )
        score = (
            search_overlap_score(query, searchable)
            + 3 * search_overlap_score(query, workflow.workflow_id)
            + 3 * phrase_match_score(query, item.keywords)
            + 2 * int(normalized_substring(query, searchable))
        )
        if score:
            scored.append((score, workflow.workflow_id, item))
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return tuple(item for _, _, item in scored[:limit])


def _validate_workflows(workflows: tuple[_WorkflowDefinition, ...]) -> None:
    identifiers = tuple(item.workflow.workflow_id for item in workflows)
    duplicates = sorted(
        identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
    )
    if duplicates:
        raise ValueError(
            "Duplicate curated workflow IDs: " + ", ".join(duplicates)
        )
    for item in workflows:
        orders = tuple(step.order for step in item.workflow.steps)
        expected = tuple(range(1, len(orders) + 1))
        if orders != expected:
            raise ValueError(
                f"Curated workflow step order is invalid: {item.workflow.workflow_id}"
            )


def _step(
    order: int,
    capability: str,
    instruction: str,
) -> CuratedWorkflowStep:
    return CuratedWorkflowStep(order, capability, instruction)


_WORKFLOWS = (
    _WorkflowDefinition(
        workflow=CuratedWorkflow(
            workflow_id="discord.community_research",
            summary=(
                "Research themes or participation across authorized Discord channels "
                "with bounded, representative sampling."
            ),
            required_grants=(),
            steps=(
                _step(
                    1,
                    "discord.search_messages",
                    "Search guild-wide with a concrete topic and bounded page size.",
                ),
                _step(
                    2,
                    "discord.read_messages",
                    "Read small windows around representative results across channels and time.",
                ),
                _step(
                    3,
                    "discord.search_messages",
                    "Run one contrasting query to reduce single-keyword selection bias.",
                ),
            ),
            stop_conditions=(
                "Enough independent samples support the recurring themes.",
                "The next page repeats existing evidence without changing the conclusion.",
                "Visibility rules leave too little evidence; state that limitation.",
            ),
        ),
        keywords=(
            "discord community research",
            "popular topic",
            "participation",
            "cross-channel",
            "server trends",
            "サーバー",
            "人気",
            "話題",
            "分析",
        ),
        required_capabilities=frozenset(
            {
                "discord.search_messages",
                "discord.read_messages",
            }
        ),
    ),
    _WorkflowDefinition(
        workflow=CuratedWorkflow(
            workflow_id="web.document_research",
            summary=(
                "Research current sources, then continue into long pages or PDFs "
                "without relying on search snippets alone."
            ),
            required_grants=("web",),
            steps=(
                _step(
                    1,
                    "web.search",
                    "Find primary sources with a narrow factual query.",
                ),
                _step(
                    2,
                    "web.fetch",
                    "Fetch the selected source and follow next_offset through needed chunks.",
                ),
                _step(
                    3,
                    "web.find",
                    "Locate exact terms or sections before fetching surrounding chunks.",
                ),
            ),
            stop_conditions=(
                "Primary sources directly support the answer.",
                "Material claims are cross-checked where independent confirmation matters.",
                "A source remains inaccessible; identify the gap instead of guessing.",
            ),
        ),
        keywords=(
            "web pdf research",
            "long document",
            "primary source",
            "current facts",
            "ページ内検索",
            "長文",
            "PDF",
            "調査",
            "文書",
            "読む",
            "根拠",
            "検索",
        ),
        required_capabilities=frozenset({"web.search", "web.fetch", "web.find"}),
    ),
    _WorkflowDefinition(
        workflow=CuratedWorkflow(
            workflow_id="file.safe_compute_transform",
            summary=(
                "Import or download a source file, inspect it in bounded chunks, "
                "transform it with safe compute, and deliver a verified derived file."
            ),
            required_grants=("files", "safe_compute"),
            steps=(
                _step(
                    1,
                    "discord.import_attachment",
                    "Import the exact requested attachment into the isolated workspace.",
                ),
                _step(
                    2,
                    "files.read",
                    "Inspect bounded chunks and preserve the imported source unchanged.",
                ),
                _step(
                    3,
                    "files.write_text",
                    "Write a separate script or derived text path with explicit inputs.",
                ),
                _step(
                    4,
                    "compute.run",
                    "Run argv-based safe compute and inspect its bounded result.",
                ),
                _step(
                    5,
                    "discord.send_file",
                    "Send only the verified derived output requested by the user.",
                ),
            ),
            stop_conditions=(
                "The derived output hash and size are verified.",
                "The source format is unsupported; explain the exact limitation.",
                "The requested operation would require network, shell, or host-file access.",
            ),
        ),
        keywords=(
            "file compute transform",
            "attachment edit",
            "workspace",
            "python calculation",
            "ファイル",
            "添付",
            "PDF",
            "内容",
            "編集",
            "修正",
            "書き換え",
            "計算",
            "変換",
            "返す",
            "送る",
        ),
        required_capabilities=frozenset(
            {
                "discord.import_attachment",
                "files.read",
                "files.write_text",
                "compute.run",
                "discord.send_file",
            }
        ),
    ),
    _WorkflowDefinition(
        workflow=CuratedWorkflow(
            workflow_id="media.save_and_deliver",
            summary=(
                "Resolve a user-supplied public media URL generically, save it into "
                "the isolated workspace, and deliver the resulting file."
            ),
            required_grants=("files", "media_download"),
            steps=(
                _step(
                    1,
                    "media.save",
                    "Resolve and save the supplied URL without platform-name dispatch rules.",
                ),
                _step(
                    2,
                    "files.list",
                    "Verify the saved filename, size, and workspace-relative path.",
                ),
                _step(
                    3,
                    "discord.send_file",
                    "Deliver the saved file only after media.save reports success.",
                ),
            ),
            stop_conditions=(
                "The terminal save result and file metadata are verified.",
                "The extractor reports authentication, challenge, or unsupported media.",
                "The file exceeds configured or Discord delivery limits.",
            ),
        ),
        keywords=(
            "media download save",
            "video URL",
            "yt-dlp",
            "deliver file",
            "動画",
            "音声",
            "ダウンロード",
            "保存",
            "送信",
            "ファイル",
            "送る",
            "届ける",
        ),
        required_capabilities=frozenset(
            {"media.save", "files.list", "discord.send_file"}
        ),
    ),
    _WorkflowDefinition(
        workflow=CuratedWorkflow(
            workflow_id="action.execute_with_receipt",
            summary=(
                "Discover and execute a requested Discord write, verify its Action "
                "Receipt, and keep a bounded Undo path when one exists."
            ),
            required_grants=("discord_message",),
            steps=(
                _step(
                    1,
                    "capability_search",
                    "Inspect the complete available index for the concrete action and object.",
                ),
                _step(
                    2,
                    "capability_describe",
                    "Load exactly one selected write contract and retain its contract ID.",
                ),
                _step(
                    3,
                    "capability_invoke",
                    "Invoke it with that contract ID, active authorization, and schema fields.",
                ),
                _step(
                    4,
                    "action.undo",
                    "Use the receipt action_id only if the user later asks to revert it.",
                ),
            ),
            stop_conditions=(
                "The capability result and action_receipt confirm success or failure.",
                "Authorization or Discord permissions reject the action; report the reason.",
                "No Undo exists; state that before any high-impact non-reversible action.",
            ),
        ),
        keywords=(
            "discord action receipt undo",
            "reversible write",
            "role pin timeout",
            "thread forum timer audio moderation",
            "操作",
            "元に戻す",
            "取り消し",
            "スレッド",
            "フォーラム",
            "役職",
            "ロール",
            "タイマー",
            "音楽",
            "一時停止",
            "モデレーション",
        ),
        required_capabilities=frozenset({"action.undo"}),
    ),
    _WorkflowDefinition(
        workflow=CuratedWorkflow(
            workflow_id="memory.selective_capture",
            summary=(
                "Reuse or selectively save an explicit stable preference or a "
                "reusable verified success/failure lesson without creating a turn log."
            ),
            required_grants=("memory",),
            steps=(
                _step(
                    1,
                    "memory.search",
                    "Search for an existing semantic key before creating a memory.",
                ),
                _step(
                    2,
                    "memory.remember",
                    "Save only a high-confidence user-stated preference or one "
                    "reusable verified success/failure lesson.",
                ),
                _step(
                    3,
                    "memory.update",
                    "Update an existing record when newer sourced evidence supersedes it.",
                ),
                _step(
                    4,
                    "memory.forget",
                    "Forget by returned memory_id only when the user explicitly asks.",
                ),
            ),
            stop_conditions=(
                "No stable or reusable fact exists; do not save anything.",
                "The information is inferred, sensitive, secret, or merely conversational.",
                "An existing memory already captures the same fact.",
            ),
        ),
        keywords=(
            "memory preference procedure",
            "remember success",
            "remember failure",
            "long-term",
            "メモリ",
            "好み",
            "以前",
            "成功",
            "失敗",
            "うまくいかなかった",
            "やり方",
            "再利用",
            "手順",
            "成功手順",
            "覚えて",
            "思い出す",
        ),
        required_capabilities=frozenset(
            {
                "memory.search",
                "memory.remember",
                "memory.update",
                "memory.forget",
            }
        ),
    ),
    _WorkflowDefinition(
        workflow=CuratedWorkflow(
            workflow_id="discord.platform_inspection",
            summary=(
                "Inspect a Discord server's members, effective permissions, presence, "
                "voice participation, channels, and low-frequency platform resources "
                "without loading unrelated resource families."
            ),
            required_grants=(),
            steps=(
                _step(
                    1,
                    "discord.list_servers",
                    "Resolve a mutually joined server and keep its exact server ID.",
                ),
                _step(
                    2,
                    "discord.list_members",
                    "Search or page members only when the question concerns people.",
                ),
                _step(
                    3,
                    "discord.inspect_user",
                    "Inspect the exact member for full Presence, activities, roles, and VC state.",
                ),
                _step(
                    4,
                    "discord.inspect_channel",
                    "Inspect settings, overwrites, and requester/Bot effective permissions.",
                ),
                _step(
                    5,
                    "discord.list_platform_resources",
                    "Request only the named resource family, such as events or audit logs.",
                ),
            ),
            stop_conditions=(
                "The requested live state is returned with complete=true.",
                "Presence intent or Discord cache completeness is false; state the limit.",
                "Either requester or Bot lacks visibility; do not infer hidden state.",
            ),
        ),
        keywords=(
            "discord platform api inspect",
            "presence voice permissions status",
            "member channel audit event",
            "Discord API",
            "ステータス",
            "オンライン",
            "VC",
            "権限",
            "管理者",
            "メンバー",
            "チャンネル",
            "監査ログ",
        ),
        required_capabilities=frozenset(
            {
                "discord.list_servers",
                "discord.list_members",
                "discord.inspect_user",
                "discord.inspect_channel",
                "discord.list_platform_resources",
            }
        ),
    ),
    _WorkflowDefinition(
        workflow=CuratedWorkflow(
            workflow_id="image.generate_review_publish",
            summary=(
                "Generate an image as a restart-safe job, inspect its bounded preview, "
                "and independently decide whether to publish the full original."
            ),
            required_grants=("image", "files"),
            steps=(
                _step(
                    1,
                    "image.generate",
                    "Submit a complete brief and wait for the terminal workspace handoff.",
                ),
                _step(
                    2,
                    "image.status",
                    "Use only when resuming a known job or confirming terminal state.",
                ),
                _step(
                    3,
                    "discord.send_file",
                    (
                        "Decide from the exact request's meaning and conversation context "
                        "whether publishing the full original fulfils the user's intent; "
                        "no particular delivery verb is required."
                    ),
                ),
            ),
            stop_conditions=(
                "The terminal result and model-visible preview are available.",
                (
                    "The request means the result should remain private for comparison or "
                    "iteration, or privacy or safety risk is unresolved; do not publish."
                ),
                "Discord attachment delivery succeeds or returns an exact permission error.",
            ),
        ),
        keywords=(
            "image generate review publish",
            "image job preview attachment",
            "画像生成",
            "画像",
            "描く",
            "添付",
            "投稿",
            "隠す",
            "見せない",
            "段階",
        ),
        required_capabilities=frozenset(
            {"image.generate", "image.status", "discord.send_file"}
        ),
    ),
)
