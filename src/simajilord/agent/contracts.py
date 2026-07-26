"""Durable agent records without selecting or enabling a model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class GoalState(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class AgentGoal:
    goal_id: str
    instruction: str
    created_at: datetime
    next_review_at: datetime | None
    state: GoalState = GoalState.ACTIVE


@dataclass(frozen=True, slots=True)
class AgentEvent:
    event_id: str
    kind: str
    occurred_at: datetime
    workspace_id: str | None
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class ActionProposal:
    capability_name: str
    reason: str
    arguments: dict[str, object]
    deduplication_key: str
