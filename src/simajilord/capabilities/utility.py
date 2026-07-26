"""Small deterministic-shape utilities reusable across chat transports."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from simajilord.core.capabilities import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError


@dataclass(frozen=True, slots=True)
class RollRequest:
    dice: int = 1
    sides: int = 6


@dataclass(frozen=True, slots=True)
class RollResponse:
    rolls: tuple[int, ...]
    total: int


@dataclass(frozen=True, slots=True)
class ChooseRequest:
    options: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChooseResponse:
    choice: str
    option_count: int


def build_utility_endpoints() -> tuple[CapabilityEndpoint, CapabilityEndpoint]:
    async def roll(request: RollRequest, _: InvocationContext) -> RollResponse:
        if not 1 <= request.dice <= 20:
            raise UserError("utility.dice_count_invalid")
        if not 2 <= request.sides <= 1_000:
            raise UserError("utility.dice_sides_invalid")
        values = tuple(secrets.randbelow(request.sides) + 1 for _ in range(request.dice))
        return RollResponse(rolls=values, total=sum(values))

    async def choose(request: ChooseRequest, _: InvocationContext) -> ChooseResponse:
        options = tuple(option.strip() for option in request.options if option.strip())
        if not 2 <= len(options) <= 20:
            raise UserError("utility.option_count_invalid")
        if any(len(option) > 100 for option in options):
            raise UserError("utility.option_too_long")
        return ChooseResponse(
            choice=secrets.choice(options),
            option_count=len(options),
        )

    return (
        endpoint(
            CapabilityDescriptor(
                name="utility.roll",
                summary="Roll bounded virtual dice using the host secure random source.",
                risk=RiskLevel.READ,
                keywords=("dice", "random", "roll", "game"),
            ),
            RollRequest,
            RollResponse,
            roll,
        ),
        endpoint(
            CapabilityDescriptor(
                name="utility.choose",
                summary="Choose one item from a bounded list.",
                risk=RiskLevel.READ,
                keywords=("choose", "pick", "random", "decision"),
            ),
            ChooseRequest,
            ChooseResponse,
            choose,
        ),
    )
