"""Bounded audited execution for requester-confirmed human UI actions."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Sequence

from simajilord.core import CapabilityRegistry, InvocationContext, RiskLevel
from simajilord.core.errors import CapabilityError

from .actions import ActionReceiptService

log = logging.getLogger(__name__)


class HumanCapabilityExecutor:
    """Run a host allowlist through typed invocation and the durable effect ledger."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        action_receipts: ActionReceiptService,
        allowed_capabilities: Sequence[str],
        write_capabilities: Sequence[str],
    ) -> None:
        self._registry = registry
        self._action_receipts = action_receipts
        self._allowed = frozenset(allowed_capabilities)
        self._writes = frozenset(write_capabilities)
        if len(self._allowed) != len(tuple(allowed_capabilities)):
            raise ValueError("human capability allowlist contains duplicates")
        if len(self._writes) != len(tuple(write_capabilities)):
            raise ValueError("human write capability policy contains duplicates")
        if not self._writes <= self._allowed:
            raise ValueError("human write capabilities must be allowlisted")
        for name in self._allowed:
            endpoint = self._registry.endpoint(name)
            is_write = endpoint.descriptor.risk in {
                RiskLevel.WRITE,
                RiskLevel.DESTRUCTIVE,
            }
            if is_write != (name in self._writes):
                raise ValueError(f"human write policy does not match capability risk: {name}")
            if is_write and not self._action_receipts.has_explicit_policy(name):
                raise ValueError(f"human write lacks an Action Receipt policy: {name}")

    async def invoke(
        self,
        name: str,
        request: object,
        context: InvocationContext,
    ) -> object:
        """Invoke once, closing every planned effect into a terminal safe state."""

        if name not in self._allowed:
            raise CapabilityError(f"Human capability is not allowlisted: {name}")
        if name not in self._writes:
            return await self._registry.invoke(name, request, context)

        effect = await self._action_receipts.plan_external_effect(
            capability=name,
            request=request,
            context=context,
            authorization_reference=context.request_id,
        )
        invocation_context = dataclasses.replace(
            context,
            external_effect_dispatch=effect,
        )
        try:
            response = await self._registry.invoke(name, request, invocation_context)
        except BaseException as exc:
            try:
                if effect.dispatched or effect.completed_without_dispatch:
                    await self._action_receipts.mark_external_effect_unknown(
                        effect.effect_id,
                        context=context,
                    )
                elif isinstance(exc, asyncio.CancelledError):
                    await self._action_receipts.cancel_external_effect(
                        effect.effect_id,
                        context=context,
                    )
                else:
                    await self._action_receipts.reject_external_effect(
                        effect.effect_id,
                        context=context,
                    )
            except Exception:
                log.critical(
                    "Human external effect could not be closed effect=%s request=%s",
                    effect.effect_id,
                    context.request_id,
                    exc_info=True,
                )
            raise

        if not effect.dispatched and not effect.completed_without_dispatch:
            await effect.dispatch()
            await self._action_receipts.mark_external_effect_unknown(
                effect.effect_id,
                context=context,
            )
            raise RuntimeError("human write returned without crossing its external effect boundary")
        await self._action_receipts.record(
            capability=name,
            request=request,
            response=response,
            context=context,
            effect_id=effect.effect_id if effect.dispatched else None,
        )
        return response
