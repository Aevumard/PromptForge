from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .budget import ContextBlock, ContextBudgetPlan, plan_context
from .deferred import build_context_packet
from .tool_output import ToolOutputItem, ToolOutputTrimResult, trim_tool_outputs


@dataclass(frozen=True)
class AgentInputPacket:
    """Single model-facing handoff for the common PromptForge path.

    It combines deterministic context budgeting, progressive disclosure, and
    optional tool-output trimming without requiring the integration to compose
    individual planners itself.
    """

    schema_version: str
    context: Mapping[str, Any]
    context_tokens: int
    budget_tokens: int
    tokens_saved: int
    deferred_catalog: Mapping[str, Any]
    tool_outputs: Sequence[Mapping[str, Any]]
    tool_output_chars_saved: int
    tool_output_tokens_saved: int
    omitted_context_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "agent-input.v1":
            raise ValueError("schema_version must be agent-input.v1")
        if self.context_tokens < 0:
            raise ValueError("context_tokens must be non-negative")
        if self.budget_tokens < 1:
            raise ValueError("budget_tokens must be positive")
        if self.tokens_saved < 0:
            raise ValueError("tokens_saved must be non-negative")
        if self.tool_output_chars_saved < 0:
            raise ValueError("tool_output_chars_saved must be non-negative")
        if self.tool_output_tokens_saved < 0:
            raise ValueError(
                "tool_output_tokens_saved must be non-negative"
            )

    @property
    def total_estimated_savings(self) -> int:
        return self.tokens_saved + self.tool_output_tokens_saved

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "context": dict(self.context),
            "context_tokens": self.context_tokens,
            "budget_tokens": self.budget_tokens,
            "tokens_saved": self.tokens_saved,
            "deferred_catalog": dict(self.deferred_catalog),
            "tool_outputs": [dict(item) for item in self.tool_outputs],
            "tool_output_chars_saved": self.tool_output_chars_saved,
            "tool_output_tokens_saved": self.tool_output_tokens_saved,
            "total_estimated_savings": self.total_estimated_savings,
            "omitted_context_ids": list(self.omitted_context_ids),
        }


@dataclass(frozen=True)
class AgentPreparation:
    """Auditable preparation state behind one model-facing packet."""

    packet: AgentInputPacket
    budget_plan: ContextBudgetPlan
    tool_output_result: ToolOutputTrimResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet": self.packet.to_dict(),
            "budget_plan": self.budget_plan.to_dict(),
            "tool_output_result": (
                None
                if self.tool_output_result is None
                else self.tool_output_result.to_dict()
            ),
        }


def prepare_agent_input(
    blocks: Sequence[ContextBlock],
    *,
    budget_tokens: int,
    reserve_tokens: int = 0,
    reserve_ratio: float = 0.0,
    min_optional_utility: float = 0.0,
    max_optional_blocks: int | None = None,
    descriptions: Mapping[str, str] | None = None,
    estimator=None,
    tool_outputs: Sequence[ToolOutputItem] = (),
    recent_tool_turns: int = 2,
    max_tool_output_chars: int = 2000,
    tool_preview_chars: int = 600,
    eligible_tools=None,
) -> AgentPreparation:
    """One-call AI fast path for context preparation.

    The function never calls a model, never executes a tool, and never
    semantically summarizes evidence. It only performs deterministic
    admission/budgeting and optional output trimming.
    """

    plan = plan_context(
        blocks,
        budget_tokens=budget_tokens,
        reserve_tokens=reserve_tokens,
        reserve_ratio=reserve_ratio,
        min_optional_utility=min_optional_utility,
        max_optional_blocks=max_optional_blocks,
        estimator=estimator,
    )
    delivery = build_context_packet(
        plan,
        blocks,
        descriptions=descriptions,
    )

    tool_result = (
        None
        if not tool_outputs
        else trim_tool_outputs(
            tool_outputs,
            recent_turns=recent_tool_turns,
            max_output_chars=max_tool_output_chars,
            preview_chars=tool_preview_chars,
            eligible_tools=eligible_tools,
        )
    )

    tool_items = (
        []
        if tool_result is None
        else tool_result.model_input
    )

    packet = AgentInputPacket(
        schema_version="agent-input.v1",
        context=delivery.context,
        context_tokens=delivery.context_tokens,
        budget_tokens=delivery.budget_tokens,
        tokens_saved=delivery.tokens_saved,
        deferred_catalog=delivery.deferred_catalog,
        tool_outputs=tool_items,
        tool_output_chars_saved=(
            0 if tool_result is None else tool_result.chars_saved
        ),
        tool_output_tokens_saved=(
            0
            if tool_result is None
            else tool_result.estimated_tokens_saved
        ),
        omitted_context_ids=tuple(
            plan.excluded_ids
        ),
    )
    return AgentPreparation(
        packet=packet,
        budget_plan=plan,
        tool_output_result=tool_result,
    )


__all__ = [
    "AgentInputPacket",
    "AgentPreparation",
    "prepare_agent_input",
]
