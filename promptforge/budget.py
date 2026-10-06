from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite
from typing import Any, Callable, Mapping, Sequence

from .core import estimate_tokens, serialize_context


TokenEstimator = Callable[[Any], int]


@dataclass(frozen=True)
class ContextBlock:
    """One independently controllable context unit.

    Utility is caller-supplied. PromptForge does not infer semantic importance
    from prose. Required blocks are pinned and are never removed while fitting
    a budget. Optional blocks compete for the remaining token budget.
    """

    block_id: str
    value: Any
    utility: float = 1.0
    required: bool = False
    path: str | None = None
    token_estimate: int | None = None
    kind: str = "context"

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id.strip():
            raise ValueError("block_id must not be empty")
        if (
            not isinstance(self.utility, (int, float))
            or isinstance(self.utility, bool)
            or not isfinite(float(self.utility))
            or float(self.utility) < 0.0
        ):
            raise ValueError("utility must be a finite non-negative number")
        if self.path is not None and (
            not isinstance(self.path, str) or not self.path.strip()
        ):
            raise ValueError("path must be non-empty when supplied")
        if self.token_estimate is not None and (
            not isinstance(self.token_estimate, int)
            or isinstance(self.token_estimate, bool)
            or self.token_estimate < 1
        ):
            raise ValueError("token_estimate must be a positive integer when supplied")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a bool")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("kind must not be empty")

    def estimate(self, estimator: TokenEstimator | None = None) -> int:
        if self.token_estimate is not None:
            return self.token_estimate
        estimate = estimator or (lambda value: estimate_tokens(serialize_context(value)))
        tokens = int(estimate(self.value))
        if tokens < 1:
            raise ValueError("token estimator must return a positive integer")
        return tokens

    def to_dict(self, *, token_cost: int | None = None) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "path": self.path,
            "kind": self.kind,
            "utility": float(self.utility),
            "required": self.required,
            "token_estimate": (
                self.token_estimate if token_cost is None else token_cost
            ),
        }


@dataclass(frozen=True)
class ContextBudgetPolicy:
    """Hard input-token budget with explicit safety/headroom reservation."""

    budget_tokens: int
    reserve_tokens: int = 0
    reserve_ratio: float = 0.0
    min_optional_utility: float = 0.0
    max_optional_blocks: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.budget_tokens, int)
            or isinstance(self.budget_tokens, bool)
            or self.budget_tokens < 1
        ):
            raise ValueError("budget_tokens must be a positive integer")
        if (
            not isinstance(self.reserve_tokens, int)
            or isinstance(self.reserve_tokens, bool)
            or self.reserve_tokens < 0
        ):
            raise ValueError("reserve_tokens must be a non-negative integer")
        if (
            not isinstance(self.reserve_ratio, (int, float))
            or isinstance(self.reserve_ratio, bool)
            or not isfinite(float(self.reserve_ratio))
            or not 0.0 <= float(self.reserve_ratio) < 1.0
        ):
            raise ValueError("reserve_ratio must be between 0.0 and 1.0")
        if (
            not isinstance(self.min_optional_utility, (int, float))
            or isinstance(self.min_optional_utility, bool)
            or not isfinite(float(self.min_optional_utility))
            or float(self.min_optional_utility) < 0.0
        ):
            raise ValueError(
                "min_optional_utility must be a finite non-negative number"
            )
        if self.max_optional_blocks is not None and (
            not isinstance(self.max_optional_blocks, int)
            or isinstance(self.max_optional_blocks, bool)
            or self.max_optional_blocks < 0
        ):
            raise ValueError(
                "max_optional_blocks must be a non-negative integer or None"
            )

    @property
    def effective_reserve_tokens(self) -> int:
        return max(
            self.reserve_tokens,
            ceil(self.budget_tokens * float(self.reserve_ratio)),
        )

    @property
    def usable_tokens(self) -> int:
        return max(0, self.budget_tokens - self.effective_reserve_tokens)


@dataclass(frozen=True)
class ContextBudgetPlan:
    """Deterministic, auditable token-budget decision."""

    schema_version: str
    budget_tokens: int
    reserve_tokens: int
    usable_tokens: int
    baseline_tokens: int
    selected_tokens: int
    tokens_saved: int
    reduction_ratio: float | None
    budget_satisfied: bool
    included_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    required_ids: tuple[str, ...]
    optional_included_ids: tuple[str, ...]
    token_costs: dict[str, int]
    reasons: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if self.schema_version != "context-budget.v1":
            raise ValueError("schema_version must be context-budget.v1")
        if self.budget_tokens < 1:
            raise ValueError("budget_tokens must be positive")
        if self.reserve_tokens < 0 or self.usable_tokens < 0:
            raise ValueError("budget reserves must be non-negative")
        if self.selected_tokens > self.usable_tokens:
            raise ValueError("selected_tokens cannot exceed usable_tokens")
        if self.baseline_tokens < 0 or self.tokens_saved < 0:
            raise ValueError("token counts must be non-negative")
        if self.reduction_ratio is not None and not 0.0 <= self.reduction_ratio <= 1.0:
            raise ValueError("reduction_ratio must be between 0.0 and 1.0")
        if len(set(self.included_ids)) != len(self.included_ids):
            raise ValueError("included_ids must be unique")
        if len(set(self.excluded_ids)) != len(self.excluded_ids):
            raise ValueError("excluded_ids must be unique")
        if set(self.included_ids).intersection(self.excluded_ids):
            raise ValueError("included_ids and excluded_ids must be disjoint")

    @property
    def omitted_ids(self) -> tuple[str, ...]:
        return self.excluded_ids

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "included_ids",
            "excluded_ids",
            "required_ids",
            "optional_included_ids",
        ):
            payload[key] = list(getattr(self, key))
        payload["reduction_ratio"] = self.reduction_ratio
        payload["reasons"] = {
            key: list(value) for key, value in self.reasons.items()
        }
        return payload

    def compact_manifest(
        self,
        blocks: Sequence[ContextBlock],
    ) -> dict[str, Any]:
        """Return a small model-facing manifest without omitted content."""

        by_id = {block.block_id: block for block in blocks}
        return {
            "schema_version": self.schema_version,
            "budget_tokens": self.budget_tokens,
            "usable_tokens": self.usable_tokens,
            "selected_tokens": self.selected_tokens,
            "tokens_saved": self.tokens_saved,
            "included": [
                {
                    "block_id": block_id,
                    "path": by_id[block_id].path,
                    "kind": by_id[block_id].kind,
                }
                for block_id in self.included_ids
                if block_id in by_id
            ],
            "omitted": [
                {
                    "block_id": block_id,
                    "path": by_id[block_id].path,
                    "kind": by_id[block_id].kind,
                    "token_estimate": self.token_costs.get(block_id),
                    "reason": list(
                        self.reasons.get(block_id, ("not selected",))
                    ),
                }
                for block_id in self.excluded_ids
                if block_id in by_id
            ],
        }

    def materialize(
        self,
        blocks: Sequence[ContextBlock],
    ) -> dict[str, Any]:
        """Materialize selected blocks into a nested mapping.

        Blocks without a path use their block id as the top-level key. If two
        blocks share a path, the later block in the supplied sequence wins.
        Integrations should avoid overlapping paths unless that overwrite is
        intentional.
        """

        selected = set(self.included_ids)
        output: dict[str, Any] = {}

        for block in blocks:
            if block.block_id not in selected:
                continue
            path = block.path or block.block_id
            parts = path.split(".")
            current = output
            for segment in parts[:-1]:
                child = current.get(segment)
                if not isinstance(child, dict):
                    child = {}
                    current[segment] = child
                current = child
            current[parts[-1]] = block.value

        return output


class ContextBudgetPlanner:
    """Pack required and useful optional context deterministically.

    Optional blocks are ranked by utility density (utility / token cost). This
    is a fast heuristic rather than an optimal knapsack solver. The caller can
    provide exact token costs when a provider tokenizer is available.
    """

    def __init__(
        self,
        policy: ContextBudgetPolicy,
        *,
        estimator: TokenEstimator | None = None,
    ) -> None:
        if not isinstance(policy, ContextBudgetPolicy):
            raise TypeError("policy must be a ContextBudgetPolicy")
        self.policy = policy
        self.estimator = estimator

    def plan(self, blocks: Sequence[ContextBlock]) -> ContextBudgetPlan:
        values = tuple(blocks)
        if not values:
            raise ValueError("blocks must contain at least one ContextBlock")

        ids = [block.block_id for block in values]
        if len(set(ids)) != len(ids):
            raise ValueError("block_id values must be unique")

        costs = {
            block.block_id: block.estimate(self.estimator)
            for block in values
        }
        baseline_tokens = sum(costs.values())

        required = tuple(block for block in values if block.required)
        optional = tuple(block for block in values if not block.required)

        required_tokens = sum(costs[block.block_id] for block in required)
        if required_tokens > self.policy.usable_tokens:
            raise ValueError(
                "required context exceeds usable token budget: "
                f"{required_tokens} > {self.policy.usable_tokens}"
            )

        included = [block.block_id for block in required]
        reasons: dict[str, tuple[str, ...]] = {
            block.block_id: ("required block pinned",)
            for block in required
        }
        remaining = self.policy.usable_tokens - required_tokens
        optional_included: list[str] = []

        ranked_optional = sorted(
            (
                block
                for block in optional
                if block.utility >= self.policy.min_optional_utility
            ),
            key=lambda block: (
                -(float(block.utility) / costs[block.block_id]),
                -float(block.utility),
                costs[block.block_id],
                block.block_id,
            ),
        )

        for block in ranked_optional:
            if (
                self.policy.max_optional_blocks is not None
                and len(optional_included) >= self.policy.max_optional_blocks
            ):
                reasons[block.block_id] = ("optional block count limit reached",)
                continue

            cost = costs[block.block_id]
            if cost <= remaining:
                included.append(block.block_id)
                optional_included.append(block.block_id)
                remaining -= cost
                reasons[block.block_id] = ("selected by utility density",)
            else:
                reasons[block.block_id] = (
                    "omitted because it does not fit remaining token budget",
                )

        for block in optional:
            reasons.setdefault(
                block.block_id,
                ("omitted by optional-utility threshold",),
            )

        included_set = set(included)
        excluded = tuple(block.block_id for block in values if block.block_id not in included_set)
        selected_tokens = sum(costs[item] for item in included)
        tokens_saved = max(0, baseline_tokens - selected_tokens)
        reduction_ratio = (
            tokens_saved / baseline_tokens
            if baseline_tokens > 0
            else None
        )

        return ContextBudgetPlan(
            schema_version="context-budget.v1",
            budget_tokens=self.policy.budget_tokens,
            reserve_tokens=self.policy.effective_reserve_tokens,
            usable_tokens=self.policy.usable_tokens,
            baseline_tokens=baseline_tokens,
            selected_tokens=selected_tokens,
            tokens_saved=tokens_saved,
            reduction_ratio=reduction_ratio,
            budget_satisfied=selected_tokens <= self.policy.usable_tokens,
            included_ids=tuple(included),
            excluded_ids=excluded,
            required_ids=tuple(block.block_id for block in required),
            optional_included_ids=tuple(optional_included),
            token_costs=costs,
            reasons=reasons,
        )


def plan_context(
    blocks: Sequence[ContextBlock],
    *,
    budget_tokens: int,
    reserve_tokens: int = 0,
    reserve_ratio: float = 0.0,
    min_optional_utility: float = 0.0,
    max_optional_blocks: int | None = None,
    estimator: TokenEstimator | None = None,
) -> ContextBudgetPlan:
    """Fast public helper for token-aware context packing."""

    return ContextBudgetPlanner(
        ContextBudgetPolicy(
            budget_tokens=budget_tokens,
            reserve_tokens=reserve_tokens,
            reserve_ratio=reserve_ratio,
            min_optional_utility=min_optional_utility,
            max_optional_blocks=max_optional_blocks,
        ),
        estimator=estimator,
    ).plan(blocks)


__all__ = [
    "TokenEstimator",
    "ContextBlock",
    "ContextBudgetPolicy",
    "ContextBudgetPlan",
    "ContextBudgetPlanner",
    "plan_context",
]
