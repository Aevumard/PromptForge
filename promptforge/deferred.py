from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .budget import ContextBlock, ContextBudgetPlan


@dataclass(frozen=True)
class DeferredContextItem:
    """Compact catalog entry for context that is available but not loaded."""

    block: ContextBlock
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.block, ContextBlock):
            raise TypeError("block must be a ContextBlock")
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")

    def catalog_entry(self, *, token_cost: int | None = None) -> dict[str, Any]:
        cost = token_cost
        if cost is None:
            cost = self.block.token_estimate or self.block.estimate()
        return {
            "block_id": self.block.block_id,
            "path": self.block.path,
            "kind": self.block.kind,
            "description": self.description,
            "token_estimate": cost,
            "required": self.block.required,
        }


@dataclass(frozen=True)
class ContextLoadResult:
    """Small, auditable result of an explicit on-demand context load."""

    schema_version: str
    requested_ids: tuple[str, ...]
    loaded_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    omitted_ids: tuple[str, ...]
    selected_tokens: int
    token_budget: int | None
    materialized: Mapping[str, Any]
    reasons: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if self.schema_version != "context-load.v1":
            raise ValueError("schema_version must be context-load.v1")
        if self.selected_tokens < 0:
            raise ValueError("selected_tokens must be non-negative")
        overlap = set(self.loaded_ids).intersection(self.omitted_ids)
        if overlap:
            raise ValueError("loaded_ids and omitted_ids must be disjoint")
        if set(self.loaded_ids).intersection(self.unknown_ids):
            raise ValueError("loaded_ids and unknown_ids must be disjoint")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requested_ids": list(self.requested_ids),
            "loaded_ids": list(self.loaded_ids),
            "unknown_ids": list(self.unknown_ids),
            "omitted_ids": list(self.omitted_ids),
            "selected_tokens": self.selected_tokens,
            "token_budget": self.token_budget,
            "materialized": dict(self.materialized),
            "reasons": {
                key: list(value) for key, value in self.reasons.items()
            },
        }


class DeferredContextCatalog:
    """Progressive-disclosure catalog for expensive optional context.

    The model-facing catalog contains metadata only. Full block values enter
    context only after an explicit load request. Loading is deterministic and
    bounded; PromptForge never guesses which deferred block the model wants.
    """

    def __init__(self, items: Sequence[DeferredContextItem]) -> None:
        values = tuple(items)
        ids = [item.block.block_id for item in values]
        if len(set(ids)) != len(ids):
            raise ValueError("deferred block_id values must be unique")
        self._items = values
        self._by_id = {item.block.block_id: item for item in values}

    @classmethod
    def from_blocks(
        cls,
        blocks: Sequence[ContextBlock],
        *,
        descriptions: Mapping[str, str] | None = None,
    ) -> "DeferredContextCatalog":
        descriptions = descriptions or {}
        return cls(
            DeferredContextItem(
                block,
                descriptions.get(block.block_id, ""),
            )
            for block in blocks
        )

    @classmethod
    def from_budget_plan(
        cls,
        plan: ContextBudgetPlan,
        blocks: Sequence[ContextBlock],
        *,
        descriptions: Mapping[str, str] | None = None,
    ) -> "DeferredContextCatalog":
        selected = set(plan.included_ids)
        return cls.from_blocks(
            [
                block
                for block in blocks
                if block.block_id not in selected
            ],
            descriptions=descriptions,
        )

    def __len__(self) -> int:
        return len(self._items)

    def catalog(self, *, include_required: bool = False) -> dict[str, Any]:
        """Return a compact catalog with no deferred block values."""

        entries = []
        for item in self._items:
            if item.block.required and not include_required:
                continue
            cost = item.block.token_estimate or item.block.estimate()
            entries.append(item.catalog_entry(token_cost=cost))
        total_tokens = sum(
            int(entry["token_estimate"]) for entry in entries
        )
        return {
            "schema_version": "context-catalog.v1",
            "count": len(entries),
            "estimated_total_tokens": total_tokens,
            "items": entries,
        }

    def load(
        self,
        requested_ids: Sequence[str],
        *,
        token_budget: int | None = None,
        max_items: int | None = None,
    ) -> ContextLoadResult:
        requested = tuple(dict.fromkeys(requested_ids))
        if any(not isinstance(item, str) or not item.strip() for item in requested):
            raise ValueError("requested_ids must contain non-empty strings")
        if token_budget is not None and (
            not isinstance(token_budget, int)
            or isinstance(token_budget, bool)
            or token_budget < 1
        ):
            raise ValueError("token_budget must be a positive integer or None")
        if max_items is not None and (
            not isinstance(max_items, int)
            or isinstance(max_items, bool)
            or max_items < 1
        ):
            raise ValueError("max_items must be a positive integer or None")

        loaded: list[str] = []
        omitted: list[str] = []
        unknown: list[str] = []
        reasons: dict[str, tuple[str, ...]] = {}
        materialized: dict[str, Any] = {}
        selected_tokens = 0
        loaded_blocks: list[ContextBlock] = []

        for block_id in requested:
            item = self._by_id.get(block_id)
            if item is None:
                unknown.append(block_id)
                reasons[block_id] = ("unknown deferred context id",)
                continue

            if max_items is not None and len(loaded) >= max_items:
                omitted.append(block_id)
                reasons[block_id] = ("load item limit reached",)
                continue

            cost = item.block.token_estimate or item.block.estimate()
            if (
                token_budget is not None
                and selected_tokens + cost > token_budget
            ):
                omitted.append(block_id)
                reasons[block_id] = ("load token budget exceeded",)
                continue

            loaded.append(block_id)
            loaded_blocks.append(item.block)
            selected_tokens += cost
            reasons[block_id] = ("explicitly requested",)

        if loaded_blocks:
            selected = {block.block_id for block in loaded_blocks}
            for block in loaded_blocks:
                if block.block_id not in selected:
                    continue
                path = block.path or block.block_id
                parts = path.split(".")
                current = materialized
                for segment in parts[:-1]:
                    child = current.get(segment)
                    if not isinstance(child, dict):
                        child = {}
                        current[segment] = child
                    current = child
                current[parts[-1]] = block.value

        return ContextLoadResult(
            schema_version="context-load.v1",
            requested_ids=requested,
            loaded_ids=tuple(loaded),
            unknown_ids=tuple(unknown),
            omitted_ids=tuple(omitted),
            selected_tokens=selected_tokens,
            token_budget=token_budget,
            materialized=materialized,
            reasons=reasons,
        )


__all__ = [
    "DeferredContextItem",
    "ContextLoadResult",
    "DeferredContextCatalog",
]
