from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import ceil
from typing import Any, Iterable, Mapping, Sequence

from .core import estimate_tokens


@dataclass(frozen=True)
class ToolOutputItem:
    """Provider-agnostic representation of one tool result in an agent turn."""

    item_id: str
    tool_name: str
    content: str
    turn_index: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("item_id", self.item_id),
            ("tool_name", self.tool_name),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if (
            not isinstance(self.turn_index, int)
            or isinstance(self.turn_index, bool)
            or self.turn_index < 0
        ):
            raise ValueError("turn_index must be a non-negative integer")


@dataclass(frozen=True)
class ToolOutputTrimPolicy:
    """Deterministic sliding-window policy for shrinking old tool outputs."""

    recent_turns: int = 2
    max_output_chars: int = 2000
    preview_chars: int = 600
    eligible_tools: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.recent_turns, int) or isinstance(
            self.recent_turns, bool
        ) or self.recent_turns < 1:
            raise ValueError("recent_turns must be at least 1")
        if not isinstance(self.max_output_chars, int) or isinstance(
            self.max_output_chars, bool
        ) or self.max_output_chars < 1:
            raise ValueError("max_output_chars must be positive")
        if not isinstance(self.preview_chars, int) or isinstance(
            self.preview_chars, bool
        ) or self.preview_chars < 1:
            raise ValueError("preview_chars must be positive")
        if self.preview_chars >= self.max_output_chars:
            raise ValueError("preview_chars must be smaller than max_output_chars")
        if self.eligible_tools is not None:
            if any(
                not isinstance(name, str) or not name.strip()
                for name in self.eligible_tools
            ):
                raise ValueError("eligible_tools must contain non-empty strings")


@dataclass(frozen=True)
class ToolOutputTrimmed:
    """One trimmed tool result plus replay metadata."""

    item_id: str
    tool_name: str
    turn_index: int
    content: str
    original_chars: int
    retained_chars: int
    chars_saved: int
    original_tokens: int
    retained_tokens: int
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "tool_name": self.tool_name,
            "turn_index": self.turn_index,
            "original_chars": self.original_chars,
            "retained_chars": self.retained_chars,
            "chars_saved": self.chars_saved,
            "original_tokens": self.original_tokens,
            "retained_tokens": self.retained_tokens,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class ToolOutputTrimResult:
    """Immutable audit result for a tool-output trimming pass."""

    schema_version: str
    items: tuple[ToolOutputItem | ToolOutputTrimmed, ...]
    trimmed_ids: tuple[str, ...]
    chars_saved: int
    estimated_tokens_saved: int
    reasons: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if self.schema_version != "tool-output-trim.v1":
            raise ValueError("schema_version must be tool-output-trim.v1")
        if self.chars_saved < 0 or self.estimated_tokens_saved < 0:
            raise ValueError("savings must be non-negative")
        if len(set(self.trimmed_ids)) != len(self.trimmed_ids):
            raise ValueError("trimmed_ids must be unique")

    @property
    def model_input(self) -> list[dict[str, Any]]:
        """Return compact provider-agnostic items for an adapter to serialize."""
        return [
            {
                "item_id": item.item_id,
                "tool_name": item.tool_name,
                "turn_index": item.turn_index,
                "content": item.content,
                "trimmed": isinstance(item, ToolOutputTrimmed),
            }
            for item in self.items
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trimmed_ids": list(self.trimmed_ids),
            "chars_saved": self.chars_saved,
            "estimated_tokens_saved": self.estimated_tokens_saved,
            "reasons": {
                key: list(value) for key, value in self.reasons.items()
            },
            "items": [
                item.to_dict() if isinstance(item, ToolOutputTrimmed)
                else {
                    "item_id": item.item_id,
                    "tool_name": item.tool_name,
                    "turn_index": item.turn_index,
                    "trimmed": False,
                }
                for item in self.items
            ],
        }


class ToolOutputTrimmer:
    """Trim bulky older tool results before they reach the next model call.

    PromptForge does not summarize tool output. It keeps a deterministic head/tail
    preview and records a digest so an integration can fetch or replay the full result.
    """

    def __init__(self, policy: ToolOutputTrimPolicy | None = None) -> None:
        self.policy = policy or ToolOutputTrimPolicy()

    def trim(self, items: Sequence[ToolOutputItem]) -> ToolOutputTrimResult:
        values = tuple(items)
        ids = [item.item_id for item in values]
        if len(set(ids)) != len(ids):
            raise ValueError("item_id values must be unique")

        output: list[ToolOutputItem | ToolOutputTrimmed] = []
        trimmed_ids: list[str] = []
        reasons: dict[str, tuple[str, ...]] = {}
        chars_saved = 0
        token_saved = 0

        for item in values:
            eligible_tool = (
                self.policy.eligible_tools is None
                or item.tool_name in self.policy.eligible_tools
            )
            if item.turn_index < self.policy.recent_turns:
                output.append(item)
                reasons[item.item_id] = ("recent turn preserved",)
                continue
            if not eligible_tool:
                output.append(item)
                reasons[item.item_id] = ("tool not eligible for trimming",)
                continue
            if len(item.content) <= self.policy.max_output_chars:
                output.append(item)
                reasons[item.item_id] = ("output below trim threshold",)
                continue

            trimmed = self._trim_item(item)
            if trimmed is None:
                output.append(item)
                reasons[item.item_id] = ("trim would not reduce content",)
                continue

            output.append(trimmed)
            trimmed_ids.append(item.item_id)
            chars_saved += trimmed.chars_saved
            token_saved += max(
                0,
                trimmed.original_tokens - trimmed.retained_tokens,
            )
            reasons[item.item_id] = (
                "older tool output exceeded trim threshold",
            )

        return ToolOutputTrimResult(
            schema_version="tool-output-trim.v1",
            items=tuple(output),
            trimmed_ids=tuple(trimmed_ids),
            chars_saved=chars_saved,
            estimated_tokens_saved=token_saved,
            reasons=reasons,
        )

    def _trim_item(self, item: ToolOutputItem) -> ToolOutputTrimmed | None:
        original = item.content
        keep = self.policy.preview_chars
        head = ceil(keep / 2)
        tail = keep - head
        if tail > 0:
            preview = (
                original[:head]
                + "\n…[PromptForge tool output omitted "
                + str(len(original) - keep)
                + " chars]…\n"
                + original[-tail:]
            )
        else:
            preview = original[:head]

        if len(preview) >= len(original):
            return None

        return ToolOutputTrimmed(
            item_id=item.item_id,
            tool_name=item.tool_name,
            turn_index=item.turn_index,
            content=preview,
            original_chars=len(original),
            retained_chars=len(preview),
            chars_saved=len(original) - len(preview),
            original_tokens=estimate_tokens(original),
            retained_tokens=estimate_tokens(preview),
            content_sha256=hashlib.sha256(
                original.encode("utf-8")
            ).hexdigest(),
        )


def trim_tool_outputs(
    items: Sequence[ToolOutputItem],
    *,
    recent_turns: int = 2,
    max_output_chars: int = 2000,
    preview_chars: int = 600,
    eligible_tools: Iterable[str] | None = None,
) -> ToolOutputTrimResult:
    """Fast one-call path for model-call preparation."""

    eligible = (
        None
        if eligible_tools is None
        else frozenset(eligible_tools)
    )
    return ToolOutputTrimmer(
        ToolOutputTrimPolicy(
            recent_turns=recent_turns,
            max_output_chars=max_output_chars,
            preview_chars=preview_chars,
            eligible_tools=eligible,
        )
    ).trim(items)


__all__ = [
    "ToolOutputItem",
    "ToolOutputTrimPolicy",
    "ToolOutputTrimmed",
    "ToolOutputTrimResult",
    "ToolOutputTrimmer",
    "trim_tool_outputs",
]
