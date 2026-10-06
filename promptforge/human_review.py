from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


HUMAN_REVIEW_DECISIONS = (
    "allow",
    "conditional",
    "blocked",
    "rejected",
    "reconsidered",
)


@dataclass(frozen=True)
class HumanReviewRecord:
    """Immutable record of one human intervention at a decision boundary.

    This records what the reviewer saw and how the workflow resumed. It does
    not imply that the review established factual truth or causal validity.
    """

    schema_version: str
    review_id: str
    reviewer_id: str
    reviewed_at: str
    evidence_snapshot_id: str
    decision: str
    rationale: tuple[str, ...]
    changes_made: tuple[str, ...]
    resulting_policy_version: str

    def __post_init__(self) -> None:
        for name, value in (
            ("schema_version", self.schema_version),
            ("review_id", self.review_id),
            ("reviewer_id", self.reviewer_id),
            ("reviewed_at", self.reviewed_at),
            ("evidence_snapshot_id", self.evidence_snapshot_id),
            ("resulting_policy_version", self.resulting_policy_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.schema_version != "human-review.v1":
            raise ValueError("schema_version must be human-review.v1")
        if self.decision not in HUMAN_REVIEW_DECISIONS:
            raise ValueError(f"unknown human review decision: {self.decision}")
        if not self.rationale:
            raise ValueError("rationale must contain at least one item")
        if any(
            not isinstance(item, str) or not item.strip()
            for item in self.rationale
        ):
            raise ValueError("rationale must contain non-empty strings")
        if any(
            not isinstance(item, str) or not item.strip()
            for item in self.changes_made
        ):
            raise ValueError("changes_made must contain non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rationale"] = list(self.rationale)
        payload["changes_made"] = list(self.changes_made)
        return payload


__all__ = ["HUMAN_REVIEW_DECISIONS", "HumanReviewRecord"]
