from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .decision import ActionDecision


TRIAGE_STAGES = (
    "admitted",
    "prioritized",
    "action_gated",
    "waiting_human",
    "executed",
    "observed",
    "closed",
)

_ALLOWED_TRANSITIONS = {
    "admitted": {"prioritized"},
    "prioritized": {"action_gated"},
    "action_gated": {"waiting_human", "executed"},
    "waiting_human": {"action_gated"},
    "executed": {"observed"},
    "observed": {"closed"},
    "closed": set(),
}


@dataclass(frozen=True)
class PriorityAssessment:
    """Typed priority state kept separate from actionability."""

    urgency: float
    importance: float
    priority_band: str
    rationale: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("urgency", self.urgency),
            ("importance", self.importance),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if not self.priority_band.strip():
            raise ValueError("priority_band must not be empty")
        if not self.rationale:
            raise ValueError("rationale must contain at least one item")
        if any(not isinstance(item, str) or not item.strip() for item in self.rationale):
            raise ValueError("rationale must contain non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "urgency": float(self.urgency),
            "importance": float(self.importance),
            "priority_band": self.priority_band,
            "rationale": list(self.rationale),
        }


@dataclass(frozen=True)
class TriageState:
    """Immutable, replayable workflow state for an operational decision."""

    schema_version: str
    case_id: str
    decision_id: str
    stage: str
    policy_version: str
    evidence_snapshot_id: str | None = None
    parent_decision_id: str | None = None
    priority: PriorityAssessment | None = None
    action_decision: ActionDecision | None = None
    human_review_reason: str | None = None
    outcome: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "triage-state.v1":
            raise ValueError("schema_version must be triage-state.v1")
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if not self.decision_id.strip():
            raise ValueError("decision_id must not be empty")
        if not self.policy_version.strip():
            raise ValueError("policy_version must not be empty")
        if self.stage not in TRIAGE_STAGES:
            raise ValueError(f"unknown triage stage: {self.stage}")
        if self.stage == "prioritized" and self.priority is None:
            raise ValueError("prioritized state requires priority")
        if self.stage in {"action_gated", "waiting_human", "executed", "observed", "closed"}:
            if self.priority is None:
                raise ValueError(f"{self.stage} state requires priority")
        if self.stage in {"executed", "observed", "closed"}:
            if self.action_decision is None:
                raise ValueError(f"{self.stage} state requires action_decision")
        if self.stage == "waiting_human":
            if not self.human_review_reason or not self.human_review_reason.strip():
                raise ValueError("waiting_human state requires human_review_reason")

    @classmethod
    def admitted(
        cls,
        *,
        case_id: str,
        decision_id: str,
        policy_version: str,
        evidence_snapshot_id: str | None = None,
    ) -> "TriageState":
        return cls(
            schema_version="triage-state.v1",
            case_id=case_id,
            decision_id=decision_id,
            stage="admitted",
            policy_version=policy_version,
            evidence_snapshot_id=evidence_snapshot_id,
        )

    def transition(
        self,
        next_stage: str,
        *,
        priority: PriorityAssessment | None = None,
        action_decision: ActionDecision | None = None,
        human_review_reason: str | None = None,
        outcome: dict[str, Any] | None = None,
        decision_id: str | None = None,
    ) -> "TriageState":
        allowed = _ALLOWED_TRANSITIONS[self.stage]
        if next_stage not in allowed:
            raise ValueError(
                f"invalid triage transition: {self.stage} -> {next_stage}"
            )

        next_priority = priority if priority is not None else self.priority
        next_action = (
            action_decision
            if action_decision is not None
            else self.action_decision
        )

        return TriageState(
            schema_version="triage-state.v1",
            case_id=self.case_id,
            decision_id=decision_id or self.decision_id,
            parent_decision_id=self.decision_id,
            stage=next_stage,
            policy_version=self.policy_version,
            evidence_snapshot_id=self.evidence_snapshot_id,
            priority=next_priority,
            action_decision=next_action,
            human_review_reason=human_review_reason,
            outcome=outcome,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["priority"] = (
            None if self.priority is None else self.priority.to_dict()
        )
        payload["action_decision"] = (
            None
            if self.action_decision is None
            else self.action_decision.to_dict()
        )
        return payload


@dataclass(frozen=True)
class TriageEnvelope:
    """Explicit handoff between priority and actionability."""

    schema_version: str
    priority: PriorityAssessment
    action_decision: ActionDecision | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "triage-envelope.v1":
            raise ValueError("schema_version must be triage-envelope.v1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "priority": self.priority.to_dict(),
            "action_decision": (
                None
                if self.action_decision is None
                else self.action_decision.to_dict()
            ),
        }


__all__ = [
    "TRIAGE_STAGES",
    "PriorityAssessment",
    "TriageEnvelope",
    "TriageState",
]
