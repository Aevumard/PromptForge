from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from .epistemic import EpistemicContextResult


@dataclass(frozen=True)
class ActionCandidate:
    """Caller-supplied operational action metadata.

    Scores are descriptive inputs from the integration. PromptForge does not
    infer them from prose and does not treat the resulting ranking as truth.
    """

    action_id: str
    description: str
    evidence_support: float
    reversibility: float
    downside: float
    operational_cost: float = 0.0
    required_evidence_ids: tuple[str, ...] = ()
    support_evidence_ids: tuple[str, ...] = ()
    depends_on_causal_claim: bool = False

    def __post_init__(self) -> None:
        if not self.action_id.strip():
            raise ValueError("action_id must not be empty")
        if not self.description.strip():
            raise ValueError("description must not be empty")
        for name, value in (
            ("evidence_support", self.evidence_support),
            ("reversibility", self.reversibility),
            ("downside", self.downside),
            ("operational_cost", self.operational_cost),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if not isinstance(self.require_support_anchors, bool):
            raise TypeError("require_support_anchors must be a bool")
        if (
            not isinstance(self.min_support_anchors, int)
            or isinstance(self.min_support_anchors, bool)
            or self.min_support_anchors < 1
        ):
            raise ValueError("min_support_anchors must be at least 1")
        if len(set(self.required_evidence_ids)) != len(self.required_evidence_ids):
            raise ValueError("required_evidence_ids must be unique")
        if len(set(self.support_evidence_ids)) != len(self.support_evidence_ids):
            raise ValueError("support_evidence_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_evidence_ids"] = list(self.required_evidence_ids)
        payload["support_evidence_ids"] = list(self.support_evidence_ids)
        return payload


@dataclass(frozen=True)
class ActionPolicy:
    """Conservative deterministic policy for action ordering."""

    evidence_weight: float = 0.40
    reversibility_weight: float = 0.30
    downside_weight: float = 0.20
    cost_weight: float = 0.10
    min_evidence_support: float = 0.40
    max_downside: float = 0.80
    require_reversible: bool = False
    causal_penalty: float = 0.10
    require_support_anchors: bool = False
    min_support_anchors: int = 1

    def __post_init__(self) -> None:
        weights = (
            self.evidence_weight,
            self.reversibility_weight,
            self.downside_weight,
            self.cost_weight,
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(float(value))
            or value < 0.0
            for value in weights
        ):
            raise ValueError("decision weights must be finite non-negative numbers")
        if sum(weights) <= 0.0:
            raise ValueError("at least one decision weight must be positive")
        for name, value in (
            ("min_evidence_support", self.min_evidence_support),
            ("max_downside", self.max_downside),
            ("causal_penalty", self.causal_penalty),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be between 0.0 and 1.0")


@dataclass(frozen=True)
class ActionDecision:
    schema_version: str
    selected_action_id: str
    ranked_action_ids: tuple[str, ...]
    blocked_action_ids: tuple[str, ...]
    scores: dict[str, float]
    reasons: dict[str, tuple[str, ...]]
    evidence_ids_available: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ranked_action_ids"] = list(self.ranked_action_ids)
        payload["blocked_action_ids"] = list(self.blocked_action_ids)
        payload["evidence_ids_available"] = list(self.evidence_ids_available)
        payload["reasons"] = {
            key: list(value) for key, value in self.reasons.items()
        }
        return payload


class UncertaintyActionGate:
    """Rank operational actions without pretending to resolve causal uncertainty.

    The gate uses caller-supplied action attributes plus the explicit evidence
    boundary. It fails closed when a candidate requires unavailable evidence or
    violates configured safety thresholds. Causal dependence receives a small
    deterministic penalty, but is never interpreted as proof or disproof.
    """

    def __init__(self, policy: ActionPolicy | None = None) -> None:
        self.policy = policy or ActionPolicy()

    def decide(
        self,
        actions: Sequence[ActionCandidate],
        *,
        evidence: EpistemicContextResult | None = None,
    ) -> ActionDecision:
        if not actions:
            raise ValueError("actions must contain at least one candidate")

        available = (
            set(evidence.included_ids)
            if evidence is not None
            else set()
        )
        seen: set[str] = set()
        scored: list[tuple[float, ActionCandidate]] = []
        blocked: list[str] = []
        reasons: dict[str, tuple[str, ...]] = {}

        for action in actions:
            if action.action_id in seen:
                raise ValueError(
                    "duplicate action_id: " + action.action_id
                )
            seen.add(action.action_id)

            why: list[str] = []
            required = set(action.required_evidence_ids)
            missing = sorted(required.difference(available))
            support_ids = tuple(action.support_evidence_ids)
            missing_support = sorted(set(support_ids).difference(available))

            if evidence is not None and missing:
                blocked.append(action.action_id)
                reasons[action.action_id] = (
                    "required evidence unavailable",
                    "missing_evidence:" + ",".join(missing),
                )
                continue

            if evidence is None and action.required_evidence_ids:
                blocked.append(action.action_id)
                reasons[action.action_id] = (
                    "required evidence boundary was not supplied",
                )
                continue

            if self.policy.require_support_anchors:
                if not support_ids:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "evidence support is unanchored",
                        "support_evidence_ids required",
                    )
                    continue
                if evidence is None:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "evidence support boundary was not supplied",
                    )
                    continue
                if missing_support:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "support evidence unavailable",
                        "missing_support_evidence:" + ",".join(missing_support),
                    )
                    continue
                if len(support_ids) < self.policy.min_support_anchors:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "insufficient support evidence anchors",
                        "support_anchor_count:" + str(len(support_ids)),
                    )
                    continue

            if action.evidence_support < self.policy.min_evidence_support:
                blocked.append(action.action_id)
                reasons[action.action_id] = ("evidence support below threshold",)
                continue

            if action.downside > self.policy.max_downside:
                blocked.append(action.action_id)
                reasons[action.action_id] = ("downside above threshold",)
                continue

            if self.policy.require_reversible and action.reversibility <= 0.0:
                blocked.append(action.action_id)
                reasons[action.action_id] = ("action is not reversible",)
                continue

            causal_penalty = (
                self.policy.causal_penalty
                if action.depends_on_causal_claim
                else 0.0
            )
            if action.depends_on_causal_claim:
                why.append("causal dependence receives a caution penalty")

            score = (
                self.policy.evidence_weight * action.evidence_support
                + self.policy.reversibility_weight * action.reversibility
                + self.policy.downside_weight * (1.0 - action.downside)
                + self.policy.cost_weight * (1.0 - action.operational_cost)
                - causal_penalty
            )

            if action.reversibility >= 0.75:
                why.append("high reversibility")
            if action.downside <= 0.25:
                why.append("low downside")
            reasons[action.action_id] = tuple(why)
            scored.append((score, action))

        if not scored:
            raise ValueError("no action candidate passed the decision gate")

        scored.sort(key=lambda item: (-item[0], item[1].action_id))
        scores = {action.action_id: float(score) for score, action in scored}
        ranked = tuple(action.action_id for _, action in scored)

        return ActionDecision(
            schema_version="uncertainty-action.v1",
            selected_action_id=ranked[0],
            ranked_action_ids=ranked,
            blocked_action_ids=tuple(blocked),
            scores=scores,
            reasons=reasons,
            evidence_ids_available=tuple(sorted(available)),
        )


__all__ = [
    "ActionCandidate",
    "ActionDecision",
    "ActionPolicy",
    "UncertaintyActionGate",
]
