from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from .epistemic import EVIDENCE_STANCES, EpistemicContextResult


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
    support_evidence_tags: tuple[str, ...] = ()
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
        if len(set(self.required_evidence_ids)) != len(self.required_evidence_ids):
            raise ValueError("required_evidence_ids must be unique")
        if len(set(self.support_evidence_ids)) != len(self.support_evidence_ids):
            raise ValueError("support_evidence_ids must be unique")
        normalized_tags = tuple(tag.strip() for tag in self.support_evidence_tags)
        if any(not tag for tag in normalized_tags):
            raise ValueError("support_evidence_tags must contain non-empty strings")
        if len(set(normalized_tags)) != len(normalized_tags):
            raise ValueError("support_evidence_tags must be unique")
        object.__setattr__(self, "support_evidence_tags", normalized_tags)

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
    require_support_stance: bool = False
    allowed_support_stances: tuple[str, ...] = ("supports",)
    require_support_tag_match: bool = False
    require_support_quality: bool = False
    min_support_relevance: float = 0.50
    min_support_reliability: float = 0.50
    require_support_provenance_diversity: bool = False
    min_distinct_support_sources: int = 2
    max_support_anchors_per_source: int | None = None

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
        if not isinstance(self.require_support_anchors, bool):
            raise TypeError("require_support_anchors must be a bool")
        if (
            not isinstance(self.min_support_anchors, int)
            or isinstance(self.min_support_anchors, bool)
            or self.min_support_anchors < 1
        ):
            raise ValueError("min_support_anchors must be at least 1")
        if not isinstance(self.require_support_stance, bool):
            raise TypeError("require_support_stance must be a bool")
        if not self.allowed_support_stances:
            raise ValueError("allowed_support_stances must not be empty")
        if any(
            stance not in EVIDENCE_STANCES
            for stance in self.allowed_support_stances
        ):
            raise ValueError(
                "allowed_support_stances must contain only known evidence stances"
            )
        if len(set(self.allowed_support_stances)) != len(self.allowed_support_stances):
            raise ValueError("allowed_support_stances must be unique")
        if not isinstance(self.require_support_tag_match, bool):
            raise TypeError("require_support_tag_match must be a bool")
        if not isinstance(self.require_support_quality, bool):
            raise TypeError("require_support_quality must be a bool")
        if not isinstance(self.require_support_provenance_diversity, bool):
            raise TypeError("require_support_provenance_diversity must be a bool")
        if (
            not isinstance(self.min_distinct_support_sources, int)
            or isinstance(self.min_distinct_support_sources, bool)
            or self.min_distinct_support_sources < 2
        ):
            raise ValueError("min_distinct_support_sources must be at least 2")
        if (
            self.max_support_anchors_per_source is not None
            and (
                not isinstance(self.max_support_anchors_per_source, int)
                or isinstance(self.max_support_anchors_per_source, bool)
                or self.max_support_anchors_per_source < 1
            )
        ):
            raise ValueError(
                "max_support_anchors_per_source must be at least 1 or None"
            )
        for name, value in (
            ("min_support_relevance", self.min_support_relevance),
            ("min_support_reliability", self.min_support_reliability),
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
    support_evidence_ids: dict[str, tuple[str, ...]]
    support_evidence_stances: dict[str, dict[str, str]]
    support_evidence_tag_matches: dict[str, dict[str, tuple[str, ...]]]
    support_evidence_quality: dict[str, dict[str, dict[str, float]]]
    support_evidence_provenance: dict[str, dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ranked_action_ids"] = list(self.ranked_action_ids)
        payload["blocked_action_ids"] = list(self.blocked_action_ids)
        payload["evidence_ids_available"] = list(self.evidence_ids_available)
        payload["support_evidence_ids"] = {
            key: list(value) for key, value in self.support_evidence_ids.items()
        }
        payload["support_evidence_stances"] = {
            key: dict(value) for key, value in self.support_evidence_stances.items()
        }
        payload["support_evidence_tag_matches"] = {
            key: {evidence_id: list(tags) for evidence_id, tags in value.items()}
            for key, value in self.support_evidence_tag_matches.items()
        }
        payload["support_evidence_quality"] = {
            key: {
                evidence_id: dict(metrics)
                for evidence_id, metrics in value.items()
            }
            for key, value in self.support_evidence_quality.items()
        }
        payload["support_evidence_provenance"] = {
            key: dict(value)
            for key, value in self.support_evidence_provenance.items()
        }
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
        support_evidence_by_action: dict[str, tuple[str, ...]] = {}
        support_evidence_stances: dict[str, dict[str, str]] = {}
        support_evidence_tag_matches: dict[str, dict[str, tuple[str, ...]]] = {}
        support_evidence_quality: dict[str, dict[str, dict[str, float]]] = {}
        support_evidence_provenance: dict[str, dict[str, str]] = {}

        snapshot_items = (
            evidence.context.get("evidence", [])
            if evidence is not None
            else ()
        )
        stance_by_id = {
            str(item.get("evidence_id", "")).strip(): str(item.get("stance", "")).strip()
            for item in snapshot_items
            if str(item.get("evidence_id", "")).strip()
        }
        tags_by_id = {
            str(item.get("evidence_id", "")).strip(): tuple(
                str(tag).strip()
                for tag in item.get("tags", ())
                if str(tag).strip()
            )
            for item in snapshot_items
            if str(item.get("evidence_id", "")).strip()
        }
        support_source_by_id = {
            str(item.get("evidence_id", "")).strip(): str(item["source"]).strip()
            for item in snapshot_items
            if str(item.get("evidence_id", "")).strip() and "source" in item
        }
        support_relevance_by_id = {
            str(item.get("evidence_id", "")).strip(): float(item["relevance"])
            for item in snapshot_items
            if str(item.get("evidence_id", "")).strip() and "relevance" in item
        }
        support_reliability_by_id = {
            str(item.get("evidence_id", "")).strip(): float(item["reliability"])
            for item in snapshot_items
            if str(item.get("evidence_id", "")).strip() and "reliability" in item
        }

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
            support_evidence_by_action[action.action_id] = support_ids
            missing_support = sorted(set(support_ids).difference(available))
            support_evidence_stances[action.action_id] = {
                evidence_id: stance_by_id[evidence_id]
                for evidence_id in support_ids
                if evidence_id in stance_by_id
            }
            expected_support_tags = action.support_evidence_tags
            support_evidence_tag_matches[action.action_id] = {
                evidence_id: tuple(
                    tag
                    for tag in tags_by_id.get(evidence_id, ())
                    if tag in expected_support_tags
                )
                for evidence_id in support_ids
            }
            support_evidence_quality[action.action_id] = {
                evidence_id: {
                    "relevance": support_relevance_by_id[evidence_id],
                    "reliability": support_reliability_by_id[evidence_id],
                }
                for evidence_id in support_ids
                if evidence_id in support_relevance_by_id
                and evidence_id in support_reliability_by_id
            }

            support_evidence_provenance[action.action_id] = {
                evidence_id: support_source_by_id[evidence_id]
                for evidence_id in support_ids
                if evidence_id in support_source_by_id
            }

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

            if self.policy.require_support_stance:
                if not support_ids:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "support stance cannot be checked without support anchors",
                    )
                    continue
                if evidence is None:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "support stance boundary was not supplied",
                    )
                    continue
                missing_stance = sorted(
                    evidence_id for evidence_id in support_ids
                    if evidence_id not in stance_by_id
                )
                if missing_stance:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "support stance unavailable",
                        "missing_support_stance:" + ",".join(missing_stance),
                    )
                    continue
                incompatible = sorted(
                    evidence_id for evidence_id in support_ids
                    if stance_by_id.get(evidence_id)
                    not in self.policy.allowed_support_stances
                )
                if incompatible:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "support evidence stance incompatible",
                        "incompatible_support_stance:" + ",".join(
                            f"{evidence_id}={stance_by_id[evidence_id]}"
                            for evidence_id in incompatible
                        ),
                    )
                    continue

            if self.policy.require_support_tag_match:
                if not support_ids:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "support relevance cannot be checked without support anchors",
                    )
                    continue
                if evidence is None:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "support relevance boundary was not supplied",
                    )
                    continue
                if not expected_support_tags:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "support relevance scope was not declared",
                        "support_evidence_tags required",
                    )
                    continue
                incompatible_relevance = sorted(
                    evidence_id
                    for evidence_id in support_ids
                    if not support_evidence_tag_matches[action.action_id].get(
                        evidence_id
                    )
                )
                if incompatible_relevance:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "support evidence relevance mismatch",
                        "missing_support_tag_match:" + ",".join(
                            incompatible_relevance
                        ),
                    )
                    continue

            if self.policy.require_support_quality:
                if not support_ids:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "support quality cannot be checked without support anchors",
                    )
                    continue
                if evidence is None:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "support quality boundary was not supplied",
                    )
                    continue
                missing_quality = sorted(
                    evidence_id
                    for evidence_id in support_ids
                    if evidence_id not in support_evidence_quality[action.action_id]
                )
                if missing_quality:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "support evidence quality metadata unavailable",
                        "missing_support_quality:" + ",".join(missing_quality),
                    )
                    continue
                low_relevance = sorted(
                    evidence_id
                    for evidence_id in support_ids
                    if support_relevance_by_id[evidence_id]
                    < self.policy.min_support_relevance
                )
                low_reliability = sorted(
                    evidence_id
                    for evidence_id in support_ids
                    if support_reliability_by_id[evidence_id]
                    < self.policy.min_support_reliability
                )
                if low_relevance or low_reliability:
                    details: list[str] = []
                    if low_relevance:
                        details.append(
                            "low_support_relevance:"
                            + ",".join(
                                f"{evidence_id}={support_relevance_by_id[evidence_id]:.6g}"
                                for evidence_id in low_relevance
                            )
                        )
                    if low_reliability:
                        details.append(
                            "low_support_reliability:"
                            + ",".join(
                                f"{evidence_id}={support_reliability_by_id[evidence_id]:.6g}"
                                for evidence_id in low_reliability
                            )
                        )
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "support evidence quality below threshold",
                        *details,
                    )
                    continue

            if self.policy.require_support_provenance_diversity:
                if not support_ids:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "support provenance diversity cannot be checked without support anchors",
                    )
                    continue
                if evidence is None:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "support provenance diversity boundary was not supplied",
                    )
                    continue
                missing_provenance = sorted(
                    evidence_id
                    for evidence_id in support_ids
                    if evidence_id not in support_source_by_id
                    or not support_source_by_id[evidence_id]
                )
                if missing_provenance:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "support evidence provenance metadata unavailable",
                        "missing_support_provenance:" + ",".join(
                            missing_provenance
                        ),
                    )
                    continue
                distinct_sources = sorted(
                    {
                        support_source_by_id[evidence_id]
                        for evidence_id in support_ids
                        if support_source_by_id[evidence_id].casefold() != "unknown"
                    }
                )
                if len(distinct_sources) < self.policy.min_distinct_support_sources:
                    blocked.append(action.action_id)
                    reasons[action.action_id] = (
                        "insufficient distinct support sources",
                        "distinct_support_source_count:" + str(len(distinct_sources)),
                    )
                    continue
                if self.policy.max_support_anchors_per_source is not None:
                    source_counts: dict[str, int] = {}
                    for evidence_id in support_ids:
                        source = support_source_by_id[evidence_id]
                        source_counts[source] = source_counts.get(source, 0) + 1
                    over_limit = sorted(
                        (source, count)
                        for source, count in source_counts.items()
                        if count > self.policy.max_support_anchors_per_source
                    )
                    if over_limit:
                        blocked.append(action.action_id)
                        reasons[action.action_id] = (
                            "support source concentration exceeds threshold",
                            "support_source_anchor_count:"
                            + ",".join(
                                f"{source}={count}"
                                for source, count in over_limit
                            ),
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
            schema_version="uncertainty-action.v5",
            selected_action_id=ranked[0],
            ranked_action_ids=ranked,
            blocked_action_ids=tuple(blocked),
            scores=scores,
            reasons=reasons,
            evidence_ids_available=tuple(sorted(available)),
            support_evidence_ids=support_evidence_by_action,
            support_evidence_stances=support_evidence_stances,
            support_evidence_tag_matches=support_evidence_tag_matches,
            support_evidence_quality=support_evidence_quality,
        )


__all__ = [
    "ActionCandidate",
    "ActionDecision",
    "ActionPolicy",
    "UncertaintyActionGate",
]
