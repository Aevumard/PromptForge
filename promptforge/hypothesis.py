from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from .epistemic import EpistemicContextResult


@dataclass(frozen=True)
class HypothesisEvidencePolicy:
    """Operational guard against source-correlated evidence inflation.

    The source field is caller-supplied provenance metadata. This policy does
    not establish statistical independence; it only caps how many support or
    contradiction records from the same declared source may contribute to a
    hypothesis assessment. Unknown sources can remain individually isolated.
    """

    max_per_source: int | None = 1
    isolate_unknown_source: bool = True

    def __post_init__(self) -> None:
        if self.max_per_source is not None and (
            not isinstance(self.max_per_source, int)
            or isinstance(self.max_per_source, bool)
            or self.max_per_source < 1
        ):
            raise ValueError("max_per_source must be a positive integer or None")
        if not isinstance(self.isolate_unknown_source, bool):
            raise TypeError("isolate_unknown_source must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



HYPOTHESIS_STATUSES = (
    "unresolved",
    "supported_by_available_evidence",
    "challenged_by_available_evidence",
    "contested",
    "insufficient_evidence",
)


@dataclass(frozen=True)
class HypothesisRecord:
    """Caller-supplied hypothesis/evidence mapping.

    Evidence membership is explicit. PromptForge does not infer which evidence
    supports or contradicts a hypothesis.
    """

    hypothesis_id: str
    statement: str
    support_evidence_ids: tuple[str, ...] = ()
    contradiction_evidence_ids: tuple[str, ...] = ()
    required_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id must not be empty")
        if not self.statement.strip():
            raise ValueError("statement must not be empty")
        fields = (
            self.support_evidence_ids,
            self.contradiction_evidence_ids,
            self.required_evidence_ids,
        )
        for values in fields:
            if len(set(values)) != len(values):
                raise ValueError("hypothesis evidence ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["support_evidence_ids"] = list(self.support_evidence_ids)
        payload["contradiction_evidence_ids"] = list(self.contradiction_evidence_ids)
        payload["required_evidence_ids"] = list(self.required_evidence_ids)
        return payload


@dataclass(frozen=True)
class HypothesisAssessment:
    hypothesis_id: str
    status: str
    support_count: int
    contradiction_count: int
    available_support_ids: tuple[str, ...]
    available_contradiction_ids: tuple[str, ...]
    missing_required_ids: tuple[str, ...]
    support_ratio: float | None
    source_excluded_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["available_support_ids"] = list(self.available_support_ids)
        payload["available_contradiction_ids"] = list(
            self.available_contradiction_ids
        )
        payload["missing_required_ids"] = list(self.missing_required_ids)
        payload["source_excluded_ids"] = list(self.source_excluded_ids)
        return payload


@dataclass(frozen=True)
class DiscriminatingExperiment:
    """Caller-specified experiment design attributes.

    The derived score is a prioritization heuristic, not a statistical power
    estimate and not evidence that an experiment will identify a true cause.
    """

    experiment_id: str
    description: str
    target_hypotheses: tuple[str, ...]
    isolation: float
    control_quality: float
    measurement_quality: float
    operational_cost: float = 0.0
    operational_risk: float = 0.0

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        if not self.description.strip():
            raise ValueError("description must not be empty")
        if len(self.target_hypotheses) < 2:
            raise ValueError("target_hypotheses must contain at least two hypotheses")
        if len(set(self.target_hypotheses)) != len(self.target_hypotheses):
            raise ValueError("target_hypotheses must be unique")
        for name, value in (
            ("isolation", self.isolation),
            ("control_quality", self.control_quality),
            ("measurement_quality", self.measurement_quality),
            ("operational_cost", self.operational_cost),
            ("operational_risk", self.operational_risk),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be between 0.0 and 1.0")

    def priority_score(self) -> float:
        separation = min(1.0, len(self.target_hypotheses) / 4.0)
        return (
            0.30 * separation
            + 0.25 * self.isolation
            + 0.20 * self.control_quality
            + 0.20 * self.measurement_quality
            - 0.15 * self.operational_cost
            - 0.10 * self.operational_risk
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_hypotheses"] = list(self.target_hypotheses)
        payload["priority_score"] = self.priority_score()
        return payload


class HypothesisLedger:
    """Maintain explicit support/contradiction state without causal invention."""

    def assess(
        self,
        hypotheses: Sequence[HypothesisRecord],
        *,
        evidence: EpistemicContextResult | None = None,
        evidence_policy: HypothesisEvidencePolicy | None = None,
    ) -> tuple[HypothesisAssessment, ...]:
        if not hypotheses:
            raise ValueError("hypotheses must contain at least one item")

        available = (
            set(evidence.included_ids)
            if evidence is not None
            else set()
        )
        effective_policy = evidence_policy
        source_by_id: dict[str, str] = {}
        ordered_ids = tuple(evidence.included_ids) if evidence is not None else ()
        if evidence is not None:
            raw_records = evidence.context.get("evidence", [])
            for raw in raw_records:
                evidence_id = str(raw.get("evidence_id", "")).strip()
                if evidence_id and evidence_id in available:
                    source = str(raw.get("source", "")).strip()
                    if effective_policy is not None and (
                        effective_policy.isolate_unknown_source
                        and source.lower() in {"", "unknown"}
                    ):
                        source = f"__unknown__:{evidence_id}"
                    source_by_id[evidence_id] = source or "__unknown__"

        def apply_source_cap(
            evidence_ids: Sequence[str],
        ) -> tuple[tuple[str, ...], tuple[str, ...]]:
            if effective_policy is None or effective_policy.max_per_source is None:
                return tuple(evidence_ids), ()
            selected: list[str] = []
            excluded: list[str] = []
            source_counts: dict[str, int] = {}
            for evidence_id in ordered_ids:
                if evidence_id not in evidence_ids:
                    continue
                source = source_by_id.get(
                    evidence_id,
                    f"__unknown__:{evidence_id}",
                )
                count = source_counts.get(source, 0)
                if count >= effective_policy.max_per_source:
                    excluded.append(evidence_id)
                    continue
                selected.append(evidence_id)
                source_counts[source] = count + 1
            return tuple(selected), tuple(excluded)

        seen: set[str] = set()
        results: list[HypothesisAssessment] = []

        for hypothesis in hypotheses:
            if hypothesis.hypothesis_id in seen:
                raise ValueError(
                    "duplicate hypothesis_id: " + hypothesis.hypothesis_id
                )
            seen.add(hypothesis.hypothesis_id)

            required = set(hypothesis.required_evidence_ids)
            missing_required = tuple(sorted(required.difference(available)))

            if evidence is None and (
                hypothesis.support_evidence_ids
                or hypothesis.contradiction_evidence_ids
                or hypothesis.required_evidence_ids
            ):
                raise ValueError(
                    "evidence snapshot is required when hypothesis evidence ids are supplied"
                )

            raw_support_ids = tuple(
                item
                for item in hypothesis.support_evidence_ids
                if item in available
            )
            raw_contradiction_ids = tuple(
                item
                for item in hypothesis.contradiction_evidence_ids
                if item in available
            )
            support_ids, support_excluded = apply_source_cap(raw_support_ids)
            contradiction_ids, contradiction_excluded = apply_source_cap(
                raw_contradiction_ids
            )
            source_excluded_ids = tuple(
                dict.fromkeys((*support_excluded, *contradiction_excluded))
            )

            if missing_required:
                status = "insufficient_evidence"
            elif support_ids and contradiction_ids:
                status = "contested"
            elif support_ids:
                status = "supported_by_available_evidence"
            elif contradiction_ids:
                status = "challenged_by_available_evidence"
            else:
                status = "unresolved"

            total = len(support_ids) + len(contradiction_ids)
            support_ratio = (
                len(support_ids) / total
                if total
                else None
            )

            results.append(
                HypothesisAssessment(
                    hypothesis_id=hypothesis.hypothesis_id,
                    status=status,
                    support_count=len(support_ids),
                    contradiction_count=len(contradiction_ids),
                    available_support_ids=support_ids,
                    available_contradiction_ids=contradiction_ids,
                    missing_required_ids=missing_required,
                    support_ratio=support_ratio,
                    source_excluded_ids=source_excluded_ids,
                )
            )

        return tuple(results)

    @staticmethod
    def rank_experiments(
        experiments: Sequence[DiscriminatingExperiment],
    ) -> tuple[DiscriminatingExperiment, ...]:
        if not experiments:
            raise ValueError("experiments must contain at least one item")
        seen: set[str] = set()
        for experiment in experiments:
            if experiment.experiment_id in seen:
                raise ValueError(
                    "duplicate experiment_id: " + experiment.experiment_id
                )
            seen.add(experiment.experiment_id)

        return tuple(
            sorted(
                experiments,
                key=lambda item: (
                    -item.priority_score(),
                    item.experiment_id,
                ),
            )
        )


__all__ = [
    "HYPOTHESIS_STATUSES",
    "HypothesisEvidencePolicy",
    "HypothesisRecord",
    "HypothesisAssessment",
    "DiscriminatingExperiment",
    "HypothesisLedger",
]
