from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

from .core import estimate_tokens, serialize_context


EVIDENCE_KINDS = ("observation", "inference", "hypothesis")
EVIDENCE_STANCES = ("supports", "contradicts", "neutral")
_CAUSAL_TAGS = frozenset(
    {"intervention", "multivariable", "confounded", "placebo_absent"}
)


def _normalize_tuple(values: Iterable[Any], *, field: str) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item:
            raise ValueError(f"{field} must not contain empty values")
        if item not in seen:
            result.append(item)
            seen.add(item)
    return tuple(result)


@dataclass(frozen=True)
class EvidenceRecord:
    """Caller-supplied evidence with explicit epistemic and provenance metadata.

    PromptForge does not infer these labels. They are part of the integration
    contract and remain descriptive unless an external evaluator validates them.
    """

    evidence_id: str
    content: str
    kind: str = "observation"
    stance: str = "neutral"
    source: str = "unknown"
    timestamp: float | None = None
    relevance: float = 1.0
    reliability: float = 1.0
    tags: tuple[str, ...] = ()
    intervention_factors: tuple[str, ...] = ()
    confounders: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("evidence_id must not be empty")
        if not self.content.strip():
            raise ValueError("content must not be empty")
        if self.kind not in EVIDENCE_KINDS:
            raise ValueError(f"kind must be one of {EVIDENCE_KINDS}")
        if self.stance not in EVIDENCE_STANCES:
            raise ValueError(f"stance must be one of {EVIDENCE_STANCES}")
        if self.timestamp is not None and (
            not isinstance(self.timestamp, (int, float))
            or isinstance(self.timestamp, bool)
            or not isfinite(float(self.timestamp))
        ):
            raise ValueError("timestamp must be a finite number or None")
        for name, value in (
            ("relevance", self.relevance),
            ("reliability", self.reliability),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        object.__setattr__(self, "tags", _normalize_tuple(self.tags, field="tags"))
        object.__setattr__(
            self,
            "intervention_factors",
            _normalize_tuple(self.intervention_factors, field="intervention_factors"),
        )
        object.__setattr__(
            self,
            "confounders",
            _normalize_tuple(self.confounders, field="confounders"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceRecord":
        allowed = {
            "evidence_id",
            "content",
            "kind",
            "stance",
            "source",
            "timestamp",
            "relevance",
            "reliability",
            "tags",
            "intervention_factors",
            "confounders",
        }
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(
                "unknown evidence fields: " + ", ".join(sorted(unknown))
            )
        return cls(
            evidence_id=str(value["evidence_id"]),
            content=str(value["content"]),
            kind=str(value.get("kind", "observation")),
            stance=str(value.get("stance", "neutral")),
            source=str(value.get("source", "unknown")),
            timestamp=value.get("timestamp"),
            relevance=float(value.get("relevance", 1.0)),
            reliability=float(value.get("reliability", 1.0)),
            tags=tuple(value.get("tags", ())),
            intervention_factors=tuple(value.get("intervention_factors", ())),
            confounders=tuple(value.get("confounders", ())),
        )

    @property
    def temporal_status(self) -> str:
        return "known" if self.timestamp is not None else "unknown"

    @property
    def causal_status(self) -> str:
        has_intervention = bool(self.intervention_factors) or "intervention" in self.tags
        if not has_intervention:
            return "observational"
        if self.confounders or "confounded" in self.tags:
            return "confounded_intervention"
        if (
            len(self.intervention_factors) > 1
            or "multivariable" in self.tags
            or "placebo_absent" in self.tags
        ):
            return "multivariable_intervention"
        return "isolated_candidate"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        payload["intervention_factors"] = list(self.intervention_factors)
        payload["confounders"] = list(self.confounders)
        payload["temporal_status"] = self.temporal_status
        payload["causal_status"] = self.causal_status
        return payload


@dataclass(frozen=True)
class EpistemicContextPolicy:
    """Deterministic evidence-selection policy with a hard temporal boundary."""

    cutoff: float | None = None
    max_records: int | None = None
    budget_tokens: int | None = None
    preserve_contradictions: bool = True
    preserve_each_kind: bool = True
    allow_unknown_time: bool = False
    required_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.cutoff is not None and not isfinite(float(self.cutoff)):
            raise ValueError("cutoff must be finite or None")
        for name, value in (
            ("max_records", self.max_records),
            ("budget_tokens", self.budget_tokens),
        ):
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        object.__setattr__(
            self,
            "required_ids",
            _normalize_tuple(self.required_ids, field="required_ids"),
        )


@dataclass(frozen=True)
class EpistemicContextResult:
    schema_version: str
    context: dict[str, Any]
    serialized_context: str
    included_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    future_excluded_ids: tuple[str, ...]
    unknown_time_ids: tuple[str, ...]
    unknown_time_excluded_ids: tuple[str, ...]
    protected_ids: tuple[str, ...]
    contradiction_ids: tuple[str, ...]
    negative_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]
    inference_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    budget_satisfied: bool
    estimated_tokens: int
    audit: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "included_ids",
            "excluded_ids",
            "future_excluded_ids",
            "unknown_time_ids",
            "unknown_time_excluded_ids",
            "protected_ids",
            "contradiction_ids",
            "negative_ids",
            "hypothesis_ids",
            "inference_ids",
            "observation_ids",
        ):
            payload[key] = list(getattr(self, key))
        return payload


class EpistemicContextCompiler:
    """Compile evidence into a compact, auditable context without semantic invention.

    Selection is deterministic:
    1. enforce the temporal cutoff;
    2. protect required evidence and, when configured, contradiction/kind coverage;
    3. rank remaining evidence by relevance, reliability, recency, and id;
    4. enforce record/token budgets;
    5. retain explicit provenance and exclusion reasons.

    The compiler does not infer whether a claim is true, causal, or high quality.
    """

    def compile(
        self,
        records: Sequence[EvidenceRecord | Mapping[str, Any]],
        *,
        policy: EpistemicContextPolicy | None = None,
    ) -> EpistemicContextResult:
        effective_policy = policy or EpistemicContextPolicy()
        normalized = self._normalize_records(records)
        by_id = {record.evidence_id: record for record in normalized}

        missing_required = [
            evidence_id
            for evidence_id in effective_policy.required_ids
            if evidence_id not in by_id
        ]
        if missing_required:
            raise ValueError(
                "required evidence ids are missing: "
                + ", ".join(missing_required)
            )

        future_excluded: list[str] = []
        unknown_time_excluded: list[str] = []
        eligible: list[EvidenceRecord] = []
        unknown_time: list[str] = []

        for record in normalized:
            if (
                effective_policy.cutoff is not None
                and record.timestamp is not None
                and record.timestamp > effective_policy.cutoff
            ):
                future_excluded.append(record.evidence_id)
                continue
            if record.timestamp is None:
                unknown_time.append(record.evidence_id)
                if (
                    effective_policy.cutoff is not None
                    and not effective_policy.allow_unknown_time
                ):
                    unknown_time_excluded.append(record.evidence_id)
                    continue
            eligible.append(record)

        unsafe_required = sorted(
            set(effective_policy.required_ids).intersection(
                future_excluded + unknown_time_excluded
            )
        )
        if unsafe_required:
            raise ValueError(
                "required evidence crosses the temporal boundary: "
                + ", ".join(unsafe_required)
            )

        required = [
            record
            for record in eligible
            if record.evidence_id in effective_policy.required_ids
        ]
        protected: list[EvidenceRecord] = []
        protected_ids: set[str] = set()

        def protect(record: EvidenceRecord) -> None:
            if record.evidence_id not in protected_ids:
                protected.append(record)
                protected_ids.add(record.evidence_id)

        for record in required:
            protect(record)

        if effective_policy.max_records is not None and (
            len(protected) > effective_policy.max_records
        ):
            raise ValueError(
                "protected evidence exceeds max_records: "
                f"{len(protected)} > {effective_policy.max_records}"
            )

        candidates = [item for item in eligible if item.evidence_id not in protected_ids]

        if effective_policy.preserve_contradictions:
            contradictions = [
                item for item in candidates if item.stance == "contradicts"
            ]
            if contradictions:
                protect(self._best(contradictions))
        if effective_policy.preserve_each_kind:
            for kind in EVIDENCE_KINDS:
                same_kind = [item for item in candidates if item.kind == kind]
                if same_kind:
                    protect(self._best(same_kind))
        if effective_policy.allow_unknown_time:
            unknown = [item for item in candidates if item.timestamp is None]
            if unknown:
                protect(self._best(unknown))

        ranked = sorted(
            (
                item
                for item in eligible
                if item.evidence_id not in protected_ids
            ),
            key=self._sort_key,
        )
        ordered = protected + ranked

        included: list[EvidenceRecord] = []
        excluded: list[tuple[str, str]] = []
        seen: set[str] = set()

        for record in ordered:
            if record.evidence_id in seen:
                continue
            if effective_policy.max_records is not None and (
                len(included) >= effective_policy.max_records
            ):
                excluded.append((record.evidence_id, "record_budget"))
                continue

            trial = included + [record]
            serialized = self._serialize(trial)
            if (
                effective_policy.budget_tokens is not None
                and estimate_tokens(serialized) > effective_policy.budget_tokens
            ):
                if record.evidence_id in protected_ids:
                    raise ValueError(
                        "protected evidence exceeds budget: "
                        f"{record.evidence_id}"
                    )
                excluded.append((record.evidence_id, "token_budget"))
                continue

            included.append(record)
            seen.add(record.evidence_id)

        included_ids = tuple(item.evidence_id for item in included)
        excluded_ids = tuple(
            list(
                item_id
                for item_id, _ in excluded
                if item_id not in future_excluded
            )
            + future_excluded
            + unknown_time_excluded
        )
        contradiction_ids = tuple(
            item.evidence_id
            for item in included
            if item.stance == "contradicts"
        )
        negative_ids = contradiction_ids
        hypothesis_ids = tuple(
            item.evidence_id for item in included if item.kind == "hypothesis"
        )
        inference_ids = tuple(
            item.evidence_id for item in included if item.kind == "inference"
        )
        observation_ids = tuple(
            item.evidence_id for item in included if item.kind == "observation"
        )

        serialized = self._serialize(included)
        estimated_tokens = estimate_tokens(serialized)
        budget_satisfied = (
            effective_policy.budget_tokens is None
            or estimated_tokens <= effective_policy.budget_tokens
        )

        context = {
            "evidence": [item.to_dict() for item in included],
            "epistemic_boundary": {
                "cutoff": effective_policy.cutoff,
                "future_excluded": list(future_excluded),
                "unknown_time": list(unknown_time),
            },
        }

        audit = {
            "input_count": len(normalized),
            "eligible_count": len(eligible),
            "included_count": len(included),
            "excluded_count": (
                len(excluded)
                + len(future_excluded)
                + len(unknown_time_excluded)
            ),
            "excluded_reasons": [
                {"evidence_id": evidence_id, "reason": reason}
                for evidence_id, reason in excluded
            ],
            "future_excluded_count": len(future_excluded),
            "unknown_time_excluded_count": len(unknown_time_excluded),
            "protected_count": len(protected_ids),
            "protected_ids": [
                item.evidence_id
                for item in protected
                if item.evidence_id in included_ids
            ],
            "contradiction_preserved": bool(contradiction_ids)
            if any(item.stance == "contradicts" for item in eligible)
            else True,
            "temporal_boundary_enforced": effective_policy.cutoff is not None,
            "unknown_time_allowed": effective_policy.allow_unknown_time,
            "kind_coverage": {
                kind: any(item.kind == kind for item in included)
                for kind in EVIDENCE_KINDS
                if any(item.kind == kind for item in eligible)
            },
            "causal_cautions": [
                {
                    "evidence_id": item.evidence_id,
                    "causal_status": item.causal_status,
                }
                for item in included
                if item.causal_status != "observational"
            ],
        }

        return EpistemicContextResult(
            schema_version="epistemic-context.v1",
            context=context,
            serialized_context=serialized,
            included_ids=included_ids,
            excluded_ids=excluded_ids,
            future_excluded_ids=tuple(future_excluded),
            unknown_time_ids=tuple(
                evidence_id
                for evidence_id in unknown_time
                if evidence_id in included_ids
            ),
            unknown_time_excluded_ids=tuple(unknown_time_excluded),
            protected_ids=tuple(
                evidence_id
                for evidence_id in protected_ids
                if evidence_id in included_ids
            ),
            contradiction_ids=contradiction_ids,
            negative_ids=negative_ids,
            hypothesis_ids=hypothesis_ids,
            inference_ids=inference_ids,
            observation_ids=observation_ids,
            budget_satisfied=budget_satisfied,
            estimated_tokens=estimated_tokens,
            audit=audit,
        )

    @staticmethod
    def _normalize_records(
        records: Sequence[EvidenceRecord | Mapping[str, Any]],
    ) -> tuple[EvidenceRecord, ...]:
        normalized: list[EvidenceRecord] = []
        seen: set[str] = set()
        for raw in records:
            record = (
                raw
                if isinstance(raw, EvidenceRecord)
                else EvidenceRecord.from_mapping(raw)
            )
            if record.evidence_id in seen:
                raise ValueError(
                    "duplicate evidence_id: " + record.evidence_id
                )
            normalized.append(record)
            seen.add(record.evidence_id)
        return tuple(normalized)

    @staticmethod
    def _sort_key(record: EvidenceRecord) -> tuple[Any, ...]:
        timestamp_key = (
            1 if record.timestamp is not None else 0,
            float(record.timestamp) if record.timestamp is not None else float("-inf"),
        )
        return (
            -float(record.relevance),
            -float(record.reliability),
            -timestamp_key[0],
            -timestamp_key[1],
            record.kind,
            record.stance,
            record.evidence_id,
        )

    @staticmethod
    def _best(records: Sequence[EvidenceRecord]) -> EvidenceRecord:
        return min(records, key=EpistemicContextCompiler._sort_key)

    @staticmethod
    def _serialize(records: Sequence[EvidenceRecord]) -> str:
        return serialize_context(
            {
                "evidence": [item.to_dict() for item in records],
            }
        )


__all__ = [
    "EVIDENCE_KINDS",
    "EVIDENCE_STANCES",
    "EvidenceRecord",
    "EpistemicContextPolicy",
    "EpistemicContextResult",
    "EpistemicContextCompiler",
]
