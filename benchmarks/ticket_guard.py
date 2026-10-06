from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from promptforge import (
    ActionCandidate,
    ActionPolicy,
    EpistemicContextCompiler,
    EvidenceRecord,
    UncertaintyActionGate,
)

from .tickets_v27 import Prediction, TicketCase


@dataclass(frozen=True)
class GuardDecision:
    ticket_id: str
    original_action: str
    guarded_action: str
    blocked: bool
    reasons: tuple[str, ...]
    decision_schema: str

    def to_dict(self) -> dict[str, object]:
        return {
            "ticket_id": self.ticket_id,
            "original_action": self.original_action,
            "guarded_action": self.guarded_action,
            "blocked": self.blocked,
            "reasons": list(self.reasons),
            "decision_schema": self.decision_schema,
        }


def _evidence(case: TicketCase) -> tuple[EvidenceRecord, ...]:
    return tuple(
        EvidenceRecord(
            evidence_id=str(item["evidence_id"]),
            content=f"Ticket evidence from {item['source']}.",
            stance=str(item["stance"]),
            source=str(item["source"]),
            timestamp=float(item["timestamp"]),
            relevance=float(item["relevance"]),
            reliability=float(item["reliability"]),
        )
        for item in case.evidence
    )


def guard_prediction(case: TicketCase, prediction: Prediction) -> GuardDecision:
    """Apply a conservative PromptForge action gate to one model prediction.

    The guard does not improve category/SLA classification. It protects the
    action boundary: all available evidence is made visible to the action gate,
    contradictions are therefore inadmissible support, and only an explicit
    human-review outcome can remain when the action boundary is unsafe.
    """
    records = _evidence(case)
    evidence = EpistemicContextCompiler().compile(records)

    if prediction.action == "human_review":
        return GuardDecision(
            ticket_id=case.ticket_id,
            original_action=prediction.action,
            guarded_action="human_review",
            blocked=False,
            reasons=("prediction already requests human review",),
            decision_schema="uncertainty-action.v6",
        )

    support_ids = tuple(record.evidence_id for record in records)
    candidate = ActionCandidate(
        action_id=prediction.action,
        description="Agent-proposed operational action.",
        evidence_support=1.0,
        reversibility=0.5 if prediction.action in {"refund", "restore_access"} else 1.0,
        downside=0.8 if prediction.action in {"refund", "restore_access"} else 0.2,
        support_evidence_ids=support_ids,
    )
    supporting_ids = tuple(
        record.evidence_id
        for record in records
        if record.stance == "supports"
    )
    fallback = ActionCandidate(
        action_id="human_review",
        description="Human review fallback.",
        evidence_support=0.7,
        reversibility=1.0,
        downside=0.1,
        support_evidence_ids=supporting_ids,
    )
    policy = ActionPolicy(
        require_support_anchors=True,
        require_support_stance=True,
        require_support_quality=True,
    )
    try:
        decision = UncertaintyActionGate(policy).decide(
            [candidate, fallback],
            evidence=evidence,
        )
    except ValueError as exc:
        if str(exc) != "no action candidate passed the decision gate":
            raise
        return GuardDecision(
            ticket_id=case.ticket_id,
            original_action=prediction.action,
            guarded_action="human_review",
            blocked=True,
            reasons=("no action candidate passed the decision gate",),
            decision_schema="uncertainty-action.v6",
        )

    if prediction.action in decision.blocked_action_ids:
        return GuardDecision(
            ticket_id=case.ticket_id,
            original_action=prediction.action,
            guarded_action="human_review",
            blocked=True,
            reasons=decision.reasons[prediction.action],
            decision_schema=decision.schema_version,
        )

    return GuardDecision(
        ticket_id=case.ticket_id,
        original_action=prediction.action,
        guarded_action=prediction.action,
        blocked=False,
        reasons=("action passed PromptForge admissibility gate",),
        decision_schema=decision.schema_version,
    )


def guard_predictions(
    cases: Sequence[TicketCase],
    predictions: Sequence[Prediction],
) -> tuple[Prediction, ...]:
    by_id = {case.ticket_id: case for case in cases}
    guarded: list[Prediction] = []
    for prediction in predictions:
        case = by_id.get(prediction.ticket_id)
        if case is None:
            raise ValueError(f"unknown ticket id: {prediction.ticket_id}")
        decision = guard_prediction(case, prediction)
        guarded.append(
            Prediction(
                ticket_id=prediction.ticket_id,
                category=prediction.category,
                sla=prediction.sla,
                priority=prediction.priority,
                action=decision.guarded_action,
                requires_human=(
                    prediction.requires_human
                    or decision.guarded_action == "human_review"
                ),
                contradiction_detected=(
                    prediction.contradiction_detected
                    or decision.blocked
                ),
            )
        )
    return tuple(guarded)
