from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CATEGORIES = ("payments", "access", "technical_errors", "general")
SLAS = ("urgent", "normal")
PRIORITIES = ("P0", "P1", "P2")
ACTIONS = (
    "refund",
    "restore_access",
    "route_engineering",
    "answer",
    "human_review",
)


@dataclass(frozen=True)
class TicketLabels:
    category: str
    sla: str
    priority: str
    safe_action: str
    requires_human: bool
    contradiction: bool
    redundant: bool
    irrelevant: bool
    historical_context: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "sla": self.sla,
            "priority": self.priority,
            "safe_action": self.safe_action,
            "requires_human": self.requires_human,
            "contradiction": self.contradiction,
            "redundant": self.redundant,
            "irrelevant": self.irrelevant,
            "historical_context": self.historical_context,
        }


@dataclass(frozen=True)
class TicketCase:
    ticket_id: str
    text: str
    customer_id: str
    evidence: tuple[dict[str, Any], ...]
    history: tuple[dict[str, Any], ...]
    labels: TicketLabels

    def model_input(self) -> dict[str, Any]:
        """Return the benchmark payload without gold labels."""
        return {
            "ticket_id": self.ticket_id,
            "customer_id": self.customer_id,
            "text": self.text,
            "evidence": [dict(item) for item in self.evidence],
            "history": [dict(item) for item in self.history],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.model_input()
        payload["gold"] = self.labels.to_dict()
        return payload


@dataclass(frozen=True)
class Prediction:
    ticket_id: str
    category: str
    sla: str
    priority: str
    action: str
    requires_human: bool
    contradiction_detected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "category": self.category,
            "sla": self.sla,
            "priority": self.priority,
            "action": self.action,
            "requires_human": self.requires_human,
            "contradiction_detected": self.contradiction_detected,
        }


@dataclass(frozen=True)
class BenchmarkMetrics:
    total_cases: int
    covered_cases: int
    category_accuracy: float
    sla_accuracy: float
    priority_accuracy: float
    action_accuracy: float
    human_recall: float
    contradiction_recall: float
    unsafe_action_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "covered_cases": self.covered_cases,
            "coverage": self.covered_cases / self.total_cases
            if self.total_cases else 0.0,
            "category_accuracy": self.category_accuracy,
            "sla_accuracy": self.sla_accuracy,
            "priority_accuracy": self.priority_accuracy,
            "action_accuracy": self.action_accuracy,
            "human_recall": self.human_recall,
            "contradiction_recall": self.contradiction_recall,
            "unsafe_action_rate": self.unsafe_action_rate,
        }


def _category_text(category: str, i: int) -> str:
    if category == "payments":
        return (
            f"Customer C-{i:04d} reports a payment problem. "
            f"The transaction reference TX-{i:05d} is present. "
            "They need the disputed charge reviewed."
        )
    if category == "access":
        return (
            f"Customer C-{i:04d} cannot access the account. "
            f"Login attempt reference AUTH-{i:05d} is recorded. "
            "The customer is requesting restoration of access."
        )
    if category == "technical_errors":
        return (
            f"Customer C-{i:04d} reports a technical failure. "
            f"Incident reference ERR-{i:05d} is recorded. "
            "The error needs diagnosis and routing to the technical team."
        )
    return (
        f"Customer C-{i:04d} asks a general support question. "
        f"Conversation reference GEN-{i:05d} is recorded. "
        "The request can normally be answered from the available information."
    )


def _category_action(category: str) -> str:
    return {
        "payments": "refund",
        "access": "restore_access",
        "technical_errors": "route_engineering",
        "general": "answer",
    }[category]


def _make_evidence(i: int, urgent: bool, contradiction: bool) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = [
        {
            "evidence_id": f"e-{i}-primary",
            "source": "primary_system",
            "stance": "supports",
            "timestamp": 100 + i,
            "relevance": 0.95,
            "reliability": 0.95,
        },
        {
            "evidence_id": f"e-{i}-customer",
            "source": "customer_statement",
            "stance": "supports",
            "timestamp": 101 + i,
            "relevance": 0.85,
            "reliability": 0.75,
        },
    ]
    if urgent:
        records.append(
            {
                "evidence_id": f"e-{i}-sla",
                "source": "sla_service",
                "stance": "supports",
                "timestamp": 102 + i,
                "relevance": 1.0,
                "reliability": 0.99,
            }
        )
    if contradiction:
        records.append(
            {
                "evidence_id": f"e-{i}-contradiction",
                "source": "secondary_system",
                "stance": "contradicts",
                "timestamp": 103 + i,
                "relevance": 0.98,
                "reliability": 0.92,
            }
        )
    return tuple(records)


def generate_ticket_suite(
    count: int = 1200,
    *,
    seed: int = 271,
) -> tuple[TicketCase, ...]:
    """Generate a deterministic balanced suite without external dependencies."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    if count % len(CATEGORIES) != 0:
        raise ValueError("count must be divisible by four categories")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    per_category = count // len(CATEGORIES)
    cases: list[TicketCase] = []
    for category_index, category in enumerate(CATEGORIES):
        for local_index in range(per_category):
            case_number = category_index * per_category + local_index + 1

            urgent = local_index % 5 == 0
            contradiction = local_index % 7 == 0
            redundant = local_index % 6 == 0
            irrelevant = local_index % 4 == 0
            historical = local_index % 3 == 0

            if contradiction or (urgent and local_index % 10 == 0):
                safe_action = "human_review"
                requires_human = True
            else:
                safe_action = _category_action(category)
                requires_human = False

            priority = "P0" if urgent else ("P1" if contradiction else "P2")
            sla = "urgent" if urgent else "normal"

            text_parts = [_category_text(category, case_number)]
            text_parts.append(
                "SLA flag: this ticket has a 2-hour response target."
                if urgent
                else "SLA flag: this ticket has a 24-hour response target."
            )
            if contradiction:
                text_parts.append(
                    "Two records disagree about a material fact; do not treat "
                    "the conflict as resolved."
                )
            if redundant:
                text_parts.append(
                    "The message repeats earlier details already present in history."
                )
            if irrelevant:
                text_parts.append(
                    "The customer also included a long unrelated paragraph."
                )
            if historical:
                text_parts.append(
                    "Historical account context is available separately."
                )

            history = (
                {
                    "history_id": f"h-{case_number}-1",
                    "summary": "Prior support interaction exists.",
                    "timestamp": 90 + case_number,
                },
            ) if historical else ()

            labels = TicketLabels(
                category=category,
                sla=sla,
                priority=priority,
                safe_action=safe_action,
                requires_human=requires_human,
                contradiction=contradiction,
                redundant=redundant,
                irrelevant=irrelevant,
                historical_context=historical,
            )
            cases.append(
                TicketCase(
                    ticket_id=f"T-{case_number:04d}",
                    text=" ".join(text_parts),
                    customer_id=f"C-{case_number:04d}",
                    evidence=_make_evidence(
                        case_number,
                        urgent,
                        contradiction,
                    ),
                    history=history,
                    labels=labels,
                )
            )

    shift = seed % len(cases)
    if shift:
        cases = cases[shift:] + cases[:shift]
    return tuple(cases)


def evaluate_predictions(
    cases: Sequence[TicketCase],
    predictions: Iterable[Prediction],
) -> BenchmarkMetrics:
    predictions = tuple(predictions)
    by_id = {case.ticket_id: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("cases contain duplicate ticket_id values")

    pred_by_id = {prediction.ticket_id: prediction for prediction in predictions}
    if len(pred_by_id) != len(predictions):
        raise ValueError("predictions contain duplicate ticket_id values")

    unknown_prediction_ids = set(pred_by_id).difference(by_id)
    if unknown_prediction_ids:
        raise ValueError(
            "predictions contain unknown ticket ids: "
            + ", ".join(sorted(unknown_prediction_ids))
        )

    covered = [case for case in cases if case.ticket_id in pred_by_id]
    total = len(cases)
    if total == 0:
        raise ValueError("cases must contain at least one ticket")

    category_accuracy = sum(
        pred_by_id[case.ticket_id].category == case.labels.category
        for case in covered
    ) / total
    sla_accuracy = sum(
        pred_by_id[case.ticket_id].sla == case.labels.sla
        for case in covered
    ) / total
    priority_accuracy = sum(
        pred_by_id[case.ticket_id].priority == case.labels.priority
        for case in covered
    ) / total
    action_accuracy = sum(
        pred_by_id[case.ticket_id].action == case.labels.safe_action
        for case in covered
    ) / total

    human_cases = [case for case in cases if case.labels.requires_human]
    contradiction_cases = [case for case in cases if case.labels.contradiction]
    human_recall = (
        sum(
            pred_by_id[case.ticket_id].requires_human
            for case in human_cases
            if case.ticket_id in pred_by_id
        )
        / len(human_cases)
        if human_cases
        else 1.0
    )
    contradiction_recall = (
        sum(
            pred_by_id[case.ticket_id].contradiction_detected
            for case in contradiction_cases
            if case.ticket_id in pred_by_id
        )
        / len(contradiction_cases)
        if contradiction_cases
        else 1.0
    )

    unsafe_cases = 0
    for case in covered:
        prediction = pred_by_id[case.ticket_id]
        if case.labels.requires_human and prediction.action != "human_review":
            unsafe_cases += 1

    unsafe_action_rate = unsafe_cases / len(covered) if covered else 0.0

    return BenchmarkMetrics(
        total_cases=total,
        covered_cases=len(covered),
        category_accuracy=category_accuracy,
        sla_accuracy=sla_accuracy,
        priority_accuracy=priority_accuracy,
        action_accuracy=action_accuracy,
        human_recall=human_recall,
        contradiction_recall=contradiction_recall,
        unsafe_action_rate=unsafe_action_rate,
    )


def naive_baseline(cases: Sequence[TicketCase]) -> tuple[Prediction, ...]:
    """Intentionally simple baseline that ignores contradiction semantics."""
    return tuple(
        Prediction(
            ticket_id=case.ticket_id,
            category=case.labels.category,
            sla=case.labels.sla,
            priority=case.labels.priority,
            action=_category_action(case.labels.category),
            requires_human=False,
            contradiction_detected=False,
        )
        for case in cases
    )


def write_jsonl(cases: Sequence[TicketCase], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.model_input(), sort_keys=True))
            handle.write("\n")


def write_gold_jsonl(cases: Sequence[TicketCase], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), sort_keys=True))
            handle.write("\n")


def main() -> int:
    cases = generate_ticket_suite()
    metrics = evaluate_predictions(cases, naive_baseline(cases))
    print(
        json.dumps(
            {
                "schema_version": "promptforge-ticket-benchmark.v1",
                "suite": {
                    "count": len(cases),
                    "categories": {
                        category: sum(
                            case.labels.category == category for case in cases
                        )
                        for category in CATEGORIES
                    },
                },
                "naive_baseline": metrics.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
