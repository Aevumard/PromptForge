from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from promptforge import (
    ContextBlock,
    EpistemicContextCompiler,
    EvidenceRecord,
    prepare_agent_input,
)

from .ticket_guard import guard_predictions
from .tickets_v27 import (
    ACTIONS,
    CATEGORIES,
    PRIORITIES,
    SLAS,
    BenchmarkMetrics,
    Prediction,
    TicketCase,
    evaluate_predictions,
    generate_ticket_suite,
    naive_baseline,
)


class AgentAdapter(Protocol):
    """Provider-neutral contract for a model-facing benchmark adapter."""

    def predict(self, model_input: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class CallableAgentAdapter:
    """Wrap a plain Python callable without adding provider dependencies."""

    function: Callable[[Mapping[str, Any]], Mapping[str, Any]]

    def predict(self, model_input: Mapping[str, Any]) -> Mapping[str, Any]:
        result = self.function(model_input)
        if not isinstance(result, Mapping):
            raise TypeError("agent adapter must return a mapping")
        return result


@dataclass(frozen=True)
class ReplayAgentAdapter:
    """Replay externally produced JSON predictions by ticket id."""

    predictions: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ReplayAgentAdapter":
        records: dict[str, Mapping[str, Any]] = {}
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        f"prediction line {line_number} must be a JSON object"
                    )
                ticket_id = str(payload.get("ticket_id", "")).strip()
                if not ticket_id:
                    raise ValueError(
                        f"prediction line {line_number} is missing ticket_id"
                    )
                if ticket_id in records:
                    raise ValueError(
                        f"duplicate prediction for ticket_id {ticket_id}"
                    )
                records[ticket_id] = payload
        return cls(records)

    def predict(self, model_input: Mapping[str, Any]) -> Mapping[str, Any]:
        ticket_id = str(model_input.get("ticket_id", "")).strip()
        if ticket_id not in self.predictions:
            raise KeyError(f"no replay prediction for {ticket_id}")
        return self.predictions[ticket_id]


@dataclass(frozen=True)
class ModelLoopRecord:
    ticket_id: str
    raw_prediction: Prediction | None
    guarded_prediction: Prediction | None
    context_tokens: int
    budget_tokens: int
    tokens_saved: int
    omitted_context_ids: tuple[str, ...]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "raw_prediction": (
                None
                if self.raw_prediction is None
                else self.raw_prediction.to_dict()
            ),
            "guarded_prediction": (
                None
                if self.guarded_prediction is None
                else self.guarded_prediction.to_dict()
            ),
            "context_tokens": self.context_tokens,
            "budget_tokens": self.budget_tokens,
            "tokens_saved": self.tokens_saved,
            "omitted_context_ids": list(self.omitted_context_ids),
            "error": self.error,
        }


@dataclass(frozen=True)
class ModelLoopReport:
    schema_version: str
    raw_metrics: BenchmarkMetrics
    guarded_metrics: BenchmarkMetrics
    records: Sequence[ModelLoopRecord]

    @property
    def total_calls(self) -> int:
        return len(self.records)

    @property
    def failed_calls(self) -> int:
        return sum(record.raw_prediction is None for record in self.records)

    @property
    def total_context_tokens(self) -> int:
        return sum(record.context_tokens for record in self.records)

    @property
    def total_tokens_saved(self) -> int:
        return sum(record.tokens_saved for record in self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "summary": {
                "total_calls": self.total_calls,
                "failed_calls": self.failed_calls,
                "context_tokens": self.total_context_tokens,
                "tokens_saved": self.total_tokens_saved,
            },
            "raw_metrics": self.raw_metrics.to_dict(),
            "guarded_metrics": self.guarded_metrics.to_dict(),
            "records": [record.to_dict() for record in self.records],
        }


def _validate_choice(name: str, value: Any, choices: Sequence[str]) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = value.strip()
    if normalized not in choices:
        raise ValueError(
            f"{name} must be one of {tuple(choices)}, got {normalized!r}"
        )
    return normalized


def _validate_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def parse_prediction(payload: Mapping[str, Any], *, ticket_id: str) -> Prediction:
    returned_id = str(payload.get("ticket_id", ticket_id)).strip()
    if returned_id != ticket_id:
        raise ValueError(
            f"prediction ticket_id mismatch: expected {ticket_id}, got {returned_id}"
        )
    return Prediction(
        ticket_id=ticket_id,
        category=_validate_choice("category", payload.get("category"), CATEGORIES),
        sla=_validate_choice("sla", payload.get("sla"), SLAS),
        priority=_validate_choice("priority", payload.get("priority"), PRIORITIES),
        action=_validate_choice("action", payload.get("action"), ACTIONS),
        requires_human=_validate_bool(
            "requires_human",
            payload.get("requires_human"),
        ),
        contradiction_detected=_validate_bool(
            "contradiction_detected",
            payload.get("contradiction_detected"),
        ),
    )


def _epistemic_summary(case: TicketCase) -> dict[str, Any]:
    records = tuple(
        EvidenceRecord(
            evidence_id=str(item["evidence_id"]),
            content=(
                "Metadata-only evidence record "
                f"{item['evidence_id']} from source {item['source']}."
            ),
            kind=str(item.get("kind", "observation")),
            stance=str(item["stance"]),
            source=str(item["source"]),
            timestamp=float(item["timestamp"]),
            relevance=float(item["relevance"]),
            reliability=float(item["reliability"]),
        )
        for item in case.evidence
    )
    result = EpistemicContextCompiler().compile(records)
    return {
        "schema_version": result.schema_version,
        "included_ids": list(result.included_ids),
        "excluded_ids": list(result.excluded_ids),
        "future_excluded_ids": list(result.future_excluded_ids),
        "unknown_time_excluded_ids": list(result.unknown_time_excluded_ids),
        "contradiction_ids": list(result.contradiction_ids),
        "negative_ids": list(result.negative_ids),
        "hypothesis_ids": list(result.hypothesis_ids),
        "inference_ids": list(result.inference_ids),
        "observation_ids": list(result.observation_ids),
        "compiler_estimated_tokens": result.estimated_tokens,
        "representation": "metadata_only",
        "budget_satisfied": result.budget_satisfied,
        "audit": {
            "input_count": result.audit["input_count"],
            "included_count": result.audit["included_count"],
            "excluded_count": result.audit["excluded_count"],
            "future_excluded_count": result.audit["future_excluded_count"],
            "unknown_time_excluded_count": result.audit[
                "unknown_time_excluded_count"
            ],
            "temporal_boundary_enforced": result.audit[
                "temporal_boundary_enforced"
            ],
        },
    }


def build_model_input(
    case: TicketCase,
    *,
    budget_tokens: int = 500,
    reserve_tokens: int = 50,
    include_epistemic: bool = True,
) -> dict[str, Any]:
    """Construct the provider-neutral model packet through PromptForge."""

    blocks = [
        ContextBlock(
            "ticket",
            {
                "ticket_id": case.ticket_id,
                "customer_id": case.customer_id,
                "text": case.text,
            },
            required=True,
            path="ticket",
        ),
        ContextBlock(
            "evidence",
            list(case.evidence),
            utility=3.0,
            path="evidence",
        ),
        ContextBlock(
            "history",
            list(case.history),
            utility=1.0,
            path="history",
        ),
    ]
    if include_epistemic:
        blocks.insert(
            1,
            ContextBlock(
                "epistemic",
                _epistemic_summary(case),
                required=True,
                utility=4.0,
                path="epistemic",
            ),
        )

    preparation = prepare_agent_input(
        tuple(blocks),
        budget_tokens=budget_tokens,
        reserve_tokens=reserve_tokens,
        descriptions={
            "epistemic": (
                "PromptForge epistemic summary: evidence provenance, "
                "contradiction preservation, exclusions, and selection audit."
            ),
            "evidence": "Current evidence available for the ticket.",
            "history": "Optional historical customer context.",
        },
    )
    packet = preparation.packet.to_dict()
    return {
        "ticket_id": case.ticket_id,
        "customer_id": case.customer_id,
        "promptforge": packet,
        "output_schema": {
            "ticket_id": "string",
            "category": list(CATEGORIES),
            "sla": list(SLAS),
            "priority": list(PRIORITIES),
            "action": list(ACTIONS),
            "requires_human": "boolean",
            "contradiction_detected": "boolean",
        },
    }


def run_model_loop(
    cases: Sequence[TicketCase],
    adapter: AgentAdapter,
    *,
    budget_tokens: int = 500,
    reserve_tokens: int = 50,
    apply_guard: bool = True,
    include_epistemic: bool = True,
) -> ModelLoopReport:
    """Run an external model adapter through PromptForge and score both stages."""

    raw_predictions: list[Prediction] = []
    guarded_predictions: list[Prediction] = []
    records: list[ModelLoopRecord] = []

    for case in cases:
        try:
            model_input = build_model_input(
                case,
                budget_tokens=budget_tokens,
                reserve_tokens=reserve_tokens,
                include_epistemic=include_epistemic,
            )
            packet = model_input["promptforge"]
            payload = adapter.predict(model_input)
            prediction = parse_prediction(payload, ticket_id=case.ticket_id)
            raw_predictions.append(prediction)

            guarded = (
                guard_predictions(cases, (prediction,))[0]
                if apply_guard
                else prediction
            )
            guarded_predictions.append(guarded)
            records.append(
                ModelLoopRecord(
                    ticket_id=case.ticket_id,
                    raw_prediction=prediction,
                    guarded_prediction=guarded,
                    context_tokens=int(packet["context_tokens"]),
                    budget_tokens=int(packet["budget_tokens"]),
                    tokens_saved=int(packet["tokens_saved"]),
                    omitted_context_ids=tuple(
                        packet["omitted_context_ids"]
                    ),
                )
            )
        except Exception as exc:
            records.append(
                ModelLoopRecord(
                    ticket_id=case.ticket_id,
                    raw_prediction=None,
                    guarded_prediction=None,
                    context_tokens=0,
                    budget_tokens=budget_tokens,
                    tokens_saved=0,
                    omitted_context_ids=(),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    raw_metrics = evaluate_predictions(cases, raw_predictions)
    guarded_metrics = evaluate_predictions(cases, guarded_predictions)

    return ModelLoopReport(
        schema_version="promptforge-v27.4-model-loop.v1",
        raw_metrics=raw_metrics,
        guarded_metrics=guarded_metrics,
        records=tuple(records),
    )


def run_replay_benchmark(
    prediction_path: str | Path,
    *,
    count: int = 1200,
    seed: int = 271,
    budget_tokens: int = 500,
    reserve_tokens: int = 50,
    apply_guard: bool = True,
) -> ModelLoopReport:
    cases = generate_ticket_suite(count=count, seed=seed)
    adapter = ReplayAgentAdapter.from_jsonl(prediction_path)
    return run_model_loop(
        cases,
        adapter,
        budget_tokens=budget_tokens,
        reserve_tokens=reserve_tokens,
        apply_guard=apply_guard,
    )


def demo_baseline_report(
    *,
    count: int = 1200,
    seed: int = 271,
) -> ModelLoopReport:
    """Exercise the model-loop contract with the existing naive baseline."""

    cases = generate_ticket_suite(count=count, seed=seed)
    baseline = {
        prediction.ticket_id: prediction.to_dict()
        for prediction in naive_baseline(cases)
    }
    return run_model_loop(
        cases,
        ReplayAgentAdapter(baseline),
    )


def main() -> int:
    report = demo_baseline_report()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
