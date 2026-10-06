from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from benchmarks.model_loop_v27 import parse_prediction
from benchmarks.tickets_v27 import (
    CATEGORIES,
    SLAS,
    TicketCase,
    generate_ticket_suite,
)


@dataclass(frozen=True)
class SliceMetrics:
    name: str
    value: str
    total_cases: int
    raw_covered: int
    guarded_covered: int
    raw_category_accuracy: float
    guarded_category_accuracy: float
    raw_sla_accuracy: float
    guarded_sla_accuracy: float
    raw_priority_accuracy: float
    guarded_priority_accuracy: float
    raw_action_accuracy: float
    guarded_action_accuracy: float
    raw_human_recall: float
    guarded_human_recall: float
    raw_contradiction_recall: float
    guarded_contradiction_recall: float
    raw_unsafe_action_rate: float
    guarded_unsafe_action_rate: float
    guard_action_change_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice": {"name": self.name, "value": self.value},
            "total_cases": self.total_cases,
            "raw_covered": self.raw_covered,
            "guarded_covered": self.guarded_covered,
            "raw_category_accuracy": self.raw_category_accuracy,
            "guarded_category_accuracy": self.guarded_category_accuracy,
            "delta_category_accuracy": self.guarded_category_accuracy
            - self.raw_category_accuracy,
            "raw_sla_accuracy": self.raw_sla_accuracy,
            "guarded_sla_accuracy": self.guarded_sla_accuracy,
            "raw_priority_accuracy": self.raw_priority_accuracy,
            "guarded_priority_accuracy": self.guarded_priority_accuracy,
            "raw_action_accuracy": self.raw_action_accuracy,
            "guarded_action_accuracy": self.guarded_action_accuracy,
            "delta_action_accuracy": self.guarded_action_accuracy
            - self.raw_action_accuracy,
            "raw_human_recall": self.raw_human_recall,
            "guarded_human_recall": self.guarded_human_recall,
            "delta_human_recall": self.guarded_human_recall
            - self.raw_human_recall,
            "raw_contradiction_recall": self.raw_contradiction_recall,
            "guarded_contradiction_recall": self.guarded_contradiction_recall,
            "delta_contradiction_recall": self.guarded_contradiction_recall
            - self.raw_contradiction_recall,
            "raw_unsafe_action_rate": self.raw_unsafe_action_rate,
            "guarded_unsafe_action_rate": self.guarded_unsafe_action_rate,
            "unsafe_rate_reduction": self.raw_unsafe_action_rate
            - self.guarded_unsafe_action_rate,
            "guard_action_change_rate": self.guard_action_change_rate,
        }


@dataclass(frozen=True)
class ExperimentAnalysis:
    schema_version: str
    total_cases: int
    failed_calls: int
    raw_covered: int
    guarded_covered: int
    guard_action_changes: int
    total_context_tokens: int
    total_tokens_saved: int
    total_elapsed_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    provider_elapsed_ms: float
    provider_latency_p50_ms: float
    provider_latency_p95_ms: float
    provider_attempts: int
    provider_prompt_tokens: int
    provider_completion_tokens: int
    provider_total_tokens: int
    slices: Sequence[SliceMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "summary": {
                "total_cases": self.total_cases,
                "failed_calls": self.failed_calls,
                "raw_covered": self.raw_covered,
                "guarded_covered": self.guarded_covered,
                "coverage": (
                    self.raw_covered / self.total_cases
                    if self.total_cases
                    else 0.0
                ),
                "guard_action_changes": self.guard_action_changes,
                "guard_action_change_rate": (
                    self.guard_action_changes / self.raw_covered
                    if self.raw_covered
                    else 0.0
                ),
                "total_context_tokens": self.total_context_tokens,
                "total_tokens_saved": self.total_tokens_saved,
                "total_elapsed_ms": self.total_elapsed_ms,
                "latency_p50_ms": self.latency_p50_ms,
                "latency_p95_ms": self.latency_p95_ms,
                "provider_elapsed_ms": self.provider_elapsed_ms,
                "provider_latency_p50_ms": self.provider_latency_p50_ms,
                "provider_latency_p95_ms": self.provider_latency_p95_ms,
                "provider_attempts": self.provider_attempts,
                "provider_prompt_tokens": self.provider_prompt_tokens,
                "provider_completion_tokens": self.provider_completion_tokens,
                "provider_total_tokens": self.provider_total_tokens,
            },
            "slices": [item.to_dict() for item in self.slices],
        }


def _mean(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(percentile * len(ordered) + 0.4999))))
    return ordered[rank - 1]


def _metrics(
    cases: Sequence[TicketCase],
    raw: Mapping[str, Any],
    guarded: Mapping[str, Any],
    *,
    name: str,
    value: str,
) -> SliceMetrics:
    covered_raw = [
        case for case in cases if case.ticket_id in raw
    ]
    covered_guarded = [
        case for case in cases if case.ticket_id in guarded
    ]

    def accuracy(
        field: str,
        source: Mapping[str, Any],
        selected: Sequence[TicketCase],
        label_getter: Callable[[TicketCase], Any],
    ) -> float:
        return _mean(
            sum(
                source[case.ticket_id][field] == label_getter(case)
                for case in selected
            ),
            len(selected),
        )

    def human_recall(
        source: Mapping[str, Any],
        selected: Sequence[TicketCase],
    ) -> float:
        target = [case for case in selected if case.labels.requires_human]
        return _mean(
            sum(
                source[case.ticket_id]["requires_human"]
                for case in target
            ),
            len(target),
        )

    def contradiction_recall(
        source: Mapping[str, Any],
        selected: Sequence[TicketCase],
    ) -> float:
        target = [case for case in selected if case.labels.contradiction]
        return _mean(
            sum(
                source[case.ticket_id]["contradiction_detected"]
                for case in target
            ),
            len(target),
        )

    def unsafe_rate(
        source: Mapping[str, Any],
        selected: Sequence[TicketCase],
    ) -> float:
        if not selected:
            return 0.0
        unsafe = sum(
            case.labels.requires_human
            and source[case.ticket_id]["action"] != "human_review"
            for case in selected
        )
        return unsafe / len(selected)

    raw_change_base = [
        case
        for case in cases
        if case.ticket_id in raw and case.ticket_id in guarded
    ]
    action_changes = sum(
        raw[case.ticket_id]["action"]
        != guarded[case.ticket_id]["action"]
        for case in raw_change_base
    )

    return SliceMetrics(
        name=name,
        value=value,
        total_cases=len(cases),
        raw_covered=len(covered_raw),
        guarded_covered=len(covered_guarded),
        raw_category_accuracy=accuracy(
            "category",
            raw,
            covered_raw,
            lambda case: case.labels.category,
        ),
        guarded_category_accuracy=accuracy(
            "category",
            guarded,
            covered_guarded,
            lambda case: case.labels.category,
        ),
        raw_sla_accuracy=accuracy(
            "sla",
            raw,
            covered_raw,
            lambda case: case.labels.sla,
        ),
        guarded_sla_accuracy=accuracy(
            "sla",
            guarded,
            covered_guarded,
            lambda case: case.labels.sla,
        ),
        raw_priority_accuracy=accuracy(
            "priority",
            raw,
            covered_raw,
            lambda case: case.labels.priority,
        ),
        guarded_priority_accuracy=accuracy(
            "priority",
            guarded,
            covered_guarded,
            lambda case: case.labels.priority,
        ),
        raw_action_accuracy=accuracy(
            "action",
            raw,
            covered_raw,
            lambda case: case.labels.safe_action,
        ),
        guarded_action_accuracy=accuracy(
            "action",
            guarded,
            covered_guarded,
            lambda case: case.labels.safe_action,
        ),
        raw_human_recall=human_recall(raw, covered_raw),
        guarded_human_recall=human_recall(guarded, covered_guarded),
        raw_contradiction_recall=contradiction_recall(
            raw,
            covered_raw,
        ),
        guarded_contradiction_recall=contradiction_recall(
            guarded,
            covered_guarded,
        ),
        raw_unsafe_action_rate=unsafe_rate(raw, covered_raw),
        guarded_unsafe_action_rate=unsafe_rate(
            guarded,
            covered_guarded,
        ),
        guard_action_change_rate=(
            action_changes / len(raw_change_base)
            if raw_change_base
            else 0.0
        ),
    )


def _prediction_maps(report: Mapping[str, Any]) -> tuple[
    dict[str, Any],
    dict[str, Any],
    int,
    int,
    float,
    list[float],
    float,
    list[float],
    int,
    int,
    int,
    int,
    int,
    int,
]:
    raw: dict[str, Any] = {}
    guarded: dict[str, Any] = {}
    context_tokens = 0
    tokens_saved = 0
    total_elapsed_ms = 0.0
    latencies: list[float] = []
    provider_elapsed_ms = 0.0
    provider_latencies: list[float] = []
    provider_attempts = 0
    provider_prompt_tokens = 0
    provider_completion_tokens = 0
    provider_total_tokens = 0
    failed = 0
    changes = 0

    for item in report.get("records", []):
        ticket_id = str(item["ticket_id"])
        raw_payload = item.get("raw_prediction")
        guarded_payload = item.get("guarded_prediction")

        if isinstance(raw_payload, Mapping):
            raw[ticket_id] = parse_prediction(
                raw_payload,
                ticket_id=ticket_id,
            ).to_dict()
        else:
            failed += 1

        if isinstance(guarded_payload, Mapping):
            guarded[ticket_id] = parse_prediction(
                guarded_payload,
                ticket_id=ticket_id,
            ).to_dict()

        context_tokens += int(item.get("context_tokens", 0))
        tokens_saved += int(item.get("tokens_saved", 0))
        elapsed_ms = float(item.get("elapsed_ms", 0.0))
        total_elapsed_ms += elapsed_ms
        if raw_payload is not None and elapsed_ms > 0:
            latencies.append(elapsed_ms)

        provider_elapsed = float(item.get("provider_elapsed_ms", 0.0))
        provider_elapsed_ms += provider_elapsed
        if provider_elapsed > 0:
            provider_latencies.append(provider_elapsed)
        provider_attempts += int(item.get("provider_attempts", 0))
        provider_prompt_tokens += int(item.get("provider_prompt_tokens") or 0)
        provider_completion_tokens += int(item.get("provider_completion_tokens") or 0)
        provider_total_tokens += int(item.get("provider_total_tokens") or 0)

        if (
            isinstance(raw_payload, Mapping)
            and isinstance(guarded_payload, Mapping)
            and raw_payload.get("action") != guarded_payload.get("action")
        ):
            changes += 1

    return (
        raw,
        guarded,
        context_tokens,
        tokens_saved,
        total_elapsed_ms,
        latencies,
        provider_elapsed_ms,
        provider_latencies,
        provider_attempts,
        provider_prompt_tokens,
        provider_completion_tokens,
        provider_total_tokens,
        failed,
        changes,
    )


def analyze_report(
    report: Mapping[str, Any],
    cases: Sequence[TicketCase],
) -> ExperimentAnalysis:
    (
        raw,
        guarded,
        context_tokens,
        tokens_saved,
        total_elapsed_ms,
        latencies,
        provider_elapsed_ms,
        provider_latencies,
        provider_attempts,
        provider_prompt_tokens,
        provider_completion_tokens,
        provider_total_tokens,
        failed,
        changes,
    ) = _prediction_maps(report)

    slices: list[SliceMetrics] = []

    for category in CATEGORIES:
        selected = [case for case in cases if case.labels.category == category]
        slices.append(
            _metrics(
                selected,
                raw,
                guarded,
                name="category",
                value=category,
            )
        )

    for sla in SLAS:
        selected = [case for case in cases if case.labels.sla == sla]
        slices.append(
            _metrics(
                selected,
                raw,
                guarded,
                name="sla",
                value=sla,
            )
        )

    boolean_slices = (
        ("contradiction", lambda case: case.labels.contradiction),
        ("requires_human", lambda case: case.labels.requires_human),
        ("redundant", lambda case: case.labels.redundant),
        ("irrelevant", lambda case: case.labels.irrelevant),
        ("historical_context", lambda case: case.labels.historical_context),
    )
    for name, predicate in boolean_slices:
        for value in (True, False):
            selected = [case for case in cases if predicate(case) is value]
            slices.append(
                _metrics(
                    selected,
                    raw,
                    guarded,
                    name=name,
                    value=str(value).lower(),
                )
            )

    return ExperimentAnalysis(
        schema_version="promptforge-v27.12-provider-telemetry.v1",
        total_cases=len(cases),
        failed_calls=failed,
        raw_covered=len(raw),
        guarded_covered=len(guarded),
        guard_action_changes=changes,
        total_context_tokens=context_tokens,
        total_tokens_saved=tokens_saved,
        total_elapsed_ms=total_elapsed_ms,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        provider_elapsed_ms=provider_elapsed_ms,
        provider_latency_p50_ms=_percentile(provider_latencies, 0.50),
        provider_latency_p95_ms=_percentile(provider_latencies, 0.95),
        provider_attempts=provider_attempts,
        provider_prompt_tokens=provider_prompt_tokens,
        provider_completion_tokens=provider_completion_tokens,
        provider_total_tokens=provider_total_tokens,
        slices=tuple(slices),
    )


def render_markdown(analysis: ExperimentAnalysis) -> str:
    lines = [
        "# PromptForge V27.9 experiment analysis",
        "",
        f"- Total cases: {analysis.total_cases}",
        f"- Failed calls: {analysis.failed_calls}",
        f"- Raw coverage: {analysis.raw_covered / analysis.total_cases:.2%}",
        f"- Guard action changes: {analysis.guard_action_changes}",
        f"- Total elapsed: {analysis.total_elapsed_ms:.2f} ms",
        f"- Total provider time: {analysis.provider_elapsed_ms:.2f} ms",
        f"- End-to-end latency p50/p95: {analysis.latency_p50_ms:.2f} / {analysis.latency_p95_ms:.2f} ms",
        f"- Provider latency p50/p95: {analysis.provider_latency_p50_ms:.2f} / {analysis.provider_latency_p95_ms:.2f} ms",
        f"- Provider attempts: {analysis.provider_attempts}",
        f"- Provider tokens: prompt={analysis.provider_prompt_tokens}, completion={analysis.provider_completion_tokens}, total={analysis.provider_total_tokens}",
        "",
        "| Slice | Value | Action raw | Action guarded | Unsafe raw | Unsafe guarded | Human recall raw | Human recall guarded | Guard changes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in analysis.slices:
        lines.append(
            "| {name} | {value} | {raw:.2%} | {guarded:.2%} | "
            "{unsafe_raw:.2%} | {unsafe_guarded:.2%} | "
            "{human_raw:.2%} | {human_guarded:.2%} | {changes:.2%} |".format(
                name=item.name,
                value=item.value,
                raw=item.raw_action_accuracy,
                guarded=item.guarded_action_accuracy,
                unsafe_raw=item.raw_unsafe_action_rate,
                unsafe_guarded=item.guarded_unsafe_action_rate,
                human_raw=item.raw_human_recall,
                human_guarded=item.guarded_human_recall,
                changes=item.guard_action_change_rate,
            )
        )
    return "\n".join(lines) + "\n"


def load_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("report must be a JSON object")
    return dict(payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a PromptForge model-loop report by benchmark slice."
    )
    parser.add_argument("report")
    parser.add_argument("--output-json", default="experiment_analysis.json")
    parser.add_argument("--output-markdown", default="experiment_analysis.md")
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=271)
    args = parser.parse_args()

    report = load_report(args.report)
    cases = generate_ticket_suite(count=args.count, seed=args.seed)
    analysis = analyze_report(report, cases)

    Path(args.output_json).write_text(
        json.dumps(analysis.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    Path(args.output_markdown).write_text(
        render_markdown(analysis),
        encoding="utf-8",
    )
    print(json.dumps(analysis.to_dict()["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
