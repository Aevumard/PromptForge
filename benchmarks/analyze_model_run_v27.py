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
            },
            "slices": [item.to_dict() for item in self.slices],
        }


def _mean(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


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


def _prediction_maps(report: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], int, int, int, int]:
    raw: dict[str, Any] = {}
    guarded: dict[str, Any] = {}
    context_tokens = 0
    tokens_saved = 0
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

        if (
            isinstance(raw_payload, Mapping)
            and isinstance(guarded_payload, Mapping)
            and raw_payload.get("action") != guarded_payload.get("action")
        ):
            changes += 1

    return raw, guarded, context_tokens, tokens_saved, failed, changes


def analyze_report(
    report: Mapping[str, Any],
    cases: Sequence[TicketCase],
) -> ExperimentAnalysis:
    raw, guarded, context_tokens, tokens_saved, failed, changes = _prediction_maps(
        report
    )

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
        schema_version="promptforge-v27.9-experiment-analysis.v1",
        total_cases=len(cases),
        failed_calls=failed,
        raw_covered=len(raw),
        guarded_covered=len(guarded),
        guard_action_changes=changes,
        total_context_tokens=context_tokens,
        total_tokens_saved=tokens_saved,
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
