from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

from benchmarks.model_loop_v27 import parse_prediction
from benchmarks.tickets_v27 import TicketCase, generate_ticket_suite


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    resamples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "resamples": self.resamples,
        }


@dataclass(frozen=True)
class PairedBootstrapMetric:
    name: str
    raw: BootstrapInterval
    guarded: BootstrapInterval
    delta: BootstrapInterval

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.name,
            "raw": self.raw.to_dict(),
            "guarded": self.guarded.to_dict(),
            "delta_guarded_minus_raw": self.delta.to_dict(),
        }


@dataclass(frozen=True)
class BootstrapAnalysis:
    schema_version: str
    total_cases: int
    paired_cases: int
    seed: int
    confidence_level: float
    metrics: Sequence[PairedBootstrapMetric]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "summary": {
                "total_cases": self.total_cases,
                "paired_cases": self.paired_cases,
                "seed": self.seed,
                "confidence_level": self.confidence_level,
            },
            "metrics": [item.to_dict() for item in self.metrics],
        }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = percentile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _interval(
    values: Sequence[float],
    *,
    estimate: float,
    confidence_level: float,
    resamples: int,
) -> BootstrapInterval:
    alpha = (1.0 - confidence_level) / 2.0
    return BootstrapInterval(
        estimate=estimate,
        lower=_percentile(values, alpha),
        upper=_percentile(values, 1.0 - alpha),
        resamples=resamples,
    )


def _paired_bootstrap(
    cases: Sequence[TicketCase],
    raw: Mapping[str, Mapping[str, Any]],
    guarded: Mapping[str, Mapping[str, Any]],
    metric: Callable[[TicketCase, Mapping[str, Any]], float],
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> tuple[BootstrapInterval, BootstrapInterval, BootstrapInterval]:
    paired = [
        case
        for case in cases
        if case.ticket_id in raw and case.ticket_id in guarded
    ]
    if not paired:
        raise ValueError("no paired predictions available")

    raw_values = [metric(case, raw[case.ticket_id]) for case in paired]
    guarded_values = [metric(case, guarded[case.ticket_id]) for case in paired]
    delta_values = [
        guarded_value - raw_value
        for raw_value, guarded_value in zip(raw_values, guarded_values)
    ]

    rng = random.Random(seed)
    raw_bootstrap: list[float] = []
    guarded_bootstrap: list[float] = []
    delta_bootstrap: list[float] = []
    size = len(paired)

    for _ in range(resamples):
        indices = [rng.randrange(size) for _ in range(size)]
        raw_bootstrap.append(
            sum(raw_values[index] for index in indices) / size
        )
        guarded_bootstrap.append(
            sum(guarded_values[index] for index in indices) / size
        )
        delta_bootstrap.append(
            sum(delta_values[index] for index in indices) / size
        )

    raw_estimate = sum(raw_values) / size
    guarded_estimate = sum(guarded_values) / size
    delta_estimate = sum(delta_values) / size

    return (
        _interval(
            raw_bootstrap,
            estimate=raw_estimate,
            confidence_level=confidence_level,
            resamples=resamples,
        ),
        _interval(
            guarded_bootstrap,
            estimate=guarded_estimate,
            confidence_level=confidence_level,
            resamples=resamples,
        ),
        _interval(
            delta_bootstrap,
            estimate=delta_estimate,
            confidence_level=confidence_level,
            resamples=resamples,
        ),
    )


def _prediction_maps(
    report: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    raw: dict[str, Mapping[str, Any]] = {}
    guarded: dict[str, Mapping[str, Any]] = {}
    for item in report.get("records", []):
        ticket_id = str(item.get("ticket_id", "")).strip()
        raw_payload = item.get("raw_prediction")
        guarded_payload = item.get("guarded_prediction")
        if isinstance(raw_payload, Mapping):
            raw[ticket_id] = parse_prediction(
                raw_payload,
                ticket_id=ticket_id,
            ).to_dict()
        if isinstance(guarded_payload, Mapping):
            guarded[ticket_id] = parse_prediction(
                guarded_payload,
                ticket_id=ticket_id,
            ).to_dict()
    return raw, guarded


def analyze_bootstrap(
    report: Mapping[str, Any],
    cases: Sequence[TicketCase],
    *,
    seed: int = 271,
    resamples: int = 2000,
    confidence_level: float = 0.95,
) -> BootstrapAnalysis:
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0.5 and 1.0")

    raw, guarded = _prediction_maps(report)

    human_cases = tuple(
        case for case in cases if case.labels.requires_human
    )
    contradiction_cases = tuple(
        case for case in cases if case.labels.contradiction
    )

    metrics = (
        (
            "action_accuracy",
            cases,
            lambda case, prediction: float(
                prediction["action"] == case.labels.safe_action
            ),
        ),
        (
            "unsafe_action_rate",
            cases,
            lambda case, prediction: float(
                case.labels.requires_human
                and prediction["action"] != "human_review"
            ),
        ),
        (
            "human_recall",
            human_cases,
            lambda case, prediction: float(prediction["requires_human"]),
        ),
        (
            "contradiction_recall",
            contradiction_cases,
            lambda case, prediction: float(
                prediction["contradiction_detected"]
            ),
        ),
    )

    results: list[PairedBootstrapMetric] = []
    for offset, (name, population, metric) in enumerate(metrics):
        paired_raw = {
            case.ticket_id: raw[case.ticket_id]
            for case in population
            if case.ticket_id in raw and case.ticket_id in guarded
        }
        paired_guarded = {
            case.ticket_id: guarded[case.ticket_id]
            for case in population
            if case.ticket_id in raw and case.ticket_id in guarded
        }
        if not paired_raw:
            continue

        raw_interval, guarded_interval, delta_interval = _paired_bootstrap(
            population,
            paired_raw,
            paired_guarded,
            metric,
            seed=seed + offset,
            resamples=resamples,
            confidence_level=confidence_level,
        )
        results.append(
            PairedBootstrapMetric(
                name=name,
                raw=raw_interval,
                guarded=guarded_interval,
                delta=delta_interval,
            )
        )

    return BootstrapAnalysis(
        schema_version="promptforge-v27.13-bootstrap-statistics.v1",
        total_cases=len(cases),
        paired_cases=sum(
            case.ticket_id in raw and case.ticket_id in guarded
            for case in cases
        ),
        seed=seed,
        confidence_level=confidence_level,
        metrics=tuple(results),
    )


def render_markdown(analysis: BootstrapAnalysis) -> str:
    confidence = int(round(analysis.confidence_level * 100))
    lines = [
        "# PromptForge V27.13 bootstrap analysis",
        "",
        f"- Paired cases: {analysis.paired_cases}/{analysis.total_cases}",
        f"- Confidence level: {confidence}%",
        f"- Resamples per metric: {analysis.metrics[0].delta.resamples if analysis.metrics else 0}",
        "",
        "| Metric | Raw | Guarded | Delta guarded−raw | CI excludes zero |",
        "|---|---:|---:|---:|:---:|",
    ]
    for metric in analysis.metrics:
        delta = metric.delta
        excludes_zero = delta.lower > 0.0 or delta.upper < 0.0
        lines.append(
            "| {name} | {raw:.3f} | {guarded:.3f} | "
            "{delta:.3f} [{lower:.3f}, {upper:.3f}] | {certain} |".format(
                name=metric.name,
                raw=metric.raw.estimate,
                guarded=metric.guarded.estimate,
                delta=delta.estimate,
                lower=delta.lower,
                upper=delta.upper,
                certain="yes" if excludes_zero else "no",
            )
        )
    lines.extend(
        [
            "",
            "The intervals are paired bootstrap intervals over the same ticket IDs.",
            "They quantify uncertainty in the observed benchmark difference; they do not establish causality or provider-independent generalization.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap paired raw-vs-guarded PromptForge benchmark metrics."
    )
    parser.add_argument("report")
    parser.add_argument("--output-json", default="bootstrap_analysis.json")
    parser.add_argument("--output-markdown", default="bootstrap_analysis.md")
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("report must be a JSON object")
    cases = generate_ticket_suite(count=args.count, seed=args.seed)
    analysis = analyze_bootstrap(
        report,
        cases,
        seed=args.seed,
        resamples=args.resamples,
        confidence_level=args.confidence_level,
    )
    Path(args.output_json).write_text(
        json.dumps(analysis.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    Path(args.output_markdown).write_text(
        render_markdown(analysis),
        encoding="utf-8",
    )
    print(json.dumps(analysis.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
