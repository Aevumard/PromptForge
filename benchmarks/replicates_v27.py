from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from benchmarks.analyze_bootstrap_v27 import (
    analyze_bootstrap,
    render_markdown as render_bootstrap_markdown,
)
from benchmarks.analyze_model_run_v27 import analyze_report, render_markdown as render_model_analysis_markdown
from benchmarks.model_loop_v27 import AgentAdapter, ModelLoopReport, Prediction
from benchmarks.providers.openai_compatible import (
    OpenAICompatibleAgentAdapter,
    OpenAICompatibleConfig,
    write_prediction_jsonl,
    write_report,
)
from benchmarks.resumable_model_loop_v27 import run_resumable_model_loop
from benchmarks.tickets_v27 import TicketCase, generate_ticket_suite

SCHEMA_VERSION = "promptforge-v27.14-replicate-variability.v1"
_SECRET_METADATA_KEYS = frozenset({"api_key", "authorization", "password", "secret", "token"})


@dataclass(frozen=True)
class ScalarSummary:
    count: int
    mean: float
    std: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
        }


def summarize(values: Sequence[float]) -> ScalarSummary:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return ScalarSummary(
        count=len(values),
        mean=statistics.fmean(values),
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
        minimum=min(values),
        maximum=max(values),
    )


@dataclass(frozen=True)
class ReplicateRun:
    replicate_index: int
    run_id: str
    report: ModelLoopReport
    analysis: Any
    bootstrap: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "replicate_index": self.replicate_index,
            "run_id": self.run_id,
            "report": self.report.to_dict(),
            "analysis": self.analysis.to_dict(),
            "bootstrap": self.bootstrap.to_dict(),
        }


@dataclass(frozen=True)
class PairwiseDisagreement:
    pairs: int
    comparable_pairs: int
    raw_action_disagreement_rate: float
    guarded_action_disagreement_rate: float
    raw_decision_disagreement_rate: float
    guarded_decision_disagreement_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "replicate_pairs": self.pairs,
            "comparable_pairs": self.comparable_pairs,
            "raw_action_disagreement_rate": self.raw_action_disagreement_rate,
            "guarded_action_disagreement_rate": self.guarded_action_disagreement_rate,
            "raw_decision_disagreement_rate": self.raw_decision_disagreement_rate,
            "guarded_decision_disagreement_rate": self.guarded_decision_disagreement_rate,
        }


@dataclass(frozen=True)
class ReplicateExperiment:
    schema_version: str
    suite: Mapping[str, int]
    condition: Mapping[str, Any]
    provider: Mapping[str, Any]
    replicate_count: int
    replicates: Sequence[ReplicateRun]
    metrics: Mapping[str, ScalarSummary]
    pairwise_disagreement: PairwiseDisagreement

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite": dict(self.suite),
            "condition": dict(self.condition),
            "provider": dict(self.provider),
            "replicate_count": self.replicate_count,
            "replicates": [item.to_dict() for item in self.replicates],
            "metrics": {name: summary.to_dict() for name, summary in self.metrics.items()},
            "pairwise_disagreement": self.pairwise_disagreement.to_dict(),
        }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _metric_values(report: ModelLoopReport) -> dict[str, float]:
    raw = report.raw_metrics.to_dict()
    guarded = report.guarded_metrics.to_dict()
    usage_records = [
        record for record in report.records
        if record.provider_total_tokens is not None
    ]
    return {
        "raw.action_accuracy": float(raw["action_accuracy"]),
        "guarded.action_accuracy": float(guarded["action_accuracy"]),
        "raw.unsafe_action_rate": float(raw["unsafe_action_rate"]),
        "guarded.unsafe_action_rate": float(guarded["unsafe_action_rate"]),
        "raw.human_recall": float(raw["human_recall"]),
        "guarded.human_recall": float(guarded["human_recall"]),
        "raw.contradiction_recall": float(raw["contradiction_recall"]),
        "guarded.contradiction_recall": float(guarded["contradiction_recall"]),
        "raw.coverage": (
            float(raw["covered_cases"]) / float(raw["total_cases"])
            if raw["total_cases"] else 0.0
        ),
        "guarded.coverage": (
            float(guarded["covered_cases"]) / float(guarded["total_cases"])
            if guarded["total_cases"] else 0.0
        ),
        "failure_rate": (
            float(report.failed_calls) / float(report.total_calls)
            if report.total_calls else 0.0
        ),
        "context_tokens": float(report.total_context_tokens),
        "tokens_saved": float(report.total_tokens_saved),
        "e2e_latency_p50_ms": _percentile(
            [record.elapsed_ms for record in report.records], 0.50
        ),
        "e2e_latency_p95_ms": _percentile(
            [record.elapsed_ms for record in report.records], 0.95
        ),
        "provider_latency_p50_ms": _percentile(
            [record.provider_elapsed_ms for record in report.records], 0.50
        ),
        "provider_latency_p95_ms": _percentile(
            [record.provider_elapsed_ms for record in report.records], 0.95
        ),
        "provider_attempts": float(report.provider_attempts),
        "provider_usage_coverage": (
            float(len(usage_records)) / float(report.total_calls)
            if report.total_calls else 0.0
        ),
        "provider_prompt_tokens": float(
            sum(record.provider_prompt_tokens or 0 for record in usage_records)
        ),
        "provider_completion_tokens": float(
            sum(record.provider_completion_tokens or 0 for record in usage_records)
        ),
        "provider_total_tokens": float(
            sum(record.provider_total_tokens or 0 for record in usage_records)
        ),
    }


def _prediction_maps(
    report: ModelLoopReport,
) -> tuple[dict[str, Prediction], dict[str, Prediction]]:
    raw: dict[str, Prediction] = {}
    guarded: dict[str, Prediction] = {}
    for record in report.records:
        if record.raw_prediction is not None:
            raw[record.ticket_id] = record.raw_prediction
        if record.guarded_prediction is not None:
            guarded[record.ticket_id] = record.guarded_prediction
    return raw, guarded


def _pairwise_disagreement(
    reports: Sequence[ModelLoopReport],
) -> PairwiseDisagreement:
    if len(reports) < 2:
        return PairwiseDisagreement(0, 0, 0.0, 0.0, 0.0, 0.0)

    raw_maps = []
    guarded_maps = []
    for report in reports:
        raw, guarded = _prediction_maps(report)
        raw_maps.append(raw)
        guarded_maps.append(guarded)

    total_pairs = len(reports) * (len(reports) - 1) // 2
    comparable_pairs = 0
    raw_action_rates: list[float] = []
    guarded_action_rates: list[float] = []
    raw_decision_rates: list[float] = []
    guarded_decision_rates: list[float] = []

    for left, right in itertools.combinations(range(len(reports)), 2):
        comparable_ids = sorted(
            set(raw_maps[left])
            & set(raw_maps[right])
            & set(guarded_maps[left])
            & set(guarded_maps[right])
        )
        if not comparable_ids:
            continue

        comparable_pairs += 1
        raw_action_rates.append(
            statistics.fmean(
                raw_maps[left][ticket_id].action
                != raw_maps[right][ticket_id].action
                for ticket_id in comparable_ids
            )
        )
        guarded_action_rates.append(
            statistics.fmean(
                guarded_maps[left][ticket_id].action
                != guarded_maps[right][ticket_id].action
                for ticket_id in comparable_ids
            )
        )
        raw_decision_rates.append(
            statistics.fmean(
                raw_maps[left][ticket_id].to_dict()
                != raw_maps[right][ticket_id].to_dict()
                for ticket_id in comparable_ids
            )
        )
        guarded_decision_rates.append(
            statistics.fmean(
                guarded_maps[left][ticket_id].to_dict()
                != guarded_maps[right][ticket_id].to_dict()
                for ticket_id in comparable_ids
            )
        )

    return PairwiseDisagreement(
        pairs=total_pairs,
        comparable_pairs=comparable_pairs,
        raw_action_disagreement_rate=(
            statistics.fmean(raw_action_rates) if raw_action_rates else 0.0
        ),
        guarded_action_disagreement_rate=(
            statistics.fmean(guarded_action_rates)
            if guarded_action_rates else 0.0
        ),
        raw_decision_disagreement_rate=(
            statistics.fmean(raw_decision_rates)
            if raw_decision_rates else 0.0
        ),
        guarded_decision_disagreement_rate=(
            statistics.fmean(guarded_decision_rates)
            if guarded_decision_rates else 0.0
        ),
    )


def _provider_metadata(config: OpenAICompatibleConfig) -> dict[str, Any]:
    split = urlsplit(config.endpoint_url)
    hostname = split.hostname or ""
    try:
        port = split.port
    except ValueError:
        port = None
    netloc = hostname if port is None else f"{hostname}:{port}"
    safe_url = urlunsplit((split.scheme, netloc, split.path, "", ""))
    prompt_hash = hashlib.sha256(
        config.system_prompt.encode("utf-8")
    ).hexdigest()
    return {
        "endpoint": safe_url,
        "model": config.model,
        "json_mode": config.json_mode,
        "timeout_seconds": config.timeout_seconds,
        "max_attempts": config.max_attempts,
        "retry_backoff_seconds": config.retry_backoff_seconds,
        "system_prompt_sha256": prompt_hash,
    }


def _safe_provider_metadata(
    provider_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not provider_metadata:
        return {}
    return {
        key: value
        for key, value in provider_metadata.items()
        if key.lower() not in _SECRET_METADATA_KEYS
    }


def _write_or_validate_manifest(
    root: Path,
    *,
    suite: Mapping[str, Any],
    condition: Mapping[str, Any],
    provider: Mapping[str, Any],
) -> None:
    manifest_path = root / "manifest.json"
    expected = {
        "suite": dict(suite),
        "condition": dict(condition),
        "provider": dict(provider),
    }
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current != expected:
            raise ValueError(
                "existing replicate output directory is bound to a different "
                "suite, condition, or provider; choose another --output-dir"
            )
        return
    manifest_path.write_text(
        json.dumps(expected, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_replicate_experiment(
    cases: Sequence[TicketCase],
    adapter_factory: Callable[[], AgentAdapter],
    *,
    replicates: int = 3,
    seed: int = 271,
    bootstrap_resamples: int = 2000,
    bootstrap_confidence: float = 0.95,
    output_dir: str | Path = "replicate_runs",
    budget_tokens: int = 500,
    reserve_tokens: int = 50,
    include_epistemic: bool = True,
    apply_guard: bool = True,
    retry_failed: bool = True,
    fsync_each_record: bool = True,
    provider_metadata: Mapping[str, Any] | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> ReplicateExperiment:
    if not cases:
        raise ValueError("cases must not be empty")
    if replicates < 2:
        raise ValueError("replicates must be at least 2")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe_provider = _safe_provider_metadata(provider_metadata)
    condition = {
        "budget_tokens": budget_tokens,
        "reserve_tokens": reserve_tokens,
        "include_epistemic": include_epistemic,
        "apply_guard": apply_guard,
    }
    suite = {"count": len(cases), "seed": seed}
    _write_or_validate_manifest(
        root,
        suite=suite,
        condition=condition,
        provider=safe_provider,
    )

    results: list[ReplicateRun] = []
    for index in range(1, replicates + 1):
        run_id = f"replicate-{index:03d}"
        run_dir = root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / "model_run.checkpoint.jsonl"

        if progress_callback is not None:
            progress_callback(
                "replicate_start",
                {"index": index, "total": replicates, "run_id": run_id},
            )
        adapter = adapter_factory()
        report = run_resumable_model_loop(
            cases,
            adapter,
            checkpoint_path=checkpoint,
            budget_tokens=budget_tokens,
            reserve_tokens=reserve_tokens,
            apply_guard=apply_guard,
            include_epistemic=include_epistemic,
            retry_failed=retry_failed,
            fsync_each_record=fsync_each_record,
        )
        analysis = analyze_report(report.to_dict(), cases)
        bootstrap = analyze_bootstrap(
            report.to_dict(),
            cases,
            seed=seed,
            resamples=bootstrap_resamples,
            confidence_level=bootstrap_confidence,
        )

        write_prediction_jsonl(report, run_dir / "predictions.jsonl")
        write_report(report, run_dir / "model_report.json")
        (run_dir / "experiment_analysis.json").write_text(
            json.dumps(analysis.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (run_dir / "experiment_analysis.md").write_text(
            render_model_analysis_markdown(analysis),
            encoding="utf-8",
        )
        (run_dir / "bootstrap_analysis.json").write_text(
            json.dumps(bootstrap.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (run_dir / "bootstrap_analysis.md").write_text(
            render_bootstrap_markdown(bootstrap),
            encoding="utf-8",
        )

        result = ReplicateRun(
            replicate_index=index,
            run_id=run_id,
            report=report,
            analysis=analysis,
            bootstrap=bootstrap,
        )
        results.append(result)
        if progress_callback is not None:
            progress_callback(
                "replicate_complete",
                {
                    "index": index,
                    "total": replicates,
                    "run_id": run_id,
                    "coverage": (
                        report.raw_metrics.covered_cases / report.raw_metrics.total_cases
                        if report.raw_metrics.total_cases
                        else 0.0
                    ),
                    "guarded_action_accuracy": report.guarded_metrics.action_accuracy,
                    "guarded_unsafe_action_rate": report.guarded_metrics.unsafe_action_rate,
                },
            )

    metric_values: dict[str, list[float]] = {}
    for result in results:
        for name, value in _metric_values(result.report).items():
            metric_values.setdefault(name, []).append(value)

    return ReplicateExperiment(
        schema_version=SCHEMA_VERSION,
        suite=suite,
        condition=condition,
        provider=safe_provider,
        replicate_count=len(results),
        replicates=tuple(results),
        metrics={
            name: summarize(values)
            for name, values in sorted(metric_values.items())
        },
        pairwise_disagreement=_pairwise_disagreement(
            [result.report for result in results]
        ),
    )


def render_markdown(experiment: ReplicateExperiment) -> str:
    lines = [
        "# PromptForge V27.14 replicate variability",
        "",
        (
            "Repeated runs use the same ticket suite and condition. "
            "Between-replicate dispersion measures model/provider variability; "
            "the paired bootstrap inside each run measures sampling uncertainty "
            "over ticket IDs. They answer different questions."
        ),
        "",
        f"- Replicates: {experiment.replicate_count}",
        (
            f"- Ticket suite: {experiment.suite['count']} tickets "
            f"(seed {experiment.suite['seed']})"
        ),
        "",
        "## Condition",
        "",
    ]
    for key, value in experiment.condition.items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Between-replicate metrics",
            "",
            "| Metric | Mean | Std | Min | Max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, summary in experiment.metrics.items():
        lines.append(
            "| {name} | {mean:.4f} | {std:.4f} | "
            "{minimum:.4f} | {maximum:.4f} |".format(
                name=name,
                mean=summary.mean,
                std=summary.std,
                minimum=summary.minimum,
                maximum=summary.maximum,
            )
        )

    disagreement = experiment.pairwise_disagreement
    lines.extend(
        [
            "",
            "## Cross-replicate prediction stability",
            "",
            f"- Replicate pairs: {disagreement.pairs}",
            f"- Comparable pairs: {disagreement.comparable_pairs}",
            f"- Raw action disagreement: "
            f"{disagreement.raw_action_disagreement_rate:.2%}",
            f"- Guarded action disagreement: "
            f"{disagreement.guarded_action_disagreement_rate:.2%}",
            f"- Raw full-decision disagreement: "
            f"{disagreement.raw_decision_disagreement_rate:.2%}",
            f"- Guarded full-decision disagreement: "
            f"{disagreement.guarded_decision_disagreement_rate:.2%}",
            "",
            (
                "A disagreement rate of 0% means comparable replicas made "
                "identical decisions. Non-zero values quantify output "
                "instability, not error by themselves."
            ),
            "",
            "## Per-replicate artifacts",
            "",
        ]
    )
    for result in experiment.replicates:
        lines.append(
            f"- {result.run_id}: "
            f"replicate-{result.replicate_index:03d}/model_report.json, "
            f"replicate-{result.replicate_index:03d}/bootstrap_analysis.json"
        )

    lines.extend(
        [
            "",
            "Provider metadata:",
            "",
            json.dumps(experiment.provider, indent=2, sort_keys=True),
            "",
        ]
    )
    return "\n".join(lines)


def write_experiment(
    experiment: ReplicateExperiment,
    *,
    output_json: str | Path,
    output_markdown: str | Path,
) -> None:
    json_path = Path(output_json)
    markdown_path = Path(output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(experiment.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_markdown(experiment),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat the same PromptForge ticket condition to measure "
            "model/provider variability across replicas."
        )
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--budget-tokens", type=int, default=500)
    parser.add_argument("--reserve-tokens", type=int, default=50)
    parser.add_argument("--output-dir", default="replicate_runs")
    parser.add_argument("--output-json", default="replicate_analysis.json")
    parser.add_argument("--output-markdown", default="replicate_analysis.md")
    parser.add_argument("--no-epistemic", action="store_true")
    parser.add_argument("--no-guard", action="store_true")
    parser.add_argument("--no-retry-failed", action="store_true")
    parser.add_argument("--no-fsync", action="store_true")
    args = parser.parse_args()

    config = OpenAICompatibleConfig.from_env()

    def adapter_factory() -> OpenAICompatibleAgentAdapter:
        return OpenAICompatibleAgentAdapter(config)

    cases = generate_ticket_suite(count=args.count, seed=args.seed)
    experiment = run_replicate_experiment(
        cases,
        adapter_factory,
        replicates=args.replicates,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_confidence=args.bootstrap_confidence,
        output_dir=args.output_dir,
        budget_tokens=args.budget_tokens,
        reserve_tokens=args.reserve_tokens,
        include_epistemic=not args.no_epistemic,
        apply_guard=not args.no_guard,
        retry_failed=not args.no_retry_failed,
        fsync_each_record=not args.no_fsync,
        provider_metadata=_provider_metadata(config),
    )
    write_experiment(
        experiment,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(
        json.dumps(
            {
                "replicates": experiment.replicate_count,
                "pairs": experiment.pairwise_disagreement.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
