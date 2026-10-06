from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import re
from typing import Any, Sequence

from benchmarks.analyze_model_run_v27 import ExperimentAnalysis, analyze_report, render_markdown
from benchmarks.model_loop_v27 import AgentAdapter, ModelLoopReport
from benchmarks.providers.openai_compatible import (
    OpenAICompatibleAgentAdapter,
    OpenAICompatibleConfig,
)
from benchmarks.resumable_model_loop_v27 import run_resumable_model_loop
from benchmarks.tickets_v27 import TicketCase, generate_ticket_suite


@dataclass(frozen=True)
class AblationVariant:
    """One controlled PromptForge configuration for the same ticket suite."""

    name: str
    budget_tokens: int = 500
    reserve_tokens: int = 50
    include_epistemic: bool = True
    apply_guard: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("variant name must not be empty")
        if self.budget_tokens < 1:
            raise ValueError("budget_tokens must be positive")
        if self.reserve_tokens < 0:
            raise ValueError("reserve_tokens must be non-negative")
        if self.reserve_tokens >= self.budget_tokens:
            raise ValueError("reserve_tokens must be smaller than budget_tokens")

    @property
    def checkpoint_name(self) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", self.name.strip()).strip("-")
        slug = slug or "variant"
        epistemic = "e1" if self.include_epistemic else "e0"
        guard = "g1" if self.apply_guard else "g0"
        return f"{slug}.b{self.budget_tokens}.r{self.reserve_tokens}.{epistemic}.{guard}"


@dataclass(frozen=True)
class AblationResult:
    variant: AblationVariant
    report: ModelLoopReport
    analysis: ExperimentAnalysis

    def to_dict(self) -> dict[str, Any]:
        payload = self.variant.__dict__.copy()
        payload["report"] = self.report.to_dict()
        payload["analysis"] = self.analysis.to_dict()
        return {"variant": payload}


@dataclass(frozen=True)
class AblationReport:
    schema_version: str
    count: int
    seed: int
    results: Sequence[AblationResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite": {"count": self.count, "seed": self.seed},
            "results": [result.to_dict() for result in self.results],
        }


DEFAULT_VARIANTS = (
    AblationVariant(
        name="baseline_budget_only",
        include_epistemic=False,
        apply_guard=False,
    ),
    AblationVariant(
        name="epistemic_budget",
        include_epistemic=True,
        apply_guard=False,
    ),
    AblationVariant(
        name="epistemic_budget_guarded",
        include_epistemic=True,
        apply_guard=True,
    ),
)


def run_ablation_suite(
    cases: Sequence[TicketCase],
    adapter: AgentAdapter,
    *,
    variants: Sequence[AblationVariant] = DEFAULT_VARIANTS,
    checkpoint_dir: str | Path = "ablation_checkpoints",
    seed: int = -1,
    retry_failed: bool = True,
    fsync_each_record: bool = True,
) -> AblationReport:
    """Run comparable variants over the same ordered ticket suite.

    Each variant receives its own resumable checkpoint, so an interrupted
    experiment can restart without re-calling successful tickets.
    """

    if not cases:
        raise ValueError("cases must not be empty")
    if not variants:
        raise ValueError("variants must not be empty")

    root = Path(checkpoint_dir)
    results: list[AblationResult] = []

    for variant in variants:
        checkpoint = root / f"{variant.checkpoint_name}.checkpoint.jsonl"
        report = run_resumable_model_loop(
            cases,
            adapter,
            checkpoint_path=checkpoint,
            budget_tokens=variant.budget_tokens,
            reserve_tokens=variant.reserve_tokens,
            apply_guard=variant.apply_guard,
            include_epistemic=variant.include_epistemic,
            retry_failed=retry_failed,
            fsync_each_record=fsync_each_record,
        )
        analysis = analyze_report(report.to_dict(), cases)
        results.append(
            AblationResult(
                variant=variant,
                report=report,
                analysis=analysis,
            )
        )

    return AblationReport(
        schema_version="promptforge-v27.10-ablation-suite.v1",
        count=len(cases),
        seed=seed,
        results=tuple(results),
    )


def render_ablation_markdown(report: AblationReport) -> str:
    lines = [
        "# PromptForge V27.10 controlled ablation",
        "",
        (
            "Each variant uses the same ordered ticket suite. The variants "
            "differ only in the PromptForge layer being ablated."
        ),
        "",
        "| Variant | Epistemic | Guard | Budget | Coverage | Raw action accuracy | Guarded action accuracy | Raw unsafe rate | Guarded unsafe rate | Context tokens | Tokens saved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for result in report.results:
        metrics = result.report.raw_metrics
        guarded = result.report.guarded_metrics
        lines.append(
            "| {name} | {ep} | {guard} | {budget} | {coverage:.2%} | "
            "{raw_action:.2%} | {guarded_action:.2%} | {raw_unsafe:.2%} | "
            "{guarded_unsafe:.2%} | {tokens} | {saved} |".format(
                name=result.variant.name,
                ep="yes" if result.variant.include_epistemic else "no",
                guard="yes" if result.variant.apply_guard else "no",
                budget=result.variant.budget_tokens,
                coverage=metrics.covered_cases / metrics.total_cases,
                raw_action=metrics.action_accuracy,
                guarded_action=guarded.action_accuracy,
                raw_unsafe=metrics.unsafe_action_rate,
                guarded_unsafe=guarded.unsafe_action_rate,
                tokens=result.report.total_context_tokens,
                saved=result.report.total_tokens_saved,
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- baseline_budget_only isolates PromptForge budgeting from the epistemic summary and action guard.",
            "- epistemic_budget adds the epistemic compiler output while keeping the action guard off.",
            "- epistemic_budget_guarded adds the action safety layer on top.",
            "- This is an ablation benchmark, not a causal proof; model nondeterminism and provider behavior can still affect comparisons.",
            "",
        ]
    )
    return "\n".join(lines)


def write_json(report: AblationReport, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_markdown(report: AblationReport, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_ablation_markdown(report),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the PromptForge V27.10 controlled ablation suite."
    )
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--checkpoint-dir", default="ablation_checkpoints")
    parser.add_argument("--output-json", default="ablation_report.json")
    parser.add_argument("--output-markdown", default="ablation_report.md")
    parser.add_argument("--no-retry-failed", action="store_true")
    parser.add_argument("--no-fsync", action="store_true")
    args = parser.parse_args()

    cases = generate_ticket_suite(count=args.count, seed=args.seed)
    config = OpenAICompatibleConfig.from_env()
    adapter = OpenAICompatibleAgentAdapter(config)
    report = run_ablation_suite(
        cases,
        adapter,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        retry_failed=not args.no_retry_failed,
        fsync_each_record=not args.no_fsync,
    )
    report = AblationReport(
        schema_version=report.schema_version,
        count=report.count,
        seed=args.seed,
        results=report.results,
    )
    write_json(report, args.output_json)
    write_markdown(report, args.output_markdown)
    print(json.dumps(report.to_dict()["suite"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
