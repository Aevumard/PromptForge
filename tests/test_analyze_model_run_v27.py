import json
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest import TestCase

from benchmarks.analyze_model_run_v27 import analyze_report, render_markdown
from benchmarks.model_loop_v27 import demo_baseline_report
from benchmarks.tickets_v27 import generate_ticket_suite


class ExperimentAnalysisTests(TestCase):
    def test_baseline_analysis_has_all_slice_families(self) -> None:
        report = demo_baseline_report(count=16, seed=271).to_dict()
        cases = generate_ticket_suite(count=16, seed=271)
        analysis = analyze_report(report, cases)

        self.assertEqual(
            analysis.schema_version,
            "promptforge-v27.12-provider-telemetry.v1",
        )
        names = {item.name for item in analysis.slices}
        self.assertEqual(
            names,
            {
                "category",
                "sla",
                "contradiction",
                "requires_human",
                "redundant",
                "irrelevant",
                "historical_context",
            },
        )
        self.assertEqual(analysis.raw_covered, 16)
        self.assertEqual(analysis.failed_calls, 0)
        self.assertGreater(analysis.guard_action_changes, 0)
        self.assertGreaterEqual(analysis.total_elapsed_ms, 0.0)
        self.assertGreaterEqual(analysis.latency_p50_ms, 0.0)
        self.assertGreaterEqual(analysis.latency_p95_ms, analysis.latency_p50_ms)
        self.assertGreaterEqual(analysis.provider_elapsed_ms, 0.0)
        self.assertGreaterEqual(analysis.provider_latency_p95_ms, analysis.provider_latency_p50_ms)
        self.assertGreaterEqual(analysis.provider_attempts, 0)
        self.assertGreaterEqual(analysis.provider_total_tokens, 0)

    def test_guard_reduces_unsafe_rate_in_human_slice(self) -> None:
        report = demo_baseline_report(count=16, seed=271).to_dict()
        cases = generate_ticket_suite(count=16, seed=271)
        analysis = analyze_report(report, cases)

        target = next(
            item
            for item in analysis.slices
            if item.name == "requires_human" and item.value == "true"
        )
        self.assertLess(
            target.guarded_unsafe_action_rate,
            target.raw_unsafe_action_rate,
        )

    def test_markdown_contains_comparison_table(self) -> None:
        report = demo_baseline_report(count=16, seed=271).to_dict()
        cases = generate_ticket_suite(count=16, seed=271)
        markdown = render_markdown(analyze_report(report, cases))
        self.assertIn("| Slice | Value |", markdown)
        self.assertIn("unsafe", markdown.lower())
        self.assertIn("End-to-end latency p50/p95", markdown)
        self.assertIn("Provider latency p50/p95", markdown)


if __name__ == "__main__":
    import unittest

    unittest.main()
