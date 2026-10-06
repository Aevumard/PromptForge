import json
from unittest import TestCase

from benchmarks.analyze_bootstrap_v27 import analyze_bootstrap, render_markdown
from benchmarks.model_loop_v27 import demo_baseline_report
from benchmarks.tickets_v27 import generate_ticket_suite


class BootstrapAnalysisTests(TestCase):
    def test_bootstrap_is_deterministic(self) -> None:
        report = demo_baseline_report(count=16, seed=271).to_dict()
        cases = generate_ticket_suite(count=16, seed=271)

        first = analyze_bootstrap(
            report,
            cases,
            seed=271,
            resamples=200,
        )
        second = analyze_bootstrap(
            report,
            cases,
            seed=271,
            resamples=200,
        )

        self.assertEqual(
            first.to_dict(),
            second.to_dict(),
        )
        self.assertEqual(
            first.schema_version,
            "promptforge-v27.13-bootstrap-statistics.v1",
        )

    def test_guard_delta_for_unsafe_rate_is_strictly_negative(self) -> None:
        report = demo_baseline_report(count=16, seed=271).to_dict()
        cases = generate_ticket_suite(count=16, seed=271)
        analysis = analyze_bootstrap(
            report,
            cases,
            seed=271,
            resamples=400,
        )
        target = next(
            item
            for item in analysis.metrics
            if item.name == "unsafe_action_rate"
        )
        self.assertLess(target.delta.estimate, 0.0)
        self.assertLess(target.delta.upper, 0.0)

    def test_markdown_reports_ci(self) -> None:
        report = demo_baseline_report(count=16, seed=271).to_dict()
        cases = generate_ticket_suite(count=16, seed=271)
        analysis = analyze_bootstrap(
            report,
            cases,
            seed=271,
            resamples=200,
        )
        markdown = render_markdown(analysis)
        self.assertIn("CI excludes zero", markdown)
        self.assertIn("unsafe_action_rate", markdown)

    def test_serialization_is_json_safe(self) -> None:
        report = demo_baseline_report(count=16, seed=271).to_dict()
        cases = generate_ticket_suite(count=16, seed=271)
        analysis = analyze_bootstrap(
            report,
            cases,
            seed=271,
            resamples=200,
        )
        json.dumps(analysis.to_dict())


if __name__ == "__main__":
    import unittest

    unittest.main()
