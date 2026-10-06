import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.ablation_v27 import (
    AblationVariant,
    DEFAULT_VARIANTS,
    render_ablation_markdown,
    run_ablation_suite,
)
from benchmarks.model_loop_v27 import CallableAgentAdapter, build_model_input
from benchmarks.tickets_v27 import generate_ticket_suite


class AblationSuiteTests(TestCase):
    def test_default_variants_is_three_layer_ablation(self) -> None:
        self.assertEqual(
            [item.name for item in DEFAULT_VARIANTS],
            [
                "baseline_budget_only",
                "epistemic_budget",
                "epistemic_budget_guarded",
            ],
        )
        self.assertFalse(DEFAULT_VARIANTS[0].include_epistemic)
        self.assertTrue(DEFAULT_VARIANTS[1].include_epistemic)
        self.assertTrue(DEFAULT_VARIANTS[2].apply_guard)

    def test_model_input_can_ablate_epistemic_block(self) -> None:
        case = generate_ticket_suite(count=4)[0]
        with_epistemic = build_model_input(case, budget_tokens=500)
        without_epistemic = build_model_input(
            case,
            budget_tokens=500,
            include_epistemic=False,
        )

        self.assertIn(
            "epistemic",
            with_epistemic["promptforge"]["context"],
        )
        self.assertNotIn(
            "epistemic",
            without_epistemic["promptforge"]["context"],
        )

    def test_ablation_suite_runs_each_variant_with_own_checkpoint(self) -> None:
        cases = generate_ticket_suite(count=4, seed=271)
        calls = []

        def agent(model_input):
            calls.append(
                (
                    model_input["ticket_id"],
                    "epistemic" in model_input["promptforge"]["context"],
                )
            )
            return {
                "ticket_id": model_input["ticket_id"],
                "category": "payments",
                "sla": "urgent",
                "priority": "P0",
                "action": "refund",
                "requires_human": False,
                "contradiction_detected": False,
            }

        with TemporaryDirectory() as temp_dir:
            report = run_ablation_suite(
                cases,
                CallableAgentAdapter(agent),
                checkpoint_dir=Path(temp_dir),
                fsync_each_record=False,
            )

            self.assertEqual(report.schema_version, "promptforge-v27.10-ablation-suite.v1")
            self.assertEqual(len(report.results), 3)
            self.assertEqual(len(calls), 12)

            checkpoints = sorted(Path(temp_dir).glob("*.checkpoint.jsonl"))
            self.assertEqual(len(checkpoints), 3)
            for checkpoint in checkpoints:
                self.assertEqual(len(checkpoint.read_text(encoding="utf-8").splitlines()), 4)

            with_epistemic = [flag for _, flag in calls]
            self.assertEqual(sum(with_epistemic), 8)

    def test_ablation_report_serializes_and_renders(self) -> None:
        cases = generate_ticket_suite(count=4)
        adapter = CallableAgentAdapter(
            lambda model_input: {
                "ticket_id": model_input["ticket_id"],
                "category": "general",
                "sla": "normal",
                "priority": "P2",
                "action": "answer",
                "requires_human": False,
                "contradiction_detected": False,
            }
        )
        with TemporaryDirectory() as temp_dir:
            report = run_ablation_suite(
                cases,
                adapter,
                checkpoint_dir=Path(temp_dir),
                fsync_each_record=False,
            )
            payload = report.to_dict()
            self.assertEqual(payload["suite"]["count"], 4)
            json.dumps(payload)
            markdown = render_ablation_markdown(report)
            self.assertIn("baseline_budget_only", markdown)
            self.assertIn("epistemic_budget_guarded", markdown)
            self.assertIn("unsafe rate", markdown.lower())
            self.assertIn("p50 ms", markdown.lower())
            self.assertIn("p95 ms", markdown.lower())

    def test_variant_rejects_invalid_budget(self) -> None:
        with self.assertRaises(ValueError):
            AblationVariant(
                name="bad",
                budget_tokens=10,
                reserve_tokens=10,
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
