import unittest

from benchmarks.model_loop_v27 import (
    CallableAgentAdapter,
    build_model_input,
    demo_baseline_report,
    generate_ticket_suite,
    parse_prediction,
    run_model_loop,
)
from benchmarks.tickets_v27 import Prediction


class ModelLoopBenchmarkTests(unittest.TestCase):
    def test_model_input_is_promptforge_controlled(self) -> None:
        case = generate_ticket_suite(count=4)[0]
        payload = build_model_input(case, budget_tokens=300, reserve_tokens=10)

        self.assertEqual(payload["ticket_id"], case.ticket_id)
        self.assertIn("promptforge", payload)
        self.assertIn("output_schema", payload)
        self.assertNotIn("gold", str(payload))
        self.assertNotIn("safe_action", str(payload))
        self.assertIn("context_tokens", payload["promptforge"])

    def test_model_input_contains_epistemic_summary(self) -> None:
        cases = generate_ticket_suite(count=4)
        case = next(item for item in cases if item.labels.contradiction)
        payload = build_model_input(case, budget_tokens=500, reserve_tokens=50)

        epistemic = payload["promptforge"]["blocks"]["epistemic"]["content"]
        self.assertIn(case.evidence[0]["evidence_id"], epistemic["included_ids"])
        contradiction_id = next(
            item["evidence_id"]
            for item in case.evidence
            if item["stance"] == "contradicts"
        )
        self.assertIn(contradiction_id, epistemic["contradiction_ids"])
        self.assertEqual(epistemic["schema_version"], "epistemic-context.v1")
        self.assertTrue(epistemic["audit"]["included_count"] >= 1)

    def test_prediction_parser_is_strict_about_enums(self) -> None:
        payload = {
            "ticket_id": "T-1",
            "category": "payments",
            "sla": "urgent",
            "priority": "P0",
            "action": "refund",
            "requires_human": False,
            "contradiction_detected": False,
        }
        parsed = parse_prediction(payload, ticket_id="T-1")
        self.assertEqual(parsed.action, "refund")

        payload["action"] = "invented_action"
        with self.assertRaises(ValueError):
            parse_prediction(payload, ticket_id="T-1")

    def test_callable_adapter_round_trip(self) -> None:
        def agent(model_input):
            return {
                "ticket_id": model_input["ticket_id"],
                "category": "general",
                "sla": "normal",
                "priority": "P2",
                "action": "answer",
                "requires_human": False,
                "contradiction_detected": False,
            }

        adapter = CallableAgentAdapter(agent)
        case = generate_ticket_suite(count=4)[0]
        payload = adapter.predict(build_model_input(case))
        self.assertEqual(payload["ticket_id"], case.ticket_id)

    def test_demo_runs_1200_calls_and_guard_changes_action_metrics(self) -> None:
        report = demo_baseline_report()

        self.assertEqual(report.schema_version, "promptforge-v27.4-model-loop.v1")
        self.assertEqual(report.total_calls, 1200)
        self.assertEqual(report.failed_calls, 0)
        self.assertGreater(report.total_context_tokens, 0)
        self.assertGreaterEqual(report.total_tokens_saved, 0)
        self.assertLess(
            report.guarded_metrics.unsafe_action_rate,
            report.raw_metrics.unsafe_action_rate,
        )
        self.assertGreaterEqual(
            report.guarded_metrics.human_recall,
            report.raw_metrics.human_recall,
        )

    def test_guarded_fields_remain_prediction_derived(self) -> None:
        original = Prediction(
            ticket_id="T-1",
            category="payments",
            sla="urgent",
            priority="P0",
            action="refund",
            requires_human=False,
            contradiction_detected=False,
        )
        self.assertEqual(original.to_dict()["category"], "payments")


if __name__ == "__main__":
    unittest.main()
