import unittest

from benchmarks.tickets_v27 import (
    CATEGORIES,
    Prediction,
    evaluate_predictions,
    generate_ticket_suite,
    naive_baseline,
)


class TicketBenchmarkTests(unittest.TestCase):
    def test_default_suite_has_1200_balanced_cases(self) -> None:
        cases = generate_ticket_suite()
        self.assertEqual(len(cases), 1200)
        counts = {
            category: sum(case.labels.category == category for case in cases)
            for category in CATEGORIES
        }
        self.assertEqual(counts, {category: 300 for category in CATEGORIES})

    def test_suite_is_deterministic_for_seed(self) -> None:
        first = generate_ticket_suite(seed=271)
        second = generate_ticket_suite(seed=271)
        self.assertEqual(first, second)

    def test_model_input_does_not_leak_gold_labels(self) -> None:
        case = generate_ticket_suite()[0]
        payload = case.model_input()
        self.assertNotIn("gold", payload)
        self.assertNotIn("safe_action", str(payload))
        self.assertNotIn("requires_human", str(payload))

    def test_red_team_dimensions_are_present(self) -> None:
        cases = generate_ticket_suite()
        self.assertTrue(any(case.labels.contradiction for case in cases))
        self.assertTrue(any(case.labels.redundant for case in cases))
        self.assertTrue(any(case.labels.irrelevant for case in cases))
        self.assertTrue(any(case.labels.historical_context for case in cases))
        self.assertTrue(any(case.labels.requires_human for case in cases))
        self.assertTrue(any(case.labels.sla == "urgent" for case in cases))

    def test_perfect_predictions_score_one(self) -> None:
        cases = generate_ticket_suite()
        predictions = tuple(
            Prediction(
                ticket_id=case.ticket_id,
                category=case.labels.category,
                sla=case.labels.sla,
                priority=case.labels.priority,
                action=case.labels.safe_action,
                requires_human=case.labels.requires_human,
                contradiction_detected=case.labels.contradiction,
            )
            for case in cases
        )
        metrics = evaluate_predictions(cases, predictions)
        self.assertEqual(metrics.covered_cases, 1200)
        self.assertEqual(metrics.category_accuracy, 1.0)
        self.assertEqual(metrics.sla_accuracy, 1.0)
        self.assertEqual(metrics.priority_accuracy, 1.0)
        self.assertEqual(metrics.action_accuracy, 1.0)
        self.assertEqual(metrics.human_recall, 1.0)
        self.assertEqual(metrics.contradiction_recall, 1.0)
        self.assertEqual(metrics.unsafe_action_rate, 0.0)

    def test_naive_baseline_exposes_safety_failure(self) -> None:
        cases = generate_ticket_suite()
        metrics = evaluate_predictions(cases, naive_baseline(cases))
        self.assertEqual(metrics.covered_cases, 1200)
        self.assertGreater(metrics.unsafe_action_rate, 0.0)
        self.assertLess(metrics.human_recall, 1.0)
        self.assertLess(metrics.contradiction_recall, 1.0)


if __name__ == "__main__":
    unittest.main()
