import unittest

from benchmarks.tickets_v27 import generate_ticket_suite, naive_baseline
from benchmarks.ticket_guard import guard_predictions, guard_prediction


class TicketGuardTests(unittest.TestCase):
    def test_contradiction_blocks_naive_operational_action(self) -> None:
        cases = generate_ticket_suite()
        case = next(case for case in cases if case.labels.contradiction)
        prediction = naive_baseline([case])[0]

        decision = guard_prediction(case, prediction)

        self.assertTrue(decision.blocked)
        self.assertEqual(decision.guarded_action, "human_review")
        self.assertTrue(
            any(
                "support evidence stance incompatible" in reason
                for reason in decision.reasons
            )
            or any(
                "no action candidate passed" in reason
                for reason in decision.reasons
            )
        )

    def test_non_contradictory_action_can_pass(self) -> None:
        cases = generate_ticket_suite()
        case = next(
            case
            for case in cases
            if not case.labels.contradiction and case.labels.category == "general"
        )
        prediction = naive_baseline([case])[0]

        decision = guard_prediction(case, prediction)

        self.assertFalse(decision.blocked)
        self.assertEqual(decision.guarded_action, prediction.action)

    def test_guard_removes_contradiction_unsafe_actions(self) -> None:
        cases = generate_ticket_suite()
        baseline = naive_baseline(cases)
        guarded = guard_predictions(cases, baseline)

        contradiction_cases = [
            case for case in cases if case.labels.contradiction
        ]
        guarded_map = {item.ticket_id: item for item in guarded}

        self.assertTrue(contradiction_cases)
        self.assertTrue(
            all(
                guarded_map[case.ticket_id].action == "human_review"
                for case in contradiction_cases
            )
        )

    def test_guard_does_not_change_classification_fields(self) -> None:
        cases = generate_ticket_suite()
        baseline = naive_baseline(cases)
        guarded = guard_predictions(cases, baseline)

        for before, after in zip(baseline, guarded):
            self.assertEqual(before.category, after.category)
            self.assertEqual(before.sla, after.sla)
            self.assertEqual(before.priority, after.priority)
