import unittest

from promptforge import ContextExplorationController


class TestExplorationController(unittest.TestCase):
    def test_coverage_gap_selects_unobserved_alternative(self):
        controller = ContextExplorationController(
            novelty_threshold=3.0,
            exploration_budget=2,
        )
        decision = controller.decide(
            experience_version=4,
            candidate_order=("preferred", "alternate", "other"),
            observed_strategies=("preferred", "preferred"),
            preferred_strategy="preferred",
        )

        self.assertTrue(decision.required)
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.reason, "coverage_gap")
        self.assertEqual(decision.target_strategy, "alternate")

    def test_novel_case_requests_exploration(self):
        controller = ContextExplorationController(novelty_threshold=2.0)
        decision = controller.decide(
            experience_version=1,
            candidate_order=("a", "b"),
            observed_strategies=("a", "b"),
            preferred_strategy="a",
            novelty_distance=4.0,
        )

        self.assertTrue(decision.required)
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.reason, "novel_case")
        self.assertEqual(decision.target_strategy, "b")

    def test_policy_refresh_requests_exploration(self):
        controller = ContextExplorationController()
        decision = controller.decide(
            experience_version=5,
            candidate_order=("a", "b"),
            observed_strategies=("a", "b"),
            preferred_strategy="a",
            policy_refresh_required=True,
        )

        self.assertTrue(decision.required)
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.reason, "policy_refresh")

    def test_cooldown_blocks_exploration(self):
        controller = ContextExplorationController(
            min_interval=3,
            exploration_budget=2,
        )
        first = controller.decide(
            experience_version=1,
            candidate_order=("a", "b"),
            observed_strategies=("a",),
            preferred_strategy="a",
            novelty_distance=5.0,
        )
        self.assertTrue(first.eligible)
        controller.record_exploration(experience_version=1)

        second = controller.decide(
            experience_version=2,
            candidate_order=("a", "b"),
            observed_strategies=("a",),
            preferred_strategy="a",
            novelty_distance=5.0,
        )
        self.assertTrue(second.required)
        self.assertFalse(second.eligible)
        self.assertEqual(second.reason, "exploration_cooldown")
        self.assertEqual(second.cooldown_remaining, 2)

    def test_budget_blocks_exploration(self):
        controller = ContextExplorationController(
            exploration_budget=1,
        )
        first = controller.decide(
            experience_version=1,
            candidate_order=("a", "b"),
            observed_strategies=("a",),
            preferred_strategy="a",
            force=True,
        )
        self.assertTrue(first.eligible)
        controller.record_exploration(experience_version=1)

        second = controller.decide(
            experience_version=3,
            candidate_order=("a", "b"),
            observed_strategies=("a",),
            preferred_strategy="a",
            force=True,
        )
        self.assertTrue(second.required)
        self.assertFalse(second.eligible)
        self.assertEqual(second.reason, "exploration_budget_exhausted")

    def test_no_alternative_is_never_exploration(self):
        controller = ContextExplorationController()
        decision = controller.decide(
            experience_version=0,
            candidate_order=("a",),
            preferred_strategy="a",
            force=True,
        )

        self.assertTrue(decision.required)
        self.assertFalse(decision.eligible)
        self.assertEqual(decision.reason, "no_alternative_strategy")
        self.assertIsNone(decision.target_strategy)

    def test_record_refresh_state_is_serializable(self):
        controller = ContextExplorationController(
            novelty_threshold=1.5,
            min_interval=2,
            exploration_budget=5,
        )
        controller.record_exploration(experience_version=3)
        payload = controller.to_dict()

        self.assertEqual(payload["novelty_threshold"], 1.5)
        self.assertEqual(payload["min_interval"], 2)
        self.assertEqual(payload["exploration_budget"], 5)
        self.assertEqual(payload["exploration_count"], 1)
        self.assertEqual(payload["last_exploration_experience_version"], 3)


if __name__ == "__main__":
    unittest.main()
