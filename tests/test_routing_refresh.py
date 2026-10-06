import unittest

from promptforge import (
    ContextRoutingModeScore,
    ContextRoutingPolicyEvidence,
    ContextRoutingPolicyHistory,
    ContextRoutingPolicyRefreshController,
)


class TestPolicyRefreshController(unittest.TestCase):
    def evidence(self, version, mode="credit"):
        return ContextRoutingPolicyEvidence(
            version=version,
            scores=(
                ContextRoutingModeScore("nearest", 6, 3, 0.4, 2.0, 0.8),
                ContextRoutingModeScore("credit", 6, 3, 0.8, 0.5, 0.2),
            ),
            selected_mode=mode,
        )

    def test_stale_history_is_refresh_eligible(self):
        history = ContextRoutingPolicyHistory()
        history.record(self.evidence(1))
        history.record(self.evidence(2))

        controller = ContextRoutingPolicyRefreshController(
            min_observations=2,
            min_stability=1.0,
            max_age=1,
        )
        decision = controller.decide(history.snapshot(), experience_version=5)

        self.assertTrue(decision.refresh_required)
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.reason, "stale_policy")
        self.assertEqual(decision.freshness_age, 3)

    def test_cooldown_blocks_second_refresh(self):
        history = ContextRoutingPolicyHistory()
        history.record(self.evidence(1))
        history.record(self.evidence(2))

        controller = ContextRoutingPolicyRefreshController(
            min_observations=2,
            min_stability=1.0,
            max_age=1,
            min_interval=3,
            refresh_budget=2,
        )
        first = controller.decide(history.snapshot(), experience_version=5)
        self.assertTrue(first.eligible)

        controller.record_refresh(experience_version=5)
        second = controller.decide(history.snapshot(), experience_version=6)

        self.assertTrue(second.refresh_required)
        self.assertFalse(second.eligible)
        self.assertEqual(second.reason, "refresh_cooldown")
        self.assertEqual(second.cooldown_remaining, 2)

    def test_budget_exhaustion_blocks_refresh(self):
        history = ContextRoutingPolicyHistory()
        history.record(self.evidence(1))
        history.record(self.evidence(2))

        controller = ContextRoutingPolicyRefreshController(
            min_observations=2,
            min_stability=1.0,
            max_age=1,
            refresh_budget=1,
        )
        first = controller.decide(history.snapshot(), experience_version=5)
        self.assertTrue(first.eligible)
        controller.record_refresh(experience_version=5)

        second = controller.decide(history.snapshot(), experience_version=8)
        self.assertTrue(second.refresh_required)
        self.assertFalse(second.eligible)
        self.assertEqual(second.reason, "refresh_budget_exhausted")

    def test_insufficient_evidence_does_not_trigger_execution(self):
        history = ContextRoutingPolicyHistory()
        history.record(self.evidence(1))

        controller = ContextRoutingPolicyRefreshController(
            min_observations=2,
            min_stability=1.0,
            max_age=0,
        )
        decision = controller.decide(history.snapshot(), experience_version=4)

        self.assertTrue(decision.refresh_required)
        self.assertFalse(decision.eligible)
        self.assertEqual(decision.reason, "insufficient_evidence")

    def test_explicit_policy_freshness_has_a_refresh_gate(self):
        controller = ContextRoutingPolicyRefreshController(
            min_observations=1,
            max_age=2,
        )
        fresh = controller.decide_for_policy(
            policy_version=8,
            experience_version=10,
        )
        stale = controller.decide_for_policy(
            policy_version=8,
            experience_version=11,
        )

        self.assertFalse(fresh.refresh_required)
        self.assertFalse(fresh.eligible)
        self.assertEqual(fresh.reason, "healthy")
        self.assertTrue(stale.refresh_required)
        self.assertTrue(stale.eligible)
        self.assertEqual(stale.reason, "stale_policy")

    def test_record_refresh_rejects_active_cooldown(self):
        controller = ContextRoutingPolicyRefreshController(
            min_interval=2,
            refresh_budget=2,
        )
        controller.record_refresh(experience_version=4)
        with self.assertRaises(RuntimeError):
            controller.record_refresh(experience_version=5)

    def test_controller_state_is_serializable(self):
        controller = ContextRoutingPolicyRefreshController(
            min_observations=2,
            min_stability=0.75,
            max_age=4,
            min_interval=3,
            refresh_budget=5,
        )
        payload = controller.to_dict()

        self.assertEqual(payload["min_observations"], 2)
        self.assertEqual(payload["min_stability"], 0.75)
        self.assertEqual(payload["max_age"], 4)
        self.assertEqual(payload["min_interval"], 3)
        self.assertEqual(payload["refresh_budget"], 5)
        self.assertEqual(payload["refresh_count"], 0)


if __name__ == "__main__":
    unittest.main()
