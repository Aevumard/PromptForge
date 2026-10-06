import unittest

from promptforge import (
    ContextRoutingModeScore,
    ContextRoutingPolicyEvidence,
    ContextRoutingPolicyHistory,
)


def evidence(version, mode):
    return ContextRoutingPolicyEvidence(
        version=version,
        scores=(
            ContextRoutingModeScore(mode="nearest", episodes=4, families=2, oracle_agreement_rate=0.5, mean_absolute_regret=1.0, mean_relative_regret=0.4),
            ContextRoutingModeScore(mode="credit", episodes=4, families=2, oracle_agreement_rate=0.75, mean_absolute_regret=0.5, mean_relative_regret=0.2),
        ),
        selected_mode=mode,
    )


class RoutingPolicyHistoryTests(unittest.TestCase):
    def test_records_bounded_immutable_history(self):
        history = ContextRoutingPolicyHistory(max_entries=2)
        self.assertEqual(history.record(evidence(1, "nearest")), 1)
        self.assertEqual(history.record(evidence(2, "credit")), 2)
        self.assertEqual(history.record(evidence(3, "credit")), 3)

        snapshot = history.snapshot()
        self.assertEqual(snapshot.version, 3)
        self.assertEqual(
            [item.version for item in snapshot.evidences],
            [2, 3],
        )
        self.assertEqual(history.latest.version, 3)

    def test_consensus_prefers_recent_tie_break(self):
        history = ContextRoutingPolicyHistory()
        history.record(evidence(1, "nearest"))
        history.record(evidence(2, "credit"))

        self.assertEqual(history.select_mode(), "credit")
        self.assertEqual(
            history.select_mode(window=1),
            "credit",
        )

    def test_consensus_can_be_stabilized_by_recent_window(self):
        history = ContextRoutingPolicyHistory()
        history.record(evidence(1, "nearest"))
        history.record(evidence(2, "nearest"))
        history.record(evidence(3, "credit"))
        history.record(evidence(4, "nearest"))

        self.assertEqual(history.select_mode(window=3), "nearest")
        self.assertEqual(history.select_mode(window=1), "nearest")

    def test_stability_is_descriptive_and_deterministic(self):
        history = ContextRoutingPolicyHistory()
        history.record(evidence(10, "nearest"))
        history.record(evidence(11, "nearest"))
        history.record(evidence(12, "credit"))
        history.record(evidence(13, "credit"))

        stability = history.stability()
        self.assertEqual(stability.history_version, 4)
        self.assertEqual(stability.observations, 4)
        self.assertEqual(stability.latest_evidence_version, 13)
        self.assertEqual(stability.latest_mode, "credit")
        self.assertEqual(stability.switch_count, 1)
        self.assertAlmostEqual(stability.stability_rate, 2.0 / 3.0)
        self.assertEqual(
            stability.mode_counts,
            (("nearest", 2), ("credit", 2)),
        )


    def test_health_requires_enough_stable_observations(self):
        history = ContextRoutingPolicyHistory()
        history.record(evidence(1, "nearest"))
        health = history.health(min_observations=2)

        self.assertFalse(health.stable)
        self.assertTrue(health.refresh_recommended)
        self.assertEqual(health.mode, "nearest")

    def test_health_detects_policy_oscillation(self):
        history = ContextRoutingPolicyHistory()
        history.record(evidence(1, "nearest"))
        history.record(evidence(2, "credit"))
        history.record(evidence(3, "nearest"))

        health = history.health(min_observations=3, min_stability=0.5)
        self.assertFalse(health.stable)
        self.assertTrue(health.refresh_recommended)
        self.assertEqual(health.switch_count, 2)
        self.assertAlmostEqual(health.stability_rate, 0.0)

    def test_health_accepts_stable_recent_window(self):
        history = ContextRoutingPolicyHistory()
        history.record(evidence(1, "nearest"))
        history.record(evidence(2, "credit"))
        history.record(evidence(3, "credit"))
        history.record(evidence(4, "credit"))

        health = history.health(window=3, min_observations=2, min_stability=1.0)
        self.assertTrue(health.stable)
        self.assertFalse(health.refresh_recommended)
        self.assertEqual(health.mode, "credit")
        self.assertEqual(health.stability_rate, 1.0)

    def test_backward_policy_versions_are_rejected(self):
        history = ContextRoutingPolicyHistory()
        history.record(evidence(5, "nearest"))

        with self.assertRaises(ValueError):
            history.record(evidence(4, "credit"))

    def test_snapshot_is_isolated_from_future_records(self):
        history = ContextRoutingPolicyHistory()
        history.record(evidence(1, "nearest"))
        snapshot = history.snapshot()

        history.record(evidence(2, "credit"))

        self.assertEqual(len(snapshot.evidences), 1)
        self.assertEqual(snapshot.latest.selected_mode, "nearest")
        self.assertEqual(history.snapshot().latest.selected_mode, "credit")


if __name__ == "__main__":
    unittest.main()
