import unittest

from promptforge import (
    ContextPortfolioController,
    ContextTopologyProfiler,
    ContextTrajectoryMonitor,
    HeuristicContextRegimeSelector,
    rank_context_candidates,
)


class AdaptiveContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = {
            "user": {
                "id": "U-7",
                "name": "Ada",
                "preferences": {
                    "language": "es",
                    "theme": "dark",
                },
            },
            "task": {
                "action": "review",
                "priority": "high",
                "commentary": "noise",
            },
            "metadata": {
                "request_id": "R-1",
            },
        }

    def test_profile_is_deterministic_and_tracks_required_boundary(self):
        profile = ContextTopologyProfiler().profile(
            self.data,
            ["user.id", "task.action", "missing.value"],
        )

        self.assertEqual(profile.schema_version, "context-topology.v1")
        self.assertEqual(profile.top_level_fields, 3)
        self.assertEqual(profile.required_count, 3)
        self.assertEqual(profile.required_present, 2)
        self.assertEqual(profile.required_missing, ("missing.value",))
        self.assertGreater(profile.node_count, profile.leaf_fields)
        self.assertGreaterEqual(profile.boundary_edges, 1)
        self.assertGreaterEqual(profile.estimated_tokens, 1)

        second = ContextTopologyProfiler().profile(
            self.data,
            ["user.id", "task.action", "missing.value"],
        )
        self.assertEqual(profile.to_dict(), second.to_dict())

    def test_hub_regime_is_explicit_not_hidden(self):
        data = {
            "task": {
                "a": 1,
                "b": 2,
                "c": 3,
                "d": 4,
                "e": 5,
                "f": 6,
                "g": 7,
                "h": 8,
                "i": 9,
            },
            "noise": "x",
        }
        profile = ContextTopologyProfiler().profile(data, ["task.a"])
        signature = HeuristicContextRegimeSelector().classify(profile)

        self.assertIn("hub_dominated", signature.flags)
        self.assertEqual(signature.primary_regime, "hub_dominated")

    def test_rank_candidates_respects_budget_and_regime_priority(self):
        profile = ContextTopologyProfiler().profile(
            self.data,
            ["user.id", "task.action", "task.priority"],
        )
        recommendation = HeuristicContextRegimeSelector().recommend(profile)
        candidates = [
            {
                "arm_id": "noop",
                "context_chars": 240,
                "estimated_tokens": 60,
                "required_values_preserved": True,
            },
            {
                "arm_id": "selection_only",
                "context_chars": 90,
                "estimated_tokens": 23,
                "required_values_preserved": True,
            },
            {
                "arm_id": "selection_representation_B",
                "context_chars": 120,
                "estimated_tokens": 30,
                "required_values_preserved": True,
            },
        ]

        ranked = rank_context_candidates(
            candidates,
            recommendation,
            budget_tokens=30,
        )

        self.assertEqual(ranked[0], "selection_only")
        self.assertNotIn("noop", ranked)

    def test_trajectory_monitor_detects_stagnation(self):
        monitor = ContextTrajectoryMonitor(stagnation_patience=2)
        state = monitor.observe(
            [
                {"cost": 10.0, "accepted": 1, "rejected": 0},
                {"cost": 9.0, "accepted": 1, "rejected": 0},
                {"cost": 9.0, "accepted": 0, "rejected": 1},
                {"cost": 9.0, "accepted": 0, "rejected": 1},
            ]
        )

        self.assertEqual(state.state, "stagnating")
        self.assertEqual(state.stagnation_length, 2)
        self.assertGreaterEqual(state.recent_gain, 0.0)

    def test_portfolio_controller_switches_on_observed_probe(self):
        trajectory = ContextTrajectoryMonitor().observe(
            [
                {"cost": 10.0, "accepted": 1, "rejected": 0},
                {"cost": 9.0, "accepted": 1, "rejected": 0},
            ]
        )
        decision = ContextPortfolioController().decide(
            trajectory=trajectory,
            current_strategy="selection_only",
            current_cost=9.0,
            probe_scores={"selection_representation_B": 8.5},
            remaining_budget_fraction=0.50,
        )

        self.assertEqual(decision.action, "switch")
        self.assertEqual(decision.target_strategy, "selection_representation_B")
        self.assertIn("bounded probe", decision.reason)


if __name__ == "__main__":
    unittest.main()
