import unittest

from promptforge import (
    ComplexContextController,
    ContextEpisode,
    ContextTopologyProfiler,
    NearestEpisodeRouter,
)


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "task": {
                "id": "T-1",
                "action": "review",
                "notes": "noise",
            },
            "user": {
                "id": "U-1",
                "preferences": {
                    "language": "es",
                },
            },
            "metadata": {"request_id": "R-1"},
        }
        self.profile = ContextTopologyProfiler().profile(
            self.data,
            ["task.id", "task.action", "user.id"],
        )

    def _candidates(self):
        return [
            {
                "arm_id": "selection_only",
                "context_chars": 80,
                "estimated_tokens": 20,
                "required_values_preserved": True,
            },
            {
                "arm_id": "selection_representation_B",
                "context_chars": 120,
                "estimated_tokens": 30,
                "required_values_preserved": True,
            },
            {
                "arm_id": "noop",
                "context_chars": 260,
                "estimated_tokens": 65,
                "required_values_preserved": True,
            },
        ]

    def test_episodic_memory_wins_when_case_is_not_novel(self):
        episodes = [
            ContextEpisode(
                "train-1",
                "family-train",
                "selection_representation_B",
                1.0,
                self.profile.to_dict(),
            )
        ]
        route = NearestEpisodeRouter(
            features=("node_count", "max_depth", "boundary_pressure"),
        ).fit(episodes).route(self.profile)

        decision = ComplexContextController(
            novelty_threshold=0.01,
        ).decide(
            profile=self.profile,
            candidates=self._candidates(),
            memory_route=route,
        )

        self.assertEqual(decision.action, "select")
        self.assertEqual(decision.strategy, "selection_representation_B")
        self.assertEqual(decision.source, "episodic_memory")
        self.assertLessEqual(decision.novelty_distance, 0.01)

    def test_structural_fallback_when_memory_is_novel(self):
        route = type(
            "Route",
            (),
            {
                "strategy": "selection_representation_B",
                "nearest_distance": 99.0,
            },
        )()

        decision = ComplexContextController(
            novelty_threshold=3.0,
        ).decide(
            profile=self.profile,
            candidates=self._candidates(),
            memory_route=route,
        )

        self.assertEqual(decision.source, "structural_regime")
        self.assertEqual(decision.strategy, "selection_only")

    def test_trajectory_control_overrides_memory(self):
        episodes = [
            ContextEpisode(
                "train-1",
                "family-train",
                "selection_representation_B",
                1.0,
                self.profile.to_dict(),
            )
        ]
        route = NearestEpisodeRouter(
            features=("node_count", "max_depth", "boundary_pressure"),
        ).fit(episodes).route(self.profile)

        trajectory = {
            "cost": 10.0,
            "accepted": 0,
            "rejected": 1,
        }
        decision = ComplexContextController().decide(
            profile=self.profile,
            candidates=self._candidates(),
            memory_route=route,
            trajectory=__import__("promptforge").ContextTrajectoryMonitor().observe(
                [
                    {"cost": 10.0, "accepted": 1, "rejected": 0},
                    {"cost": 10.0, "accepted": 0, "rejected": 1},
                    {"cost": 10.0, "accepted": 0, "rejected": 1},
                    {"cost": 10.0, "accepted": 0, "rejected": 1},
                ]
            ),
            current_strategy="selection_only",
            current_cost=10.0,
            remaining_budget_fraction=0.5,
        )

        self.assertIn(decision.action, {"intensify", "stop", "continue"})


if __name__ == "__main__":
    unittest.main()
