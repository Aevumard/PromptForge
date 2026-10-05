import unittest

from promptforge import (
    ContextEpisode,
    NearestEpisodeRouter,
    episode_oracle,
    leave_one_family_out,
    routing_summary,
)


class EpisodicRoutingTests(unittest.TestCase):
    def _episodes(self):
        return [
            ContextEpisode(
                "a1", "family-a", "selection_only", 1.0,
                {"node_count": 10, "max_depth": 2, "boundary_pressure": 0.1},
            ),
            ContextEpisode(
                "a1", "family-a", "representation_B", 2.0,
                {"node_count": 10, "max_depth": 2, "boundary_pressure": 0.1},
            ),
            ContextEpisode(
                "b1", "family-b", "selection_only", 2.5,
                {"node_count": 30, "max_depth": 5, "boundary_pressure": 0.4},
            ),
            ContextEpisode(
                "b1", "family-b", "representation_B", 1.5,
                {"node_count": 30, "max_depth": 5, "boundary_pressure": 0.4},
            ),
            ContextEpisode(
                "c1", "family-c", "selection_only", 1.2,
                {"node_count": 11, "max_depth": 2, "boundary_pressure": 0.1},
            ),
            ContextEpisode(
                "c1", "family-c", "representation_B", 2.1,
                {"node_count": 11, "max_depth": 2, "boundary_pressure": 0.1},
            ),
        ]

    def test_nearest_episode_router_uses_observed_cases(self):
        episodes = self._episodes()
        router = NearestEpisodeRouter(
            features=("node_count", "max_depth", "boundary_pressure"),
        ).fit(episodes[:4])

        self.assertEqual(router.predict(episodes[4].topology), "selection_only")
        self.assertEqual(
            router.training_episode_ids,
            ("a1", "b1"),
        )

    def test_episode_oracle_is_explicit_observed_outcome(self):
        oracle = episode_oracle(self._episodes())
        self.assertEqual(oracle["a1"], "selection_only")
        self.assertEqual(oracle["b1"], "representation_B")

    def test_leave_one_family_out_prevents_episode_leakage(self):
        episodes = self._episodes()
        folds = leave_one_family_out(
            episodes,
            router_factory=lambda: NearestEpisodeRouter(
                features=("node_count", "max_depth", "boundary_pressure"),
            ),
        )

        self.assertEqual(
            [family for family, _ in folds],
            ["family-a", "family-b", "family-c"],
        )

        for family, evaluations in folds:
            self.assertTrue(evaluations)
            for evaluation in evaluations:
                self.assertEqual(evaluation.family_id, family)

    def test_summary_has_no_hidden_oracle_claim(self):
        evaluations = leave_one_family_out(self._episodes())[0][1]
        summary = routing_summary(evaluations)

        self.assertEqual(summary["episodes"], float(len(evaluations)))
        self.assertGreaterEqual(summary["oracle_agreement_rate"], 0.0)
        self.assertGreaterEqual(summary["mean_relative_regret"], 0.0)


if __name__ == "__main__":
    unittest.main()
