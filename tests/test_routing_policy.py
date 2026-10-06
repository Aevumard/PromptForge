import unittest

from promptforge import (
    ContextEpisode,
    ContextRoutingModeScore,
    ContextRoutingPolicyEvidence,
    ContextRoutingPolicyEvaluator,
    ContextRoutingPolicySelector,
    ContextTopologyProfiler,
)


class RoutingPolicyTests(unittest.TestCase):
    def setUp(self):
        profile = ContextTopologyProfiler().profile(
            {"task": {"id": "T", "action": "review"}},
            ["task.id"],
        )
        self.profile = profile.to_dict()

    def episode(self, episode_id, family, strategy, cost, node_count):
        topology = dict(self.profile)
        topology["node_count"] = node_count
        return ContextEpisode(
            episode_id,
            family,
            strategy,
            float(cost),
            topology,
        )

    def test_policy_selector_prefers_lower_relative_regret(self):
        evidence = ContextRoutingPolicyEvidence(
            version=7,
            scores=(
                ContextRoutingModeScore("nearest", 10, 3, 0.4, 2.0, 0.5),
                ContextRoutingModeScore("credit", 10, 3, 0.7, 0.9, 0.2),
            ),
            selected_mode="nearest",
        )
        selected = ContextRoutingPolicySelector().select(evidence)

        self.assertEqual(selected, "credit")

    def test_policy_selector_falls_back_when_evidence_is_insufficient(self):
        evidence = ContextRoutingPolicyEvidence(
            version=1,
            scores=(
                ContextRoutingModeScore("nearest", 0, 0, 0.0, 0.0, 0.0),
                ContextRoutingModeScore("credit", 0, 0, 0.0, 0.0, 0.0),
            ),
            selected_mode="credit",
        )
        selected = ContextRoutingPolicySelector(min_episodes=2).select(evidence)

        self.assertEqual(selected, "nearest")

    def test_evaluator_uses_whole_family_holdout(self):
        episodes = [
            self.episode("A", "family-a", "s1", 1.0, 10),
            self.episode("A", "family-a", "s2", 3.0, 10),
            self.episode("B", "family-b", "s1", 3.0, 20),
            self.episode("B", "family-b", "s2", 1.0, 20),
            self.episode("C", "family-c", "s1", 1.0, 30),
            self.episode("C", "family-c", "s2", 2.0, 30),
        ]
        evidence = ContextRoutingPolicyEvaluator(
            features=("node_count", "max_depth"),
            top_k=4,
        ).evaluate(episodes, version=3)

        self.assertEqual(evidence.version, 3)
        self.assertEqual(evidence.selected_mode, "nearest")
        self.assertEqual(len(evidence.scores), 2)
        self.assertEqual(
            {score.mode for score in evidence.scores},
            {"nearest", "credit"},
        )
        for score in evidence.scores:
            self.assertEqual(score.episodes, 3)
            self.assertEqual(score.families, 3)

    def test_policy_evidence_is_serializable(self):
        evidence = ContextRoutingPolicyEvidence(
            version=2,
            scores=(
                ContextRoutingModeScore("nearest", 4, 2, 0.5, 1.0, 0.25),
                ContextRoutingModeScore("credit", 4, 2, 0.75, 0.5, 0.1),
            ),
            selected_mode="credit",
        )
        payload = evidence.to_dict()

        self.assertEqual(payload["version"], 2)
        self.assertEqual(payload["selected_mode"], "credit")
        self.assertEqual(payload["scores"][1]["mode"], "credit")


if __name__ == "__main__":
    unittest.main()
