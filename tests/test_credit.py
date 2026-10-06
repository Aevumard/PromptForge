import unittest

from promptforge import (
    ContextEpisode,
    ContextExperienceStore,
    ContextMemoryConsolidator,
    ContextMemoryCreditPolicy,
    ContextTopologyProfiler,
)


class MemoryCreditTests(unittest.TestCase):
    def setUp(self):
        profile = ContextTopologyProfiler().profile(
            {"task": {"id": "T", "action": "review"}},
            ["task.id"],
        )
        self.profile = profile.to_dict()

    def episode(self, episode_id, family, strategy, cost):
        return ContextEpisode(
            episode_id,
            family,
            strategy,
            float(cost),
            self.profile,
        )

    def test_credit_is_deterministic_and_decays_by_recency(self):
        episodes = [
            self.episode("A-1", "A", "s1", 2.0),
            self.episode("A-2", "A", "s1", 2.0),
            self.episode("B-1", "B", "s2", 1.0),
        ]
        policy = ContextMemoryCreditPolicy(half_life=2.0)
        first = policy.assess(episodes)
        second = policy.assess(episodes)

        self.assertEqual(first, second)
        self.assertEqual(first[0].evidence_count, 2)
        self.assertGreater(first[1].recency_weight, first[0].recency_weight)
        self.assertGreater(first[1].credit, first[0].credit)

    def test_comparable_observations_reward_lower_observed_cost(self):
        episodes = [
            self.episode("X", "A", "slow", 5.0),
            self.episode("X", "A", "fast", 2.0),
            self.episode("Y", "A", "slow", 6.0),
            self.episode("Y", "A", "fast", 2.5),
            self.episode("Z", "A", "slow", 7.0),
        ]
        policy = ContextMemoryCreditPolicy(half_life=100.0)
        credits = policy.assess(episodes)

        slow = [item for item in credits if item.strategy == "slow"][0]
        fast = [item for item in credits if item.strategy == "fast"][0]

        self.assertEqual(fast.win_count, 2)
        self.assertEqual(fast.loss_count, 0)
        self.assertEqual(slow.win_count, 0)
        self.assertEqual(slow.loss_count, 2)
        self.assertGreater(fast.observed_win_rate, slow.observed_win_rate)
        self.assertGreater(fast.credit, slow.credit)

    def test_snapshot_credit_isolated_from_later_observations(self):
        store = ContextExperienceStore(
            [self.episode("A", "A", "s1", 3.0)]
        )
        snapshot = store.snapshot()
        before = snapshot.credit()

        store.record(self.episode("B", "A", "s1", 1.0))

        self.assertEqual(snapshot.credit(), before)
        self.assertEqual(len(snapshot.episodes), 1)
        self.assertEqual(len(store.credit()), 2)

    def test_credit_aware_consolidation_uses_credit_after_coverage(self):
        episodes = [
            self.episode("A-1", "A", "s1", 2.0),
            self.episode("A-2", "A", "s1", 2.0),
            self.episode("B-1", "B", "s2", 10.0),
            self.episode("C-1", "C", "s3", 1.0),
            self.episode("C-2", "C", "s3", 1.0),
            self.episode("A-3", "A", "s1", 2.0),
        ]
        policy = ContextMemoryCreditPolicy(half_life=100.0)
        batch = ContextMemoryConsolidator(
            recent_fraction=0.50,
            credit_policy=policy,
        ).replay(episodes, max_items=4)

        self.assertEqual(len(batch.episodes), 4)
        self.assertGreaterEqual(batch.family_count, 3)
        self.assertGreaterEqual(batch.strategy_count, 3)
        self.assertIn("A-3", {item.episode_id for item in batch.episodes})


if __name__ == "__main__":
    unittest.main()
