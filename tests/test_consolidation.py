import unittest

from promptforge import (
    ContextEpisode,
    ContextExperienceStore,
    ContextMemoryConsolidator,
    ContextTopologyProfiler,
)


class ConsolidationTests(unittest.TestCase):
    def setUp(self):
        profile = ContextTopologyProfiler().profile(
            {"task": {"id": "T", "action": "review"}},
            ["task.id"],
        )
        self.profile = profile.to_dict()

    def episode(self, i, family, strategy):
        return ContextEpisode(
            f"e{i}", family, strategy, float(i), self.profile
        )

    def test_replay_is_bounded_deterministic_and_diverse(self):
        episodes = [
            self.episode(1, "A", "s1"),
            self.episode(2, "A", "s1"),
            self.episode(3, "B", "s2"),
            self.episode(4, "C", "s3"),
            self.episode(5, "C", "s3"),
            self.episode(6, "D", "s4"),
        ]
        policy = ContextMemoryConsolidator(recent_fraction=0.5)
        first = policy.replay(episodes, max_items=4)
        second = policy.replay(episodes, max_items=4)

        self.assertEqual(first.episodes, second.episodes)
        self.assertEqual(len(first.episodes), 4)
        self.assertGreaterEqual(first.family_count, 3)
        self.assertGreaterEqual(first.strategy_count, 3)

    def test_store_consolidation_keeps_existing_snapshot_intact(self):
        episodes = [
            self.episode(1, "A", "s1"),
            self.episode(2, "B", "s2"),
            self.episode(3, "C", "s3"),
            self.episode(4, "D", "s4"),
        ]
        store = ContextExperienceStore(episodes)
        snapshot = store.snapshot()
        old_version = store.version

        batch = store.consolidate(max_episodes=2)

        self.assertEqual(batch.source_size, 4)
        self.assertEqual(store.size, 2)
        self.assertEqual(store.version, old_version + 1)
        self.assertEqual(len(snapshot.episodes), 4)
        self.assertEqual(snapshot.version, old_version)

    def test_consolidating_at_capacity_is_a_noop(self):
        store = ContextExperienceStore(
            [self.episode(1, "A", "s1"), self.episode(2, "B", "s2")]
        )
        before = store.version
        batch = store.consolidate(max_episodes=2)

        self.assertEqual(batch.source_size, 2)
        self.assertEqual(store.size, 2)
        self.assertEqual(store.version, before)

    def test_invalid_capacity_is_rejected(self):
        policy = ContextMemoryConsolidator()
        with self.assertRaises(ValueError):
            policy.replay([self.episode(1, "A", "s1")], max_items=0)


if __name__ == "__main__":
    unittest.main()
