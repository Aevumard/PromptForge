import unittest

from promptforge import (
    ContextEpisode,
    ContextExperienceStore,
    ContextTopologyProfiler,
)


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.profile = ContextTopologyProfiler().profile(
            {"task": {"id": "T-1", "action": "review"}},
            ["task.id", "task.action"],
        )

    def _episode(self, episode_id, strategy, cost):
        return ContextEpisode(
            episode_id,
            "family-a",
            strategy,
            cost,
            self.profile.to_dict(),
        )

    def test_snapshot_is_frozen_against_later_learning(self):
        store = ContextExperienceStore()
        store.record(self._episode("e1", "selection_only", 10.0))

        snapshot = store.snapshot()
        store.record(self._episode("e2", "selection_representation_B", 1.0))

        self.assertEqual(snapshot.version, 1)
        self.assertEqual(len(snapshot.episodes), 1)
        self.assertEqual(store.version, 2)

    def test_snapshot_routes_only_from_frozen_evidence(self):
        store = ContextExperienceStore(
            [self._episode("e1", "selection_only", 10.0)]
        )
        snapshot = store.snapshot()
        store.record(
            self._episode("e2", "selection_representation_B", 1.0)
        )

        route = snapshot.route(
            self.profile,
            features=("node_count", "max_depth"),
        )

        self.assertEqual(route.strategy, "selection_only")
        self.assertEqual(route.nearest_episode_id, "e1")

    def test_empty_snapshot_cannot_route(self):
        store = ContextExperienceStore()
        with self.assertRaises(ValueError):
            store.snapshot().route(self.profile)

    def test_record_requires_episode(self):
        store = ContextExperienceStore()
        with self.assertRaises(TypeError):
            store.record("not-an-episode")


if __name__ == "__main__":
    unittest.main()
