import unittest

from promptforge import (
    ContextEpisode,
    ContextExperienceStore,
    ContextMemoryAwareRouter,
    ContextMemoryCreditPolicy,
    ContextTopologyProfiler,
)


class MemoryAwareRoutingTests(unittest.TestCase):
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

    def test_repeated_credit_can_overrule_one_nearest_memory(self):
        episodes = [
            self.episode("a1", "family-a", "sparse", 3.0),
            self.episode("b1", "family-a", "dense", 1.0),
            self.episode("b2", "family-a", "dense", 1.0),
            self.episode("b3", "family-a", "dense", 1.0),
        ]
        router = ContextMemoryAwareRouter(
            features=("node_count", "max_depth"),
            top_k=4,
            credit_policy=ContextMemoryCreditPolicy(half_life=100.0),
        ).fit(episodes)

        route = router.route(self.profile)

        self.assertEqual(route.strategy, "dense")
        self.assertEqual(route.candidate_count, 4)
        self.assertEqual(route.nearest_episode_id, "b3")
        self.assertGreater(route.selected_credit, 0.0)
        self.assertEqual(route.strategy_scores[0][0], "dense")

    def test_router_is_deterministic(self):
        episodes = [
            self.episode("a1", "family-a", "sparse", 3.0),
            self.episode("b1", "family-a", "dense", 1.0),
            self.episode("b2", "family-a", "dense", 1.0),
        ]
        first = ContextMemoryAwareRouter(
            features=("node_count", "max_depth"),
            top_k=3,
        ).fit(episodes).route(self.profile)
        second = ContextMemoryAwareRouter(
            features=("node_count", "max_depth"),
            top_k=3,
        ).fit(episodes).route(self.profile)

        self.assertEqual(first, second)
        self.assertEqual(first.strategy, "dense")

    def test_snapshot_memory_aware_route_is_frozen(self):
        store = ContextExperienceStore(
            [self.episode("a1", "family-a", "sparse", 3.0)]
        )
        snapshot = store.snapshot()
        before = snapshot.memory_aware_route(self.profile)

        store.record(self.episode("b1", "family-a", "dense", 1.0))

        self.assertEqual(snapshot.memory_aware_route(self.profile), before)
        self.assertEqual(store.memory_aware_route(self.profile).strategy, "dense")

    def test_invalid_top_k_is_rejected(self):
        with self.assertRaises(ValueError):
            ContextMemoryAwareRouter(top_k=0)


if __name__ == "__main__":
    unittest.main()
