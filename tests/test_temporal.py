import unittest

from promptforge import (
    ContextEpisode,
    ContextMemoryAwareRouter,
    ContextMemoryTemporalPolicy,
    ContextTopologyProfiler,
)


class TemporalMemoryGuardTests(unittest.TestCase):
    def setUp(self):
        self.profile = ContextTopologyProfiler().profile(
            {"task": {"id": "T", "action": "review"}},
            ["task.id"],
        ).to_dict()

    def episode(self, episode_id, family, strategy, cost, source="trusted"):
        return ContextEpisode(
            episode_id,
            family,
            strategy,
            float(cost),
            self.profile,
            source=source,
        )

    def test_future_memory_is_excluded_at_historical_as_of_boundary(self):
        episodes = [
            self.episode("past", "family-a", "safe", 2.0),
            self.episode("current", "family-b", "safe", 2.0),
            self.episode("future", "family-c", "dangerous", 0.1),
        ]
        policy = ContextMemoryTemporalPolicy()
        result = policy.assess(episodes, as_of_index=1)

        self.assertEqual(result.as_of_index, 1)
        self.assertEqual(result.included_ids, ("past", "current"))
        self.assertEqual(result.future_excluded_ids, ("future",))

    def test_stale_memory_is_hard_excluded(self):
        episodes = [
            self.episode("old", "family-a", "legacy", 0.1),
            self.episode("recent", "family-b", "safe", 2.0),
            self.episode("current", "family-c", "safe", 2.0),
        ]
        policy = ContextMemoryTemporalPolicy(max_age=1)
        result = policy.assess(episodes)

        self.assertEqual(result.stale_excluded_ids, ("old",))
        self.assertEqual(result.included_ids, ("recent", "current"))

    def test_blocked_source_cannot_win_by_structural_closeness(self):
        episodes = [
            self.episode(
                "untrusted",
                "family-a",
                "poisoned",
                0.1,
                source="untrusted",
            ),
            self.episode(
                "trusted",
                "family-b",
                "safe",
                2.0,
                source="trusted",
            ),
        ]
        router = ContextMemoryAwareRouter(
            features=("node_count", "max_depth"),
            top_k=2,
            temporal_policy=ContextMemoryTemporalPolicy(
                blocked_sources=("untrusted",),
            ),
        ).fit(episodes)

        route = router.route(self.profile)

        self.assertEqual(route.strategy, "safe")
        self.assertEqual(route.temporal_source_excluded_ids, ("untrusted",))
        self.assertTrue(route.temporal_policy_applied)
        self.assertEqual(route.evidence_count, 1)

    def test_blocked_memory_cannot_inflate_allowed_memory_credit(self):
        base = [
            self.episode("trusted", "family-a", "safe", 1.0, source="trusted"),
            self.episode("current", "family-b", "safe", 1.0, source="trusted"),
        ]
        poisoned = [
            self.episode(
                "poisoned",
                "family-a",
                "safe",
                0.01,
                source="untrusted",
            ),
            *base,
        ]
        policy = ContextMemoryTemporalPolicy(blocked_sources=("untrusted",))

        baseline = ContextMemoryAwareRouter(
            features=("node_count", "max_depth"),
            top_k=2,
            temporal_policy=policy,
        ).fit(base).route(self.profile)

        guarded = ContextMemoryAwareRouter(
            features=("node_count", "max_depth"),
            top_k=3,
            temporal_policy=policy,
        ).fit(poisoned).route(self.profile)

        self.assertEqual(guarded.strategy, baseline.strategy)
        self.assertAlmostEqual(
            guarded.selected_credit,
            baseline.selected_credit,
        )

    def test_router_fails_closed_when_no_memory_is_temporally_admissible(self):
        episodes = [
            self.episode("old-1", "family-a", "legacy", 0.1),
            self.episode("old-2", "family-b", "legacy", 0.2),
        ]
        router = ContextMemoryAwareRouter(
            top_k=2,
            temporal_policy=ContextMemoryTemporalPolicy(
                blocked_sources=("trusted",),
            ),
        ).fit(episodes)

        with self.assertRaises(ValueError):
            router.route(self.profile)

    def test_as_of_index_requires_explicit_temporal_policy(self):
        episodes = [
            self.episode("a", "family-a", "safe", 1.0),
            self.episode("b", "family-b", "safe", 1.0),
        ]
        router = ContextMemoryAwareRouter(
            top_k=2,
        ).fit(episodes)

        with self.assertRaises(ValueError):
            router.route(self.profile, as_of_index=0)

    def test_temporal_result_is_deterministic(self):
        episodes = [
            self.episode("a", "family-a", "safe", 1.0),
            self.episode("b", "family-b", "safe", 1.0),
            self.episode("c", "family-c", "safe", 1.0),
        ]
        policy = ContextMemoryTemporalPolicy(
            max_age=1,
            blocked_sources=("untrusted",),
        )

        first = policy.assess(episodes, as_of_index=1)
        second = policy.assess(episodes, as_of_index=1)

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.to_dict()["schema_version"], "memory-temporal.v1")
        self.assertEqual(
            first.future_excluded_ids,
            ("c",),
        )


if __name__ == "__main__":
    unittest.main()
