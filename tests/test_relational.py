import unittest

from promptforge import (
    ContextRelation,
    ContextRelationProfiler,
)


class RelationalTests(unittest.TestCase):
    def test_profile_is_deterministic_and_detects_hub(self):
        relations = [
            ContextRelation("task", "goal", "targets"),
            ContextRelation("task", "user", "owned_by"),
            ContextRelation("task", "evidence", "supported_by"),
        ]
        profile = ContextRelationProfiler().profile(relations)

        self.assertEqual(profile.schema_version, "context-relational.v1")
        self.assertEqual(profile.relation_count, 3)
        self.assertEqual(profile.relation_nodes, 4)
        self.assertEqual(profile.relation_components, 1)
        self.assertEqual(profile.relation_max_degree, 3)
        self.assertEqual(profile.relation_kind_count, 3)
        self.assertIn("targets", profile.relation_kinds)
        self.assertGreater(profile.relation_hub_ratio, 1.0)
        self.assertEqual(profile.to_dict(), ContextRelationProfiler().profile(relations).to_dict())

    def test_disconnected_components_are_explicit(self):
        profile = ContextRelationProfiler().profile(
            [
                {"source": "a", "target": "b"},
                {"source": "c", "target": "d"},
            ]
        )
        self.assertEqual(profile.relation_nodes, 4)
        self.assertEqual(profile.relation_components, 2)

    def test_duplicate_relations_are_not_double_counted(self):
        relation = ContextRelation("a", "b", "related")
        profile = ContextRelationProfiler().profile([relation, relation])
        self.assertEqual(profile.relation_count, 1)
        self.assertEqual(profile.relation_max_degree, 1)

    def test_self_relation_is_rejected(self):
        with self.assertRaises(ValueError):
            ContextRelation("a", "a")


if __name__ == "__main__":
    unittest.main()
