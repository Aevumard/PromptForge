import unittest

from harness.agent import prepare


class AgentFacadeTests(unittest.TestCase):

    def test_prepare_works_without_knowing_arm(self):
        result = prepare("T003")

        self.assertEqual(
            result["schema_version"],
            "agent-prepare.v1",
        )
        self.assertEqual(result["task_id"], "T003")
        self.assertEqual(
            result["policy"],
            "minimal_serialized_context",
        )
        self.assertEqual(
            result["selected_arm"],
            "selection_only",
        )
        self.assertEqual(result["context_chars"], 48)
        self.assertEqual(
            result["context"],
            {
                "entity": "A-17",
                "score": 0.87,
                "status": "stable",
            },
        )
        self.assertEqual(
            result["excluded"],
            ["trace", "commentary"],
        )

    def test_prepare_returns_auditable_candidates(self):
        result = prepare("T003")

        self.assertEqual(len(result["candidates"]), 8)
        self.assertEqual(
            {candidate["arm_id"] for candidate in result["candidates"]},
            {
                "noop",
                "selection_only",
                "representation_only",
                "representation_A",
                "representation_B",
                "selection_representation",
                "selection_representation_A",
                "selection_representation_B",
            },
        )

        for candidate in result["candidates"]:
            self.assertTrue(
                candidate["required_values_preserved"],
                candidate["arm_id"],
            )

    def test_minimal_policy_selects_smallest_context(self):
        result = prepare("T003")

        selected_size = result["context_chars"]

        self.assertEqual(
            selected_size,
            min(
                candidate["context_chars"]
                for candidate in result["candidates"]
            ),
        )

    def test_explicit_arm_is_available_without_search(self):
        result = prepare(
            "T003",
            arm_id="selection_representation_B",
        )

        self.assertEqual(
            result["policy"],
            "explicit_arm",
        )
        self.assertEqual(
            result["selected_arm"],
            "selection_representation_B",
        )
        self.assertTrue(result["required_values_preserved"])
        self.assertTrue(result["validation"]["passed"])

    def test_unknown_policy_fails(self):
        with self.assertRaises(ValueError):
            prepare("T003", policy="magic_optimization")

    def test_unknown_arm_fails(self):
        with self.assertRaises(ValueError):
            prepare("T003", arm_id="not_an_arm")

    def test_policy_and_arm_are_not_ambiguous(self):
        with self.assertRaises(ValueError):
            prepare(
                "T003",
                policy="something_else",
                arm_id="selection_only",
            )


if __name__ == "__main__":
    unittest.main()