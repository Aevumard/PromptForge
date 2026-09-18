import unittest

from harness.agent_context import (
    compile_for_agent,
    compile_task_by_id,
    get_task,
    load_task_suite,
)


class AgentContextTests(unittest.TestCase):

    def test_public_suite_contains_four_tasks(self):
        tasks = load_task_suite()
        self.assertEqual(
            [task["task_id"] for task in tasks],
            ["T001", "T002", "T003", "T004"],
        )

    def test_selection_compiles_all_public_tasks(self):
        for task_id in ("T001", "T002", "T003", "T004"):
            result = compile_task_by_id(task_id, "selection_only")
            self.assertTrue(result["required_values_preserved"])
            self.assertEqual(
                set(result["included"]),
                set(result["required"]),
            )

    def test_t003_selection_removes_known_noise(self):
        result = compile_task_by_id("T003", "selection_only")
        self.assertEqual(
            result["value"],
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
        self.assertEqual(result["context_chars"], 48)

    def test_generic_representation_is_lossless(self):
        task = get_task("T003")
        result = compile_for_agent(task, "representation_only")

        expected_fields = [
            {"key": key, "value": value}
            for key, value in task["data"].items()
        ]

        self.assertEqual(
            result["value"],
            {"fields": expected_fields},
        )
        self.assertTrue(result["required_values_preserved"])
        self.assertTrue(result["validation"]["passed"])

    def test_selection_generic_representation_preserves_values(self):
        result = compile_task_by_id("T003", "selection_representation")
        self.assertTrue(result["required_values_preserved"])
        self.assertTrue(result["validation"]["passed"])

    def test_all_public_arms_preserve_required_values(self):
        task = get_task("T003")
        arms = (
            "noop",
            "selection_only",
            "representation_only",
            "representation_A",
            "representation_B",
            "selection_representation",
            "selection_representation_A",
            "selection_representation_B",
        )

        for arm_id in arms:
            result = compile_for_agent(task, arm_id)
            self.assertTrue(
                result["required_values_preserved"],
                arm_id,
            )
            self.assertTrue(
                result["validation"]["passed"],
                arm_id,
            )

    def test_representation_b_round_trip(self):
        result = compile_task_by_id(
            "T003",
            "selection_representation_B",
        )

        self.assertEqual(
            result["value"],
            {
                "pairs": [
                    ["entity", "A-17"],
                    ["score", 0.87],
                    ["status", "stable"],
                ]
            },
        )

        self.assertTrue(result["validation"]["passed"])

    def test_unknown_task_fails(self):
        with self.assertRaises(ValueError):
            get_task("DOES_NOT_EXIST")

    def test_unknown_arm_fails(self):
        task = get_task("T003")
        with self.assertRaises(ValueError):
            compile_for_agent(task, "not_a_real_arm")


if __name__ == "__main__":
    unittest.main()