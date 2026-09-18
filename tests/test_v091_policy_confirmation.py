import json
import unittest

from harness.runner import v091_policy_confirmation as module


class V091PolicyConfirmationTests(unittest.TestCase):

    def test_constants(self):
        self.assertEqual(module.TASK_ID, "T003")
        self.assertEqual(module.MODEL_ID, "deepseek")
        self.assertEqual(module.REPETITIONS, 40)
        self.assertEqual(module.EXPECTED_RUNS, 120)

    def test_schedule_integrity(self):
        schedule = json.loads(
            module.SCHEDULE_FILE.read_text(
                encoding="utf-8"
            )
        )

        module.validate_environment(schedule)

        self.assertEqual(
            len(schedule["rows"]),
            120,
        )

    def test_conditions(self):
        schedule = json.loads(
            module.SCHEDULE_FILE.read_text(
                encoding="utf-8"
            )
        )

        observed = {
            row["condition_id"]
            for row in schedule["rows"]
        }

        self.assertEqual(
            observed,
            {
                "noop_compact",
                "pairs_compact",
                "pairs_bare_compact",
            },
        )

    def test_only_t003(self):
        schedule = json.loads(
            module.SCHEDULE_FILE.read_text(
                encoding="utf-8"
            )
        )

        self.assertTrue(
            all(
                row["task_id"] == "T003"
                for row in schedule["rows"]
            )
        )

    def test_no_flat(self):
        schedule = json.loads(
            module.SCHEDULE_FILE.read_text(
                encoding="utf-8"
            )
        )

        self.assertTrue(
            all(
                row["condition_id"] != "flat_compact"
                for row in schedule["rows"]
            )
        )


if __name__ == "__main__":
    unittest.main()