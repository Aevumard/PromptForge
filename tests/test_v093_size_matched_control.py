import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "harness" / "runner" / "v093_size_matched_control.py"
SCHEDULE = ROOT / "harness" / "runner" / "v093_size_matched_schedule.json"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "v093_size_matched_control",
        RUNNER,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


class V093Tests(unittest.TestCase):

    def test_constants(self):
        module = load_module()

        self.assertEqual(
            module.TASKS,
            ["T003"],
        )

        self.assertEqual(
            module.CONDITIONS,
            [
                "noop_compact",
                "noop_padded",
                "pairs_compact",
                "pairs_bare_compact",
            ],
        )

        self.assertEqual(
            module.REPETITIONS,
            15,
        )

        self.assertEqual(
            module.EXPECTED_RUNS,
            240,
        )

    def test_schedule_count(self):
        payload = json.loads(
            SCHEDULE.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            payload["scheduled_runs"],
            240,
        )

        self.assertEqual(
            len(payload["rows"]),
            240,
        )

    def test_schedule_positions(self):
        payload = json.loads(
            SCHEDULE.read_text(
                encoding="utf-8"
            )
        )

        observed = [
            int(
                row["global_execution_position"]
            )
            for row in payload["rows"]
        ]

        self.assertEqual(
            observed,
            list(range(240)),
        )

    def test_repetition_balance(self):
        payload = json.loads(
            SCHEDULE.read_text(
                encoding="utf-8"
            )
        )

        for repetition in range(1, 16):

            rows = [
                row
                for row in payload["rows"]
                if int(
                    row["repetition"]
                ) == repetition
            ]

            self.assertEqual(
                len(rows),
                16,
            )

            self.assertEqual(
                {
                    row["variant_id"]
                    for row in rows
                },
                {
                    "canonical_t003",
                    "noise_long",
                    "key_order",
                    "content_shift",
                },
            )

            self.assertEqual(
                {
                    row["condition_id"]
                    for row in rows
                },
                {
                    "noop_compact",
                    "noop_padded",
                    "pairs_compact",
                    "pairs_bare_compact",
                },
            )

    def test_no_flat(self):
        payload = json.loads(
            SCHEDULE.read_text(
                encoding="utf-8"
            )
        )

        self.assertTrue(
            all(
                row["condition_id"]
                != "flat_compact"
                for row in payload["rows"]
            )
        )

    def test_canonical_preflight(self):
        module = load_module()

        task = module.preflight_task()

        observed = (
            module.validate_canonical_preflight(
                task
            )
        )

        self.assertEqual(
            observed["noop_compact"],
            (85, 307),
        )

        self.assertEqual(
            observed["noop_padded"],
            (105, 327),
        )

        self.assertEqual(
            observed["pairs_compact"],
            (105, 327),
        )

        self.assertEqual(
            observed["pairs_bare_compact"],
            (95, 317),
        )

    def test_exact_size_matching(self):
        module = load_module()

        task = module.preflight_task()

        padded = module.compile_condition(
            task,
            "noop_padded",
        )[1]

        pairs = module.compile_condition(
            task,
            "pairs_compact",
        )[1]

        padded_chars = len(
            json.dumps(
                padded,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

        pairs_chars = len(
            json.dumps(
                pairs,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

        self.assertEqual(
            padded_chars,
            pairs_chars,
        )

    def test_required_values_preserved(self):
        module = load_module()

        task = module.preflight_task()

        noop = module.compile_condition(
            task,
            "noop_compact",
        )[1]

        padded = module.compile_condition(
            task,
            "noop_padded",
        )[1]

        for key in task["required"]:
            self.assertEqual(
                noop[key],
                padded[key],
            )


if __name__ == "__main__":
    unittest.main()