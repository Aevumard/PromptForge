import json
import unittest

from harness.runner import v092_perturbation_validation as module


class V092Tests(unittest.TestCase):

    def setUp(self):
        self.runner = module.load_v09_module()
        module.install_execution_surface(
            self.runner
        )

    def test_constants(self):
        self.assertEqual(
            module.TASK_ID,
            "T003",
        )

        self.assertEqual(
            module.TASK_FAMILY,
            "structured_analysis",
        )

        self.assertEqual(
            module.MODEL_ID,
            "deepseek",
        )

        self.assertEqual(
            module.REPETITIONS,
            15,
        )

        self.assertEqual(
            module.EXPECTED_RUNS,
            180,
        )

    def test_variants(self):
        self.assertEqual(
            set(module.VARIANTS),
            {
                "canonical_t003",
                "noise_long",
                "key_order",
                "content_shift",
            },
        )

    def test_conditions(self):
        self.assertEqual(
            set(module.CONDITIONS),
            {
                "noop_compact",
                "pairs_compact",
                "pairs_bare_compact",
            },
        )

    def test_schedule(self):
        schedule = json.loads(
            module.SCHEDULE_FILE.read_text(
                encoding="utf-8"
            )
        )

        module.validate_environment(
            schedule
        )

    def test_compact_execution_surface(self):
        variants = module.load_variants()

        task = module.build_task(
            variants[
                "canonical_t003"
            ]
        )

        expected = {
            "noop_compact": (
                "noop",
                85,
                307,
            ),
            "pairs_compact": (
                "representation_B",
                105,
                327,
            ),
            "pairs_bare_compact": (
                "representation_B",
                95,
                317,
            ),
        }

        for condition, expected_value in expected.items():

            compiled, serialized = (
                self.runner.compile_condition(
                    task,
                    condition,
                )
            )

            self.assertEqual(
                compiled[
                    "transform_id"
                ],
                expected_value[0],
            )

            context_chars = len(
                json.dumps(
                    serialized,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                )
            )

            prompt = (
                self.runner.build_prompt(
                    task,
                    serialized,
                )
            )

            self.assertEqual(
                context_chars,
                expected_value[1],
            )

            self.assertEqual(
                len(prompt),
                expected_value[2],
            )

    def test_canonical_preflight(self):
        module.validate_canonical_preflight(
            self.runner,
            module.load_variants(),
        )

    def test_no_flat(self):
        schedule = json.loads(
            module.SCHEDULE_FILE.read_text(
                encoding="utf-8"
            )
        )

        self.assertTrue(
            all(
                row["condition_id"]
                != "flat_compact"
                for row in schedule["rows"]
            )
        )


if __name__ == "__main__":
    unittest.main()