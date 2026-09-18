import unittest

from harness.runner.v061_isolation import (
    ARMS,
    ARM_COUNT,
    SCHEMES,
    build_schedule,
    validate_schedule,
)


class V061IsolationTests(unittest.TestCase):

    def test_all_schemes_have_three_field_rows(self):

        for scheme in SCHEMES:

            schedule = validate_schedule(
                scheme,
                10,
                6101,
            )

            self.assertEqual(
                len(schedule),
                40,
            )

            for row in schedule:
                self.assertEqual(
                    len(row),
                    3,
                    msg=(
                        f"{scheme}: "
                        f"invalid row {row!r}"
                    ),
                )

    def test_all_schemes_have_exact_pairs(self):

        expected = {
            (
                repetition,
                arm_id,
            )
            for repetition in range(
                1,
                11,
            )
            for arm_id in ARMS
        }

        for scheme in SCHEMES:

            schedule = validate_schedule(
                scheme,
                10,
                6101,
            )

            observed = {
                (
                    row[0],
                    row[1],
                )
                for row in schedule
            }

            self.assertEqual(
                observed,
                expected,
                msg=scheme,
            )

            self.assertEqual(
                sorted(
                    row[2]
                    for row in schedule
                ),
                list(range(40)),
                msg=scheme,
            )

    def test_interleaved_schedule(self):

        schedule = validate_schedule(
            "interleaved",
            10,
            6101,
        )

        for repetition in range(
            1,
            11,
        ):

            rows = [
                row
                for row in schedule
                if row[0] == repetition
            ]

            self.assertEqual(
                [
                    row[1]
                    for row in rows
                ],
                ARMS,
            )

            self.assertEqual(
                [
                    row[2]
                    for row in rows
                ],
                list(
                    range(
                        (repetition - 1)
                        * ARM_COUNT,
                        repetition
                        * ARM_COUNT,
                    )
                ),
            )

    def test_randomized_schedule_is_valid(self):

        schedule = validate_schedule(
            "randomized",
            10,
            6101,
        )

        for repetition in range(
            1,
            11,
        ):

            rows = [
                row
                for row in schedule
                if row[0] == repetition
            ]

            self.assertEqual(
                sorted(
                    row[1]
                    for row in rows
                ),
                sorted(ARMS),
            )

    def test_randomized_same_seed(self):

        first = build_schedule(
            "randomized",
            10,
            6101,
        )

        second = build_schedule(
            "randomized",
            10,
            6101,
        )

        self.assertEqual(
            first,
            second,
        )

    def test_randomized_different_seed(self):

        first = build_schedule(
            "randomized",
            10,
            6101,
        )

        second = build_schedule(
            "randomized",
            10,
            6102,
        )

        self.assertNotEqual(
            first,
            second,
        )

    def test_blocked_schedule(self):

        schedule = validate_schedule(
            "blocked",
            10,
            6101,
        )

        for arm_index, arm_id in enumerate(
            ARMS
        ):

            rows = [
                row
                for row in schedule
                if row[1] == arm_id
            ]

            self.assertEqual(
                [
                    row[0]
                    for row in rows
                ],
                list(range(1, 11)),
            )

            self.assertEqual(
                [
                    row[2]
                    for row in rows
                ],
                list(
                    range(
                        arm_index * 10,
                        (arm_index + 1) * 10,
                    )
                ),
            )

    def test_latin_square_balance(self):

        schedule = validate_schedule(
            "latin_square",
            10,
            6101,
        )

        first_cycle = [
            row
            for row in schedule
            if row[0] <= ARM_COUNT
        ]

        for arm_id in ARMS:

            relative_positions = sorted(
                row[2] % ARM_COUNT
                for row in first_cycle
                if row[1] == arm_id
            )

            self.assertEqual(
                relative_positions,
                list(range(ARM_COUNT)),
                msg=arm_id,
            )

    def test_latin_square_deterministic(self):

        first = build_schedule(
            "latin_square",
            10,
            6101,
        )

        second = build_schedule(
            "latin_square",
            10,
            6101,
        )

        self.assertEqual(
            first,
            second,
        )


if __name__ == "__main__":
    unittest.main()
