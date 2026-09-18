from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "harness" / "runner"
if str(RUNNER) not in sys.path:
    sys.path.insert(0, str(RUNNER))

import v08_plan


class TestV08Preregistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = v08_plan.build_schedule_document()
        cls.positions = cls.document["positions"]

    def test_exact_total_runs(self) -> None:
        self.assertEqual(self.document["total_runs"], 360)
        self.assertEqual(len(self.positions), 360)
        self.assertEqual(
            [row["global_execution_position"] for row in self.positions],
            list(range(360)),
        )

    def test_exact_arm_set(self) -> None:
        self.assertEqual(set(self.document["arms"]), {"noop", "representation"})
        self.assertNotIn("representation_A", self.document["arms"])
        self.assertNotIn("representation_B", self.document["arms"])
        self.assertTrue(all(row["arm_id"] in {"noop", "representation"} for row in self.positions))

    def test_excluded_arms_are_explicit(self) -> None:
        self.assertEqual(
            self.document["excluded_arms"],
            ["representation_A", "representation_B"],
        )

    def test_cell_balance(self) -> None:
        for task in v08_plan.TASKS:
            for model in v08_plan.MODELS:
                for arm in v08_plan.ARMS:
                    count = sum(
                        1
                        for row in self.positions
                        if row["task_id"] == task
                        and row["model_id"] == model
                        and row["arm_id"] == arm
                    )
                    self.assertEqual(count, 15, (task, model, arm))

    def test_global_model_distribution(self) -> None:
        for model in v08_plan.MODELS:
            count = sum(1 for row in self.positions if row["model_id"] == model)
            self.assertEqual(count, 120, model)

    def test_global_arm_distribution(self) -> None:
        self.assertEqual(
            sum(1 for row in self.positions if row["arm_id"] == "noop"),
            180,
        )
        self.assertEqual(
            sum(1 for row in self.positions if row["arm_id"] == "representation"),
            180,
        )

    def test_task_distribution(self) -> None:
        for task in v08_plan.TASKS:
            self.assertEqual(
                sum(1 for row in self.positions if row["task_id"] == task),
                90,
            )

    def test_repetition_block_balance(self) -> None:
        for repetition in range(1, v08_plan.REPETITIONS + 1):
            rep_rows = [r for r in self.positions if r["repetition"] == repetition]
            self.assertEqual(len(rep_rows), 24)
            for model in v08_plan.MODELS:
                self.assertEqual(sum(1 for r in rep_rows if r["model_id"] == model), 8)
            for arm in v08_plan.ARMS:
                self.assertEqual(sum(1 for r in rep_rows if r["arm_id"] == arm), 12)
            for task in v08_plan.TASKS:
                self.assertEqual(sum(1 for r in rep_rows if r["task_id"] == task), 6)

    def test_schedule_hash_is_self_consistent(self) -> None:
        recomputed = v08_plan.compute_schedule_sha256(self.positions)
        self.assertEqual(self.document["schedule_sha256"], recomputed)


if __name__ == "__main__":
    unittest.main(verbosity=2)