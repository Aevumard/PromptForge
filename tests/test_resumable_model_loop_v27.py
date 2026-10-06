import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.model_loop_v27 import CallableAgentAdapter
from benchmarks.resumable_model_loop_v27 import (
    CHECKPOINT_SCHEMA,
    run_resumable_model_loop,
)
from benchmarks.tickets_v27 import generate_ticket_suite


class ResumableModelLoopTests(TestCase):
    def test_checkpoint_resume_skips_successes_and_retries_failures(self) -> None:
        cases = generate_ticket_suite(count=4)

        with TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "run.jsonl"
            first_calls = []

            def flaky_first_pass(model_input):
                ticket_id = model_input["ticket_id"]
                first_calls.append(ticket_id)
                if len(first_calls) > 2:
                    raise RuntimeError("simulated interruption")
                return {
                    "ticket_id": ticket_id,
                    "category": "payments",
                    "sla": "urgent",
                    "priority": "P0",
                    "action": "refund",
                    "requires_human": False,
                    "contradiction_detected": False,
                }

            first = run_resumable_model_loop(
                cases,
                CallableAgentAdapter(flaky_first_pass),
                checkpoint_path=checkpoint,
                apply_guard=False,
                fsync_each_record=False,
            )
            self.assertEqual(first.raw_metrics.covered_cases, 2)

            persisted = checkpoint.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(persisted), 4)
            self.assertTrue(all("elapsed_ms" in json.loads(line) for line in persisted))
            self.assertTrue(
                all(
                    json.loads(line)["schema_version"] == CHECKPOINT_SCHEMA
                    for line in persisted
                )
            )

            second_calls = []

            def second_pass(model_input):
                ticket_id = model_input["ticket_id"]
                second_calls.append(ticket_id)
                return {
                    "ticket_id": ticket_id,
                    "category": "payments",
                    "sla": "urgent",
                    "priority": "P0",
                    "action": "refund",
                    "requires_human": False,
                    "contradiction_detected": False,
                }

            second = run_resumable_model_loop(
                cases,
                CallableAgentAdapter(second_pass),
                checkpoint_path=checkpoint,
                apply_guard=False,
                fsync_each_record=False,
            )
            self.assertEqual(second.raw_metrics.covered_cases, 4)
            self.assertEqual(len(second_calls), 2)
            self.assertEqual(
                set(second_calls),
                {cases[2].ticket_id, cases[3].ticket_id},
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
