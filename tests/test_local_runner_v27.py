import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from benchmarks.local_runner_v27 import _progress
from benchmarks.model_loop_v27 import CallableAgentAdapter
from benchmarks.replicates_v27 import run_replicate_experiment
from benchmarks.tickets_v27 import generate_ticket_suite


class LocalRunnerTests(TestCase):
    def test_progress_messages_are_human_readable(self):
        _progress(
            "replicate_complete",
            {
                "index": 1,
                "total": 3,
                "coverage": 1.0,
                "guarded_action_accuracy": 0.75,
                "guarded_unsafe_action_rate": 0.0,
            },
        )

    def test_progress_callback_is_emitted(self):
        events = []

        def agent(model_input):
            return {
                "ticket_id": model_input["ticket_id"],
                "category": "payments",
                "sla": "urgent",
                "priority": "P0",
                "action": "refund",
                "requires_human": False,
                "contradiction_detected": False,
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            run_replicate_experiment(
                generate_ticket_suite(count=4, seed=271),
                lambda: CallableAgentAdapter(agent),
                replicates=2,
                output_dir=Path(temp_dir),
                include_epistemic=False,
                apply_guard=False,
                fsync_each_record=False,
                progress_callback=lambda event, payload: events.append((event, payload)),
            )

        self.assertEqual(
            [item[0] for item in events],
            [
                "replicate_start",
                "replicate_complete",
                "replicate_start",
                "replicate_complete",
            ],
        )
