import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.model_loop_v27 import CallableAgentAdapter
from benchmarks.replicates_v27 import (
    SCHEMA_VERSION,
    render_markdown,
    run_replicate_experiment,
    summarize,
)
from benchmarks.tickets_v27 import generate_ticket_suite


class ReplicateRunnerTests(TestCase):
    def test_summarize_uses_sample_standard_deviation(self) -> None:
        summary = summarize((1.0, 2.0, 3.0))
        self.assertEqual(summary.count, 3)
        self.assertEqual(summary.mean, 2.0)
        self.assertAlmostEqual(summary.std, 1.0)
        self.assertEqual(summary.minimum, 1.0)
        self.assertEqual(summary.maximum, 3.0)

    def test_replicate_runner_creates_independent_checkpoints(self) -> None:
        cases = generate_ticket_suite(count=8, seed=271)
        factory_calls = {"count": 0}

        def factory():
            factory_calls["count"] += 1
            replicate_index = factory_calls["count"]

            def agent(model_input):
                return {
                    "ticket_id": model_input["ticket_id"],
                    "category": "payments",
                    "sla": "urgent",
                    "priority": "P0",
                    "action": "refund" if replicate_index % 2 else "answer",
                    "requires_human": False,
                    "contradiction_detected": False,
                }

            return CallableAgentAdapter(agent)

        with TemporaryDirectory() as temp_dir:
            experiment = run_replicate_experiment(
                cases,
                factory,
                replicates=3,
                output_dir=Path(temp_dir),
                include_epistemic=False,
                apply_guard=False,
                fsync_each_record=False,
            )

            self.assertEqual(experiment.schema_version, SCHEMA_VERSION)
            self.assertEqual(experiment.replicate_count, 3)
            self.assertEqual(factory_calls["count"], 3)
            checkpoints = sorted(
                Path(temp_dir).glob("replicate-*/model_run.checkpoint.jsonl")
            )
            self.assertEqual(len(checkpoints), 3)
            self.assertTrue(
                all(
                    len(path.read_text(encoding="utf-8").splitlines()) == 8
                    for path in checkpoints
                )
            )
            self.assertGreater(
                experiment.pairwise_disagreement.raw_action_disagreement_rate,
                0.0,
            )
            self.assertEqual(
                experiment.metrics["raw.action_accuracy"].count,
                3,
            )

    def test_manifest_prevents_mixing_conditions(self) -> None:
        cases = generate_ticket_suite(count=4, seed=271)

        def factory():
            return CallableAgentAdapter(
                lambda model_input: {
                    "ticket_id": model_input["ticket_id"],
                    "category": "payments",
                    "sla": "urgent",
                    "priority": "P0",
                    "action": "refund",
                    "requires_human": False,
                    "contradiction_detected": False,
                }
            )

        with TemporaryDirectory() as temp_dir:
            run_replicate_experiment(
                cases,
                factory,
                replicates=2,
                output_dir=Path(temp_dir),
                budget_tokens=500,
                include_epistemic=False,
                apply_guard=False,
                fsync_each_record=False,
            )
            with self.assertRaises(ValueError):
                run_replicate_experiment(
                    cases,
                    factory,
                    replicates=2,
                    output_dir=Path(temp_dir),
                    budget_tokens=300,
                    include_epistemic=False,
                    apply_guard=False,
                    fsync_each_record=False,
                )

    def test_resume_does_not_recall_completed_replicates(self) -> None:
        cases = generate_ticket_suite(count=4, seed=271)
        calls = {"count": 0}

        def factory():
            def agent(model_input):
                calls["count"] += 1
                return {
                    "ticket_id": model_input["ticket_id"],
                    "category": "payments",
                    "sla": "urgent",
                    "priority": "P0",
                    "action": "refund",
                    "requires_human": False,
                    "contradiction_detected": False,
                }

            return CallableAgentAdapter(agent)

        with TemporaryDirectory() as temp_dir:
            run_replicate_experiment(
                cases,
                factory,
                replicates=2,
                output_dir=Path(temp_dir),
                include_epistemic=False,
                apply_guard=False,
                fsync_each_record=False,
            )
            self.assertEqual(calls["count"], 8)

            run_replicate_experiment(
                cases,
                factory,
                replicates=2,
                output_dir=Path(temp_dir),
                include_epistemic=False,
                apply_guard=False,
                fsync_each_record=False,
            )
            self.assertEqual(calls["count"], 8)

    def test_serialization_filters_secret_provider_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            experiment = run_replicate_experiment(
                generate_ticket_suite(count=4, seed=271),
                lambda: CallableAgentAdapter(
                    lambda model_input: {
                        "ticket_id": model_input["ticket_id"],
                        "category": "general",
                        "sla": "normal",
                        "priority": "P2",
                        "action": "answer",
                        "requires_human": False,
                        "contradiction_detected": False,
                    }
                ),
                replicates=2,
                output_dir=Path(temp_dir),
                include_epistemic=False,
                apply_guard=False,
                fsync_each_record=False,
                provider_metadata={
                    "endpoint": "http://localhost:8000/v1/chat/completions",
                    "api_key": "SECRET",
                },
            )
            payload = experiment.to_dict()
            json_text = json.dumps(payload)
            self.assertNotIn("SECRET", json_text)
            self.assertIn(
                "Between-replicate metrics",
                render_markdown(experiment),
            )

    def test_single_replicate_is_rejected(self) -> None:
        cases = generate_ticket_suite(count=4)
        with self.assertRaises(ValueError):
            run_replicate_experiment(
                cases,
                lambda: CallableAgentAdapter(lambda _: {}),
                replicates=1,
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
