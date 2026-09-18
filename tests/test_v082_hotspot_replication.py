import json
import subprocess
import sys
import unittest
from pathlib import Path


class V082HotspotReplicationTests(unittest.TestCase):

    def test_schedule(self):

        root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        path = (
            root
            / "harness"
            / "runner"
            / "v082_hotspot_schedule.json"
        )

        document = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            document["task_id"],
            "T003",
        )

        self.assertEqual(
            document["model_id"],
            "deepseek",
        )

        self.assertEqual(
            document["arms"],
            [
                "noop",
                "representation",
            ],
        )

        self.assertEqual(
            document["repetitions"],
            30,
        )

        self.assertEqual(
            document["scheduled_runs"],
            60,
        )

        positions = [
            row[
                "global_execution_position"
            ]
            for row in document[
                "runs"
            ]
        ]

        self.assertEqual(
            positions,
            list(
                range(60)
            ),
        )

    def test_runner_compiles(self):

        root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        runner = (
            root
            / "harness"
            / "runner"
            / "v082_hotspot_replication.py"
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "py_compile",
                str(runner),
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=result.stderr,
        )

    def test_dry_run(self):

        root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "harness.runner.v082_hotspot_replication",
                "--dry-run",
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=result.stdout + result.stderr,
        )

        self.assertIn(
            "V082_DRY_RUN=PASS",
            result.stdout,
        )

        self.assertIn(
            "V082_API_CALLS=0",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()