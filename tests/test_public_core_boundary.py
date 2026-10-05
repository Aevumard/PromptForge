import ast
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harness.agent import prepare_context as legacy_prepare_context
from promptforge import (
    POLICY_BUDGET_CONSTRAINED,
    prepare_context,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = tuple(
    sorted((ROOT / "promptforge").glob("*.py"))
)


class PublicCoreBoundaryTests(unittest.TestCase):
    def test_public_core_contains_no_harness_imports(self):
        for path in PUBLIC_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertNotIn(
                        "harness",
                        {alias.name.split(".")[0] for alias in node.names},
                    )
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotEqual(node.module.split(".")[0], "harness")

    def test_public_core_can_run_without_harness_tree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            isolated_root = Path(temp_dir) / "isolated"
            shutil.copytree(ROOT / "promptforge", isolated_root / "promptforge")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(isolated_root)
            env.pop("PYTHONSTARTUP", None)

            code = """
import importlib.util
from promptforge import prepare_context

result = prepare_context(
    {"entity": "A-17", "noise": "x"},
    ["entity"],
)

assert result["context"] == {"entity": "A-17"}
assert importlib.util.find_spec("harness") is None
"""
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=temp_dir,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                0,
                completed.stderr,
            )

    def test_public_core_works_through_top_level_package(self):
        result = prepare_context(
            {
                "user": {
                    "id": "U-7",
                    "name": "Ada",
                },
                "task": {
                    "action": "review",
                    "commentary": "noise",
                },
            },
            ["user.id", "task.action"],
        )

        self.assertEqual(
            result["context"],
            {
                "user": {"id": "U-7"},
                "task": {"action": "review"},
            },
        )
        self.assertEqual(result["selected_arm"], "selection_only")
        self.assertTrue(result["required_values_preserved"])
        self.assertTrue(result["validation"]["passed"])

    def test_public_core_matches_legacy_generic_contract(self):
        data = {
            "user": {
                "id": "U-7",
                "name": "Ada",
            },
            "task": {
                "action": "review",
                "priority": "high",
                "commentary": "noise",
            },
        }
        required = ["user.id", "task.action", "task.priority"]

        legacy = legacy_prepare_context(
            data,
            required,
            task_id="PARITY-001",
            task_family="parity",
        )
        current = prepare_context(
            data,
            required,
            task_id="PARITY-001",
            task_family="parity",
        )

        self.assertEqual(current["task_id"], legacy["task_id"])
        self.assertEqual(current["task_family"], legacy["task_family"])
        self.assertEqual(current["policy"], legacy["policy"])
        self.assertEqual(current["selected_arm"], legacy["selected_arm"])
        self.assertEqual(current["context"], legacy["context"])
        self.assertEqual(current["serialized_context"], legacy["serialized_context"])
        self.assertEqual(current["context_chars"], legacy["context_chars"])
        self.assertEqual(current["estimated_tokens"], legacy["estimated_tokens"])
        self.assertEqual(current["included_paths"], legacy["included_paths"])
        self.assertEqual(current["excluded_paths"], legacy["excluded_paths"])
        self.assertEqual(current["required_values_preserved"], legacy["required_values_preserved"])
        self.assertEqual(current["validation"], legacy["validation"])
        self.assertEqual(
            [
                (item["arm_id"], item["context_chars"], item["estimated_tokens"])
                for item in current["candidates"]
            ],
            [
                (item["arm_id"], item["context_chars"], item["estimated_tokens"])
                for item in legacy["candidates"]
            ],
        )

    def test_public_core_retains_budget_contract(self):
        result = prepare_context(
            {
                "entity": "A-17",
                "score": 0.87,
                "status": "stable",
                "trace": "noise",
            },
            ["entity", "score", "status"],
            policy=POLICY_BUDGET_CONSTRAINED,
            budget_tokens=12,
        )

        self.assertEqual(result["policy"], POLICY_BUDGET_CONSTRAINED)
        self.assertTrue(result["budget_satisfied"])
        self.assertLessEqual(result["estimated_tokens"], 12)


if __name__ == "__main__":
    unittest.main()
