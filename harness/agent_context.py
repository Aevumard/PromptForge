from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from harness.runner.transforms import (
    V06_ARM_SEQUENCES,
    compile_v06_arm,
    decode_representation_a,
    decode_representation_b,
)

ROOT = Path(__file__).resolve().parents[1]
TASK_FILE = ROOT / "tasks" / "suite.json"


def load_task_suite() -> list[dict[str, Any]]:
    """Load the public PromptForge task suite."""
    document = json.loads(TASK_FILE.read_text(encoding="utf-8"))
    tasks = document.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("tasks must be a list")
    return deepcopy(tasks)


def get_task(task_id: str) -> dict[str, Any]:
    matches = [
        task
        for task in load_task_suite()
        if isinstance(task, dict) and task.get("task_id") == task_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one task with task_id={task_id}; found {len(matches)}"
        )
    return deepcopy(matches[0])


def _decode_compiled_value(compiled: dict[str, Any]) -> dict[str, Any]:
    sequence = compiled.get("transform_sequence", [])
    value = compiled["value"]

    # The frozen transforms map the generic "representation" stage
    # to the same lossless fields representation implemented by
    # representation_A. Decode both names accordingly.
    if "representation_A" in sequence or "representation" in sequence:
        return decode_representation_a(value)

    if "representation_B" in sequence:
        return decode_representation_b(value)

    if not isinstance(value, dict):
        raise ValueError("compiled context value is not a mapping")

    return dict(value)


def validate_required_values(
    task: dict[str, Any],
    compiled: dict[str, Any],
) -> dict[str, Any]:
    decoded = _decode_compiled_value(compiled)
    mismatches: dict[str, Any] = {}

    for key in task["required"]:
        expected = task["data"].get(key)

        if key not in decoded:
            mismatches[key] = {
                "expected": expected,
                "actual": None,
                "reason": "missing",
            }
            continue

        actual = decoded[key]

        if actual != expected:
            mismatches[key] = {
                "expected": expected,
                "actual": actual,
                "reason": "value_mismatch",
            }

    return {
        "passed": not mismatches,
        "required_count": len(task["required"]),
        "mismatches": mismatches,
    }


def compile_for_agent(
    task: dict[str, Any],
    arm_id: str = "selection_only",
) -> dict[str, Any]:
    """Compile task context into an agent-facing artifact."""
    if arm_id not in V06_ARM_SEQUENCES:
        raise ValueError(f"unknown PromptForge arm: {arm_id}")

    context = {
        "required": list(task["required"]),
        "data": deepcopy(task["data"]),
    }

    compiled = compile_v06_arm(context, arm_id)
    validation = validate_required_values(task, compiled)

    if not validation["passed"]:
        raise ValueError(
            "required values were not preserved: "
            + json.dumps(
                validation["mismatches"],
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    serialized_context = json.dumps(
        compiled["value"],
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return {
        "schema_version": "agent-context.v1",
        "task_id": task["task_id"],
        "task_family": task["task_family"],
        "arm_id": compiled["arm_id"],
        "transform_id": compiled["transform_id"],
        "transform_sequence": list(compiled["transform_sequence"]),
        "included": list(compiled["included"]),
        "excluded": list(compiled["excluded"]),
        "required": list(task["required"]),
        "value": deepcopy(compiled["value"]),
        "serialized_context": serialized_context,
        "context_chars": len(serialized_context),
        "required_values_preserved": True,
        "validation": validation,
    }


def compile_task_by_id(
    task_id: str,
    arm_id: str = "selection_only",
) -> dict[str, Any]:
    """Load a public task and compile it for an agent."""
    return compile_for_agent(get_task(task_id), arm_id=arm_id)