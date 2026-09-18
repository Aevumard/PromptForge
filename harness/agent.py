from __future__ import annotations

from typing import Any, Iterable

from harness.agent_context import (
    compile_for_agent,
    get_task,
)
from harness.runner.transforms import V06_ARM_SEQUENCES


ARM_ORDER = (
    "selection_only",
    "noop",
    "representation_only",
    "representation_A",
    "representation_B",
    "selection_representation",
    "selection_representation_A",
    "selection_representation_B",
)

POLICY_MINIMAL = "minimal"
POLICY_MINIMAL_SERIALIZED_CONTEXT = "minimal_serialized_context"


def _candidate_summary(compiled: dict[str, Any]) -> dict[str, Any]:
    return {
        "arm_id": compiled["arm_id"],
        "transform_id": compiled["transform_id"],
        "transform_sequence": list(compiled["transform_sequence"]),
        "context_chars": compiled["context_chars"],
        "included": list(compiled["included"]),
        "excluded": list(compiled["excluded"]),
        "required_values_preserved": bool(
            compiled["required_values_preserved"]
        ),
    }


def _public_result(
    compiled: dict[str, Any],
    *,
    policy: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = next(
        (
            candidate
            for candidate in candidates
            if candidate["arm_id"] == "noop"
        ),
        None,
    )
    baseline_chars = (
        baseline["context_chars"]
        if baseline is not None
        else None
    )
    saved_chars = (
        baseline_chars - compiled["context_chars"]
        if baseline_chars is not None
        else None
    )
    reduction_ratio = (
        saved_chars / baseline_chars
        if baseline_chars not in (None, 0)
        else None
    )

    return {
        "schema_version": "agent-prepare.v1",
        "task_id": compiled["task_id"],
        "task_family": compiled["task_family"],
        "policy": policy,
        "selected_arm": compiled["arm_id"],
        "transform_id": compiled["transform_id"],
        "transform_sequence": list(compiled["transform_sequence"]),
        "required": list(compiled["required"]),
        "included": list(compiled["included"]),
        "excluded": list(compiled["excluded"]),
        "context": compiled["value"],
        "serialized_context": compiled["serialized_context"],
        "context_chars": compiled["context_chars"],
        "baseline_arm": "noop" if baseline is not None else None,
        "baseline_context_chars": baseline_chars,
        "context_chars_saved": saved_chars,
        "context_reduction_ratio": reduction_ratio,
        "required_values_preserved": bool(
            compiled["required_values_preserved"]
        ),
        "validation": compiled["validation"],
        "selection_basis": (
            "smallest_serialized_context"
            if policy == POLICY_MINIMAL_SERIALIZED_CONTEXT
            else "explicit_arm"
        ),
        "candidates": candidates,
    }



def _prepare_minimal(task: dict[str, Any]) -> dict[str, Any]:
    compiled_candidates: list[dict[str, Any]] = []

    for arm_id in ARM_ORDER:
        if arm_id not in V06_ARM_SEQUENCES:
            raise ValueError(
                "agent facade references unknown arm: " + arm_id
            )

        compiled_candidates.append(
            compile_for_agent(task, arm_id)
        )

    selected = min(
        enumerate(compiled_candidates),
        key=lambda item: (
            item[1]["context_chars"],
            item[0],
        ),
    )[1]

    candidates = [
        _candidate_summary(compiled)
        for compiled in compiled_candidates
    ]

    return _public_result(
        selected,
        policy=POLICY_MINIMAL_SERIALIZED_CONTEXT,
        candidates=candidates,
    )


def _prepare_task(
    task: dict[str, Any],
    *,
    policy: str,
    arm_id: str | None,
) -> dict[str, Any]:
    if arm_id is not None:
        if policy != POLICY_MINIMAL:
            raise ValueError(
                "arm_id cannot be combined with policy=" + repr(policy)
            )

        compiled = compile_for_agent(task, arm_id)

        return _public_result(
            compiled,
            policy="explicit_arm",
            candidates=[_candidate_summary(compiled)],
        )

    if policy == POLICY_MINIMAL:
        return _prepare_minimal(task)

    raise ValueError(
        "unknown agent policy: " + repr(policy)
    )


def prepare(
    task_id: str,
    *,
    policy: str = POLICY_MINIMAL,
    arm_id: str | None = None,
) -> dict[str, Any]:
    """Prepare a validated public task for direct agent consumption."""
    return _prepare_task(
        get_task(task_id),
        policy=policy,
        arm_id=arm_id,
    )


def prepare_context(
    data: dict[str, Any],
    required: Iterable[str],
    *,
    task_id: str = "adhoc",
    task_family: str = "agent_request",
    policy: str = POLICY_MINIMAL,
    arm_id: str | None = None,
) -> dict[str, Any]:
    """Prepare arbitrary agent context without a provider or fixture task."""
    if not isinstance(data, dict):
        raise TypeError("data must be a mapping")

    required_list = list(required)
    if any(not isinstance(key, str) for key in required_list):
        raise TypeError("required must contain only string field names")

    task = {
        "task_id": task_id,
        "task_family": task_family,
        "required": required_list,
        "data": data,
    }

    return _prepare_task(
        task,
        policy=policy,
        arm_id=arm_id,
    )
