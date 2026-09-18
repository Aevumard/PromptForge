from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from harness.agent_context import (
    compile_for_agent,
    get_task,
)
from harness.context import (
    estimate_tokens,
    inspect_context,
    missing_required_paths,
    normalize_required_paths,
    select_required_paths,
    serialize_context,
    validate_field_schema,
)
from harness.runner.transforms import (
    V06_ARM_SEQUENCES,
    compile_v06_arm,
    decode_representation_a,
    decode_representation_b,
)


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
POLICY_BUDGET_CONSTRAINED = "budget_constrained"


def _decode_value(compiled: dict[str, Any]) -> dict[str, Any]:
    sequence = compiled.get("transform_sequence", [])
    value = compiled["value"]

    if "representation_A" in sequence or "representation" in sequence:
        return decode_representation_a(value)

    if "representation_B" in sequence:
        return decode_representation_b(value)

    if not isinstance(value, dict):
        raise ValueError("compiled context value is not a mapping")

    return dict(value)


def _candidate_summary(compiled: dict[str, Any]) -> dict[str, Any]:
    return {
        "arm_id": compiled["arm_id"],
        "transform_id": compiled["transform_id"],
        "transform_sequence": list(compiled["transform_sequence"]),
        "context_chars": compiled["context_chars"],
        "estimated_tokens": compiled["estimated_tokens"],
        "included": list(compiled["included"]),
        "excluded": list(compiled["excluded"]),
        "included_paths": list(
            compiled.get("included_paths", compiled["included"])
        ),
        "excluded_paths": list(
            compiled.get("excluded_paths", compiled["excluded"])
        ),
        "required_values_preserved": bool(
            compiled["required_values_preserved"]
        ),
    }


def _public_result(
    compiled: dict[str, Any],
    *,
    policy: str,
    candidates: list[dict[str, Any]],
    budget_tokens: int | None = None,
    schema_validation: dict[str, Any] | None = None,
    inspection: dict[str, Any] | None = None,
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
    baseline_tokens = (
        baseline["estimated_tokens"]
        if baseline is not None
        else None
    )
    saved_chars = (
        baseline_chars - compiled["context_chars"]
        if baseline_chars is not None
        else None
    )
    saved_tokens = (
        baseline_tokens - compiled["estimated_tokens"]
        if baseline_tokens is not None
        else None
    )
    reduction_ratio = (
        saved_chars / baseline_chars
        if baseline_chars not in (None, 0)
        else None
    )

    return {
        "schema_version": "agent-prepare.v2",
        "task_id": compiled["task_id"],
        "task_family": compiled["task_family"],
        "policy": policy,
        "selected_arm": compiled["arm_id"],
        "transform_id": compiled["transform_id"],
        "transform_sequence": list(compiled["transform_sequence"]),
        "required": list(compiled["required"]),
        "required_paths": list(
            compiled.get("required_paths", compiled["required"])
        ),
        "included": list(compiled["included"]),
        "excluded": list(compiled["excluded"]),
        "included_paths": list(
            compiled.get("included_paths", compiled["included"])
        ),
        "excluded_paths": list(
            compiled.get("excluded_paths", compiled["excluded"])
        ),
        "context": compiled["value"],
        "serialized_context": compiled["serialized_context"],
        "context_chars": compiled["context_chars"],
        "estimated_tokens": compiled["estimated_tokens"],
        "token_estimator": "utf8_bytes_div_4_ceiling",
        "baseline_arm": "noop" if baseline is not None else None,
        "baseline_context_chars": baseline_chars,
        "baseline_estimated_tokens": baseline_tokens,
        "context_chars_saved": saved_chars,
        "estimated_tokens_saved": saved_tokens,
        "context_reduction_ratio": reduction_ratio,
        "budget_tokens": budget_tokens,
        "budget_satisfied": (
            budget_tokens is None
            or compiled["estimated_tokens"] <= budget_tokens
        ),
        "required_values_preserved": bool(
            compiled["required_values_preserved"]
        ),
        "validation": compiled["validation"],
        "schema_validation": schema_validation,
        "selection_basis": (
            "smallest_serialized_context"
            if policy == POLICY_MINIMAL_SERIALIZED_CONTEXT
            else (
                "smallest_serialized_context_within_budget"
                if policy == POLICY_BUDGET_CONSTRAINED
                else "explicit_arm"
            )
        ),
        "provenance": compiled.get("provenance", []),
        "inspection": inspection,
        "candidates": candidates,
    }


def _compile_generic_arm(
    data: Mapping[str, Any],
    required_paths: list[str],
    task_id: str,
    task_family: str,
    arm_id: str,
) -> dict[str, Any]:
    sequence = V06_ARM_SEQUENCES.get(arm_id)
    if sequence is None:
        raise ValueError(f"unknown PromptForge arm: {arm_id}")

    uses_selection = "selection" in sequence
    working_data = (
        select_required_paths(data, required_paths)
        if uses_selection
        else dict(data)
    )

    required_roots = []
    for path in required_paths:
        root = path.split(".", 1)[0]
        if root not in required_roots:
            required_roots.append(root)

    compiled = compile_v06_arm(
        {
            "required": required_roots,
            "data": working_data,
        },
        arm_id,
    )

    decoded = _decode_value(compiled)
    missing = missing_required_paths(decoded, required_paths)
    mismatches = {}

    for path in required_paths:
        original = data
        expected_marker = object()

        def lookup(source: Any, field_path: str) -> Any:
            current = source
            for segment in field_path.split("."):
                if not isinstance(current, Mapping):
                    return expected_marker
                current = current.get(segment, expected_marker)
                if current is expected_marker:
                    return expected_marker
            return current

        expected = lookup(original, path)
        actual = lookup(decoded, path)

        if actual is expected_marker or expected is expected_marker:
            continue

        if actual != expected:
            mismatches[path] = {
                "expected": expected,
                "actual": actual,
                "reason": "value_mismatch",
            }

    validation = {
        "passed": not missing and not mismatches,
        "required_count": len(required_paths),
        "mismatches": {
            **{
                path: {
                    "expected": "present",
                    "actual": None,
                    "reason": "missing",
                }
                for path in missing
            },
            **mismatches,
        },
    }

    if not validation["passed"]:
        raise ValueError(
            "required values were not preserved: "
            + serialize_context(validation["mismatches"])
        )

    serialized_context = serialize_context(compiled["value"])

    all_leaf_paths = _leaf_paths(dict(data))
    included_leaf_paths = _leaf_paths(working_data)
    included_paths = list(included_leaf_paths)
    excluded_paths = [
        path
        for path in all_leaf_paths
        if path not in set(included_leaf_paths)
    ]

    provenance = [
        {
            "path": path,
            "source": "input",
            "required": True,
            "included": path in included_paths
            or any(item.startswith(path + ".") for item in included_paths),
        }
        for path in required_paths
    ]

    result = {
        "schema_version": "agent-context.v2",
        "task_id": task_id,
        "task_family": task_family,
        "arm_id": compiled["arm_id"],
        "transform_id": compiled["transform_id"],
        "transform_sequence": list(compiled["transform_sequence"]),
        "included": list(compiled["included"]),
        "excluded": list(compiled["excluded"]),
        "included_paths": included_paths,
        "excluded_paths": excluded_paths,
        "required": list(required_roots),
        "required_paths": list(required_paths),
        "value": compiled["value"],
        "serialized_context": serialized_context,
        "context_chars": len(serialized_context),
        "estimated_tokens": estimate_tokens(serialized_context),
        "required_values_preserved": True,
        "validation": validation,
        "provenance": provenance,
    }
    return result


def _leaf_paths(value: Any, prefix: str = "") -> list[str]:
    if not isinstance(value, Mapping) or not value:
        return [prefix] if prefix else []

    paths: list[str] = []
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping) and child:
            paths.extend(_leaf_paths(child, path))
        else:
            paths.append(path)
    return paths


def _prepare_minimal(task: dict[str, Any]) -> dict[str, Any]:
    compiled_candidates: list[dict[str, Any]] = []

    for arm_id in ARM_ORDER:
        if arm_id not in V06_ARM_SEQUENCES:
            raise ValueError(
                "agent facade references unknown arm: " + arm_id
            )

        compiled = compile_for_agent(task, arm_id)
        compiled["estimated_tokens"] = estimate_tokens(
            compiled["serialized_context"]
        )
        compiled["required_paths"] = list(task["required"])
        compiled["included_paths"] = list(compiled["included"])
        compiled["excluded_paths"] = list(compiled["excluded"])
        compiled["provenance"] = [
            {
                "path": path,
                "source": "fixture",
                "required": True,
                "included": path in compiled["included"],
            }
            for path in task["required"]
        ]
        compiled_candidates.append(compiled)

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


def _prepare_generic(
    data: Mapping[str, Any],
    required: Iterable[str],
    *,
    task_id: str,
    task_family: str,
    policy: str,
    arm_id: str | None,
    budget_tokens: int | None,
    schema: Mapping[str, type | tuple[type, ...]] | None,
) -> dict[str, Any]:
    required_paths = normalize_required_paths(required)
    if schema is not None:
        schema_paths = normalize_required_paths(schema.keys())
        required_paths = normalize_required_paths(
            [*required_paths, *schema_paths]
        )

    inspection = inspect_context(data, required_paths)

    if not required_paths:
        raise ValueError("required must contain at least one field path")

    if arm_id is not None:
        if policy != POLICY_MINIMAL:
            raise ValueError(
                "arm_id cannot be combined with policy=" + repr(policy)
            )

        compiled = _compile_generic_arm(
            data,
            required_paths,
            task_id,
            task_family,
            arm_id,
        )
        if budget_tokens is not None and (
            compiled["estimated_tokens"] > budget_tokens
        ):
            raise ValueError(
                "explicit arm exceeds budget: "
                f"{compiled['estimated_tokens']} > {budget_tokens} tokens"
            )

        schema_validation = (
            validate_field_schema(_decode_value(compiled), schema)
            if schema is not None
            else None
        )
        if schema_validation is not None and not schema_validation["passed"]:
            raise ValueError(
                "schema validation failed: "
                + serialize_context(schema_validation["mismatches"])
            )

        candidates = [_candidate_summary(compiled)]
        return _public_result(
            compiled,
            policy="explicit_arm",
            candidates=candidates,
            budget_tokens=budget_tokens,
            schema_validation=schema_validation,
            inspection=inspection,
        )

    if policy not in {
        POLICY_MINIMAL,
        POLICY_BUDGET_CONSTRAINED,
    }:
        raise ValueError(
            "unknown agent policy: " + repr(policy)
        )

    effective_policy = policy
    if budget_tokens is not None and policy == POLICY_MINIMAL:
        effective_policy = POLICY_BUDGET_CONSTRAINED

    candidates_compiled = [
        _compile_generic_arm(
            data,
            required_paths,
            task_id,
            task_family,
            candidate_arm,
        )
        for candidate_arm in ARM_ORDER
    ]

    candidates = [
        _candidate_summary(compiled)
        for compiled in candidates_compiled
    ]

    if effective_policy == POLICY_MINIMAL:
        selected = min(
            enumerate(candidates_compiled),
            key=lambda item: (
                item[1]["context_chars"],
                item[0],
            ),
        )[1]
    elif effective_policy == POLICY_BUDGET_CONSTRAINED:
        if budget_tokens is None:
            raise ValueError(
                "budget_constrained policy requires budget_tokens"
            )

        feasible = [
            (index, compiled)
            for index, compiled in enumerate(candidates_compiled)
            if compiled["estimated_tokens"] <= budget_tokens
        ]
        if not feasible:
            smallest = min(
                compiled["estimated_tokens"]
                for compiled in candidates_compiled
            )
            raise ValueError(
                "no context candidate fits budget: "
                f"{smallest} > {budget_tokens} tokens"
            )

        selected = min(
            feasible,
            key=lambda item: (
                item[1]["context_chars"],
                item[0],
            ),
        )[1]
    else:
        raise ValueError(
            "unknown agent policy: " + repr(policy)
        )

    schema_validation = (
        validate_field_schema(_decode_value(selected), schema)
        if schema is not None
        else None
    )
    if schema_validation is not None and not schema_validation["passed"]:
        raise ValueError(
            "schema validation failed: "
            + serialize_context(schema_validation["mismatches"])
        )

    return _public_result(
        selected,
        policy=effective_policy,
        candidates=candidates,
        budget_tokens=budget_tokens,
        schema_validation=schema_validation,
        inspection=inspection,
    )


def prepare(
    task_id: str,
    *,
    policy: str = POLICY_MINIMAL,
    arm_id: str | None = None,
    budget_tokens: int | None = None,
) -> dict[str, Any]:
    """Prepare a validated public fixture task for agent consumption."""
    task = get_task(task_id)

    if budget_tokens is not None and budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")

    if arm_id is not None:
        if policy != POLICY_MINIMAL:
            raise ValueError(
                "arm_id cannot be combined with policy=" + repr(policy)
            )

        compiled = compile_for_agent(task, arm_id)
        compiled["estimated_tokens"] = estimate_tokens(
            compiled["serialized_context"]
        )
        compiled["required_paths"] = list(task["required"])
        compiled["included_paths"] = list(compiled["included"])
        compiled["excluded_paths"] = list(compiled["excluded"])
        compiled["provenance"] = [
            {
                "path": path,
                "source": "fixture",
                "required": True,
                "included": path in compiled["included"],
            }
            for path in task["required"]
        ]

        if (
            budget_tokens is not None
            and compiled["estimated_tokens"] > budget_tokens
        ):
            raise ValueError(
                "explicit arm exceeds budget: "
                f"{compiled['estimated_tokens']} > {budget_tokens} tokens"
            )

        return _public_result(
            compiled,
            policy="explicit_arm",
            candidates=[_candidate_summary(compiled)],
            budget_tokens=budget_tokens,
        )

    if policy not in {
        POLICY_MINIMAL,
        POLICY_BUDGET_CONSTRAINED,
    }:
        raise ValueError(
            "unknown agent policy: " + repr(policy)
        )

    if policy == POLICY_BUDGET_CONSTRAINED or budget_tokens is not None:
        compiled_candidates = []
        for candidate_arm in ARM_ORDER:
            compiled = compile_for_agent(task, candidate_arm)
            compiled["estimated_tokens"] = estimate_tokens(
                compiled["serialized_context"]
            )
            compiled["required_paths"] = list(task["required"])
            compiled["included_paths"] = list(compiled["included"])
            compiled["excluded_paths"] = list(compiled["excluded"])
            compiled["provenance"] = [
                {
                    "path": path,
                    "source": "fixture",
                    "required": True,
                    "included": path in compiled["included"],
                }
                for path in task["required"]
            ]
            compiled_candidates.append(compiled)

        if budget_tokens is None:
            raise ValueError(
                "budget_constrained policy requires budget_tokens"
            )

        feasible = [
            (index, compiled)
            for index, compiled in enumerate(compiled_candidates)
            if compiled["estimated_tokens"] <= budget_tokens
        ]
        if not feasible:
            raise ValueError("no context candidate fits budget")

        selected = min(
            feasible,
            key=lambda item: (
                item[1]["context_chars"],
                item[0],
            ),
        )[1]
        return _public_result(
            selected,
            policy=POLICY_BUDGET_CONSTRAINED,
            candidates=[
                _candidate_summary(compiled)
                for compiled in compiled_candidates
            ],
            budget_tokens=budget_tokens,
        )

    if policy == POLICY_MINIMAL:
        return _prepare_minimal(task)

    raise ValueError(
        "unknown agent policy: " + repr(policy)
    )


def prepare_context(
    data: Mapping[str, Any],
    required: Iterable[str],
    *,
    task_id: str = "adhoc",
    task_family: str = "agent_request",
    policy: str = POLICY_MINIMAL,
    arm_id: str | None = None,
    budget_tokens: int | None = None,
    schema: Mapping[str, type | tuple[type, ...]] | None = None,
) -> dict[str, Any]:
    """Prepare arbitrary provider-agnostic context for an agent."""
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping")

    if budget_tokens is not None and budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")

    return _prepare_generic(
        data,
        required,
        task_id=task_id,
        task_family=task_family,
        policy=policy,
        arm_id=arm_id,
        budget_tokens=budget_tokens,
        schema=schema,
    )


def inspect(
    data: Mapping[str, Any],
    required: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Inspect context size and required-field coverage without transforming it."""
    return inspect_context(data, required)
