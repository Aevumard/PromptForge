from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any


POLICY_MINIMAL = "minimal"
POLICY_MINIMAL_SERIALIZED_CONTEXT = "minimal_serialized_context"
POLICY_BUDGET_CONSTRAINED = "budget_constrained"

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

ARM_SEQUENCES: dict[str, tuple[str, ...]] = {
    "noop": ("noop",),
    "selection_only": ("selection",),
    "representation_only": ("representation",),
    "representation_A": ("representation_A",),
    "representation_B": ("representation_B",),
    "selection_representation": ("selection", "representation"),
    "selection_representation_A": ("selection", "representation_A"),
    "selection_representation_B": ("selection", "representation_B"),
}

MISSING = object()


def normalize_required_paths(required: Iterable[str]) -> list[str]:
    if isinstance(required, str):
        raise TypeError("required must be an iterable of field paths, not a string")

    paths: list[str] = []
    seen: set[str] = set()

    for path in required:
        if not isinstance(path, str):
            raise TypeError("required must contain only string field paths")

        path = path.strip()
        if not path:
            raise ValueError("required field paths must not be empty")

        if path not in seen:
            paths.append(path)
            seen.add(path)

    normalized: list[str] = []
    for path in paths:
        parts = path.split(".")
        if any(not part for part in parts):
            raise ValueError(f"invalid field path: {path!r}")

        if any(
            existing == path
            or (
                path.startswith(existing + ".")
                and existing in normalized
            )
            for existing in normalized
        ):
            continue

        normalized = [
            existing
            for existing in normalized
            if not existing.startswith(path + ".")
        ]
        normalized.append(path)

    return normalized


def get_path(data: Mapping[str, Any], path: str) -> Any:
    current: Any = data

    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return MISSING
        current = current[segment]

    return current


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = target

    for segment in parts[:-1]:
        child = current.get(segment)
        if not isinstance(child, dict):
            child = {}
            current[segment] = child
        current = child

    current[parts[-1]] = deepcopy(value)


def select_required_paths(
    data: Mapping[str, Any],
    required: Iterable[str],
) -> dict[str, Any]:
    selected: dict[str, Any] = {}

    for path in normalize_required_paths(required):
        value = get_path(data, path)
        if value is not MISSING:
            _set_path(selected, path, value)

    return selected


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


def missing_required_paths(
    data: Mapping[str, Any],
    required: Iterable[str],
) -> list[str]:
    return [
        path
        for path in normalize_required_paths(required)
        if get_path(data, path) is MISSING
    ]


def estimate_tokens(serialized_context: str) -> int:
    """Estimate tokens using UTF-8 byte length divided by four."""
    if not serialized_context:
        return 0

    byte_count = len(serialized_context.encode("utf-8"))
    return max(1, math.ceil(byte_count / 4))


def serialize_context(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def validate_field_schema(
    data: Mapping[str, Any],
    schema: Mapping[str, type | tuple[type, ...]],
) -> dict[str, Any]:
    mismatches: dict[str, Any] = {}

    for path, expected_type in schema.items():
        value = get_path(data, path)

        if value is MISSING:
            mismatches[path] = {
                "reason": "missing",
                "expected_type": _type_name(expected_type),
            }
            continue

        if not isinstance(value, expected_type):
            mismatches[path] = {
                "reason": "type_mismatch",
                "expected_type": _type_name(expected_type),
                "actual_type": type(value).__name__,
            }

    return {
        "passed": not mismatches,
        "checked_count": len(schema),
        "mismatches": mismatches,
    }


def _type_name(expected_type: type | tuple[type, ...]) -> str:
    if isinstance(expected_type, tuple):
        return "|".join(item.__name__ for item in expected_type)
    return expected_type.__name__


def inspect(
    data: Mapping[str, Any],
    required: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Inspect context without transforming or mutating it."""
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping")

    required_paths = normalize_required_paths(required or [])
    serialized = serialize_context(data)
    missing = missing_required_paths(data, required_paths)

    return {
        "schema_version": "context-inspect.v1",
        "top_level_fields": len(data),
        "leaf_fields": len(_leaf_paths(data)),
        "required_count": len(required_paths),
        "required_present": len(required_paths) - len(missing),
        "required_missing": missing,
        "context_chars": len(serialized),
        "context_bytes": len(serialized.encode("utf-8")),
        "estimated_tokens": estimate_tokens(serialized),
        "token_estimator": "utf8_bytes_div_4_ceiling",
    }


def _decode_representation(value: dict[str, Any], stage: str) -> dict[str, Any]:
    if stage in {"representation", "representation_A"}:
        fields = value.get("fields")
        if not isinstance(fields, list):
            raise ValueError("invalid representation_A payload")

        decoded: dict[str, Any] = {}
        for item in fields:
            if not isinstance(item, dict) or "key" not in item:
                raise ValueError("invalid representation_A field")
            decoded[item["key"]] = item.get("value")
        return decoded

    if stage == "representation_B":
        pairs = value.get("pairs")
        if not isinstance(pairs, list):
            raise ValueError("invalid representation_B payload")

        decoded = {}
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("invalid representation_B pair")
            decoded[pair[0]] = pair[1]
        return decoded

    raise ValueError(f"unknown representation stage: {stage}")


def _apply_arm(
    data: Mapping[str, Any],
    required_paths: list[str],
    arm_id: str,
) -> dict[str, Any]:
    sequence = ARM_SEQUENCES.get(arm_id)
    if sequence is None:
        raise ValueError(f"unknown PromptForge arm: {arm_id}")

    uses_selection = "selection" in sequence
    working_data = (
        select_required_paths(data, required_paths)
        if uses_selection
        else dict(data)
    )

    value: Any = deepcopy(working_data)

    for stage in sequence:
        if stage in {"noop", "selection"}:
            continue

        if stage in {"representation", "representation_A"}:
            value = {
                "fields": [
                    {"key": key, "value": deepcopy(item)}
                    for key, item in value.items()
                ]
            }
            continue

        if stage == "representation_B":
            value = {
                "pairs": [
                    [key, deepcopy(item)]
                    for key, item in value.items()
                ]
            }
            continue

        raise ValueError(f"unknown transform stage: {stage}")

    if not isinstance(value, dict):
        raise ValueError("compiled context value is not a mapping")

    representation_stage = next(
        (
            stage
            for stage in reversed(sequence)
            if stage in {"representation", "representation_A", "representation_B"}
        ),
        None,
    )
    decoded = (
        _decode_representation(value, representation_stage)
        if representation_stage is not None
        else value
    )

    missing = missing_required_paths(decoded, required_paths)
    mismatches: dict[str, Any] = {}

    for path in required_paths:
        expected = get_path(data, path)
        actual = get_path(decoded, path)

        if expected is MISSING or actual is MISSING:
            continue

        if expected != actual:
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

    serialized = serialize_context(value)
    all_leaf_paths = _leaf_paths(dict(data))
    included_leaf_paths = _leaf_paths(working_data)
    included_set = set(included_leaf_paths)

    required_roots: list[str] = []
    for path in required_paths:
        root = path.split(".", 1)[0]
        if root not in required_roots:
            required_roots.append(root)

    provenance = [
        {
            "path": path,
            "source": "input",
            "required": True,
            "included": path in included_leaf_paths
            or any(item.startswith(path + ".") for item in included_leaf_paths),
        }
        for path in required_paths
    ]

    return {
        "schema_version": "agent-context.v2",
        "task_id": "adhoc",
        "task_family": "agent_request",
        "arm_id": arm_id,
        "transform_id": "|".join(sequence),
        "transform_sequence": list(sequence),
        "included": required_roots if uses_selection else list(dict(data).keys()),
        "excluded": [
            path
            for path in all_leaf_paths
            if path not in included_set
        ],
        "included_paths": included_leaf_paths,
        "excluded_paths": [
            path
            for path in all_leaf_paths
            if path not in included_set
        ],
        "required": required_roots,
        "required_paths": list(required_paths),
        "value": value,
        "serialized_context": serialized,
        "context_chars": len(serialized),
        "estimated_tokens": estimate_tokens(serialized),
        "required_values_preserved": True,
        "validation": validation,
        "provenance": provenance,
    }


def _candidate_summary(compiled: dict[str, Any]) -> dict[str, Any]:
    return {
        "arm_id": compiled["arm_id"],
        "transform_id": compiled["transform_id"],
        "transform_sequence": list(compiled["transform_sequence"]),
        "context_chars": compiled["context_chars"],
        "estimated_tokens": compiled["estimated_tokens"],
        "included": list(compiled["included"]),
        "excluded": list(compiled["excluded"]),
        "included_paths": list(compiled["included_paths"]),
        "excluded_paths": list(compiled["excluded_paths"]),
        "required_values_preserved": True,
    }


def _public_result(
    compiled: dict[str, Any],
    *,
    policy: str,
    candidates: list[dict[str, Any]],
    required_paths: list[str],
    budget_tokens: int | None,
    schema_validation: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline = next(
        (candidate for candidate in candidates if candidate["arm_id"] == "noop"),
        None,
    )
    baseline_chars = baseline["context_chars"] if baseline else None
    baseline_tokens = baseline["estimated_tokens"] if baseline else None

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
        "required_paths": list(required_paths),
        "included": list(compiled["included"]),
        "excluded": list(compiled["excluded"]),
        "included_paths": list(compiled["included_paths"]),
        "excluded_paths": list(compiled["excluded_paths"]),
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
        "required_values_preserved": True,
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
        "provenance": compiled["provenance"],
        "inspection": inspect(
            {
                "required": required_paths,
                "context": compiled["value"],
            },
            ["required"],
        ),
        "candidates": candidates,
    }


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
    """Prepare arbitrary provider-agnostic context without the research harness."""
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping")

    if budget_tokens is not None and budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")

    required_paths = normalize_required_paths(required)
    if schema is not None:
        required_paths = normalize_required_paths(
            [*required_paths, *schema.keys()]
        )

    if not required_paths:
        raise ValueError("required must contain at least one field path")

    inspection = inspect(data, required_paths)

    if arm_id is not None:
        if policy != POLICY_MINIMAL:
            raise ValueError(
                "arm_id cannot be combined with policy=" + repr(policy)
            )
        compiled = _apply_arm(data, required_paths, arm_id)
        if budget_tokens is not None and compiled["estimated_tokens"] > budget_tokens:
            raise ValueError(
                "explicit arm exceeds budget: "
                f"{compiled['estimated_tokens']} > {budget_tokens} tokens"
            )
        candidates = [_candidate_summary(compiled)]
        effective_policy = "explicit_arm"
    else:
        if policy not in {POLICY_MINIMAL, POLICY_BUDGET_CONSTRAINED}:
            raise ValueError("unknown agent policy: " + repr(policy))

        effective_policy = (
            POLICY_BUDGET_CONSTRAINED
            if budget_tokens is not None and policy == POLICY_MINIMAL
            else policy
        )

        compiled_candidates = [
            _apply_arm(data, required_paths, candidate_arm)
            for candidate_arm in ARM_ORDER
        ]
        candidates = [
            _candidate_summary(compiled)
            for compiled in compiled_candidates
        ]

        if effective_policy == POLICY_BUDGET_CONSTRAINED:
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
                smallest = min(
                    compiled["estimated_tokens"]
                    for compiled in compiled_candidates
                )
                raise ValueError(
                    "no context candidate fits budget: "
                    f"{smallest} > {budget_tokens} tokens"
                )

            compiled = min(
                feasible,
                key=lambda item: (item[1]["context_chars"], item[0]),
            )[1]
        else:
            compiled = min(
                enumerate(compiled_candidates),
                key=lambda item: (item[1]["context_chars"], item[0]),
            )[1]

    schema_validation = (
        validate_field_schema(
            (
                _decode_representation(
                    compiled["value"],
                    next(
                        stage
                        for stage in reversed(compiled["transform_sequence"])
                        if stage in {
                            "representation",
                            "representation_A",
                            "representation_B",
                        }
                    ),
                )
                if any(
                    stage in {
                        "representation",
                        "representation_A",
                        "representation_B",
                    }
                    for stage in compiled["transform_sequence"]
                )
                else compiled["value"]
            ),
            schema,
        )
        if schema is not None
        else None
    )

    if schema_validation is not None and not schema_validation["passed"]:
        raise ValueError(
            "schema validation failed: "
            + serialize_context(schema_validation["mismatches"])
        )

    compiled["task_id"] = task_id
    compiled["task_family"] = task_family

    result = _public_result(
        compiled,
        policy=effective_policy,
        candidates=candidates,
        required_paths=required_paths,
        budget_tokens=budget_tokens,
        schema_validation=schema_validation,
    )
    result["inspection"] = inspection
    return result


__all__ = [
    "POLICY_BUDGET_CONSTRAINED",
    "POLICY_MINIMAL",
    "POLICY_MINIMAL_SERIALIZED_CONTEXT",
    "inspect",
    "prepare_context",
]
