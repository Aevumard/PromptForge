from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Iterable


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
    paths = normalize_required_paths(required)
    selected: dict[str, Any] = {}

    for path in paths:
        value = get_path(data, path)
        if value is MISSING:
            continue
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
    """Estimate tokens without a tokenizer dependency.

    The estimator uses UTF-8 byte length / 4, rounded up. It is a
    planning heuristic, not a provider-specific tokenizer measurement.
    """
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
        return "|".join(t.__name__ for t in expected_type)
    return expected_type.__name__


def inspect_context(
    data: Mapping[str, Any],
    required: Iterable[str] | None = None,
) -> dict[str, Any]:
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
