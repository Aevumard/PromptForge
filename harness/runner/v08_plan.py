"""PromptForge V0.8 deterministic factorial schedule generator.

This module only generates the frozen experimental plan and schedule.
It never imports a provider and never executes an API call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "PROMPTFORGE_V0.8_HETEROGENEITY"
SCHEME = "cyclic_balanced_factorial"
REPETITIONS = 15

TASKS = (
    "T001",
    "T002",
    "T003",
    "T004",
)

MODELS = (
    "deepseek",
    "openai_gpt5_mini",
    "gemini_3_6_flash",
)

ARMS = (
    "noop",
    "representation",
)

# Explicitly excluded from V0.8 primary search space.
EXCLUDED_ARMS = (
    "representation_A",
    "representation_B",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def build_base_factorial() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for model in MODELS:
            for arm in ARMS:
                rows.append(
                    {
                        "task_id": task,
                        "model_id": model,
                        "arm_id": arm,
                    }
                )
    return rows


def rotate(rows: list[dict[str, Any]], offset: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    offset = offset % len(rows)
    return rows[offset:] + rows[:offset]


def build_schedule() -> list[dict[str, Any]]:
    base = build_base_factorial()
    schedule: list[dict[str, Any]] = []
    position = 0

    # Every repetition contains the complete 4 x 3 x 2 factorial once.
    # Cyclic rotation changes execution order deterministically between reps.
    for repetition in range(1, REPETITIONS + 1):
        offset = ((repetition - 1) * 5) % len(base)
        ordered = rotate(base, offset)
        if repetition % 2 == 0:
            ordered = list(reversed(ordered))

        for cell in ordered:
            schedule.append(
                {
                    "global_execution_position": position,
                    "task_id": cell["task_id"],
                    "model_id": cell["model_id"],
                    "arm_id": cell["arm_id"],
                    "repetition": repetition,
                }
            )
            position += 1

    return schedule


def compute_schedule_sha256(positions: list[dict[str, Any]]) -> str:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "scheme": SCHEME,
        "tasks": list(TASKS),
        "models": list(MODELS),
        "arms": list(ARMS),
        "repetitions": REPETITIONS,
        "positions": positions,
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def build_schedule_document() -> dict[str, Any]:
    positions = build_schedule()
    return {
        "experiment_id": EXPERIMENT_ID,
        "version": "v0.8",
        "scheme": SCHEME,
        "tasks": list(TASKS),
        "models": list(MODELS),
        "arms": list(ARMS),
        "excluded_arms": list(EXCLUDED_ARMS),
        "repetitions": REPETITIONS,
        "total_runs": len(positions),
        "schedule_sha256": compute_schedule_sha256(positions),
        "positions": positions,
    }


def write_schedule(path: Path) -> dict[str, Any]:
    document = build_schedule_document()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return document


def validate_document(document: dict[str, Any]) -> None:
    assert document["experiment_id"] == EXPERIMENT_ID
    assert document["scheme"] == SCHEME
    assert document["tasks"] == list(TASKS)
    assert document["models"] == list(MODELS)
    assert document["arms"] == list(ARMS)
    assert document["excluded_arms"] == list(EXCLUDED_ARMS)
    assert document["repetitions"] == REPETITIONS
    positions = document["positions"]
    assert document["total_runs"] == 360
    assert len(positions) == 360
    assert [row["global_execution_position"] for row in positions] == list(range(360))
    assert set(row["task_id"] for row in positions) == set(TASKS)
    assert set(row["model_id"] for row in positions) == set(MODELS)
    assert set(row["arm_id"] for row in positions) == set(ARMS)
    assert all(row["arm_id"] not in EXCLUDED_ARMS for row in positions)

    expected_hash = compute_schedule_sha256(positions)
    assert document["schedule_sha256"] == expected_hash

    for task in TASKS:
        task_rows = [r for r in positions if r["task_id"] == task]
        assert len(task_rows) == 90
        for model in MODELS:
            cell_rows = [r for r in task_rows if r["model_id"] == model]
            assert len(cell_rows) == 30
            for arm in ARMS:
                exact_cell = [r for r in cell_rows if r["arm_id"] == arm]
                assert len(exact_cell) == 15

    for model in MODELS:
        assert sum(1 for r in positions if r["model_id"] == model) == 120

    for arm in ARMS:
        assert sum(1 for r in positions if r["arm_id"] == arm) == 180

    for repetition in range(1, REPETITIONS + 1):
        rep_rows = [r for r in positions if r["repetition"] == repetition]
        assert len(rep_rows) == 24
        for model in MODELS:
            assert sum(1 for r in rep_rows if r["model_id"] == model) == 8
        for arm in ARMS:
            assert sum(1 for r in rep_rows if r["arm_id"] == arm) == 12
        for task in TASKS:
            assert sum(1 for r in rep_rows if r["task_id"] == task) == 6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("v08_schedule.json"),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
    )
    args = parser.parse_args()

    document = build_schedule_document()
    validate_document(document)

    if not args.validate_only:
        write_schedule(args.output)
        print(f"SCHEDULE_WRITTEN={args.output}")

    print("V08_PLAN_VALID=PASS")
    print(f"RUNS={document['total_runs']}")
    print(f"SCHEDULE_SHA256={document['schedule_sha256']}")
    print(f"MODELS={','.join(MODELS)}")
    print(f"ARMS={','.join(ARMS)}")
    print(f"EXCLUDED_ARMS={','.join(EXCLUDED_ARMS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())