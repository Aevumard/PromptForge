from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPERIMENT_ID = "PROMPTFORGE_V0.9_POLICY_SEARCH"

TASKS = [
    "T001",
    "T002",
    "T003",
    "T004",
]

CONDITIONS = [
    "noop_compact",
    "pairs_compact",
    "pairs_bare_compact",
    "flat_compact",
]

REPETITIONS = 15


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_schedule():
    rows = []
    position = 0

    for repetition in range(1, REPETITIONS + 1):

        cells = []

        for task_id in TASKS:
            for condition_id in CONDITIONS:
                cells.append(
                    {
                        "task_id": task_id,
                        "condition_id": condition_id,
                    }
                )

        offset = ((repetition - 1) * 3) % len(cells)

        cells = cells[offset:] + cells[:offset]

        if repetition % 2 == 0:
            cells.reverse()

        for within_rep, cell in enumerate(cells, start=1):

            rows.append(
                {
                    "repetition": repetition,
                    "task_id": cell["task_id"],
                    "condition_id": cell["condition_id"],
                    "position_within_rep": within_rep,
                    "global_execution_position": position,
                }
            )

            position += 1

    return rows


def schedule_hash(rows):
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "tasks": TASKS,
        "conditions": CONDITIONS,
        "repetitions": REPETITIONS,
        "rows": rows,
    }

    return hashlib.sha256(
        canonical(payload)
    ).hexdigest().upper()


def build_document():
    rows = build_schedule()

    return {
        "experiment_id": EXPERIMENT_ID,
        "version": "0.9-discovery",
        "tasks": TASKS,
        "conditions": CONDITIONS,
        "repetitions": REPETITIONS,
        "scheduled_runs": len(rows),
        "schedule_sha256": schedule_hash(rows),
        "rows": rows,
    }


def validate(document):

    rows = document["rows"]

    assert len(rows) == 240

    positions = [
        row["global_execution_position"]
        for row in rows
    ]

    assert positions == list(range(240))

    for repetition in range(1, REPETITIONS + 1):

        rep_rows = [
            row
            for row in rows
            if row["repetition"] == repetition
        ]

        assert len(rep_rows) == 16

        for task_id in TASKS:

            task_rows = [
                row
                for row in rep_rows
                if row["task_id"] == task_id
            ]

            assert len(task_rows) == 4

            assert {
                row["condition_id"]
                for row in task_rows
            } == set(CONDITIONS)

    for task_id in TASKS:

        task_rows = [
            row
            for row in rows
            if row["task_id"] == task_id
        ]

        assert len(task_rows) == 60

        for condition_id in CONDITIONS:

            cell_rows = [
                row
                for row in task_rows
                if row["condition_id"] == condition_id
            ]

            assert len(cell_rows) == 15

    assert (
        document["schedule_sha256"]
        == schedule_hash(rows)
    )


if __name__ == "__main__":

    output = Path(__file__).with_name(
        "v09_policy_search_schedule.json"
    )

    document = build_document()

    validate(document)

    output.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("V09_PLAN_VALID=PASS")
    print("V09_RUNS=240")
    print(
        "V09_SCHEDULE_SHA256="
        + document["schedule_sha256"]
    )