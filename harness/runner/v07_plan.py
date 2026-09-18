from pathlib import Path
import json

TASKS = ["T001", "T002", "T003", "T004"]
ARMS = ["selection_only", "selection_representation", "selection_representation_A"]
REPETITIONS = 15
OUTPUT = Path(r"D:\PROMPTFORGE\promptforge-harness-v0.1\harness\runner\v07_schedule.json")


def build_schedule():
    rows = []

    for task in TASKS:
        for repetition in range(1, REPETITIONS + 1):
            shift = (repetition - 1) % len(ARMS)
            order = ARMS[shift:] + ARMS[:shift]

            for relative_position, arm in enumerate(order):
                rows.append({
                    "task": task,
                    "repetition": repetition,
                    "arm_id": arm,
                    "relative_position": relative_position,
                    "global_execution_position": None
                })

    # IMPORTANT: global position is assigned only after the complete
    # 180-row sequence exists. This prevents task-local resets.
    for global_execution_position, row in enumerate(rows):
        row["global_execution_position"] = global_execution_position

    return rows


def validate_schedule(rows):
    expected_total = len(TASKS) * REPETITIONS * len(ARMS)

    assert len(rows) == expected_total

    # Global execution positions must be exactly 0..179.
    observed_global_positions = sorted(
        row["global_execution_position"]
        for row in rows
    )

    assert observed_global_positions == list(range(expected_total))

    for task in TASKS:
        task_rows = [
            row for row in rows
            if row["task"] == task
        ]

        assert len(task_rows) == REPETITIONS * len(ARMS)

        for repetition in range(1, REPETITIONS + 1):
            repetition_rows = [
                row for row in task_rows
                if row["repetition"] == repetition
            ]

            assert len(repetition_rows) == len(ARMS)
            assert {
                row["arm_id"]
                for row in repetition_rows
            } == set(ARMS)

            assert {
                row["relative_position"]
                for row in repetition_rows
            } == set(range(len(ARMS)))

        for arm in ARMS:
            arm_positions = [
                row["relative_position"]
                for row in task_rows
                if row["arm_id"] == arm
            ]

            assert len(arm_positions) == REPETITIONS

            for position in range(len(ARMS)):
                assert arm_positions.count(position) == 5

        assert all(
            row["arm_id"] != "selection_representation_B"
            for row in task_rows
        )

    return True


def main():
    rows = build_schedule()
    validate_schedule(rows)

    payload = {
        "status": "VALID",
        "tasks": TASKS,
        "arms": ARMS,
        "repetitions": REPETITIONS,
        "runs_per_task": REPETITIONS * len(ARMS),
        "total_runs": len(rows),
        "scheme": "latin_square",
        "api_calls": 0,
        "rows": rows
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8"
    )

    print("V07_SCHEDULE=PASS")
    print(f"TASKS={len(TASKS)}")
    print(f"ARMS={len(ARMS)}")
    print(f"REPETITIONS={REPETITIONS}")
    print(f"RUNS_PER_TASK={REPETITIONS * len(ARMS)}")
    print(f"TOTAL_RUNS={len(rows)}")
    print("GLOBAL_POSITIONS=0..179")
    print("SCHEME=latin_square")
    print("API_CALLS=0")
    print(f"OUTPUT={OUTPUT}")


if __name__ == "__main__":
    main()
