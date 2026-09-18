import importlib.util
from pathlib import Path

ROOT = Path(r"D:\PROMPTFORGE\promptforge-harness-v0.1")
PLAN = ROOT / "harness" / "runner" / "v07_plan.py"


def load_module():
    spec = importlib.util.spec_from_file_location("v07_plan", PLAN)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_constants():
    module = load_module()
    assert module.TASKS == ["T001", "T002", "T003", "T004"]
    assert module.ARMS == [
        "selection_only",
        "selection_representation",
        "selection_representation_A"
    ]
    assert module.REPETITIONS == 15


def test_total_size():
    module = load_module()
    rows = module.build_schedule()
    assert len(rows) == 180


def test_task_size():
    module = load_module()
    rows = module.build_schedule()

    for task in module.TASKS:
        task_rows = [
            row for row in rows
            if row["task"] == task
        ]

        assert len(task_rows) == 45


def test_repetition_coverage():
    module = load_module()
    rows = module.build_schedule()

    for task in module.TASKS:
        task_rows = [
            row for row in rows
            if row["task"] == task
        ]

        assert {
            row["repetition"]
            for row in task_rows
        } == set(range(1, 16))


def test_latin_square_balance():
    module = load_module()
    rows = module.build_schedule()

    for task in module.TASKS:
        task_rows = [
            row for row in rows
            if row["task"] == task
        ]

        for arm in module.ARMS:
            positions = [
                row["relative_position"]
                for row in task_rows
                if row["arm_id"] == arm
            ]

            assert len(positions) == 15

            for position in range(3):
                assert positions.count(position) == 5


def test_global_execution_positions_are_unique_and_complete():
    module = load_module()
    rows = module.build_schedule()

    positions = sorted(
        row["global_execution_position"]
        for row in rows
    )

    assert positions == list(range(180))
    assert len(set(positions)) == 180


def test_no_B():
    module = load_module()
    rows = module.build_schedule()

    assert all(
        row["arm_id"] != "selection_representation_B"
        for row in rows
    )


def test_internal_validation():
    module = load_module()
    rows = module.build_schedule()
    assert module.validate_schedule(rows) is True
