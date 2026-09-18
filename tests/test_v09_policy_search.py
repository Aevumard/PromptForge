import importlib.util
import json
from pathlib import Path

ROOT = Path(
    r"D:\PROMPTFORGE\promptforge-harness-v0.1"
)

PLAN = (
    ROOT
    / "harness"
    / "runner"
    / "v09_policy_search_plan.py"
)

SCHEDULE = (
    ROOT
    / "harness"
    / "runner"
    / "v09_policy_search_schedule.json"
)


def load_plan():

    spec = importlib.util.spec_from_file_location(
        "v09_plan",
        PLAN,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


def test_runs():

    module = load_plan()

    rows = module.build_schedule()

    assert len(rows) == 240


def test_positions():

    module = load_plan()

    rows = module.build_schedule()

    assert [
        x["global_execution_position"]
        for x in rows
    ] == list(range(240))


def test_task_balance():

    module = load_plan()

    rows = module.build_schedule()

    for task_id in module.TASKS:

        task_rows = [
            x
            for x in rows
            if x["task_id"] == task_id
        ]

        assert len(task_rows) == 60

        for condition in module.CONDITIONS:

            cell_rows = [
                x
                for x in task_rows
                if x["condition_id"] == condition
            ]

            assert len(cell_rows) == 15


def test_schedule_file():

    module = load_plan()

    document = json.loads(
        SCHEDULE.read_text(
            encoding="utf-8"
        )
    )

    expected = module.build_document()

    assert document == expected