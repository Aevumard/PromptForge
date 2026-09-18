import importlib.util
import json
from pathlib import Path

ROOT = Path(r"D:\PROMPTFORGE\promptforge-harness-v0.1")
RUNNER = ROOT / "harness" / "runner" / "v07_runner.py"
SCHEDULE = ROOT / "harness" / "runner" / "v07_schedule.json"


def load_module():
    spec = importlib.util.spec_from_file_location("v07_runner", RUNNER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_exists():
    assert RUNNER.exists()


def test_schedule_is_confirmatory():
    payload = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    assert payload["status"] == "VALID"
    assert payload["tasks"] == ["T001", "T002", "T003", "T004"]
    assert payload["arms"] == [
        "selection_only",
        "selection_representation",
        "selection_representation_A",
    ]
    assert payload["repetitions"] == 15
    assert payload["total_runs"] == 180
    assert payload["scheme"] == "latin_square"
    assert payload["api_calls"] == 0


def test_global_positions_are_unique_and_complete():
    payload = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    positions = sorted(
        row["global_execution_position"]
        for row in payload["rows"]
    )
    assert positions == list(range(180))
    assert len(set(positions)) == 180


def test_task_ranges_are_global():
    payload = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    tasks = ["T001", "T002", "T003", "T004"]

    for index, task in enumerate(tasks):
        rows = [
            row for row in payload["rows"]
            if row["task"] == task
        ]

        positions = sorted(
            row["global_execution_position"]
            for row in rows
        )

        assert positions == list(
            range(index * 45, (index + 1) * 45)
        )


def test_runner_schedule_validation():
    module = load_module()
    payload = module.load_and_validate_schedule()
    assert payload["total_runs"] == 180


def test_atomic_writer_round_trip(tmp_path):
    module = load_module()
    target = tmp_path / "atomic.json"
    payload = {"status": "OK", "runs": [1, 2, 3]}

    module.atomic_write_json(target, payload)

    assert json.loads(
        target.read_text(encoding="utf-8")
    ) == payload


def test_no_B():
    payload = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    assert all(
        row["arm_id"] != "selection_representation_B"
        for row in payload["rows"]
    )
