from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]

SCHEDULE = (
    ROOT
    / "harness"
    / "runner"
    / "v084_zoom_schedule.json"
)

PREREG = (
    ROOT
    / "harness"
    / "prereg"
    / "v084_zoom_preregistration.md"
)


def test_schedule_exists():
    assert SCHEDULE.exists()


def test_prereg_exists():
    assert PREREG.exists()


def test_schedule_shape():
    payload = json.loads(
        SCHEDULE.read_text(encoding="utf-8")
    )

    assert payload["experiment"] == "V0.8.4_ZOOM"
    assert payload["task_ids"] == ["T003"]
    assert payload["repetitions"] == 30
    assert payload["runs"] == 120
    assert len(payload["conditions"]) == 4
    assert len(payload["rows"]) == 120


def test_positions_are_contiguous():
    payload = json.loads(
        SCHEDULE.read_text(encoding="utf-8")
    )

    positions = [
        row["global_execution_pos"]
        for row in payload["rows"]
    ]

    assert positions == list(range(120))


def test_each_repetition_contains_all_conditions():
    payload = json.loads(
        SCHEDULE.read_text(encoding="utf-8")
    )

    expected = set(payload["conditions"])

    for repetition in range(1, 31):
        rows = [
            row
            for row in payload["rows"]
            if row["repetition"] == repetition
        ]

        assert len(rows) == 4
        assert {
            row["arm_id"]
            for row in rows
        } == expected


def test_each_repetition_has_unique_positions():
    payload = json.loads(
        SCHEDULE.read_text(encoding="utf-8")
    )

    for repetition in range(1, 31):
        rows = [
            row
            for row in payload["rows"]
            if row["repetition"] == repetition
        ]

        positions = [
            row["position_within_rep"]
            for row in rows
        ]

        assert positions == [1, 2, 3, 4]
