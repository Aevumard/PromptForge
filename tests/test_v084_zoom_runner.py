import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "harness" / "runner" / "v084_zoom_runner.py"
SCHEDULE = ROOT / "harness" / "runner" / "v084_zoom_schedule.json"

sys.path.insert(0, str(ROOT))

from harness.runner.v084_zoom_runner import CONDITIONS, preflight_task, preflight


def test_runner_exists():
    assert RUNNER.exists()


def test_schedule_exists():
    assert SCHEDULE.exists()


def test_condition_set():
    assert CONDITIONS == [
        "noop_compact",
        "pairs_compact",
        "pairs_bare_compact",
        "flat_compact",
    ]


def test_preflight_dimensions():
    task = preflight_task()
    observed = preflight(task)

    expected = {
        "noop_compact": (85, 307),
        "pairs_compact": (105, 327),
        "pairs_bare_compact": (95, 317),
        "flat_compact": (85, 307),
    }

    for condition, sizes in expected.items():
        assert (
            observed[condition]["context_chars"],
            observed[condition]["prompt_chars"],
        ) == sizes


def test_dry_run_no_api_calls():
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "V084_DRY_RUN=PASS" in completed.stdout
    assert "V084_API_CALLS=0" in completed.stdout


def test_real_report_not_created_by_dry_run():
    report = ROOT / "harness" / "reports" / "v084_zoom_deepseek.json"
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data.get("status") in {"RUNNING", "COMPLETE"}
