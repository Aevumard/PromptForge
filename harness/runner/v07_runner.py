import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TASK_FILE = ROOT / "tasks" / "suite.json"
SCHEDULE_FILE = ROOT / "harness" / "runner" / "v07_schedule.json"
DEFAULT_REPORT = ROOT / "harness" / "reports" / "v07_confirmatory_deepseek.json"

EXPECTED_TASKS = ["T001", "T002", "T003", "T004"]
EXPECTED_ARMS = [
    "selection_only",
    "selection_representation",
    "selection_representation_A",
]
EXPECTED_REPETITIONS = 15
EXPECTED_TOTAL_RUNS = 180
SCHEME = "latin_square"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_schedule():
    payload = json.loads(
        SCHEDULE_FILE.read_text(encoding="utf-8")
    )

    if payload.get("status") != "VALID":
        raise RuntimeError("schedule status is not VALID")

    if payload.get("tasks") != EXPECTED_TASKS:
        raise RuntimeError("schedule task set mismatch")

    if payload.get("arms") != EXPECTED_ARMS:
        raise RuntimeError("schedule arm set mismatch")

    if payload.get("repetitions") != EXPECTED_REPETITIONS:
        raise RuntimeError("schedule repetition count mismatch")

    if payload.get("total_runs") != EXPECTED_TOTAL_RUNS:
        raise RuntimeError("schedule total run count mismatch")

    if payload.get("scheme") != SCHEME:
        raise RuntimeError("schedule scheme mismatch")

    rows = payload.get("rows")

    if not isinstance(rows, list):
        raise RuntimeError("schedule rows missing")

    if len(rows) != EXPECTED_TOTAL_RUNS:
        raise RuntimeError("schedule row count mismatch")

    required_fields = {
        "task",
        "repetition",
        "arm_id",
        "relative_position",
        "global_execution_position",
    }

    for row in rows:
        if set(row.keys()) != required_fields:
            raise RuntimeError("unexpected schedule row schema")

    positions = sorted(
        row["global_execution_position"]
        for row in rows
    )

    if positions != list(range(EXPECTED_TOTAL_RUNS)):
        raise RuntimeError(
            "global execution positions are not exactly 0..179"
        )

    expected = {
        (task, repetition, arm)
        for task in EXPECTED_TASKS
        for repetition in range(1, EXPECTED_REPETITIONS + 1)
        for arm in EXPECTED_ARMS
    }

    observed = {
        (
            row["task"],
            row["repetition"],
            row["arm_id"],
        )
        for row in rows
    }

    if observed != expected:
        raise RuntimeError("schedule coverage mismatch")

    return payload


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )

    temp_path = Path(temp_name)

    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )

        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)

    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def build_task(raw):
    task = dict(raw)

    task["instruction"] = (
        "Extract the required fields from the supplied context. "
        "Return only valid JSON. Do not explain. Do not use markdown."
    )

    task["expected"] = {
        key: raw["data"][key]
        for key in raw["required"]
    }

    task["schema"] = {
        key: "value"
        for key in raw["required"]
    }

    return task


def load_suite():
    suite = json.loads(
        TASK_FILE.read_text(encoding="utf-8")
    )

    tasks = {}

    for raw in suite["tasks"]:
        task = build_task(raw)
        tasks[task["task_id"]] = task

    if set(tasks.keys()) != set(EXPECTED_TASKS):
        raise RuntimeError("task suite mismatch")

    return tasks


def make_failure_execution(message):
    return {
        "provider_status": "PROVIDER_EXCEPTION",
        "error": message,
        "response_id": None,
        "model": None,
        "response_status": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "latency_ms": 0,
        "raw_text": "",
        "output": None,
    }


def run_one(task, row):
    from harness.evaluators.deterministic import evaluate
    from harness.runner.real_executor import execute
    from harness.runner.transforms import compile_v06_arm

    context = {
        "required": task["required"],
        "data": task["data"],
    }

    compiled = compile_v06_arm(
        context,
        row["arm_id"],
    )

    start_utc = utc_now()
    start_mono = time.monotonic()

    try:
        execution = execute(task, compiled)
        runner_exception = None
    except Exception as exc:
        runner_exception = (
            type(exc).__name__ + ": " + str(exc)
        )
        execution = make_failure_execution(
            runner_exception
        )

    end_utc = utc_now()
    end_mono = time.monotonic()

    verification = {
        "evaluator": "deterministic_exact_fields",
        "passed": False,
        "missing_or_incorrect": [],
    }

    if execution.get("output") is not None:
        passed, verification = evaluate(
            task["expected"],
            execution["output"],
        )
        verification["passed"] = passed

    return {
        "schema_version": "0.7",
        "task_id": task["task_id"],
        "task_family": task["task_family"],
        "provider": "deepseek",
        "model": execution.get("model"),
        "arm_id": row["arm_id"],
        "transform_id": compiled["transform_id"],
        "transform_sequence": compiled["transform_sequence"],
        "repetition": row["repetition"],
        "scheme": SCHEME,
        "relative_position": row["relative_position"],
        "global_execution_position": row["global_execution_position"],
        "timestamp_utc": start_utc,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "monotonic_elapsed_ms": (
            end_mono - start_mono
        ) * 1000.0,
        "context_change": {
            "included": compiled["included"],
            "excluded": compiled["excluded"],
        },
        "metrics": {
            "input_tokens": execution["input_tokens"],
            "output_tokens": execution["output_tokens"],
            "reasoning_tokens": execution["reasoning_tokens"],
            "total_tokens": execution["total_tokens"],
            "latency_ms": execution["latency_ms"],
        },
        "execution": {
            "provider_status": execution["provider_status"],
            "response_status": execution["response_status"],
            "response_id": execution["response_id"],
            "error": execution["error"],
        },
        "verification": verification,
        "quality_pass": bool(verification["passed"]),
        "raw_text": execution["raw_text"],
        "output": execution["output"],
        "status": (
            "PASS"
            if verification["passed"]
            else execution["provider_status"]
        ),
        "runner_exception": runner_exception,
    }


def new_report(schedule_hash):
    return {
        "schema_version": "0.7",
        "status": "RUNNING",
        "started_utc": utc_now(),
        "provider": "deepseek",
        "model": None,
        "task_set": EXPECTED_TASKS,
        "arms": EXPECTED_ARMS,
        "repetitions": EXPECTED_REPETITIONS,
        "scheduled_runs": EXPECTED_TOTAL_RUNS,
        "scheme": SCHEME,
        "schedule_sha256": schedule_hash,
        "persist_mode": "atomic_json_replace_per_run",
        "api_calls": 0,
        "completed_runs": 0,
        "failed_runs": 0,
        "runs": [],
    }


def validate_existing_report(report, schedule_hash):
    if report.get("schema_version") != "0.7":
        raise RuntimeError("existing report schema mismatch")

    if report.get("schedule_sha256") != schedule_hash:
        raise RuntimeError("existing report schedule hash mismatch")

    if report.get("task_set") != EXPECTED_TASKS:
        raise RuntimeError("existing report task mismatch")

    if report.get("arms") != EXPECTED_ARMS:
        raise RuntimeError("existing report arm mismatch")

    runs = report.get("runs")

    if not isinstance(runs, list):
        raise RuntimeError("existing report runs invalid")

    positions = [
        run["global_execution_position"]
        for run in runs
    ]

    if len(positions) != len(set(positions)):
        raise RuntimeError("existing report has duplicate positions")

    if any(
        p < 0 or p >= EXPECTED_TOTAL_RUNS
        for p in positions
    ):
        raise RuntimeError("existing report has invalid positions")


def print_dry_run(schedule):
    print("dry_run=YES")

    for row in schedule:
        print(
            "position={} task={} rep={} relative={} arm={}"
            .format(
                row["global_execution_position"],
                row["task"],
                row["repetition"],
                row["relative_position"],
                row["arm_id"],
            )
        )

    print("api_calls=0")
    print("V07_DRY_RUN=PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )
    args = parser.parse_args()

    print("============================================================")
    print("PROMPTFORGE V0.7 CONFIRMATORY RUNNER")
    print("============================================================")

    schedule_payload = load_and_validate_schedule()
    schedule = schedule_payload["rows"]
    schedule_hash = sha256_file(SCHEDULE_FILE)

    print("tasks={}".format(len(EXPECTED_TASKS)))
    print("arms={}".format(len(EXPECTED_ARMS)))
    print("repetitions={}".format(EXPECTED_REPETITIONS))
    print("scheduled_runs={}".format(EXPECTED_TOTAL_RUNS))
    print("scheme={}".format(SCHEME))
    print("schedule_sha256={}".format(schedule_hash))

    if args.dry_run:
        print_dry_run(schedule)
        return 0

    tasks = load_suite()
    report_path = args.report

    if report_path.exists():
        report = json.loads(
            report_path.read_text(encoding="utf-8")
        )
        validate_existing_report(
            report,
            schedule_hash,
        )
        print("resume_existing_runs={}".format(
            len(report["runs"])
        ))
    else:
        report = new_report(schedule_hash)
        atomic_write_json(report_path, report)
        print("resume_existing_runs=0")

    completed = {
        run["global_execution_position"]
        for run in report["runs"]
    }

    try:
        for row in schedule:
            position = row["global_execution_position"]

            if position in completed:
                print(
                    "SKIP position={} already persisted"
                    .format(position),
                    flush=True,
                )
                continue

            task = tasks[row["task"]]

            print(
                "START position={} task={} rep={} arm={}"
                .format(
                    position,
                    row["task"],
                    row["repetition"],
                    row["arm_id"],
                ),
                flush=True,
            )

            result = run_one(task, row)

            report["runs"].append(result)
            report["api_calls"] += 1
            report["completed_runs"] = len(report["runs"])

            if result["status"] != "PASS":
                report["failed_runs"] += 1

            if report["model"] is None:
                report["model"] = result["model"]

            completed.add(position)

            atomic_write_json(report_path, report)

            print(
                "DONE position={} status={} reasoning={} total={}"
                .format(
                    position,
                    result["status"],
                    result["metrics"]["reasoning_tokens"],
                    result["metrics"]["total_tokens"],
                ),
                flush=True,
            )

    except KeyboardInterrupt:
        report["status"] = "PARTIAL"
        report["finished_utc"] = utc_now()
        atomic_write_json(report_path, report)
        print("status=PARTIAL")
        return 130

    except Exception as exc:
        report["status"] = "PARTIAL"
        report["finished_utc"] = utc_now()
        report["runner_error"] = (
            type(exc).__name__ + ": " + str(exc)
        )
        atomic_write_json(report_path, report)
        raise

    if len(report["runs"]) == EXPECTED_TOTAL_RUNS:
        report["status"] = "COMPLETE"
    else:
        report["status"] = "PARTIAL"

    report["finished_utc"] = utc_now()
    atomic_write_json(report_path, report)

    print("status={}".format(report["status"]))
    print("runs={}".format(len(report["runs"])))
    print("api_calls={}".format(report["api_calls"]))
    print("report={}".format(report_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
