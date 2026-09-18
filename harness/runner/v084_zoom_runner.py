import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK_FILE = ROOT / "tasks" / "suite.json"
SCHEDULE_FILE = ROOT / "harness" / "runner" / "v084_zoom_schedule.json"
REPORT_FILE = ROOT / "harness" / "reports" / "v084_zoom_deepseek.json"

TASK_ID = "T003"
EXPERIMENT_ID = "PROMPTFORGE_V0.8.4_ZOOM"
MODEL_ID = "deepseek"
EXPECTED_RUNS = 120
REPETITIONS = 30

CONDITIONS = [
    "noop_compact",
    "pairs_compact",
    "pairs_bare_compact",
    "flat_compact",
]

EXPECTED_SCHEDULE_SHA256 = "4D6B9B81CB2C2B10126E549469616539342CC7FCC3ADF40C83158E4989EDB23F"
EXPECTED_TRANSFORMS_SHA256 = "D851443F75B02876DEF4A971B0C97B0A34A6DE044ED49326E133A1D421EC8E4A"
EXPECTED_EXECUTOR_SHA256 = "5589F8C777A594E81BBBFA15D14BF181B141957825DB176EAEA1ABB5DD2D9842"
EXPECTED_DEEPSEEK_SHA256 = "4913395597AFAB924A8FB8927CB61288603B4B87A5B1B0BC47F6959707D0E614"

PROTECTED_FILES = {
    "transforms.py": ROOT / "harness" / "runner" / "transforms.py",
    "real_executor.py": ROOT / "harness" / "runner" / "real_executor.py",
    "deepseek_adapter.py": ROOT / "harness" / "providers" / "deepseek_adapter.py",
}

sys.path.insert(0, str(ROOT))

from harness.evaluators.deterministic import evaluate
from harness.providers.deepseek_adapter import DeepSeekProviderAdapter
from harness.runner.transforms import compile_v06_arm


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    with temp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


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


def result_to_dict(result):
    if isinstance(result, dict):
        return dict(result)

    fields = [
        "provider",
        "model",
        "status",
        "provider_status",
        "error",
        "response_id",
        "response_status",
        "input_tokens",
        "reasoning_tokens",
        "output_tokens",
        "total_tokens",
        "latency_ms",
        "raw_text",
        "text",
    ]

    out = {}
    for field in fields:
        if hasattr(result, field):
            out[field] = getattr(result, field)

    return out


def parse_json(text):
    if not text:
        return None, "empty_model_output"

    text = text.strip()

    try:
        return json.loads(text), None
    except Exception:
        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1]), None
            except Exception as exc:
                return None, "json_parse_error: {}".format(exc)

        return None, "json_parse_error: no_object"


def build_context(task):
    return {
        "required": task["required"],
        "data": task["data"],
    }


def compile_condition(task, condition):
    context = build_context(task)

    if condition == "noop_compact":
        compiled = compile_v06_arm(context, "noop")
        serialized_value = compiled["value"]

    elif condition == "pairs_compact":
        compiled = compile_v06_arm(context, "representation_B")
        serialized_value = compiled["value"]

    elif condition == "pairs_bare_compact":
        compiled = compile_v06_arm(context, "representation_B")
        serialized_value = compiled["value"]["pairs"]

    elif condition == "flat_compact":
        compiled = compile_v06_arm(context, "noop")
        serialized_value = task["data"]

    else:
        raise ValueError("unknown V0.8.4 condition: {}".format(condition))

    return compiled, serialized_value


def build_prompt(task, serialized_value):
    return (
        task["instruction"]
        + "\n\nCONTEXT:\n"
        + json.dumps(
            serialized_value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n\nRETURN ONLY THIS JSON OBJECT:\n"
        + json.dumps(
            task["schema"],
            ensure_ascii=False,
            indent=2,
        )
    )


def load_schedule():
    raw = json.loads(
        SCHEDULE_FILE.read_text(encoding="utf-8")
    )
    return raw


def validate_schedule(schedule):
    if schedule["experiment"] != "V0.8.4_ZOOM":
        raise RuntimeError("SCHEDULE_EXPERIMENT_ID=FAIL")

    if schedule["task_ids"] != [TASK_ID]:
        raise RuntimeError("SCHEDULE_TASKS=FAIL")

    if int(schedule["repetitions"]) != REPETITIONS:
        raise RuntimeError("SCHEDULE_REPETITIONS=FAIL")

    if schedule["conditions"] != CONDITIONS:
        raise RuntimeError("SCHEDULE_CONDITIONS=FAIL")

    if int(schedule["runs"]) != EXPECTED_RUNS:
        raise RuntimeError("SCHEDULE_RUN_COUNT=FAIL")

    rows = schedule["rows"]

    if len(rows) != EXPECTED_RUNS:
        raise RuntimeError("SCHEDULE_ROWS=FAIL")

    positions = [
        int(row["global_execution_pos"])
        for row in rows
    ]

    if positions != list(range(EXPECTED_RUNS)):
        raise RuntimeError("SCHEDULE_POSITIONS=FAIL")

    for repetition in range(1, REPETITIONS + 1):
        current = [
            row["arm_id"]
            for row in rows
            if int(row["repetition"]) == repetition
        ]

        if len(current) != 4:
            raise RuntimeError("SCHEDULE_REPETITION_SIZE=FAIL")

        if set(current) != set(CONDITIONS):
            raise RuntimeError("SCHEDULE_REPETITION_CONDITIONS=FAIL")


def validate_environment(schedule):
    actual_schedule_sha = sha256_file(SCHEDULE_FILE)

    if actual_schedule_sha != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError("SCHEDULE_SHA256=FAIL")

    expected = {
        "transforms.py": EXPECTED_TRANSFORMS_SHA256,
        "real_executor.py": EXPECTED_EXECUTOR_SHA256,
        "deepseek_adapter.py": EXPECTED_DEEPSEEK_SHA256,
    }

    for name, expected_hash in expected.items():
        path = PROTECTED_FILES[name]

        if not path.exists():
            raise RuntimeError("PROTECTED_FILE_MISSING=" + name)

        actual = sha256_file(path)

        if actual != expected_hash:
            raise RuntimeError("PROTECTED_HASH_FAIL=" + name)

    return actual_schedule_sha


def preflight_task():
    suite = json.loads(
        TASK_FILE.read_text(encoding="utf-8")
    )

    raw_task = next(
        item
        for item in suite["tasks"]
        if item["task_id"] == TASK_ID
    )

    return build_task(raw_task)


def preflight(task):
    expected = {
        "noop_compact": (85, 307),
        "pairs_compact": (105, 327),
        "pairs_bare_compact": (95, 317),
        "flat_compact": (85, 307),
    }

    observed = {}

    for condition in CONDITIONS:
        compiled, serialized_value = compile_condition(task, condition)
        context_chars = len(
            json.dumps(
                serialized_value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        prompt = build_prompt(task, serialized_value)
        prompt_chars = len(prompt)

        observed[condition] = {
            "context_chars": context_chars,
            "prompt_chars": prompt_chars,
            "transform_id": compiled["transform_id"],
            "transform_sequence": compiled["transform_sequence"],
        }

        if (context_chars, prompt_chars) != expected[condition]:
            raise RuntimeError(
                "PREFLIGHT_SIZE_FAIL=" + condition
            )

    return observed


def make_report(schedule, schedule_sha, prereg_sha):
    return {
        "schema_version": "0.8.4",
        "experiment_id": EXPERIMENT_ID,
        "status": "RUNNING",
        "provider_phase": "REAL_COLLECTION",
        "task_id": TASK_ID,
        "model_id": MODEL_ID,
        "conditions": CONDITIONS,
        "repetitions": REPETITIONS,
        "scheduled_runs": EXPECTED_RUNS,
        "schedule_sha256": schedule_sha,
        "preregistration_sha256": prereg_sha,
        "protected_sha256": {
            "transforms.py": EXPECTED_TRANSFORMS_SHA256,
            "real_executor.py": EXPECTED_EXECUTOR_SHA256,
            "deepseek_adapter.py": EXPECTED_DEEPSEEK_SHA256,
        },
        "runs": [],
    }


def execute_one(task, condition, repetition, position, adapter):
    compiled, serialized_value = compile_condition(task, condition)
    prompt = build_prompt(task, serialized_value)

    started_utc = utc_now()
    started_monotonic = time.monotonic()

    provider_exception = None

    try:
        normalized = adapter.generate(prompt)
        execution = result_to_dict(normalized)
    except Exception as exc:
        provider_exception = "{}: {}".format(
            type(exc).__name__,
            exc,
        )
        execution = {
            "provider": "deepseek",
            "model": MODEL_ID,
            "status": "PROVIDER_EXCEPTION",
            "provider_status": "PROVIDER_EXCEPTION",
            "error": provider_exception,
            "response_id": None,
            "response_status": None,
            "input_tokens": 0,
            "reasoning_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0,
            "raw_text": "",
        }

    finished_utc = utc_now()
    finished_monotonic = time.monotonic()

    raw_text = execution.get(
        "raw_text",
        execution.get("text", ""),
    )

    provider_status = execution.get(
        "provider_status",
        execution.get("status", "ERROR"),
    )

    parsed_output = None
    parse_error = None

    if provider_status == "MODEL_OK":
        parsed_output, parse_error = parse_json(raw_text)
        if parse_error is not None:
            provider_status = "OUTPUT_ERROR"

    verification = {
        "evaluator": "deterministic_exact_fields",
        "passed": False,
        "missing_or_incorrect": [],
    }

    if parsed_output is not None:
        passed, verification_result = evaluate(
            task["expected"],
            parsed_output,
        )
        verification = dict(verification_result)
        verification["passed"] = bool(passed)

    metrics = {
        "input_tokens": int(execution.get("input_tokens", 0) or 0),
        "reasoning_tokens": int(execution.get("reasoning_tokens", 0) or 0),
        "output_tokens": int(execution.get("output_tokens", 0) or 0),
        "total_tokens": int(execution.get("total_tokens", 0) or 0),
        "latency_ms": float(execution.get("latency_ms", 0) or 0),
    }

    return {
        "schema_version": "0.8.4",
        "run_id": "T003::deepseek::{}::rep_{:03d}".format(condition, repetition),
        "timestamp_utc": started_utc,
        "task_id": TASK_ID,
        "provider": execution.get("provider", MODEL_ID),
        "model_id": MODEL_ID,
        "model": execution.get("model"),
        "condition_id": condition,
        "transform_id": compiled["transform_id"],
        "transform_sequence": compiled["transform_sequence"],
        "serialization": "compact",
        "representation_payload": "pairs_bare" if condition == "pairs_bare_compact" else condition.split("_compact")[0],
        "repetition": repetition,
        "global_execution_position": position,
        "prompt_chars": len(prompt),
        "context_chars": len(json.dumps(serialized_value, ensure_ascii=False, separators=(",", ":"))),
        "metrics": metrics,
        "execution": {
            "provider_status": provider_status,
            "response_status": execution.get("response_status"),
            "response_id": execution.get("response_id"),
            "error": parse_error or provider_exception or execution.get("error"),
        },
        "verification": verification,
        "quality_pass": bool(verification["passed"]),
        "raw_text": raw_text,
        "output": parsed_output,
        "status": "PASS" if verification["passed"] else provider_status,
        "start_utc": started_utc,
        "end_utc": finished_utc,
        "monotonic_elapsed_ms": (finished_monotonic - started_monotonic) * 1000.0,
    }


def dry_run():
    schedule = load_schedule()
    validate_schedule(schedule)
    schedule_sha = validate_environment(schedule)

    task = preflight_task()
    observed = preflight(task)

    print("V084_VALIDATE_SCHEDULE=PASS")
    print("V084_VALIDATE_ENVIRONMENT=PASS")
    print("V084_PREFLIGHT=PASS")
    print("V084_TASK=T003")
    print("V084_MODEL=deepseek")
    print("V084_CONDITIONS=4")
    print("V084_RUNS=120")
    print("V084_SCHEDULE_SHA256=" + schedule_sha)

    for condition in CONDITIONS:
        data = observed[condition]
        print(
            "V084_PREFLIGHT_{}={}::{}".format(
                condition.upper(),
                data["context_chars"],
                data["prompt_chars"],
            )
        )

    print("V084_API_CALLS=0")
    print("V084_DRY_RUN=PASS")


def real_run():
    schedule = load_schedule()
    validate_schedule(schedule)
    schedule_sha = validate_environment(schedule)

    prereg_path = ROOT / "harness" / "prereg" / "v084_zoom_preregistration.md"
    prereg_sha = sha256_file(prereg_path)

    task = preflight_task()
    preflight(task)

    if REPORT_FILE.exists():
        report = json.loads(
            REPORT_FILE.read_text(encoding="utf-8")
        )

        if report.get("schedule_sha256") != schedule_sha:
            raise RuntimeError("REPORT_SCHEDULE_SHA256=FAIL")

        if report.get("preregistration_sha256") != prereg_sha:
            raise RuntimeError("REPORT_PREREGISTRATION_SHA256=FAIL")
    else:
        report = make_report(
            schedule,
            schedule_sha,
            prereg_sha,
        )
        atomic_write(REPORT_FILE, report)

    completed_positions = {
        int(row["global_execution_position"])
        for row in report["runs"]
    }

    adapter = DeepSeekProviderAdapter()

    print("V084_REAL_COLLECTION=START")
    print("RESUME_EXISTING_RUNS=" + str(len(completed_positions)))

    for row in schedule["rows"]:
        position = int(row["global_execution_pos"])

        if position in completed_positions:
            continue

        repetition = int(row["repetition"])
        condition = row["arm_id"]

        print(
            "START position={} condition={} rep={}".format(
                position,
                condition,
                repetition,
            ),
            flush=True,
        )

        result = execute_one(
            task,
            condition,
            repetition,
            position,
            adapter,
        )

        report["runs"].append(result)
        report["runs"].sort(
            key=lambda item: int(item["global_execution_position"])
        )

        atomic_write(REPORT_FILE, report)

        m = result["metrics"]

        print(
            "DONE position={} condition={} rep={} status={} quality={} input={} reasoning={} output={} total={}".format(
                position,
                condition,
                repetition,
                result["status"],
                result["quality_pass"],
                m["input_tokens"],
                m["reasoning_tokens"],
                m["output_tokens"],
                m["total_tokens"],
            ),
            flush=True,
        )

    if len(report["runs"]) != EXPECTED_RUNS:
        raise RuntimeError("RUN_COUNT_INCOMPLETE")

    report["status"] = "COMPLETE"
    report["completed_runs"] = len(report["runs"])
    report["last_completed_position"] = max(
        int(row["global_execution_position"])
        for row in report["runs"]
    )
    report["finished_utc"] = utc_now()

    atomic_write(REPORT_FILE, report)

    print("V084_REAL_COLLECTION=COMPLETE")
    print("V084_RUNS=" + str(len(report["runs"])))
    print("V084_REPORT=" + str(REPORT_FILE))


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--real-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    real_run()


if __name__ == "__main__":
    main()
