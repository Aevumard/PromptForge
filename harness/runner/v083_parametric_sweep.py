import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\PROMPTFORGE\promptforge-harness-v0.1")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUNNER_DIR = ROOT / "harness" / "runner"

SCHEDULE_FILE = RUNNER_DIR / "v083_schedule.json"
TASK_FILE = RUNNER_DIR / "v083_task_snapshot.json"
MANIFEST_FILE = RUNNER_DIR / "v083_manifest.json"
REPORT_FILE = ROOT / "harness" / "reports" / "v083_parametric_sweep_deepseek.json"

EXPECTED_RUNS = 120
TASK_ID = "T003"
MODEL_ID = "deepseek"

EXPECTED_HASHES = {
    "transforms.py": "d851443f75b02876def4a971b0c97b0a34a6de044ed49326e133a1d421ec8e4a",
    "v082_hotspot_replication.py": "3fe7fb524feb1e553ce118ff8c13cc1ed8020784de6cf50df2c430a4be530fc6",
    "deepseek_adapter.py": "4913395597afab924a8fb8927cb61288603b4b87a5b1b0bc47f6959707d0e614",
    "real_executor.py": "5589F8C777A594E81BBBFA15D14BF181B141957825DB176EAEA1ABB5DD2D9842",
}

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha256_file(path):
    return sha256_bytes(path.read_bytes())

def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with temp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)

def validate_protected():
    paths = {
        "transforms.py": RUNNER_DIR / "transforms.py",
        "v082_hotspot_replication.py": RUNNER_DIR / "v082_hotspot_replication.py",
        "deepseek_adapter.py": ROOT / "harness" / "providers" / "deepseek_adapter.py",
        "real_executor.py": RUNNER_DIR / "real_executor.py",
    }
    for name, path in paths.items():
        actual = sha256_file(path).lower()
        expected = EXPECTED_HASHES[name].lower()
        if actual != expected:
            raise RuntimeError("PROTECTED_HASH_MISMATCH:" + name)

def validate_schedule(schedule):
    payload = dict(schedule)
    stored = payload.pop("schedule_sha256")
    actual = sha256_bytes(canonical(payload))
    if stored != actual:
        raise RuntimeError("SCHEDULE_SHA_MISMATCH")
    if schedule["task_id"] != TASK_ID:
        raise RuntimeError("TASK_ID_MISMATCH")
    if schedule["model_id"] != MODEL_ID:
        raise RuntimeError("MODEL_ID_MISMATCH")
    if len(schedule["runs"]) != EXPECTED_RUNS:
        raise RuntimeError("SCHEDULE_RUN_COUNT_MISMATCH")
    positions = [int(x["global_execution_position"]) for x in schedule["runs"]]
    if positions != list(range(EXPECTED_RUNS)):
        raise RuntimeError("SCHEDULE_POSITION_MISMATCH")

def load_task():
    raw = load_json(TASK_FILE)

    if raw["task_id"] != TASK_ID:
        raise RuntimeError("TASK_SNAPSHOT_ID_MISMATCH")

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

def condition_lookup(manifest):
    return {
        x["condition_id"]: x
        for x in manifest["conditions"]
    }

def build_compiled(task, structure):
    from harness.runner.transforms import compile_v06_arm
    context = {
        "required": task["required"],
        "data": task["data"],
    }
    return compile_v06_arm(context, structure)

def serialize_context(value, serialization):
    if serialization == "pretty":
        return json.dumps(value, ensure_ascii=False, indent=2)
    if serialization == "compact":
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    raise ValueError("UNKNOWN_SERIALIZATION:" + serialization)

def build_prompt(task, compiled_value, serialization):
    context_text = serialize_context(compiled_value, serialization)
    schema_text = json.dumps(task["schema"], ensure_ascii=False, indent=2)
    prompt = (
        task["instruction"]
        + "\n\nCONTEXT:\n"
        + context_text
        + "\n\nRETURN ONLY THIS JSON OBJECT:\n"
        + schema_text
    )
    return prompt, context_text

def normalize_result(result):
    if isinstance(result, dict):
        return dict(result)
    fields = [
        "provider", "model", "status", "provider_status",
        "error", "response_id", "response_status",
        "input_tokens", "reasoning_tokens", "output_tokens",
        "total_tokens", "latency_ms", "raw_text", "text"
    ]
    output = {}
    for field in fields:
        if hasattr(result, field):
            output[field] = getattr(result, field)
    return output

def preflight(task, conditions):
    results = []
    for condition in conditions.values():
        compiled = build_compiled(task, condition["arm_id"])
        prompt, context_text = build_prompt(
            task, compiled["value"], condition["serialization"]
        )
        if not prompt:
            raise RuntimeError("EMPTY_PROMPT")
        results.append({
            "condition_id": condition["condition_id"],
            "structure": condition["structure"],
            "serialization": condition["serialization"],
            "context_chars": len(context_text),
            "prompt_chars": len(prompt),
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        })
    return results

def execute_one(task, condition, repetition, position, adapter):
    from harness.runner.v082_hotspot_replication import parse_json, evaluate

    compiled = build_compiled(task, condition["arm_id"])
    prompt, context_text = build_prompt(
        task, compiled["value"], condition["serialization"]
    )

    started = time.monotonic()
    started_utc = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()

    provider_exception = None

    try:
        normalized = adapter.generate(prompt)
        execution = normalize_result(normalized)
    except Exception as exc:
        provider_exception = f"{type(exc).__name__}: {exc}"
        execution = {
            "provider": MODEL_ID,
            "model": MODEL_ID,
            "provider_status": "PROVIDER_EXCEPTION",
            "status": "PROVIDER_EXCEPTION",
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

    raw_text = execution.get("raw_text", execution.get("text", ""))
    if not isinstance(raw_text, str):
        raw_text = ""

    provider_status = execution.get(
        "provider_status",
        execution.get("status", "ERROR")
    )

    parsed = None
    parse_error = None

    if provider_status == "MODEL_OK":
        parsed, parse_error = parse_json(raw_text)
        if parse_error is not None:
            provider_status = "OUTPUT_ERROR"

    verification = {
        "evaluator": "deterministic_exact_fields",
        "passed": False,
        "missing_or_incorrect": [],
    }

    if parsed is not None:
        passed, verification_result = evaluate(task["expected"], parsed)
        verification = dict(verification_result)
        verification["passed"] = bool(passed)

    elapsed = (time.monotonic() - started) * 1000.0

    metrics = {
        "input_tokens": int(execution.get("input_tokens", 0) or 0),
        "reasoning_tokens": int(execution.get("reasoning_tokens", 0) or 0),
        "output_tokens": int(execution.get("output_tokens", 0) or 0),
        "total_tokens": int(execution.get("total_tokens", 0) or 0),
        "latency_ms": float(execution.get("latency_ms", 0) or 0),
        "context_chars": len(context_text),
        "prompt_chars": len(prompt),
    }

    finished_utc = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()

    return {
        "schema_version": "0.8.3",
        "run_id": (
            f"T003::deepseek::{condition['condition_id']}::rep_{repetition:03d}"
        ),
        "timestamp_utc": started_utc,
        "task_id": TASK_ID,
        "task_family": task["task_family"],
        "provider": execution.get("provider", MODEL_ID),
        "model_id": MODEL_ID,
        "model": execution.get("model"),
        "condition_id": condition["condition_id"],
        "structure": condition["structure"],
        "serialization": condition["serialization"],
        "arm_id": condition["arm_id"],
        "transform_id": compiled["transform_id"],
        "transform_sequence": compiled["transform_sequence"],
        "repetition": repetition,
        "global_execution_position": position,
        "context_change": {
            "included": compiled["included"],
            "excluded": compiled["excluded"],
        },
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
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
        "output": parsed,
        "status": "PASS" if verification["passed"] else provider_status,
        "end_utc": finished_utc,
        "monotonic_elapsed_ms": elapsed,
    }

def make_report(schedule, manifest, preflight_results):
    return {
        "schema_version": "0.8.3",
        "experiment_id": "PROMPTFORGE_V0.8.3_PARAMETRIC_SWEEP",
        "status": "RUNNING",
        "started_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "provider_phase": "REAL_COLLECTION",
        "task_id": TASK_ID,
        "model_id": MODEL_ID,
        "repetitions": 20,
        "scheduled_runs": EXPECTED_RUNS,
        "schedule_sha256": schedule["schedule_sha256"],
        "task_snapshot_sha256": sha256_file(TASK_FILE),
        "protected_sha256": EXPECTED_HASHES,
        "preflight": preflight_results,
        "runs": [],
    }

def dry_run():
    validate_protected()
    schedule = load_json(SCHEDULE_FILE)
    validate_schedule(schedule)
    manifest = load_json(MANIFEST_FILE)
    task = load_task()
    conditions = condition_lookup(manifest)
    preflight_results = preflight(task, conditions)
    print("V083_PROTECTED=PASS")
    print("V083_SCHEDULE=PASS")
    print("V083_PREFLIGHT=PASS")
    print("V083_CONDITIONS=" + str(len(preflight_results)))
    print("V083_RUNS=120")
    print("V083_API_CALLS=0")

def real_run():
    validate_protected()
    schedule = load_json(SCHEDULE_FILE)
    validate_schedule(schedule)
    manifest = load_json(MANIFEST_FILE)
    task = load_task()
    conditions = condition_lookup(manifest)

    preflight_results = preflight(task, conditions)

    print("V083_PROTECTED=PASS")
    print("V083_SCHEDULE=PASS")
    print("V083_PREFLIGHT=PASS")

    from harness.providers.deepseek_adapter import DeepSeekProviderAdapter
    adapter = DeepSeekProviderAdapter()

    if REPORT_FILE.exists():
        report = load_json(REPORT_FILE)
        if report.get("schedule_sha256") != schedule["schedule_sha256"]:
            raise RuntimeError("REPORT_SCHEDULE_MISMATCH")
    else:
        report = make_report(schedule, manifest, preflight_results)
        atomic_write(REPORT_FILE, report)

    completed = {
        int(x["global_execution_position"])
        for x in report["runs"]
    }

    print("RESUME_EXISTING_RUNS=" + str(len(completed)))
    print("REAL_COLLECTION=START")

    for row in schedule["runs"]:
        position = int(row["global_execution_position"])
        if position in completed:
            continue

        condition = conditions[row["condition_id"]]
        repetition = int(row["repetition"])

        print(
            f"START position={position} condition={row['condition_id']} rep={repetition}",
            flush=True
        )

        result = execute_one(
            task, condition, repetition, position, adapter
        )

        report["runs"].append(result)
        report["runs"].sort(
            key=lambda x: int(x["global_execution_position"])
        )
        atomic_write(REPORT_FILE, report)

        m = result["metrics"]

        print(
            f"DONE position={position} condition={row['condition_id']} "
            f"rep={repetition} status={result['status']} "
            f"quality={result['quality_pass']} "
            f"input={m['input_tokens']} "
            f"reasoning={m['reasoning_tokens']} "
            f"output={m['output_tokens']} "
            f"total={m['total_tokens']} "
            f"latency_ms={m['latency_ms']}",
            flush=True
        )

    if len(report["runs"]) != EXPECTED_RUNS:
        raise RuntimeError("RUN_COUNT_INCOMPLETE")

    positions = sorted(
        int(x["global_execution_position"])
        for x in report["runs"]
    )

    if positions != list(range(EXPECTED_RUNS)):
        raise RuntimeError("FINAL_POSITION_SET_INVALID")

    report["status"] = "COMPLETE"
    report["completed_runs"] = len(report["runs"])
    report["last_completed_position"] = max(positions)
    report["finished_utc"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()

    atomic_write(REPORT_FILE, report)

    print("")
    print("V083_REAL_COLLECTION=COMPLETE")
    print("V083_RUNS=" + str(len(report["runs"])))
    print("V083_REPORT=" + str(REPORT_FILE))

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--real-run", action="store_true")
args = parser.parse_args()

if args.dry_run == args.real_run:
    raise SystemExit("SELECT_ONE_MODE")

if args.dry_run:
    dry_run()
else:
    real_run()
