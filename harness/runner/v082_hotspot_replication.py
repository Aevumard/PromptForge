import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from harness.evaluators.deterministic import evaluate
from harness.providers.deepseek_adapter import DeepSeekProviderAdapter
from harness.runner.transforms import compile_v06_arm


ROOT = Path(__file__).resolve().parents[2]

TASK_FILE = ROOT / "tasks" / "suite.json"

SCHEDULE_FILE = (
    ROOT
    / "harness"
    / "runner"
    / "v082_hotspot_schedule.json"
)

REPORT_FILE = (
    ROOT
    / "harness"
    / "reports"
    / "v082_hotspot_replication_deepseek.json"
)

TASK_ID = "T003"
MODEL_ID = "deepseek"

ARMS = [
    "noop",
    "representation",
]

REPETITIONS = 30
EXPECTED_RUNS = 60

EXPERIMENT_ID = (
    "PROMPTFORGE_V0.8.2_HOTSPOT_REPLICATION"
)

SCHEMA_VERSION = "0.8.2"

EXPECTED_EXECUTOR_SHA256 = (
    "5589F8C777A594E81BBBFA15D14BF181B141957825DB176EAEA1ABB5DD2D9842"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def schedule_hash(document):

    payload = {
        key: value
        for key, value in document.items()
        if key != "schedule_sha256"
    }

    return hashlib.sha256(
        canonical_json(payload)
    ).hexdigest()


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


def build_prompt(task, compiled):

    return (
        task["instruction"]
        + "\n\nCONTEXT:\n"
        + json.dumps(
            compiled["value"],
            ensure_ascii=False,
            indent=2,
        )
        + "\n\nRETURN ONLY THIS JSON OBJECT:\n"
        + json.dumps(
            task["schema"],
            ensure_ascii=False,
            indent=2,
        )
    )


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
                return (
                    json.loads(
                        text[start:end + 1]
                    ),
                    None,
                )

            except Exception as exc:

                return (
                    None,
                    f"json_parse_error: {exc}",
                )

        return (
            None,
            "json_parse_error: no_object",
        )


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

    output = {}

    for field in fields:

        if hasattr(result, field):
            output[field] = getattr(
                result,
                field,
            )

    return output


def atomic_write(path, payload):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    with temp.open("wb") as handle:

        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())

    temp.replace(path)


def load_schedule():

    document = json.loads(
        SCHEDULE_FILE.read_text(
            encoding="utf-8"
        )
    )

    positions = [
        int(
            row["global_execution_position"]
        )
        for row in document["runs"]
    ]

    if positions != list(range(EXPECTED_RUNS)):
        raise RuntimeError(
            "SCHEDULE_POSITIONS_INVALID"
        )

    if document["task_id"] != TASK_ID:
        raise RuntimeError(
            "SCHEDULE_TASK_INVALID"
        )

    if document["model_id"] != MODEL_ID:
        raise RuntimeError(
            "SCHEDULE_MODEL_INVALID"
        )

    if document["arms"] != ARMS:
        raise RuntimeError(
            "SCHEDULE_ARMS_INVALID"
        )

    if int(
        document["repetitions"]
    ) != REPETITIONS:
        raise RuntimeError(
            "SCHEDULE_REPETITIONS_INVALID"
        )

    if int(
        document["scheduled_runs"]
    ) != EXPECTED_RUNS:
        raise RuntimeError(
            "SCHEDULE_RUNS_INVALID"
        )

    if (
        document["schedule_sha256"]
        != schedule_hash(document)
    ):
        raise RuntimeError(
            "SCHEDULE_HASH_INVALID"
        )

    return document


def validate_environment(
    schedule_document,
):

    executor = (
        ROOT
        / "harness"
        / "runner"
        / "real_executor.py"
    )

    executor_hash = (
        hashlib.sha256(
            executor.read_bytes()
        )
        .hexdigest()
        .upper()
    )

    if (
        executor_hash
        != EXPECTED_EXECUTOR_SHA256
    ):
        raise RuntimeError(
            "EXECUTOR_CHANGED="
            + executor_hash
        )

    if (
        schedule_document["experiment_id"]
        != EXPERIMENT_ID
    ):
        raise RuntimeError(
            "EXPERIMENT_ID_INVALID"
        )

    suite = json.loads(
        TASK_FILE.read_text(
            encoding="utf-8"
        )
    )

    task_ids = {
        item["task_id"]
        for item in suite["tasks"]
    }

    if TASK_ID not in task_ids:
        raise RuntimeError(
            "T003_MISSING"
        )


def make_report(schedule_document):

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "RUNNING",
        "started_utc": utc_now(),
        "provider_phase": "REAL_COLLECTION",
        "tasks": [TASK_ID],
        "models": [MODEL_ID],
        "arms": ARMS,
        "repetitions": REPETITIONS,
        "scheduled_runs": EXPECTED_RUNS,
        "scheme": (
            schedule_document["scheme"]
        ),
        "schedule_sha256": (
            schedule_document[
                "schedule_sha256"
            ]
        ),
        "executor_sha256": (
            EXPECTED_EXECUTOR_SHA256
        ),
        "runs": [],
    }


def execute_one(
    task,
    arm_id,
    repetition,
    position,
    adapter,
):

    context = {
        "required": task["required"],
        "data": task["data"],
    }

    compiled_arm = (
        "representation_only"
        if arm_id == "representation"
        else arm_id
    )

    compiled = compile_v06_arm(
        context,
        compiled_arm,
    )

    prompt = build_prompt(
        task,
        compiled,
    )

    started_utc = utc_now()
    started_monotonic = time.monotonic()

    provider_exception = None

    try:

        normalized = adapter.generate(
            prompt
        )

        execution = result_to_dict(
            normalized
        )

    except Exception as exc:

        provider_exception = (
            f"{type(exc).__name__}: {exc}"
        )

        execution = {
            "provider": MODEL_ID,
            "model": MODEL_ID,
            "status": (
                "PROVIDER_EXCEPTION"
            ),
            "provider_status": (
                "PROVIDER_EXCEPTION"
            ),
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

    finished_monotonic = time.monotonic()
    finished_utc = utc_now()

    raw_text = execution.get(
        "raw_text",
        execution.get(
            "text",
            "",
        ),
    )

    provider_status = execution.get(
        "provider_status",
        execution.get(
            "status",
            "ERROR",
        ),
    )

    parsed_output = None
    parse_error = None

    if provider_status == "MODEL_OK":

        parsed_output, parse_error = (
            parse_json(raw_text)
        )

        if parse_error is not None:

            provider_status = (
                "OUTPUT_ERROR"
            )

    verification = {
        "evaluator": (
            "deterministic_exact_fields"
        ),
        "passed": False,
        "missing_or_incorrect": [],
    }

    if parsed_output is not None:

        passed, verification_result = evaluate(
            task["expected"],
            parsed_output,
        )

        verification = dict(
            verification_result
        )

        verification["passed"] = bool(
            passed
        )

    metrics = {
        "input_tokens": int(
            execution.get(
                "input_tokens",
                0,
            )
            or 0
        ),
        "reasoning_tokens": int(
            execution.get(
                "reasoning_tokens",
                0,
            )
            or 0
        ),
        "output_tokens": int(
            execution.get(
                "output_tokens",
                0,
            )
            or 0
        ),
        "total_tokens": int(
            execution.get(
                "total_tokens",
                0,
            )
            or 0
        ),
        "latency_ms": float(
            execution.get(
                "latency_ms",
                0,
            )
            or 0
        ),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": (
            f"{TASK_ID}::"
            f"{MODEL_ID}::"
            f"{arm_id}::"
            f"rep_{repetition:03d}"
        ),
        "timestamp_utc": started_utc,
        "task_id": TASK_ID,
        "task_family": task[
            "task_family"
        ],
        "provider": execution.get(
            "provider",
            MODEL_ID,
        ),
        "model_id": MODEL_ID,
        "model": execution.get(
            "model"
        ),
        "arm_id": arm_id,
        "transform_id": compiled[
            "transform_id"
        ],
        "transform_sequence": compiled[
            "transform_sequence"
        ],
        "repetition": repetition,
        "global_execution_position": position,
        "context_change": {
            "included": compiled[
                "included"
            ],
            "excluded": compiled[
                "excluded"
            ],
        },
        "metrics": metrics,
        "execution": {
            "provider_status": provider_status,
            "response_status": execution.get(
                "response_status"
            ),
            "response_id": execution.get(
                "response_id"
            ),
            "error": (
                parse_error
                or provider_exception
                or execution.get(
                    "error"
                )
            ),
        },
        "verification": verification,
        "quality_pass": bool(
            verification["passed"]
        ),
        "raw_text": raw_text,
        "output": parsed_output,
        "status": (
            "PASS"
            if verification["passed"]
            else provider_status
        ),
        "start_utc": started_utc,
        "end_utc": finished_utc,
        "monotonic_elapsed_ms": (
            finished_monotonic
            - started_monotonic
        ) * 1000.0,
    }


def dry_run():

    schedule_document = load_schedule()

    validate_environment(
        schedule_document
    )

    print(
        "V082_VALIDATE_SCHEDULE=PASS"
    )

    print(
        "V082_VALIDATE_ENVIRONMENT=PASS"
    )

    print(
        "V082_DRY_RUN=PASS"
    )

    print(
        "V082_RUNS=60"
    )

    print(
        "V082_MODEL=deepseek"
    )

    print(
        "V082_TASK=T003"
    )

    print(
        "V082_ARMS=noop,representation"
    )

    print(
        "V082_API_CALLS=0"
    )


def real_run():

    schedule_document = load_schedule()

    validate_environment(
        schedule_document
    )

    suite = json.loads(
        TASK_FILE.read_text(
            encoding="utf-8"
        )
    )

    raw_task = next(
        item
        for item in suite["tasks"]
        if item["task_id"] == TASK_ID
    )

    task = build_task(
        raw_task
    )

    adapter = (
        DeepSeekProviderAdapter()
    )

    if REPORT_FILE.exists():

        report = json.loads(
            REPORT_FILE.read_text(
                encoding="utf-8"
            )
        )

        if (
            report.get(
                "schedule_sha256"
            )
            != schedule_document[
                "schedule_sha256"
            ]
        ):
            raise RuntimeError(
                "REPORT_SCHEDULE_MISMATCH"
            )

    else:

        report = make_report(
            schedule_document
        )

        atomic_write(
            REPORT_FILE,
            report
        )

    completed_positions = {
        int(
            row[
                "global_execution_position"
            ]
        )
        for row in report["runs"]
    }

    print(
        "REAL_COLLECTION=START"
    )

    print(
        "RESUME_EXISTING_RUNS="
        + str(
            len(
                completed_positions
            )
        )
    )

    for row in schedule_document["runs"]:

        position = int(
            row[
                "global_execution_position"
            ]
        )

        if position in completed_positions:
            continue

        repetition = int(
            row["repetition"]
        )

        arm_id = row["arm_id"]

        print(
            "START "
            f"position={position} "
            f"arm={arm_id} "
            f"rep={repetition}",
            flush=True,
        )

        result = execute_one(
            task,
            arm_id,
            repetition,
            position,
            adapter,
        )

        report["runs"].append(
            result
        )

        report["runs"].sort(
            key=lambda item: int(
                item[
                    "global_execution_position"
                ]
            )
        )

        atomic_write(
            REPORT_FILE,
            report
        )

        metrics = result[
            "metrics"
        ]

        print(
            "DONE "
            f"position={position} "
            f"arm={arm_id} "
            f"rep={repetition} "
            f"status={result['status']} "
            f"quality={result['quality_pass']} "
            f"input={metrics['input_tokens']} "
            f"reasoning={metrics['reasoning_tokens']} "
            f"output={metrics['output_tokens']} "
            f"total={metrics['total_tokens']}",
            flush=True,
        )

    if len(
        report["runs"]
    ) != EXPECTED_RUNS:

        raise RuntimeError(
            "RUN_COUNT_INCOMPLETE"
        )

    report["status"] = (
        "COMPLETE"
    )

    report[
        "last_completed_position"
    ] = max(
        int(
            row[
                "global_execution_position"
            ]
        )
        for row in report["runs"]
    )

    report[
        "completed_runs"
    ] = len(
        report["runs"]
    )

    report[
        "finished_utc"
    ] = utc_now()

    atomic_write(
        REPORT_FILE,
        report
    )

    print("")
    print(
        "V082_REAL_COLLECTION=COMPLETE"
    )
    print(
        f"V082_RUNS={len(report['runs'])}"
    )
    print(
        f"V082_REPORT={REPORT_FILE}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--real",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        not args.dry_run
        and not args.real
    ):
        parser.error(
            "use --dry-run or --real"
        )

    if args.dry_run:
        dry_run()
        return

    real_run()


if __name__ == "__main__":
    main()