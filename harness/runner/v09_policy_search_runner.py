from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASK_FILE = ROOT / "tasks" / "suite.json"

SCHEDULE_FILE = (
    ROOT
    / "harness"
    / "runner"
    / "v09_policy_search_schedule.json"
)

PREREG_FILE = (
    ROOT
    / "harness"
    / "prereg"
    / "v09_policy_search_preregistration.md"
)

REPORT_FILE = (
    ROOT
    / "harness"
    / "reports"
    / "v09_policy_discovery_deepseek.json"
)

TASKS = [
    "T001",
    "T002",
    "T003",
    "T004",
]

CONDITIONS = [
    "noop_compact",
    "pairs_compact",
    "pairs_bare_compact",
    "flat_compact",
]

REPETITIONS = 15
EXPECTED_RUNS = 240


def load_v084():
    path = (
        ROOT
        / "harness"
        / "runner"
        / "v084_zoom_runner.py"
    )

    spec = importlib.util.spec_from_file_location(
        "promptforge_v084",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("V084_IMPORT_FAIL")

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


V084 = load_v084()

from harness.evaluators.deterministic import evaluate
from harness.providers.deepseek_adapter import DeepSeekProviderAdapter


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest().upper()


def atomic_write(path, payload):
    temp = path.with_suffix(".tmp")

    temp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    temp.replace(path)


def build_task(raw):
    task = dict(raw)

    task["instruction"] = (
        "Extract the required fields from "
        "the supplied context. "
        "Return only valid JSON. "
        "Do not explain. "
        "Do not use markdown."
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


def load_schedule():
    schedule = json.loads(
        SCHEDULE_FILE.read_text(
            encoding="utf-8"
        )
    )

    if len(schedule["rows"]) != EXPECTED_RUNS:
        raise RuntimeError("SCHEDULE_SIZE_FAIL")

    positions = [
        row["global_execution_position"]
        for row in schedule["rows"]
    ]

    if positions != list(range(EXPECTED_RUNS)):
        raise RuntimeError("SCHEDULE_POSITION_FAIL")

    return schedule


def validate_environment():

    expected = {
        "transforms.py": (
            "D851443F75B02876DEF4A971B0C97B0A34A6DE044ED49326E133A1D421EC8E4A"
        ),
        "real_executor.py": (
            "5589F8C777A594E81BBBFA15D14BF181B141957825DB176EAEA1ABB5DD2D9842"
        ),
        "deepseek_adapter.py": (
            "4913395597AFAB924A8FB8927CB61288603B4B87A5B1B0BC47F6959707D0E614"
        ),
        "v084_zoom_runner.py": (
            "DA6A123B9CE8F3D6066D6A97E81EEEE3EDAA5CE9B37EF6051CBB5C94A3167C06"
        ),
    }

    files = {
        "transforms.py":
            ROOT / "harness" / "runner" / "transforms.py",

        "real_executor.py":
            ROOT / "harness" / "runner" / "real_executor.py",

        "deepseek_adapter.py":
            ROOT / "harness" / "providers" / "deepseek_adapter.py",

        "v084_zoom_runner.py":
            ROOT / "harness" / "runner" / "v084_zoom_runner.py",
    }

    for name in expected:

        actual = sha256_file(files[name])

        if actual != expected[name]:
            raise RuntimeError(
                "PROTECTED_HASH_FAIL=" + name
            )


def preflight():

    suite = json.loads(
        TASK_FILE.read_text(
            encoding="utf-8"
        )
    )

    observed = {}

    for task_id in TASKS:

        raw = next(
            item
            for item in suite["tasks"]
            if item["task_id"] == task_id
        )

        task = build_task(raw)

        observed[task_id] = {}

        for condition in CONDITIONS:

            compiled, value = (
                V084.compile_condition(
                    task,
                    condition,
                )
            )

            prompt = V084.build_prompt(
                task,
                value,
            )

            observed[task_id][condition] = {
                "context_chars": len(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
                "prompt_chars": len(prompt),
                "transform_id": compiled[
                    "transform_id"
                ],
            }

    return observed


def parse_output(text):

    if not isinstance(text, str):
        return None, "OUTPUT_NOT_TEXT"

    try:
        return json.loads(text.strip()), None
    except Exception as exc:
        return None, (
            type(exc).__name__
            + ": "
            + str(exc)
        )


def execute_one(
    task,
    condition,
    repetition,
    position,
    adapter,
):

    compiled, value = (
        V084.compile_condition(
            task,
            condition,
        )
    )

    prompt = V084.build_prompt(
        task,
        value,
    )

    started = now()
    started_mono = time.monotonic()

    try:
        result = adapter.generate(prompt)

        if isinstance(result, dict):
            execution = dict(result)
        else:
            execution = {}

            for field in [
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
            ]:
                if hasattr(result, field):
                    execution[field] = getattr(
                        result,
                        field,
                    )

    except Exception as exc:

        execution = {
            "provider": "deepseek",
            "model": "deepseek",
            "provider_status": "PROVIDER_EXCEPTION",
            "error": (
                type(exc).__name__
                + ": "
                + str(exc)
            ),
            "input_tokens": 0,
            "reasoning_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0,
            "raw_text": "",
        }

    provider_status = execution.get(
        "provider_status",
        execution.get(
            "status",
            "ERROR",
        ),
    )

    raw_text = execution.get(
        "raw_text",
        execution.get(
            "text",
            "",
        ),
    )

    parsed = None
    parse_error = None

    if provider_status == "MODEL_OK":
        parsed, parse_error = parse_output(raw_text)

        if parse_error is not None:
            provider_status = "OUTPUT_ERROR"

    verification = {
        "evaluator": "deterministic_exact_fields",
        "passed": False,
        "missing_or_incorrect": [],
    }

    if (
        provider_status == "MODEL_OK"
        and parsed is not None
    ):

        passed, verification = evaluate(
            task["expected"],
            parsed,
        )

        verification["passed"] = passed

    metrics = {
        "input_tokens": int(
            execution.get(
                "input_tokens",
                0,
            ) or 0
        ),
        "reasoning_tokens": int(
            execution.get(
                "reasoning_tokens",
                0,
            ) or 0
        ),
        "output_tokens": int(
            execution.get(
                "output_tokens",
                0,
            ) or 0
        ),
        "total_tokens": int(
            execution.get(
                "total_tokens",
                0,
            ) or 0
        ),
        "latency_ms": float(
            execution.get(
                "latency_ms",
                0,
            ) or 0
        ),
    }

    return {
        "schema_version": "0.9",
        "timestamp_utc": started,
        "task_id": task["task_id"],
        "condition_id": condition,
        "transform_id": compiled[
            "transform_id"
        ],
        "transform_sequence": compiled[
            "transform_sequence"
        ],
        "repetition": repetition,
        "global_execution_position": position,
        "prompt_chars": len(prompt),
        "context_chars": len(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ),
        "metrics": metrics,
        "execution": {
            "provider_status": provider_status,
            "response_id": execution.get(
                "response_id"
            ),
            "error": (
                parse_error
                or execution.get("error")
            ),
        },
        "verification": verification,
        "quality_pass": bool(
            verification["passed"]
        ),
        "raw_text": raw_text,
        "output": parsed,
        "status": (
            "PASS"
            if verification["passed"]
            else provider_status
        ),
        "start_utc": started,
        "end_utc": now(),
        "monotonic_elapsed_ms": (
            time.monotonic()
            - started_mono
        ) * 1000.0,
    }


def make_report(schedule):

    return {
        "schema_version": "0.9",
        "experiment_id": (
            "PROMPTFORGE_V0.9_POLICY_SEARCH"
        ),
        "status": "RUNNING",
        "provider_phase": "REAL_COLLECTION",
        "tasks": TASKS,
        "model_id": "deepseek",
        "conditions": CONDITIONS,
        "repetitions": REPETITIONS,
        "scheduled_runs": EXPECTED_RUNS,
        "schedule_sha256": schedule[
            "schedule_sha256"
        ],
        "preregistration_sha256": sha256_file(
            PREREG_FILE
        ),
        "protected_sha256": {
            "transforms.py": (
                "D851443F75B02876DEF4A971B0C97B0A34A6DE044ED49326E133A1D421EC8E4A"
            ),
            "real_executor.py": (
                "5589F8C777A594E81BBBFA15D14BF181B141957825DB176EAEA1ABB5DD2D9842"
            ),
            "deepseek_adapter.py": (
                "4913395597AFAB924A8FB8927CB61288603B4B87A5B1B0BC47F6959707D0E614"
            ),
            "v084_zoom_runner.py": (
                "DA6A123B9CE8F3D6066D6A97E81EEEE3EDAA5CE9B37EF6051CBB5C94A3167C06"
            ),
        },
        "runs": [],
    }


def summarize(report):

    output = {
        "cells": {},
        "paired_vs_noop": {},
    }

    usable = [
        row
        for row in report["runs"]
        if row["execution"][
            "provider_status"
        ] == "MODEL_OK"
    ]

    for task_id in TASKS:

        output["cells"][task_id] = {}
        output["paired_vs_noop"][task_id] = {}

        controls = {
            int(row["repetition"]): row
            for row in usable
            if (
                row["task_id"] == task_id
                and row["condition_id"]
                == "noop_compact"
            )
        }

        for condition in CONDITIONS:

            values = [
                row["metrics"]
                for row in usable
                if (
                    row["task_id"] == task_id
                    and row["condition_id"]
                    == condition
                )
            ]

            if values:

                output["cells"][task_id][
                    condition
                ] = {
                    "n": len(values),
                    "mean_input": statistics.mean(
                        x["input_tokens"]
                        for x in values
                    ),
                    "mean_reasoning": statistics.mean(
                        x["reasoning_tokens"]
                        for x in values
                    ),
                    "mean_total": statistics.mean(
                        x["total_tokens"]
                        for x in values
                    ),
                    "mean_latency_ms": statistics.mean(
                        x["latency_ms"]
                        for x in values
                    ),
                }

            if condition == "noop_compact":
                continue

            deltas = []

            for row in usable:

                if (
                    row["task_id"] != task_id
                    or row["condition_id"]
                    != condition
                ):
                    continue

                control = controls.get(
                    int(row["repetition"])
                )

                if control is None:
                    continue

                deltas.append(
                    row["metrics"]["total_tokens"]
                    - control["metrics"][
                        "total_tokens"
                    ]
                )

            if deltas:

                mean_delta = statistics.mean(
                    deltas
                )

                if mean_delta < 0:
                    signal = "BENEFIT_SIGNAL"
                elif mean_delta > 0:
                    signal = "HARM_SIGNAL"
                else:
                    signal = "UNCERTAIN"

                output[
                    "paired_vs_noop"
                ][task_id][condition] = {
                    "n": len(deltas),
                    "mean_delta_total": mean_delta,
                    "signal": signal,
                }

    return output


def dry_run():

    schedule = load_schedule()

    validate_environment()

    observed = preflight()

    print("V09_VALIDATE_SCHEDULE=PASS")
    print("V09_VALIDATE_ENVIRONMENT=PASS")
    print("V09_PREFLIGHT=PASS")
    print("V09_RUNS=240")
    print("V09_TASKS=" + ",".join(TASKS))
    print(
        "V09_CONDITIONS="
        + ",".join(CONDITIONS)
    )
    print(
        "V09_SCHEDULE_SHA256="
        + schedule["schedule_sha256"]
    )
    print(
        "V09_PREREG_SHA256="
        + sha256_file(PREREG_FILE)
    )
    print(
        "V09_PREFLIGHT_SIZES="
        + json.dumps(
            observed,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    print("V09_API_CALLS=0")
    print("V09_DRY_RUN=PASS")


def real_run():

    schedule = load_schedule()

    validate_environment()

    suite = json.loads(
        TASK_FILE.read_text(
            encoding="utf-8"
        )
    )

    tasks = {
        item["task_id"]: build_task(item)
        for item in suite["tasks"]
        if item["task_id"] in TASKS
    }

    prereg_sha = sha256_file(
        PREREG_FILE
    )

    if REPORT_FILE.exists():

        report = json.loads(
            REPORT_FILE.read_text(
                encoding="utf-8"
            )
        )

        if report.get(
            "schedule_sha256"
        ) != schedule["schedule_sha256"]:

            raise RuntimeError(
                "REPORT_SCHEDULE_MISMATCH"
            )

        if report.get(
            "preregistration_sha256"
        ) != prereg_sha:

            raise RuntimeError(
                "REPORT_PREREG_MISMATCH"
            )

    else:

        report = make_report(
            schedule
        )

        atomic_write(
            REPORT_FILE,
            report
        )

    done = {
        int(
            row["global_execution_position"]
        )
        for row in report["runs"]
    }

    adapter = DeepSeekProviderAdapter()

    print("V09_REAL_COLLECTION=START")
    print(
        "RESUME_EXISTING_RUNS="
        + str(len(done))
    )

    for row in schedule["rows"]:

        position = int(
            row["global_execution_position"]
        )

        if position in done:
            continue

        task_id = row["task_id"]
        condition = row["condition_id"]
        repetition = int(
            row["repetition"]
        )

        print(
            "START "
            + "position=" + str(position)
            + " task=" + task_id
            + " condition=" + condition
            + " rep=" + str(repetition),
            flush=True,
        )

        result = execute_one(
            tasks[task_id],
            condition,
            repetition,
            position,
            adapter,
        )

        report["runs"].append(
            result
        )

        report["runs"].sort(
            key=lambda x:
                int(
                    x[
                        "global_execution_position"
                    ]
                )
        )

        atomic_write(
            REPORT_FILE,
            report,
        )

        m = result["metrics"]

        print(
            "DONE "
            + "position=" + str(position)
            + " task=" + task_id
            + " condition=" + condition
            + " rep=" + str(repetition)
            + " status=" + result["status"]
            + " quality=" + str(
                result["quality_pass"]
            )
            + " input=" + str(
                m["input_tokens"]
            )
            + " reasoning=" + str(
                m["reasoning_tokens"]
            )
            + " output=" + str(
                m["output_tokens"]
            )
            + " total=" + str(
                m["total_tokens"]
            ),
            flush=True,
        )

    if len(report["runs"]) != EXPECTED_RUNS:
        raise RuntimeError(
            "RUN_COUNT_INCOMPLETE"
        )

    report["status"] = "COMPLETE"
    report["completed_runs"] = len(
        report["runs"]
    )
    report["finished_utc"] = now()
    report["summary"] = summarize(
        report
    )

    atomic_write(
        REPORT_FILE,
        report
    )

    print("")
    print("V09_REAL_COLLECTION=COMPLETE")
    print(
        "V09_RUNS="
        + str(len(report["runs"]))
    )
    print(
        "V09_REPORT="
        + str(REPORT_FILE)
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--real-run",
        action="store_true",
    )

    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    elif args.real_run:
        real_run()
    else:
        raise SystemExit(
            "Use --dry-run or --real-run"
        )