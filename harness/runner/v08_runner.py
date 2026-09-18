from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.evaluators.deterministic import evaluate
from harness.providers.deepseek_adapter import DeepSeekProviderAdapter
from harness.providers.openai_adapter import OpenAIProviderAdapter
from harness.providers.gemini_adapter import GeminiProviderAdapter
from harness.runner.transforms import compile_v06_arm


ROOT = Path(__file__).resolve().parents[2]

TASK_FILE = ROOT / "tasks" / "suite.json"
SCHEDULE_FILE = ROOT / "harness" / "runner" / "v08_schedule.json"
PREREG_FILE = ROOT / "harness" / "prereg" / "v08_preregistration.md"
REPORT_FILE = ROOT / "harness" / "reports" / "v08_heterogeneity.json"

EXPECTED_SCHEDULE_SHA256 = (
    "ae909249485308bb6b63596a9875868c80caa7483990ac2ea0ce7d2e61515f5e"
)

EXPECTED_EXECUTOR_SHA256 = (
    "5589F8C777A594E81BBBFA15D14BF181B141957825DB176EAEA1ABB5DD2D9842"
)

EXPECTED_PREREG_SHA256 = (
    "A3C4873D52C65F3B1D3EA5D35E6DBF1392B365BC773719D47BDAC9D09E904424"
)

EXPECTED_RUNS = 360

TASKS = (
    "T001",
    "T002",
    "T003",
    "T004",
)

MODELS = (
    "deepseek",
    "openai_gpt5_mini",
    "gemini_3_6_flash",
)

ARMS = (
    "noop",
    "representation",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def build_task(raw: dict[str, Any]) -> dict[str, Any]:
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


def result_to_dict(result: Any) -> dict[str, Any]:
    if is_dataclass(result):
        return asdict(result)

    if isinstance(result, dict):
        return dict(result)

    fields = (
        "provider",
        "model",
        "status",
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
    )

    out: dict[str, Any] = {}

    for field in fields:
        if hasattr(result, field):
            out[field] = getattr(result, field)

    if "raw_text" not in out and "text" in out:
        out["raw_text"] = out["text"]

    return out


def parse_json(text: str | None) -> tuple[Any, str | None]:
    if not text:
        return None, "empty_model_output"

    cleaned = text.strip()

    try:
        return json.loads(cleaned), None
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1]), None
            except Exception as exc:
                return None, "json_parse_error: {}".format(exc)

        return None, "json_parse_error: no_object"


def build_prompt(
    task: dict[str, Any],
    compiled: dict[str, Any],
) -> str:
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


def build_adapters() -> dict[str, Any]:
    return {
        "deepseek": DeepSeekProviderAdapter(),
        "openai_gpt5_mini": OpenAIProviderAdapter(
            model="gpt-5-mini",
            reasoning_effort=os.getenv(
                "OPENAI_REASONING_EFFORT",
                "low",
            ).strip() or "low",
            max_output_tokens=int(
                os.getenv(
                    "OPENAI_MAX_OUTPUT_TOKENS",
                    "512",
                )
            ),
        ),
        "gemini_3_6_flash": GeminiProviderAdapter(
            model="gemini-3.6-flash",
        ),
    }


def validate_schedule(
    document: dict[str, Any],
) -> list[dict[str, Any]]:
    if document["experiment_id"] != "PROMPTFORGE_V0.8_HETEROGENEITY":
        raise RuntimeError("SCHEDULE_EXPERIMENT_ID=FAIL")

    if document["version"] != "v0.8":
        raise RuntimeError("SCHEDULE_VERSION=FAIL")

    if document["tasks"] != list(TASKS):
        raise RuntimeError("SCHEDULE_TASKS=FAIL")

    if document["models"] != list(MODELS):
        raise RuntimeError("SCHEDULE_MODELS=FAIL")

    if document["arms"] != list(ARMS):
        raise RuntimeError("SCHEDULE_ARMS=FAIL")

    if document["repetitions"] != 15:
        raise RuntimeError("SCHEDULE_REPETITIONS=FAIL")

    positions = document["positions"]

    if len(positions) != EXPECTED_RUNS:
        raise RuntimeError(
            "SCHEDULE_RUN_COUNT=FAIL "
            + str(len(positions))
        )

    if [
        row["global_execution_position"]
        for row in positions
    ] != list(range(EXPECTED_RUNS)):
        raise RuntimeError("SCHEDULE_POSITIONS=FAIL")

    if any(
        row["task_id"] not in TASKS
        or row["model_id"] not in MODELS
        or row["arm_id"] not in ARMS
        for row in positions
    ):
        raise RuntimeError("SCHEDULE_FACTORS=FAIL")

    if document["schedule_sha256"] != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError(
            "SCHEDULE_DIGEST_FIELD=FAIL"
        )

    payload = {
        "experiment_id": document["experiment_id"],
        "scheme": document["scheme"],
        "tasks": document["tasks"],
        "models": document["models"],
        "arms": document["arms"],
        "repetitions": document["repetitions"],
        "positions": positions,
    }

    actual = hashlib.sha256(
        canonical_json(payload)
    ).hexdigest()

    if actual != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError(
            "SCHEDULE_CANONICAL_HASH=FAIL "
            + actual
        )

    return positions


def validate_environment() -> None:
    executor = ROOT / "harness" / "runner" / "real_executor.py"

    executor_hash = sha256_file(executor)

    if executor_hash != EXPECTED_EXECUTOR_SHA256:
        raise RuntimeError(
            "EXECUTOR_FROZEN_HASH=FAIL "
            + executor_hash
        )

    prereg_hash = sha256_file(PREREG_FILE)

    if prereg_hash != EXPECTED_PREREG_SHA256:
        raise RuntimeError(
            "PREREGISTRATION_HASH=FAIL "
            + prereg_hash
        )

    schedule_hash = sha256_file(SCHEDULE_FILE)

    if schedule_hash != (
        "6BED608E2F53012CCCD94173A38623BB61096099073E456F2CAF95A502F9AF29"
    ):
        raise RuntimeError(
            "SCHEDULE_FILE_HASH=FAIL "
            + schedule_hash
        )


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with tmp.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(tmp, path)


def load_report() -> dict[str, Any] | None:
    if not REPORT_FILE.exists():
        return None

    try:
        data = json.loads(
            REPORT_FILE.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "REPORT_READ_ERROR: {}".format(exc)
        )

    if data.get("experiment_id") != (
        "PROMPTFORGE_V0.8_HETEROGENEITY"
    ):
        raise RuntimeError(
            "REPORT_EXPERIMENT_ID=FAIL"
        )

    if data.get("schedule_sha256") != (
        EXPECTED_SCHEDULE_SHA256
    ):
        raise RuntimeError(
            "REPORT_SCHEDULE_SHA256=FAIL"
        )

    if data.get("preregistration_sha256") != (
        EXPECTED_PREREG_SHA256
    ):
        raise RuntimeError(
            "REPORT_PREREGISTRATION_SHA256=FAIL"
        )

    return data


def make_initial_report(
    schedule_document: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "0.8",
        "experiment_id": (
            "PROMPTFORGE_V0.8_HETEROGENEITY"
        ),
        "status": "RUNNING",
        "started_utc": utc_now(),
        "provider_phase": "REAL_COLLECTION",
        "tasks": list(TASKS),
        "models": list(MODELS),
        "arms": list(ARMS),
        "repetitions": 15,
        "scheduled_runs": EXPECTED_RUNS,
        "scheme": schedule_document["scheme"],
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "executor_sha256": EXPECTED_EXECUTOR_SHA256,
        "runs": [],
    }


def normalize_v08_arm_id(arm_id):
    if arm_id == "representation":
        return "representation_only"
    return arm_id

def is_rate_limit_error(execution):
    error = execution.get("error")
    if error is None:
        return False

    text = str(error).lower()

    markers = (
        "ratelimiterror",
        "rate limit",
        "too_many_requests",
        "quota exceeded",
        "quota_exceeded",
        "http 429",
        "code: 429",
        "error code: 429",
        "status code 429",
    )

    return any(marker in text for marker in markers)

def execute_one(
    task: dict[str, Any],
    model_id: str,
    arm_id: str,
    repetition: int,
    global_position: int,
    adapter: Any,
) -> dict[str, Any]:
    context = {
        "required": task["required"],
        "data": task["data"],
    }

    compiled = compile_v06_arm(
        context,
        normalize_v08_arm_id(arm_id),
    )

    prompt = build_prompt(
        task,
        compiled,
    )

    started_utc = utc_now()
    started_monotonic = time.monotonic()

    try:
        normalized = adapter.generate(prompt)
        execution = result_to_dict(normalized)
        provider_exception = None

        if is_rate_limit_error(execution):
            raise RuntimeError(
                "V08_RATE_LIMIT_ABORT: "
                + str(execution.get("error"))
            )
    except Exception as exc:
        if str(exc).startswith("V08_RATE_LIMIT_ABORT:"):
            print(
                str(exc),
                flush=True,
            )
            raise

        execution = {
            "provider": getattr(
                adapter,
                "provider_name",
                model_id,
            ),
            "model": getattr(
                adapter,
                "model",
                model_id,
            ),
            "status": "PROVIDER_EXCEPTION",
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
            "response_id": None,
            "response_status": None,
            "input_tokens": 0,
            "reasoning_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0,
            "raw_text": "",
        }
        provider_exception = execution["error"]

    finished_monotonic = time.monotonic()
    finished_utc = utc_now()

    raw_text = execution.get(
        "raw_text",
        execution.get("text", ""),
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
        parsed_output, parse_error = parse_json(
            raw_text
        )

        if parse_error is not None:
            provider_status = "OUTPUT_ERROR"

    verification = {
        "evaluator": "deterministic_exact_fields",
        "passed": False,
        "missing_or_incorrect": [],
    }

    if parsed_output is not None:
        passed, verification = evaluate(
            task["expected"],
            parsed_output,
        )
        verification["passed"] = bool(passed)

    metrics = {
        "input_tokens": int(
            execution.get("input_tokens", 0) or 0
        ),
        "reasoning_tokens": int(
            execution.get("reasoning_tokens", 0) or 0
        ),
        "output_tokens": int(
            execution.get("output_tokens", 0) or 0
        ),
        "total_tokens": int(
            execution.get("total_tokens", 0) or 0
        ),
        "latency_ms": float(
            execution.get("latency_ms", 0) or 0
        ),
    }

    return {
        "schema_version": "0.8",
        "run_id": (
            f"{task['task_id']}::"
            f"{model_id}::"
            f"{arm_id}::"
            f"rep_{repetition:03d}"
        ),
        "timestamp_utc": started_utc,
        "task_id": task["task_id"],
        "task_family": task["task_family"],
        "provider": execution.get(
            "provider",
            model_id,
        ),
        "model_id": model_id,
        "model": execution.get("model"),
        "arm_id": arm_id,
        "transform_id": compiled["transform_id"],
        "transform_sequence": compiled[
            "transform_sequence"
        ],
        "repetition": repetition,
        "global_execution_position": global_position,
        "context_change": {
            "included": compiled["included"],
            "excluded": compiled["excluded"],
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
                or execution.get("error")
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


def dry_run(schedule: list[dict[str, Any]]) -> None:
    if len(schedule) != EXPECTED_RUNS:
        raise RuntimeError("DRY_RUN_RUNS=FAIL")

    expected = list(range(EXPECTED_RUNS))

    observed = [
        row["global_execution_position"]
        for row in schedule
    ]

    if observed != expected:
        raise RuntimeError(
            "DRY_RUN_POSITIONS=FAIL"
        )

    if any(
        row["model_id"] not in MODELS
        or row["arm_id"] not in ARMS
        or row["task_id"] not in TASKS
        for row in schedule
    ):
        raise RuntimeError(
            "DRY_RUN_FACTORS=FAIL"
        )

    print(
        "V08_DRY_RUN=PASS",
        flush=True,
    )
    print(
        f"DRY_RUN_POSITIONS=0..{EXPECTED_RUNS - 1}",
        flush=True,
    )
    print(
        "API_CALLS=0",
        flush=True,
    )


def main() -> int:
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

    if not args.dry_run and not args.real:
        parser.error(
            "use --dry-run or --real"
        )

    validate_environment()

    schedule_document = json.loads(
        SCHEDULE_FILE.read_text(
            encoding="utf-8"
        )
    )

    schedule = validate_schedule(
        schedule_document
    )

    print(
        "============================================================"
    )
    print(
        "PROMPTFORGE V0.8 HETEROGENEITY RUNNER"
    )
    print(
        "============================================================"
    )
    print(
        f"RUNS={len(schedule)}"
    )
    print(
        "MODELS=deepseek,openai_gpt5_mini,gemini_3_6_flash"
    )
    print(
        "ARMS=noop,representation"
    )
    print(
        f"SCHEDULE_SHA256={EXPECTED_SCHEDULE_SHA256}"
    )
    print(
        f"PREREGISTRATION_SHA256={EXPECTED_PREREG_SHA256}"
    )
    print(
        f"EXECUTOR_SHA256={EXPECTED_EXECUTOR_SHA256}"
    )

    if args.dry_run:
        dry_run(schedule)
        return 0

    if args.real:
        print("")
        print(
            "REAL_COLLECTION=START"
        )

        suite = json.loads(
            TASK_FILE.read_text(
                encoding="utf-8"
            )
        )

        task_map = {
            item["task_id"]: build_task(item)
            for item in suite["tasks"]
        }

        adapters = build_adapters()

        unknown_tasks = (
            set(TASKS)
            - set(task_map)
        )

        if unknown_tasks:
            raise RuntimeError(
                "MISSING_TASKS="
                + ",".join(sorted(unknown_tasks))
            )

        unknown_adapters = (
            set(MODELS)
            - set(adapters)
        )

        if unknown_adapters:
            raise RuntimeError(
                "MISSING_ADAPTERS="
                + ",".join(sorted(unknown_adapters))
            )

        report = load_report()

        if report is None:
            report = make_initial_report(
                schedule_document
            )
            atomic_write(
                REPORT_FILE,
                report,
            )

        completed = {
            int(
                row["global_execution_position"]
            )
            for row in report["runs"]
        }

        if any(
            position < 0
            or position >= EXPECTED_RUNS
            for position in completed
        ):
            raise RuntimeError(
                "REPORT_POSITION_RANGE=FAIL"
            )

        print(
            f"RESUME_EXISTING_RUNS={len(completed)}",
            flush=True,
        )

        for row in schedule:
            position = int(
                row["global_execution_position"]
            )

            if position in completed:
                print(
                    "SKIP "
                    f"position={position} "
                    f"task={row['task_id']} "
                    f"model={row['model_id']} "
                    f"arm={row['arm_id']} "
                    f"rep={row['repetition']}",
                    flush=True,
                )
                continue

            task_id = row["task_id"]
            model_id = row["model_id"]
            arm_id = row["arm_id"]
            repetition = int(row["repetition"])

            print(
                "START "
                f"position={position} "
                f"task={task_id} "
                f"model={model_id} "
                f"arm={arm_id} "
                f"rep={repetition}",
                flush=True,
            )

            run = execute_one(
                task=task_map[task_id],
                model_id=model_id,
                arm_id=normalize_v08_arm_id(arm_id),
                repetition=repetition,
                global_position=position,
                adapter=adapters[model_id],
            )

            report["runs"].append(run)

            report["runs"].sort(
                key=lambda item: int(
                    item["global_execution_position"]
                )
            )

            report["last_completed_position"] = position
            report["completed_runs"] = len(
                report["runs"]
            )

            atomic_write(
                REPORT_FILE,
                report,
            )

            print(
                "DONE "
                f"position={position} "
                f"task={task_id} "
                f"model={model_id} "
                f"arm={arm_id} "
                f"rep={repetition} "
                f"status={run['status']} "
                f"quality={run['quality_pass']} "
                f"input={run['metrics']['input_tokens']} "
                f"reasoning={run['metrics']['reasoning_tokens']} "
                f"output={run['metrics']['output_tokens']} "
                f"total={run['metrics']['total_tokens']} "
                f"latency_ms={run['metrics']['latency_ms']:.0f}",
                flush=True,
            )

        positions = [
            int(
                item["global_execution_position"]
            )
            for item in report["runs"]
        ]

        if sorted(positions) != list(
            range(EXPECTED_RUNS)
        ):
            report["status"] = "INCOMPLETE"
            atomic_write(
                REPORT_FILE,
                report,
            )
            raise RuntimeError(
                "COLLECTION_INCOMPLETE"
            )

        report["status"] = "COMPLETE"
        report["finished_utc"] = utc_now()
        report["completed_runs"] = EXPECTED_RUNS

        atomic_write(
            REPORT_FILE,
            report,
        )

        print(
            "============================================================"
        )
        print(
            "V0.8 REAL COLLECTION COMPLETE"
        )
        print(
            "============================================================"
        )
        print(
            "status=COMPLETE"
        )
        print(
            f"runs={EXPECTED_RUNS}"
        )
        print(
            "api_calls=360"
        )
        print(
            f"report={REPORT_FILE}"
        )
        print(
            f"schedule_sha256={EXPECTED_SCHEDULE_SHA256}"
        )
        print(
            f"preregistration_sha256={EXPECTED_PREREG_SHA256}"
        )

        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
