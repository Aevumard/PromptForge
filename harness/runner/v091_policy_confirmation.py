import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCHEDULE_FILE = ROOT / "harness" / "runner" / "v091_schedule.json"
PREREG_FILE = ROOT / "harness" / "prereg" / "v091_policy_confirmation.md"
REPORT_FILE = ROOT / "harness" / "reports" / "v091_policy_confirmation_deepseek.json"
TASK_FILE = ROOT / "tasks" / "suite.json"

V09_SOURCE_FILE = ROOT / "harness" / "runner" / "v09_policy_search_runner.py"

EXPECTED_SCHEDULE_SHA256 = "B365DA2E348ABE3CD028E2BB78FD5372C2918A633CC4450A55FA53B28870CBFB"
EXPECTED_PREREG_SHA256 = "47E1874D001973E621F8291D9F0DCBD27EB24734CC44A629F58938FA77F6782D"
EXPECTED_V09_SOURCE_SHA256 = "353EA5B6D91F4653C659B299A276A369B6396BF915F8496AF0BC7819446E1E38"
EXPECTED_SUITE_SHA256 = "12FA0D70631609B231400CEEAF727C1A22AF87334A0075148CF9C66F8C5BDEF3"

EXPECTED_TRANSFORMS_SHA256 = "D851443F75B02876DEF4A971B0C97B0A34A6DE044ED49326E133A1D421EC8E4A"
EXPECTED_EXECUTOR_SHA256 = "5589F8C777A594E81BBBFA15D14BF181B141957825DB176EAEA1ABB5DD2D9842"
EXPECTED_DEEPSEEK_SHA256 = "4913395597AFAB924A8FB8927CB61288603B4B87A5B1B0BC47F6959707D0E614"

PROTECTED_FILES = {
    "transforms.py": ROOT / "harness" / "runner" / "transforms.py",
    "real_executor.py": ROOT / "harness" / "runner" / "real_executor.py",
    "deepseek_adapter.py": ROOT / "harness" / "providers" / "deepseek_adapter.py",
}

TASK_ID = "T003"
MODEL_ID = "deepseek"

CONDITIONS = [
    "noop_compact",
    "pairs_compact",
    "pairs_bare_compact",
]

REPETITIONS = 40
EXPECTED_RUNS = 120


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


def load_v09_module():
    if not V09_SOURCE_FILE.exists():
        raise RuntimeError("V09_SOURCE_MISSING")

    spec = importlib.util.spec_from_file_location(
        "promptforge_v09_source",
        V09_SOURCE_FILE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("V09_SOURCE_IMPORT_SPEC_FAIL")

    module = importlib.util.module_from_spec(spec)

    sys.modules["promptforge_v09_source"] = module

    spec.loader.exec_module(module)

    if not hasattr(module, "execute_one"):
        raise RuntimeError("V09_EXECUTE_ONE_MISSING")

    return module


def validate_environment(schedule_document):
    required = {
        SCHEDULE_FILE: EXPECTED_SCHEDULE_SHA256,
        PREREG_FILE: EXPECTED_PREREG_SHA256,
        V09_SOURCE_FILE: EXPECTED_V09_SOURCE_SHA256,
        TASK_FILE: EXPECTED_SUITE_SHA256,
    }

    for path, expected in required.items():
        if not path.exists():
            raise RuntimeError("REQUIRED_FILE_MISSING=" + str(path))

        actual = sha256_file(path)

        if actual != expected:
            raise RuntimeError(
                "REQUIRED_HASH_FAIL={}".format(path.name)
            )

    protected_expected = {
        "transforms.py": EXPECTED_TRANSFORMS_SHA256,
        "real_executor.py": EXPECTED_EXECUTOR_SHA256,
        "deepseek_adapter.py": EXPECTED_DEEPSEEK_SHA256,
    }

    for name, expected in protected_expected.items():
        path = PROTECTED_FILES[name]

        if not path.exists():
            raise RuntimeError("PROTECTED_FILE_MISSING=" + name)

        actual = sha256_file(path)

        if actual != expected:
            raise RuntimeError("PROTECTED_HASH_FAIL=" + name)

    rows = schedule_document["rows"]

    if len(rows) != EXPECTED_RUNS:
        raise RuntimeError("SCHEDULE_RUN_COUNT_FAIL")

    positions = [
        int(row["global_execution_position"])
        for row in rows
    ]

    if positions != list(range(EXPECTED_RUNS)):
        raise RuntimeError("SCHEDULE_POSITIONS_FAIL")

    for row in rows:
        if row["task_id"] != TASK_ID:
            raise RuntimeError("SCHEDULE_TASK_FAIL")

        if row["model_id"] != MODEL_ID:
            raise RuntimeError("SCHEDULE_MODEL_FAIL")

        if row["condition_id"] not in CONDITIONS:
            raise RuntimeError("SCHEDULE_CONDITION_FAIL")

    for repetition in range(1, REPETITIONS + 1):
        current = [
            row["condition_id"]
            for row in rows
            if int(row["repetition"]) == repetition
        ]

        if len(current) != 3:
            raise RuntimeError("SCHEDULE_REPETITION_SIZE_FAIL")

        if set(current) != set(CONDITIONS):
            raise RuntimeError("SCHEDULE_REPETITION_BALANCE_FAIL")


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


def load_task():
    suite = json.loads(
        TASK_FILE.read_text(encoding="utf-8")
    )

    for raw in suite["tasks"]:
        if raw["task_id"] == TASK_ID:
            return build_task(raw)

    raise RuntimeError("TASK_T003_MISSING")


def load_report():
    if not REPORT_FILE.exists():
        return None

    return json.loads(
        REPORT_FILE.read_text(encoding="utf-8")
    )


def make_initial_report(schedule_document):
    return {
        "schema_version": "0.9.1",
        "experiment_id": "PROMPTFORGE_V0.9.1_POLICY_CONFIRMATION",
        "status": "RUNNING",
        "provider_phase": "REAL_COLLECTION",
        "task_id": TASK_ID,
        "model_id": MODEL_ID,
        "conditions": CONDITIONS,
        "repetitions": REPETITIONS,
        "scheduled_runs": EXPECTED_RUNS,
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "v09_source_sha256": EXPECTED_V09_SOURCE_SHA256,
        "task_suite_sha256": EXPECTED_SUITE_SHA256,
        "protected_sha256": {
            "transforms.py": EXPECTED_TRANSFORMS_SHA256,
            "real_executor.py": EXPECTED_EXECUTOR_SHA256,
            "deepseek_adapter.py": EXPECTED_DEEPSEEK_SHA256,
        },
        "runs": [],
        "started_utc": utc_now(),
        "finished_utc": None,
    }


def dry_run(schedule_document):
    validate_environment(schedule_document)

    module = load_v09_module()

    if not hasattr(module, "execute_one"):
        raise RuntimeError("V09_EXECUTE_ONE_MISSING")

    print("============================================================")
    print("PROMPTFORGE V0.9.1 POLICY CONFIRMATION")
    print("============================================================")
    print("TASK=T003")
    print("MODEL=deepseek")
    print("CONDITIONS=noop_compact,pairs_compact,pairs_bare_compact")
    print("REPETITIONS=40")
    print("RUNS=120")
    print("PRIMARY=pairs_compact_vs_noop_compact")
    print("NEGATIVE_CONTROL=pairs_bare_compact")
    print("API_CALLS=0")
    print("V091_DRY_RUN=PASS")


def real_run(schedule_document):
    validate_environment(schedule_document)

    module = load_v09_module()
    task = load_task()

    from harness.providers.deepseek_adapter import DeepSeekProviderAdapter

    adapter = DeepSeekProviderAdapter()

    report = load_report()

    if report is None:
        report = make_initial_report(schedule_document)
        atomic_write(REPORT_FILE, report)

    completed = {
        int(row["global_execution_position"])
        for row in report["runs"]
    }

    print("============================================================")
    print("PROMPTFORGE V0.9.1 POLICY CONFIRMATION")
    print("============================================================")
    print("REAL_COLLECTION=START")
    print("TASK=T003")
    print("MODEL=deepseek")
    print("REPETITIONS=40")
    print("RUNS=120")
    print("NO_RETRIES=TRUE")
    print("")

    for row in schedule_document["rows"]:
        position = int(row["global_execution_position"])

        if position in completed:
            continue

        condition = row["condition_id"]
        repetition = int(row["repetition"])

        print(
            "START position={} task={} condition={} rep={}".format(
                position,
                TASK_ID,
                condition,
                repetition,
            ),
            flush=True,
        )

        result = module.execute_one(
            task,
            condition,
            repetition,
            position,
            adapter,
        )

        result["experiment_id"] = (
            "PROMPTFORGE_V0.9.1_POLICY_CONFIRMATION"
        )

        result["confirmation_role"] = {
            "noop_compact": "BASELINE",
            "pairs_compact": "PRIMARY",
            "pairs_bare_compact": "NEGATIVE_CONTROL",
        }[condition]

        report["runs"].append(result)

        report["runs"].sort(
            key=lambda item: int(
                item["global_execution_position"]
            )
        )

        atomic_write(
            REPORT_FILE,
            report,
        )

        print(
            "DONE position={} task={} condition={} rep={} status={} quality={}".format(
                position,
                TASK_ID,
                condition,
                repetition,
                result.get("status"),
                result.get("quality_pass"),
            ),
            flush=True,
        )

    if len(report["runs"]) != EXPECTED_RUNS:
        raise RuntimeError(
            "COLLECTION_INCOMPLETE={}".format(
                len(report["runs"])
            )
        )

    report["status"] = "COMPLETE"
    report["finished_utc"] = utc_now()

    atomic_write(
        REPORT_FILE,
        report,
    )

    print("")
    print("============================================================")
    print("V0.9.1 REAL COLLECTION COMPLETE")
    print("============================================================")
    print("STATUS=COMPLETE")
    print("RUNS={}".format(len(report["runs"])))
    print("REPORT={}".format(REPORT_FILE))
    print(
        "SCHEDULE_SHA256={}".format(
            EXPECTED_SCHEDULE_SHA256
        )
    )
    print(
        "PREREGISTRATION_SHA256={}".format(
            EXPECTED_PREREG_SHA256
        )
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

    if args.dry_run == args.real:
        parser.error(
            "use exactly one of --dry-run or --real"
        )

    schedule_document = json.loads(
        SCHEDULE_FILE.read_text(
            encoding="utf-8"
        )
    )

    if args.dry_run:
        dry_run(schedule_document)
        return 0

    real_run(schedule_document)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())