import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SCHEDULE_FILE = ROOT / "harness" / "runner" / "v092_perturbation_schedule.json"
VARIANT_FILE = ROOT / "harness" / "runner" / "v092_variant_snapshot.json"
PREREG_FILE = ROOT / "harness" / "prereg" / "v092_perturbation_validation.md"
REPORT_FILE = ROOT / "harness" / "reports" / "v092_perturbation_validation_deepseek.json"
V09_SOURCE_FILE = ROOT / "harness" / "runner" / "v09_policy_search_runner.py"
TASK_FILE = ROOT / "tasks" / "suite.json"

EXPECTED_SCHEDULE_SHA256 = "21203DE648E16BC3AE6E4FB8110DEB51FEC33291C55A8B9B1DF97566D53AD35E"
EXPECTED_VARIANT_SHA256 = "A83422BD2B58DEDE1EC44C0D9F0B040B040F96EC467B4BD23DB161110F781584"
EXPECTED_PREREG_SHA256 = "1A6451FAE91434E4F18C4D27CC33FC06BBE234A422DEEA0A09DACEA8739E0B3B"
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
TASK_FAMILY = "structured_analysis"
MODEL_ID = "deepseek"

VARIANTS = [
    "canonical_t003",
    "noise_long",
    "key_order",
    "content_shift",
]

CONDITIONS = [
    "noop_compact",
    "pairs_compact",
    "pairs_bare_compact",
]

REPETITIONS = 15
EXPECTED_RUNS = 180


def utc_now():
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


def load_v09_module():
    spec = importlib.util.spec_from_file_location(
        "promptforge_v09_source",
        V09_SOURCE_FILE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "V09_IMPORT_SPEC_FAIL"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        "promptforge_v09_source"
    ] = module

    spec.loader.exec_module(module)

    if not hasattr(module, "execute_one"):
        raise RuntimeError(
            "V09_EXECUTE_ONE_MISSING"
        )

    return module


def install_execution_surface(module):
    from harness.runner.transforms import (
        compile_v06_arm,
    )

    def compile_condition(
        task,
        condition,
    ):
        context = {
            "required": task["required"],
            "data": task["data"],
        }

        if condition == "noop_compact":
            compiled = compile_v06_arm(
                context,
                "noop",
            )

            serialized_value = compiled["value"]

            return (
                compiled,
                serialized_value,
            )

        if condition == "pairs_compact":
            compiled = compile_v06_arm(
                context,
                "representation_B",
            )

            serialized_value = compiled["value"]

            return (
                compiled,
                serialized_value,
            )

        if condition == "pairs_bare_compact":
            compiled = compile_v06_arm(
                context,
                "representation_B",
            )

            serialized_value = compiled[
                "value"
            ]["pairs"]

            return (
                compiled,
                serialized_value,
            )

        raise ValueError(
            "unknown V0.9.2 condition: "
            + condition
        )

    def build_prompt(
        task,
        serialized_value,
    ):
        return (
            task["instruction"]
            + "\n\nCONTEXT:\n"
            + json.dumps(
                serialized_value,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )
            + "\n\nRETURN ONLY THIS JSON OBJECT:\n"
            + json.dumps(
                task["schema"],
                ensure_ascii=False,
                indent=2,
            )
        )

    module.compile_condition = (
        compile_condition
    )

    module.build_prompt = (
        build_prompt
    )


def load_variants():
    document = json.loads(
        VARIANT_FILE.read_text(
            encoding="utf-8"
        )
    )

    result = {}

    for item in document["variants"]:
        result[item["variant_id"]] = item

    if set(result) != set(VARIANTS):
        raise RuntimeError(
            "VARIANT_SET_FAIL"
        )

    return result


def build_task(variant):
    data = dict(
        variant["data"]
    )

    required = list(
        variant["required"]
    )

    return {
        "task_id": TASK_ID,
        "task_family": TASK_FAMILY,
        "required": required,
        "data": data,
        "expected": {
            key: data[key]
            for key in required
        },
        "instruction": (
            "Extract the required fields from the supplied context. "
            "Return only valid JSON. Do not explain. Do not use markdown."
        ),
        "schema": {
            key: "value"
            for key in required
        },
    }


def validate_environment(schedule):
    required_files = {
        SCHEDULE_FILE: EXPECTED_SCHEDULE_SHA256,
        VARIANT_FILE: EXPECTED_VARIANT_SHA256,
        PREREG_FILE: EXPECTED_PREREG_SHA256,
        V09_SOURCE_FILE: EXPECTED_V09_SOURCE_SHA256,
        TASK_FILE: EXPECTED_SUITE_SHA256,
    }

    for path, expected in required_files.items():

        if not path.exists():
            raise RuntimeError(
                "REQUIRED_FILE_MISSING="
                + path.name
            )

        actual = sha256_file(path)

        if actual != expected:
            raise RuntimeError(
                "REQUIRED_HASH_FAIL="
                + path.name
            )

    for name, expected in {
        "transforms.py": EXPECTED_TRANSFORMS_SHA256,
        "real_executor.py": EXPECTED_EXECUTOR_SHA256,
        "deepseek_adapter.py": EXPECTED_DEEPSEEK_SHA256,
    }.items():

        path = PROTECTED_FILES[name]

        if not path.exists():
            raise RuntimeError(
                "PROTECTED_FILE_MISSING="
                + name
            )

        actual = sha256_file(path)

        if actual != expected:
            raise RuntimeError(
                "PROTECTED_HASH_FAIL="
                + name
            )

    rows = schedule["rows"]

    if len(rows) != EXPECTED_RUNS:
        raise RuntimeError(
            "SCHEDULE_COUNT_FAIL"
        )

    positions = [
        int(
            row["global_execution_position"]
        )
        for row in rows
    ]

    if positions != list(
        range(EXPECTED_RUNS)
    ):
        raise RuntimeError(
            "SCHEDULE_POSITIONS_FAIL"
        )

    for row in rows:

        if row["task_id"] != TASK_ID:
            raise RuntimeError(
                "SCHEDULE_TASK_FAIL"
            )

        if row["task_family"] != TASK_FAMILY:
            raise RuntimeError(
                "SCHEDULE_TASK_FAMILY_FAIL"
            )

        if row["model_id"] != MODEL_ID:
            raise RuntimeError(
                "SCHEDULE_MODEL_FAIL"
            )

        if row["variant_id"] not in VARIANTS:
            raise RuntimeError(
                "SCHEDULE_VARIANT_FAIL"
            )

        if row["condition_id"] not in CONDITIONS:
            raise RuntimeError(
                "SCHEDULE_CONDITION_FAIL"
            )

    for variant_id in VARIANTS:

        variant_rows = [
            row
            for row in rows
            if row["variant_id"] == variant_id
        ]

        if len(variant_rows) != 45:
            raise RuntimeError(
                "VARIANT_COUNT_FAIL="
                + variant_id
            )

        for repetition in range(
            1,
            REPETITIONS + 1,
        ):

            current = [
                row["condition_id"]
                for row in variant_rows
                if int(
                    row["repetition"]
                ) == repetition
            ]

            if set(current) != set(
                CONDITIONS
            ):
                raise RuntimeError(
                    "VARIANT_PAIRING_FAIL="
                    + variant_id
            )


def validate_canonical_preflight(
    module,
    variants,
):
    task = build_task(
        variants["canonical_t003"]
    )

    expected = {
        "noop_compact": (
            85,
            307,
        ),
        "pairs_compact": (
            105,
            327,
        ),
        "pairs_bare_compact": (
            95,
            317,
        ),
    }

    observed = {}

    for condition in CONDITIONS:

        compiled, serialized_value = (
            module.compile_condition(
                task,
                condition,
            )
        )

        context_chars = len(
            json.dumps(
                serialized_value,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )
        )

        prompt = module.build_prompt(
            task,
            serialized_value,
        )

        observed[condition] = (
            context_chars,
            len(prompt),
        )

    if observed != expected:
        raise RuntimeError(
            "CANONICAL_PREFLIGHT_FAIL="
            + repr(observed)
        )


def make_initial_report():
    return {
        "schema_version": "0.9.2",
        "experiment_id": (
            "PROMPTFORGE_V0.9.2_PERTURBATION_VALIDATION"
        ),
        "status": "RUNNING",
        "provider_phase": "REAL_COLLECTION",
        "task_id": TASK_ID,
        "task_family": TASK_FAMILY,
        "model_id": MODEL_ID,
        "variants": VARIANTS,
        "conditions": CONDITIONS,
        "repetitions": REPETITIONS,
        "scheduled_runs": EXPECTED_RUNS,
        "primary_endpoint": (
            "pooled_delta_total_pairs_compact_minus_noop_compact"
        ),
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "variant_snapshot_sha256": EXPECTED_VARIANT_SHA256,
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


def dry_run(
    schedule,
    module,
    variants,
):
    validate_environment(
        schedule
    )

    install_execution_surface(
        module
    )

    validate_canonical_preflight(
        module,
        variants,
    )

    print(
        "============================================================"
    )
    print(
        "PROMPTFORGE V0.9.2 PERTURBATION VALIDATION"
    )
    print(
        "============================================================"
    )
    print(
        "TASK=T003"
    )
    print(
        "TASK_FAMILY=structured_analysis"
    )
    print(
        "MODEL=deepseek"
    )
    print(
        "VARIANTS=4"
    )
    print(
        "CONDITIONS=3"
    )
    print(
        "REPETITIONS=15"
    )
    print(
        "RUNS=180"
    )
    print(
        "PRIMARY=pairs_compact_vs_noop_compact"
    )
    print(
        "NEGATIVE_CONTROL=pairs_bare_compact"
    )
    print(
        "CANONICAL_PREFLIGHT=PASS"
    )
    print(
        "API_CALLS=0"
    )
    print(
        "V092_DRY_RUN=PASS"
    )


def real_run(
    schedule,
    module,
    variants,
):
    validate_environment(
        schedule
    )

    install_execution_surface(
        module
    )

    validate_canonical_preflight(
        module,
        variants,
    )

    from harness.providers.deepseek_adapter import (
        DeepSeekProviderAdapter,
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

    else:

        report = make_initial_report()

        atomic_write(
            REPORT_FILE,
            report,
        )

    completed = {
        int(
            row[
                "global_execution_position"
            ]
        )
        for row in report["runs"]
    }

    print(
        "============================================================"
    )
    print(
        "PROMPTFORGE V0.9.2 PERTURBATION VALIDATION"
    )
    print(
        "============================================================"
    )
    print(
        "REAL_COLLECTION=START"
    )
    print(
        "TASK=T003"
    )
    print(
        "MODEL=deepseek"
    )
    print(
        "VARIANTS=4"
    )
    print(
        "RUNS=180"
    )
    print(
        "NO_RETRIES=TRUE"
    )
    print(
        ""
    )

    for row in schedule["rows"]:

        position = int(
            row[
                "global_execution_position"
            ]
        )

        if position in completed:
            continue

        variant_id = row[
            "variant_id"
        ]

        condition = row[
            "condition_id"
        ]

        repetition = int(
            row["repetition"]
        )

        task = build_task(
            variants[variant_id]
        )

        print(
            "START position={} variant={} condition={} rep={}".format(
                position,
                variant_id,
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
            "PROMPTFORGE_V0.9.2_PERTURBATION_VALIDATION"
        )

        result["variant_id"] = (
            variant_id
        )

        result["confirmation_role"] = {
            "noop_compact": "BASELINE",
            "pairs_compact": "PRIMARY",
            "pairs_bare_compact": "NEGATIVE_CONTROL",
        }[condition]

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
            report,
        )

        print(
            "DONE position={} variant={} condition={} rep={} status={} quality={}".format(
                position,
                variant_id,
                condition,
                repetition,
                result.get("status"),
                result.get("quality_pass"),
            ),
            flush=True,
        )

    if len(report["runs"]) != EXPECTED_RUNS:
        raise RuntimeError(
            "COLLECTION_INCOMPLETE="
            + str(
                len(report["runs"])
            )
        )

    report["status"] = "COMPLETE"
    report["finished_utc"] = utc_now()

    atomic_write(
        REPORT_FILE,
        report,
    )

    print(
        ""
    )

    print(
        "============================================================"
    )
    print(
        "V0.9.2 REAL COLLECTION COMPLETE"
    )
    print(
        "============================================================"
    )
    print(
        "STATUS=COMPLETE"
    )
    print(
        "RUNS={}".format(
            len(report["runs"])
        )
    )
    print(
        "REPORT={}".format(
            REPORT_FILE
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

    schedule = json.loads(
        SCHEDULE_FILE.read_text(
            encoding="utf-8"
        )
    )

    variants = load_variants()
    module = load_v09_module()

    if args.dry_run:
        dry_run(
            schedule,
            module,
            variants,
        )
        return 0

    real_run(
        schedule,
        module,
        variants,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )