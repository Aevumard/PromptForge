import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SCHEDULE_FILE = ROOT / "harness" / "runner" / "v093_size_matched_schedule.json"
VARIANT_FILE = ROOT / "harness" / "runner" / "v093_variant_snapshot.json"
PREREG_FILE = ROOT / "harness" / "prereg" / "v093_size_matched_control.md"
REPORT_FILE = ROOT / "harness" / "reports" / "v093_size_matched_control_deepseek.json"
V09_SOURCE_FILE = ROOT / "harness" / "runner" / "v09_policy_search_runner.py"
TASK_FILE = ROOT / "tasks" / "suite.json"

EXPECTED_SCHEDULE_SHA256 = "8C072CBB8E1F764C7E9A60C3EE1CB4758CB2E4621C3E23BD12EB91ED5AB70CAC"
EXPECTED_VARIANT_SHA256 = "A83422BD2B58DEDE1EC44C0D9F0B040B040F96EC467B4BD23DB161110F781584"
EXPECTED_PREREG_SHA256 = "EFE3356CF047BDCD892ECDDAADB58185342600B0BE9D778168DA93A5796D4FA6"
EXPECTED_VARIANT_SNAPSHOT_SHA256 = "A83422BD2B58DEDE1EC44C0D9F0B040B040F96EC467B4BD23DB161110F781584"
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
    "noop_padded",
    "pairs_compact",
    "pairs_bare_compact",
]
REPETITIONS = 15
EXPECTED_RUNS = 240
# PROMPTFORGE_V093_PREFLIGHT_CONTRACT
TASKS = ["T003"]


def preflight_task():
    import copy
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    suite_path = repo_root / "tasks" / "suite.json"

    with suite_path.open("r", encoding="utf-8") as handle:
        suite = json.load(handle)

    tasks = suite.get("tasks")

    if not isinstance(tasks, list):
        raise RuntimeError("TASKS_COLLECTION_MISSING")

    matches = [
        task
        for task in tasks
        if isinstance(task, dict)
        and task.get("task_id") == "T003"
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"T003_EXPECTED_EXACTLY_ONE_MATCH_GOT_{len(matches)}"
        )

    raw_task = copy.deepcopy(matches[0])
    return build_task(raw_task)

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



    module.compile_condition = (
        compile_condition
    )

    module.build_prompt = (
        build_prompt
    )

    # V0.9.3 EXECUTION SURFACE BRIDGE
    if not hasattr(module, 'V084'):
        raise RuntimeError('V093_V084_SURFACE_MISSING')
    module.V084.compile_condition = compile_condition
    module.V084.build_prompt = build_prompt


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

        if schedule["task_family"] != TASK_FAMILY:
            raise RuntimeError(
                "SCHEDULE_TASK_FAMILY_FAIL"
            )

        if schedule["model_id"] != MODEL_ID:
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

        if len(variant_rows) != len(CONDITIONS) * REPETITIONS:
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



# V0.9.3 PUBLIC TEST SURFACE
# compile_condition/build_prompt are module-level functions.

def compile_condition(task, condition):
    from harness.runner.transforms import compile_v06_arm
    context = {'required': task['required'], 'data': task['data']}
    if condition == 'noop_compact':
        compiled = compile_v06_arm(context, 'noop')
        serialized_value = compiled['value']
        return (compiled, serialized_value)
    if condition == 'pairs_compact':
        compiled = compile_v06_arm(context, 'representation_B')
        serialized_value = compiled['value']
        return (compiled, serialized_value)
    if condition == 'pairs_bare_compact':
        compiled = compile_v06_arm(context, 'representation_B')
        serialized_value = compiled['value']['pairs']
        return (compiled, serialized_value)
    if condition == 'noop_padded':
        import copy
        import json

        pairs_compact = compile_v06_arm(
            context,
            'representation_B',
        )

        target_context_chars = len(
            json.dumps(
                pairs_compact['value'],
                ensure_ascii=False,
                separators=(',', ':'),
            )
        )

        padded_compiled = copy.deepcopy(
            compile_v06_arm(
                context,
                'noop',
            )
        )

        base_value = copy.deepcopy(
            padded_compiled['value']
        )

        for pad_len in range(0, 1001):
            candidate = copy.deepcopy(
                base_value
            )

            candidate['_padding'] = 'x' * pad_len

            candidate_chars = len(
                json.dumps(
                    candidate,
                    ensure_ascii=False,
                    separators=(',', ':'),
                )
            )

            if candidate_chars == target_context_chars:
                padded_compiled['value'] = candidate

                return (
                    padded_compiled,
                    candidate,
                )

        raise RuntimeError(
            'NOOP_PADDED_TARGET_UNREACHABLE='
            + str(target_context_chars)
        )

    raise ValueError('unknown V0.9.3 condition: ' + condition)

def build_prompt(task, serialized_value):
    import json
    return task['instruction'] + '\n\nCONTEXT:\n' + json.dumps(serialized_value, ensure_ascii=False, separators=(',', ':')) + '\n\nRETURN ONLY THIS JSON OBJECT:\n' + json.dumps(task['schema'], ensure_ascii=False, indent=2)

def validate_canonical_preflight(task, expected=None):
    expected = {
        "noop_compact": (85, 307),
        "noop_padded": (105, 327),
        "pairs_compact": (105, 327),
        "pairs_bare_compact": (95, 317),
    }

    observed = {}

    for condition in CONDITIONS:
        compiled, serialized_value = compile_condition(
            task,
            condition,
        )

        prompt = build_prompt(
            task,
            serialized_value,
        )

        context_chars = len(
            json.dumps(
                serialized_value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

        prompt_chars = len(prompt)

        observed[condition] = (
            context_chars,
            prompt_chars,
        )

    if observed != expected:
        raise RuntimeError(
            "SIZE_MATCHED_PREFLIGHT_FAIL={}".format(
                observed
            )
        )

    noop = compile_condition(
        task,
        "noop_compact",
    )[1]

    padded = compile_condition(
        task,
        "noop_padded",
    )[1]

    pairs = compile_condition(
        task,
        "pairs_compact",
    )[1]

    noop_required = {
        key: noop[key]
        for key in task["required"]
    }

    padded_required = {
        key: padded[key]
        for key in task["required"]
    }

    pairs_required = {
        pair[0]: pair[1]
        for pair in pairs["pairs"]
        if pair[0] in task["required"]
    }

    if noop_required != padded_required:
        raise RuntimeError(
            "NOOP_PADDED_REQUIRED_VALUES_FAIL"
        )

    if padded_required != pairs_required:
        raise RuntimeError(
            "SIZE_MATCHED_REQUIRED_VALUES_FAIL"
        )

    return observed

def make_initial_report():
    return {
        "schema_version": "0.9.3",
        "experiment_id": (
            "PROMPTFORGE_V0.9.3_PERTURBATION_VALIDATION"
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
        preflight_task(),
        variants,
    )

    print(
        "============================================================"
    )
    print(
        "PROMPTFORGE V0.9.3 PERTURBATION VALIDATION"
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
        "CONDITIONS=4"
    )
    print(
        "REPETITIONS=15"
    )
    print(
        "RUNS=240"
    )
    print(
        "PRIMARY=pairs_compact_vs_noop_padded"
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
        "V093_DRY_RUN=PASS"
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
        preflight_task(),
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
        "PROMPTFORGE V0.9.3 PERTURBATION VALIDATION"
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
        "RUNS=240"
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
            "PROMPTFORGE_V0.9.3_PERTURBATION_VALIDATION"
        )

        result["variant_id"] = (
            variant_id
        )

        result["confirmation_role"] = {
            "noop_compact": "BASELINE",
            "noop_padded": "SIZE_MATCHED_BASELINE",
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
        "V0.9.3 REAL COLLECTION COMPLETE"
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
