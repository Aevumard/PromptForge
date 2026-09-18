import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from harness.evaluators.deterministic import evaluate
from harness.runner.real_executor import execute
from harness.runner.transforms import compile_v06_arm


ROOT = Path(__file__).resolve().parents[2]

TASK_FILE = ROOT / "tasks" / "suite.json"

DEFAULT_REPORT = (
    ROOT
    / "harness"
    / "reports"
    / "v061_variance_isolation.json"
)

TASK_ID = "T003"

ARMS = [
    "selection_representation",
    "selection_representation_A",
    "selection_representation_B",
    "selection_only",
]

ARM_COUNT = len(ARMS)

SCHEMES = [
    "interleaved",
    "randomized",
    "blocked",
    "latin_square",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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


# ------------------------------------------------------------
# Schedule row:
#
#     (repetition, arm_id, global_execution_position)
#
# global_execution_position is always exactly 0..N-1.
# ------------------------------------------------------------

def interleaved_schedule(repetitions):
    schedule = []

    for global_position in range(
        repetitions * ARM_COUNT
    ):
        repetition = (
            global_position // ARM_COUNT
        ) + 1

        arm_position = (
            global_position % ARM_COUNT
        )

        arm_id = ARMS[arm_position]

        schedule.append(
            (
                repetition,
                arm_id,
                global_position,
            )
        )

    return schedule


def randomized_schedule(repetitions, seed):
    rng = random.Random(seed)

    schedule = []
    global_position = 0

    for repetition in range(
        1,
        repetitions + 1,
    ):
        arms = list(ARMS)
        rng.shuffle(arms)

        for arm_id in arms:
            schedule.append(
                (
                    repetition,
                    arm_id,
                    global_position,
                )
            )

            global_position += 1

    return schedule


def blocked_schedule(repetitions):
    schedule = []
    global_position = 0

    for arm_id in ARMS:
        for repetition in range(
            1,
            repetitions + 1,
        ):
            schedule.append(
                (
                    repetition,
                    arm_id,
                    global_position,
                )
            )

            global_position += 1

    return schedule


def latin_square_schedule(repetitions, seed):
    rng = random.Random(seed)

    arm_rotation = rng.randrange(
        ARM_COUNT
    )

    row_rotation = rng.randrange(
        ARM_COUNT
    )

    schedule = []
    global_position = 0

    for repetition in range(
        1,
        repetitions + 1,
    ):
        row = (
            (repetition - 1 + row_rotation)
            % ARM_COUNT
        )

        for position in range(
            ARM_COUNT
        ):
            arm_id = ARMS[
                (
                    row
                    + position
                    + arm_rotation
                )
                % ARM_COUNT
            ]

            schedule.append(
                (
                    repetition,
                    arm_id,
                    global_position,
                )
            )

            global_position += 1

    return schedule


def build_schedule(
    scheme,
    repetitions,
    seed,
):
    if scheme == "interleaved":
        return interleaved_schedule(
            repetitions
        )

    if scheme == "randomized":
        return randomized_schedule(
            repetitions,
            seed,
        )

    if scheme == "blocked":
        return blocked_schedule(
            repetitions
        )

    if scheme == "latin_square":
        return latin_square_schedule(
            repetitions,
            seed,
        )

    raise ValueError(
        f"Unknown scheme: {scheme}"
    )


def validate_schedule(
    scheme,
    repetitions,
    seed,
):
    schedule = build_schedule(
        scheme,
        repetitions,
        seed,
    )

    expected_count = (
        repetitions * ARM_COUNT
    )

    # --------------------------------------------------------
    # Exact tuple shape
    # --------------------------------------------------------

    for row in schedule:
        if not isinstance(row, tuple):
            raise AssertionError(
                f"{scheme}: schedule row "
                f"is not a tuple: {row!r}"
            )

        if len(row) != 3:
            raise AssertionError(
                f"{scheme}: schedule row "
                f"must contain exactly 3 fields: {row!r}"
            )

    # --------------------------------------------------------
    # Exact run count
    # --------------------------------------------------------

    if len(schedule) != expected_count:
        raise AssertionError(
            f"{scheme}: expected "
            f"{expected_count} rows, got "
            f"{len(schedule)}"
        )

    # --------------------------------------------------------
    # Exact repetition × arm coverage
    # --------------------------------------------------------

    expected_pairs = {
        (
            repetition,
            arm_id,
        )
        for repetition in range(
            1,
            repetitions + 1,
        )
        for arm_id in ARMS
    }

    observed_pairs = {
        (
            repetition,
            arm_id,
        )
        for (
            repetition,
            arm_id,
            global_position,
        ) in schedule
    }

    if observed_pairs != expected_pairs:
        raise AssertionError(
            f"{scheme}: repetition/arm "
            f"coverage mismatch"
        )

    if len(observed_pairs) != len(
        schedule
    ):
        raise AssertionError(
            f"{scheme}: duplicate "
            f"repetition/arm assignment"
        )

    # --------------------------------------------------------
    # Global temporal position MUST be exactly 0..N-1
    # --------------------------------------------------------

    observed_positions = sorted(
        global_position
        for (
            repetition,
            arm_id,
            global_position,
        ) in schedule
    )

    expected_positions = list(
        range(expected_count)
    )

    if observed_positions != expected_positions:
        raise AssertionError(
            f"{scheme}: global execution "
            f"positions are invalid: "
            f"{observed_positions}"
        )

    # --------------------------------------------------------
    # Basic ranges
    # --------------------------------------------------------

    for (
        repetition,
        arm_id,
        global_position,
    ) in schedule:

        if not (
            1
            <= repetition
            <= repetitions
        ):
            raise AssertionError(
                f"{scheme}: invalid repetition "
                f"{repetition}"
            )

        if arm_id not in ARMS:
            raise AssertionError(
                f"{scheme}: unknown arm "
                f"{arm_id}"
            )

        if not (
            0
            <= global_position
            < expected_count
        ):
            raise AssertionError(
                f"{scheme}: invalid global "
                f"position {global_position}"
            )

    # --------------------------------------------------------
    # Interleaved invariant
    # --------------------------------------------------------

    if scheme == "interleaved":

        for repetition in range(
            1,
            repetitions + 1,
        ):
            rows = [
                row
                for row in schedule
                if row[0] == repetition
            ]

            observed_arms = [
                row[1]
                for row in rows
            ]

            if observed_arms != ARMS:
                raise AssertionError(
                    f"interleaved: invalid arm "
                    f"order at repetition "
                    f"{repetition}"
                )

    # --------------------------------------------------------
    # Blocked invariant
    #
    # Each arm owns one contiguous global block.
    # --------------------------------------------------------

    if scheme == "blocked":

        for arm_index, arm_id in enumerate(
            ARMS
        ):
            rows = [
                row
                for row in schedule
                if row[1] == arm_id
            ]

            expected_repetitions = list(
                range(
                    1,
                    repetitions + 1,
                )
            )

            observed_repetitions = [
                row[0]
                for row in rows
            ]

            if observed_repetitions != (
                expected_repetitions
            ):
                raise AssertionError(
                    f"blocked: invalid repetition "
                    f"order for {arm_id}: "
                    f"{observed_repetitions}"
                )

            expected_global = list(
                range(
                    arm_index * repetitions,
                    (arm_index + 1) * repetitions,
                )
            )

            observed_global = [
                row[2]
                for row in rows
            ]

            if observed_global != (
                expected_global
            ):
                raise AssertionError(
                    f"blocked: invalid global "
                    f"positions for {arm_id}: "
                    f"{observed_global}"
                )

    # --------------------------------------------------------
    # Latin-square invariant
    #
    # Within each complete 4-repetition cycle,
    # every arm appears exactly once in every
    # relative position 0..3.
    # --------------------------------------------------------

    if scheme == "latin_square":

        if repetitions < ARM_COUNT:
            raise AssertionError(
                "latin_square requires at least "
                f"{ARM_COUNT} repetitions"
            )

        first_cycle = [
            row
            for row in schedule
            if row[0] <= ARM_COUNT
        ]

        for arm_id in ARMS:

            positions = sorted(
                row[2] % ARM_COUNT
                for row in first_cycle
                if row[1] == arm_id
            )

            expected_relative_positions = list(
                range(ARM_COUNT)
            )

            if positions != (
                expected_relative_positions
            ):
                raise AssertionError(
                    "latin_square: arm "
                    f"{arm_id} does not occupy "
                    f"each relative position "
                    f"exactly once: "
                    f"{positions}"
                )

    return schedule


def run_one(
    task,
    arm_id,
    repetition,
    scheme,
    global_execution_position,
    seed,
):
    context = {
        "required": task["required"],
        "data": task["data"],
    }

    compiled = compile_v06_arm(
        context,
        arm_id,
    )

    started_utc = utc_now()
    started_monotonic = time.monotonic()

    execution = execute(
        task,
        compiled,
    )

    finished_monotonic = time.monotonic()
    finished_utc = utc_now()

    verification = {
        "evaluator": (
            "deterministic_exact_fields"
        ),
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
        "schema_version": "0.6.1",
        "timestamp_utc": started_utc,
        "task_id": task["task_id"],
        "task_family": task["task_family"],
        "provider": "deepseek",
        "model": execution.get("model"),
        "arm_id": arm_id,
        "transform_id": compiled["transform_id"],
        "transform_sequence": compiled[
            "transform_sequence"
        ],
        "repetition": repetition,
        "scheme": scheme,
        "global_execution_position": (
            global_execution_position
        ),
        "seed": seed,
        "start_utc": started_utc,
        "end_utc": finished_utc,
        "start_monotonic": started_monotonic,
        "end_monotonic": finished_monotonic,
        "monotonic_elapsed_ms": (
            finished_monotonic
            - started_monotonic
        ) * 1000.0,
        "context_change": {
            "included": compiled["included"],
            "excluded": compiled["excluded"],
        },
        "metrics": {
            "input_tokens": execution[
                "input_tokens"
            ],
            "reasoning_tokens": execution[
                "reasoning_tokens"
            ],
            "output_tokens": execution[
                "output_tokens"
            ],
            "total_tokens": execution[
                "total_tokens"
            ],
            "latency_ms": execution[
                "latency_ms"
            ],
        },
        "execution": {
            "provider_status": execution[
                "provider_status"
            ],
            "response_status": execution[
                "response_status"
            ],
            "response_id": execution[
                "response_id"
            ],
            "error": execution["error"],
        },
        "verification": verification,
        "quality_pass": bool(
            verification["passed"]
        ),
        "raw_text": execution["raw_text"],
        "output": execution["output"],
        "status": (
            "PASS"
            if verification["passed"]
            else execution[
                "provider_status"
            ]
        ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scheme",
        choices=SCHEMES,
        default="interleaved",
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=6101,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )

    args = parser.parse_args()

    if args.repetitions < 1:
        raise ValueError(
            "repetitions must be >= 1"
        )

    schedule = validate_schedule(
        args.scheme,
        args.repetitions,
        args.seed,
    )

    print(
        "============================================================"
    )
    print(
        "PROMPTFORGE V0.6.1 VARIANCE ISOLATION"
    )
    print(
        "============================================================"
    )
    print(
        f"scheme={args.scheme}"
    )
    print(
        f"task={TASK_ID}"
    )
    print(
        f"arms={ARM_COUNT}"
    )
    print(
        f"repetitions={args.repetitions}"
    )
    print(
        f"seed={args.seed}"
    )
    print(
        f"scheduled_runs={len(schedule)}"
    )

    if args.dry_run:

        print("dry_run=YES")

        for (
            repetition,
            arm_id,
            global_position,
        ) in schedule:

            print(
                f"rep={repetition:02d} "
                f"global_position={global_position} "
                f"arm={arm_id}"
            )

        print("api_calls=0")

        print(
            "============================================================"
        )

        return 0

    suite = json.loads(
        TASK_FILE.read_text(
            encoding="utf-8"
        )
    )

    matches = [
        task
        for task in suite["tasks"]
        if task["task_id"] == TASK_ID
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {TASK_ID}; "
            f"found {len(matches)}"
        )

    task = build_task(
        matches[0]
    )

    report = {
        "schema_version": "0.6.1",
        "status": "RUNNING",
        "started_utc": utc_now(),
        "provider": "deepseek",
        "task_id": TASK_ID,
        "task_family": task["task_family"],
        "scheme": args.scheme,
        "seed": args.seed,
        "repetitions": args.repetitions,
        "arms": ARMS,
        "arm_count": ARM_COUNT,
        "scheduled_runs": len(schedule),
        "schedule_row_schema": [
            "repetition",
            "arm_id",
            "global_execution_position",
        ],
        "runs": [],
    }

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    def save():
        args.report.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    save()

    for (
        repetition,
        arm_id,
        global_position,
    ) in schedule:

        print(
            f"START "
            f"scheme={args.scheme} "
            f"rep={repetition} "
            f"global_position={global_position} "
            f"arm={arm_id}",
            flush=True,
        )

        row = run_one(
            task=task,
            arm_id=arm_id,
            repetition=repetition,
            scheme=args.scheme,
            global_execution_position=global_position,
            seed=args.seed,
        )

        report["runs"].append(row)

        save()

        print(
            f"DONE "
            f"scheme={args.scheme} "
            f"rep={repetition} "
            f"global_position={global_position} "
            f"arm={arm_id} "
            f"status={row['status']} "
            f"input={row['metrics']['input_tokens']} "
            f"reasoning={row['metrics']['reasoning_tokens']} "
            f"output={row['metrics']['output_tokens']} "
            f"total={row['metrics']['total_tokens']} "
            f"latency_ms={row['metrics']['latency_ms']:.0f}",
            flush=True,
        )

    report["status"] = "COMPLETE"
    report["finished_utc"] = utc_now()

    save()

    print(
        "============================================================"
    )
    print("V0.6.1 STATUS")
    print(
        "============================================================"
    )
    print("status=COMPLETE")
    print(
        f"scheme={args.scheme}"
    )
    print(
        f"runs={len(report['runs'])}"
    )
    print(
        f"report={args.report}"
    )
    print(
        "============================================================"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
