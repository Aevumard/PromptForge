import argparse
import json
import math
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REPORT = (
    ROOT
    / "harness"
    / "reports"
    / "v061_variance_isolation.json"
)

DEFAULT_OUTPUT = (
    ROOT
    / "harness"
    / "analysis"
    / "v061_variance_isolation_report.md"
)

BASELINE = "selection_only"

ARMS = [
    "selection_representation",
    "selection_representation_A",
    "selection_representation_B",
]


def pearson(xs, ys):

    if len(xs) != len(ys):
        return None

    if len(xs) < 2:
        return None

    mx = mean(xs)
    my = mean(ys)

    numerator = sum(
        (x - mx) * (y - my)
        for x, y in zip(xs, ys)
    )

    den_x = math.sqrt(
        sum(
            (x - mx) ** 2
            for x in xs
        )
    )

    den_y = math.sqrt(
        sum(
            (y - my) ** 2
            for y in ys
        )
    )

    if den_x == 0 or den_y == 0:
        return None

    return numerator / (
        den_x * den_y
    )


def sign(value):

    if value < 0:
        return -1

    if value > 0:
        return 1

    return 0


def sign_agreement(xs, ys):

    if not xs:
        return None

    return sum(
        sign(x) == sign(y)
        for x, y in zip(xs, ys)
    ) / len(xs)


def paired(rows, treatment):

    baseline = {
        int(row["repetition"]): row
        for row in rows
        if row["arm_id"] == BASELINE
    }

    treated = {
        int(row["repetition"]): row
        for row in rows
        if row["arm_id"] == treatment
    }

    common = sorted(
        set(baseline)
        & set(treated)
    )

    result = []

    for repetition in common:

        a = baseline[repetition]
        b = treated[repetition]

        result.append(
            {
                "repetition": repetition,
                "delta_input": (
                    b["metrics"]["input_tokens"]
                    - a["metrics"]["input_tokens"]
                ),
                "delta_reasoning": (
                    b["metrics"]["reasoning_tokens"]
                    - a["metrics"]["reasoning_tokens"]
                ),
                "delta_output": (
                    b["metrics"]["output_tokens"]
                    - a["metrics"]["output_tokens"]
                ),
                "delta_total": (
                    b["metrics"]["total_tokens"]
                    - a["metrics"]["total_tokens"]
                ),
                "delta_latency": (
                    b["metrics"]["latency_ms"]
                    - a["metrics"]["latency_ms"]
                ),
                "execution_position": (
                    b["execution_position"]
                ),
                "block_position": (
                    b["block_position"]
                ),
                "quality_pass": (
                    bool(
                        a["quality_pass"]
                        and b["quality_pass"]
                    )
                ),
            }
        )

    return result


def fmt(value):

    if value is None:
        return "n/a"

    if isinstance(value, int):
        return f"{value:+d}"

    return f"{value:+.4f}"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    if not args.report.is_file():
        raise FileNotFoundError(
            args.report
        )

    report = json.loads(
        args.report.read_text(
            encoding="utf-8"
        )
    )

    rows = report.get("runs", [])

    lines = []

    lines.append(
        "# PromptForge v0.6.1 — Variance Isolation"
    )

    lines.append("")

    lines.append(
        f"- Source: `{args.report}`"
    )

    lines.append(
        "- API calls by analyzer: **0**"
    )

    lines.append("")

    scheme = report.get(
        "scheme",
        "unknown",
    )

    lines.append(
        f"## Scheme: `{scheme}`"
    )

    lines.append("")

    quality_rate = (
        sum(
            bool(row.get("quality_pass"))
            for row in rows
        )
        / len(rows)
        if rows
        else 0
    )

    lines.append(
        f"- Runs: **{len(rows)}**"
    )

    lines.append(
        f"- Quality pass rate: **{quality_rate:.3f}**"
    )

    lines.append("")

    lines.append(
        "### Paired contrasts against selection_only"
    )

    lines.append("")

    lines.append(
        "| Arm | n | Mean ΔR | Median ΔR | "
        "Min ΔR | Max ΔR | Corr(position, ΔR) |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|"
    )

    caches = {}

    for arm in ARMS:

        pairs = paired(
            rows,
            arm,
        )

        caches[arm] = pairs

        reasoning = [
            row["delta_reasoning"]
            for row in pairs
        ]

        positions = [
            row["execution_position"]
            for row in pairs
        ]

        correlation = pearson(
            positions,
            reasoning,
        )

        lines.append(
            f"| `{arm}` | "
            f"{len(reasoning)} | "
            f"{fmt(mean(reasoning)) if reasoning else 'n/a'} | "
            f"{fmt(median(reasoning)) if reasoning else 'n/a'} | "
            f"{fmt(min(reasoning)) if reasoning else 'n/a'} | "
            f"{fmt(max(reasoning)) if reasoning else 'n/a'} | "
            f"{fmt(correlation)} |"
        )

    lines.append("")

    lines.append(
        "### Rep-by-rep Δreasoning"
    )

    lines.append("")

    lines.append(
        "| Rep | SR | A | B |"
    )

    lines.append(
        "|---:|---:|---:|---:|"
    )

    for repetition in range(
        1,
        int(report["repetitions"]) + 1,
    ):

        cells = []

        for arm in ARMS:

            pair = next(
                (
                    row
                    for row in caches[arm]
                    if row["repetition"]
                    == repetition
                ),
                None,
            )

            cells.append(
                "n/a"
                if pair is None
                else fmt(
                    pair["delta_reasoning"]
                )
            )

        lines.append(
            f"| {repetition} | "
            f"{cells[0]} | "
            f"{cells[1]} | "
            f"{cells[2]} |"
        )

    lines.append("")

    lines.append(
        "### Cross-arm sign agreement"
    )

    lines.append("")

    combinations = [
        (
            ARMS[0],
            ARMS[1],
        ),
        (
            ARMS[0],
            ARMS[2],
        ),
        (
            ARMS[1],
            ARMS[2],
        ),
    ]

    lines.append(
        "| Pair | Sign agreement | Pearson ΔR |"
    )

    lines.append(
        "|---|---:|---:|"
    )

    for first, second in combinations:

        first_by_rep = {
            row["repetition"]: row
            for row in caches[first]
        }

        second_by_rep = {
            row["repetition"]: row
            for row in caches[second]
        }

        common = sorted(
            set(first_by_rep)
            & set(second_by_rep)
        )

        xs = [
            first_by_rep[rep]["delta_reasoning"]
            for rep in common
        ]

        ys = [
            second_by_rep[rep]["delta_reasoning"]
            for rep in common
        ]

        lines.append(
            f"| `{first}` vs `{second}` | "
            f"{fmt(sign_agreement(xs, ys))} | "
            f"{fmt(pearson(xs, ys))} |"
        )

    lines.append("")

    lines.append(
        "### Position data"
    )

    lines.append("")

    for arm in [
        BASELINE,
        *ARMS,
    ]:

        arm_rows = [
            row
            for row in rows
            if row["arm_id"] == arm
        ]

        lines.append(
            f"#### `{arm}`"
        )

        lines.append("")

        lines.append(
            "| Rep | Global position | Block position |"
        )

        lines.append(
            "|---:|---:|---:|"
        )

        for row in sorted(
            arm_rows,
            key=lambda item: item["repetition"],
        ):

            lines.append(
                f"| {row['repetition']} | "
                f"{row['execution_position']} | "
                f"{row['block_position']} |"
            )

        lines.append("")

    lines.append(
        "## Interpretation boundary"
    )

    lines.append("")

    lines.append(
        "Observed associations only. "
        "No causal claim about backend state, "
        "server load, model cognition, or prompt "
        "sensitivity is made by this analyzer."
    )

    lines.append("")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        "============================================================"
    )
    print(
        "PROMPTFORGE V0.6.1 STATIC ANALYZER"
    )
    print(
        "============================================================"
    )
    print("status=COMPLETE")
    print(
        f"source={args.report}"
    )
    print(
        f"output={args.output}"
    )
    print("api_calls=0")
    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()
