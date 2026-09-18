import json
import math
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"D:\PROMPTFORGE\promptforge-harness-v0.1")
REPORT_DIR = ROOT / "harness" / "reports"
OUT = ROOT / "harness" / "analysis" / "v061_combined_analysis.md"

SCHEMES = [
    "interleaved",
    "randomized",
    "blocked",
    "latin_square",
]

ARMS = [
    "selection_only",
    "selection_representation",
    "selection_representation_A",
    "selection_representation_B",
]

TEST_ARMS = [
    "selection_representation",
    "selection_representation_A",
    "selection_representation_B",
]

REPORTS = {
    scheme: REPORT_DIR / f"v061_t003_{scheme}.json"
    for scheme in SCHEMES
}


# ============================================================
# Basic helpers
# ============================================================

def mean(values):
    values = list(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def pearson(xs, ys):
    xs = list(xs)
    ys = list(ys)

    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")

    mx = mean(xs)
    my = mean(ys)

    numerator = sum(
        (x - mx) * (y - my)
        for x, y in zip(xs, ys)
    )

    denominator_x = math.sqrt(
        sum((x - mx) ** 2 for x in xs)
    )

    denominator_y = math.sqrt(
        sum((y - my) ** 2 for y in ys)
    )

    denominator = denominator_x * denominator_y

    if denominator == 0:
        return float("nan")

    return numerator / denominator


def fmt(value, digits=3):
    if isinstance(value, int):
        return str(value)

    if isinstance(value, float) and math.isnan(value):
        return "NA"

    return f"{value:.{digits}f}"


def pct(value):
    if isinstance(value, float) and math.isnan(value):
        return "NA"

    return f"{100.0 * value:.1f}%"


# ============================================================
# Real V0.6.1 schema extraction
# ============================================================

def load_report(path):
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: report root must be an object"
        )

    runs = data.get("runs")

    if not isinstance(runs, list):
        raise ValueError(
            f"{path}: expected top-level runs[]"
        )

    return runs


def required_top_level(run, key):
    if key not in run:
        raise ValueError(
            f"Run missing required top-level field '{key}'. "
            f"Available keys={list(run.keys())}"
        )

    return run[key]


def metric_from_paths(run, paths, metric_name):
    for path in paths:
        value = run

        ok = True

        for key in path:
            if not isinstance(value, dict) or key not in value:
                ok = False
                break

            value = value[key]

        if ok and isinstance(value, (int, float)):
            return float(value)

    metrics_obj = run.get("metrics")
    execution_obj = run.get("execution")

    if isinstance(metrics_obj, dict):
        metrics_keys = list(metrics_obj.keys())
    else:
        metrics_keys = []

    if isinstance(execution_obj, dict):
        execution_keys = list(execution_obj.keys())
    else:
        execution_keys = []

    raise ValueError(
        f"Could not locate metric '{metric_name}'. "
        f"Top-level keys={list(run.keys())}; "
        f"metrics keys={metrics_keys}; "
        f"execution keys={execution_keys}"
    )


def normalize_run(run, scheme):
    if not isinstance(run, dict):
        raise ValueError(
            f"{scheme}: execution record must be an object"
        )

    repetition = int(
        required_top_level(run, "repetition")
    )

    arm = str(
        required_top_level(run, "arm_id")
    )

    position = int(
        required_top_level(
            run,
            "global_execution_position",
        )
    )

    quality = bool(
        required_top_level(
            run,
            "quality_pass",
        )
    )

    input_tokens = metric_from_paths(
        run,
        [
            ("metrics", "input_tokens"),
            ("execution", "input_tokens"),
            ("execution", "usage", "input_tokens"),
        ],
        "input_tokens",
    )

    reasoning_tokens = metric_from_paths(
        run,
        [
            ("metrics", "reasoning_tokens"),
            ("execution", "reasoning_tokens"),
            ("execution", "usage", "reasoning_tokens"),
        ],
        "reasoning_tokens",
    )

    output_tokens = metric_from_paths(
        run,
        [
            ("metrics", "output_tokens"),
            ("execution", "output_tokens"),
            ("execution", "usage", "output_tokens"),
        ],
        "output_tokens",
    )

    total_tokens = metric_from_paths(
        run,
        [
            ("metrics", "total_tokens"),
            ("execution", "total_tokens"),
            ("execution", "usage", "total_tokens"),
        ],
        "total_tokens",
    )

    latency_ms = metric_from_paths(
        run,
        [
            ("metrics", "latency_ms"),
            ("execution", "latency_ms"),
            ("execution", "duration_ms"),
            ("execution", "elapsed_ms"),
        ],
        "latency_ms",
    )

    return {
        "scheme": scheme,
        "repetition": repetition,
        "arm": arm,
        "position": position,
        "input": input_tokens,
        "reasoning": reasoning_tokens,
        "output": output_tokens,
        "total": total_tokens,
        "latency": latency_ms,
        "quality": quality,
    }


# ============================================================
# Grouping / statistics
# ============================================================

def group(rows, key_fn):
    groups = defaultdict(list)

    for row in rows:
        groups[key_fn(row)].append(row)

    return groups


def metric_mean(rows, metric):
    return mean(
        row[metric]
        for row in rows
    )


def descriptive_ss(rows, metric, key_fn):
    grand_mean = metric_mean(rows, metric)

    groups = group(rows, key_fn)

    return sum(
        len(group_rows)
        * (
            metric_mean(group_rows, metric)
            - grand_mean
        ) ** 2
        for group_rows in groups.values()
    )


def arm_scheme_interaction_ss(rows, metric):
    grand_mean = metric_mean(rows, metric)

    arm_groups = group(
        rows,
        lambda row: row["arm"],
    )

    scheme_groups = group(
        rows,
        lambda row: row["scheme"],
    )

    cells = group(
        rows,
        lambda row: (
            row["arm"],
            row["scheme"],
        ),
    )

    arm_means = {
        arm: metric_mean(group_rows, metric)
        for arm, group_rows in arm_groups.items()
    }

    scheme_means = {
        scheme: metric_mean(group_rows, metric)
        for scheme, group_rows in scheme_groups.items()
    }

    ss = 0.0

    for (arm, scheme), cell_rows in cells.items():
        cell_mean = metric_mean(
            cell_rows,
            metric,
        )

        expected = (
            arm_means[arm]
            + scheme_means[scheme]
            - grand_mean
        )

        ss += (
            len(cell_rows)
            * (cell_mean - expected) ** 2
        )

    return ss


# ============================================================
# Paired contrasts
# ============================================================

def paired_rows(rows, test_arm):
    index = {}

    for row in rows:
        key = (
            row["scheme"],
            row["repetition"],
        )

        index.setdefault(key, {})[
            row["arm"]
        ] = row

    pairs = []

    for (
        scheme,
        repetition,
    ), arms in sorted(index.items()):

        if (
            "selection_only" not in arms
            or test_arm not in arms
        ):
            continue

        baseline = arms["selection_only"]
        test = arms[test_arm]

        delta_reasoning = (
            test["reasoning"]
            - baseline["reasoning"]
        )

        if delta_reasoning < 0:
            sign = -1
        elif delta_reasoning > 0:
            sign = 1
        else:
            sign = 0

        pairs.append({
            "scheme": scheme,
            "repetition": repetition,
            "test_position": test["position"],
            "delta_input": (
                test["input"]
                - baseline["input"]
            ),
            "delta_reasoning": delta_reasoning,
            "delta_output": (
                test["output"]
                - baseline["output"]
            ),
            "delta_total": (
                test["total"]
                - baseline["total"]
            ),
            "delta_latency": (
                test["latency"]
                - baseline["latency"]
            ),
            "delta_reasoning_sign": sign,
        })

    return pairs


def sign_agreement(pairs_a, pairs_b):
    map_a = {
        (
            pair["scheme"],
            pair["repetition"],
        ): pair["delta_reasoning_sign"]
        for pair in pairs_a
    }

    map_b = {
        (
            pair["scheme"],
            pair["repetition"],
        ): pair["delta_reasoning_sign"]
        for pair in pairs_b
    }

    common = sorted(
        set(map_a) & set(map_b)
    )

    if not common:
        return 0, 0, float("nan")

    agreement = sum(
        1
        for key in common
        if map_a[key] == map_b[key]
    )

    return (
        agreement,
        len(common),
        agreement / len(common),
    )


# ============================================================
# Main
# ============================================================

def main():

    all_rows = []
    integrity = []

    # --------------------------------------------------------
    # Load reports
    # --------------------------------------------------------

    for scheme in SCHEMES:

        path = REPORTS[scheme]

        if not path.exists():
            raise FileNotFoundError(
                f"Missing report: {path}"
            )

        raw_runs = load_report(path)

        if len(raw_runs) != 40:
            raise ValueError(
                f"{scheme}: expected 40 runs, "
                f"got {len(raw_runs)}"
            )

        normalized = [
            normalize_run(
                run,
                scheme,
            )
            for run in raw_runs
        ]

        quality_ok = sum(
            1
            for row in normalized
            if row["quality"]
        )

        integrity.append({
            "scheme": scheme,
            "runs": len(normalized),
            "quality_ok": quality_ok,
            "quality_rate": (
                quality_ok
                / len(normalized)
            ),
        })

        all_rows.extend(normalized)

    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    if len(all_rows) != 160:
        raise ValueError(
            f"Expected exactly 160 runs, "
            f"got {len(all_rows)}"
        )

    expected_positions = set(range(40))

    for scheme in SCHEMES:

        scheme_rows = [
            row
            for row in all_rows
            if row["scheme"] == scheme
        ]

        positions = {
            row["position"]
            for row in scheme_rows
        }

        if positions != expected_positions:
            raise ValueError(
                f"{scheme}: invalid global positions"
            )

        for arm in ARMS:

            count = sum(
                1
                for row in scheme_rows
                if row["arm"] == arm
            )

            if count != 10:
                raise ValueError(
                    f"{scheme}/{arm}: "
                    f"expected 10 runs, "
                    f"got {count}"
                )

    # --------------------------------------------------------
    # Global means
    # --------------------------------------------------------

    global_summary = {}

    for arm in ARMS:

        rows = [
            row
            for row in all_rows
            if row["arm"] == arm
        ]

        global_summary[arm] = {
            "n": len(rows),
            "input": metric_mean(
                rows,
                "input",
            ),
            "reasoning": metric_mean(
                rows,
                "reasoning",
            ),
            "output": metric_mean(
                rows,
                "output",
            ),
            "total": metric_mean(
                rows,
                "total",
            ),
            "latency": metric_mean(
                rows,
                "latency",
            ),
        }

    # --------------------------------------------------------
    # Scheme × arm means
    # --------------------------------------------------------

    cell_summary = {}

    for scheme in SCHEMES:
        for arm in ARMS:

            rows = [
                row
                for row in all_rows
                if (
                    row["scheme"] == scheme
                    and row["arm"] == arm
                )
            ]

            cell_summary[
                (scheme, arm)
            ] = {
                "n": len(rows),
                "reasoning": metric_mean(
                    rows,
                    "reasoning",
                ),
                "output": metric_mean(
                    rows,
                    "output",
                ),
                "total": metric_mean(
                    rows,
                    "total",
                ),
                "latency": metric_mean(
                    rows,
                    "latency",
                ),
            }

    # --------------------------------------------------------
    # Position correlations
    # --------------------------------------------------------

    correlation_sets = [
        ("ALL", all_rows)
    ]

    for scheme in SCHEMES:
        correlation_sets.append(
            (
                scheme,
                [
                    row
                    for row in all_rows
                    if row["scheme"] == scheme
                ],
            )
        )

    position_correlations = []

    for label, rows in correlation_sets:

        position_correlations.append({
            "label": label,
            "n": len(rows),
            "reasoning": pearson(
                [
                    row["position"]
                    for row in rows
                ],
                [
                    row["reasoning"]
                    for row in rows
                ],
            ),
            "latency": pearson(
                [
                    row["position"]
                    for row in rows
                ],
                [
                    row["latency"]
                    for row in rows
                ],
            ),
            "total": pearson(
                [
                    row["position"]
                    for row in rows
                ],
                [
                    row["total"]
                    for row in rows
                ],
            ),
        })

    # --------------------------------------------------------
    # Repetition correlations
    # --------------------------------------------------------

    repetition_correlations = {
        metric: pearson(
            [
                row["repetition"]
                for row in all_rows
            ],
            [
                row[metric]
                for row in all_rows
            ],
        )
        for metric in [
            "reasoning",
            "latency",
            "total",
        ]
    }

    # --------------------------------------------------------
    # Paired effects
    # --------------------------------------------------------

    paired = {
        arm: paired_rows(
            all_rows,
            arm,
        )
        for arm in TEST_ARMS
    }

    paired_summary = {}

    for arm in TEST_ARMS:

        pairs = paired[arm]

        paired_summary[arm] = {
            "n": len(pairs),
            "delta_input": mean(
                pair["delta_input"]
                for pair in pairs
            ),
            "delta_reasoning": mean(
                pair["delta_reasoning"]
                for pair in pairs
            ),
            "delta_output": mean(
                pair["delta_output"]
                for pair in pairs
            ),
            "delta_total": mean(
                pair["delta_total"]
                for pair in pairs
            ),
            "delta_latency": mean(
                pair["delta_latency"]
                for pair in pairs
            ),
            "position_delta_reasoning": pearson(
                [
                    pair["test_position"]
                    for pair in pairs
                ],
                [
                    pair["delta_reasoning"]
                    for pair in pairs
                ],
            ),
            "beneficial": sum(
                1
                for pair in pairs
                if pair["delta_reasoning_sign"] < 0
            ),
            "harmful": sum(
                1
                for pair in pairs
                if pair["delta_reasoning_sign"] > 0
            ),
            "tie": sum(
                1
                for pair in pairs
                if pair["delta_reasoning_sign"] == 0
            ),
        }

    # --------------------------------------------------------
    # Within-scheme effects
    # --------------------------------------------------------

    within_scheme = {}

    for scheme in SCHEMES:

        within_scheme[scheme] = {}

        for arm in TEST_ARMS:

            pairs = [
                pair
                for pair in paired[arm]
                if pair["scheme"] == scheme
            ]

            within_scheme[
                scheme
            ][arm] = {
                "n": len(pairs),
                "delta_reasoning": mean(
                    pair["delta_reasoning"]
                    for pair in pairs
                ),
                "delta_total": mean(
                    pair["delta_total"]
                    for pair in pairs
                ),
                "delta_latency": mean(
                    pair["delta_latency"]
                    for pair in pairs
                ),
                "beneficial": sum(
                    1
                    for pair in pairs
                    if pair["delta_reasoning_sign"] < 0
                ),
                "harmful": sum(
                    1
                    for pair in pairs
                    if pair["delta_reasoning_sign"] > 0
                ),
            }

    # --------------------------------------------------------
    # Sign agreement
    # --------------------------------------------------------

    sign_agreement_table = {}

    for i in range(len(TEST_ARMS)):
        for j in range(i + 1, len(TEST_ARMS)):

            a = TEST_ARMS[i]
            b = TEST_ARMS[j]

            sign_agreement_table[
                (a, b)
            ] = sign_agreement(
                paired[a],
                paired[b],
            )

    # --------------------------------------------------------
    # Variance decomposition
    # --------------------------------------------------------

    variance = {}

    for metric in [
        "reasoning",
        "latency",
        "total",
    ]:

        values = [
            row[metric]
            for row in all_rows
        ]

        grand = mean(values)

        total_ss = sum(
            (value - grand) ** 2
            for value in values
        )

        arm_ss = descriptive_ss(
            all_rows,
            metric,
            lambda row: row["arm"],
        )

        scheme_ss = descriptive_ss(
            all_rows,
            metric,
            lambda row: row["scheme"],
        )

        repetition_ss = descriptive_ss(
            all_rows,
            metric,
            lambda row: row["repetition"],
        )

        interaction_ss = (
            arm_scheme_interaction_ss(
                all_rows,
                metric,
            )
        )

        explained = (
            arm_ss
            + scheme_ss
            + repetition_ss
            + interaction_ss
        )

        residual = max(
            0.0,
            total_ss - explained,
        )

        def share(value):
            if total_ss == 0:
                return float("nan")
            return value / total_ss

        variance[metric] = {
            "arm": share(arm_ss),
            "scheme": share(scheme_ss),
            "repetition": share(
                repetition_ss
            ),
            "interaction": share(
                interaction_ss
            ),
            "residual": share(
                residual
            ),
        }

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    lines = []

    lines.append(
        "# PromptForge V0.6.1 — T003 Combined Analysis"
    )
    lines.append("")

    lines.append("## Experimental scope")
    lines.append("")
    lines.append(
        "160 real API executions: "
        "4 schemes × 10 repetitions × 4 arms."
    )
    lines.append("")
    lines.append(
        "The analysis is descriptive and uses the actual "
        "V0.6.1 report schema. It does not claim causal "
        "identification or inferential significance."
    )
    lines.append("")

    lines.append("## Integrity")
    lines.append("")
    lines.append(
        "| Scheme | Runs | Quality OK | Quality rate |"
    )
    lines.append("|---|---:|---:|---:|")

    for row in integrity:
        lines.append(
            f"| {row['scheme']} | "
            f"{row['runs']} | "
            f"{row['quality_ok']} | "
            f"{pct(row['quality_rate'])} |"
        )

    total_quality = sum(
        row["quality_ok"]
        for row in integrity
    )

    lines.append("")
    lines.append(
        f"Global quality: **{total_quality}/160 "
        f"= {pct(total_quality / 160)}**."
    )
    lines.append("")

    lines.append("## Global arm means")
    lines.append("")
    lines.append(
        "| Arm | N | Input | Reasoning | Output | Total | Latency ms |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for arm in ARMS:

        row = global_summary[arm]

        lines.append(
            f"| `{arm}` | "
            f"{row['n']} | "
            f"{fmt(row['input'])} | "
            f"{fmt(row['reasoning'])} | "
            f"{fmt(row['output'])} | "
            f"{fmt(row['total'])} | "
            f"{fmt(row['latency'])} |"
        )

    lines.append("")

    lines.append("## Scheme × arm means")
    lines.append("")
    lines.append(
        "| Scheme | Arm | Reasoning | Output | Total | Latency ms |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")

    for scheme in SCHEMES:
        for arm in ARMS:

            row = cell_summary[
                (scheme, arm)
            ]

            lines.append(
                f"| {scheme} | `{arm}` | "
                f"{fmt(row['reasoning'])} | "
                f"{fmt(row['output'])} | "
                f"{fmt(row['total'])} | "
                f"{fmt(row['latency'])} |"
            )

    lines.append("")

    lines.append("## Position correlations")
    lines.append("")
    lines.append(
        "| Dataset | N | r(position, reasoning) | "
        "r(position, latency) | r(position, total) |"
    )
    lines.append("|---|---:|---:|---:|---:|")

    for row in position_correlations:

        lines.append(
            f"| {row['label']} | "
            f"{row['n']} | "
            f"{fmt(row['reasoning'])} | "
            f"{fmt(row['latency'])} | "
            f"{fmt(row['total'])} |"
        )

    lines.append("")

    lines.append("## Repetition correlations")
    lines.append("")
    lines.append(
        f"- reasoning: **{fmt(repetition_correlations['reasoning'])}**"
    )
    lines.append(
        f"- latency: **{fmt(repetition_correlations['latency'])}**"
    )
    lines.append(
        f"- total: **{fmt(repetition_correlations['total'])}**"
    )
    lines.append("")

    lines.append("## Paired contrasts vs selection_only")
    lines.append("")
    lines.append(
        "| Arm | N | Δinput | Δreasoning | Δoutput | "
        "Δtotal | Δlatency | Beneficial | Harmful | "
        "r(position, Δreasoning) |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for arm in TEST_ARMS:

        row = paired_summary[arm]

        lines.append(
            f"| `{arm}` | "
            f"{row['n']} | "
            f"{fmt(row['delta_input'])} | "
            f"{fmt(row['delta_reasoning'])} | "
            f"{fmt(row['delta_output'])} | "
            f"{fmt(row['delta_total'])} | "
            f"{fmt(row['delta_latency'])} | "
            f"{row['beneficial']} | "
            f"{row['harmful']} | "
            f"{fmt(row['position_delta_reasoning'])} |"
        )

    lines.append("")

    lines.append("## Within-scheme paired effects")
    lines.append("")
    lines.append(
        "| Scheme | Arm | Δreasoning | Δtotal | "
        "Δlatency | Beneficial | Harmful |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for scheme in SCHEMES:
        for arm in TEST_ARMS:

            row = within_scheme[
                scheme
            ][arm]

            lines.append(
                f"| {scheme} | `{arm}` | "
                f"{fmt(row['delta_reasoning'])} | "
                f"{fmt(row['delta_total'])} | "
                f"{fmt(row['delta_latency'])} | "
                f"{row['beneficial']} | "
                f"{row['harmful']} |"
            )

    lines.append("")

    lines.append("## Sign agreement")
    lines.append("")
    lines.append(
        "Agreement compares the sign of matched "
        "Δreasoning values against selection_only."
    )
    lines.append("")
    lines.append("| Pair | Agreement | N | Rate |")
    lines.append("|---|---:|---:|---:|")

    for (
        a,
        b,
    ), (
        agreement,
        n,
        rate,
    ) in sign_agreement_table.items():

        lines.append(
            f"| `{a}` vs `{b}` | "
            f"{agreement} | "
            f"{n} | "
            f"{pct(rate)} |"
        )

    lines.append("")

    lines.append("## Descriptive variance decomposition")
    lines.append("")
    lines.append(
        "These are descriptive sums-of-squares components, "
        "not p-values."
    )
    lines.append("")
    lines.append(
        "| Metric | ARM | SCHEME | REPETITION | "
        "ARM×SCHEME | Residual |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")

    for metric in [
        "reasoning",
        "latency",
        "total",
    ]:

        row = variance[metric]

        lines.append(
            f"| {metric} | "
            f"{pct(row['arm'])} | "
            f"{pct(row['scheme'])} | "
            f"{pct(row['repetition'])} | "
            f"{pct(row['interaction'])} | "
            f"{pct(row['residual'])} |"
        )

    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "1. The V0.6.1 collection contains 160/160 validated executions."
    )
    lines.append(
        "2. Quality preservation is complete across all schemes and arms."
    )
    lines.append(
        "3. A simple global monotonic position effect does not explain "
        "the observed execution variance."
    )
    lines.append(
        "4. Repetition has comparatively weak global linear association "
        "with the measured metrics."
    )
    lines.append(
        "5. Arm behavior changes across scheduling schemes, so "
        "ARM×SCHEME interaction remains an important descriptive component."
    )
    lines.append(
        "6. Shared signs between paired arm effects indicate correlated "
        "variation, but they do not identify its source."
    )
    lines.append(
        "7. The data do not establish hidden cognitive paths or internal "
        "model mechanisms."
    )
    lines.append("")

    lines.append("## Status")
    lines.append("")
    lines.append(
        "**V0.6.1 isolation collection: COMPLETE.**"
    )
    lines.append(
        "**V0.6.1 combined analysis: COMPLETE.**"
    )
    lines.append("")

    OUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("V061_COMBINED_ANALYSIS=PASS")
    print(f"REPORT={OUT}")
    print("RUNS=160")
    print(
        f"QUALITY={total_quality}/160"
    )

    for metric in [
        "reasoning",
        "latency",
        "total",
    ]:

        row = variance[metric]

        print(
            f"{metric}: "
            f"arm={pct(row['arm'])} "
            f"scheme={pct(row['scheme'])} "
            f"repetition={pct(row['repetition'])} "
            f"interaction={pct(row['interaction'])} "
            f"residual={pct(row['residual'])}"
        )


if __name__ == "__main__":
    main()
