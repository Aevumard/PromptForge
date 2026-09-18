import json
import math
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"D:\PROMPTFORGE\promptforge-harness-v0.1")
REPORT_DIR = ROOT / "harness" / "reports"
OUT = ROOT / "harness" / "analysis" / "v063_robust_baseline.md"

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


# ============================================================
# Basic helpers
# ============================================================

def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def median(values):
    values = sorted(values)

    if not values:
        return float("nan")

    n = len(values)
    mid = n // 2

    if n % 2:
        return float(values[mid])

    return (values[mid - 1] + values[mid]) / 2.0


def stdev(values):
    values = list(values)

    if len(values) < 2:
        return float("nan")

    m = mean(values)

    return math.sqrt(
        sum((x - m) ** 2 for x in values)
        / (len(values) - 1)
    )


def mad(values):
    values = list(values)

    if not values:
        return float("nan")

    m = median(values)

    return median(
        [abs(x - m) for x in values]
    )


def trimmed_mean(values, trim_each_side=1):
    values = sorted(values)

    if len(values) <= 2 * trim_each_side:
        return mean(values)

    trimmed = values[
        trim_each_side:
        len(values) - trim_each_side
    ]

    return mean(trimmed)


def pearson(xs, ys):
    xs = list(xs)
    ys = list(ys)

    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")

    mx = mean(xs)
    my = mean(ys)

    num = sum(
        (x - mx) * (y - my)
        for x, y in zip(xs, ys)
    )

    den_x = math.sqrt(
        sum((x - mx) ** 2 for x in xs)
    )

    den_y = math.sqrt(
        sum((y - my) ** 2 for y in ys)
    )

    den = den_x * den_y

    if den == 0:
        return float("nan")

    return num / den


def fmt(x, digits=3):
    if isinstance(x, float) and math.isnan(x):
        return "NA"

    return f"{x:.{digits}f}"


def pct(x):
    if isinstance(x, float) and math.isnan(x):
        return "NA"

    return f"{100.0 * x:.1f}%"


# ============================================================
# V0.6.1 schema
# ============================================================

def load_report(path):
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: report root must be object"
        )

    runs = data.get("runs")

    if not isinstance(runs, list):
        raise ValueError(
            f"{path}: missing runs[]"
        )

    return runs


def required(run, key):
    if key not in run:
        raise ValueError(
            f"Missing field '{key}'"
        )

    return run[key]


def metric(run, paths, name):
    for path in paths:

        value = run
        ok = True

        for key in path:

            if (
                not isinstance(value, dict)
                or key not in value
            ):
                ok = False
                break

            value = value[key]

        if ok and isinstance(value, (int, float)):
            return float(value)

    metrics = run.get("metrics", {})
    execution = run.get("execution", {})

    metrics_keys = (
        list(metrics.keys())
        if isinstance(metrics, dict)
        else []
    )

    execution_keys = (
        list(execution.keys())
        if isinstance(execution, dict)
        else []
    )

    raise ValueError(
        f"Metric '{name}' not found. "
        f"metrics={metrics_keys}; "
        f"execution={execution_keys}"
    )


def normalize(run, scheme):
    return {
        "scheme": scheme,
        "repetition": int(
            required(run, "repetition")
        ),
        "arm": str(
            required(run, "arm_id")
        ),
        "position": int(
            required(
                run,
                "global_execution_position",
            )
        ),
        "quality": bool(
            required(
                run,
                "quality_pass",
            )
        ),
        "input": metric(
            run,
            [
                ("metrics", "input_tokens"),
                ("execution", "input_tokens"),
                (
                    "execution",
                    "usage",
                    "input_tokens",
                ),
            ],
            "input_tokens",
        ),
        "reasoning": metric(
            run,
            [
                ("metrics", "reasoning_tokens"),
                ("execution", "reasoning_tokens"),
                (
                    "execution",
                    "usage",
                    "reasoning_tokens",
                ),
            ],
            "reasoning_tokens",
        ),
        "output": metric(
            run,
            [
                ("metrics", "output_tokens"),
                ("execution", "output_tokens"),
                (
                    "execution",
                    "usage",
                    "output_tokens",
                ),
            ],
            "output_tokens",
        ),
        "total": metric(
            run,
            [
                ("metrics", "total_tokens"),
                ("execution", "total_tokens"),
                (
                    "execution",
                    "usage",
                    "total_tokens",
                ),
            ],
            "total_tokens",
        ),
        "latency": metric(
            run,
            [
                ("metrics", "latency_ms"),
                ("execution", "latency_ms"),
                ("execution", "duration_ms"),
                ("execution", "elapsed_ms"),
            ],
            "latency_ms",
        ),
    }


# ============================================================
# Grouping
# ============================================================

def group(rows, key_fn):
    groups = defaultdict(list)

    for row in rows:
        groups[key_fn(row)].append(row)

    return groups


# ============================================================
# Baseline estimators
# ============================================================

def baseline_samples(rows, scheme, excluded_repetition):
    """
    Returns all selection_only reasoning observations
    from the same scheme except the current repetition.
    """
    values = [
        row["reasoning"]
        for row in rows
        if (
            row["scheme"] == scheme
            and row["arm"] == "selection_only"
            and row["repetition"] != excluded_repetition
        )
    ]

    if len(values) != 9:
        raise ValueError(
            f"{scheme}/rep={excluded_repetition}: "
            f"expected 9 leave-one-out baseline samples, "
            f"got {len(values)}"
        )

    return values


def baseline_estimators(values):
    return {
        "loo_mean": mean(values),
        "loo_median": median(values),
        "loo_trimmed_mean_20": trimmed_mean(
            values,
            trim_each_side=1,
        ),
    }


# ============================================================
# Build robust contrasts
# ============================================================

def build_contrasts(rows):
    """
    For each test arm and scheme×repetition, calculate:
      - original single-baseline delta
      - LOO mean delta
      - LOO median delta
      - LOO trimmed-mean delta
    """
    blocks = group(
        rows,
        lambda r: (
            r["scheme"],
            r["repetition"],
        ),
    )

    contrasts = []

    for (
        scheme,
        repetition,
    ), block in sorted(blocks.items()):

        by_arm = {
            row["arm"]: row
            for row in block
        }

        if "selection_only" not in by_arm:
            raise ValueError(
                f"Missing selection_only: "
                f"{scheme}/rep={repetition}"
            )

        contemporaneous = by_arm[
            "selection_only"
        ]

        baseline_values = baseline_samples(
            rows,
            scheme,
            repetition,
        )

        estimators = baseline_estimators(
            baseline_values
        )

        for arm in TEST_ARMS:

            test = by_arm.get(arm)

            if test is None:
                raise ValueError(
                    f"Missing {arm}: "
                    f"{scheme}/rep={repetition}"
                )

            contrasts.append({
                "scheme": scheme,
                "repetition": repetition,
                "arm": arm,
                "position": test["position"],

                "test_reasoning":
                    test["reasoning"],

                "original_baseline":
                    contemporaneous["reasoning"],

                "loo_mean_baseline":
                    estimators["loo_mean"],

                "loo_median_baseline":
                    estimators["loo_median"],

                "loo_trimmed_baseline":
                    estimators[
                        "loo_trimmed_mean_20"
                    ],

                "original_delta":
                    test["reasoning"]
                    - contemporaneous[
                        "reasoning"
                    ],

                "loo_mean_delta":
                    test["reasoning"]
                    - estimators[
                        "loo_mean"
                    ],

                "loo_median_delta":
                    test["reasoning"]
                    - estimators[
                        "loo_median"
                    ],

                "loo_trimmed_delta":
                    test["reasoning"]
                    - estimators[
                        "loo_trimmed_mean_20"
                    ],

                "delta_total_original":
                    test["total"]
                    - contemporaneous[
                        "total"
                    ],

                "delta_latency_original":
                    test["latency"]
                    - contemporaneous[
                        "latency"
                    ],
            })

    return contrasts


# ============================================================
# Summaries
# ============================================================

def summarize_method(rows, delta_field):
    deltas = [
        row[delta_field]
        for row in rows
    ]

    negatives = sum(
        1
        for x in deltas
        if x < 0
    )

    positives = sum(
        1
        for x in deltas
        if x > 0
    )

    ties = sum(
        1
        for x in deltas
        if x == 0
    )

    return {
        "n": len(deltas),
        "mean": mean(deltas),
        "sd": stdev(deltas),
        "abs_mean": mean(
            abs(x)
            for x in deltas
        ),
        "min": min(deltas),
        "max": max(deltas),
        "beneficial": negatives,
        "harmful": positives,
        "tie": ties,
    }


def summarize_by_scheme(rows, delta_field):
    result = {}

    for scheme in SCHEMES:

        scheme_rows = [
            row
            for row in rows
            if row["scheme"] == scheme
        ]

        result[scheme] = summarize_method(
            scheme_rows,
            delta_field,
        )

    return result


# ============================================================
# Baseline diagnostics
# ============================================================

def baseline_diagnostics(rows):
    result = {}

    for scheme in SCHEMES:

        values = [
            row["reasoning"]
            for row in rows
            if (
                row["scheme"] == scheme
                and row["arm"] == "selection_only"
            )
        ]

        result[scheme] = {
            "n": len(values),
            "mean": mean(values),
            "median": median(values),
            "trimmed_mean":
                trimmed_mean(
                    values,
                    trim_each_side=1,
                ),
            "sd": stdev(values),
            "mad": mad(values),
            "min": min(values),
            "max": max(values),
            "range":
                max(values) - min(values),
        }

    return result


# ============================================================
# Method agreement
# ============================================================

def sign(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0


def agreement(rows, field_a, field_b):
    a = [
        sign(row[field_a])
        for row in rows
    ]

    b = [
        sign(row[field_b])
        for row in rows
    ]

    agree = sum(
        x == y
        for x, y in zip(a, b)
    )

    return {
        "n": len(rows),
        "agreement": agree,
        "rate": agree / len(rows)
        if rows else float("nan"),
    }


# ============================================================
# Robustness against baseline outliers
# ============================================================

def baseline_influence(rows):
    """
    For each contrast:
      influence = contemporaneous baseline
                  - robust baseline
    """
    result = []

    for row in rows:

        result.append({
            "scheme": row["scheme"],
            "repetition": row["repetition"],
            "arm": row["arm"],
            "original_baseline":
                row["original_baseline"],
            "loo_trimmed_baseline":
                row["loo_trimmed_baseline"],
            "baseline_shift":
                row["original_baseline"]
                - row["loo_trimmed_baseline"],
            "delta_shift":
                row["original_delta"]
                - row["loo_trimmed_delta"],
        })

    return result


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load all frozen reports
    # --------------------------------------------------------

    rows = []

    for scheme in SCHEMES:

        path = (
            REPORT_DIR
            / f"v061_t003_{scheme}.json"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        raw = load_report(path)

        if len(raw) != 40:
            raise ValueError(
                f"{scheme}: expected 40 runs, "
                f"got {len(raw)}"
            )

        rows.extend(
            normalize(run, scheme)
            for run in raw
        )

    if len(rows) != 160:
        raise ValueError(
            f"Expected 160 runs, got {len(rows)}"
        )

    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    if not all(
        row["quality"]
        for row in rows
    ):
        raise ValueError(
            "Not all runs have quality_pass=True"
        )

    for scheme in SCHEMES:

        scheme_rows = [
            row
            for row in rows
            if row["scheme"] == scheme
        ]

        if len(scheme_rows) != 40:
            raise ValueError(
                f"{scheme}: expected 40 runs"
            )

        for arm in ARMS:

            count = sum(
                1
                for row in scheme_rows
                if row["arm"] == arm
            )

            if count != 10:
                raise ValueError(
                    f"{scheme}/{arm}: expected 10"
                )

    # --------------------------------------------------------
    # Baseline diagnostics
    # --------------------------------------------------------

    base_diag = baseline_diagnostics(rows)

    # --------------------------------------------------------
    # Build robust contrasts
    # --------------------------------------------------------

    contrasts = build_contrasts(rows)

    if len(contrasts) != 120:
        raise ValueError(
            f"Expected 120 test-arm contrasts, "
            f"got {len(contrasts)}"
        )

    # --------------------------------------------------------
    # Method summaries
    # --------------------------------------------------------

    methods = {
        "single_observation":
            "original_delta",
        "loo_mean":
            "loo_mean_delta",
        "loo_median":
            "loo_median_delta",
        "loo_trimmed_mean_20":
            "loo_trimmed_delta",
    }

    global_method_summary = {}

    for method_name, field in methods.items():

        global_method_summary[
            method_name
        ] = {}

        for arm in TEST_ARMS:

            arm_rows = [
                row
                for row in contrasts
                if row["arm"] == arm
            ]

            global_method_summary[
                method_name
            ][arm] = summarize_method(
                arm_rows,
                field,
            )

    # --------------------------------------------------------
    # By-scheme method summaries
    # --------------------------------------------------------

    scheme_method_summary = {}

    for method_name, field in methods.items():

        scheme_method_summary[
            method_name
        ] = {}

        for arm in TEST_ARMS:

            scheme_method_summary[
                method_name
            ][arm] = {}

            for scheme in SCHEMES:

                subset = [
                    row
                    for row in contrasts
                    if (
                        row["arm"] == arm
                        and row["scheme"] == scheme
                    )
                ]

                scheme_method_summary[
                    method_name
                ][arm][scheme] = (
                    summarize_method(
                        subset,
                        field,
                    )
                )

    # --------------------------------------------------------
    # Method agreement
    # --------------------------------------------------------

    agreement_table = {}

    method_items = list(
        methods.items()
    )

    for i in range(len(method_items)):

        for j in range(i + 1, len(method_items)):

            name_a, field_a = method_items[i]
            name_b, field_b = method_items[j]

            agreement_table[
                (name_a, name_b)
            ] = agreement(
                contrasts,
                field_a,
                field_b,
            )

    # --------------------------------------------------------
    # Baseline influence
    # --------------------------------------------------------

    influence = baseline_influence(
        contrasts
    )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    correlation_table = {}

    for arm in TEST_ARMS:

        arm_rows = [
            row
            for row in contrasts
            if row["arm"] == arm
        ]

        correlation_table[arm] = {
            "original_vs_trimmed":
                pearson(
                    [
                        row[
                            "original_delta"
                        ]
                        for row in arm_rows
                    ],
                    [
                        row[
                            "loo_trimmed_delta"
                        ]
                        for row in arm_rows
                    ],
                ),
            "mean_vs_trimmed":
                pearson(
                    [
                        row[
                            "loo_mean_delta"
                        ]
                        for row in arm_rows
                    ],
                    [
                        row[
                            "loo_trimmed_delta"
                        ]
                        for row in arm_rows
                    ],
                ),
            "median_vs_trimmed":
                pearson(
                    [
                        row[
                            "loo_median_delta"
                        ]
                        for row in arm_rows
                    ],
                    [
                        row[
                            "loo_trimmed_delta"
                        ]
                        for row in arm_rows
                    ],
                ),
        }

    # --------------------------------------------------------
    # Extreme baseline influence cases
    # --------------------------------------------------------

    extreme_influence = sorted(
        influence,
        key=lambda x: abs(
            x["baseline_shift"]
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # Robust baseline aggregate across schemes
    # --------------------------------------------------------

    baseline_global = {
        "mean":
            mean(
                row["mean"]
                for row in base_diag.values()
            ),
        "median":
            median(
                row["median"]
                for row in base_diag.values()
            ),
        "trimmed":
            mean(
                row["trimmed_mean"]
                for row in base_diag.values()
            ),
    }

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    lines = []

    lines.append(
        "# PromptForge V0.6.3 — Robust Baseline Estimation"
    )
    lines.append("")

    lines.append(
        "Frozen input: V0.6.1 T003, 160 real API executions."
    )
    lines.append(
        "New API executions: **0**."
    )
    lines.append("")

    lines.append("## Objective")
    lines.append("")
    lines.append(
        "Replace the single contemporaneous `selection_only` "
        "baseline with multi-sample same-scheme estimators."
    )
    lines.append("")
    lines.append(
        "Primary estimator: **leave-one-out trimmed mean 20%** "
        "of the nine other `selection_only` observations within "
        "the same scheme."
    )
    lines.append("")
    lines.append(
        "Sensitivity estimators: leave-one-out mean and "
        "leave-one-out median."
    )
    lines.append("")

    lines.append("## Integrity")
    lines.append("")
    lines.append(
        "- Frozen runs: **160/160**"
    )
    lines.append(
        "- Quality: **160/160**"
    )
    lines.append(
        "- Schemes: **4**"
    )
    lines.append(
        "- Repetitions per scheme: **10**"
    )
    lines.append(
        "- Robust test-arm contrasts: **120**"
    )
    lines.append("")

    # Baseline diagnostics

    lines.append(
        "## 1. Baseline `selection_only` diagnostics"
    )
    lines.append("")
    lines.append(
        "| Scheme | N | Mean | Median | "
        "Trimmed mean | SD | MAD | Min | Max | Range |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for scheme in SCHEMES:

        x = base_diag[scheme]

        lines.append(
            f"| {scheme} | "
            f"{x['n']} | "
            f"{fmt(x['mean'])} | "
            f"{fmt(x['median'])} | "
            f"{fmt(x['trimmed_mean'])} | "
            f"{fmt(x['sd'])} | "
            f"{fmt(x['mad'])} | "
            f"{fmt(x['min'])} | "
            f"{fmt(x['max'])} | "
            f"{fmt(x['range'])} |"
        )

    lines.append("")

    # Global effects

    lines.append(
        "## 2. Global effect by baseline estimator"
    )
    lines.append("")
    lines.append(
        "| Baseline method | Arm | N | Mean Δreasoning | "
        "SD | Mean |Δ| | Min | Max | Beneficial | Harmful |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for method_name in methods:

        for arm in TEST_ARMS:

            x = global_method_summary[
                method_name
            ][arm]

            lines.append(
                f"| `{method_name}` | "
                f"`{arm}` | "
                f"{x['n']} | "
                f"{fmt(x['mean'])} | "
                f"{fmt(x['sd'])} | "
                f"{fmt(x['abs_mean'])} | "
                f"{fmt(x['min'])} | "
                f"{fmt(x['max'])} | "
                f"{x['beneficial']} | "
                f"{x['harmful']} |"
            )

    lines.append("")

    # Scheme effects, primary estimator

    lines.append(
        "## 3. Primary estimator by scheme"
    )
    lines.append("")
    lines.append(
        "Primary = leave-one-out trimmed mean 20%."
    )
    lines.append("")
    lines.append(
        "| Scheme | Arm | N | Mean Δreasoning | "
        "SD | Min | Max | Beneficial | Harmful |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )

    primary_name = "loo_trimmed_mean_20"

    for scheme in SCHEMES:

        for arm in TEST_ARMS:

            x = scheme_method_summary[
                primary_name
            ][arm][scheme]

            lines.append(
                f"| {scheme} | "
                f"`{arm}` | "
                f"{x['n']} | "
                f"{fmt(x['mean'])} | "
                f"{fmt(x['sd'])} | "
                f"{fmt(x['min'])} | "
                f"{fmt(x['max'])} | "
                f"{x['beneficial']} | "
                f"{x['harmful']} |"
            )

    lines.append("")

    # Method agreement

    lines.append(
        "## 4. Agreement between baseline estimators"
    )
    lines.append("")
    lines.append(
        "| Method A | Method B | N | "
        "Sign agreement | Rate |"
    )
    lines.append(
        "|---|---|---:|---:|---:|"
    )

    for (
        name_a,
        name_b,
    ), x in agreement_table.items():

        lines.append(
            f"| `{name_a}` | "
            f"`{name_b}` | "
            f"{x['n']} | "
            f"{x['agreement']} | "
            f"{pct(x['rate'])} |"
        )

    lines.append("")

    # Correlations

    lines.append(
        "## 5. Correlation of estimated effects"
    )
    lines.append("")
    lines.append(
        "| Arm | Original vs trimmed | "
        "LOO mean vs trimmed | "
        "LOO median vs trimmed |"
    )
    lines.append(
        "|---|---:|---:|---:|"
    )

    for arm in TEST_ARMS:

        x = correlation_table[arm]

        lines.append(
            f"| `{arm}` | "
            f"{fmt(x['original_vs_trimmed'])} | "
            f"{fmt(x['mean_vs_trimmed'])} | "
            f"{fmt(x['median_vs_trimmed'])} |"
        )

    lines.append("")

    # Baseline influence

    lines.append(
        "## 6. Baseline influence"
    )
    lines.append("")
    lines.append(
        "Positive baseline_shift means the contemporaneous "
        "single baseline was above the robust leave-one-out "
        "trimmed estimate; negative means it was below it."
    )
    lines.append("")
    lines.append(
        "| Scheme | Rep | Arm | Single baseline | "
        "Robust baseline | Baseline shift | Effect shift |"
    )
    lines.append(
        "|---|---:|---|---:|---:|---:|---:|"
    )

    for x in extreme_influence[:20]:

        lines.append(
            f"| {x['scheme']} | "
            f"{x['repetition']} | "
            f"`{x['arm']}` | "
            f"{fmt(x['original_baseline'])} | "
            f"{fmt(x['loo_trimmed_baseline'])} | "
            f"{fmt(x['baseline_shift'])} | "
            f"{fmt(x['delta_shift'])} |"
        )

    lines.append("")

    # Primary scheme table

    lines.append(
        "## 7. Primary estimator stability across schemes"
    )
    lines.append("")
    lines.append(
        "| Arm | Interleaved | Randomized | Blocked | Latin square |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    for arm in TEST_ARMS:

        vals = []

        for scheme in SCHEMES:
            vals.append(
                scheme_method_summary[
                    primary_name
                ][arm][scheme]["mean"]
            )

        lines.append(
            f"| `{arm}` | "
            + " | ".join(
                fmt(x)
                for x in vals
            )
            + " |"
        )

    lines.append("")

    # Interpretation

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The primary question is whether representation-related "
        "effects survive replacing the contemporaneous single "
        "baseline with a multi-sample same-scheme estimator."
    )
    lines.append("")
    lines.append(
        "A result that remains directionally similar across the "
        "LOO mean, LOO median and LOO trimmed mean is more robust "
        "to baseline outliers than the original single-observation "
        "contrast."
    )
    lines.append("")
    lines.append(
        "Large disagreement between estimators indicates that "
        "the baseline distribution itself is materially influencing "
        "the apparent arm effect."
    )
    lines.append("")
    lines.append(
        "The primary estimator does not remove scheme dependence. "
        "It only reduces sensitivity to individual baseline "
        "observations within each scheme."
    )
    lines.append("")

    lines.append("## Status")
    lines.append("")
    lines.append(
        "**V0.6.3 Robust Baseline Estimation: COMPLETE.**"
    )
    lines.append(
        "**Frozen input: V0.6.1 160-run dataset.**"
    )
    lines.append(
        "**New API calls: 0.**"
    )
    lines.append("")

    OUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("V063_ROBUST_BASELINE=PASS")
    print(f"REPORT={OUT}")
    print("INPUT_RUNS=160")
    print("ROBUST_CONTRASTS=120")
    print("NEW_API_CALLS=0")

    print("")
    print("PRIMARY ESTIMATOR = LOO_TRIMMED_MEAN_20")

    for arm in TEST_ARMS:

        x = global_method_summary[
            "loo_trimmed_mean_20"
        ][arm]

        print(
            f"{arm}: "
            f"mean_delta={fmt(x['mean'])} "
            f"sd={fmt(x['sd'])} "
            f"abs_mean={fmt(x['abs_mean'])} "
            f"min={fmt(x['min'])} "
            f"max={fmt(x['max'])} "
            f"beneficial={x['beneficial']} "
            f"harmful={x['harmful']}"
        )


if __name__ == "__main__":
    main()
