import json
import math
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"D:\PROMPTFORGE\promptforge-harness-v0.1")
REPORT_DIR = ROOT / "harness" / "reports"
OUT = ROOT / "harness" / "analysis" / "v062_residual_forensics.md"

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
# Helpers
# ============================================================

def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def variance(values):
    values = list(values)

    if len(values) < 2:
        return float("nan")

    m = mean(values)

    return sum(
        (x - m) ** 2
        for x in values
    ) / (len(values) - 1)


def stdev(values):
    v = variance(values)

    if math.isnan(v):
        return float("nan")

    return math.sqrt(v)


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


def fmt(value, digits=3):
    if isinstance(value, float) and math.isnan(value):
        return "NA"

    return f"{value:.{digits}f}"


def pct(value):
    if isinstance(value, float) and math.isnan(value):
        return "NA"

    return f"{100.0 * value:.1f}%"


def load_report(path):
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: root must be object"
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
            f"Missing required field '{key}'"
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

    metrics_obj = run.get("metrics", {})
    execution_obj = run.get("execution", {})

    metrics_keys = (
        list(metrics_obj.keys())
        if isinstance(metrics_obj, dict)
        else []
    )

    execution_keys = (
        list(execution_obj.keys())
        if isinstance(execution_obj, dict)
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


def group(rows, key):
    result = defaultdict(list)

    for row in rows:
        result[key(row)].append(row)

    return result


# ============================================================
# Residual construction
# ============================================================

def repetition_blocks(rows):
    return group(
        rows,
        lambda r: (
            r["scheme"],
            r["repetition"],
        ),
    )


def attach_relative_position(rows):
    blocks = repetition_blocks(rows)
    result = []

    for key, block in blocks.items():

        ordered = sorted(
            block,
            key=lambda r: r["position"],
        )

        for relative_position, row in enumerate(
            ordered
        ):
            x = dict(row)
            x["relative_position"] = (
                relative_position
            )
            x["block_key"] = key
            result.append(x)

    return result


def build_paired_deltas(rows):
    blocks = repetition_blocks(rows)

    deltas = []

    for key, block in blocks.items():

        by_arm = {
            row["arm"]: row
            for row in block
        }

        baseline = by_arm.get(
            "selection_only"
        )

        if baseline is None:
            raise ValueError(
                f"Missing selection_only in {key}"
            )

        for arm in TEST_ARMS:

            test = by_arm.get(arm)

            if test is None:
                raise ValueError(
                    f"Missing {arm} in {key}"
                )

            deltas.append({
                "scheme": key[0],
                "repetition": key[1],
                "arm": arm,
                "position": test["position"],
                "relative_position":
                    test["position"]
                    - min(
                        r["position"]
                        for r in block
                    ),
                "delta_reasoning":
                    test["reasoning"]
                    - baseline["reasoning"],
                "delta_output":
                    test["output"]
                    - baseline["output"],
                "delta_total":
                    test["total"]
                    - baseline["total"],
                "delta_latency":
                    test["latency"]
                    - baseline["latency"],
                "delta_input":
                    test["input"]
                    - baseline["input"],
            })

    return deltas


# ============================================================
# Sequence analysis
# ============================================================

def sequence_rows(rows):
    result = []

    blocks = repetition_blocks(rows)

    for (scheme, repetition), block in blocks.items():

        ordered = sorted(
            block,
            key=lambda r: r["position"],
        )

        for i in range(1, len(ordered)):

            prev = ordered[i - 1]
            curr = ordered[i]

            result.append({
                "scheme": scheme,
                "repetition": repetition,
                "previous_arm": prev["arm"],
                "current_arm": curr["arm"],
                "position": curr["position"],
                "reasoning": curr["reasoning"],
                "latency": curr["latency"],
                "delta_reasoning":
                    curr["reasoning"]
                    - prev["reasoning"],
                "delta_latency":
                    curr["latency"]
                    - prev["latency"],
            })

    return result


# ============================================================
# Main
# ============================================================

def main():

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
                f"{scheme}: expected 40 runs"
            )

        rows.extend(
            normalize(run, scheme)
            for run in raw
        )

    if len(rows) != 160:
        raise ValueError(
            f"Expected 160 rows, got {len(rows)}"
        )

    # --------------------------------------------------------
    # Integrity
    # --------------------------------------------------------

    if not all(row["quality"] for row in rows):
        raise ValueError(
            "Quality preservation failed"
        )

    for scheme in SCHEMES:

        sr = [
            r
            for r in rows
            if r["scheme"] == scheme
        ]

        if len(sr) != 40:
            raise ValueError(
                f"{scheme}: incorrect count"
            )

        for arm in ARMS:

            n = sum(
                1
                for r in sr
                if r["arm"] == arm
            )

            if n != 10:
                raise ValueError(
                    f"{scheme}/{arm}: expected 10"
                )

    # --------------------------------------------------------
    # Add relative positions
    # --------------------------------------------------------

    rows = attach_relative_position(rows)

    # --------------------------------------------------------
    # 1. Intra-repetition dispersion
    # --------------------------------------------------------

    blocks = repetition_blocks(rows)

    block_dispersion = []

    for key, block in sorted(blocks.items()):

        reasoning_values = [
            r["reasoning"]
            for r in block
        ]

        total_values = [
            r["total"]
            for r in block
        ]

        latency_values = [
            r["latency"]
            for r in block
        ]

        block_dispersion.append({
            "scheme": key[0],
            "repetition": key[1],
            "reasoning_sd":
                stdev(reasoning_values),
            "reasoning_range":
                max(reasoning_values)
                - min(reasoning_values),
            "total_sd":
                stdev(total_values),
            "total_range":
                max(total_values)
                - min(total_values),
            "latency_sd":
                stdev(latency_values),
            "latency_range":
                max(latency_values)
                - min(latency_values),
        })

    # --------------------------------------------------------
    # 2. Arm-specific residual variance
    # --------------------------------------------------------

    arm_dispersion = {}

    for arm in ARMS:

        ar = [
            r
            for r in rows
            if r["arm"] == arm
        ]

        arm_dispersion[arm] = {
            "n": len(ar),
            "reasoning_mean":
                mean(
                    r["reasoning"]
                    for r in ar
                ),
            "reasoning_sd":
                stdev(
                    r["reasoning"]
                    for r in ar
                ),
            "total_sd":
                stdev(
                    r["total"]
                    for r in ar
                ),
            "latency_sd":
                stdev(
                    r["latency"]
                    for r in ar
                ),
        }

    # --------------------------------------------------------
    # 3. Paired deltas
    # --------------------------------------------------------

    deltas = build_paired_deltas(rows)

    delta_summary = {}

    for arm in TEST_ARMS:

        dr = [
            d
            for d in deltas
            if d["arm"] == arm
        ]

        delta_summary[arm] = {
            "n": len(dr),
            "reasoning_mean":
                mean(
                    d["delta_reasoning"]
                    for d in dr
                ),
            "reasoning_sd":
                stdev(
                    d["delta_reasoning"]
                    for d in dr
                ),
            "reasoning_abs_mean":
                mean(
                    abs(d["delta_reasoning"])
                    for d in dr
                ),
            "reasoning_min":
                min(
                    d["delta_reasoning"]
                    for d in dr
                ),
            "reasoning_max":
                max(
                    d["delta_reasoning"]
                    for d in dr
                ),
            "total_mean":
                mean(
                    d["delta_total"]
                    for d in dr
                ),
            "total_sd":
                stdev(
                    d["delta_total"]
                    for d in dr
                ),
            "latency_mean":
                mean(
                    d["delta_latency"]
                    for d in dr
                ),
        }

    # --------------------------------------------------------
    # 4. Delta correlations
    # --------------------------------------------------------

    delta_correlations = {}

    for arm in TEST_ARMS:

        dr = [
            d
            for d in deltas
            if d["arm"] == arm
        ]

        delta_correlations[arm] = {
            "position_vs_reasoning":
                pearson(
                    [
                        d["position"]
                        for d in dr
                    ],
                    [
                        d["delta_reasoning"]
                        for d in dr
                    ],
                ),
            "relative_position_vs_reasoning":
                pearson(
                    [
                        d["relative_position"]
                        for d in dr
                    ],
                    [
                        d["delta_reasoning"]
                        for d in dr
                    ],
                ),
            "delta_reasoning_vs_latency":
                pearson(
                    [
                        d["delta_reasoning"]
                        for d in dr
                    ],
                    [
                        d["delta_latency"]
                        for d in dr
                    ],
                ),
            "delta_reasoning_vs_delta_output":
                pearson(
                    [
                        d["delta_reasoning"]
                        for d in dr
                    ],
                    [
                        d["delta_output"]
                        for d in dr
                    ],
                ),
            "delta_reasoning_vs_delta_total":
                pearson(
                    [
                        d["delta_reasoning"]
                        for d in dr
                    ],
                    [
                        d["delta_total"]
                        for d in dr
                    ],
                ),
        }

    # --------------------------------------------------------
    # 5. Sequence / adjacency effects
    # --------------------------------------------------------

    seq = sequence_rows(rows)

    transition_groups = group(
        seq,
        lambda r: (
            r["previous_arm"],
            r["current_arm"],
        ),
    )

    transition_summary = []

    for key, group_rows in sorted(
        transition_groups.items()
    ):

        transition_summary.append({
            "previous_arm": key[0],
            "current_arm": key[1],
            "n": len(group_rows),
            "mean_reasoning_delta":
                mean(
                    r["delta_reasoning"]
                    for r in group_rows
                ),
            "mean_latency_delta":
                mean(
                    r["delta_latency"]
                    for r in group_rows
                ),
        })

    # --------------------------------------------------------
    # 6. Extreme residual cases
    # --------------------------------------------------------

    extreme = sorted(
        deltas,
        key=lambda d: abs(
            d["delta_reasoning"]
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # 7. Per-scheme delta dispersion
    # --------------------------------------------------------

    scheme_delta_summary = {}

    for scheme in SCHEMES:

        scheme_delta_summary[scheme] = {}

        for arm in TEST_ARMS:

            dr = [
                d
                for d in deltas
                if (
                    d["scheme"] == scheme
                    and d["arm"] == arm
                )
            ]

            scheme_delta_summary[
                scheme
            ][arm] = {
                "mean_reasoning":
                    mean(
                        d["delta_reasoning"]
                        for d in dr
                    ),
                "sd_reasoning":
                    stdev(
                        d["delta_reasoning"]
                        for d in dr
                    ),
                "min_reasoning":
                    min(
                        d["delta_reasoning"]
                        for d in dr
                    ),
                "max_reasoning":
                    max(
                        d["delta_reasoning"]
                        for d in dr
                    ),
            }

    # --------------------------------------------------------
    # 8. Residual proxy:
    #    variance remaining after subtracting paired baseline
    # --------------------------------------------------------

    residual_proxy = {}

    for arm in TEST_ARMS:

        dr = [
            d
            for d in deltas
            if d["arm"] == arm
        ]

        delta_var = variance(
            d["delta_reasoning"]
            for d in dr
        )

        residual_proxy[arm] = {
            "delta_reasoning_variance":
                delta_var,
            "delta_reasoning_sd":
                math.sqrt(delta_var)
                if not math.isnan(delta_var)
                else float("nan"),
        }

    # --------------------------------------------------------
    # 9. Cross-arm matched delta correlations
    # --------------------------------------------------------

    matched = defaultdict(dict)

    for d in deltas:

        key = (
            d["scheme"],
            d["repetition"],
        )

        matched[key][
            d["arm"]
        ] = d["delta_reasoning"]

    cross_arm_delta = []

    for i in range(len(TEST_ARMS)):

        for j in range(i + 1, len(TEST_ARMS)):

            arm_a = TEST_ARMS[i]
            arm_b = TEST_ARMS[j]

            xs = []
            ys = []

            for values in matched.values():

                if (
                    arm_a in values
                    and arm_b in values
                ):
                    xs.append(
                        values[arm_a]
                    )
                    ys.append(
                        values[arm_b]
                    )

            cross_arm_delta.append({
                "arm_a": arm_a,
                "arm_b": arm_b,
                "n": len(xs),
                "r":
                    pearson(xs, ys),
            })

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    lines = []

    lines.append(
        "# PromptForge V0.6.2 — T003 Residual Variance Forensics"
    )
    lines.append("")
    lines.append(
        "This analysis reuses the frozen 160-run V0.6.1 "
        "dataset. No new API executions are performed."
    )
    lines.append("")

    lines.append("## Integrity")
    lines.append("")
    lines.append(
        "- Runs: **160**"
    )
    lines.append(
        "- Quality: **160/160**"
    )
    lines.append(
        "- Schemes: **4**"
    )
    lines.append(
        "- Arms: **4**"
    )
    lines.append(
        "- Repetitions per arm/scheme: **10**"
    )
    lines.append("")

    lines.append(
        "## 1. Intra-repetition dispersion"
    )
    lines.append("")
    lines.append(
        "| Scheme | Mean reasoning SD | Mean reasoning range | "
        "Mean total SD | Mean latency SD |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    for scheme in SCHEMES:

        sr = [
            x
            for x in block_dispersion
            if x["scheme"] == scheme
        ]

        lines.append(
            f"| {scheme} | "
            f"{fmt(mean(x['reasoning_sd'] for x in sr))} | "
            f"{fmt(mean(x['reasoning_range'] for x in sr))} | "
            f"{fmt(mean(x['total_sd'] for x in sr))} | "
            f"{fmt(mean(x['latency_sd'] for x in sr))} |"
        )

    lines.append("")

    lines.append(
        "## 2. Arm-specific dispersion"
    )
    lines.append("")
    lines.append(
        "| Arm | N | Reasoning mean | Reasoning SD | "
        "Total SD | Latency SD |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    for arm in ARMS:

        x = arm_dispersion[arm]

        lines.append(
            f"| `{arm}` | "
            f"{x['n']} | "
            f"{fmt(x['reasoning_mean'])} | "
            f"{fmt(x['reasoning_sd'])} | "
            f"{fmt(x['total_sd'])} | "
            f"{fmt(x['latency_sd'])} |"
        )

    lines.append("")

    lines.append(
        "## 3. Paired delta dispersion vs selection_only"
    )
    lines.append("")
    lines.append(
        "| Arm | N | Mean Δreasoning | SD Δreasoning | "
        "Mean |Δreasoning| | Min | Max | "
        "Mean Δtotal | Mean Δlatency |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for arm in TEST_ARMS:

        x = delta_summary[arm]

        lines.append(
            f"| `{arm}` | "
            f"{x['n']} | "
            f"{fmt(x['reasoning_mean'])} | "
            f"{fmt(x['reasoning_sd'])} | "
            f"{fmt(x['reasoning_abs_mean'])} | "
            f"{fmt(x['reasoning_min'])} | "
            f"{fmt(x['reasoning_max'])} | "
            f"{fmt(x['total_mean'])} | "
            f"{fmt(x['latency_mean'])} |"
        )

    lines.append("")

    lines.append(
        "## 4. Correlations of paired residual proxies"
    )
    lines.append("")
    lines.append(
        "| Arm | r(abs position, Δreasoning) | "
        "r(relative position, Δreasoning) | "
        "r(Δreasoning, Δlatency) | "
        "r(Δreasoning, Δoutput) | "
        "r(Δreasoning, Δtotal) |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    for arm in TEST_ARMS:

        x = delta_correlations[arm]

        lines.append(
            f"| `{arm}` | "
            f"{fmt(x['position_vs_reasoning'])} | "
            f"{fmt(x['relative_position_vs_reasoning'])} | "
            f"{fmt(x['delta_reasoning_vs_latency'])} | "
            f"{fmt(x['delta_reasoning_vs_delta_output'])} | "
            f"{fmt(x['delta_reasoning_vs_delta_total'])} |"
        )

    lines.append("")

    lines.append(
        "## 5. Sequence / adjacency transitions"
    )
    lines.append("")
    lines.append(
        "| Previous arm | Current arm | N | "
        "Mean Δreasoning | Mean Δlatency |"
    )
    lines.append(
        "|---|---|---:|---:|---:|"
    )

    for x in transition_summary:

        lines.append(
            f"| `{x['previous_arm']}` | "
            f"`{x['current_arm']}` | "
            f"{x['n']} | "
            f"{fmt(x['mean_reasoning_delta'])} | "
            f"{fmt(x['mean_latency_delta'])} |"
        )

    lines.append("")

    lines.append(
        "## 6. Per-scheme residual behavior"
    )
    lines.append("")
    lines.append(
        "| Scheme | Arm | Mean Δreasoning | SD | Min | Max |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|"
    )

    for scheme in SCHEMES:

        for arm in TEST_ARMS:

            x = scheme_delta_summary[
                scheme
            ][arm]

            lines.append(
                f"| {scheme} | `{arm}` | "
                f"{fmt(x['mean_reasoning'])} | "
                f"{fmt(x['sd_reasoning'])} | "
                f"{fmt(x['min_reasoning'])} | "
                f"{fmt(x['max_reasoning'])} |"
            )

    lines.append("")

    lines.append(
        "## 7. Largest paired reasoning deviations"
    )
    lines.append("")
    lines.append(
        "| Rank | Scheme | Repetition | Arm | "
        "Position | Δreasoning |"
    )
    lines.append(
        "|---:|---|---:|---|---:|---:|"
    )

    for rank, x in enumerate(
        extreme[:20],
        start=1,
    ):

        lines.append(
            f"| {rank} | "
            f"{x['scheme']} | "
            f"{x['repetition']} | "
            f"`{x['arm']}` | "
            f"{x['position']} | "
            f"{fmt(x['delta_reasoning'])} |"
        )

    lines.append("")

    lines.append(
        "## 8. Cross-arm matched residual correlation"
    )
    lines.append("")
    lines.append(
        "| Arm A | Arm B | N | "
        "r(matched Δreasoning) |"
    )
    lines.append(
        "|---|---|---:|---:|"
    )

    for x in cross_arm_delta:

        lines.append(
            f"| `{x['arm_a']}` | "
            f"`{x['arm_b']}` | "
            f"{x['n']} | "
            f"{fmt(x['r'])} |"
        )

    lines.append("")

    lines.append(
        "## Interpretation rules"
    )
    lines.append("")
    lines.append(
        "The residual proxy here is deliberately descriptive: "
        "paired Δreasoning removes the contemporaneous "
        "`selection_only` baseline within each scheme×repetition."
    )
    lines.append("")
    lines.append(
        "A large remaining SD in paired Δreasoning means that "
        "the unexplained variation is not removed simply by "
        "subtracting the matched baseline."
    )
    lines.append("")
    lines.append(
        "Cross-arm correlation of matched deltas is evidence of "
        "shared variation only. It does not identify its cause."
    )
    lines.append("")
    lines.append(
        "Sequence transition effects are exploratory and do not "
        "establish carry-over causality."
    )
    lines.append("")

    lines.append(
        "## Status"
    )
    lines.append("")
    lines.append(
        "**V0.6.2 residual forensics: COMPLETE.**"
    )
    lines.append(
        "**Input dataset: frozen V0.6.1 160-run collection.**"
    )
    lines.append(
        "**New API calls: 0.**"
    )
    lines.append("")

    OUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("V062_RESIDUAL_FORENSICS=PASS")
    print(f"REPORT={OUT}")
    print("INPUT_RUNS=160")
    print("NEW_API_CALLS=0")

    print("")
    print("Top residual proxy results:")

    for arm in TEST_ARMS:

        x = delta_summary[arm]

        print(
            f"{arm}: "
            f"mean_delta={fmt(x['reasoning_mean'])} "
            f"sd_delta={fmt(x['reasoning_sd'])} "
            f"abs_mean={fmt(x['reasoning_abs_mean'])} "
            f"min={fmt(x['reasoning_min'])} "
            f"max={fmt(x['reasoning_max'])}"
        )


if __name__ == "__main__":
    main()
