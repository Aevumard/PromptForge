import json
import math
from pathlib import Path

from collections import defaultdict

ROOT = Path(r"D:\PROMPTFORGE\promptforge-harness-v0.1")
REPORT_DIR = ROOT / "harness" / "reports"
OUT = ROOT / "harness" / "analysis" / "v064_confirmatory_comparison.md"

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

BASELINE = "selection_only"
PRIMARY_METHOD = "loo_trimmed_mean_20"

METRIC = "reasoning"


# ============================================================
# Basic statistics
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


def median(values):
    values = sorted(values)

    if not values:
        return float("nan")

    n = len(values)
    k = n // 2

    if n % 2:
        return float(values[k])

    return (values[k - 1] + values[k]) / 2.0


def trimmed_mean(values, trim_each_side=1):
    values = sorted(values)

    if len(values) <= 2 * trim_each_side:
        return mean(values)

    return mean(
        values[
            trim_each_side:
            len(values) - trim_each_side
        ]
    )


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

    dx = math.sqrt(
        sum((x - mx) ** 2 for x in xs)
    )

    dy = math.sqrt(
        sum((y - my) ** 2 for y in ys)
    )

    denominator = dx * dy

    if denominator == 0:
        return float("nan")

    return numerator / denominator


def fmt(x, digits=4):
    if isinstance(x, float) and math.isnan(x):
        return "NA"

    return f"{x:.{digits}f}"


def pct(x):
    if isinstance(x, float) and math.isnan(x):
        return "NA"

    return f"{100.0 * x:.2f}%"


# ============================================================
# Distribution helpers
# ============================================================

def normal_cdf(x):
    return 0.5 * (
        1.0
        + math.erf(
            x / math.sqrt(2.0)
        )
    )


def normal_two_sided_p(z):
    return 2.0 * (
        1.0 - normal_cdf(abs(z))
    )


def normal_ci(mean_value, se, z=1.959963984540054):
    return (
        mean_value - z * se,
        mean_value + z * se,
    )


# ============================================================
# Report loading
# ============================================================

def load_report(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as fh:
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
            f"Missing required field: {key}"
        )

    return run[key]


def metric(run, paths, name):

    for path in paths:

        value = run
        valid = True

        for key in path:

            if (
                not isinstance(value, dict)
                or key not in value
            ):
                valid = False
                break

            value = value[key]

        if valid and isinstance(
            value,
            (int, float),
        ):
            return float(value)

    raise ValueError(
        f"Could not locate metric '{name}'"
    )


def normalize(run, scheme):
    return {
        "scheme": scheme,
        "repetition": int(
            required(
                run,
                "repetition",
            )
        ),
        "arm": str(
            required(
                run,
                "arm_id",
            )
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
        "reasoning": metric(
            run,
            [
                (
                    "metrics",
                    "reasoning_tokens",
                ),
                (
                    "execution",
                    "reasoning_tokens",
                ),
                (
                    "execution",
                    "usage",
                    "reasoning_tokens",
                ),
            ],
            "reasoning_tokens",
        ),
    }


# ============================================================
# Robust baseline
# ============================================================

def baseline_samples(
    rows,
    scheme,
    excluded_repetition,
):
    values = [
        row["reasoning"]
        for row in rows
        if (
            row["scheme"] == scheme
            and row["arm"] == BASELINE
            and row["repetition"]
            != excluded_repetition
        )
    ]

    if len(values) != 9:
        raise ValueError(
            f"{scheme}/rep={excluded_repetition}: "
            f"expected 9 leave-one-out baseline values, "
            f"got {len(values)}"
        )

    return values


def robust_baseline(
    rows,
    scheme,
    repetition,
):
    values = baseline_samples(
        rows,
        scheme,
        repetition,
    )

    return trimmed_mean(
        values,
        trim_each_side=1,
    )


def build_primary_contrasts(rows):

    blocks = defaultdict(dict)

    for row in rows:
        key = (
            row["scheme"],
            row["repetition"],
        )

        blocks[key][row["arm"]] = row

    contrasts = []

    for (
        scheme,
        repetition,
    ), arms in sorted(blocks.items()):

        robust_base = robust_baseline(
            rows,
            scheme,
            repetition,
        )

        for arm in TEST_ARMS:

            test = arms.get(arm)

            if test is None:
                raise ValueError(
                    f"Missing arm {arm} "
                    f"in {scheme}/rep={repetition}"
                )

            contrasts.append({
                "scheme": scheme,
                "repetition": repetition,
                "block": (
                    f"{scheme}:{repetition}"
                ),
                "arm": arm,
                "position": test["position"],
                "delta_reasoning":
                    test["reasoning"]
                    - robust_base,
                "test_reasoning":
                    test["reasoning"],
                "robust_baseline":
                    robust_base,
            })

    return contrasts


# ============================================================
# Pairwise block-level comparisons
# ============================================================

def matched_arm_values(
    contrasts,
    arm_a,
    arm_b,
):
    index = defaultdict(dict)

    for row in contrasts:

        index[
            (
                row["scheme"],
                row["repetition"],
            )
        ][row["arm"]] = row["delta_reasoning"]

    values_a = []
    values_b = []

    for key in sorted(index):

        if (
            arm_a not in index[key]
            or arm_b not in index[key]
        ):
            continue

        values_a.append(
            index[key][arm_a]
        )

        values_b.append(
            index[key][arm_b]
        )

    return values_a, values_b


def paired_difference_stats(
    contrasts,
    arm_a,
    arm_b,
):
    a, b = matched_arm_values(
        contrasts,
        arm_a,
        arm_b,
    )

    differences = [
        x - y
        for x, y in zip(a, b)
    ]

    n = len(differences)

    if n < 2:
        raise ValueError(
            f"Insufficient paired observations "
            f"for {arm_a} vs {arm_b}"
        )

    m = mean(differences)
    sd = stdev(differences)
    se = sd / math.sqrt(n)

    t_approx = (
        m / se
        if se > 0
        else float("nan")
    )

    p_approx = (
        normal_two_sided_p(t_approx)
        if not math.isnan(t_approx)
        else float("nan")
    )

    ci_low, ci_high = normal_ci(
        m,
        se,
    )

    return {
        "n": n,
        "mean_difference": m,
        "sd": sd,
        "se": se,
        "t_approx": t_approx,
        "p_approx": p_approx,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cohen_dz":
            m / sd
            if sd > 0
            else float("nan"),
    }


# ============================================================
# Try scipy / statsmodels
# ============================================================

def optional_statsmodels():
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        return sm, smf
    except Exception:
        return None, None


def optional_scipy():
    try:
        from scipy import stats

        return stats
    except Exception:
        return None


# ============================================================
# Factorial analysis
# ============================================================

def build_dataframe(contrasts):
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "pandas is required for factorial analysis"
        ) from exc

    return pd.DataFrame([
        {
            "delta_reasoning": row[
                "delta_reasoning"
            ],
            "arm": row["arm"],
            "scheme": row["scheme"],
            "block": row["block"],
            "repetition": row["repetition"],
        }
        for row in contrasts
    ])


def factorial_analysis(contrasts):
    sm, smf = optional_statsmodels()

    if smf is None:
        return {
            "available": False,
            "reason": (
                "statsmodels unavailable"
            ),
        }

    df = build_dataframe(
        contrasts
    )

    formula = (
        "delta_reasoning "
        "~ C(arm) * C(scheme)"
    )

    model = smf.ols(
        formula,
        data=df,
    ).fit()

    cluster_model = model.get_robustcov_results(
        cov_type="cluster",
        groups=df["block"],
    )

    # Keep conventional ANOVA descriptive only.
    anova = sm.stats.anova_lm(
        model,
        typ=2,
    )

    return {
        "available": True,
        "model": model,
        "cluster_model": cluster_model,
        "anova": anova,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "n": len(df),
    }


# ============================================================
# Type-II descriptive effect sizes
# ============================================================

def eta_squared_from_anova(
    anova,
    term,
):
    if term not in anova.index:
        return float("nan")

    ss_term = float(
        anova.loc[term, "sum_sq"]
    )

    residual_term = "Residual"

    if residual_term not in anova.index:
        return float("nan")

    ss_error = float(
        anova.loc[
            residual_term,
            "sum_sq",
        ]
    )

    denominator = (
        ss_term + ss_error
    )

    if denominator == 0:
        return float("nan")

    return ss_term / denominator


# ============================================================
# Model contrast extraction
# ============================================================

def cluster_contrast(
    cluster_model,
    term,
):
    params = cluster_model.params
    covariance = cluster_model.cov_params()

    if term not in params.index:
        return None

    estimate = float(
        params[term]
    )

    se = float(
        math.sqrt(
            covariance.loc[
                term,
                term,
            ]
        )
    )

    z = (
        estimate / se
        if se > 0
        else float("nan")
    )

    p = (
        normal_two_sided_p(z)
        if not math.isnan(z)
        else float("nan")
    )

    low, high = normal_ci(
        estimate,
        se,
    )

    return {
        "estimate": estimate,
        "se": se,
        "z": z,
        "p": p,
        "ci_low": low,
        "ci_high": high,
    }


# ============================================================
# Explicit pairwise contrasts via cell means
# ============================================================

def cell_mean(
    contrasts,
    arm,
    scheme=None,
):
    values = [
        row["delta_reasoning"]
        for row in contrasts
        if (
            row["arm"] == arm
            and (
                scheme is None
                or row["scheme"] == scheme
            )
        )
    ]

    return mean(values)


def bootstrap_ci(
    values,
    seed=6401,
    iterations=20000,
):
    try:
        import numpy as np
    except Exception:
        return (
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(seed)

    values = np.asarray(
        list(values),
        dtype=float,
    )

    if len(values) < 2:
        return (
            float("nan"),
            float("nan"),
        )

    samples = rng.choice(
        values,
        size=(
            iterations,
            len(values),
        ),
        replace=True,
    )

    means = samples.mean(
        axis=1
    )

    return (
        float(
            np.quantile(
                means,
                0.025,
            )
        ),
        float(
            np.quantile(
                means,
                0.975,
            )
        ),
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load frozen dataset
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
                f"{scheme}: expected 40 runs"
            )

        rows.extend(
            normalize(
                run,
                scheme,
            )
            for run in raw
        )

    if len(rows) != 160:
        raise ValueError(
            f"Expected 160 runs, "
            f"got {len(rows)}"
        )

    if not all(
        row["quality"]
        for row in rows
    ):
        raise ValueError(
            "Quality validation failed"
        )

    # --------------------------------------------------------
    # Primary robust contrasts
    # --------------------------------------------------------

    contrasts = build_primary_contrasts(
        rows
    )

    if len(contrasts) != 120:
        raise ValueError(
            f"Expected 120 robust contrasts, "
            f"got {len(contrasts)}"
        )

    # --------------------------------------------------------
    # Pairwise comparisons
    # --------------------------------------------------------

    pair_specs = [
        (
            "A_vs_representation",
            "selection_representation_A",
            "selection_representation",
        ),
        (
            "B_vs_representation",
            "selection_representation_B",
            "selection_representation",
        ),
        (
            "B_vs_A",
            "selection_representation_B",
            "selection_representation_A",
        ),
    ]

    pairwise = {}

    for name, arm_a, arm_b in pair_specs:

        result = paired_difference_stats(
            contrasts,
            arm_a,
            arm_b,
        )

        differences = []

        values_a, values_b = matched_arm_values(
            contrasts,
            arm_a,
            arm_b,
        )

        differences = [
            a - b
            for a, b in zip(
                values_a,
                values_b,
            )
        ]

        boot_low, boot_high = bootstrap_ci(
            differences,
            seed=6401,
        )

        result[
            "bootstrap_ci_low"
        ] = boot_low

        result[
            "bootstrap_ci_high"
        ] = boot_high

        pairwise[name] = result

    # --------------------------------------------------------
    # Global robust means
    # --------------------------------------------------------

    arm_means = {}

    for arm in TEST_ARMS:

        values = [
            row["delta_reasoning"]
            for row in contrasts
            if row["arm"] == arm
        ]

        boot_low, boot_high = bootstrap_ci(
            values,
            seed=6401,
        )

        arm_means[arm] = {
            "n": len(values),
            "mean": mean(values),
            "sd": stdev(values),
            "bootstrap_low": boot_low,
            "bootstrap_high": boot_high,
        }

    # --------------------------------------------------------
    # Scheme cell means
    # --------------------------------------------------------

    scheme_cells = {}

    for scheme in SCHEMES:

        scheme_cells[scheme] = {}

        for arm in TEST_ARMS:

            values = [
                row["delta_reasoning"]
                for row in contrasts
                if (
                    row["scheme"] == scheme
                    and row["arm"] == arm
                )
            ]

            scheme_cells[
                scheme
            ][arm] = {
                "n": len(values),
                "mean": mean(values),
                "sd": stdev(values),
            }

    # --------------------------------------------------------
    # Factorial model
    # --------------------------------------------------------

    factorial = factorial_analysis(
        contrasts
    )

    # --------------------------------------------------------
    # Agreement / difference distribution
    # --------------------------------------------------------

    a_values, rep_values = matched_arm_values(
        contrasts,
        "selection_representation_A",
        "selection_representation",
    )

    a_minus_rep = [
        a - b
        for a, b in zip(
            a_values,
            rep_values,
        )
    ]

    paired_sign = {
        "positive": sum(
            x > 0
            for x in a_minus_rep
        ),
        "negative": sum(
            x < 0
            for x in a_minus_rep
        ),
        "zero": sum(
            x == 0
            for x in a_minus_rep
        ),
    }

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    lines = []

    lines.append(
        "# PromptForge V0.6.4 — Confirmatory-Style Comparison"
    )
    lines.append("")
    lines.append(
        "Frozen input: V0.6.1 T003, 160 real API runs."
    )
    lines.append(
        "Primary response: robust Δreasoning using "
        "LOO_TRIMMED_MEAN_20 baseline."
    )
    lines.append(
        "New API executions: **0**."
    )
    lines.append("")

    lines.append("## Important inferential scope")
    lines.append("")
    lines.append(
        "This is a confirmatory-style analysis of the frozen "
        "dataset, not an independently preregistered replication."
    )
    lines.append("")
    lines.append(
        "Failure to reject a difference of zero is not itself "
        "evidence of equivalence. A formal equivalence claim "
        "requires a pre-specified practical equivalence margin."
    )
    lines.append("")

    lines.append("## 1. Primary robust arm effects")
    lines.append("")
    lines.append(
        "| Arm | N | Mean Δreasoning | SD | "
        "Bootstrap 95% CI |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    for arm in TEST_ARMS:

        x = arm_means[arm]

        lines.append(
            f"| `{arm}` | "
            f"{x['n']} | "
            f"{fmt(x['mean'])} | "
            f"{fmt(x['sd'])} | "
            f"[{fmt(x['bootstrap_low'])}, "
            f"{fmt(x['bootstrap_high'])}] |"
        )

    lines.append("")

    lines.append(
        "## 2. Direct paired comparisons"
    )
    lines.append("")
    lines.append(
        "Differences are computed within each scheme×repetition "
        "block using the robust baseline-derived Δreasoning."
    )
    lines.append("")
    lines.append(
        "| Contrast | N | Mean difference | SD | "
        "SE | Approx p | 95% CI | Cohen dz | "
        "Bootstrap CI |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for name, _arm_a, _arm_b in pair_specs:

        x = pairwise[name]

        lines.append(
            f"| `{name}` | "
            f"{x['n']} | "
            f"{fmt(x['mean_difference'])} | "
            f"{fmt(x['sd'])} | "
            f"{fmt(x['se'])} | "
            f"{fmt(x['p_approx'])} | "
            f"[{fmt(x['ci_low'])}, "
            f"{fmt(x['ci_high'])}] | "
            f"{fmt(x['cohen_dz'])} | "
            f"[{fmt(x['bootstrap_ci_low'])}, "
            f"{fmt(x['bootstrap_ci_high'])}] |"
        )

    lines.append("")

    lines.append(
        "## 3. A vs representation: paired sign distribution"
    )
    lines.append("")
    lines.append(
        f"- A > representation in Δreasoning: "
        f"**{paired_sign['positive']}**"
    )
    lines.append(
        f"- A < representation in Δreasoning: "
        f"**{paired_sign['negative']}**"
    )
    lines.append(
        f"- Ties: **{paired_sign['zero']}**"
    )
    lines.append("")

    lines.append(
        "## 4. Scheme × arm robust cell means"
    )
    lines.append("")
    lines.append(
        "| Scheme | Representation | A | B |"
    )
    lines.append(
        "|---|---:|---:|---:|"
    )

    for scheme in SCHEMES:

        r = scheme_cells[
            scheme
        ]

        lines.append(
            f"| {scheme} | "
            f"{fmt(r['selection_representation']['mean'])} | "
            f"{fmt(r['selection_representation_A']['mean'])} | "
            f"{fmt(r['selection_representation_B']['mean'])} |"
        )

    lines.append("")

    lines.append(
        "## 5. Factorial model"
    )
    lines.append("")
    lines.append(
        "Model: `Δreasoning ~ ARM + SCHEME + ARM×SCHEME`."
    )
    lines.append(
        "Because all arms in a scheme×repetition block share the "
        "same robust baseline construction, cluster-robust standard "
        "errors by `scheme×repetition` are reported where available."
    )
    lines.append("")

    if not factorial["available"]:

        lines.append(
            f"Factorial model unavailable: "
            f"{factorial['reason']}"
        )

    else:

        anova = factorial["anova"]

        lines.append(
            f"- N = **{factorial['n']}**"
        )
        lines.append(
            f"- R² = **{fmt(factorial['r_squared'])}**"
        )
        lines.append(
            f"- Adjusted R² = **{fmt(factorial['adj_r_squared'])}**"
        )
        lines.append("")

        lines.append(
            "### Descriptive Type-II sums-of-squares effect sizes"
        )
        lines.append("")

        lines.append(
            "| Term | η² | Sum Sq | df | F | p |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|"
        )

        for term in [
            "C(arm)",
            "C(scheme)",
            "C(arm):C(scheme)",
        ]:

            if term not in anova.index:
                continue

            row = anova.loc[term]

            eta = eta_squared_from_anova(
                anova,
                term,
            )

            lines.append(
                f"| `{term}` | "
                f"{fmt(eta)} | "
                f"{fmt(float(row['sum_sq']))} | "
                f"{fmt(float(row['df']))} | "
                f"{fmt(float(row['F']))} | "
                f"{fmt(float(row['PR(>F)']))} |"
            )

        lines.append("")

        lines.append(
            "### Cluster-robust model coefficients"
        )
        lines.append("")

        cluster_model = factorial[
            "cluster_model"
        ]

        lines.append(
            "| Term | Estimate | SE | z | p | 95% CI |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|"
        )

        # statsmodels 0.15 returns robust covariance results
        # with params/bse as numpy arrays. Preserve the original
        # parameter names from the fitted model and map by position.
        param_names = list(
            factorial["model"].params.index
        )

        robust_params = list(
            cluster_model.params
        )

        robust_bse = list(
            cluster_model.bse
        )

        if len(param_names) != len(robust_params):
            raise ValueError(
                "Cluster-robust parameter count mismatch: "
                f"names={len(param_names)} "
                f"params={len(robust_params)}"
            )

        if len(param_names) != len(robust_bse):
            raise ValueError(
                "Cluster-robust SE count mismatch: "
                f"names={len(param_names)} "
                f"bse={len(robust_bse)}"
            )

        for idx, term in enumerate(
            param_names
        ):

            estimate = float(
                robust_params[idx]
            )

            se = float(
                robust_bse[idx]
            )

            z = (
                estimate / se
                if se > 0
                else float("nan")
            )

            p = (
                normal_two_sided_p(z)
                if not math.isnan(z)
                else float("nan")
            )

            low, high = normal_ci(
                estimate,
                se,
            )

            lines.append(
                f"| `{term}` | "
                f"{fmt(estimate)} | "
                f"{fmt(se)} | "
                f"{fmt(z)} | "
                f"{fmt(p)} | "
                f"[{fmt(low)}, {fmt(high)}] |"
            )

    lines.append("")

    lines.append(
        "## 6. Interpretation"
    )
    lines.append("")
    lines.append(
        "The primary confirmatory contrast is A vs representation."
    )
    lines.append("")
    lines.append(
        "A difference whose confidence interval contains zero "
        "is compatible with no detectable difference in this "
        "dataset; it is not automatically evidence of equivalence."
    )
    lines.append("")
    lines.append(
        "The B comparisons are treated as secondary/exploratory "
        "contrasts. Their interpretation must retain the observed "
        "ARM×SCHEME interaction."
    )
    lines.append("")
    lines.append(
        "Formal equivalence would require a practical margin "
        "specified independently of the observed effect."
    )
    lines.append("")

    lines.append(
        "## Status"
    )
    lines.append("")
    lines.append(
        "**V0.6.4 confirmatory-style analysis: COMPLETE.**"
    )
    lines.append(
        "**Frozen input: 160 API runs.**"
    )
    lines.append(
        "**New API calls: 0.**"
    )
    lines.append("")

    OUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        "V064_CONFIRMATORY_ANALYSIS=PASS"
    )
    print(
        f"REPORT={OUT}"
    )
    print(
        "INPUT_RUNS=160"
    )
    print(
        "ROBUST_CONTRASTS=120"
    )
    print(
        "NEW_API_CALLS=0"
    )

    print("")
    print(
        "PRIMARY A VS REPRESENTATION"
    )

    x = pairwise[
        "A_vs_representation"
    ]

    print(
        f"mean_difference={fmt(x['mean_difference'])}"
    )

    print(
        f"ci95=[{fmt(x['ci_low'])}, "
        f"{fmt(x['ci_high'])}]"
    )

    print(
        f"p_approx={fmt(x['p_approx'])}"
    )

    print(
        f"cohen_dz={fmt(x['cohen_dz'])}"
    )

    print("")

    if factorial["available"]:

        print(
            f"R2={fmt(factorial['r_squared'])}"
        )

        anova = factorial["anova"]

        for term in [
            "C(arm)",
            "C(scheme)",
            "C(arm):C(scheme)",
        ]:

            if term in anova.index:

                eta = eta_squared_from_anova(
                    anova,
                    term,
                )

                print(
                    f"{term}: "
                    f"eta2={fmt(eta)} "
                    f"F={fmt(float(anova.loc[term, 'F']))} "
                    f"p={fmt(float(anova.loc[term, 'PR(>F)']))}"
                )
    else:
        print(
            "FACTORIAL_MODEL=UNAVAILABLE"
        )


if __name__ == "__main__":
    main()
