from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]

REPORT_FILE = ROOT / "harness" / "reports" / "v07_confirmatory_deepseek.json"
OUTPUT_MD = ROOT / "harness" / "analysis" / "v07_confirmatory_analysis.md"
OUTPUT_JSON = ROOT / "harness" / "analysis" / "v07_confirmatory_analysis.json"

TASKS = ["T001", "T002", "T003", "T004"]

ARMS = [
    "selection_only",
    "selection_representation",
    "selection_representation_A",
]

BASELINE = "selection_only"
REPRESENTATION = "selection_representation"
ARM_A = "selection_representation_A"

REPS = list(range(1, 16))

EXPECTED_RUNS = 180

ALPHA = 0.05
DZ_THRESHOLD = 0.30
TOST_MARGIN = 10.0


def fail(message: str):
    raise RuntimeError(message)


def mean(values):
    return statistics.fmean(values)


def sd(values):
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def trimmed_mean_20(values):
    """
    20% two-sided trimming for the fixed 14-value LOO pool:
    remove min and max, mean remaining 12.
    """
    if len(values) != 14:
        fail(f"Expected 14 LOO values, got {len(values)}")

    ordered = sorted(values)

    return mean(ordered[1:-1])


def one_sample_t(values):
    n = len(values)

    if n < 2:
        return {
            "t": float("nan"),
            "df": max(0, n - 1),
            "p": float("nan"),
        }

    m = mean(values)
    s = sd(values)

    if s == 0.0:
        if m == 0.0:
            return {
                "t": 0.0,
                "df": n - 1,
                "p": 1.0,
            }

        return {
            "t": math.inf if m > 0 else -math.inf,
            "df": n - 1,
            "p": 0.0,
        }

    statistic = m / (s / math.sqrt(n))

    p_value = 2.0 * (1.0 - t.cdf(abs(statistic), n - 1))

    return {
        "t": statistic,
        "df": n - 1,
        "p": float(p_value),
    }


def cohens_dz(values):
    s = sd(values)

    if s == 0.0:
        if mean(values) == 0.0:
            return 0.0
        return math.inf if mean(values) > 0 else -math.inf

    return mean(values) / s


def holm_bonferroni(p_map):
    ordered = sorted(p_map.items(), key=lambda item: item[1])

    m = len(ordered)
    adjusted = {}

    running = 0.0

    for rank, (task_id, p_value) in enumerate(ordered, start=1):
        candidate = min(1.0, (m - rank + 1) * p_value)
        running = max(running, candidate)
        adjusted[task_id] = running

    return adjusted


def tost(values, margin):
    """
    Global TOST, alpha=0.05, equivalence interval [-margin,+margin].
    90% CI is reported because TOST alpha=.05.
    """

    n = len(values)

    if n < 2:
        fail("TOST requires at least two paired observations")

    m = mean(values)
    s = sd(values)
    se = s / math.sqrt(n)
    df = n - 1

    if se == 0.0:
        ci_low = m
        ci_high = m

        p_lower = 0.0 if m > -margin else 1.0
        p_upper = 0.0 if m < margin else 1.0

        equivalent = (
            p_lower < ALPHA
            and p_upper < ALPHA
            and ci_low >= -margin
            and ci_high <= margin
        )

        return {
            "n": n,
            "mean": m,
            "sd": s,
            "se": se,
            "df": df,
            "p_lower": p_lower,
            "p_upper": p_upper,
            "ci90_low": ci_low,
            "ci90_high": ci_high,
            "equivalent": equivalent,
        }

    # H01: mean <= -margin
    t_lower = (m + margin) / se
    p_lower = 1.0 - t.cdf(t_lower, df)

    # H02: mean >= +margin
    t_upper = (m - margin) / se
    p_upper = t.cdf(t_upper, df)

    critical = t.ppf(1.0 - ALPHA, df)

    ci90_low = m - critical * se
    ci90_high = m + critical * se

    equivalent = (
        p_lower < ALPHA
        and p_upper < ALPHA
        and ci90_low >= -margin
        and ci90_high <= margin
    )

    return {
        "n": n,
        "mean": m,
        "sd": s,
        "se": se,
        "df": df,
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "ci90_low": float(ci90_low),
        "ci90_high": float(ci90_high),
        "equivalent": bool(equivalent),
    }


def load_report():
    if not REPORT_FILE.exists():
        fail(f"Report not found: {REPORT_FILE}")

    raw = REPORT_FILE.read_bytes()

    report_hash = hashlib.sha256(raw).hexdigest()

    report = json.loads(raw.decode("utf-8"))

    return report, report_hash


def index_runs(report):
    runs = report.get("runs")

    if not isinstance(runs, list):
        fail("report.runs is not a list")

    if len(runs) != EXPECTED_RUNS:
        fail(f"Expected {EXPECTED_RUNS} runs, got {len(runs)}")

    by_cell = {}
    positions = []

    for run in runs:
        task_id = run.get("task_id")
        arm_id = run.get("arm_id")
        repetition = run.get("repetition")

        key = (task_id, arm_id, repetition)

        if key in by_cell:
            fail(f"Duplicate cell: {key}")

        by_cell[key] = run

        positions.append(
            run.get("global_execution_position")
        )

    if sorted(positions) != list(range(EXPECTED_RUNS)):
        fail("global_execution_position is not exactly 0..179")

    return by_cell


def verify_collection(report, by_cell):
    if report.get("status") != "COMPLETE":
        fail(f"Report status is not COMPLETE: {report.get('status')}")

    if report.get("provider") != "deepseek":
        fail(f"Unexpected provider: {report.get('provider')}")

    if report.get("scheme") != "latin_square":
        fail(f"Unexpected scheme: {report.get('scheme')}")

    if report.get("repetitions") != 15:
        fail(f"Unexpected repetitions: {report.get('repetitions')}")

    for task_id in TASKS:
        for repetition in REPS:
            for arm_id in ARMS:
                key = (task_id, arm_id, repetition)

                if key not in by_cell:
                    fail(f"Missing experimental cell: {key}")

                run = by_cell[key]

                execution = run.get("execution", {})
                verification = run.get("verification", {})

                provider_status = execution.get("provider_status")

                if provider_status != "MODEL_OK":
                    fail(
                        f"Provider failure at "
                        f"{task_id}/{arm_id}/rep{repetition}: "
                        f"{provider_status}"
                    )

                if verification.get("passed") is not True:
                    fail(
                        f"Verification failure at "
                        f"{task_id}/{arm_id}/rep{repetition}"
                    )

                if run.get("quality_pass") is not True:
                    fail(
                        f"quality_pass != true at "
                        f"{task_id}/{arm_id}/rep{repetition}"
                    )

    return {
        "runs": len(by_cell),
        "quality_passes": EXPECTED_RUNS,
    }


def metric_value(run, metric):
    return run["metrics"][metric]


def build_task_analysis(task_id, by_cell):
    baseline_runs = {}
    representation_runs = {}
    a_runs = {}

    for rep in REPS:
        baseline_runs[rep] = by_cell[(task_id, BASELINE, rep)]
        representation_runs[rep] = by_cell[(task_id, REPRESENTATION, rep)]
        a_runs[rep] = by_cell[(task_id, ARM_A, rep)]

    robust_delta_reasoning_rep = []
    robust_delta_reasoning_a = []

    robust_delta_output_rep = []
    robust_delta_output_a = []

    robust_delta_total_rep = []
    robust_delta_total_a = []

    raw_reasoning_baseline = []
    raw_reasoning_rep = []
    raw_reasoning_a = []

    raw_output_baseline = []
    raw_output_rep = []
    raw_output_a = []

    raw_total_baseline = []
    raw_total_rep = []
    raw_total_a = []

    row_details = []

    for rep in REPS:
        loo_baseline_reasoning = [
            metric_value(
                baseline_runs[other_rep],
                "reasoning_tokens",
            )
            for other_rep in REPS
            if other_rep != rep
        ]

        loo_baseline_output = [
            metric_value(
                baseline_runs[other_rep],
                "output_tokens",
            )
            for other_rep in REPS
            if other_rep != rep
        ]

        loo_baseline_total = [
            metric_value(
                baseline_runs[other_rep],
                "total_tokens",
            )
            for other_rep in REPS
            if other_rep != rep
        ]

        baseline_reasoning = metric_value(
            baseline_runs[rep],
            "reasoning_tokens",
        )

        baseline_output = metric_value(
            baseline_runs[rep],
            "output_tokens",
        )

        baseline_total = metric_value(
            baseline_runs[rep],
            "total_tokens",
        )

        rep_reasoning = metric_value(
            representation_runs[rep],
            "reasoning_tokens",
        )

        rep_output = metric_value(
            representation_runs[rep],
            "output_tokens",
        )

        rep_total = metric_value(
            representation_runs[rep],
            "total_tokens",
        )

        a_reasoning = metric_value(
            a_runs[rep],
            "reasoning_tokens",
        )

        a_output = metric_value(
            a_runs[rep],
            "output_tokens",
        )

        a_total = metric_value(
            a_runs[rep],
            "total_tokens",
        )

        robust_baseline_reasoning = trimmed_mean_20(
            loo_baseline_reasoning
        )

        robust_baseline_output = trimmed_mean_20(
            loo_baseline_output
        )

        robust_baseline_total = trimmed_mean_20(
            loo_baseline_total
        )

        delta_rep_reasoning = (
            rep_reasoning
            - robust_baseline_reasoning
        )

        delta_a_reasoning = (
            a_reasoning
            - robust_baseline_reasoning
        )

        delta_rep_output = (
            rep_output
            - robust_baseline_output
        )

        delta_a_output = (
            a_output
            - robust_baseline_output
        )

        delta_rep_total = (
            rep_total
            - robust_baseline_total
        )

        delta_a_total = (
            a_total
            - robust_baseline_total
        )

        robust_delta_reasoning_rep.append(
            delta_rep_reasoning
        )

        robust_delta_reasoning_a.append(
            delta_a_reasoning
        )

        robust_delta_output_rep.append(
            delta_rep_output
        )

        robust_delta_output_a.append(
            delta_a_output
        )

        robust_delta_total_rep.append(
            delta_rep_total
        )

        robust_delta_total_a.append(
            delta_a_total
        )

        raw_reasoning_baseline.append(
            baseline_reasoning
        )

        raw_reasoning_rep.append(
            rep_reasoning
        )

        raw_reasoning_a.append(
            a_reasoning
        )

        raw_output_baseline.append(
            baseline_output
        )

        raw_output_rep.append(
            rep_output
        )

        raw_output_a.append(
            a_output
        )

        raw_total_baseline.append(
            baseline_total
        )

        raw_total_rep.append(
            rep_total
        )

        raw_total_a.append(
            a_total
        )

        row_details.append(
            {
                "repetition": rep,
                "robust_baseline_reasoning": robust_baseline_reasoning,
                "robust_delta_reasoning_representation": delta_rep_reasoning,
                "robust_delta_reasoning_A": delta_a_reasoning,
                "robust_baseline_output": robust_baseline_output,
                "robust_delta_output_representation": delta_rep_output,
                "robust_delta_output_A": delta_a_output,
                "robust_baseline_total": robust_baseline_total,
                "robust_delta_total_representation": delta_rep_total,
                "robust_delta_total_A": delta_a_total,
            }
        )

    rep_test = one_sample_t(
        robust_delta_reasoning_rep
    )

    a_test = one_sample_t(
        robust_delta_reasoning_a
    )

    return {
        "task_id": task_id,
        "n": 15,
        "baseline": {
            "reasoning_mean": mean(raw_reasoning_baseline),
            "reasoning_sd": sd(raw_reasoning_baseline),
            "output_mean": mean(raw_output_baseline),
            "output_sd": sd(raw_output_baseline),
            "total_mean": mean(raw_total_baseline),
            "total_sd": sd(raw_total_baseline),
        },
        "representation": {
            "mean_robust_delta_reasoning": mean(
                robust_delta_reasoning_rep
            ),
            "sd_robust_delta_reasoning": sd(
                robust_delta_reasoning_rep
            ),
            "min_robust_delta_reasoning": min(
                robust_delta_reasoning_rep
            ),
            "max_robust_delta_reasoning": max(
                robust_delta_reasoning_rep
            ),
            "cohen_dz": cohens_dz(
                robust_delta_reasoning_rep
            ),
            "t": rep_test["t"],
            "df": rep_test["df"],
            "p_raw": rep_test["p"],
            "reasoning_mean": mean(raw_reasoning_rep),
            "reasoning_sd": sd(raw_reasoning_rep),
            "mean_robust_delta_output": mean(
                robust_delta_output_rep
            ),
            "mean_robust_delta_total": mean(
                robust_delta_total_rep
            ),
        },
        "A": {
            "mean_robust_delta_reasoning": mean(
                robust_delta_reasoning_a
            ),
            "sd_robust_delta_reasoning": sd(
                robust_delta_reasoning_a
            ),
            "min_robust_delta_reasoning": min(
                robust_delta_reasoning_a
            ),
            "max_robust_delta_reasoning": max(
                robust_delta_reasoning_a
            ),
            "cohen_dz": cohens_dz(
                robust_delta_reasoning_a
            ),
            "t": a_test["t"],
            "df": a_test["df"],
            "p_raw": a_test["p"],
            "reasoning_mean": mean(raw_reasoning_a),
            "reasoning_sd": sd(raw_reasoning_a),
            "mean_robust_delta_output": mean(
                robust_delta_output_a
            ),
            "mean_robust_delta_total": mean(
                robust_delta_total_a
            ),
        },
        "rows": row_details,
    }


def global_tost(by_cell):
    differences = []

    for task_id in TASKS:
        for rep in REPS:
            rep_value = metric_value(
                by_cell[(task_id, REPRESENTATION, rep)],
                "reasoning_tokens",
            )

            a_value = metric_value(
                by_cell[(task_id, ARM_A, rep)],
                "reasoning_tokens",
            )

            differences.append(
                a_value - rep_value
            )

    result = tost(
        differences,
        TOST_MARGIN,
    )

    result["endpoint"] = "A_minus_representation"
    result["margin_tokens"] = TOST_MARGIN
    result["pair_count"] = len(differences)

    return result


def build_decision(task_results, tost_result):
    raw_p = {}

    for task_id in TASKS:
        raw_p[task_id] = task_results[
            task_id
        ]["representation"]["p_raw"]

    holm = holm_bonferroni(raw_p)

    rows = []

    universal = True

    for task_id in TASKS:
        result = task_results[task_id]

        rep = result["representation"]
        baseline = result["baseline"]

        mean_pass = rep[
            "mean_robust_delta_reasoning"
        ] < 0

        holm_pass = holm[task_id] < ALPHA

        dz_pass = rep["cohen_dz"] > DZ_THRESHOLD

        variance_pass = (
            rep["reasoning_sd"]
            <= baseline["reasoning_sd"]
        )

        task_pass = (
            mean_pass
            and holm_pass
            and dz_pass
            and variance_pass
        )

        if not task_pass:
            universal = False

        rows.append(
            {
                "task_id": task_id,
                "mean_pass": mean_pass,
                "holm_pass": holm_pass,
                "dz_pass": dz_pass,
                "variance_pass": variance_pass,
                "task_pass": task_pass,
                "mean_robust_delta": rep[
                    "mean_robust_delta_reasoning"
                ],
                "p_raw": rep["p_raw"],
                "p_holm": holm[task_id],
                "cohen_dz": rep["cohen_dz"],
                "reasoning_sd": rep["reasoning_sd"],
                "baseline_reasoning_sd": baseline[
                    "reasoning_sd"
                ],
            }
        )

    equivalence = bool(
        tost_result["equivalent"]
    )

    freeze_pass = (
        universal
        and equivalence
    )

    return {
        "holm_adjusted_p": holm,
        "primary_task_rows": rows,
        "representation_universal_pass": universal,
        "A_vs_representation_equivalence": equivalence,
        "freeze_pass": freeze_pass,
        "decision": (
            "FREEZE_PASS"
            if freeze_pass
            else "FREEZE_FAIL"
        ),
    }


def fmt(value, digits=6):
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"

        if math.isinf(value):
            return "inf" if value > 0 else "-inf"

    return f"{value:.{digits}f}"


def render_markdown(
    metadata,
    collection,
    task_results,
    tost_result,
    decision,
):
    lines = []

    lines.append("# PromptForge V0.7 Confirmatory Analysis")
    lines.append("")
    lines.append(
        f"- Generated UTC: {metadata['generated_utc']}"
    )
    lines.append(
        f"- Source SHA256: `{metadata['source_sha256']}`"
    )
    lines.append(
        f"- Runs: {collection['runs']}"
    )
    lines.append(
        f"- Quality passes: {collection['quality_passes']}"
    )
    lines.append("- Analysis API calls: 0")
    lines.append("")

    lines.append("## Primary Endpoint")
    lines.append("")
    lines.append(
        "| Task | Mean robust Δreasoning | p raw | p Holm | dz | "
        "Rep SD | Baseline SD | Mean<0 | Holm<.05 | dz>.30 | SD criterion | PASS |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|"
    )

    for row in decision["primary_task_rows"]:
        lines.append(
            f"| {row['task_id']} "
            f"| {fmt(row['mean_robust_delta'])} "
            f"| {fmt(row['p_raw'])} "
            f"| {fmt(row['p_holm'])} "
            f"| {fmt(row['cohen_dz'])} "
            f"| {fmt(row['reasoning_sd'])} "
            f"| {fmt(row['baseline_reasoning_sd'])} "
            f"| {row['mean_pass']} "
            f"| {row['holm_pass']} "
            f"| {row['dz_pass']} "
            f"| {row['variance_pass']} "
            f"| {row['task_pass']} |"
        )

    lines.append("")
    lines.append("## Global TOST: A vs Representation")
    lines.append("")
    lines.append(
        f"- N matched pairs: {tost_result['pair_count']}"
    )
    lines.append(
        f"- Mean A - Representation: {fmt(tost_result['mean'])} tokens"
    )
    lines.append(
        f"- 90% CI: [{fmt(tost_result['ci90_low'])}, {fmt(tost_result['ci90_high'])}]"
    )
    lines.append(
        f"- Lower one-sided p: {fmt(tost_result['p_lower'])}"
    )
    lines.append(
        f"- Upper one-sided p: {fmt(tost_result['p_upper'])}"
    )
    lines.append(
        f"- Equivalence: {tost_result['equivalent']}"
    )

    lines.append("")
    lines.append("## Final Decision")
    lines.append("")
    lines.append(
        f"**{decision['decision']}**"
    )
    lines.append("")
    lines.append(
        f"- Universal representation criterion: "
        f"{decision['representation_universal_pass']}"
    )
    lines.append(
        f"- A-vs-representation equivalence: "
        f"{decision['A_vs_representation_equivalence']}"
    )

    lines.append("")
    lines.append(
        "No runs, tasks, arms, thresholds, baseline estimator, "
        "or collection rules were altered during analysis."
    )

    return "\n".join(lines) + "\n"


def main():
    report, source_hash = load_report()

    by_cell = index_runs(report)

    collection = verify_collection(
        report,
        by_cell,
    )

    task_results = {}

    for task_id in TASKS:
        task_results[task_id] = build_task_analysis(
            task_id,
            by_cell,
        )

    tost_result = global_tost(
        by_cell
    )

    decision = build_decision(
        task_results,
        tost_result,
    )

    metadata = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_sha256": source_hash,
        "baseline_estimator": "LOO_TRIMMED_MEAN_20",
        "tasks": TASKS,
        "arms": ARMS,
        "repetitions": 15,
        "expected_runs": EXPECTED_RUNS,
        "holm_alpha": ALPHA,
        "cohen_dz_threshold": DZ_THRESHOLD,
        "tost_alpha": ALPHA,
        "tost_margin_tokens": TOST_MARGIN,
        "analysis_api_calls": 0,
    }

    output = {
        "metadata": metadata,
        "collection": collection,
        "tasks": task_results,
        "global_A_vs_representation_TOST": tost_result,
        "decision": decision,
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    OUTPUT_MD.write_text(
        render_markdown(
            metadata,
            collection,
            task_results,
            tost_result,
            decision,
        ),
        encoding="utf-8",
    )

    print("")
    print("============================================================")
    print("V0.7 CONFIRMATORY ANALYSIS COMPLETE")
    print("============================================================")
    print("")
    print(f"SOURCE_SHA256={source_hash}")
    print(f"RUNS={collection['runs']}")
    print(f"QUALITY_PASSES={collection['quality_passes']}")
    print("")

    print("PRIMARY TASK RESULTS")
    print("")

    for row in decision["primary_task_rows"]:
        print(
            f"{row['task_id']} "
            f"delta={row['mean_robust_delta']:.6f} "
            f"p_raw={row['p_raw']:.8g} "
            f"p_holm={row['p_holm']:.8g} "
            f"dz={row['cohen_dz']:.6f} "
            f"rep_sd={row['reasoning_sd']:.6f} "
            f"baseline_sd={row['baseline_reasoning_sd']:.6f} "
            f"PASS={row['task_pass']}"
        )

    print("")
    print("GLOBAL TOST")
    print("")
    print(
        f"N={tost_result['pair_count']}"
    )
    print(
        f"MEAN_A_MINUS_REP={tost_result['mean']:.6f}"
    )
    print(
        f"CI90_LOW={tost_result['ci90_low']:.6f}"
    )
    print(
        f"CI90_HIGH={tost_result['ci90_high']:.6f}"
    )
    print(
        f"P_LOWER={tost_result['p_lower']:.8g}"
    )
    print(
        f"P_UPPER={tost_result['p_upper']:.8g}"
    )
    print(
        f"EQUIVALENT={tost_result['equivalent']}"
    )

    print("")
    print("============================================================")
    print(
        "REPRESENTATION_UNIVERSAL_PASS="
        + str(
            decision["representation_universal_pass"]
        )
    )
    print(
        "A_VS_REP_EQUIVALENCE="
        + str(
            decision[
                "A_vs_representation_equivalence"
            ]
        )
    )
    print(
        "DECISION="
        + decision["decision"]
    )
    print("============================================================")
    print("")
    print(f"MARKDOWN={OUTPUT_MD}")
    print(f"JSON={OUTPUT_JSON}")
    print("API_CALLS=0")

    return 0


if __name__ == "__main__":
    sys.exit(main())
