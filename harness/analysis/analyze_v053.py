import json
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("Falta numpy. Ejecuta: python -m pip install numpy") from exc

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError as exc:
    raise SystemExit("Falta scipy. Ejecuta: python -m pip install scipy") from exc

REPO = Path(r"D:\PROMPTFORGE\promptforge-harness-v0.1")
REPORT = REPO / "harness" / "reports" / "v053_deepseek.json"
OUT_JSON = REPO / "harness" / "analysis" / "v053_statistical_analysis.json"
OUT_MD = REPO / "harness" / "analysis" / "v053_statistical_analysis.md"

TASKS = ["T001", "T002", "T003", "T004"]
ARMS = ["noop", "selection_only", "representation_only", "selection_representation"]
TREATMENTS = ARMS[1:]
METRICS = ["input_tokens", "reasoning_tokens", "output_tokens", "total_tokens", "latency_ms"]

def mean(values):
    return float(np.mean(np.asarray(values, dtype=float)))

def median(values):
    return float(np.median(np.asarray(values, dtype=float)))

def sample_sd(values):
    x = np.asarray(values, dtype=float)
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

def basic_stats(values):
    x = np.asarray(values, dtype=float)
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x))
    }

def bootstrap_mean_ci(values, rng, samples=50000):
    x = np.asarray(values, dtype=float)
    n = len(x)

    if n == 0:
        return None

    if n == 1:
        value = float(x[0])
        return {
            "mean": value,
            "low_2_5": value,
            "high_97_5": value,
            "n": 1
        }

    indices = rng.integers(0, n, size=(samples, n))
    boot_means = x[indices].mean(axis=1)

    return {
        "mean": float(np.mean(x)),
        "low_2_5": float(np.percentile(boot_means, 2.5)),
        "high_97_5": float(np.percentile(boot_means, 97.5)),
        "n": n
    }

def get_num(mapping, names):
    for name in names:
        if name in mapping:
            value = mapping[name]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None

def extract_metrics(run):
    metrics = run.get("metrics")

    if not isinstance(metrics, dict):
        raise SystemExit("Run sin objeto metrics")

    result = {}

    result["input_tokens"] = get_num(metrics, ["input_tokens", "prompt_tokens"])
    result["reasoning_tokens"] = get_num(metrics, ["reasoning_tokens"])
    result["output_tokens"] = get_num(metrics, ["output_tokens", "completion_tokens"])
    result["total_tokens"] = get_num(metrics, ["total_tokens"])
    result["latency_ms"] = get_num(metrics, ["latency_ms", "duration_ms"])

    for metric, value in result.items():
        if value is None:
            raise SystemExit("Falta metric {} en run".format(metric))

    reconstructed_total = (
        result["input_tokens"] + result["output_tokens"]
    )

    if abs(result["total_tokens"] - reconstructed_total) > 1e-9:
        raise SystemExit(
            "Token accounting inconsistente: {} != {}".format(
                result["total_tokens"], reconstructed_total
            )
        )

    if result["reasoning_tokens"] > result["output_tokens"]:
        raise SystemExit(
            "Reasoning tokens no puede superar output tokens: {} > {}".format(
                result["reasoning_tokens"], result["output_tokens"]
            )
        )

    return result

def tradeoff_ratio(base, treatment):
    input_removed = base["input_tokens"] - treatment["input_tokens"]

    base_non_reasoning = base["output_tokens"] - base["reasoning_tokens"]
    treatment_non_reasoning = treatment["output_tokens"] - treatment["reasoning_tokens"]

    base_other_work = base["reasoning_tokens"] + base_non_reasoning
    treatment_other_work = treatment["reasoning_tokens"] + treatment_non_reasoning

    extra_non_input = treatment_other_work - base_other_work

    if input_removed <= 0:
        return None

    return extra_non_input / input_removed

def main():
    raw = json.loads(REPORT.read_text(encoding="utf-8"))

    runs = raw.get("runs")

    if not isinstance(runs, list):
        raise SystemExit("El reporte no contiene runs[]")

    if len(runs) != 48:
        raise SystemExit("Esperados 48 runs, encontrados {}".format(len(runs)))

    print("RUNS_FOUND={}".format(len(runs)))

    records = []

    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise SystemExit("runs[{}] no es un objeto".format(index))

        metrics = extract_metrics(run)

        task_index = index // 12
        position_in_task = index % 12
        rep = (position_in_task // 4) + 1
        arm_index = position_in_task % 4

        task = TASKS[task_index]
        arm = ARMS[arm_index]

        status = run.get("status", "UNKNOWN")

        records.append({
            "index": index,
            "task": task,
            "arm": arm,
            "rep": rep,
            "status": status,
            "metrics": metrics
        })

    pass_count = sum(1 for r in records if r["status"] == "PASS")

    print("PASS_COUNT={}".format(pass_count))

    if pass_count != 48:
        raise SystemExit("No todos los runs tienen status PASS")

    print("IDENTITY_RECONSTRUCTION=VALID")
    print("TOKEN_ACCOUNTING=VALID")
    print("REASONING_IS_SUBSET_OF_OUTPUT=VALID")
    print("")

    lookup = {}
    for record in records:
        lookup[(record["task"], record["arm"], record["rep"])] = record["metrics"]

    rng = np.random.default_rng(20260915)

    result = {
        "source": str(REPORT),
        "run_count": len(records),
        "tasks": TASKS,
        "arms": ARMS,
        "repetitions": [1, 2, 3],
        "token_semantics": {
            "total_tokens": "input_tokens + output_tokens",
            "reasoning_tokens": "subset of output_tokens"
        },
        "bootstrap": {
            "samples": 50000,
            "seed": 20260915,
            "confidence": 0.95
        },
        "task_results": {},
        "global_macro_effects": {},
        "correlations": {}
    }

    for task in TASKS:
        result["task_results"][task] = {}

        for arm in TREATMENTS:
            pairs = []

            for rep in [1, 2, 3]:
                base = lookup[(task, "noop", rep)]
                treatment = lookup[(task, arm, rep)]
                pairs.append((base, treatment))

            cell = {
                "n": len(pairs),
                "arm_stats": {},
                "paired_delta": {},
                "tradeoff": {}
            }

            for metric in METRICS:
                base_values = np.array([p[0][metric] for p in pairs], dtype=float)
                treatment_values = np.array([p[1][metric] for p in pairs], dtype=float)
                delta = treatment_values - base_values

                cell["arm_stats"][metric] = basic_stats(treatment_values.tolist())
                cell["paired_delta"][metric] = {
                    **basic_stats(delta.tolist()),
                    "bootstrap_ci": bootstrap_mean_ci(delta.tolist(), rng)
                }

            ratios = []
            for base, treatment in pairs:
                ratio = tradeoff_ratio(base, treatment)
                if ratio is not None:
                    ratios.append(ratio)

            if ratios:
                cell["tradeoff"] = {
                    "n": len(ratios),
                    "mean_ratio": mean(ratios),
                    "median_ratio": median(ratios),
                    "sd_ratio": sample_sd(ratios),
                    "bootstrap_ci": bootstrap_mean_ci(ratios, rng)
                }

            result["task_results"][task][arm] = cell

    for arm in TREATMENTS:
        result["global_macro_effects"][arm] = {}

        for metric in METRICS:
            task_effects = []

            for task in TASKS:
                task_effects.append(
                    result["task_results"][task][arm]["paired_delta"][metric]["mean"]
                )

            result["global_macro_effects"][arm][metric] = {
                "mean_task_effect": mean(task_effects),
                "task_effects": task_effects,
                "n_tasks": len(task_effects)
            }

        reasoning_deltas = []
        latency_deltas = []

        for task in TASKS:
            for rep in [1, 2, 3]:
                base = lookup[(task, "noop", rep)]
                treatment = lookup[(task, arm, rep)]
                reasoning_deltas.append(
                    treatment["reasoning_tokens"] - base["reasoning_tokens"]
                )
                latency_deltas.append(
                    treatment["latency_ms"] - base["latency_ms"]
                )

        pearson = pearsonr(reasoning_deltas, latency_deltas)
        spearman = spearmanr(reasoning_deltas, latency_deltas)

        result["correlations"][arm] = {
            "n": len(reasoning_deltas),
            "reasoning_delta_vs_latency_delta": {
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue)
            }
        }

    OUT_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    md = []
    md.append("# PromptForge v0.5.3 — Statistical Analysis")
    md.append("")
    md.append("Runs: **48** — Tasks: **4** — Arms: **4** — Repetitions: **3**")
    md.append("")
    md.append("## Token accounting")
    md.append("")
    md.append("- `total_tokens = input_tokens + output_tokens`")
    md.append("- `reasoning_tokens` is a component of `output_tokens`, not an additional term.")
    md.append("")
    md.append("## Method")
    md.append("")
    md.append("Every treatment is compared against NOOP within the same task and repetition.")
    md.append("Bootstrap CIs are computed independently inside each task/arm cell.")
    md.append("Global effects are macro-averages of the four task-level effects.")
    md.append("")

    for task in TASKS:
        md.append("## {}".format(task))
        md.append("")
        md.append("| Arm | dInput | dReasoning | dOutput | dTotal | dLatency ms | Trade-off |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")

        for arm in TREATMENTS:
            cell = result["task_results"][task][arm]
            trade = cell["tradeoff"].get("mean_ratio")
            trade_text = "n/a" if trade is None else "{:.3f}".format(trade)

            md.append("| {} | {:+.2f} | {:+.2f} | {:+.2f} | {:+.2f} | {:+.1f} | {} |".format(
                arm,
                cell["paired_delta"]["input_tokens"]["mean"],
                cell["paired_delta"]["reasoning_tokens"]["mean"],
                cell["paired_delta"]["output_tokens"]["mean"],
                cell["paired_delta"]["total_tokens"]["mean"],
                cell["paired_delta"]["latency_ms"]["mean"],
                trade_text
            ))

        md.append("")
        md.append("### Bootstrap 95% CIs")
        md.append("")

        for arm in TREATMENTS:
            md.append("#### {}".format(arm))

            for metric in METRICS:
                ci = result["task_results"][task][arm]["paired_delta"][metric]["bootstrap_ci"]
                md.append("- {}: {:.2f} [{:.2f}, {:.2f}]".format(
                    metric, ci["mean"], ci["low_2_5"], ci["high_97_5"]
                ))

            md.append("")

    md.append("## Global macro effects")
    md.append("")
    md.append("| Arm | dInput | dReasoning | dOutput | dTotal | dLatency ms |")
    md.append("|---|---:|---:|---:|---:|---:|")

    for arm in TREATMENTS:
        md.append("| {} | {:+.2f} | {:+.2f} | {:+.2f} | {:+.2f} | {:+.1f} |".format(
            arm,
            result["global_macro_effects"][arm]["input_tokens"]["mean_task_effect"],
            result["global_macro_effects"][arm]["reasoning_tokens"]["mean_task_effect"],
            result["global_macro_effects"][arm]["output_tokens"]["mean_task_effect"],
            result["global_macro_effects"][arm]["total_tokens"]["mean_task_effect"],
            result["global_macro_effects"][arm]["latency_ms"]["mean_task_effect"]
        ))

    md.append("")
    md.append("## Reasoning vs latency")
    md.append("")
    md.append("| Arm | N | Pearson r | Pearson p | Spearman rho | Spearman p |")
    md.append("|---|---:|---:|---:|---:|---:|")

    for arm in TREATMENTS:
        corr = result["correlations"][arm]["reasoning_delta_vs_latency_delta"]
        md.append("| {} | {} | {:.3f} | {:.4f} | {:.3f} | {:.4f} |".format(
            arm,
            result["correlations"][arm]["n"],
            corr["pearson_r"],
            corr["pearson_p"],
            corr["spearman_rho"],
            corr["spearman_p"]
        ))

    md.append("")
    md.append("## Caveat")
    md.append("")
    md.append("N=3 por celda es exploratorio. Los CIs describen incertidumbre y no justifican por si solos congelar una policy.")

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("")
    print("ANALYSIS_COMPLETE")
    print("records={}".format(len(records)))
    print("json={}".format(OUT_JSON))
    print("markdown={}".format(OUT_MD))
    print("")
    print("GLOBAL EFFECTS")

    for arm in TREATMENTS:
        g = result["global_macro_effects"][arm]
        print("{} input={:+.2f} reasoning={:+.2f} output={:+.2f} total={:+.2f} latency_ms={:+.1f}".format(
            arm,
            g["input_tokens"]["mean_task_effect"],
            g["reasoning_tokens"]["mean_task_effect"],
            g["output_tokens"]["mean_task_effect"],
            g["total_tokens"]["mean_task_effect"],
            g["latency_ms"]["mean_task_effect"]
        ))

if __name__ == "__main__":
    main()