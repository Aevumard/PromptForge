# V27.13 — paired bootstrap statistics

V27.13 adds deterministic paired bootstrap confidence intervals to the raw-vs-guarded benchmark.

It resamples the same ticket IDs jointly, preserving the pairing between the raw and guarded decisions.

Metrics:
- action accuracy
- unsafe action rate
- human recall
- contradiction recall

Default: 2,000 resamples and 95% confidence.

Run:

    python -m benchmarks.analyze_bootstrap_v27 model_report.json --count 1200

Outputs:
- `bootstrap_analysis.json`
- `bootstrap_analysis.md`

Interpretation:
- the delta is guarded minus raw;
- for unsafe action rate, a negative delta favors the guard;
- “CI excludes zero” means the observed paired difference is separated from zero under the bootstrap distribution;
- this is uncertainty quantification, not a causal claim and not a substitute for repeated independent model runs.
