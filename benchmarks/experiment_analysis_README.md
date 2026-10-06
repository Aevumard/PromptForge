# V27.11 — sliced experiment analysis + operational telemetry

The model-loop report contains global raw-vs-guarded metrics. V27.9 adds a
deterministic analysis layer that slices those same results by:

- category
- SLA
- contradiction presence
- human-review requirement
- redundancy
- irrelevant content
- historical context

The analysis reports classification quality separately from action safety, and now includes total elapsed time plus latency p50/p95 for covered model calls.

Run:

    python -m benchmarks.analyze_model_run_v27 model_report.json \
      --output-json experiment_analysis.json \
      --output-markdown experiment_analysis.md

The analysis reuses the same deterministic 1,200-ticket suite seed so the
gold labels remain aligned with the live run.

Important interpretation:

- delta_* values describe guarded minus raw performance.
- unsafe_rate_reduction describes how much PromptForge reduced unsafe actions.
- guard_action_change_rate measures how often the guard altered the model's
  proposed action.
- A safer action does not imply better category/SLA/priority classification.
- latency p50/p95 describes observed wall-clock run time per covered ticket; it is not token billing.

This keeps safety gains and predictive gains analytically separate.
