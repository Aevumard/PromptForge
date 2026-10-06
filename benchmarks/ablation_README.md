# V27.11 — controlled PromptForge ablation + operational telemetry

V27.10 turns the 1,200-ticket benchmark into a controlled layer-ablation
experiment.

The default suite runs the same ticket cases through three configurations:

1. `baseline_budget_only` — PromptForge context budgeting, without the
   epistemic summary and without the action guard.
2. `epistemic_budget` — adds the PromptForge epistemic summary, guard still
   disabled.
3. `epistemic_budget_guarded` — keeps the epistemic summary and enables the
   PromptForge action guard.

Each configuration has its own resumable checkpoint.

## Live run

Set the same OpenAI-compatible endpoint variables used by V27.8/V27.9, then:

    python -m benchmarks.ablation_v27 --count 1200

Outputs:

- `ablation_report.json`
- `ablation_report.md`
- `ablation_checkpoints/*.checkpoint.jsonl`

The checkpoint directory allows an interrupted condition to resume without
repeating successful tickets. Provider telemetry includes observed provider latency,
retry attempts, and token usage when the endpoint reports usage metadata.

## What this isolates

This experiment separates three effects that were previously mixed together:

- context budgeting alone,
- epistemic context control,
- action safety enforcement.

The report keeps raw model quality and guarded safety metrics separate. It also records per-ticket elapsed time and reports latency p50/p95 so quality gains can be evaluated against operational cost.

This does not establish causal effect in a strict statistical sense. The model
may still be nondeterministic, and provider-side behavior can vary between calls.
The result is a controlled engineering ablation over a deterministic benchmark. Latency is wall-clock benchmark telemetry, not provider billing or token-usage accounting.
