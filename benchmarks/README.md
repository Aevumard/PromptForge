# V27.2 — 1,200-ticket benchmark

This benchmark turns the V27 control layer into a reproducible support-ticket evaluation surface.

The default suite contains exactly **1,200 synthetic tickets**, balanced across:

- payments
- access
- technical errors
- general queries

Every case has deterministic gold labels for category, SLA, priority, safe action, human-review requirement, contradiction, redundancy, irrelevant content, and historical context.

The benchmark deliberately keeps gold labels out of `TicketCase.model_input()`.

## Run

```bash
python benchmarks/tickets_v27.py
python -m unittest tests/test_tickets_v27.py
```

The built-in naive baseline deliberately ignores contradiction semantics. Its purpose is to establish a safety/error floor before adding a real model adapter.

## Model-in-the-loop next step

An external runner can consume `TicketCase.model_input()`, return `Prediction` records, and pass them into `evaluate_predictions()`.

This keeps dataset generation, scoring, and PromptForge's deterministic control surface separate from the provider/model integration.
