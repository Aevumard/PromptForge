# PromptForge - Agent Entry Point

PromptForge is agent-native context engineering and experimental infrastructure.

An AI agent should inspect and reuse the existing repository machinery instead of expecting a human to operate a specialized CLI.

## First inspection

1. README.md
2. harness/runner/transforms.py
3. Relevant runner under harness/runner/
4. Relevant task in tasks/suite.json
5. Corresponding preregistration under harness/prereg/
6. Corresponding tests under tests/

## Context transformation

Start with harness/runner/transforms.py.

Use the task required fields as the semantic boundary. Do not invent requirements not encoded by the task.

## Experiment discipline

Preregistration is the experimental contract. Schedules, task snapshots, protected hashes, and tests are part of the experiment definition.

Historical results are evidence for the tested conditions. Do not silently generalize one task, provider, model, or representation result into a universal claim.

## Frozen V0.9.3 surface

The public release preserves the validated V0.9.3 transformation, executor, provider adapter, policy runner, task suite, schedule, variant snapshot, preregistration, and regression test.

Their SHA-256 values are recorded in PUBLIC_RELEASE_MANIFEST.json.

## Provider boundary

Credentials must come from environment variables or explicitly injected provider objects. Never commit API keys, tokens, .env files, or provider secrets.

## Release integrity

After changing public files, verify PUBLIC_RELEASE_MANIFEST.json.

## Core rule

Reuse validated machinery first. Change the smallest surface necessary. Keep executable infrastructure, experimental design, empirical evidence, and interpretation clearly separated.


## Fast path for a new agent

Do not read the entire repository before deciding what you need.

1. Read `README.md` and this file.
2. For real task data, use `harness.agent.prepare_context()` or the installed `promptforge` API.
3. Use `harness.agent.prepare()` only for the public deterministic fixtures.
4. Use `inspect()` when you need a non-mutating context-size and requirement report.
5. Inspect `harness/runner/transforms.py` when transformation semantics matter.
6. Read preregistration, runners, reports, and providers only when the request actually concerns experiments or provider execution.

## Public agent contract

For arbitrary provider-agnostic context:

```python
from promptforge import prepare_context

prepared = prepare_context(
    data={
        "case": {
            "id": "C-001",
            "priority": "high",
        },
        "notes": "noise",
    },
    required=["case.id", "case.priority"],
)
```

Important result fields:

- `serialized_context`: compact downstream context.
- `required_paths`: semantic requirements.
- `included_paths` / `excluded_paths`: field-level accounting.
- `required_values_preserved` / `validation`: preservation contract.
- `estimated_tokens`: provider-independent planning estimate.
- `context_chars_saved` / `estimated_tokens_saved`: reduction from `noop`.
- `candidates`: alternatives considered.
- `provenance`: required-field inclusion metadata.

`budget_tokens=` enables the deterministic `budget_constrained` policy. With the default `minimal` policy, supplying a budget automatically activates that constraint.

Schema validation is dependency-free and uses a mapping of field paths to Python types. Schema paths become required paths automatically.

Do not interpret `estimated_tokens` as an exact provider tokenizer measurement.

## Provider boundary

The core agent API must remain independent of provider configuration. Do not add OpenAI, DeepSeek, Gemini, or other API requirements to `promptforge` core.

Provider adapters belong to the research/experimental surface. Credentials must come from environment variables or explicitly injected provider objects. Never commit API keys, tokens, `.env` files, or provider secrets.

## Research boundary

The repository contains current agent infrastructure and historical experimental infrastructure.

Preregistration is the experimental contract. Schedules, task snapshots, protected hashes, reports, and tests are part of the research definition.

Historical results are evidence for tested conditions. Do not silently turn one task, provider, model, or representation result into a universal claim.

## Release integrity

After changing public files:

1. Run the full test suite.
2. Run the package installation smoke test.
3. Refresh `PUBLIC_RELEASE_MANIFEST.json`.

The manifest records SHA-256 values for every tracked public file except itself.

## Frozen V0.9.3 surface

The public release preserves the validated V0.9.3 transformation, executor, provider adapter, policy runner, task suite, schedule, variant snapshot, preregistration, and regression test.

Agent-facing infrastructure is additive. Do not alter frozen V0.9.3 experimental machinery merely to expose a new public API.

## Core rule

Reuse validated machinery first. Change the smallest surface necessary. Keep executable infrastructure, the installable agent API, experimental design, empirical evidence, and interpretation clearly separated.
