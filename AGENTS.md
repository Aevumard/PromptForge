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

1. Read `README.md`.
2. Use `harness.agent.prepare_context()` when you have real task data.
3. Use `harness.agent.prepare()` only for the public fixture tasks.
4. Inspect `harness/runner/transforms.py` when you need to understand or modify transformation semantics.
5. Inspect preregistration, runners, reports, and providers only when the task actually concerns experiments or provider execution.

The core agent-context API is provider-agnostic. It does not require OpenAI, DeepSeek, Gemini, or any other API configuration.

## Agent-facing contract

For arbitrary task context:

```python
from harness.agent import prepare_context

prepared = prepare_context(
    data={
        "case_id": "C-001",
        "customer": "Ana",
        "priority": "high",
        "deadline": "2026-09-20",
        "notes": "noise",
    },
    required=["case_id", "customer", "priority", "deadline"],
)
```

The returned object is the handoff artifact. Inspect:

- `serialized_context`: the compact context to use.
- `selected_arm`: the transformation selected.
- `included` / `excluded`: what was retained or removed.
- `required_values_preserved` and `validation`: semantic safety checks.
- `context_chars_saved` / `context_reduction_ratio`: reduction against the `noop` baseline.
- `candidates`: the alternatives considered.

Do not treat the minimum-size policy as a model-quality guarantee. It is a deterministic selection rule over the currently exposed arms.

## Repository navigation rule

The repository contains both current agent-facing infrastructure and historical experimental infrastructure. Prefer the current facade first.

Do not start by reading large retained JSON reports or provider adapters unless the request requires them. The reports and adapters exist for reproducibility and historical evidence, not because every agent task needs them.

## Direct agent usage

Use the public agent-context bridge when an agent needs a compiled task context rather than raw task data.

```python
from harness.agent_context import compile_task_by_id

compiled = compile_task_by_id("T003", "selection_only")
```

The bridge loads the public task suite, applies a validated V0.6 transform arm, exposes included and excluded context, serializes the result, and validates that required values survive decoding.

This is additive infrastructure. Do not alter the frozen V0.9.3 experimental machinery to implement agent-facing access.

## Preferred agent entry point

For ordinary agent-facing context preparation, prefer the facade before using lower-level modules:

```python
from harness.agent import prepare

prepared = prepare("T003")
```

The default policy is deterministic minimal_serialized_context: it compiles the public arms, validates required values, and selects the smallest serialized context. Use arm_id= only when an explicit representation is required.

Treat the returned candidate audit as evidence of what PromptForge considered. Do not convert the minimal-size policy into a claim about model quality.
