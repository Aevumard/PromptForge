# PromptForge - Agent Entry Point

PromptForge has two deliberately separated surfaces:

- **Agent core:** provider-agnostic context preparation, validation, budgets, inspection, and deterministic policies.
- **Research harness:** historical experiments, preregistration, schedules, reports, runners, and provider adapters.

An AI agent should use the smallest validated surface that solves the task. Do not begin by reading the entire repository.

## First inspection

1. Read `README.md`.
2. Read this file.
3. For real context work, use the installable `promptforge` API.
4. For repository fixtures, use `harness.agent.prepare()`.
5. Inspect `harness/runner/transforms.py` only when transformation semantics matter.
6. Inspect historical experiments, reports, and providers only when the request actually concerns research or provider execution.

## Public core contract

The installable public API is:

```python
from promptforge import prepare_context, inspect

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

It is provider-agnostic. It requires no API key, network access, or model configuration.

The handoff artifact exposes:

- `context`
- `serialized_context`
- `selected_arm`
- `required_paths`
- `included_paths` / `excluded_paths`
- `required_values_preserved`
- `validation`
- `estimated_tokens`
- `context_chars_saved` / `estimated_tokens_saved`
- `candidates`
- `provenance`

`estimated_tokens` is a dependency-free planning heuristic based on UTF-8 byte length divided by four and rounded up. It is not an exact provider tokenizer measurement.

## Policies

The default policy is `minimal`, implemented as deterministic `minimal_serialized_context`.

Available public policies:

- `minimal`
- `budget_constrained`

Supplying `budget_tokens=` with the default `minimal` policy automatically activates `budget_constrained`.

The budget policy selects the smallest serialized candidate that fits the requested estimated-token budget. If none fits, fail explicitly rather than silently violating the budget.

Do not convert this mechanical context-size policy into a claim about model quality.

## Nested requirements

Required field paths use dot notation:

```python
required=["user.id", "task.priority"]
```

The core selects only the required branches for selection-based candidates and validates that required values survive the transformation.

Do not silently invent required fields. Treat the caller's required paths as the semantic boundary.

## Lightweight schema validation

Schema validation is dependency-free:

```python
schema={
    "user.id": str,
    "score": float,
}
```

Schema paths automatically become required paths. Validation is applied to the semantic decoded context, not only to an internal structural representation.

## Inspection

Use `inspect()` for non-mutating planning:

```python
report = inspect(
    context,
    required=["task.id", "task.priority"],
)
```

This reports field counts, required coverage, serialized size, bytes, and estimated tokens.

## Repository fixture API

`harness.agent.prepare()` is a repository fixture surface for deterministic regression tasks such as `T003`.

It depends on the checked-out repository task suite and is intentionally separate from the installable core API.

Use it for reproduction, not as the generic public integration interface.

## Provider boundary

The installable core must remain independent of OpenAI, DeepSeek, Gemini, or any other provider.

Provider adapters belong to the research/experimental surface. Credentials must come from environment variables or explicitly injected provider objects.

Never commit API keys, tokens, `.env` files, or provider secrets.

## Research boundary

Preregistration is the experimental contract. Schedules, task snapshots, protected hashes, reports, runners, and tests are part of the research definition.

Historical results are evidence for the tested conditions. Do not silently generalize one task, provider, model, representation, or experiment into a universal claim.

## Frozen V0.9.3 surface

The public release preserves the validated V0.9.3 transformation, executor, provider adapter, policy runner, task suite, schedule, variant snapshot, preregistration, and regression test.

Agent-facing infrastructure is additive. Do not alter frozen V0.9.3 experimental machinery merely to expose or polish the current public API.

## Release integrity

After changing public files:

1. Run the full test suite.
2. Verify the installable package smoke test.
3. Refresh `PUBLIC_RELEASE_MANIFEST.json`.
4. Confirm CI is green.

The manifest records SHA-256 values for every tracked public file except itself.

## Core rule

Reuse validated machinery first. Change the smallest surface necessary. Keep the installable agent API, executable infrastructure, experimental design, empirical evidence, and interpretation clearly separated.
