# PromptForge

Agent-native context engineering and experimental harness.

PromptForge is a Python layer for transforming, compiling, validating, and experimentally evaluating the context sent to language models.

## What PromptForge does

- Select relevant context for a task.
- Compile alternative structural representations.
- Construct prompts from compiled context.
- Normalize provider execution results.
- Run controlled and reproducible experiments.

## Agent-native workflow

An agent can inspect the task and context, select relevant fields, compile a representation, validate invariants, and execute or evaluate the resulting prompt.

Primary transformation surface:
harness/runner/transforms.py

Task definitions:
tasks/suite.json

Provider adapters:
harness/providers/

Experimental contracts:
harness/prereg/

Retained evidence:
harness/reports/

Regression tests:
tests/

Read AGENTS.md before making experimental or architectural changes.

## V0.9.3 public evidence

The public release retains the V0.9.3 size-matched control: 4 context variants, 4 conditions, 15 repetitions per variant/condition, and 240 scheduled runs.

The primary comparison is pairs_compact versus noop_padded, where the control is padded to the same serialized context size.

V0.9.3 does not establish a universal economic advantage for one representation. Its purpose is to separate structural representation from serialized context size under the tested conditions.

Relevant files:
harness/prereg/v093_size_matched_control.md
harness/runner/v093_size_matched_control.py
harness/reports/v093_size_matched_control_deepseek.json

## Quick validation

python -m compileall -q harness tests

python -m unittest discover -s tests -p test_*.py

## Scientific boundary

PromptForge makes context transformations explicit, testable, reproducible, and comparable. Historical results are evidence about the tested conditions, not universal guarantees across every task, model, provider, or prompting strategy.


## 30-second agent path

When an agent enters the repository, the intended order is:

1. Read this README and `AGENTS.md`.
2. Use `harness.agent.prepare_context()` for real task data.
3. Inspect `harness/runner/transforms.py` only when the transformation semantics matter.
4. Inspect historical runners, reports, and providers only when the request is experimental or provider-specific.

The primary agent-facing API is provider-agnostic and does not require an API key.

## Direct Agent Preparation

For repository fixtures, use:

```python
from harness.agent import prepare

prepared = prepare("T003")
```

For real work, pass the task's actual context and required fields:

```python
from harness.agent import prepare_context

prepared = prepare_context(
    data={
        "entity": "A-17",
        "score": 0.87,
        "status": "stable",
        "trace": "noise",
        "commentary": "noise",
    },
    required=["entity", "score", "status"],
)
```

The default policy is deterministic `minimal_serialized_context`: PromptForge evaluates the public deterministic transformation arms, validates required-value preservation, and selects the smallest serialized context. The returned artifact includes the selected arm, excluded fields, serialized context, candidate audit, and context-size reduction relative to the `noop` baseline.

This policy is a mechanical context-size policy. It does not claim that the smallest context is universally better for model quality.

## Public-fixture efficiency

Across the four public fixture tasks, the `selection_only` result reduces the serialized context from 400 characters of `noop` input to 253 characters after required-field selection: 147 characters, or 36.75%, fewer.

For T003 specifically, the reduction is 85 → 48 characters (43.5%).

These are character counts for the small public fixtures, not token counts and not a general performance guarantee. They demonstrate the current context-compaction behavior; model-quality effects require separate evaluation.

## Architecture map

| Area | Purpose | Start here |
| --- | --- | --- |
| `harness/agent.py` | Stable agent-facing facade | Real agent use |
| `harness/agent_context.py` | Task loading, compilation, validation | Context contract |
| `harness/runner/transforms.py` | Core transformation machinery | Semantics |
| `tasks/suite.json` | Public deterministic fixtures | Reproduction |
| `harness/prereg/` | Experimental contracts | Research design |
| `harness/reports/` | Retained empirical evidence | Results |
| `harness/runner/` | Historical experiment execution | Reproduction |
| `harness/providers/` | Provider adapters for experiments | Provider-specific work |
| `tests/` | Regression and contract tests | Verification |

Provider adapters are experimental infrastructure. They are not required by the core agent-context API.

## What PromptForge is not

PromptForge is not a provider-specific SDK, a universal prompt optimizer, or a claim that one representation is always superior. It is a context-engineering layer that makes selection, transformation, validation, and experimental comparison explicit and reproducible.

## Agent Context API

PromptForge also exposes a direct agent-facing context compilation API.

```python
from harness.agent_context import compile_task_by_id

compiled = compile_task_by_id("T003", "selection_only")
```

The returned artifact records the selected and excluded fields, the transform sequence, the compiled representation, serialized context, context size, required-value preservation, and validation metadata.

This API is additive to the public harness and does not modify the frozen V0.9.3 experimental surface.

## Direct Agent Preparation

The simplest public agent-facing entry point is:

```python
from harness.agent import prepare

prepared = prepare("T003")
```

With the default minimal policy, PromptForge evaluates the public deterministic arms, preserves required values, and selects the candidate with the smallest serialized context. The result records the selected arm, excluded fields, serialized context, context size, validation, and candidate audit data.

This policy is a mechanical context-size policy. It does not claim that the smallest context is universally better for model quality.
