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


## Agent Context API

PromptForge also exposes a direct agent-facing context compilation API.

```python
from harness.agent_context import compile_task_by_id

compiled = compile_task_by_id("T003", "selection_only")
```

The returned artifact records the selected and excluded fields, the transform sequence, the compiled representation, serialized context, context size, required-value preservation, and validation metadata.

This API is additive to the public harness and does not modify the frozen V0.9.3 experimental surface.
