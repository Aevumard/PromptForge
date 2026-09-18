# PromptForge V0.8.4 — Zoom Preregistration

## Experiment

V0.8.4 narrows the investigation to T003 after the V0.8.3 parametric sweep.

Conditions:

1. noop_compact
2. pairs_compact
3. pairs_bare_compact
4. flat_compact

## Task

- Task: T003
- Provider family: DeepSeek
- Repetitions: 30
- Conditions: 4
- Total runs: 120
- Paired by repetition

## Primary endpoint

	otal_tokens

## Secondary endpoints

- input_tokens
- reasoning_tokens
- output_tokens
- latency_ms
- quality

## Purpose

Test whether the execution-regime effect observed under structured representation can be retained while reducing deterministic serialization overhead.

No universal winner is assumed.

No production policy is frozen from this experiment alone.

## Schedule

harness/runner/v084_zoom_schedule.json

SHA256:

$ScheduleHash

## Frozen invariants

- T003 only
- same provider/model
- same task fixture
- same executor
- 30 repetitions
- 4 conditions
- paired analysis by repetition
- no adaptive arm removal
- no adaptive repetition increase
- no threshold changes
- no silent replacement of failed observations
- no post-hoc primary endpoint substitution

## Interpretation

A reduction in reasoning tokens is not by itself an economic improvement.

A representation is not called an optimization unless the total execution cost improves without violating quality or experimental integrity.

## Schedule Integrity Binding

Schedule SHA256:

`4D6B9B81CB2C2B10126E549469616539342CC7FCC3ADF40C83158E4989EDB23F`