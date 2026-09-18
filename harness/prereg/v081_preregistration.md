# PromptForge V0.8.1 — No-Gemini Heterogeneity Experiment

## Identity

Experiment: PROMPTFORGE_V0.8.1_NO_GEMINI

This experiment is a separate follow-up to V0.8.

It does not modify or replace the frozen V0.8 protocol.

## Scope

Tasks:

- T001
- T002
- T003
- T004

Models:

- deepseek
- openai_gpt5_mini

Arms:

-
oop
-
epresentation

Repetitions per task × model × arm: 15.

Total scheduled runs:

2 models × 4 tasks × 2 arms × 15 repetitions = 240

## Purpose

Estimate task × model heterogeneity for the representation transformation after removal of Gemini from the executable comparison because Gemini free-tier quota produced repeated 429 quota failures during V0.8 collection.

This is not a universal freeze test.

No production winner is declared by this experiment.

## Baseline

For every comparison,
oop is the baseline within the same:

- task
- model
- repetition

No baseline is pooled across models.

## Primary descriptive endpoint

For each task × model cell:

delta_reasoning = representation_reasoning_tokens - noop_reasoning_tokens

The cell is reported descriptively as:

- BENEFIT_SIGNAL when mean delta < 0 and the CI upper bound is < 0.
- HARM_SIGNAL when mean delta > 0 and the CI lower bound is > 0.
- UNCERTAIN otherwise.

These labels are descriptive only.

## Secondary metrics

Report:

- input tokens
- reasoning tokens
- output tokens
- total tokens
- latency
- quality pass rate
- model status
- task × model × arm cell counts

## Integrity rules

- Exactly 240 scheduled positions.
- Positions must be 0..239 without gaps.
- No provider retries.
- No silent replacement of failed runs.
- Provider errors remain explicit in the report.
- Deterministic evaluator remains unchanged.
-
eal_executor.py remains unchanged.
- V0.7 frozen files remain unchanged.
- No Gemini adapter is instantiated.
- No Gemini API request is permitted.
- Schedule generation performs zero API calls.
- Dry-run performs zero API calls.
- Analysis performs zero API calls.
- A rate-limit/quota error is fail-closed.

## Transformation identity

Schedule arm
epresentation maps to the existing V0.6 canonical transform:


epresentation_only

The underlying V0.6 transform implementation is not modified.

## Model routing

deepseek → DeepSeekProviderAdapter

openai_gpt5_mini → OpenAIProviderAdapter with:

- model: gpt-5-mini
- max output tokens: 512
- reasoning effort: low

## Explicit exclusions

Gemini is excluded from execution in V0.8.1.

This exclusion is a protocol change relative to V0.8 and therefore V0.8.1 is treated as a separate experiment.

## Freeze rule

No universal policy freeze is defined in V0.8.1.

Results are evidence for subsequent TASK × MODEL × POLICY analysis.
