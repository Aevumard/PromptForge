# PromptForge - Public Release Scope

## Publication state

Repository: Aevumard/PromptForge
Default branch: main
Visibility: public
Release type: curated public GitHub release

## Research provenance

The public release was curated from a private laboratory tree.

Source branch: experiment/v0.9.3-size-matched-control
Source commit: 433e6d3f0d9b599e604bea74efb1f52082db2841
Source commit message: fix(experiment): make v0.9.3 padding size matched per variant

The private local filesystem path is intentionally not published here.

## Included

Execution infrastructure, context transformation machinery, provider adapters, experiment runners, schedules, analysis source, preregistration documents, task suite, tests, selected retained research reports, agent documentation, and CI.

## Excluded

Backups, repair logs, temporary provider probes, invalid or quarantined reports, rate-limit-aborted outputs, and untrusted V0.9.3 forensic artifacts.

## Scientific boundary

Retained experiments are evidence for the tested conditions. They are not universal guarantees across all tasks, models, providers, prompts, or representations.

V0.9.3 is specifically a size-matched control study intended to isolate structural representation from serialized context size.

## Agent-facing public surface

The public release exposes an installable, provider-agnostic agent core:

- `promptforge` provides the end-user import surface.
- `prepare_context(data, required)` supports arbitrary task context.
- Nested mapping paths are supported for required fields.
- `budget_tokens=` enables deterministic budget-constrained preparation.
- Lightweight field-type schema validation is dependency-free.
- `inspect()` provides non-mutating context planning metadata.
- Provider credentials and network access are not required for the core API.
- Provider adapters remain available for historical and experimental execution.

The installable package is additive to the frozen V0.9.3 experimental machinery.

## Integrity boundary

PUBLIC_RELEASE_MANIFEST.json records SHA-256 values for every tracked public file except the manifest itself.
