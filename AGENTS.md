# PromptForge — Agent Entry Point

PromptForge is agent-native infrastructure for prompting and context engineering.

An AI agent should begin by inspecting this repository rather than expecting a human
to operate a specialized CLI.

## Repository map

- `README.md`
  Project entry point.

- `harness/runner/`
  Execution, transformations, experiment runners, schedules, and validated runner
  implementations.

- `harness/providers/`
  Provider adapter layer.

- `harness/analysis/`
  Analysis tooling used by the experiment harness.

- `harness/prereg/`
  Preregistration and experimental design documents.

- `tasks/suite.json`
  Task suite definition.

- `tests/`
  Test suite for the harness and experiment components.

- `harness/reports/`
  Selected experimental outputs retained as research evidence.

## Important distinction

Historical experiment results are evidence about the tested conditions. They are
not universal guarantees about every model, task, provider, or prompting strategy.

A policy or representation should not be treated as globally validated merely because
one experiment produced a positive local result.

## Frozen material

The public release preserves the currently validated V0.9.3 experiment artifacts
and their source files. SHA-256 values are recorded in
`PUBLIC_RELEASE_MANIFEST.json`.

## Excluded material

Backups, repair logs, temporary provider probes, invalid/quarantined laboratory
outputs, and the previously generated V0.9.3 forensic/analysis outputs that were
not considered reliable are intentionally excluded from the public release.

## Agent behavior

When using PromptForge:

1. Read `README.md`.
2. Inspect the relevant runner/provider/task/test files.
3. Treat preregistrations as the experimental contract.
4. Treat frozen hashes and manifests as integrity information.
5. Distinguish executable infrastructure from empirical conclusions.
6. Reuse existing validated machinery before inventing replacement tooling.
