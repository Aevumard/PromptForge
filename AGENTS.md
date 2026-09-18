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
