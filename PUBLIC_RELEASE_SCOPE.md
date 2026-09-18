# PromptForge — Public Release Scope

## Source

$SourceRoot

## Curated destination

$PublicRoot

## Source revision used

- HEAD: $SourceHead
- Branch: $SourceBranch
- Commit: $SourceMessage

## Included

- Core execution infrastructure
- Prompt/context transformation machinery
- Provider adapters
- Experiment runners and schedules
- Analysis source code
- Preregistration documents
- Task suite
- Test suite
- Selected validated research reports
- AGENTS.md
- Machine-readable SHA-256 manifest

## Excluded

- ackups/
- harness/backups/
- repair/debug .log files
- temporary provider bridge probes
- invalid/quarantined/rate-limit-aborted reports
- 093_deep_*
- 093_scientific_analysis.*

## Important scientific boundary

The inclusion of an experimental report means the artifact is retained as evidence
and provenance. It does not mean the measured prompting/policy effect is universal,
economically optimal, or frozen for every model or task.

## Publication state

This directory is a curated PUBLIC RELEASE STAGING TREE.

It has NOT been pushed to GitHub by this script.
