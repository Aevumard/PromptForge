# V27.16 — local Ollama runner

V27.16 turns the local benchmark into a one-command workflow.

## One command

From the repository root:

    python -m benchmarks.local_runner_v27 --replicates 3 --count 1200

The runner:

1. checks that Ollama is reachable
2. lists installed models
3. selects the requested model or the deterministic first model
4. validates the model before any benchmark call
5. starts the repeated-run benchmark
6. prints progress when each replicate starts and finishes
7. writes the existing per-replicate and aggregate artifacts

## Choose a model

    python -m benchmarks.local_runner_v27 --model qwen3:8b --replicates 3 --count 1200

## Preflight only

    python -m benchmarks.local_runner_v27 --dry-run

JSON preflight:

    python -m benchmarks.local_runner_v27 --dry-run --json

## Failure behavior

A missing Ollama server or missing model fails during preflight, before the
benchmark starts. A configured output directory remains protected by the
V27.14 manifest so incompatible experiments cannot be mixed.

The runner does not automatically download models. Model downloads can be
multi-gigabyte operations and should remain an explicit user action.
