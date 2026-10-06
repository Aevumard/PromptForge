# V27.14 — repeated-run variability

V27.13 measures uncertainty from resampling the same paired tickets. V27.14
adds a different axis: repeated executions of the same model/provider condition.

The runner keeps the ticket suite, PromptForge budget, epistemic setting and
guard setting fixed while giving every replicate its own resumable checkpoint.

## What it measures

Per replicate:

- existing raw and guarded benchmark metrics
- end-to-end and provider latency
- provider attempts and token usage
- the existing paired bootstrap analysis

Across replicates:

- mean, sample standard deviation, minimum and maximum for key metrics
- pairwise raw and guarded action disagreement
- pairwise raw and guarded full-decision disagreement
- provider usage coverage

The between-replicate standard deviation estimates variability across repeated
model/provider executions. It is not the same quantity as the paired bootstrap
interval, which quantifies ticket-sampling uncertainty inside one run.

## Live run

Configure the same environment variables used by the OpenAI-compatible adapter:

    set PROMPTFORGE_MODEL_URL=http://localhost:8000/v1/chat/completions
    set PROMPTFORGE_MODEL_NAME=TU_MODELO
    set PROMPTFORGE_MODEL_API_KEY=TU_KEY

Then run three replicas:

    python -m benchmarks.replicates_v27 --replicates 3 --count 1200

Artifacts:

- replicate_analysis.json
- replicate_analysis.md
- replicate_runs/replicate-001/
- replicate_runs/replicate-002/
- replicate_runs/replicate-003/

Each replicate directory contains its own checkpoint, raw predictions,
model report, experiment analysis and bootstrap report.

Re-running the same command resumes successful tickets from each replicate's
checkpoint rather than issuing them again.

The output directory is manifest-bound to the ticket suite, condition and
provider metadata. Reusing it with different settings fails rather than
mixing incompatible experiments.

## Interpretation

Do not collapse bootstrap uncertainty and repeated-run variability into one
number. A moving metric across replicas indicates model/provider variability;
a wide bootstrap interval indicates ticket-sample uncertainty. Neither
measurement by itself is a causal proof.

The runner stores a SHA-256 fingerprint of the system prompt rather than the
prompt text and filters secret provider metadata before persistence.
