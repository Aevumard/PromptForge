# PromptForge V0.8.2 - Hotspot Replication

Experiment:
`PROMPTFORGE_V0.8.2_HOTSPOT_REPLICATION`

Purpose:

Replicate the strongest observed V0.8.1 hotspot before local parameter search.

Frozen scope:

- Task: T003
- Task family: structured_analysis
- Model: deepseek
- Arms: noop, representation
- Repetitions: 30
- Scheduled runs: 60

The representation arm is the existing V0.6 representation path.

Primary endpoint:

Paired reasoning-token delta:

representation_reasoning - noop_reasoning

The comparison is within the same task, model and repetition.

Only pairs where both arms have quality_pass=true are confirmatory-valid.

Secondary endpoints:

- input_tokens
- output_tokens
- total_tokens
- latency
- quality validity

Integrity:

- real_executor.py remains unchanged.
- evaluator remains unchanged.
- no adaptive retries.
- no failed-run replacement.
- no tuning during this replication.
- V0.8.1 remains historical evidence.
- V0.8.2 is a separate report.

This experiment does not freeze a production policy.

If the hotspot reproduces, the next experiment is a local parameter sweep targeting lower input overhead while preserving quality and the favorable execution signal.

If it does not reproduce, investigate instability before productization.

Schedule:

Odd repetitions:
noop -> representation

Even repetitions:
representation -> noop