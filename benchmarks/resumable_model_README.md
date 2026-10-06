# V27.6 — resumable model benchmark

V27.6 adds an append-only checkpoint boundary to the V27.4/V27.5 model loop.

Every ticket produces one checkpoint record, including failures. On restart:

- successful tickets are skipped;
- failed tickets are retried by default;
- successful work is never discarded;
- the checkpoint can be replayed or inspected as JSONL.

Example:

    from benchmarks.providers.openai_compatible import OpenAICompatibleAgentAdapter, OpenAICompatibleConfig
    from benchmarks.resumable_model_loop_v27 import run_resumable_model_loop

    adapter = OpenAICompatibleAgentAdapter(OpenAICompatibleConfig(
        endpoint_url="http://localhost:8000/v1/chat/completions",
        model="my-model",
    ))

    report = run_resumable_model_loop(
        cases,
        adapter,
        checkpoint_path="artifacts/tickets-v27.6.checkpoint.jsonl",
    )

Use fsync_each_record=True for crash-safe local runs. Set it to False when the
storage layer already guarantees durable writes and maximum throughput matters.

A failed provider response does not erase earlier successful tickets. The same
checkpoint file is intentionally append-only; the latest record for a ticket is
the active state.
