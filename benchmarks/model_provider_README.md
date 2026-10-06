# V27.5 — OpenAI-compatible provider adapter

V27.5 connects the provider-neutral V27.4 model loop to any local or hosted
endpoint that accepts the OpenAI-style Chat Completions request shape.

No SDK or provider package is required. The benchmark remains deterministic and
provider-neutral; this adapter is an optional live-model boundary.

## Environment

Required:

- PROMPTFORGE_MODEL_URL
- PROMPTFORGE_MODEL_NAME

Optional:

- PROMPTFORGE_MODEL_API_KEY
- PROMPTFORGE_MODEL_TIMEOUT (default: 60)
- PROMPTFORGE_MODEL_MAX_ATTEMPTS (default: 3)
- PROMPTFORGE_MODEL_RETRY_BACKOFF (default: 1 second)
- PROMPTFORGE_MODEL_JSON_MODE (default: 0)
- PROMPTFORGE_MODEL_SYSTEM_PROMPT

## Run 1,200 tickets

From the repository root:

    python -m benchmarks.providers.openai_compatible \
      --output predictions.jsonl \
      --report model_report.json \
      --checkpoint model_run.checkpoint.jsonl

The command sends only the PromptForge-prepared model packet. It writes:

- predictions.jsonl: successfully parsed raw model predictions
- model_report.json: raw vs guarded metrics and per-ticket records
- model_run.checkpoint.jsonl: append-only per-ticket execution state

The checkpoint is active by default. Re-running the same command resumes
successful tickets and retries failed ones. Use --no-retry-failed to preserve
failed records without retrying them.

The live adapter is intentionally not part of CI. CI tests the HTTP contract
offline with a fake response, while real provider execution happens explicitly
against the user's chosen endpoint.

The same predictions.jsonl can be replayed through:

    from benchmarks.model_loop_v27 import run_replay_benchmark
    report = run_replay_benchmark("predictions.jsonl")

## Resumable execution

For a 1,200-ticket live run, use the V27.6 checkpoint runner so an interrupted
process does not lose completed calls:

    from benchmarks.resumable_model_loop_v27 import run_resumable_model_loop

    report = run_resumable_model_loop(
        cases,
        adapter,
        checkpoint_path="artifacts/tickets.checkpoint.jsonl",
    )

The checkpoint is append-only. Successful tickets are skipped on resume and
failed tickets are retried by default.

## Local Ollama auto-detection

You do not need to create an OpenAI endpoint manually when Ollama is running
locally. Ollama exposes an OpenAI-compatible /v1/chat/completions endpoint
through its local server, and its /api/tags endpoint lists installed models.

With no PROMPTFORGE_MODEL_URL or PROMPTFORGE_MODEL_NAME, PromptForge now:

1. connects to http://localhost:11434
2. discovers an installed model through /api/tags
3. selects the first model name in deterministic lexical order
4. sends requests to http://localhost:11434/v1/chat/completions

The local Ollama API key is not a real credential; PromptForge uses the value
ollama, which Ollama ignores for local requests.

So the simplest local run is just:

    python -m benchmarks.providers.openai_compatible --count 1200

For repeated runs:

    python -m benchmarks.replicates_v27 --replicates 3 --count 1200

Set PROMPTFORGE_MODEL_NAME when you want a specific installed Ollama model,
without configuring an endpoint. Use PROMPTFORGE_MODEL_URL only when targeting
a different OpenAI-compatible provider or endpoint.