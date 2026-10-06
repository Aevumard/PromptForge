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
- PROMPTFORGE_MODEL_JSON_MODE (default: 0)
- PROMPTFORGE_MODEL_SYSTEM_PROMPT

## Run 1,200 tickets

From the repository root:

    python -m benchmarks.providers.openai_compatible --output predictions.jsonl --report model_report.json

The command sends only the PromptForge-prepared model packet. It writes:

- predictions.jsonl: successfully parsed raw model predictions
- model_report.json: raw vs guarded metrics and per-ticket records

The live adapter is intentionally not part of CI. CI tests the HTTP contract
offline with a fake response, while real provider execution happens explicitly
against the user's chosen endpoint.

The same predictions.jsonl can be replayed through:

    from benchmarks.model_loop_v27 import run_replay_benchmark
    report = run_replay_benchmark("predictions.jsonl")
