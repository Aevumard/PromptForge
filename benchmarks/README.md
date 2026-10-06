# V27.1 adversarial benchmark

This benchmark is a deterministic red-team suite for the V27 AI-native control layer.

It does not call an LLM, external provider, or live tool. It tests the boundaries that PromptForge is responsible for enforcing:

- priority remains separate from actionability;
- future evidence cannot cross an epistemic cutoff;
- provenance diversity can block concentrated support;
- required context survives hard budgets while low-value context is omitted;
- deferred context stays out of the model-facing payload until explicitly loaded;
- old tool outputs preserve head/tail material and a replay digest;
- external execution is idempotency-aware;
- human review requires a fresh action decision before resuming.

Run it from the repository root:

```bash
python benchmarks/v27_adversarial.py
python -m unittest tests/test_v27_adversarial.py
```

The benchmark is intentionally separate from the frozen historical research harness. A future version can add model-in-the-loop A/B evaluation on top of these deterministic control checks without changing the historical harness semantics.
