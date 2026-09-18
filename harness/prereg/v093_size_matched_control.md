# PromptForge V0.9.3 — Size-Matched Structural Control

## Experiment

- Experiment ID: `PROMPTFORGE_V0.9.3_SIZE_MATCHED_CONTROL`
- Task: `T003`
- Task family: `structured_analysis`
- Model: `deepseek`
- Variants: `canonical_t003`, `noise_long`, `key_order`, `content_shift`
- Conditions: `noop_compact`, `noop_padded`, `pairs_compact`, `pairs_bare_compact`
- Repetitions: `15`
- Runs: `240`

## Primary question

Does `pairs_compact` change execution cost beyond the effect explained by serialized context size?

## Primary endpoint

`pairs_compact - noop_padded`

Paired by:

`variant_id + repetition`

## Controls

### noop_compact

Frozen NOOP compact representation.

### noop_padded

Frozen NOOP plus an irrelevant padding field.

The padding is generated independently for each variant so that:

`context_chars(noop_padded) == context_chars(pairs_compact)`

The required task values remain unchanged.

### pairs_compact

Frozen `representation_B` encoded as:

`{"pairs":[...]}`

### pairs_bare_compact

Frozen `representation_B`, with only the bare `pairs` array serialized.

## Secondary endpoints

- `noop_padded - noop_compact`
- `pairs_compact - pairs_bare_compact`

## Measurements

Every run records:

- input tokens
- reasoning tokens
- output tokens
- total tokens
- latency
- prompt characters
- context characters
- provider status
- deterministic quality
- raw output
- parsed output
- variant
- condition
- repetition
- global execution position

## Integrity

The following remain frozen:

- V0.9 source
- `transforms.py`
- `real_executor.py`
- `deepseek_adapter.py`
- V0.9.2 variant snapshot

No API calls during:

- installation
- compilation
- unit tests
- preflight
- dry-run

No retries or silent replacements are permitted.

## Interpretation

If `pairs_compact - noop_padded` is near zero, serialized size explains most of the observed effect.

If it remains positive, representation contributes additional execution cost after size matching.

If it is negative, this is evidence for a structural benefit after size matching.

These interpretation rules do not constitute a freeze criterion.

## No universal freeze

V0.9.3 does not establish a universal representation policy or production routing rule.
