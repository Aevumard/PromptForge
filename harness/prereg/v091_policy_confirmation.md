# PromptForge V0.9.1 — Policy Confirmation

## Experiment

Experiment ID:
`PROMPTFORGE_V0.9.1_POLICY_CONFIRMATION`

Model:
`deepseek`

Task:
`T003`

Conditions:

- `noop_compact`
- `pairs_compact`
- `pairs_bare_compact`

Repetitions:
`40`

Scheduled runs:
`120`

## Primary comparison

`pairs_compact` vs `noop_compact`

`pairs_compact` corresponds to `representation_B`.

The purpose is to confirm or reject the V0.9 T003 hotspot under a larger paired sample.

## Negative control

`pairs_bare_compact`

This remains the same representation operator with alternate serialization.

## Primary endpoint

Paired total-token delta:

`Δtotal = pairs_compact - noop_compact`

Pairing is by repetition.

## Secondary endpoints

- input-token delta
- reasoning-token delta
- output-token delta
- latency delta
- provider status
- deterministic verification
- quality pass

## Frozen rules

- No post-hoc retries.
- No adaptive repetition increase.
- No condition removal.
- No threshold changes.
- No baseline replacement.
- Provider failures remain recorded.
- Quality failures remain recorded.
- Execution order is fixed by schedule.
- Report persistence is atomic.
- Resume is by global execution position.

## Scope

This experiment can produce evidence for a localized PolicyCandidate.

It does not directly modify runtime behavior.

It does not establish a universal representation policy.

## Integrity

Schedule SHA256:

B365DA2E348ABE3CD028E2BB78FD5372C2918A633CC4450A55FA53B28870CBFB

V0.9 source runner:

harness/runner/v09_policy_search_runner.py

V0.9 source SHA256:

353EA5B6D91F4653C659B299A276A369B6396BF915F8496AF0BC7819446E1E38

Task suite SHA256:

12FA0D70631609B231400CEEAF727C1A22AF87334A0075148CF9C66F8C5BDEF3