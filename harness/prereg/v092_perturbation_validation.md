# PromptForge V0.9.2 — Perturbation Validation

Experiment:
PROMPTFORGE_V0.9.2_PERTURBATION_VALIDATION

Task:
T003

Task family:
structured_analysis

Model:
deepseek

Conditions:
- noop_compact
- pairs_compact
- pairs_bare_compact

Variants:
- canonical_t003
- noise_long
- key_order
- content_shift

Repetitions:
15

Scheduled runs:
180

Primary comparison:
pairs_compact vs noop_compact

Primary endpoint:
Pooled paired delta in total tokens.

Pairing:
variant_id + repetition

Negative control:
pairs_bare_compact

Purpose:
Test whether the V0.9/V0.9.1 T003 pairs_compact signal survives controlled
perturbations of the same task family.

Frozen rules:
- no post-hoc retries
- no adaptive repetition changes
- no variant removal
- no condition removal
- no threshold changes
- no baseline replacement
- failures remain recorded
- schedule remains immutable
- runtime protected files remain unchanged
- V0.9 source runner remains unchanged

This experiment produces evidence for a localized PolicyCandidate.
It does not directly modify runtime behavior.
It does not freeze a universal representation policy.

Integrity:

Schedule SHA256:
21203DE648E16BC3AE6E4FB8110DEB51FEC33291C55A8B9B1DF97566D53AD35E

Variant snapshot SHA256:
A83422BD2B58DEDE1EC44C0D9F0B040B040F96EC467B4BD23DB161110F781584

V0.9 source SHA256:
353EA5B6D91F4653C659B299A276A369B6396BF915F8496AF0BC7819446E1E38

Task suite SHA256:
12FA0D70631609B231400CEEAF727C1A22AF87334A0075148CF9C66F8C5BDEF3