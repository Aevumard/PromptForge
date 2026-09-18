# PromptForge V0.7 - T001-T004 Confirmatory Protocol

STATUS: PRE-REGISTRATION CANDIDATE - NO API EXECUTION

## 1. Objective

Evaluate whether the representation transformation produces a reproducible reduction in reasoning tokens across T001-T004.

Evaluate formal equivalence between selection_representation and selection_representation_A.

## 2. Frozen tasks

T001
T002
T003
T004

## 3. Confirmatory arms

selection_only
selection_representation
selection_representation_A

selection_representation_B is excluded from the primary confirmatory collection.

## 4. Sample size

15 repetitions per task.

4 tasks x 15 repetitions x 3 arms = 180 API executions.

No adaptive increase or decrease of repetitions is allowed after inspecting results.

## 5. Schedule

Primary schedule: deterministic Latin Square.

For the three confirmatory arms, each arm occupies each relative position equally often across complete 3 repetition cycles.

15 repetitions = 5 complete Latin Square cycles per task.

## 6. Primary baseline

Primary baseline estimator: LOO_TRIMMED_MEAN_20.

For each task and repetition, collect the other 14 selection_only reasoning observations from the same task.

Remove the lowest and highest value.

Average the remaining 12 observations.

Do not use the contemporaneous selection_only observation as the primary baseline estimator.

## 7. Primary endpoint

robust_delta_reasoning = arm_reasoning_tokens - LOO_TRIMMED_MEAN_20(selection_only_reasoning_tokens)

Negative values indicate lower reasoning cost than the robust baseline.

## 8. Representation freeze rule

For selection_representation, every task must satisfy all criteria:

1. Mean robust_delta_reasoning < 0.
2. Holm adjusted p < 0.05.
3. Cohen dz > 0.30.
4. Raw reasoning SD of selection_representation <= raw reasoning SD of selection_only.

All four tasks must pass for confirmatory freeze.

## 9. Equivalence test for representation_A

Comparison: representation_A minus representation.

Equivalence margin: plus or minus 10 reasoning tokens.

Test: TOST with alpha = 0.05.

Decision rule: the 90 percent confidence interval for A minus representation must lie completely inside [-10, +10].

Analysis population: 4 tasks x 15 repetitions = 60 matched pairs.

Failure to reject a zero difference is not itself evidence of equivalence.

## 10. Multiplicity

The four task level representation tests use Holm correction.

The A versus representation TOST is one global equivalence family with alpha 0.05.

## 11. B

selection_representation_B is exploratory only and receives zero API budget in this confirmatory protocol.

Any future B experiment requires a separate preregistration.

## 12. Secondary outcomes

Secondary outcomes are descriptive only:

output tokens
total tokens
latency
input token overhead
reasoning variance
robust delta output
robust delta total

Secondary outcomes cannot rescue an arm that fails a primary criterion.

## 13. Quality gate

Every included execution must satisfy the frozen deterministic verifier.

quality_pass must be true.

Quality failures are not converted into successes because token metrics look favorable.

## 14. Integrity rules

Frozen before collection:

task set
arm set
repetition count
Latin Square schedule
baseline estimator
equivalence margin
alpha
effect size threshold
variance criterion
freeze rule

After collection begins:

no task removal after seeing results
no arm removal after seeing results
no threshold changes
no replacement of the primary baseline estimator
no adaptive repetitions

## 15. Scope

Results apply only to the V0.7 task suite, model, provider, fixtures, and protocol used in the experiment.

## 16. Status

V0.7 PREREGISTERED DESIGN READY - COLLECTION NOT STARTED
