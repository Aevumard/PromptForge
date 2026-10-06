# PromptForge - Agent Entry Point

PromptForge has two deliberately separated surfaces:

- **Agent core:** provider-agnostic context preparation, validation, budgets, inspection, and deterministic policies.
- **Research harness:** historical experiments, preregistration, schedules, reports, runners, and provider adapters.

An AI agent should use the smallest validated surface that solves the task. Do not begin by reading the entire repository.

## First inspection

1. Read `README.md`.
2. Read this file.
3. For real context work, use the installable `promptforge` API.
4. For repository fixtures, use `harness.agent.prepare()`.
5. Inspect `promptforge/core.py` when public core semantics matter.
6. Inspect `harness/runner/transforms.py` only when historical transformation semantics matter.
7. Inspect historical experiments, reports, and providers only when the request actually concerns research or provider execution.

## Public core contract

The installable public API is:

```python
from promptforge import prepare_context, inspect

prepared = prepare_context(
    data={
        "case": {
            "id": "C-001",
            "priority": "high",
        },
        "notes": "noise",
    },
    required=["case.id", "case.priority"],
)
```

The public core implementation lives in `promptforge/core.py`. It must remain importable without importing `harness` at all.

It is provider-agnostic. It requires no API key, network access, or model configuration.

The handoff artifact exposes:

- `context`
- `serialized_context`
- `selected_arm`
- `required_paths`
- `included_paths` / `excluded_paths`
- `required_values_preserved`
- `validation`
- `estimated_tokens`
- `context_chars_saved` / `estimated_tokens_saved`
- `candidates`
- `provenance`

`estimated_tokens` is a dependency-free planning heuristic based on UTF-8 byte length divided by four and rounded up. It is not an exact provider tokenizer measurement.

## Complexity-aware control

The optional adaptive layer lives in `promptforge/adaptive.py` and remains provider-agnostic and harness-free.

Its contract is intentionally separated into seven concerns:

1. **Structure** — `ContextTopologyProfiler` measures the nested-context topology and the required-field boundary.
2. **Regime** — `HeuristicContextRegimeSelector` produces transparent descriptive flags and a bounded candidate preference order.
3. **Local refinement** — `ContextBlockRefiner` explores bounded add/remove moves over optional blocks while required paths remain pinned.
4. **Trajectory** — `ContextTrajectoryMonitor` consumes only the observed prefix of an external search/evaluation trace.
5. **Control** — `ContextPortfolioController` can continue, intensify, switch, or stop using observed scores and remaining budget.
6. **Memory** — `NearestEpisodeRouter` may route from observed historical episodes; `leave_one_family_out()` is the mandatory research boundary for transfer evaluation.
7. **Unified orchestration** — `ComplexContextController` may combine structural, episodic, trajectory, and bounded-control signals, but it must preserve the explicit source and novelty boundary in its result.

This is a control architecture, not a model-quality oracle. Do not infer language-model quality, universal optimality, or SOTA transfer from the topology heuristics. Keep task-level evaluation signals external and explicit.

## Memory consolidation and active credit

`ContextMemoryConsolidator` provides deterministic replay and bounded retention. Recent episodes are retained first, and remaining capacity favors coverage of observed families and strategies.

`ContextMemoryCreditPolicy` is an explicit, opt-in bookkeeping layer. It may assign episode credit from recency decay, repeated family/strategy evidence, observed cost stability, and within-episode comparisons where multiple strategies were measured.

Use credit as an operational retention/ranking signal only. Do not describe it as truth probability, causal attribution, confidence calibration, or model-quality evidence beyond the supplied measurements.

When credit is supplied to consolidation, it is only a secondary selection criterion after observed family/strategy coverage. The default consolidator remains unchanged unless a credit policy is explicitly provided.

`ContextExperienceSnapshot.credit()` evaluates only its frozen evidence. `ContextExperienceSnapshot.memory_aware_route()` uses credit-weighted nearby evidence within the same frozen snapshot. `ContextExperienceStore.consolidate()` and `memory_aware_route()` operate on current mutable memory through fresh immutable snapshots. Existing snapshots remain unchanged and continue to represent their original evidence boundary.

## Credit-aware routing

`ContextMemoryAwareRouter` is an explicit opt-in layer. It uses the same topology scaling contract as the nearest-case router, selects the closest `top_k` observed episodes, and aggregates each strategy's support from structural similarity multiplied by memory credit.

This is an operational preference over observed evidence. Keep the default `NearestEpisodeRouter` behavior unchanged. For research transfer evaluation, fit the aware router only on the training partition and apply the same whole-family holdout boundary already used by `leave_one_family_out()`.

## Cognitive experience records

`ContextEpisode` may retain decision metadata and externally supplied outcome fields in addition to routing topology.

When writing an observation, preserve the selected action, decision source, regime, novelty distance, trajectory state, and measured cost. Optional outcome metadata must remain caller-supplied; do not invent quality or causal labels.

These fields are descriptive state. They can be analyzed later, but they do not become validated causal rules merely because they are stored.
## Relational topology

`ContextRelationProfiler` provides an explicit relational layer over context nodes. It may measure supplied cross-links, connected components, degree concentration, density, and relation kinds.

Relations must be supplied by the integration. Do not infer or fabricate semantic edges and then treat them as ground truth. When relational descriptors are stored in episodes, they become part of the observed routing evidence alongside the hierarchical topology.


## Epistemic evidence boundary

For evidence-heavy reasoning where timestamps, contradictions, interventions, or competing claim types matter, use `promptforge.epistemic` as an explicit additive boundary.

1. `EvidenceRecord` is caller-supplied metadata. PromptForge must not infer its epistemic labels.
2. `EpistemicContextPolicy.cutoff` is a hard exclusion boundary for future evidence.
3. When a cutoff is active, unknown-time evidence is excluded by default; integrations must explicitly opt in to `allow_unknown_time`.
4. Required evidence that crosses the temporal boundary must fail closed rather than being silently dropped.
5. Contradictory evidence and observation/inference/hypothesis coverage can be protected during compression.
6. Intervention metadata is descriptive only. `multivariable_intervention` and `confounded_intervention` are caution states, not causal conclusions.
7. Keep evidence selection auditable through `EpistemicContextResult.audit`.
8. Do not turn relevance, reliability, or causal-status metadata into universal quality or truth claims.

The intended sequence is:

`evidence -> temporal gate -> contradiction/kind preservation -> ranking -> budget -> audited context`

## Uncertainty-aware action boundary

`promptforge.decision` is an opt-in action-selection aid layered above explicit evidence boundaries.

1. Action scores are caller-supplied descriptive metadata; do not infer them from prose and do not call them truth.
2. An action that requires unavailable evidence must fail closed.
3. Reversibility and downside can be hard gates, not only soft preferences.
4. Causal dependence is a caution signal only. It must never become an automatic causal conclusion.
5. Preserve the complete ranking, blocked set, numeric scores, and reasons for auditability.
6. Keep action choice separate from hypothesis identification: a reversible action can be preferred even while causal attribution remains unresolved.

Intended sequence:

`epistemic boundary -> feasibility -> reversibility/downside -> ranking -> action decision`

## Hypothesis and experiment boundary

`promptforge.hypothesis` is an explicit evidence bookkeeping layer.

1. Hypothesis/evidence relationships must be supplied by the integration.
2. `insufficient_evidence` and `unresolved` must not be treated as disproof.
3. Support and contradiction must remain separately visible.
4. Future or unavailable evidence cannot support a hypothesis inside the current epistemic boundary.
5. Experiment priority is a transparent design heuristic, not statistical power or a causal guarantee.
6. Keep hypothesis state separate from operational action choice; an action can be preferred while the causal hypothesis remains contested.

Intended sequence:

`temporal evidence -> support/contradiction ledger -> competing hypotheses -> discriminating experiment priority`

## Memory corroboration boundary

`ContextMemoryAwareRouter` may use `min_family_count` and `min_strategy_evidence` as explicit admissibility gates.

1. Do not let a single structurally similar episode become decisive solely because it is recent or highly credited.
2. Corroboration thresholds are operational safeguards, not truth estimates.
3. When no strategy satisfies the thresholds, retain the transparent scored fallback rather than inventing a winner.
4. Preserve the frozen snapshot boundary; corroboration counts must come only from the fitted training snapshot.

## Temporal memory guard

`ContextMemoryTemporalPolicy` is an opt-in admissibility boundary for episodic memory.

It adds a second, explicit clock on top of ordinary snapshot immutability:

- the experience sequence is the temporal clock;
- `as_of_index` provides an inclusive point-in-time boundary for historical evaluation;
- entries after that boundary are excluded as future memory;
- `max_age` can hard-exclude stale memory;
- `min_freshness_weight` can require a minimum recency signal;
- allowed and blocked sources are caller-supplied trust boundaries;
- the full exclusion audit is retained in `ContextMemoryAwareRoute`.

When enabled, eligible memory receives an additional deterministic freshness weight in the memory-aware score. When nothing passes the temporal gate, routing fails closed rather than silently falling back to future or stale evidence.

The default router remains unchanged when no temporal policy is supplied.

The intended boundary is:

`frozen memory -> temporal/source gate -> freshness weighting -> corroboration -> routing`

This is a temporal contamination defense, not a truth score or source-trust oracle.

## Cognitive loop

`ContextCognitiveLoop` is the end-to-end public orchestration surface for an online adaptive cycle. It may optionally consume a separate routing-policy meta-memory store for adaptive mode selection.

Its boundary is explicit:

1. `propose()` reads a frozen `ContextExperienceSnapshot`, profiles the current context, and produces a `ContextCognitiveProposal`.
2. The caller executes the selected strategy outside PromptForge.
3. `observe()` writes the externally measured outcome back as a `ContextEpisode`.

A proposal must retain the experience version it was based on. In adaptive mode it may also retain the policy-evidence version, policy-history version, stability rate, freshness age, and refresh recommendation that governed the concrete routing mode. Do not let later observations mutate the evidence represented by an existing proposal. The loop must never invent model-quality outcomes, provider responses, or hidden labels.

Use the cognitive loop when an integration needs the complete adaptive lifecycle. Use the lower-level adaptive, memory, routing-policy, routing-history, and orchestration APIs when an experiment needs finer control over individual stages.

## Routing-policy meta-memory

`ContextRoutingPolicyHistory` is a bounded operational meta-memory layer over completed routing-policy evaluations. It is separate from `ContextExperienceStore`: ordinary episodes record task outcomes, while policy history records prior policy evaluations.

Keep the evidence clocks separate. A policy-evidence `version` refers to the frozen episodic snapshot used by that evaluation; the policy-history `version` refers to the number of policy-evidence records stored in meta-memory.

Adaptive routing may use either explicit frozen policy evidence or a policy-history snapshot. When history is used, consensus is deterministic and falls back to `nearest` when the configured evidence threshold is not met. Optional stability and freshness gates can also force `nearest` when policy selection is unstable or stale relative to the current experience version, and the proposal records that refresh signal. These are operational controls, not quality estimates.

The policy-history boundary must remain leakage-safe: the outcome of the current proposal cannot be used to create or select the policy for that same proposal.

## Controlled policy refresh

`ContextRoutingPolicyRefreshController` is the explicit execution gate for routing-policy reevaluation.

Keep these boundaries intact:

1. `ContextRoutingPolicyHealth` determines whether existing policy evidence is stable/fresh enough.
2. `ContextRoutingPolicyRefreshController` decides whether reevaluation is eligible, considering evidence thresholds, cooldown, and optional refresh budget.
3. `ContextCognitiveLoop.refresh_memory_routing_policy()` performs holdout evaluation and records the new evidence only after the gate permits it.
4. `record_refresh()` is called only after successful evidence persistence.

Do not turn the refresh controller into an automatic model-quality loop. It schedules an evidence refresh; it does not generate outcomes, causal claims, or hidden labels. Reasons such as `insufficient_evidence`, `unstable_policy`, `stale_policy`, `refresh_cooldown`, and `refresh_budget_exhausted` must remain explicit and auditable.

## Active exploration

`ContextExplorationController` is a bounded data-collection mechanism. Keep exploration separate from claims about strategy quality.

Rules:

1. Explore only among already-available feasible strategies.
2. Use current observed coverage, novelty, policy-refresh signals, cooldown, and explicit budget as inputs.
3. Record an exploration only after the corresponding probe outcome is observed.
4. Do not label an unexplored or probed strategy as better merely because it was selected for exploration.
5. Preserve the existing temporal/leakage boundary: the current outcome cannot select the policy or exploration decision for the same proposal.

Use `source="bounded_exploration"` and `action="probe"` for proposal provenance when the exploration controller changes the concrete strategy.

## Exploration evidence integrity

The exploration adjudicator now adds an explicit evidence-integrity boundary before a challenger can pass the adoption gate.

1. Multiple probe outcomes for the same challenger within one episode are aggregated into one episode-level comparison instead of inflating the sample size.
2. Repeated probe costs are averaged within that episode; the adjudicator does not cherry-pick the best repeated outcome.
3. Strict gains are required for wins. Exact ties are recorded separately and reduce the win rate rather than counting as victories.
4. Adoption requires a consistent observed baseline strategy across the retained comparison set. Mixed baselines are surfaced as a mismatch and block adoption.
5. Unique episode coverage is retained separately from raw comparison count for auditability.
6. The adjudicator remains descriptive: it does not mutate policy, memory, or strategy preference.

The strengthened boundary is:

`probe -> episode-level aggregation -> strict win/tie accounting -> baseline consistency -> family/coverage gates -> adoption gate`

## Exploration adjudication

`ContextExplorationAdjudicator` is the evidence boundary after a probe.

Do not treat a single successful probe as an adopted strategy. Require the configured comparison and family thresholds, and keep the evidence within the same episode when computing challenger gain.

`ContextCognitiveLoop.exploration_adoption_decision()` is descriptive and conservative. It does not mutate policy, memory routing mode, or strategy preferences.

Keep the sequence explicit:

`probe -> observe -> compare within episode -> aggregate across families -> adoption gate`

## Policies

The default policy is `minimal`, implemented as deterministic `minimal_serialized_context`.

Available public policies:

- `minimal`
- `budget_constrained`

Supplying `budget_tokens=` with the default `minimal` policy automatically activates `budget_constrained`.

The budget policy selects the smallest serialized candidate that fits the requested estimated-token budget. If none fits, fail explicitly rather than silently violating the budget.

Do not convert this mechanical context-size policy into a claim about model quality.

## Nested requirements

Required field paths use dot notation:

```python
required=["user.id", "task.priority"]
```

The core selects only the required branches for selection-based candidates and validates that required values survive the transformation.

Do not silently invent required fields. Treat the caller's required paths as the semantic boundary.

## Lightweight schema validation

Schema validation is dependency-free:

```python
schema={
    "user.id": str,
    "score": float,
}
```

Schema paths automatically become required paths. Validation is applied to the semantic decoded context, not only to an internal structural representation.

## Inspection

Use `inspect()` for non-mutating planning:

```python
report = inspect(
    context,
    required=["task.id", "task.priority"],
)
```

This reports field counts, required coverage, serialized size, bytes, and estimated tokens.

## Repository fixture API

`harness.agent.prepare()` is a repository fixture surface for deterministic regression tasks such as `T003`.

It depends on the checked-out repository task suite and is intentionally separate from the installable core API.

Use it for reproduction, not as the generic public integration interface.

## Provider boundary

The installable core must remain independent of OpenAI, DeepSeek, Gemini, or any other provider.

Provider adapters belong to the research/experimental surface. Credentials must come from environment variables or explicitly injected provider objects.

Never commit API keys, tokens, `.env` files, or provider secrets.

## Research boundary

Preregistration is the experimental contract. Schedules, task snapshots, protected hashes, reports, runners, and tests are part of the research definition.

Historical results are evidence for the tested conditions. Do not silently generalize one task, provider, model, representation, or experiment into a universal claim.

## Frozen V0.9.3 surface

The public release preserves the validated V0.9.3 transformation, executor, provider adapter, policy runner, task suite, schedule, variant snapshot, preregistration, and regression test.

Agent-facing infrastructure is additive. Do not alter frozen V0.9.3 experimental machinery merely to expose or polish the current public API.

## Release integrity

After changing public files:

1. Run the full test suite.
2. Verify the installable package smoke test.
3. Refresh `PUBLIC_RELEASE_MANIFEST.json`.
4. Confirm CI is green.

The manifest records SHA-256 values for every tracked public file except itself.

## Core rule

Reuse validated machinery first. Change the smallest surface necessary. Keep the installable agent API, executable infrastructure, experimental design, empirical evidence, and interpretation clearly separated.


## Calibration family scope

Confidence calibration can now use an explicit family scope when historical confidence behavior is not transferable across task families.

1. ConfidenceObservation.family is now an active optional boundary through ConfidenceCalibrationPolicy.target_family.
2. When target_family is set, observations from other families are excluded before binning, smoothing, isotonic calibration, or metrics.
3. min_family_observations can fail closed when the target family does not contain enough historical observations.
4. Excluded family observations are retained in ConfidenceCalibrationModel.family_excluded_ids for auditability.
5. Required observations that fall outside the target family remain a hard failure.
6. When no target family is supplied, existing pooled behavior remains unchanged.

The intended boundary is:

historical labeled outcomes -> family scope -> temporal gate -> supported bins -> monotone calibration -> bounded adjustment

Family scope is an evidence-admissibility control, not a claim that family labels are causally meaningful.

## Action evidence stance admissibility

`ActionPolicy.require_support_stance` optionally requires action support anchors to carry compatible, explicitly declared evidence stances. The default admissible stance is `supports`; other stances require an explicit policy. Missing stance metadata or a missing evidence boundary fails closed. No stance is inferred from prose. The default remains backward compatible.

Boundary: epistemic evidence -> support anchors -> stance admissibility -> feasibility and safety gates -> action ranking

## Action evidence anchoring

Action support can be made auditable by declaring support evidence ids on each action. An explicit strict policy can fail closed when support is missing, outside the current evidence snapshot, or below the configured anchor count. Anchor count is not statistical independence. The default remains backward compatible.

Boundary: epistemic evidence -> explicit action support anchors -> feasibility and safety gates -> action ranking

## Action evidence relevance guard

`ActionCandidate.support_evidence_tags` is an explicit caller-supplied relevance scope for support anchors. When `ActionPolicy.require_support_tag_match=True`, every support anchor must share at least one declared tag with that scope. Missing scope, missing boundary, or missing tag overlap fails closed. PromptForge does not infer relevance from evidence prose.

`ActionDecision.support_evidence_tag_matches` preserves the observed tag overlap for auditability. Tag overlap is an admissibility control, not a semantic relevance or causal sufficiency claim.

Boundary: epistemic evidence -> support anchors -> stance admissibility -> explicit relevance-tag admissibility -> feasibility and safety gates -> action ranking

## Action support quality guard

`ActionPolicy.require_support_quality=True` adds an explicit per-anchor quality gate using the retained evidence's caller-supplied `relevance` and `reliability` metadata. `min_support_relevance` and `min_support_reliability` define the floors. Missing metadata or a value below either floor fails closed. `ActionDecision.support_evidence_quality` preserves the declared values for auditability.

These fields are admissibility metadata only. PromptForge does not independently validate them and does not interpret them as truth, causal strength, or statistical power. The default remains unchanged when the guard is disabled.

Boundary: epistemic evidence -> support anchors -> stance admissibility -> explicit relevance tags -> quality floors -> feasibility and safety gates -> action ranking

## Action support provenance diversity

`ActionPolicy.require_support_provenance_diversity=True` adds an optional provenance-structure gate for action support anchors. `min_distinct_support_sources` defaults to 2 and requires that the current support set contains at least that many explicitly declared, non-unknown source labels. `max_support_anchors_per_source` can additionally cap concentration from any one declared source. Missing source metadata fails closed, while `unknown` does not count toward distinct-source diversity.

`ActionDecision.support_evidence_provenance` preserves the caller-supplied source labels for auditability. These labels are provenance metadata only: different source labels do not establish statistical independence, truth, causal sufficiency, or absence of shared upstream dependencies. The guard is opt-in and remains backward compatible when disabled.

Boundary: epistemic evidence -> support anchors -> stance admissibility -> relevance tags -> quality floors -> provenance diversity -> feasibility and safety gates -> action ranking

## Hypothesis source cap

HypothesisEvidencePolicy can limit how many support or contradiction records from one declared source contribute to an assessment. Selected and excluded evidence ids remain auditable. Unknown sources can remain isolated per evidence id. This is an operational corroboration safeguard, not a truth claim.

Boundary: explicit hypothesis mapping -> optional source cap -> support/contradiction state -> experiment priority

## Confidence calibration boundary

The public core now includes an opt-in confidence calibration layer in
promptforge.confidence for integrations that already have externally labeled
historical outcomes.

Confidence calibration is deliberately separated from action selection:

1. ConfidenceObservation requires the integration to supply both a confidence
   value and the measured binary outcome.
2. ConfidenceCalibrator.fit() uses only that explicit historical sample.
3. An optional cutoff is a hard temporal boundary; future and, by default,
   unknown-time observations are excluded.
4. Required calibration observations that cross the boundary fail closed.
5. Supported confidence bins need a minimum observation count before they can
   change a live confidence.
6. A monotone isotonic fit prevents calibration from creating a confidence
   curve that reverses the observed ordering.
7. Smoothing limits extreme rates in small supported bins.
8. max_adjustment bounds the amount of confidence correction.
9. Sparse bins may fall back to the global historical rate, while insufficient
   total evidence leaves the current confidence unchanged.
10. Brier score, expected calibration error, and maximum calibration error are
    descriptive metrics over the supplied sample. They are not universal
    truth scores or model-quality guarantees.

The intended control boundary is:

historical labeled outcomes -> temporal gate -> supported bins -> monotone
calibration -> bounded adjustment -> audited confidence

The calibration model does not consume the outcome of the current confidence
being assessed. This prevents the calibration layer from silently self-
validating on the same decision it is supposed to regulate.


## Decision and triage separation boundary

For tasks that require triage, prioritization, escalation, or operational action, keep context admissibility, priority, and actionability as distinct decision layers.

The required control boundary is:

evidence -> temporal/admissibility gate -> priority decision -> actionability/safety gate -> action -> audit

### Priority is not evidence confidence

1. Priority answers how much the case should be advanced, not whether the requested action is safe.
2. Keep urgency (deadline/SLA pressure and time-to-breach), importance (impact, scope, severity, or cost of delay), and actionability (whether the available evidence is sufficient to execute a concrete action) as separate dimensions.
3. Evidence relevance/reliability/confidence must not be used as an automatic penalty on priority merely because evidence is weak.
4. Weak evidence can coexist with extreme urgency. A high-priority case can legitimately be blocked for action.
5. An unresolved contradiction can block an action without demoting the case's priority.
6. If the caller supplies an explicit priority formula that intentionally uses evidence quality, preserve that as caller policy and label it as such. Do not invent that coupling as a PromptForge rule.
7. Do not invent numeric weights, thresholds, or SLA bands merely to make an example look precise. When the task does not supply them, use qualitative ordering or explicitly label proposed numbers as illustrative policy.

### Source precedence is not truth

Do not create a universal source-of-truth hierarchy such as system log > database > user and silently discard the lower-ranked claim.

Instead, preserve materially conflicting evidence, retain provenance and timestamps, use source type as caller-supplied context rather than a truth oracle, resolve contradictions only when the task supplies an explicit adjudication policy or additional evidence, and let the action gate block only the actions that depend on the unresolved fact.

Different sources or source labels do not by themselves prove correctness, independence, or causality.

### Priority and action gates must stay separate

The UncertaintyActionGate in promptforge.decision is an action-selection boundary. It must not be repurposed as a ticket-priority scorer.

For a triage system, produce separate audit objects such as:

priority = {
    urgency,
    importance,
    priority_band,
    rationale
}

actionability = {
    allowed,
    blocked,
    reasons,
    required_evidence
}

A valid outcome is therefore:

priority = P0, actionability = BLOCKED

Do not lower priority merely because an automatic action is blocked.

### Temporal evidence belongs to admissibility

Future or unknown-time evidence under a hard cutoff is excluded from the historical decision context. Its exclusion does not imply that the underlying case was low priority.

The original decision must be reproducible from the admissible evidence available at the decision time.

### Model-facing rule

When an integration asks PromptForge to solve a prioritization problem, first compile the admissible context, then decide priority, and only afterward evaluate whether an action is safe and supported. Never collapse those layers into one generic confidence score.

The intended ordering is:

what can be used -> how urgent/important is the case -> can we safely act -> what action -> why

## External architecture pattern corpus

Before extending agent-facing orchestration rules, consult `docs/agent_corpus_external_patterns.md`.

The corpus is informed by current public architecture patterns from LangGraph, DSPy, Pydantic AI/pydantic-graph, Guardrails, and LlamaIndex. The adopted principles are deliberately generic:

- explicit state and checkpoint boundaries for resumable workflows;
- contract-first interfaces and metric-first optimization;
- typed, serializable stage state and output validation;
- bounded repair/retry behavior with preserved failures;
- retrieval as candidate generation followed by explicit evidence admission;
- deterministic rules separated from model-generated interpretation;
- human review represented as a durable state transition;
- replayable lineage and leakage-safe learning.

External frameworks are references for architecture patterns, not dependencies and not sources of truth. Do not copy framework-specific claims into PromptForge without verifying their actual semantics.

## Typed triage state boundary

For operational triage, use `promptforge.triage` when the workflow needs explicit lifecycle state.

The public boundary is:

`admitted -> prioritized -> action_gated -> waiting_human/executed -> observed -> closed`

Use `PriorityAssessment` for urgency, importance, priority band, and rationale. Use `ActionDecision` for action feasibility and safety. Do not merge these objects into a single confidence score.

Use `TriageState.transition()` to create immutable successor states. Each successor records the parent decision id so historical states remain replayable.

A high-priority case may legitimately transition to `waiting_human` while retaining its priority. Later observations are stored on successor states and must not mutate the earlier evidence boundary.

The JSON contract is `schemas/triage-state.v1.json`.

## Durable human review boundary

When an operational action pauses for human intervention, the review must be represented as data rather than an out-of-band edit.

`promptforge.human_review.HumanReviewRecord` captures:
- reviewer identity;
- review timestamp;
- the exact evidence snapshot reviewed;
- the review decision;
- reviewer rationale;
- declared changes;
- the resulting policy version.

`TriageState` enforces two additional replay invariants:
1. a `waiting_human -> action_gated` transition must carry a completed `HumanReviewRecord`;
2. resumption must provide a fresh `ActionDecision`, so an old action decision is never silently reused after human intervention.

The reviewed evidence snapshot must match the triage state's evidence snapshot. A human review therefore cannot silently adjudicate one snapshot and resume another.

A review record does not establish factual truth, causal validity, or statistical independence. It records an intervention at a workflow boundary and makes that intervention replayable and auditable.

Schema: `schemas/human-review.v1.json`.

## External action execution boundary

`promptforge.execution` handles the boundary around side effects without executing them.

`ActionExecutionGuard` requires an integration-owned idempotency key by default and classifies a proposed attempt as:
- `execute` when the key has no prior record;
- `retry` when prior attempts failed and the explicit retry budget allows another attempt;
- `duplicate` when the same key already succeeded or is in flight;
- `blocked` when the key is missing or the retry budget is exhausted.

`ActionExecutionRecord` is the immutable external receipt. A `TriageState` cannot enter `executed`, `observed`, or `closed` without a successful execution receipt whose `action_id` matches the selected action.

This prevents the workflow from claiming that a side effect happened merely because an action was selected. PromptForge still does not execute the side effect; the integration owns the real-world operation and receipt.

Idempotency keys are operational duplicate-action controls, not proofs of business correctness or delivery semantics.

Schema: `schemas/action-execution.v1.json`.

## Token-efficient context path

For long or noisy agent tasks, prefer the smallest context contract that preserves required information.

`promptforge.budget` adds a provider-agnostic packing layer:

1. Create `ContextBlock` values for independently controllable context units.
2. Mark load-bearing information with `required=True`.
3. Give optional blocks caller-supplied `utility`; PromptForge does not infer semantic importance.
4. Use `plan_context(..., budget_tokens=...)` to reserve headroom and pack optional blocks deterministically.
5. Use `plan.compact_manifest(blocks)` as the small model-facing receipt of what was included and omitted.
6. Use `plan.materialize(blocks)` to construct the selected nested context.

Example:

```python
from promptforge import ContextBlock, plan_context

blocks = (
    ContextBlock("ticket", ticket, required=True, path="ticket"),
    ContextBlock("customer_history", history, utility=0.7, path="history"),
    ContextBlock("old_notes", old_notes, utility=0.1, path="old_notes"),
)
plan = plan_context(blocks, budget_tokens=1800, reserve_ratio=0.10)
context = plan.materialize(blocks)
manifest = plan.compact_manifest(blocks)
```

Use an exact provider tokenizer through `estimator=` when available. The default byte-based estimate is only a planning heuristic.

Never drop required information silently to make a budget fit. Never claim that token reduction is itself evidence of better model quality.
