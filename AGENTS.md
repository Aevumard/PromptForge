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
