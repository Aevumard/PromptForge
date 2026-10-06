# PromptForge

Agent-native context engineering and experimental harness.

PromptForge is a Python layer for transforming, compiling, validating, and experimentally evaluating the context sent to language models.

## What PromptForge does

- Select relevant context for a task.
- Compile alternative structural representations.
- Construct prompts from compiled context.
- Normalize provider execution results.
- Run controlled and reproducible experiments.

## Agent-native workflow

An agent can inspect the task and context, select relevant fields, compile a representation, validate invariants, and execute or evaluate the resulting prompt.

Primary public core:
promptforge/core.py

Primary research transformation surface:
harness/runner/transforms.py

Task definitions:
tasks/suite.json

Provider adapters:
harness/providers/

Experimental contracts:
harness/prereg/

Retained evidence:
harness/reports/

Regression tests:
tests/

## Complexity-aware control layer

The public core now includes an optional, provider-agnostic adaptive layer for structurally complex contexts:

- `ContextTopologyProfiler` builds a deterministic structural signature from nested mappings and required paths.
- `HeuristicContextRegimeSelector` exposes transparent regimes such as `hub_dominated`, `deep_hierarchical`, `fragmented`, and `wide_sparse`.
- `rank_context_candidates()` orders already-evaluated preparation arms for bounded probing without pretending the heuristic is an oracle.
- `ContextBlockRefiner` performs bounded local add/remove moves over optional context blocks while pinning required fields and enforcing an optional token budget.
- `ContextTrajectoryMonitor` detects active, stagnating, exploratory, premature-collapse, and extinct search states from observed prefixes.
- `ContextPortfolioController` decides whether to continue, intensify, switch, or stop using only observed task-level scores and a remaining budget.

This architecture is deliberately graph-inspired: it borrows the pattern **structure -> regime -> portfolio -> trajectory -> control** that proved useful in the author's graph-optimization work, while keeping the PromptForge implementation domain-specific to context. No graph benchmark result is transferred to language-model quality, and no universal routing claim is implied.

The adaptive layer is optional. The default `prepare_context()` contract and the frozen research harness remain unchanged.

## Episodic routing and memory

PromptForge can retain measured context episodes and route a new context toward strategies that performed well on structurally similar prior episodes.

The memory contract is:

`episode = (episode_id, family_id, topology, strategy, observed_cost)`

`NearestEpisodeRouter` performs nearest-case routing in a scaled topology space. The router never receives hidden labels or future results; it only sees supplied historical episodes.

For research, `leave_one_family_out()` removes an entire context family from fitting before prediction. Held-out predictions are then compared against the best observed strategy for that held-out episode. This preserves the same evidence boundary used by the graph-routing work: routing performance is an empirical question, not an assumption.

This layer is intentionally separate from `prepare_context()`: memory can guide strategy selection, but it does not silently become the default policy.

## Unified adaptive decision

`ComplexContextController` composes the layers into one decision envelope:

`structure -> regime -> candidates -> memory -> trajectory -> control`

A non-novel episodic match may provide the strategy. A novel memory match falls back to structural routing. Observed trajectory/probe evidence has precedence when an active search is already underway.

The returned decision records its source and novelty distance so downstream systems can audit why a strategy was selected.



## Cognitive loop

The public API now exposes an explicit observe-decide-learn cycle through ContextCognitiveLoop.

A cycle is deliberately split into two operations:

1. propose() profiles the current context, freezes the current experience snapshot, routes from prior observations when the case is sufficiently familiar, and delegates the final choice to ComplexContextController.
2. observe() accepts the externally measured outcome and records it as a ContextEpisode for future cycles.

The resulting state transition is:

context -> topology -> regime -> memory -> control -> proposal -> external execution -> observed outcome -> memory

A minimal one-call integration is available through `ContextCognitiveLoop.prepare()`:

```python
from promptforge import ContextCognitiveLoop

loop = ContextCognitiveLoop()
result = loop.prepare(
    cycle_id="T-001-cycle-1",
    data={"task": {"id": "T-001", "action": "review"}},
    required=["task.id", "task.action"],
)

print(result.proposal.decision.strategy)
print(result.prepared["serialized_context"])
```

When relations are supplied, the same call incorporates their structural profile into future experience routing.

The loop does not call a model or invent a quality score. Execution remains outside PromptForge, while the observed outcome is explicitly written back into experience memory. Every proposal records the experience version it was based on, making the online learning boundary auditable.

ContextExperienceSnapshot is immutable. New observations can improve future routing without rewriting the evidence used by an earlier proposal or evaluation.

## Relational context topology

PromptForge can also accept an explicit relation graph over context nodes. `ContextRelationProfiler` measures relation count, connected components, degree concentration, density, and relation kinds without trying to infer semantics from raw text.

These descriptors are added to the routing feature space used by the cognitive loop. When relational evidence is supplied, future experience can therefore be matched using both hierarchical shape and explicit cross-links.

The relation graph is caller-supplied by design. PromptForge does not treat a guessed semantic relationship as ground truth; the integration must provide the relation evidence it wants the adaptive system to use.

## Cognitive experience record

Each observed episode now retains both the routing topology and the decision state that produced the outcome:

`episode = (context topology, relational topology, regime, strategy, action, source, novelty, trajectory state, observed cost, outcome)`

This separates two kinds of memory:

- structural memory: what the context looked like and which relationships were present;
- operational memory: what PromptForge decided, why it decided it, and what happened afterward.

The richer metadata is descriptive evidence. It is not automatically treated as causal knowledge. Future research can analyze which states precede lower observed cost while keeping the outcome external and auditable.
## Experience self-observation

experience_summary() provides a descriptive summary of accumulated episodes: counts by family, strategy, action, decision source, regime, and trajectory state, plus observed mean cost and mean novelty distance. It is an inspection surface only; it does not infer causal effects.

## Memory consolidation and replay

`ContextMemoryConsolidator` provides deterministic bounded replay and retention. Recent episodes are retained first; remaining capacity is allocated to increase observed family and strategy coverage.

`ContextMemoryCreditPolicy` adds an opt-in active-memory signal. Each episode gets explicit components for recency decay, repeated family/strategy evidence, observed cost stability, and within-episode comparisons when multiple strategies were measured. The resulting credit is a retention/ranking aid, not a probability of truth, a causal estimate, or a model-quality guarantee.

When supplied to `ContextMemoryConsolidator`, credit is used only after family/strategy coverage has been considered. The default consolidator remains behavior-compatible and does not apply credit unless requested.

`ContextExperienceSnapshot.credit()` computes credit inside the snapshot's frozen evidence boundary. `ContextExperienceStore.consolidate(credit_policy=...)` can then use that explicit signal for bounded forgetting without mutating older snapshots.

This gives PromptForge an explicit memory lifecycle:

`observe -> store -> assess credit -> consolidate/replay -> route -> decide`

## Credit-aware routing

`ContextMemoryAwareRouter` is an opt-in routing layer above the nearest-case baseline. It keeps the structural distance contract, examines the closest `top_k` observed episodes, and aggregates evidence by:

`strategy score = sum(credit × structural similarity)`

This lets repeated, recent, and stable observations influence a routing decision without pretending that the closest single case is sufficient evidence. The returned `ContextMemoryAwareRoute` exposes the selected credit, candidate count, and per-strategy scores for auditability.

The aware router must be fitted only on the intended training snapshot. For transfer evaluation, preserve the existing whole-family holdout boundary; never calculate credit from held-out outcomes.

## Cognitive memory mode

`ContextCognitiveLoop` exposes three explicit memory-routing modes:

- `nearest` — the original `NearestEpisodeRouter` behavior and the default.
- `credit` — the opt-in `ContextMemoryAwareRouter`, with a bounded `top_k` neighborhood and explicit memory credit.
- `adaptive` — selects between the two concrete modes using policy evidence that was evaluated before the current proposal.

The selected requested mode, concrete mode, `top_k`, and policy provenance are recorded in `ContextCognitiveProposal`. The proposal schema is now `context-cognitive-proposal.v8`.

```python
from promptforge import ContextCognitiveLoop

loop = ContextCognitiveLoop(
    memory_routing_mode="credit",
    memory_top_k=5,
)
```

This remains provider-agnostic and observational: routing changes how already-observed evidence is aggregated; it does not fabricate outcomes or execute a model.

## Routing-policy learning

PromptForge can compare its routing modes before allowing a future cycle to choose between them.

\`ContextRoutingPolicyEvaluator\` evaluates \`nearest\` and \`credit\` using whole-family holdout folds. Each training fold constructs routing evidence only from the non-held-out families; the held-out family is evaluated afterward.

\`ContextRoutingPolicyEvidence\` freezes those results. \`ContextRoutingPolicySelector\` chooses the mode from observed relative regret, with a deterministic fallback to \`nearest\` when evidence is insufficient.

The cognitive loop accepts \`memory_routing_mode="adaptive"\` plus optional previously evaluated policy evidence:

\`\`\`python
from promptforge import ContextCognitiveLoop

loop = ContextCognitiveLoop()
policy = loop.evaluate_memory_routing_policy()
adaptive_loop = ContextCognitiveLoop(
    memory_routing_mode="adaptive",
    memory_routing_policy_evidence=policy,
)
\`\`\`

The current case never contributes its outcome to the policy used for that same proposal. This keeps policy learning temporally separated from decision evidence.

## Routing-policy meta-memory

`ContextRoutingPolicyHistory` stores previously evaluated policy evidence separately from ordinary task episodes. Its history version is an internal meta-memory clock; each evidence item retains the episodic snapshot version on which it was computed.

The history is bounded and immutable through `ContextRoutingPolicyHistorySnapshot`. A deterministic consensus selector can use the full history or a recent window, with `nearest` as the conservative fallback when observations are insufficient. `ContextRoutingPolicyStability` reports mode switches and a descriptive stability rate; `ContextRoutingPolicyHealth` can turn that history into an explicit stability gate. The health state is not a confidence score or a model-quality estimate.

An adaptive cognitive loop can consume this meta-memory directly:

```python
from promptforge import ContextCognitiveLoop, ContextRoutingPolicyHistory

history = ContextRoutingPolicyHistory(max_entries=32)
loop = ContextCognitiveLoop(
    memory_routing_mode="adaptive",
    memory_routing_policy_history=history,
)

evidence = loop.evaluate_and_record_memory_routing_policy()
```

This creates the explicit control chain:

`episodes -> policy evaluation -> frozen policy evidence -> policy history -> adaptive routing -> proposal -> execution -> observation`

The history is operational meta-memory, not a learned model and not a causal oracle.

An adaptive loop can optionally require minimum policy stability and maximum policy age before honoring historical consensus. When either gate fails, the loop falls back to the conservative `nearest` mode and records `memory_policy_refresh_recommended=True`, along with `memory_policy_freshness_age`, in proposal v7.

## Controlled policy refresh

A policy can be stable enough to use and still become operationally stale as new episodic experience arrives. PromptForge therefore separates **refresh recommendation** from **refresh execution**.

`ContextRoutingPolicyRefreshController` is a bounded gate with four explicit controls:

- minimum policy observations required before a refresh can execute;
- minimum historical stability;
- maximum allowed policy age relative to the current experience version;
- optional cooldown and refresh budget.

`ContextCognitiveLoop.policy_refresh_decision()` exposes the gate without changing memory. `ContextCognitiveLoop.refresh_memory_routing_policy()` evaluates and records a new policy only when the gate says the refresh is eligible. A successful refresh advances the controller cooldown/budget state through `record_refresh()`.

The resulting lifecycle is:

`experience -> policy health -> refresh gate -> holdout evaluation -> policy history -> adaptive routing`

A blocked refresh is descriptive rather than an error: insufficient evidence, instability, staleness, cooldown, and exhausted budget are surfaced explicitly. The refresh controller never executes a model and never fabricates evidence.

## Active exploration

`ContextExplorationController` adds a bounded exploration layer above routing and policy refresh. It does not infer that an unexplored strategy is better; its purpose is to collect missing evidence when exploration is operationally useful.

The controller considers:

- alternative strategies already available in the current candidate set;
- strategy coverage in observed experience;
- structural novelty;
- an explicit policy-refresh requirement;
- exploration cooldown and exploration budget.

When exploration is eligible, `ContextCognitiveLoop.propose()` can emit `action="probe"` with `source="bounded_exploration"` and an explicit target strategy. `observe()` records the measured outcome and advances the exploration budget state only when that probe strategy was actually observed.

The resulting control chain is:

`policy health -> refresh gate -> exploration gate -> probe -> observe -> policy evaluation`

Exploration is an operational data-collection mechanism. It is not a quality oracle, confidence estimate, causal attribution, or automatic provider executor.

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

Exploration now has a separate evidence-adjudication step. `ContextExplorationAdjudicator` groups observations by episode, identifies probe outcomes, and compares each challenger against the best observed incumbent from the same episode.

An explored strategy is eligible for operational consideration only when the configured evidence gate is met: minimum comparable episodes, minimum family coverage, minimum win rate, and minimum mean relative gain.

`ContextCognitiveLoop.exploration_evidence()` exposes the descriptive challenger evidence, while `exploration_adoption_decision()` exposes the conservative gate. Neither method mutates policy or automatically replaces the incumbent.

This preserves the distinction between:

`probe -> observed outcome -> comparative evidence -> adoption gate`

rather than treating one successful probe as a learned rule.

## Epistemic evidence control

The public core now includes an opt-in `EpistemicContextCompiler` for evidence-heavy tasks where temporal boundaries, contradictory results, provenance, and causal caution matter.

The compiler accepts caller-supplied `EvidenceRecord` items and a deterministic `EpistemicContextPolicy`. It can:

- enforce a hard numeric cutoff and reject required evidence that crosses the boundary;
- exclude unknown-time evidence by default when a cutoff is active;
- preserve explicit contradictions and representation coverage while respecting record/token budgets;
- keep observation, inference, and hypothesis labels separate instead of collapsing them during compression;
- retain source, timestamp, relevance, reliability, intervention factors, and confounder metadata;
- expose descriptive causal cautions for multivariable or confounded interventions.

This layer does not infer truth, causality, confidence, or semantic relevance. Those labels are supplied by the integration and remain auditable metadata.

The handoff is:

`records -> temporal gate -> protected evidence -> deterministic ranking -> budget -> audited context`

Use it as an additive evidence-control layer above the existing structural, memory, trajectory, and cognitive orchestration surfaces.

## Uncertainty-aware action control

`UncertaintyActionGate` adds an opt-in decision boundary for tasks where the best operational action should not be equated with the strongest causal hypothesis.

`ActionCandidate` carries caller-supplied evidence support, reversibility, downside, operational cost, required evidence ids, and an explicit causal-dependence flag.

`UncertaintyActionGate` then:

- fails closed when an action requires evidence outside the current epistemic boundary;
- can require reversibility and cap downside;
- trades evidence support against reversibility, downside, and operational cost with transparent deterministic weights;
- applies only a configurable caution penalty to actions that explicitly depend on an unresolved causal claim;
- returns the full ranking, blocked actions, scores, and reasons for audit.

The gate is a decision aid, not a causal oracle. Action attributes and support values remain caller-supplied.

The intended sequence is:

`evidence boundary -> action feasibility -> risk/reversibility gate -> deterministic ranking -> audited action decision`

### Memory corroboration gate

`ContextMemoryAwareRouter` also supports an opt-in corroboration gate through `min_family_count` and `min_strategy_evidence`. This prevents one isolated episode or one-family cluster from dominating a memory-aware route solely because it is structurally close or recent.

The gate does not discard the nearest-case baseline. When no strategy satisfies the thresholds, the router deterministically falls back to the scored candidate set. The selected thresholds and resulting route remain explicit configuration rather than hidden trust.

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

## Explicit hypothesis ledger

`HypothesisLedger` keeps hypothesis state separate from the evidence that currently supports or contradicts it.

`HypothesisRecord` requires the integration to declare support, contradiction, and required-evidence relationships. The ledger then reports only descriptive states such as `unresolved`, `supported_by_available_evidence`, `challenged_by_available_evidence`, `contested`, and `insufficient_evidence`.

`DiscriminatingExperiment` provides caller-supplied design attributes and a deterministic prioritization score based on hypothesis coverage, isolation, controls, measurement quality, cost, and operational risk. The score is explicitly a prioritization heuristic, not statistical power and not proof of causal identification.

The important boundary is:

`evidence boundary -> hypothesis state -> competing-explanation visibility -> experiment prioritization`

No hypothesis is treated as disproved merely because supporting evidence is absent from the current boundary.


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

Action support anchors can optionally be checked against the stance explicitly declared on each retained `EvidenceRecord`.

1. `ActionPolicy.require_support_stance=True` fails closed unless every declared support anchor has an available epistemic stance compatible with `allowed_support_stances`.
2. The default allowed stance is `supports`; `neutral` and `contradicts` are not admissible unless the integrator explicitly opts into them.
3. Missing stance metadata or an absent epistemic boundary fails closed; PromptForge does not infer stance from evidence content.
4. `ActionDecision.support_evidence_stances` preserves the declared stance mapping for auditability.
5. This is an evidence-admissibility control, not a semantic or causal interpretation of the underlying claim.
6. The default remains backward compatible when the explicit stance gate is disabled.

The strengthened boundary is:

`epistemic evidence -> support anchors -> stance admissibility -> feasibility/safety gates -> action ranking`

## Action evidence relevance guard

Action support anchors can optionally be checked for explicit, caller-supplied relevance tags.

1. `ActionCandidate.support_evidence_tags` declares the tags expected on an action's support evidence.
2. `ActionPolicy.require_support_tag_match=True` fails closed unless every support anchor has at least one matching tag in the current epistemic snapshot.
3. A missing support scope, missing evidence boundary, or absent tag match is blocked rather than inferred from evidence prose.
4. `ActionDecision.support_evidence_tag_matches` preserves the matched tags for auditability.
5. Tag overlap is an explicit admissibility signal, not a semantic relevance oracle, and it does not establish causal sufficiency.
6. The default remains backward compatible when the guard is disabled.

The strengthened boundary is:

`epistemic evidence -> support anchors -> stance admissibility -> explicit relevance-tag admissibility -> feasibility/safety gates -> action ranking`

## Action evidence anchoring

The uncertainty action gate can require explicit evidence anchors for caller-supplied action support scores.

1. `ActionCandidate.support_evidence_ids` records the evidence ids claimed to support the action.
2. `ActionPolicy.require_support_anchors=True` fails closed when an action has no support anchors, when the evidence boundary is absent, or when any declared support id is outside the current evidence snapshot.
3. `min_support_anchors` can require multiple retained anchors without interpreting their count as statistical independence.
4. The resulting `ActionDecision` preserves the support ids for auditability.
5. The default policy remains backward compatible: unanchored `evidence_support` scores remain allowed unless the explicit guard is enabled.

The strengthened boundary is:

`epistemic evidence -> explicit action support anchors -> feasibility/safety gates -> action ranking`

This is an evidence-admissibility safeguard, not a truth score or causal inference mechanism.

## Action support quality guard

Action support anchors can optionally be required to meet explicit evidence-quality floors using the caller-supplied `EvidenceRecord.relevance` and `EvidenceRecord.reliability` metadata.

1. `ActionPolicy.require_support_quality=True` fails closed unless every support anchor has available quality metadata.
2. `min_support_relevance` and `min_support_reliability` set explicit per-anchor floors; defaults are `0.50` when the guard is enabled.
3. An anchor below either floor is blocked even when its stance and relevance tags are otherwise compatible.
4. `ActionDecision.support_evidence_quality` preserves the declared relevance/reliability values for auditability.
5. These values are caller-supplied admissibility metadata, not independently verified truth, causal strength, or statistical power.
6. The default remains backward compatible when the guard is disabled.

The strengthened boundary is:

`epistemic evidence -> support anchors -> stance -> relevance tags -> quality floors -> feasibility/safety gates -> action ranking`

## Action support provenance diversity

Action support anchors can optionally be subjected to a provenance-structure gate.

1. `ActionPolicy.require_support_provenance_diversity=True` requires an explicit current evidence boundary and support anchors.
2. `min_distinct_support_sources` defaults to `2` and counts only explicitly declared, non-`unknown` source labels.
3. Missing source metadata fails closed when the provenance guard is enabled.
4. `max_support_anchors_per_source` can limit concentration from any one declared source.
5. `ActionDecision.support_evidence_provenance` preserves the caller-supplied source labels for auditability.
6. Different source labels do not prove statistical independence, truth, causal sufficiency, or absence of shared upstream dependencies.
7. The guard is opt-in and remains backward compatible when disabled.

The strengthened boundary is:

`epistemic evidence -> support anchors -> stance -> relevance tags -> quality floors -> provenance diversity -> feasibility/safety gates -> action ranking`

## Hypothesis evidence corroboration

Hypothesis assessment can optionally cap contribution from the same declared evidence source.

1. HypothesisEvidencePolicy.max_per_source limits how many support or contradiction records from one declared source can contribute to an assessment.
2. The cap is applied separately to support and contradiction evidence, so a single source can still surface an explicit conflict.
3. Selected evidence ids remain auditable through available_support_ids, available_contradiction_ids, and source_excluded_ids.
4. Unknown or unreported sources can remain isolated per evidence id when isolate_unknown_source=True.
5. The guard is an evidence-inflation control, not a statistical independence test and not a truth estimator.
6. When no policy is supplied, existing evidence-id counting behavior remains unchanged.

The strengthened boundary is:

temporal evidence -> explicit hypothesis mapping -> optional source corroboration gate -> support/contradiction state -> experiment priority

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

## Public core boundary

The installable `promptforge` API is provider-agnostic and self-contained. Its public import surface does not depend on `harness`, provider SDKs, API keys, or network access.

The research harness remains intentionally separate: `harness/` contains fixture APIs, historical runners, provider adapters, preregistration, schedules, reports, and experimental compatibility machinery.

This separation lets the public core evolve without rewriting or importing the frozen research surface.

Read AGENTS.md before making experimental or architectural changes.

## V0.9.3 public evidence

The public release retains the V0.9.3 size-matched control: 4 context variants, 4 conditions, 15 repetitions per variant/condition, and 240 scheduled runs.

The primary comparison is pairs_compact versus noop_padded, where the control is padded to the same serialized context size.

V0.9.3 does not establish a universal economic advantage for one representation. Its purpose is to separate structural representation from serialized context size under the tested conditions.

Relevant files:
harness/prereg/v093_size_matched_control.md
harness/runner/v093_size_matched_control.py
harness/reports/v093_size_matched_control_deepseek.json

## Quick validation

python -m compileall -q harness promptforge tests

python -m unittest discover -s tests -p test_*.py

## Scientific boundary

PromptForge makes context transformations explicit, testable, reproducible, and comparable. Historical results are evidence about the tested conditions, not universal guarantees across every task, model, provider, or prompting strategy.


## 30-second agent path

PromptForge has two public surfaces:

- **Agent core:** provider-agnostic context preparation for real work.
- **Research harness:** reproducible historical experiments, providers, schedules, reports, and preregistration.

When an agent enters the repository, start here:

1. Read `README.md` and `AGENTS.md`.
2. Use `promptforge.prepare_context()` for arbitrary task data.
3. Use `harness.agent.prepare()` only for the public deterministic fixtures.
4. Read historical runners/reports/providers only when the task actually requires them.

## Install

PromptForge has no runtime dependency on an LLM provider.

```bash
pip install .
```

Then:

```python
from promptforge import prepare_context

prepared = prepare_context(
    data={
        "user": {
            "id": "U-7",
            "name": "Ada",
        },
        "task": {
            "action": "review",
            "commentary": "noise",
        },
    },
    required=["user.id", "task.action"],
)

print(prepared["serialized_context"])
```

The core is usable without OpenAI, DeepSeek, Gemini, API keys, network access, or provider configuration.

## What the agent receives

The handoff artifact contains:

- `context`: the selected structured context.
- `serialized_context`: compact JSON ready for downstream use.
- `selected_arm`: transformation selected.
- `required_paths`: semantic fields that must survive.
- `included_paths` / `excluded_paths`: field-level accounting.
- `required_values_preserved` / `validation`: preservation checks.
- `estimated_tokens`: dependency-free planning estimate.
- `context_chars_saved` / `estimated_tokens_saved`: reduction against the `noop` baseline.
- `candidates`: the alternatives considered.
- `provenance`: source and inclusion metadata for required paths.

Token estimates use UTF-8 byte length divided by four and rounded up. They are a planning heuristic, not an exact provider tokenizer measurement.

## Policies

The default policy is deterministic `minimal_serialized_context`: evaluate the public deterministic transformation arms, validate preservation, and select the smallest serialized context.

For real context under a hard budget:

```python
from promptforge import prepare_context, POLICY_BUDGET_CONSTRAINED

prepared = prepare_context(
    data=my_context,
    required=["task.id", "task.priority"],
    policy=POLICY_BUDGET_CONSTRAINED,
    budget_tokens=1200,
)
```

When `budget_tokens` is supplied with the default `minimal` policy, PromptForge automatically switches to `budget_constrained`.

The policy is a mechanical context-size rule. It is not a claim that the smallest context is universally better for model quality.

## Lightweight schema validation

No validation dependency is required for the core schema contract:

```python
prepared = prepare_context(
    data={
        "task": {"id": "T-17"},
        "score": 0.87,
    },
    required=["task.id"],
    schema={
        "task.id": str,
        "score": float,
    },
)
```

Schema paths are also included in the required set, so the validated fields cannot be silently dropped.

## Inspect before preparing

For planning without transformation:

```python
from promptforge import inspect

report = inspect(
    context,
    required=["task.id", "task.priority"],
)
```

This reports field counts, required-field coverage, serialized size, byte size, and estimated tokens without calling any provider.

## Public fixture

For deterministic repository fixtures:

```python
from harness.agent import prepare

prepared = prepare("T003")
```

T003 demonstrates a reduction from 85 to 48 serialized characters (43.5%) while preserving its required fields. This is a small deterministic fixture, not a general token-savings or model-quality guarantee.

## Architecture map

| Area | Purpose | Start here |
| --- | --- | --- |
| `promptforge/` | Installable public API | End-user integration |
| `promptforge/core.py` | Standalone provider-agnostic core | Core behavior |
| `promptforge/adaptive.py` | Complexity-aware structure, local refinement, trajectory, and control | Complex-system orchestration |
| `promptforge/memory.py` | Episodic case memory and topology routing | Learned-from-experience orchestration |
| `promptforge/experience.py` | Mutable write path, frozen snapshots, and memory credit assessment | Online experience boundary |
| `promptforge/consolidation.py` | Deterministic replay, bounded retention, and optional credit-aware selection | Memory lifecycle control |
| `promptforge/credit.py` | Credit/decay bookkeeping and optional credit-aware memory routing | Active memory scoring |
| `promptforge/routing_policy.py` | Leakage-safe comparison and selection of routing modes | Routing policy learning |
| `promptforge/cognitive.py` | Unified observe-decide-learn cycle with explicit memory-routing modes and policy learning | End-to-end adaptive orchestration |
| `promptforge/relational.py` | Explicit cross-link topology and relational descriptors | Relational context structure |
| `harness/agent.py` | Research/fixture agent facade | Fixture reproduction |
| `harness/context.py` | Generic paths, inspection, token estimation, schema checks | Core context utilities |
| `harness/agent_context.py` | Fixture loading and validated compilation | Repository contract |
| `harness/runner/transforms.py` | Transformation machinery | Transformation semantics |
| `tasks/suite.json` | Public deterministic fixtures | Reproduction |
| `harness/prereg/` | Experimental contracts | Research design |
| `harness/reports/` | Retained empirical evidence | Results |
| `harness/runner/` | Historical experiment execution | Reproduction |
| `harness/providers/` | Experimental provider adapters | Provider-specific work |
| `tests/` | Regression and contract tests | Verification |

Provider adapters are experimental infrastructure. They are not required by the core agent API.

## Research surface

The public release retains the V0.9.3 size-matched control: 4 context variants, 4 conditions, 15 repetitions per variant/condition, and 240 scheduled runs.

V0.9.3 does not establish a universal economic advantage for one representation. Its purpose is to separate structural representation from serialized context size under the tested conditions.

Relevant files:

- `harness/prereg/v093_size_matched_control.md`
- `harness/runner/v093_size_matched_control.py`
- `harness/reports/v093_size_matched_control_deepseek.json`

## Scientific boundary

PromptForge makes context transformations explicit, testable, reproducible, and comparable. Historical results are evidence about the tested conditions, not universal guarantees across every task, model, provider, or prompting strategy.

## What PromptForge is not

PromptForge is not a provider-specific SDK, a universal prompt optimizer, or a claim that one representation is always superior.

The public core product is a provider-agnostic context-engineering layer. The research harness exists to make hypotheses, transformations, execution conditions, and historical evidence inspectable and reproducible.


## Decision and triage separation

For triage and operational decision tasks, PromptForge keeps three different questions separate:

1. What evidence is admissible?
2. How should the case be prioritized?
3. What action is safe and sufficiently supported?

The intended control boundary is:

evidence -> temporal/admissibility gate -> priority -> actionability/safety gate -> action -> audit

Priority should distinguish urgency from importance. Evidence quality belongs primarily to the actionability/evidence gate, not as an automatic downward adjustment to urgency or importance. A ticket can therefore be high priority and action-blocked at the same time.

PromptForge also does not define a universal source hierarchy such as system log always wins. Source labels are caller-supplied provenance metadata; contradictions should remain visible until an explicit policy or additional evidence resolves them.

Do not invent precise priority weights or SLA thresholds unless the task or integration supplies them. When illustrative numbers are useful, mark them as proposed policy rather than PromptForge semantics.

For model-facing integrations, keep separate audit objects for priority and actionability. The UncertaintyActionGate belongs to the latter: it is not a ticket-priority scorer.

This boundary is deliberate because collapsing priority and actionability can produce the wrong operational behavior: uncertain evidence may justify blocking an automatic action without making an urgent incident less urgent.
