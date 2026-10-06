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

`ContextCognitiveLoop` now exposes two explicit memory-routing modes:

- `nearest` — the original `NearestEpisodeRouter` behavior and the default.
- `credit` — the opt-in `ContextMemoryAwareRouter`, with a bounded `top_k` neighborhood and explicit memory credit.

The selected mode and `top_k` are recorded in `ContextCognitiveProposal`. The proposal schema is versioned because the audit envelope now includes routing-mode state.

```python
from promptforge import ContextCognitiveLoop

loop = ContextCognitiveLoop(
    memory_routing_mode="credit",
    memory_top_k=5,
)
```

This is still provider-agnostic and observational: the credit-aware mode changes how already-observed evidence is aggregated; it does not fabricate outcomes or execute a model.

## Routing-policy learning

PromptForge can compare its routing modes before allowing a future cycle to choose between them.

\`ContextRoutingPolicyEvaluator\` evaluates \`nearest\` and \`credit\` using whole-family holdout folds. Each training fold constructs routing evidence only from the non-held-out families; the held-out family is evaluated afterward.

\`ContextRoutingPolicyEvidence\` freezes those results. \`ContextRoutingPolicySelector\` chooses the mode from observed relative regret, with a deterministic fallback to \`nearest\` when evidence is insufficient.

The cognitive loop accepts \`memory_routing_mode="adaptive"\` plus optional previously evaluated policy evidence:

\`\`\`python
from promptforge import ContextCognitiveLoop

policy = loop.evaluate_memory_routing_policy()
adaptive_loop = ContextCognitiveLoop(
    memory_routing_mode="adaptive",
    memory_routing_policy_evidence=policy,
)
\`\`\`

The current case never contributes its outcome to the policy used for that same proposal. This keeps policy learning temporally separated from decision evidence.

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
| `promptforge/credit.py` | Credit/decay bookkeeping and optional credit-aware memory routing | Active memory scoring |\n| `promptforge/routing_policy.py` | Leakage-safe comparison and selection of routing modes | Routing policy learning |
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
