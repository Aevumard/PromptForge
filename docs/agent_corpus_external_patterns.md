# Agent Corpus: External Architecture Patterns

This document records architecture patterns borrowed from mature open-source agent systems after inspecting their current public repositories on 2026-10-06.

The purpose is not to copy framework semantics into PromptForge. It is to strengthen the agent-facing decision discipline with patterns that are already useful in production-oriented orchestration systems.

## 1. Stateful execution and checkpoints

Reference pattern: LangGraph

LangGraph treats long-running agent execution as explicit stateful workflow execution. Checkpoints preserve graph state, support interruption and resumption, and make human-in-the-loop execution durable. Its documentation also emphasizes putting non-deterministic work and side effects behind resumable task boundaries and keeping those operations idempotent when retries are possible.

PromptForge adoption:
- represent each material decision stage as an explicit state transition;
- persist a decision snapshot at meaningful boundaries;
- keep the evidence boundary and policy version inside the snapshot;
- do not mutate an old decision record when a later observation arrives;
- mark side-effecting actions as retry-sensitive and require idempotency or an explicit duplicate-action policy;
- make human intervention a state transition, not an out-of-band edit.

This is an orchestration pattern, not a claim that PromptForge should become a graph runtime.

## 2. Declarative contracts before implementation

Reference pattern: DSPy

DSPy separates a declarative signature from the implementation used to realize it. Modules can then be composed into larger programs, while optimization is driven by an explicit metric and data.

PromptForge adoption:
- define the required input/output contract before choosing a context strategy;
- keep semantic requirements separate from the wording used to satisfy them;
- compose context stages from explicit contracts rather than free-form prose;
- never optimize a strategy without an explicit evaluation metric;
- keep optimization data separate from the current decision and from held-out evaluation data;
- treat a strategy improvement as an empirical result under a defined task family, not as a universal upgrade.

## 3. Typed state, dependencies, and outputs

Reference pattern: Pydantic AI / pydantic-graph

Pydantic AI uses typed dependencies and validated output contracts, while pydantic-graph models complex control flow as typed nodes and state transitions.

PromptForge adoption:
- every orchestration stage should have explicit input and output fields;
- required paths are part of the semantic contract, not merely a formatting preference;
- state passed between stages must be serializable when persistence or replay is required;
- distinguish construction-time schema checks from runtime evidence validation;
- prefer explicit state transitions over hidden mutable flags.

When a workflow is simple, do not force it into a graph. Use explicit state only where branching, resumption, or auditability materially benefits from it.

## 4. Output validation and bounded repair

Reference pattern: Guardrails

Guardrails separates schema/validator definitions from failure handling and supports bounded corrective behaviors such as deterministic fixes, field filtering, refrain, exception, and re-asking.

PromptForge adoption:
- validate every machine-consumed handoff before the next stage;
- classify failures explicitly as repairable, retryable, blocking, or escalation-worthy;
- prefer deterministic repair before expensive re-generation when the repair is semantics-preserving;
- bound all retries/reasks;
- preserve the failed artifact and validation reasons;
- never keep retrying until a preferred answer appears;
- a validation pass means the structure satisfied the validator, not that the content is true.

For evidence-heavy decisions, repair must never silently rewrite caller-supplied evidence or alter the historical cutoff.

## 5. Retrieval is candidate generation, not evidence admission

Reference pattern: LlamaIndex

LlamaIndex treats context augmentation as a separate layer involving indexes, query engines, agents, workflows, and evaluation/observability.

PromptForge adoption:
- retrieval may generate candidates;
- candidate retrieval does not imply admission into the decision context;
- retrieved material still passes temporal, provenance, task-relevance, contradiction, and policy gates;
- preserve why each retrieved item was admitted or rejected;
- separate retrieval failure from evidence insufficiency;
- never treat retrieval rank as truth or confidence.

The resulting boundary is:

retrieval -> candidate set -> admissibility -> protected context -> decision

## 6. Deterministic and agentic steps must remain distinguishable

Reference pattern: LangGraph

LangGraph explicitly supports combining deterministic logic with agentic steps. PromptForge should use the same separation at the corpus level.

Deterministic responsibilities include:
- temporal cutoff;
- schema validation;
- required-field preservation;
- hard safety gates;
- budget limits;
- state/version checks;
- idempotency checks.

Model responsibilities may include:
- interpretation;
- hypothesis generation;
- language normalization;
- candidate explanation;
- proposing actions within the allowed boundary.

Do not ask a model to decide a property that a deterministic rule can establish reliably.

## 7. Human intervention is a first-class outcome

A blocked action is not an error in the system.

The corpus should distinguish:
- ALLOW — proceed;
- CONDITIONAL — proceed only after specified conditions;
- WAITING_FOR_HUMAN — execution paused for review;
- BLOCKED — action is not permitted with the current evidence;
- REJECTED — proposed action conflicts with policy;
- EXECUTED — action completed;
- OBSERVED — external outcome recorded.

When human review occurs, capture:
- reviewer decision;
- timestamp;
- evidence snapshot/version reviewed;
- changes made;
- resulting policy state.

## 8. Replay, lineage, and reproducibility

A decision record should be replayable without relying on later mutable state.

Minimum lineage:
- decision id;
- parent decision or cycle id;
- context/evidence snapshot identifier;
- cutoff timestamp;
- policy version;
- schema version;
- selected strategy;
- validation results;
- action decision;
- human intervention, when present;
- external outcome, when later observed.

Later observations may improve future routing or calibration, but they must not rewrite the evidence boundary of the original decision.

## 9. Evaluation must be metric-first

Optimization is allowed only when all of the following are explicit:
- target metric;
- comparison baseline;
- task family;
- training/evidence partition;
- evaluation/holdout partition;
- allowed interventions;
- stopping rule.

Never optimize against the same outcome used to claim improvement.

A strategy can be operationally preferred while the measured benefit remains uncertain. Do not convert an optimization score into truth, causality, or confidence without an explicit validated calibration procedure.

## 10. Decision graph for evidence-heavy tasks

For a complex task, the preferred corpus-level sequence is:
1. Ingest — preserve raw inputs and provenance.
2. Normalize — map inputs to typed fields without changing their meaning.
3. Retrieve — collect candidate context when needed.
4. Admit — apply temporal, relevance, source, and policy boundaries.
5. Protect — preserve contradictions and observation/inference/hypothesis distinctions.
6. Validate — check schemas and required fields.
7. Prioritize — compute urgency/importance without conflating actionability.
8. Assess actionability — apply support, stance, relevance, quality, provenance, reversibility, downside, and causal-caution gates.
9. Decide — choose an action or an explicit blocked/escalated state.
10. Checkpoint — persist the exact state required for replay.
11. Execute — perform side effects only after the action gate allows them.
12. Observe — record external outcomes separately from the original evidence.
13. Learn — use observed outcomes only for future routing/calibration under the declared evidence boundary.

The central invariant is:

later outcomes may improve future policy evidence, but they cannot retroactively become evidence for the earlier decision.

## 11. Anti-patterns added to the corpus

Reject these shortcuts:
- one scalar confidence score representing every uncertainty dimension;
- source rank treated as truth;
- retrieval rank treated as evidence quality;
- evidence quality automatically lowering priority;
- actionability used as a proxy for priority;
- an action gate reused as a ticket-ranking function;
- hidden retries;
- unbounded reasks;
- mutation of historical decision state;
- current outcomes leaking into current policy selection;
- optimizer tuning on the same holdout used for evaluation;
- graph complexity introduced without a branching/resume/audit need;
- schema validity interpreted as factual validity.

## 12. What these external projects add to PromptForge

The combined pattern set strengthens PromptForge along five dimensions:

State: explicit lifecycle, checkpoints, replay, and human intervention.

Contracts: typed stage boundaries, required fields, schemas, and serializable state.

Validation: bounded repair/reask behavior with preserved failure evidence.

Retrieval: clear separation between candidate generation and evidence admission.

Evaluation: metric-first optimization with leakage-safe holdouts and explicit baselines.

These patterns complement PromptForge's existing epistemic, temporal, memory, exploration, confidence, and action-control boundaries. They do not replace them.

## 13. Token-efficient context management

Additional external pattern basis: current Pydantic AI / Harness compaction, LangGraph pre-model context hooks, and LlamaIndex token-window helpers.

Observed external pattern:
- long-running agents need explicit context-window management before model calls;
- compaction can trim, clear, or summarize older material while preserving recent/tool-call integrity;
- load-bearing information can be pinned so compaction cannot silently remove it;
- compaction receipts can tell the model that older history is secondhand and should be re-verified;
- token limits belong to the context-management layer rather than being left to a provider failure;
- retrieval/truncation utilities should expose token limits explicitly.

PromptForge adoption:
1. `ContextBudgetPolicy` makes the input-token budget and reserved headroom explicit.
2. `ContextBlock(required=True)` is the pinning boundary: required material cannot be displaced by optional material.
3. Optional blocks are packed deterministically using caller-supplied utility per token cost. PromptForge does not infer semantic utility from prose.
4. `ContextBudgetPlan.compact_manifest()` exposes only a small audit/disclosure manifest for omitted blocks instead of re-injecting their full contents.
5. `ContextBudgetPlan.tokens_saved` and `reduction_ratio` quantify mechanical savings against the supplied baseline.
6. A caller can inject an exact provider tokenizer; otherwise the existing dependency-free byte-based estimate is used.
7. Budget overflow of required information fails closed. PromptForge never silently drops a required field to satisfy a token budget.
8. This layer does not summarize or rewrite evidence. Semantic compression belongs to an explicitly validated upstream/downstream component.

The efficient boundary is:

`inspect -> identify required/pinned blocks -> reserve headroom -> utility-aware packing -> compact omission manifest -> model request`

Token reduction must remain an operational optimization, not a claim that fewer tokens always improves model quality.

## 14. Progressive disclosure for model context

Additional external pattern basis: Pydantic AI's on-demand capabilities and deferred tool loading. Mature agent systems can expose a compact catalog first and reveal the full capability only after the model explicitly requests it.

PromptForge adoption:
- deferred context is represented by stable ids plus compact metadata, not by eagerly sending its values;
- descriptions are caller-supplied and may remain empty;
- the model-facing catalog exposes token cost, kind, path, and description but never the deferred payload;
- an explicit load request returns only the requested blocks, subject to a token/item budget;
- unknown ids and budget omissions remain auditable;
- a single `ContextDeliveryPacket` combines current context with the deferred catalog so an agent can operate without reconstructing internal planning state.

This is progressive disclosure, not autonomous retrieval. PromptForge does not guess what the model wants to load and does not treat catalog descriptions as evidence.

The efficient agent handoff is:

`budget/admission -> current context -> deferred catalog -> explicit load request -> bounded context expansion`

## 15. Tool-output trimming before model calls

Reference pattern: OpenAI Agents Python (`ToolOutputTrimmer`, current public repository inspected 2026-10-06). The implementation applies a model-input filter immediately before model calls, protects recent turns, trims oversized older tool outputs, and preserves tool-call identity.

PromptForge adoption:
1. `ToolOutputItem` gives provider-neutral identity, tool name, content, and turn age.
2. `ToolOutputTrimPolicy` defines a sliding recent-turn protection window and explicit size threshold.
3. Only explicitly eligible tools can be trimmed when the integration supplies an allowlist.
4. The default deterministic transformation preserves both the head and tail of old output, because errors/results often occur at either boundary.
5. `ToolOutputTrimmed` retains original/retained size, estimated token savings, turn metadata, and a SHA-256 digest of the original content.
6. The trimmer never invents a semantic summary and never claims that truncated content was irrelevant.
7. The model-facing representation marks trimmed items explicitly so downstream reasoning knows that a preview replaced the original payload.

The efficient model-call boundary is:

`tool result -> recent-turn protection -> size/tool eligibility gate -> deterministic preview -> digest/audit -> model input`

This complements progressive disclosure: a trimmed result can be re-fetched from the integration using its item/call identity and digest when the full payload is actually needed.

## 16. One-call agent preparation

Reference pattern: modern agent SDKs push context shaping into a model-input boundary immediately before inference. OpenAI Agents exposes input filters/compaction, Pydantic AI exposes model-agnostic compaction, and the previous PromptForge patterns already separate budget planning, deferred context, and tool-output trimming.

PromptForge adoption:
- `prepare_agent_input()` is the single-call fast path for the common integration case;
- it composes context budgeting, progressive disclosure, and tool-output trimming without coupling PromptForge to a provider;
- it returns a compact `AgentInputPacket` for the model and an `AgentPreparation` audit object for the integration;
- no provider call, retrieval call, semantic summarization, or side effect occurs inside this helper;
- the packet exposes total estimated token savings so an integration can inspect the mechanical cost of its context strategy before sending it.

AI-oriented API principle:

`one high-level call for the common path; lower-level deterministic components remain available when the integration needs control.`
