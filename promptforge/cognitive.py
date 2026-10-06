from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .adaptive import ContextTopologyProfile, ContextTopologyProfiler, ContextTrajectoryState
from .core import POLICY_MINIMAL, prepare_context
from .experience import ContextExperienceSnapshot, ContextExperienceStore
from .credit import ContextMemoryCreditPolicy
from .routing_policy import (
    ROUTING_POLICY_MODES,
    ContextRoutingPolicyEvidence,
    ContextRoutingPolicyEvaluator,
)
from .routing_history import (
    ContextRoutingPolicyHistory,
    ContextRoutingPolicyHistorySnapshot,
)
from .routing_refresh import (
    ContextRoutingPolicyRefreshController,
    ContextRoutingPolicyRefreshDecision,
)
from .exploration import (
    ContextExplorationAdjudicator,
    ContextExplorationController,
    ContextExplorationDecision,
    ContextExplorationAdoptionDecision,
    ContextExplorationStrategyEvidence,
)
from .memory import ContextEpisode, ContextRoute, ROUTING_FEATURES
from .relational import ContextRelation, ContextRelationalProfile, ContextRelationProfiler
from .epistemic import (
    EpistemicContextCompiler,
    EpistemicContextPolicy,
    EpistemicContextResult,
    EvidenceRecord,
)
from .decision import ActionCandidate, ActionDecision, UncertaintyActionGate
from .hypothesis import HypothesisAssessment, HypothesisLedger, HypothesisRecord
from .temporal import ContextMemoryTemporalPolicy
from .orchestration import ComplexContextController, ContextAdaptiveDecision


MEMORY_ROUTING_MODES = ROUTING_POLICY_MODES


@dataclass(frozen=True)
class ContextCognitiveResult:
    """Decision plus the concrete provider-agnostic prepared context."""

    proposal: ContextCognitiveProposal
    prepared: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "prepared": self.prepared,
        }


@dataclass(frozen=True)
class ContextCognitiveProposal:
    """A complete, auditable decision proposal before execution."""

    schema_version: str
    cycle_id: str
    experience_version: int
    profile: ContextTopologyProfile
    relational_profile: ContextRelationalProfile
    decision: ContextAdaptiveDecision
    memory_route: ContextRoute | None
    memory_routing_mode: str = "nearest"
    memory_routing_selected_mode: str = "nearest"
    memory_top_k: int = 5
    memory_min_family_count: int = 1
    memory_min_strategy_evidence: int = 1
    memory_policy_version: int | None = None
    memory_policy_history_version: int | None = None
    memory_policy_stability_rate: float | None = None
    memory_policy_refresh_recommended: bool | None = None
    memory_policy_freshness_age: int | None = None
    memory_policy_refresh_required: bool | None = None
    memory_policy_refresh_reason: str | None = None
    memory_policy_refresh_eligible: bool | None = None
    exploration_required: bool | None = None
    exploration_reason: str | None = None
    exploration_eligible: bool | None = None
    exploration_target_strategy: str | None = None
    exploration_count: int | None = None
    trajectory: ContextTrajectoryState | None = None
    candidate_order: tuple[str, ...] = ()
    epistemic_context: EpistemicContextResult | None = None
    hypothesis_assessments: tuple[HypothesisAssessment, ...] = ()
    action_decision: ActionDecision | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["profile"] = self.profile.to_dict()
        payload["decision"] = self.decision.to_dict()
        payload["memory_route"] = (
            self.memory_route.to_dict() if self.memory_route is not None else None
        )
        payload["trajectory"] = (
            self.trajectory.to_dict() if self.trajectory is not None else None
        )
        payload["candidate_order"] = list(self.candidate_order)
        payload["epistemic_context"] = (
            self.epistemic_context.to_dict()
            if self.epistemic_context is not None
            else None
        )
        payload["hypothesis_assessments"] = [
            item.to_dict() for item in self.hypothesis_assessments
        ]
        payload["action_decision"] = (
            self.action_decision.to_dict()
            if self.action_decision is not None
            else None
        )
        return payload


class ContextCognitiveLoop:
    """Stateful observe-decide-learn loop for provider-agnostic orchestration.

    PromptForge does not execute a model here. It owns structural analysis,
    strategy proposal, and explicit experience write-back. The caller owns
    execution and supplies the measured outcome afterward.

    The loop is intentionally two-phase:
    1. propose() reads a frozen experience snapshot and makes a decision.
    2. observe() records the externally measured outcome for future cycles.

    This keeps online learning explicit while preserving a reproducible
    evidence boundary for every proposal already emitted.
    """

    def __init__(
        self,
        *,
        experience: ContextExperienceStore | None = None,
        profiler: ContextTopologyProfiler | None = None,
        controller: ComplexContextController | None = None,
        routing_features: Sequence[str] | None = None,
        scale_mode: str = "iqr",
        memory_routing_mode: str = "nearest",
        memory_top_k: int = 5,
        memory_credit_policy: ContextMemoryCreditPolicy | None = None,
        memory_min_family_count: int = 1,
        memory_min_strategy_evidence: int = 1,
        memory_temporal_policy: ContextMemoryTemporalPolicy | None = None,
        memory_routing_policy_evidence: ContextRoutingPolicyEvidence | None = None,
        memory_routing_policy_history: ContextRoutingPolicyHistory | None = None,
        memory_policy_history_window: int | None = None,
        memory_policy_history_min_observations: int = 1,
        memory_policy_min_stability: float | None = None,
        memory_policy_max_age: int | None = None,
        memory_policy_refresh_controller: ContextRoutingPolicyRefreshController | None = None,
        exploration_controller: ContextExplorationController | None = None,
        exploration_adjudicator: ContextExplorationAdjudicator | None = None,
        epistemic_compiler: EpistemicContextCompiler | None = None,
        hypothesis_ledger: HypothesisLedger | None = None,
        action_gate: UncertaintyActionGate | None = None,
    ) -> None:
        self.experience = experience or ContextExperienceStore()
        self.profiler = profiler or ContextTopologyProfiler()
        self.controller = controller or ComplexContextController()
        self.relation_profiler = ContextRelationProfiler()
        self.routing_features = (
            tuple(routing_features)
            if routing_features is not None
            else ROUTING_FEATURES
        )
        if memory_routing_mode not in MEMORY_ROUTING_MODES:
            raise ValueError(
                "memory_routing_mode must be one of: "
                + ", ".join(MEMORY_ROUTING_MODES)
            )
        if memory_top_k < 1:
            raise ValueError("memory_top_k must be at least 1")
        if memory_min_family_count < 1:
            raise ValueError("memory_min_family_count must be at least 1")
        if memory_min_strategy_evidence < 1:
            raise ValueError("memory_min_strategy_evidence must be at least 1")
        if memory_credit_policy is not None and not isinstance(
            memory_credit_policy, ContextMemoryCreditPolicy
        ):
            raise TypeError(
                "memory_credit_policy must be a ContextMemoryCreditPolicy or None"
            )
        self.scale_mode = scale_mode
        self.memory_routing_mode = memory_routing_mode
        self.memory_top_k = int(memory_top_k)
        self.memory_min_family_count = int(memory_min_family_count)
        self.memory_min_strategy_evidence = int(memory_min_strategy_evidence)
        if memory_temporal_policy is not None and not isinstance(
            memory_temporal_policy,
            ContextMemoryTemporalPolicy,
        ):
            raise TypeError(
                "memory_temporal_policy must be a ContextMemoryTemporalPolicy or None"
            )
        self.memory_temporal_policy = memory_temporal_policy
        if memory_routing_policy_evidence is not None and not isinstance(
            memory_routing_policy_evidence,
            ContextRoutingPolicyEvidence,
        ):
            raise TypeError(
                "memory_routing_policy_evidence must be "
                "a ContextRoutingPolicyEvidence or None"
            )
        if memory_routing_policy_history is not None and not isinstance(
            memory_routing_policy_history,
            ContextRoutingPolicyHistory,
        ):
            raise TypeError(
                "memory_routing_policy_history must be "
                "a ContextRoutingPolicyHistory or None"
            )
        if (
            memory_policy_history_window is not None
            and memory_policy_history_window < 1
        ):
            raise ValueError("memory_policy_history_window must be at least 1")
        if memory_policy_history_min_observations < 1:
            raise ValueError(
                "memory_policy_history_min_observations must be at least 1"
            )
        if memory_policy_max_age is not None and memory_policy_max_age < 0:
            raise ValueError("memory_policy_max_age must be non-negative")
        if (
            memory_policy_min_stability is not None
            and not 0.0 <= memory_policy_min_stability <= 1.0
        ):
            raise ValueError(
                "memory_policy_min_stability must be between 0.0 and 1.0"
            )
        self.memory_credit_policy = memory_credit_policy
        self.memory_routing_policy_evidence = memory_routing_policy_evidence
        self.memory_routing_policy_history = memory_routing_policy_history
        self.memory_policy_history_window = memory_policy_history_window
        self.memory_policy_history_min_observations = int(
            memory_policy_history_min_observations
        )
        self.memory_policy_min_stability = memory_policy_min_stability
        self.memory_policy_max_age = memory_policy_max_age
        if memory_policy_refresh_controller is not None and not isinstance(
            memory_policy_refresh_controller,
            ContextRoutingPolicyRefreshController,
        ):
            raise TypeError(
                "memory_policy_refresh_controller must be a ContextRoutingPolicyRefreshController or None"
            )
        self.memory_policy_refresh_controller = (
            memory_policy_refresh_controller
            if memory_policy_refresh_controller is not None
            else ContextRoutingPolicyRefreshController(
                min_observations=memory_policy_history_min_observations,
                min_stability=(
                    0.0
                    if memory_policy_min_stability is None
                    else memory_policy_min_stability
                ),
                max_age=memory_policy_max_age,
            )
        )
        if exploration_controller is not None and not isinstance(
            exploration_controller,
            ContextExplorationController,
        ):
            raise TypeError(
                "exploration_controller must be a ContextExplorationController or None"
            )
        self.exploration_controller = exploration_controller
        if exploration_adjudicator is not None and not isinstance(
            exploration_adjudicator,
            ContextExplorationAdjudicator,
        ):
            raise TypeError(
                "exploration_adjudicator must be a ContextExplorationAdjudicator or None"
            )
        self.exploration_adjudicator = (
            exploration_adjudicator
            if exploration_adjudicator is not None
            else ContextExplorationAdjudicator()
        )
        self.epistemic_compiler = epistemic_compiler or EpistemicContextCompiler()
        self.hypothesis_ledger = hypothesis_ledger or HypothesisLedger()
        self.action_gate = action_gate or UncertaintyActionGate()

    def snapshot(self) -> ContextExperienceSnapshot:
        """Return the current immutable evidence boundary."""
        return self.experience.snapshot()

    def propose(
        self,
        *,
        cycle_id: str,
        data: Mapping[str, Any],
        required: Sequence[str],
        candidates: Sequence[Mapping[str, Any]],
        relations: Sequence[ContextRelation | Mapping[str, Any]] = (),
        trajectory: ContextTrajectoryState | None = None,
        current_strategy: str | None = None,
        current_cost: float | None = None,
        probe_scores: Mapping[str, float] | None = None,
        remaining_budget_fraction: float = 1.0,
        budget_tokens: int | None = None,
        evidence: Sequence[EvidenceRecord | Mapping[str, Any]] = (),
        epistemic_policy: EpistemicContextPolicy | None = None,
        hypotheses: Sequence[HypothesisRecord] = (),
        action_candidates: Sequence[ActionCandidate] = (),
    ) -> ContextCognitiveProposal:
        if not cycle_id:
            raise ValueError("cycle_id must not be empty")

        epistemic_context = (
            self.epistemic_compiler.compile(evidence, policy=epistemic_policy)
            if evidence
            else None
        )
        hypothesis_assessments = (
            self.hypothesis_ledger.assess(
                hypotheses,
                evidence=epistemic_context,
            )
            if hypotheses
            else ()
        )
        action_decision = (
            self.action_gate.decide(
                action_candidates,
                evidence=epistemic_context,
            )
            if action_candidates
            else None
        )

        profile = self.profiler.profile(data, required)
        relational_profile = self.relation_profiler.profile(relations)
        evidence = self.experience.snapshot()
        routing_topology = profile.to_dict()
        routing_topology.update(relational_profile.to_dict())
        selected_mode = self.memory_routing_mode
        policy_version = None
        policy_history_version = None
        policy_stability_rate = None
        policy_refresh_recommended = None
        policy_freshness_age = None
        policy_refresh_required = None
        policy_refresh_reason = None
        policy_refresh_eligible = None
        exploration_required = None
        exploration_reason = None
        exploration_eligible = None
        exploration_target_strategy = None
        exploration_count = None
        if selected_mode == "adaptive":
            if self.memory_routing_policy_evidence is not None:
                policy_version = self.memory_routing_policy_evidence.version
                selected_mode = self.memory_routing_policy_evidence.selected_mode
                refresh_decision = self.memory_policy_refresh_controller.decide_for_policy(
                    policy_version=policy_version,
                    experience_version=evidence.version,
                )
                policy_freshness_age = refresh_decision.freshness_age
                policy_refresh_recommended = refresh_decision.refresh_required
                policy_refresh_required = refresh_decision.refresh_required
                policy_refresh_reason = refresh_decision.reason
                policy_refresh_eligible = refresh_decision.eligible
                if refresh_decision.refresh_required:
                    selected_mode = "nearest"
            elif self.memory_routing_policy_history is not None:
                history = self.memory_routing_policy_history.snapshot()
                selected_mode = history.select_mode(
                    window=self.memory_policy_history_window,
                    min_observations=self.memory_policy_history_min_observations,
                )
                latest = history.latest
                policy_version = latest.version if latest is not None else None
                policy_history_version = history.version if history.evidences else None
                if (
                    self.memory_policy_min_stability is not None
                    or self.memory_policy_max_age is not None
                ):
                    health = history.health(
                        window=self.memory_policy_history_window,
                        min_observations=self.memory_policy_history_min_observations,
                        min_stability=(
                            0.0
                            if self.memory_policy_min_stability is None
                            else self.memory_policy_min_stability
                        ),
                        current_experience_version=evidence.version,
                        max_age=self.memory_policy_max_age,
                    )
                    policy_stability_rate = health.stability_rate
                    policy_freshness_age = health.freshness_age
                    policy_refresh_recommended = health.refresh_recommended
                    refresh_decision = self.memory_policy_refresh_controller.decide(
                        history,
                        experience_version=evidence.version,
                    )
                    policy_refresh_required = refresh_decision.refresh_required
                    policy_refresh_reason = refresh_decision.reason
                    policy_refresh_eligible = refresh_decision.eligible
                    if not health.healthy or refresh_decision.refresh_required:
                        selected_mode = "nearest"
            else:
                selected_mode = "nearest"

        if evidence.episodes:
            if selected_mode == "credit":
                memory_route = evidence.memory_aware_route(
                    routing_topology,
                    features=self.routing_features,
                    scale_mode=self.scale_mode,
                    top_k=self.memory_top_k,
                    policy=self.memory_credit_policy,
                    min_family_count=self.memory_min_family_count,
                    min_strategy_evidence=self.memory_min_strategy_evidence,
                    temporal_policy=self.memory_temporal_policy,
                )
            else:
                memory_route = evidence.route(
                    routing_topology,
                    features=self.routing_features,
                    scale_mode=self.scale_mode,
                )
        else:
            memory_route = None

        decision = self.controller.decide(
            profile=profile,
            candidates=candidates,
            memory_route=memory_route,
            trajectory=trajectory,
            current_strategy=current_strategy,
            current_cost=current_cost,
            probe_scores=probe_scores,
            remaining_budget_fraction=remaining_budget_fraction,
            budget_tokens=budget_tokens,
        )

        candidate_order = tuple(
            ranked
            for ranked in self.controller.regime_selector.recommend(profile).preferred_arms
            if ranked in {str(candidate.get("arm_id", "")) for candidate in candidates}
        )

        if self.exploration_controller is not None:
            observed_strategies = tuple(
                episode.strategy for episode in evidence.episodes
            )
            exploration = self.exploration_controller.decide(
                experience_version=evidence.version,
                candidate_order=candidate_order,
                observed_strategies=observed_strategies,
                preferred_strategy=decision.strategy,
                novelty_distance=decision.novelty_distance,
                policy_refresh_required=bool(policy_refresh_required),
            )
            exploration_required = exploration.required
            exploration_reason = exploration.reason
            exploration_eligible = exploration.eligible
            exploration_target_strategy = exploration.target_strategy
            exploration_count = exploration.exploration_count

            if exploration.eligible and exploration.target_strategy is not None:
                target = exploration.target_strategy
                decision = ContextAdaptiveDecision(
                    schema_version=decision.schema_version,
                    action="probe",
                    strategy=target,
                    source="bounded_exploration",
                    regime=decision.regime,
                    regime_flags=decision.regime_flags,
                    novelty_distance=decision.novelty_distance,
                    reason=exploration.reason,
                )

        return ContextCognitiveProposal(
            schema_version="context-cognitive-proposal.v8",
            cycle_id=cycle_id,
            experience_version=evidence.version,
            profile=profile,
            relational_profile=relational_profile,
            decision=decision,
            memory_route=memory_route,
            memory_routing_mode=self.memory_routing_mode,
            memory_routing_selected_mode=selected_mode,
            memory_top_k=self.memory_top_k,
            memory_min_family_count=self.memory_min_family_count,
            memory_min_strategy_evidence=self.memory_min_strategy_evidence,
            memory_policy_version=policy_version,
            memory_policy_history_version=policy_history_version,
            memory_policy_stability_rate=policy_stability_rate,
            memory_policy_refresh_recommended=policy_refresh_recommended,
            memory_policy_freshness_age=policy_freshness_age,
            memory_policy_refresh_required=policy_refresh_required,
            memory_policy_refresh_reason=policy_refresh_reason,
            memory_policy_refresh_eligible=policy_refresh_eligible,
            exploration_required=exploration_required,
            exploration_reason=exploration_reason,
            exploration_eligible=exploration_eligible,
            exploration_target_strategy=exploration_target_strategy,
            exploration_count=exploration_count,
            trajectory=trajectory,
            candidate_order=candidate_order,
            epistemic_context=epistemic_context,
            hypothesis_assessments=tuple(hypothesis_assessments),
            action_decision=action_decision,
        )

    def evaluate_memory_routing_policy(
        self,
        *,
        version: int | None = None,
        top_k: int | None = None,
    ) -> ContextRoutingPolicyEvidence:
        """Evaluate routing modes on a frozen experience snapshot."""
        snapshot = self.experience.snapshot()
        evaluator = ContextRoutingPolicyEvaluator(
            features=self.routing_features,
            scale_mode=self.scale_mode,
            top_k=(
                self.memory_top_k
                if top_k is None
                else top_k
            ),
            credit_policy=self.memory_credit_policy,
        )
        return evaluator.evaluate(
            snapshot.episodes,
            version=version if version is not None else snapshot.version,
        )

    def policy_history_snapshot(self) -> ContextRoutingPolicyHistorySnapshot:
        """Return the current immutable routing-policy meta-memory snapshot."""
        if self.memory_routing_policy_history is None:
            return ContextRoutingPolicyHistorySnapshot(
                version=0,
                evidences=(),
            )
        return self.memory_routing_policy_history.snapshot()

    def record_memory_routing_policy(
        self,
        evidence: ContextRoutingPolicyEvidence,
    ) -> int:
        """Persist policy evidence in the explicit meta-memory store."""
        if self.memory_routing_policy_history is None:
            raise ValueError(
                "memory_routing_policy_history must be configured"
            )
        return self.memory_routing_policy_history.record(evidence)

    def evaluate_and_record_memory_routing_policy(
        self,
        *,
        version: int | None = None,
        top_k: int | None = None,
    ) -> ContextRoutingPolicyEvidence:
        """Evaluate routing policy on a frozen snapshot and record the evidence."""
        evidence = self.evaluate_memory_routing_policy(
            version=version,
            top_k=top_k,
        )
        self.record_memory_routing_policy(evidence)
        return evidence

    def exploration_evidence(
        self,
    ) -> tuple[ContextExplorationStrategyEvidence, ...]:
        """Assess comparable exploration outcomes on the current frozen snapshot."""
        snapshot = self.experience.snapshot()
        return self.exploration_adjudicator.assess(snapshot.episodes)

    def exploration_adoption_decision(
        self,
        *,
        preferred_strategy: str | None = None,
    ) -> ContextExplorationAdoptionDecision:
        """Return the conservative challenger-adoption gate for current evidence."""
        snapshot = self.experience.snapshot()
        return self.exploration_adjudicator.decide(
            snapshot.episodes,
            preferred_strategy=preferred_strategy,
        )

    def policy_refresh_decision(
        self,
    ) -> ContextRoutingPolicyRefreshDecision | None:
        """Return the current refresh gate without executing a refresh."""
        snapshot = self.experience.snapshot()
        if self.memory_routing_policy_history is not None:
            return self.memory_policy_refresh_controller.decide(
                self.memory_routing_policy_history.snapshot(),
                experience_version=snapshot.version,
            )
        if self.memory_routing_policy_evidence is not None:
            return self.memory_policy_refresh_controller.decide_for_policy(
                policy_version=self.memory_routing_policy_evidence.version,
                experience_version=snapshot.version,
            )
        return None

    def refresh_memory_routing_policy(
        self,
        *,
        top_k: int | None = None,
        version: int | None = None,
    ) -> ContextRoutingPolicyEvidence | None:
        """Refresh routing policy only when the explicit gate permits it."""
        if self.memory_routing_policy_history is None:
            raise ValueError(
                "memory_routing_policy_history must be configured"
            )
        decision = self.policy_refresh_decision()
        if decision is None or not decision.eligible:
            return None

        snapshot = self.experience.snapshot()
        evidence = self.evaluate_memory_routing_policy(
            version=version if version is not None else snapshot.version,
            top_k=top_k,
        )
        self.record_memory_routing_policy(evidence)
        self.memory_policy_refresh_controller.record_refresh(
            experience_version=snapshot.version,
        )
        return evidence

    def prepare(
        self,
        *,
        cycle_id: str,
        data: Mapping[str, Any],
        required: Sequence[str],
        candidates: Sequence[Mapping[str, Any]] | None = None,
        relations: Sequence[ContextRelation | Mapping[str, Any]] = (),
        task_id: str = "adhoc",
        task_family: str = "agent_request",
        budget_tokens: int | None = None,
        trajectory: ContextTrajectoryState | None = None,
        current_strategy: str | None = None,
        current_cost: float | None = None,
        probe_scores: Mapping[str, float] | None = None,
        remaining_budget_fraction: float = 1.0,
        evidence: Sequence[EvidenceRecord | Mapping[str, Any]] = (),
        epistemic_policy: EpistemicContextPolicy | None = None,
        hypotheses: Sequence[HypothesisRecord] = (),
        action_candidates: Sequence[ActionCandidate] = (),
    ) -> ContextCognitiveResult:
        """Plan and materialize the selected context preparation in one call."""
        if candidates is None:
            probe = prepare_context(
                data,
                required,
                task_id=task_id,
                task_family=task_family,
                policy=POLICY_MINIMAL,
                budget_tokens=budget_tokens,
            )
            candidates = probe["candidates"]

        proposal = self.propose(
            cycle_id=cycle_id,
            data=data,
            required=required,
            candidates=candidates,
            relations=relations,
            trajectory=trajectory,
            current_strategy=current_strategy,
            current_cost=current_cost,
            probe_scores=probe_scores,
            remaining_budget_fraction=remaining_budget_fraction,
            budget_tokens=budget_tokens,
            evidence=evidence,
            epistemic_policy=epistemic_policy,
            hypotheses=hypotheses,
            action_candidates=action_candidates,
        )
        if proposal.decision.action == "stop":
            raise ValueError(
                "cognitive decision is stop; no preparation was materialized"
            )

        prepared = prepare_context(
            data,
            required,
            task_id=task_id,
            task_family=task_family,
            policy=POLICY_MINIMAL,
            arm_id=proposal.decision.strategy,
            budget_tokens=budget_tokens,
        )
        if proposal.epistemic_context is not None:
            prepared["epistemic_context"] = proposal.epistemic_context.to_dict()
        if proposal.hypothesis_assessments:
            prepared["hypothesis_assessments"] = [
                item.to_dict() for item in proposal.hypothesis_assessments
            ]
        if proposal.action_decision is not None:
            prepared["action_decision"] = proposal.action_decision.to_dict()
        return ContextCognitiveResult(proposal=proposal, prepared=prepared)

    def observe(
        self,
        proposal: ContextCognitiveProposal,
        *,
        family_id: str,
        cost: float,
        strategy: str | None = None,
        outcome: Mapping[str, Any] | None = None,
    ) -> int:
        """Record the externally measured outcome of a prior proposal."""
        if not isinstance(proposal, ContextCognitiveProposal):
            raise TypeError("proposal must be a ContextCognitiveProposal")
        selected_strategy = strategy or proposal.decision.strategy
        version = self.experience.record(
            ContextEpisode(
                episode_id=proposal.cycle_id,
                family_id=family_id,
                strategy=selected_strategy,
                cost=cost,
                topology={
                    **proposal.profile.to_dict(),
                    **proposal.relational_profile.to_dict(),
                },
                action=proposal.decision.action,
                source=proposal.decision.source,
                regime=proposal.decision.regime,
                novelty_distance=proposal.decision.novelty_distance,
                trajectory_state=(
                    proposal.trajectory.state
                    if proposal.trajectory is not None
                    else None
                ),
                trajectory=(
                    proposal.trajectory.to_dict()
                    if proposal.trajectory is not None
                    else {}
                ),
                regime_flags=proposal.decision.regime_flags,
                decision_reason=proposal.decision.reason,
                candidate_order=proposal.candidate_order,
                outcome=dict(outcome or {}),
            )
        )
        if (
            self.exploration_controller is not None
            and proposal.decision.source == "bounded_exploration"
            and selected_strategy == proposal.exploration_target_strategy
        ):
            self.exploration_controller.record_exploration(
                experience_version=version,
            )
        return version


__all__ = [
    "MEMORY_ROUTING_MODES",
    "ContextCognitiveLoop",
    "ContextCognitiveProposal",
    "ContextCognitiveResult",
]
