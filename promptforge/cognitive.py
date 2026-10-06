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
    ContextRoutingPolicyHealth,
    ContextRoutingPolicyHistory,
    ContextRoutingPolicyHistorySnapshot,
)
from .memory import ContextEpisode, ContextRoute, ROUTING_FEATURES
from .relational import ContextRelation, ContextRelationalProfile, ContextRelationProfiler
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
    memory_policy_version: int | None = None
    memory_policy_history_version: int | None = None
    memory_policy_stability_rate: float | None = None
    memory_policy_refresh_recommended: bool | None = None
    trajectory: ContextTrajectoryState | None = None
    candidate_order: tuple[str, ...] = ()

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
        memory_routing_policy_evidence: ContextRoutingPolicyEvidence | None = None,
        memory_routing_policy_history: ContextRoutingPolicyHistory | None = None,
        memory_policy_history_window: int | None = None,
        memory_policy_history_min_observations: int = 1,
        memory_policy_min_stability: float | None = None,
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
        if memory_credit_policy is not None and not isinstance(
            memory_credit_policy, ContextMemoryCreditPolicy
        ):
            raise TypeError(
                "memory_credit_policy must be a ContextMemoryCreditPolicy or None"
            )
        self.scale_mode = scale_mode
        self.memory_routing_mode = memory_routing_mode
        self.memory_top_k = int(memory_top_k)
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
    ) -> ContextCognitiveProposal:
        if not cycle_id:
            raise ValueError("cycle_id must not be empty")

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
        if selected_mode == "adaptive":
            if self.memory_routing_policy_evidence is not None:
                policy_version = self.memory_routing_policy_evidence.version
                selected_mode = self.memory_routing_policy_evidence.selected_mode
            elif self.memory_routing_policy_history is not None:
                history = self.memory_routing_policy_history.snapshot()
                selected_mode = history.select_mode(
                    window=self.memory_policy_history_window,
                    min_observations=self.memory_policy_history_min_observations,
                )
                latest = history.latest
                policy_version = latest.version if latest is not None else None
                policy_history_version = history.version if history.evidences else None
                if self.memory_policy_min_stability is not None:
                    health = history.health(
                        window=self.memory_policy_history_window,
                        min_observations=self.memory_policy_history_min_observations,
                        min_stability=self.memory_policy_min_stability,
                    )
                    policy_stability_rate = health.stability_rate
                    policy_refresh_recommended = health.refresh_recommended
                    if not health.stable:
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

        return ContextCognitiveProposal(
            schema_version="context-cognitive-proposal.v5",
            cycle_id=cycle_id,
            experience_version=evidence.version,
            profile=profile,
            relational_profile=relational_profile,
            decision=decision,
            memory_route=memory_route,
            memory_routing_mode=self.memory_routing_mode,
            memory_routing_selected_mode=selected_mode,
            memory_top_k=self.memory_top_k,
            memory_policy_version=policy_version,
            memory_policy_history_version=policy_history_version,
            memory_policy_stability_rate=policy_stability_rate,
            memory_policy_refresh_recommended=policy_refresh_recommended,
            trajectory=trajectory,
            candidate_order=tuple(
                ranked
                for ranked in self.controller.regime_selector.recommend(profile).preferred_arms
                if ranked in {str(candidate.get("arm_id", "")) for candidate in candidates}
            ),
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
        return self.experience.record(
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


__all__ = [
    "MEMORY_ROUTING_MODES",
    "ContextCognitiveLoop",
    "ContextCognitiveProposal",
    "ContextCognitiveResult",
]
