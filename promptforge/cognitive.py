from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .adaptive import ContextTopologyProfile, ContextTopologyProfiler, ContextTrajectoryState
from .core import POLICY_MINIMAL, prepare_context
from .experience import ContextExperienceSnapshot, ContextExperienceStore
from .credit import ContextMemoryCreditPolicy
from .memory import ContextEpisode, ContextRoute, ROUTING_FEATURES
from .relational import ContextRelation, ContextRelationalProfile, ContextRelationProfiler
from .orchestration import ComplexContextController, ContextAdaptiveDecision


MEMORY_ROUTING_MODES = ("nearest", "credit")


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
    memory_top_k: int = 5
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
        self.memory_credit_policy = memory_credit_policy

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
        if evidence.episodes:
            if self.memory_routing_mode == "credit":
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
            schema_version="context-cognitive-proposal.v1",
            cycle_id=cycle_id,
            experience_version=evidence.version,
            profile=profile,
            relational_profile=relational_profile,
            decision=decision,
            memory_route=memory_route,
            trajectory=trajectory,
            candidate_order=tuple(
                ranked
                for ranked in self.controller.regime_selector.recommend(profile).preferred_arms
                if ranked in {str(candidate.get("arm_id", "")) for candidate in candidates}
            ),
        )

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
