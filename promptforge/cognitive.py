from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .adaptive import ContextTopologyProfile, ContextTopologyProfiler, ContextTrajectoryState
from .core import POLICY_MINIMAL, prepare_context
from .experience import ContextExperienceSnapshot, ContextExperienceStore
from .memory import ContextEpisode, ContextRoute, ROUTING_FEATURES
from .relational import ContextRelation, ContextRelationalProfile, ContextRelationProfiler
from .orchestration import ComplexContextController, ContextAdaptiveDecision


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

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["profile"] = self.profile.to_dict()
        payload["decision"] = self.decision.to_dict()
        payload["memory_route"] = (
            self.memory_route.to_dict() if self.memory_route is not None else None
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
        self.scale_mode = scale_mode

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
        memory_route = (
            evidence.route(
                routing_topology,
                features=self.routing_features,
                scale_mode=self.scale_mode,
            )
            if evidence.episodes
            else None
        )

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
            )
        )


__all__ = [
    "ContextCognitiveLoop",
    "ContextCognitiveProposal",
    "ContextCognitiveResult",
]
