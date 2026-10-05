from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .adaptive import (
    ContextPortfolioController,
    ContextRegimeSignature,
    ContextStrategyRecommendation,
    ContextTopologyProfile,
    ContextTrajectoryState,
    HeuristicContextRegimeSelector,
    rank_context_candidates,
)
from .memory import ContextRoute


@dataclass(frozen=True)
class ContextAdaptiveDecision:
    """Unified decision envelope across structure, memory, trajectory, and control."""

    schema_version: str
    action: str
    strategy: str
    source: str
    regime: str
    regime_flags: tuple[str, ...]
    novelty_distance: float | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ComplexContextController:
    """Combine structural routing, episodic memory, and trajectory control.

    Precedence:
    1. observed trajectory/probe evidence can force a transition;
    2. a non-novel episodic route can supply the strategy;
    3. otherwise the transparent structural recommendation supplies the strategy.

    Novelty is a relative distance, not a calibrated confidence score.
    """

    def __init__(
        self,
        *,
        regime_selector: HeuristicContextRegimeSelector | None = None,
        portfolio_controller: ContextPortfolioController | None = None,
        novelty_threshold: float = 3.0,
    ) -> None:
        if novelty_threshold < 0.0:
            raise ValueError("novelty_threshold must be non-negative")
        self.regime_selector = (
            regime_selector or HeuristicContextRegimeSelector()
        )
        self.portfolio_controller = (
            portfolio_controller or ContextPortfolioController()
        )
        self.novelty_threshold = novelty_threshold

    def decide(
        self,
        *,
        profile: ContextTopologyProfile,
        candidates: Sequence[Mapping[str, Any]],
        memory_route: ContextRoute | None = None,
        trajectory: ContextTrajectoryState | None = None,
        current_strategy: str | None = None,
        current_cost: float | None = None,
        probe_scores: Mapping[str, float] | None = None,
        remaining_budget_fraction: float = 1.0,
        budget_tokens: int | None = None,
    ) -> ContextAdaptiveDecision:
        signature: ContextRegimeSignature = self.regime_selector.classify(profile)
        recommendation: ContextStrategyRecommendation = (
            self.regime_selector.recommend(profile)
        )
        ranked = rank_context_candidates(
            candidates,
            recommendation,
            budget_tokens=budget_tokens,
        )
        if not ranked:
            raise ValueError("no feasible candidates supplied")

        candidate_ids = set(ranked)

        if trajectory is not None or current_strategy is not None:
            if trajectory is None or current_strategy is None:
                raise ValueError(
                    "trajectory and current_strategy must be supplied together"
                )
            if current_cost is None:
                raise ValueError(
                    "current_cost is required when trajectory control is used"
                )

            control = self.portfolio_controller.decide(
                trajectory=trajectory,
                current_strategy=current_strategy,
                current_cost=float(current_cost),
                probe_scores=probe_scores,
                remaining_budget_fraction=remaining_budget_fraction,
            )
            if control.action in {"switch", "continue", "intensify", "stop"}:
                target = control.target_strategy or current_strategy
                if target in candidate_ids:
                    return ContextAdaptiveDecision(
                        schema_version="context-adaptive.v1",
                        action=control.action,
                        strategy=target,
                        source="trajectory_control",
                        regime=signature.primary_regime,
                        regime_flags=signature.flags,
                        novelty_distance=(
                            memory_route.nearest_distance
                            if memory_route is not None
                            else None
                        ),
                        reason=control.reason,
                    )

        if (
            memory_route is not None
            and memory_route.strategy in candidate_ids
            and memory_route.nearest_distance <= self.novelty_threshold
        ):
            return ContextAdaptiveDecision(
                schema_version="context-adaptive.v1",
                action="select",
                strategy=memory_route.strategy,
                source="episodic_memory",
                regime=signature.primary_regime,
                regime_flags=signature.flags,
                novelty_distance=memory_route.nearest_distance,
                reason=(
                    "nearest observed episode is within the configured novelty "
                    "threshold"
                ),
            )

        strategy = next(
            (arm for arm in ranked if arm in candidate_ids),
            ranked[0],
        )
        reason = (
            "episodic memory was too novel or unavailable; "
            "using the structural candidate ranking"
        )
        return ContextAdaptiveDecision(
            schema_version="context-adaptive.v1",
            action="select",
            strategy=strategy,
            source="structural_regime",
            regime=signature.primary_regime,
            regime_flags=signature.flags,
            novelty_distance=(
                memory_route.nearest_distance
                if memory_route is not None
                else None
            ),
            reason=reason,
        )


__all__ = [
    "ComplexContextController",
    "ContextAdaptiveDecision",
]
