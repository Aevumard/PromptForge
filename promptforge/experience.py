from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .adaptive import ContextTopologyProfile
from .consolidation import ContextMemoryConsolidator, ContextReplayBatch
from .credit import (
    ContextMemoryAwareRoute,
    ContextMemoryAwareRouter,
    ContextMemoryCredit,
    ContextMemoryCreditPolicy,
)
from .memory import ContextEpisode, ContextRoute, NearestEpisodeRouter
from .temporal import ContextMemoryTemporalPolicy


@dataclass(frozen=True)
class ContextExperienceSnapshot:
    """Immutable experience boundary for one routing/evaluation cycle."""

    episodes: tuple[ContextEpisode, ...]
    version: int

    def router(
        self,
        *,
        features: Sequence[str] | None = None,
        scale_mode: str = "iqr",
    ) -> NearestEpisodeRouter:
        if not self.episodes:
            raise ValueError("snapshot contains no episodes")
        return NearestEpisodeRouter(
            features=features,
            scale_mode=scale_mode,
        ).fit(self.episodes)

    def route(
        self,
        topology: Mapping[str, Any] | ContextTopologyProfile,
        *,
        features: Sequence[str] | None = None,
        scale_mode: str = "iqr",
    ) -> ContextRoute:
        return self.router(
            features=features,
            scale_mode=scale_mode,
        ).route(topology)

    def memory_aware_route(
        self,
        topology: Mapping[str, Any] | ContextTopologyProfile,
        *,
        features: Sequence[str] | None = None,
        scale_mode: str = "iqr",
        top_k: int = 5,
        policy: ContextMemoryCreditPolicy | None = None,
        min_family_count: int = 1,
        min_strategy_evidence: int = 1,
        temporal_policy: ContextMemoryTemporalPolicy | None = None,
        as_of_index: int | None = None,
    ) -> ContextMemoryAwareRoute:
        """Route using credit-weighted nearby evidence inside this snapshot."""
        router = ContextMemoryAwareRouter(
            features=features,
            scale_mode=scale_mode,
            top_k=top_k,
            credit_policy=policy,
            min_family_count=min_family_count,
            min_strategy_evidence=min_strategy_evidence,
            temporal_policy=temporal_policy,
        ).fit(self.episodes)
        return router.route(topology, as_of_index=as_of_index)

    def credit(
        self,
        *,
        policy: ContextMemoryCreditPolicy | None = None,
    ) -> tuple[ContextMemoryCredit, ...]:
        """Assess memory credit inside this frozen evidence boundary."""
        if policy is not None and not isinstance(policy, ContextMemoryCreditPolicy):
            raise TypeError(
                "policy must be a ContextMemoryCreditPolicy or None"
            )
        assessor = policy or ContextMemoryCreditPolicy()
        return assessor.assess(self.episodes)


class ContextExperienceStore:
    """Small auditable experience system for the adaptive decision loop.

    The store separates two moments that must not be confused:
    - recording an observed outcome;
    - taking a frozen snapshot used for routing/evaluation.

    A snapshot never changes when new episodes are recorded, so an evaluation
    can freeze its evidence boundary before generating predictions.
    """

    def __init__(self, episodes: Sequence[ContextEpisode] = ()) -> None:
        self._episodes: list[ContextEpisode] = []
        self._version = 0
        for episode in episodes:
            self.record(episode)

    @property
    def version(self) -> int:
        return self._version

    @property
    def size(self) -> int:
        return len(self._episodes)

    def record(self, episode: ContextEpisode) -> int:
        if not isinstance(episode, ContextEpisode):
            raise TypeError("episode must be a ContextEpisode")
        self._episodes.append(episode)
        self._version += 1
        return self._version

    def snapshot(self) -> ContextExperienceSnapshot:
        return ContextExperienceSnapshot(
            episodes=tuple(self._episodes),
            version=self._version,
        )

    def route(
        self,
        topology: Mapping[str, Any] | ContextTopologyProfile,
        *,
        features: Sequence[str] | None = None,
        scale_mode: str = "iqr",
    ) -> ContextRoute:
        return self.snapshot().route(
            topology,
            features=features,
            scale_mode=scale_mode,
        )

    def memory_aware_route(
        self,
        topology: Mapping[str, Any] | ContextTopologyProfile,
        *,
        features: Sequence[str] | None = None,
        scale_mode: str = "iqr",
        top_k: int = 5,
        policy: ContextMemoryCreditPolicy | None = None,
        min_family_count: int = 1,
        min_strategy_evidence: int = 1,
        temporal_policy: ContextMemoryTemporalPolicy | None = None,
        as_of_index: int | None = None,
    ) -> ContextMemoryAwareRoute:
        """Route current memory using an immutable evidence boundary."""
        return self.snapshot().memory_aware_route(
            topology,
            features=features,
            scale_mode=scale_mode,
            top_k=top_k,
            policy=policy,
            min_family_count=min_family_count,
            min_strategy_evidence=min_strategy_evidence,
            temporal_policy=temporal_policy,
            as_of_index=as_of_index,
        )

    def credit(
        self,
        *,
        policy: ContextMemoryCreditPolicy | None = None,
    ) -> tuple[ContextMemoryCredit, ...]:
        """Assess current memory using a frozen snapshot boundary."""
        return self.snapshot().credit(policy=policy)

    def consolidate(
        self,
        *,
        max_episodes: int,
        consolidator: ContextMemoryConsolidator | None = None,
        credit_policy: ContextMemoryCreditPolicy | None = None,
    ) -> ContextReplayBatch:
        """Retain bounded memory while preserving prior snapshots."""
        policy = consolidator or ContextMemoryConsolidator(
            credit_policy=credit_policy
        )
        batch = policy.replay(
            self._episodes,
            max_items=max_episodes,
            credit_policy=credit_policy,
        )
        if len(batch.episodes) != len(self._episodes):
            self._episodes = list(batch.episodes)
            self._version += 1
        return batch

    def clear(self) -> None:
        self._episodes.clear()
        self._version += 1


__all__ = [
    "ContextExperienceSnapshot",
    "ContextExperienceStore",
    "ContextReplayBatch",
]
