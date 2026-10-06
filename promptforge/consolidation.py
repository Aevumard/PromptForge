from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Sequence

from .credit import ContextMemoryCreditPolicy
from .memory import ContextEpisode


@dataclass(frozen=True)
class ContextReplayBatch:
    """Deterministic subset of historical episodes for replay or refitting."""

    episodes: tuple[ContextEpisode, ...]
    source_size: int
    family_count: int
    strategy_count: int

    def to_dict(self) -> dict:
        return {
            "episodes": [episode.to_dict() for episode in self.episodes],
            "source_size": self.source_size,
            "family_count": self.family_count,
            "strategy_count": self.strategy_count,
        }


class ContextMemoryConsolidator:
    """Bounded, deterministic memory consolidation with explicit forgetting.

    Retention keeps recent observations first, then maximizes observed
    family/strategy coverage. An optional memory-credit policy supplies a
    deterministic tie-break based on recency, repeated evidence, stability,
    and within-episode observed comparisons.
    """

    def __init__(
        self,
        *,
        recent_fraction: float = 0.50,
        credit_policy: ContextMemoryCreditPolicy | None = None,
    ) -> None:
        if not 0.0 < recent_fraction <= 1.0:
            raise ValueError("recent_fraction must be in (0, 1]")
        if credit_policy is not None and not isinstance(
            credit_policy, ContextMemoryCreditPolicy
        ):
            raise TypeError(
                "credit_policy must be a ContextMemoryCreditPolicy or None"
            )
        self.recent_fraction = recent_fraction
        self.credit_policy = credit_policy

    @staticmethod
    def _validate(
        episodes: Sequence[ContextEpisode],
    ) -> tuple[ContextEpisode, ...]:
        values = tuple(episodes)
        if any(not isinstance(item, ContextEpisode) for item in values):
            raise TypeError("episodes must contain only ContextEpisode values")
        return values

    def replay(
        self,
        episodes: Sequence[ContextEpisode],
        *,
        max_items: int,
        credit_policy: ContextMemoryCreditPolicy | None = None,
    ) -> ContextReplayBatch:
        values = self._validate(episodes)
        if max_items < 1:
            raise ValueError("max_items must be at least 1")
        if not values:
            return ContextReplayBatch((), 0, 0, 0)

        policy = credit_policy or self.credit_policy
        if policy is not None and not isinstance(policy, ContextMemoryCreditPolicy):
            raise TypeError(
                "credit_policy must be a ContextMemoryCreditPolicy or None"
            )

        limit = min(max_items, len(values))
        recent_count = min(limit, max(1, ceil(limit * self.recent_fraction)))
        selected: list[tuple[int, ContextEpisode]] = [
            (index, values[index])
            for index in range(len(values) - recent_count, len(values))
        ]
        selected_indices = {index for index, _ in selected}

        credit_by_index: dict[int, float] = {}
        if policy is not None:
            credit_by_index = {
                item.sequence_index: item.credit
                for item in policy.assess(values)
            }

        # Fill remaining capacity by maximizing family/strategy coverage first.
        # Memory credit is the next deterministic criterion when enabled.
        remaining = [
            (index, item)
            for index, item in enumerate(values)
            if index not in selected_indices
        ]
        seen_families = {item.family_id for _, item in selected}
        seen_strategies = {item.strategy for _, item in selected}

        while len(selected) < limit and remaining:
            index, chosen = min(
                remaining,
                key=lambda pair: (
                    -(
                        int(pair[1].family_id not in seen_families)
                        + int(pair[1].strategy not in seen_strategies)
                    ),
                    -credit_by_index.get(pair[0], 0.0),
                    -pair[0],
                    pair[1].family_id,
                    pair[1].strategy,
                    pair[1].episode_id,
                ),
            )
            selected.append((index, chosen))
            remaining = [pair for pair in remaining if pair[0] != index]
            seen_families.add(chosen.family_id)
            seen_strategies.add(chosen.strategy)

        selected.sort(key=lambda pair: pair[0])
        return ContextReplayBatch(
            episodes=tuple(item for _, item in selected),
            source_size=len(values),
            family_count=len({item.family_id for _, item in selected}),
            strategy_count=len({item.strategy for _, item in selected}),
        )

    def consolidate(
        self,
        episodes: Sequence[ContextEpisode],
        *,
        max_episodes: int,
        credit_policy: ContextMemoryCreditPolicy | None = None,
    ) -> tuple[ContextEpisode, ...]:
        """Return the bounded retained memory set without mutating input."""
        return self.replay(
            episodes,
            max_items=max_episodes,
            credit_policy=credit_policy,
        ).episodes


__all__ = ["ContextMemoryConsolidator", "ContextReplayBatch"]
