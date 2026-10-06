from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Sequence

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

    Retention is based only on episode order, family/strategy coverage, and
    a configurable recent fraction. No hidden quality score is inferred.
    """

    def __init__(self, *, recent_fraction: float = 0.50) -> None:
        if not 0.0 < recent_fraction <= 1.0:
            raise ValueError("recent_fraction must be in (0, 1]")
        self.recent_fraction = recent_fraction

    @staticmethod
    def _validate(episodes: Sequence[ContextEpisode]) -> tuple[ContextEpisode, ...]:
        values = tuple(episodes)
        if any(not isinstance(item, ContextEpisode) for item in values):
            raise TypeError("episodes must contain only ContextEpisode values")
        return values

    def replay(
        self,
        episodes: Sequence[ContextEpisode],
        *,
        max_items: int,
    ) -> ContextReplayBatch:
        values = self._validate(episodes)
        if max_items < 1:
            raise ValueError("max_items must be at least 1")
        if not values:
            return ContextReplayBatch((), 0, 0, 0)

        limit = min(max_items, len(values))
        recent_count = min(limit, max(1, ceil(limit * self.recent_fraction)))
        selected: list[ContextEpisode] = list(values[-recent_count:])
        selected_ids = {id(item) for item in selected}

        # Fill remaining capacity by maximizing family/strategy coverage first,
        # then recency. This is deterministic and avoids random replay noise.
        remaining = [(index, item) for index, item in enumerate(values) if id(item) not in selected_ids]
        seen_families = {item.family_id for item in selected}
        seen_strategies = {item.strategy for item in selected}

        while len(selected) < limit and remaining:
            index, chosen = min(
                remaining,
                key=lambda pair: (
                    -(int(pair[1].family_id not in seen_families)
                      + int(pair[1].strategy not in seen_strategies)),
                    -pair[0],
                    pair[1].family_id,
                    pair[1].strategy,
                    pair[1].episode_id,
                ),
            )
            selected.append(chosen)
            remaining = [pair for pair in remaining if pair[0] != index]
            seen_families.add(chosen.family_id)
            seen_strategies.add(chosen.strategy)

        selected_ids_in_order = {id(item): index for index, item in enumerate(values)}
        selected = sorted(selected, key=lambda item: selected_ids_in_order[id(item)])
        return ContextReplayBatch(
            episodes=tuple(selected),
            source_size=len(values),
            family_count=len({item.family_id for item in selected}),
            strategy_count=len({item.strategy for item in selected}),
        )

    def consolidate(
        self,
        episodes: Sequence[ContextEpisode],
        *,
        max_episodes: int,
    ) -> tuple[ContextEpisode, ...]:
        """Return the bounded retained memory set without mutating input."""
        return self.replay(episodes, max_items=max_episodes).episodes


__all__ = ["ContextMemoryConsolidator", "ContextReplayBatch"]
