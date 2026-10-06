from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Sequence

from .memory import ContextEpisode


@dataclass(frozen=True)
class ContextMemoryCredit:
    """Operational credit assigned to one observed episode.

    The score combines only observable bookkeeping:
    recency, repeated evidence for the same family/strategy, observed
    consistency of cost, and within-episode comparative outcomes.

    It is a retention/ranking aid, not a probability of truth or a causal
    estimate of model quality.
    """

    episode_id: str
    family_id: str
    strategy: str
    sequence_index: int
    recency_weight: float
    evidence_count: int
    comparable_count: int
    win_count: int
    loss_count: int
    tie_count: int
    observed_win_rate: float
    stability: float
    credit: float

    def to_dict(self) -> dict:
        return asdict(self)


class ContextMemoryCreditPolicy:
    """Deterministic credit/decay policy over an explicit episode snapshot."""

    def __init__(self, *, half_life: float = 8.0) -> None:
        if not isfinite(half_life) or half_life <= 0.0:
            raise ValueError("half_life must be finite and positive")
        self.half_life = float(half_life)

    @staticmethod
    def _validate(
        episodes: Sequence[ContextEpisode],
    ) -> tuple[ContextEpisode, ...]:
        values = tuple(episodes)
        if any(not isinstance(item, ContextEpisode) for item in values):
            raise TypeError("episodes must contain only ContextEpisode values")
        return values

    @staticmethod
    def _stability(costs: Sequence[float]) -> float:
        if not costs:
            return 0.0
        mean = sum(float(cost) for cost in costs) / len(costs)
        if mean == 0.0:
            return 1.0 if all(float(cost) == 0.0 for cost in costs) else 0.0
        variance = sum((float(cost) - mean) ** 2 for cost in costs) / len(costs)
        coefficient_of_variation = sqrt(variance) / abs(mean)
        return 1.0 / (1.0 + coefficient_of_variation)

    def assess(
        self,
        episodes: Sequence[ContextEpisode],
    ) -> tuple[ContextMemoryCredit, ...]:
        values = self._validate(episodes)
        if not values:
            return ()

        # Repetition support and observed cost stability are computed within
        # a family + strategy bucket.
        costs_by_bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
        for episode in values:
            costs_by_bucket[(episode.family_id, episode.strategy)].append(
                float(episode.cost)
            )

        # Comparative evidence exists only when one episode id has at least
        # two observed strategies. This keeps "wins" explicit rather than
        # fabricating a comparison for single-strategy observations.
        episode_groups: dict[tuple[str, str], list[ContextEpisode]] = defaultdict(list)
        for episode in values:
            episode_groups[(episode.family_id, episode.episode_id)].append(episode)

        comparisons: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
        for (family_id, episode_id), group in episode_groups.items():
            costs_by_strategy: dict[str, list[float]] = defaultdict(list)
            for episode in group:
                costs_by_strategy[episode.strategy].append(float(episode.cost))
            means = {
                strategy: sum(costs) / len(costs)
                for strategy, costs in costs_by_strategy.items()
            }
            if len(means) < 2:
                continue
            oracle_cost = min(means.values())
            best_strategies = {
                strategy
                for strategy, mean_cost in means.items()
                if mean_cost == oracle_cost
            }
            for strategy, mean_cost in means.items():
                if strategy in best_strategies and len(best_strategies) > 1:
                    outcome = 0.0
                elif strategy in best_strategies:
                    outcome = -1.0
                else:
                    outcome = 1.0
                comparisons[(family_id, strategy)].append(
                    (outcome, strategy)
                )

        result: list[ContextMemoryCredit] = []
        latest_index = len(values) - 1
        for index, episode in enumerate(values):
            bucket = (episode.family_id, episode.strategy)
            bucket_costs = costs_by_bucket[bucket]
            support_count = len(bucket_costs)
            evidence_factor = support_count / (support_count + 1.0)

            age = latest_index - index
            recency_weight = 2.0 ** (-age / self.half_life)

            comparable = comparisons.get(bucket, [])
            comparable_count = len(comparable)
            win_count = sum(delta < 0.0 for delta, _ in comparable)
            loss_count = sum(delta > 0.0 for delta, _ in comparable)
            tie_count = comparable_count - win_count - loss_count

            observed_win_rate = (
                (win_count + 0.5 * tie_count) / comparable_count
                if comparable_count
                else 0.5
            )
            stability = self._stability(bucket_costs)
            comparative_factor = 0.5 + 0.5 * observed_win_rate
            credit = (
                recency_weight
                * evidence_factor
                * stability
                * comparative_factor
            )

            result.append(
                ContextMemoryCredit(
                    episode_id=episode.episode_id,
                    family_id=episode.family_id,
                    strategy=episode.strategy,
                    sequence_index=index,
                    recency_weight=recency_weight,
                    evidence_count=support_count,
                    comparable_count=comparable_count,
                    win_count=win_count,
                    loss_count=loss_count,
                    tie_count=tie_count,
                    observed_win_rate=observed_win_rate,
                    stability=stability,
                    credit=credit,
                )
            )

        return tuple(result)

    def rank(
        self,
        episodes: Sequence[ContextEpisode],
    ) -> tuple[ContextMemoryCredit, ...]:
        credits = self.assess(episodes)
        return tuple(
            sorted(
                credits,
                key=lambda item: (
                    -item.credit,
                    -item.recency_weight,
                    item.family_id,
                    item.strategy,
                    item.episode_id,
                    item.sequence_index,
                ),
            )
        )


__all__ = ["ContextMemoryCredit", "ContextMemoryCreditPolicy"]
