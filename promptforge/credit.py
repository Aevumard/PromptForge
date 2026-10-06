from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Sequence

from .memory import (
    ContextEpisode,
    ContextRoute,
    _distance,
    _feature_scales,
    resolve_routing_features,
    topology_vector,
)


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
            for strategy in means:
                if strategy in best_strategies and len(best_strategies) > 1:
                    outcome = 0
                elif strategy in best_strategies:
                    outcome = -1
                else:
                    outcome = 1
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


@dataclass(frozen=True)
class ContextMemoryAwareRoute(ContextRoute):
    """Route result with explicit active-memory evidence."""

    selected_credit: float
    candidate_count: int
    top_k: int
    strategy_scores: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["strategy_scores"] = [list(item) for item in self.strategy_scores]
        return payload


class ContextMemoryAwareRouter:
    """Route among nearby observed episodes using explicit memory credit.

    Training is always explicit. Structural distance is computed with the
    same scaling contract as NearestEpisodeRouter. Nearby evidence is then
    aggregated by strategy using credit-weighted structural similarity.

    This changes strategy preference only when explicitly selected. It is not
    a confidence estimate and never introduces an unobserved outcome.
    """

    def __init__(
        self,
        *,
        features: Sequence[str] | None = None,
        scale_mode: str = "iqr",
        top_k: int = 5,
        credit_policy: ContextMemoryCreditPolicy | None = None,
        min_family_count: int = 1,
        min_strategy_evidence: int = 1,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if min_family_count < 1:
            raise ValueError("min_family_count must be at least 1")
        if min_strategy_evidence < 1:
            raise ValueError("min_strategy_evidence must be at least 1")
        self.min_family_count = int(min_family_count)
        self.min_strategy_evidence = int(min_strategy_evidence)
        self.features = resolve_routing_features(features)
        if scale_mode not in {"minmax", "std", "iqr"}:
            raise ValueError("scale_mode must be one of: minmax, std, iqr")
        self.scale_mode = scale_mode
        self.top_k = int(top_k)
        if credit_policy is not None and not isinstance(
            credit_policy, ContextMemoryCreditPolicy
        ):
            raise TypeError(
                "credit_policy must be a ContextMemoryCreditPolicy or None"
            )
        self.credit_policy = credit_policy
        self._episodes: tuple[ContextEpisode, ...] = ()
        self._credits: tuple[ContextMemoryCredit, ...] = ()
        self._scale: tuple[float, ...] = ()

    @property
    def training_episode_ids(self) -> tuple[str, ...]:
        return tuple(sorted({episode.episode_id for episode in self._episodes}))

    @property
    def strategies(self) -> tuple[str, ...]:
        return tuple(sorted({episode.strategy for episode in self._episodes}))

    def fit(self, episodes: Sequence[ContextEpisode]) -> "ContextMemoryAwareRouter":
        values = tuple(episodes)
        if not values:
            raise ValueError("episodes must not be empty")
        if any(not isinstance(item, ContextEpisode) for item in values):
            raise TypeError("episodes must contain only ContextEpisode values")

        vectors = [topology_vector(item.topology, self.features) for item in values]
        self._scale = _feature_scales(vectors, self.scale_mode)
        self._episodes = values
        policy = self.credit_policy or ContextMemoryCreditPolicy()
        self._credits = policy.assess(values)
        return self

    def route(
        self,
        topology: Mapping[str, object],
    ) -> ContextMemoryAwareRoute:
        if not self._episodes:
            raise RuntimeError("router must be fitted before route()")

        vector = topology_vector(topology, self.features)
        credit_by_index = {
            item.sequence_index: item
            for item in self._credits
        }
        rows = []
        for index, episode in enumerate(self._episodes):
            distance = _distance(
                vector,
                topology_vector(episode.topology, self.features),
                self._scale,
            )
            credit = credit_by_index[index].credit
            similarity = 1.0 / (1.0 + distance)
            rows.append((index, episode, distance, credit, similarity))

        rows.sort(
            key=lambda row: (
                row[2],
                -row[3],
                row[1].strategy,
                row[1].episode_id,
                row[0],
            )
        )
        nearest = rows[0]
        candidate_rows = rows[: min(self.top_k, len(rows))]

        weighted_scores: dict[str, float] = {}
        family_sets: dict[str, set[str]] = {}
        evidence_counts: dict[str, int] = {}
        for _index, episode, _distance_value, credit, similarity in candidate_rows:
            weight = credit * similarity
            weighted_scores[episode.strategy] = (
                weighted_scores.get(episode.strategy, 0.0) + weight
            )
            family_sets.setdefault(episode.strategy, set()).add(episode.family_id)
            evidence_counts[episode.strategy] = (
                evidence_counts.get(episode.strategy, 0) + 1
            )

        admissible = {
            strategy
            for strategy in weighted_scores
            if len(family_sets[strategy]) >= self.min_family_count
            and evidence_counts[strategy] >= self.min_strategy_evidence
        }
        ranked_strategies = sorted(
            admissible or weighted_scores,
            key=lambda strategy: (-weighted_scores[strategy], strategy),
        )
        strategy = ranked_strategies[0]
        selected_credit = max(
            credit_by_index[row[0]].credit
            for row in candidate_rows
            if row[1].strategy == strategy
        )

        return ContextMemoryAwareRoute(
            strategy=strategy,
            nearest_distance=nearest[2],
            nearest_episode_id=nearest[1].episode_id,
            evidence_count=sum(
                episode.strategy == strategy for episode in self._episodes
            ),
            selected_credit=selected_credit,
            candidate_count=len(candidate_rows),
            top_k=self.top_k,
            strategy_scores=tuple(
                (name, weighted_scores[name]) for name in ranked_strategies
            ),
        )

    def predict(self, topology: Mapping[str, object]) -> str:
        return self.route(topology).strategy


__all__ = [
    "ContextMemoryCredit",
    "ContextMemoryCreditPolicy",
    "ContextMemoryAwareRoute",
    "ContextMemoryAwareRouter",
]
