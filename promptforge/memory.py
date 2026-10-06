from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence

from .adaptive import ContextTopologyProfile


ROUTING_FEATURES = (
    "top_level_fields",
    "node_count",
    "leaf_fields",
    "max_depth",
    "avg_depth",
    "depth_std",
    "max_branching",
    "avg_branching",
    "hub_ratio",
    "required_leaf_ratio",
    "active_root_ratio",
    "boundary_pressure",
    "relation_count",
    "relation_nodes",
    "relation_density",
    "relation_max_degree",
    "relation_avg_degree",
    "relation_hub_ratio",
    "relation_components",
    "relation_kind_count",
    "estimated_tokens",
)

SCALERS = ("minmax", "std", "iqr")


@dataclass(frozen=True)
class ContextEpisode:
    """One observed strategy outcome for one context episode."""

    episode_id: str
    family_id: str
    strategy: str
    cost: float
    topology: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if not self.family_id:
            raise ValueError("family_id must not be empty")
        if not self.strategy:
            raise ValueError("strategy must not be empty")
        cost = float(self.cost)
        if not isfinite(cost) or cost < 0.0:
            raise ValueError("cost must be a finite non-negative number")

    def vector(self, features: Sequence[str] | None = None) -> tuple[float, ...]:
        selected = resolve_routing_features(features)
        return topology_vector(self.topology, selected)


@dataclass(frozen=True)
class ContextRoute:
    """Observed-case routing result with an explicit novelty signal."""

    strategy: str
    nearest_distance: float
    nearest_episode_id: str
    evidence_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextRoutingEvaluation:
    episode_id: str
    family_id: str
    selected_strategy: str
    oracle_strategy: str
    selected_cost: float
    oracle_cost: float
    absolute_regret: float
    relative_regret: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_routing_features(
    features: Sequence[str] | None = None,
) -> tuple[str, ...]:
    selected = ROUTING_FEATURES if features is None else tuple(features)
    if not selected:
        raise ValueError("features must contain at least one routing feature")
    unknown = sorted(set(selected).difference(ROUTING_FEATURES))
    if unknown:
        raise ValueError(
            "unknown routing features: " + ", ".join(unknown)
        )
    if len(set(selected)) != len(selected):
        raise ValueError("features must not contain duplicates")
    return selected


def topology_vector(
    topology: Mapping[str, Any] | ContextTopologyProfile,
    features: Sequence[str] | None = None,
) -> tuple[float, ...]:
    selected = resolve_routing_features(features)
    if isinstance(topology, ContextTopologyProfile):
        topology = topology.to_dict()

    values: list[float] = []
    for name in selected:
        value = topology.get(name, 0.0)
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        if number != number:
            number = 0.0
        values.append(number)
    return tuple(values)


def _feature_scales(
    vectors: Sequence[Sequence[float]],
    mode: str,
) -> tuple[float, ...]:
    if mode not in SCALERS:
        raise ValueError("scale_mode must be one of: " + ", ".join(SCALERS))
    if not vectors:
        raise ValueError("vectors must not be empty")

    width = len(vectors[0])
    scales: list[float] = []
    for i in range(width):
        values = sorted(float(vector[i]) for vector in vectors)
        if mode == "minmax":
            scale = values[-1] - values[0]
        elif mode == "std":
            mean = sum(values) / len(values)
            scale = sqrt(
                sum((value - mean) ** 2 for value in values) / len(values)
            )
        else:
            if len(values) == 1:
                scale = 0.0
            else:
                q1_pos = 0.25 * (len(values) - 1)
                q3_pos = 0.75 * (len(values) - 1)
                q1_lo = int(q1_pos)
                q1_hi = min(q1_lo + 1, len(values) - 1)
                q3_lo = int(q3_pos)
                q3_hi = min(q3_lo + 1, len(values) - 1)
                q1 = values[q1_lo] + (
                    values[q1_hi] - values[q1_lo]
                ) * (q1_pos - q1_lo)
                q3 = values[q3_lo] + (
                    values[q3_hi] - values[q3_lo]
                ) * (q3_pos - q3_lo)
                scale = q3 - q1
        scales.append(scale if scale > 0.0 else 1.0)
    return tuple(scales)


def _distance(
    left: Sequence[float],
    right: Sequence[float],
    scale: Sequence[float],
) -> float:
    return sqrt(
        sum(
            ((x - y) / divisor) ** 2
            for x, y, divisor in zip(left, right, scale)
            if divisor > 0.0
        )
    )


class NearestEpisodeRouter:
    """Experience router using nearest observed context episodes.

    The router never invents an outcome. It ranks strategies by the nearest
    observed episodes in topology space. Fit data must be supplied explicitly;
    held-out evaluation is handled separately.
    """

    def __init__(
        self,
        *,
        features: Sequence[str] | None = None,
        scale_mode: str = "iqr",
    ) -> None:
        self.features = resolve_routing_features(features)
        if scale_mode not in SCALERS:
            raise ValueError("scale_mode must be one of: " + ", ".join(SCALERS))
        self.scale_mode = scale_mode
        self._training: tuple[
            tuple[str, str, str, tuple[float, ...], float], ...
        ] = ()
        self._scale: tuple[float, ...] = ()

    @property
    def strategies(self) -> tuple[str, ...]:
        return tuple(sorted({row[2] for row in self._training}))

    @property
    def training_episode_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row[0] for row in self._training}))

    def fit(self, episodes: Sequence[ContextEpisode]) -> "NearestEpisodeRouter":
        if not episodes:
            raise ValueError("episodes must not be empty")

        vectors = [episode.vector(self.features) for episode in episodes]
        self._scale = _feature_scales(vectors, self.scale_mode)
        self._training = tuple(
            sorted(
                (
                    episode.episode_id,
                    episode.family_id,
                    episode.strategy,
                    vector,
                    float(episode.cost),
                )
                for episode, vector in zip(episodes, vectors)
            )
        )
        return self

    def rank(self, topology: Mapping[str, Any] | ContextTopologyProfile) -> tuple[str, ...]:
        if not self._training:
            raise RuntimeError("router must be fitted before rank()")

        vector = topology_vector(topology, self.features)
        ranked_rows = sorted(
            self._training,
            key=lambda row: (
                _distance(vector, row[3], self._scale),
                row[4],
                row[2],
                row[0],
            ),
        )

        ranked: list[str] = []
        for row in ranked_rows:
            strategy = row[2]
            if strategy not in ranked:
                ranked.append(strategy)
        return tuple(ranked)

    def route(
        self,
        topology: Mapping[str, Any] | ContextTopologyProfile,
    ) -> ContextRoute:
        if not self._training:
            raise RuntimeError("router must be fitted before route()")

        vector = topology_vector(topology, self.features)
        nearest = min(
            self._training,
            key=lambda row: (
                _distance(vector, row[3], self._scale),
                row[4],
                row[2],
                row[0],
            ),
        )
        distance = _distance(vector, nearest[3], self._scale)
        evidence_count = sum(1 for row in self._training if row[2] == nearest[2])

        return ContextRoute(
            strategy=nearest[2],
            nearest_distance=distance,
            nearest_episode_id=nearest[0],
            evidence_count=evidence_count,
        )

    def novelty(
        self,
        topology: Mapping[str, Any] | ContextTopologyProfile,
    ) -> float:
        """Return distance to the nearest observed episode.

        This is a relative structural novelty signal, not a probability or
        calibrated confidence score.
        """
        return self.route(topology).nearest_distance

    def predict(self, topology: Mapping[str, Any] | ContextTopologyProfile) -> str:
        return self.route(topology).strategy


def episode_oracle(episodes: Sequence[ContextEpisode]) -> dict[str, str]:
    """Choose one best observed strategy per episode using supplied episodes."""
    grouped: dict[str, list[ContextEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.episode_id].append(episode)

    result: dict[str, str] = {}
    for episode_id, group in grouped.items():
        costs_by_strategy: dict[str, list[float]] = defaultdict(list)
        for episode in group:
            costs_by_strategy[episode.strategy].append(float(episode.cost))
        means = {
            strategy: sum(costs) / len(costs)
            for strategy, costs in costs_by_strategy.items()
        }
        result[episode_id] = min(
            means,
            key=lambda strategy: (means[strategy], strategy),
        )
    return result


def evaluate_holdout(
    episodes: Sequence[ContextEpisode],
    predictions: Mapping[str, str],
) -> tuple[ContextRoutingEvaluation, ...]:
    grouped: dict[str, list[ContextEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.episode_id].append(episode)

    evaluations: list[ContextRoutingEvaluation] = []
    for episode_id, group in sorted(grouped.items()):
        if episode_id not in predictions:
            raise ValueError(f"missing prediction for episode {episode_id!r}")

        selected = predictions[episode_id]
        costs = {
            strategy: sum(
                episode.cost
                for episode in group
                if episode.strategy == strategy
            )
            / sum(episode.strategy == strategy for episode in group)
            for strategy in {episode.strategy for episode in group}
        }
        oracle = min(costs, key=lambda strategy: (costs[strategy], strategy))
        if selected not in costs:
            raise ValueError(
                f"predicted strategy {selected!r} was not observed for episode {episode_id!r}"
            )

        selected_cost = float(costs[selected])
        oracle_cost = float(costs[oracle])
        absolute_regret = selected_cost - oracle_cost
        relative_regret = (
            absolute_regret / oracle_cost if oracle_cost else 0.0
        )
        family_id = str(group[0].family_id)

        evaluations.append(
            ContextRoutingEvaluation(
                episode_id=str(episode_id),
                family_id=family_id,
                selected_strategy=selected,
                oracle_strategy=oracle,
                selected_cost=selected_cost,
                oracle_cost=oracle_cost,
                absolute_regret=absolute_regret,
                relative_regret=relative_regret,
            )
        )

    return tuple(evaluations)


def leave_one_family_out(
    episodes: Sequence[ContextEpisode],
    *,
    router_factory: Any = None,
) -> tuple[tuple[str, tuple[ContextRoutingEvaluation, ...]], ...]:
    """Evaluate routing without fitting on any held-out context family."""

    if not episodes:
        raise ValueError("episodes must not be empty")

    families = sorted({episode.family_id for episode in episodes})
    if len(families) < 2:
        raise ValueError("at least two families are required")

    if router_factory is None:
        router_factory = NearestEpisodeRouter

    available_ids = {episode.episode_id for episode in episodes}
    folds: list[tuple[str, tuple[ContextRoutingEvaluation, ...]]] = []

    for held_out_family in families:
        train = [
            episode
            for episode in episodes
            if episode.family_id != held_out_family
        ]
        holdout = [
            episode
            for episode in episodes
            if episode.family_id == held_out_family
        ]
        train_ids = {episode.episode_id for episode in train}
        holdout_ids = {episode.episode_id for episode in holdout}

        if train_ids.intersection(holdout_ids):
            raise AssertionError("episode leakage across family split")
        if not train or not holdout:
            raise ValueError(f"empty train/holdout split for {held_out_family}")

        router = router_factory()
        if not hasattr(router, "fit") or not hasattr(router, "predict"):
            raise TypeError("router_factory must provide fit() and predict()")
        router.fit(train)

        if set(router.training_episode_ids).difference(available_ids):
            raise AssertionError("router retained an unknown training episode")

        predictions = {
            episode_id: router.predict(
                next(
                    episode.topology
                    for episode in holdout
                    if episode.episode_id == episode_id
                )
            )
            for episode_id in sorted(holdout_ids)
        }
        evaluations = evaluate_holdout(holdout, predictions)
        folds.append((held_out_family, evaluations))

    return tuple(folds)


def routing_summary(
    evaluations: Sequence[ContextRoutingEvaluation],
) -> dict[str, int | float]:
    if not evaluations:
        return {
            "episodes": 0,
            "oracle_agreement_rate": 0.0,
            "mean_absolute_regret": 0.0,
            "mean_relative_regret": 0.0,
        }

    agreement = sum(
        item.selected_strategy == item.oracle_strategy
        for item in evaluations
    )
    return {
        "episodes": len(evaluations),
        "oracle_agreement_rate": agreement / len(evaluations),
        "mean_absolute_regret": (
            sum(item.absolute_regret for item in evaluations)
            / len(evaluations)
        ),
        "mean_relative_regret": (
            sum(item.relative_regret for item in evaluations)
            / len(evaluations)
        ),
    }


__all__ = [
    "ContextEpisode",
    "ContextRoutingEvaluation",
    "ContextRoute",
    "NearestEpisodeRouter",
    "ROUTING_FEATURES",
    "SCALERS",
    "episode_oracle",
    "evaluate_holdout",
    "leave_one_family_out",
    "resolve_routing_features",
    "routing_summary",
    "topology_vector",
]
