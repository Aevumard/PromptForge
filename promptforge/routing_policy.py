from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .credit import ContextMemoryAwareRouter, ContextMemoryCreditPolicy
from .memory import (
    ContextEpisode,
    ContextRoutingEvaluation,
    NearestEpisodeRouter,
    evaluate_holdout,
)

ROUTING_POLICY_MODES = ("nearest", "credit", "adaptive")


@dataclass(frozen=True)
class ContextRoutingModeScore:
    """Observed holdout performance for one routing mode."""

    mode: str
    episodes: int
    families: int
    oracle_agreement_rate: float
    mean_absolute_regret: float
    mean_relative_regret: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ContextRoutingPolicyEvidence:
    """Immutable evidence used to select a future routing mode.

    This evidence is produced by a whole-family holdout evaluation. It must
    never include the current decision's held-out outcome.
    """

    version: int
    scores: tuple[ContextRoutingModeScore, ...]
    selected_mode: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["scores"] = [score.to_dict() for score in self.scores]
        return payload


class ContextRoutingPolicyEvaluator:
    """Compare nearest and credit-aware routing under family holdout."""

    def __init__(
        self,
        *,
        features: Sequence[str] | None = None,
        scale_mode: str = "iqr",
        top_k: int = 5,
        credit_policy: ContextMemoryCreditPolicy | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.features = tuple(features) if features is not None else None
        self.scale_mode = scale_mode
        self.top_k = int(top_k)
        if credit_policy is not None and not isinstance(
            credit_policy, ContextMemoryCreditPolicy
        ):
            raise TypeError(
                "credit_policy must be a ContextMemoryCreditPolicy or None"
            )
        self.credit_policy = credit_policy

    @staticmethod
    def _score(
        mode: str,
        evaluations: Sequence[ContextRoutingEvaluation],
    ) -> ContextRoutingModeScore:
        if not evaluations:
            return ContextRoutingModeScore(
                mode=mode,
                episodes=0,
                families=0,
                oracle_agreement_rate=0.0,
                mean_absolute_regret=0.0,
                mean_relative_regret=0.0,
            )
        return ContextRoutingModeScore(
            mode=mode,
            episodes=len(evaluations),
            families=len({item.family_id for item in evaluations}),
            oracle_agreement_rate=sum(
                item.selected_strategy == item.oracle_strategy
                for item in evaluations
            ) / len(evaluations),
            mean_absolute_regret=sum(
                item.absolute_regret for item in evaluations
            ) / len(evaluations),
            mean_relative_regret=sum(
                item.relative_regret for item in evaluations
            ) / len(evaluations),
        )

    def _evaluate_mode(
        self,
        episodes: Sequence[ContextEpisode],
        mode: str,
    ) -> tuple[ContextRoutingEvaluation, ...]:
        families = sorted({episode.family_id for episode in episodes})
        if len(families) < 2:
            raise ValueError("at least two families are required")
        if not episodes:
            raise ValueError("episodes must not be empty")

        evaluations: list[ContextRoutingEvaluation] = []
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
                raise AssertionError(
                    "episode leakage across family split"
                )

            if mode == "nearest":
                router = NearestEpisodeRouter(
                    features=self.features,
                    scale_mode=self.scale_mode,
                ).fit(train)
            elif mode == "credit":
                router = ContextMemoryAwareRouter(
                    features=self.features,
                    scale_mode=self.scale_mode,
                    top_k=self.top_k,
                    credit_policy=self.credit_policy,
                ).fit(train)
            else:
                raise ValueError(
                    "mode must be one of: nearest, credit"
                )

            predictions = {}
            for episode_id in sorted(holdout_ids):
                topology = next(
                    episode.topology
                    for episode in holdout
                    if episode.episode_id == episode_id
                )
                predictions[episode_id] = router.predict(topology)

            evaluations.extend(evaluate_holdout(holdout, predictions))

        return tuple(evaluations)

    def evaluate(
        self,
        episodes: Sequence[ContextEpisode],
        *,
        version: int = 1,
        selected_mode: str | None = None,
    ) -> ContextRoutingPolicyEvidence:
        values = tuple(episodes)
        if not values:
            raise ValueError("episodes must not be empty")

        scores = tuple(
            self._score(
                mode,
                self._evaluate_mode(values, mode),
            )
            for mode in ("nearest", "credit")
        )
        selector = ContextRoutingPolicySelector()
        mode = selector.select(
            ContextRoutingPolicyEvidence(
                version=version,
                scores=scores,
                selected_mode=selected_mode or "nearest",
            )
        )
        return ContextRoutingPolicyEvidence(
            version=version,
            scores=scores,
            selected_mode=mode,
        )


class ContextRoutingPolicySelector:
    """Deterministically choose among evaluated routing modes."""

    def __init__(self, *, min_episodes: int = 1) -> None:
        if min_episodes < 1:
            raise ValueError("min_episodes must be at least 1")
        self.min_episodes = int(min_episodes)

    def select(self, evidence: ContextRoutingPolicyEvidence) -> str:
        available = [
            score
            for score in evidence.scores
            if score.episodes >= self.min_episodes
            and score.mode in ("nearest", "credit")
        ]
        if not available:
            return "nearest"

        return min(
            available,
            key=lambda score: (
                score.mean_relative_regret,
                score.mean_absolute_regret,
                -score.oracle_agreement_rate,
                0 if score.mode == "nearest" else 1,
            ),
        ).mode


__all__ = [
    "ROUTING_POLICY_MODES",
    "ContextRoutingModeScore",
    "ContextRoutingPolicyEvidence",
    "ContextRoutingPolicyEvaluator",
    "ContextRoutingPolicySelector",
]
