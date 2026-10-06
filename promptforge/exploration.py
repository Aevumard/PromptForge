from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class ContextExplorationDecision:
    """Auditable decision for one bounded exploration opportunity."""

    experience_version: int
    preferred_strategy: str | None
    candidate_order: tuple[str, ...]
    observed_strategy_counts: tuple[tuple[str, int], ...]
    target_strategy: str | None
    eligible: bool
    required: bool
    reason: str
    cooldown_remaining: int
    exploration_count: int
    exploration_budget: int | None
    novelty_distance: float | None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["observed_strategy_counts"] = {
            strategy: count for strategy, count in self.observed_strategy_counts
        }
        payload["candidate_order"] = list(self.candidate_order)
        return payload


class ContextExplorationController:
    """Bounded deterministic exploration over already-available strategies.

    Exploration is a data-collection action, not a claim about strategy
    quality. The controller uses only current evidence and explicit caller
    signals. It never executes a provider or fabricates outcomes.
    """

    def __init__(
        self,
        *,
        novelty_threshold: float = 3.0,
        min_interval: int = 0,
        exploration_budget: int | None = None,
    ) -> None:
        if novelty_threshold < 0.0:
            raise ValueError("novelty_threshold must be non-negative")
        if min_interval < 0:
            raise ValueError("min_interval must be non-negative")
        if exploration_budget is not None and exploration_budget < 1:
            raise ValueError(
                "exploration_budget must be at least 1 when provided"
            )

        self.novelty_threshold = float(novelty_threshold)
        self.min_interval = int(min_interval)
        self.exploration_budget = (
            None
            if exploration_budget is None
            else int(exploration_budget)
        )
        self._last_exploration_experience_version: int | None = None
        self._exploration_count = 0

    @property
    def exploration_count(self) -> int:
        return self._exploration_count

    @property
    def last_exploration_experience_version(self) -> int | None:
        return self._last_exploration_experience_version

    def _cooldown_remaining(self, experience_version: int) -> int:
        if self._last_exploration_experience_version is None:
            return 0
        elapsed = max(
            0,
            experience_version - self._last_exploration_experience_version,
        )
        return max(0, self.min_interval - elapsed)

    def _budget_available(self) -> bool:
        return (
            self.exploration_budget is None
            or self._exploration_count < self.exploration_budget
        )

    @staticmethod
    def _counts(
        observed_strategies: Sequence[str],
    ) -> tuple[tuple[str, int], ...]:
        counts: dict[str, int] = {}
        for strategy in observed_strategies:
            key = str(strategy)
            counts[key] = counts.get(key, 0) + 1
        return tuple(sorted(counts.items()))

    def decide(
        self,
        *,
        experience_version: int,
        candidate_order: Sequence[str],
        observed_strategies: Sequence[str] = (),
        preferred_strategy: str | None = None,
        novelty_distance: float | None = None,
        policy_refresh_required: bool = False,
        force: bool = False,
    ) -> ContextExplorationDecision:
        if experience_version < 0:
            raise ValueError("experience_version must be non-negative")
        if novelty_distance is not None and novelty_distance < 0.0:
            raise ValueError("novelty_distance must be non-negative")
        if not isinstance(force, bool):
            raise TypeError("force must be a bool")

        candidates = tuple(dict.fromkeys(str(item) for item in candidate_order))
        counts = self._counts(observed_strategies)
        count_map = dict(counts)
        cooldown_remaining = self._cooldown_remaining(experience_version)
        budget_available = self._budget_available()

        alternatives = tuple(
            strategy
            for strategy in candidates
            if strategy != preferred_strategy
        )

        required = bool(
            force
            or policy_refresh_required
            or (
                novelty_distance is not None
                and novelty_distance > self.novelty_threshold
            )
            or any(count_map.get(strategy, 0) == 0 for strategy in alternatives)
        )

        if not alternatives:
            reason = "no_alternative_strategy"
            eligible = False
            target = None
        elif cooldown_remaining > 0:
            reason = "exploration_cooldown"
            eligible = False
            target = None
        elif not budget_available:
            reason = "exploration_budget_exhausted"
            eligible = False
            target = None
        elif not required:
            reason = "exploit"
            eligible = False
            target = None
        else:
            unobserved = [
                strategy for strategy in alternatives
                if count_map.get(strategy, 0) == 0
            ]
            target = min(
                unobserved or list(alternatives),
                key=lambda strategy: (count_map.get(strategy, 0), candidates.index(strategy)),
            )
            if force or policy_refresh_required:
                reason = "policy_refresh"
            elif novelty_distance is not None and novelty_distance > self.novelty_threshold:
                reason = "novel_case"
            elif target in unobserved:
                reason = "coverage_gap"
            else:
                reason = "bounded_exploration"
            eligible = True

        return ContextExplorationDecision(
            experience_version=experience_version,
            preferred_strategy=preferred_strategy,
            candidate_order=candidates,
            observed_strategy_counts=counts,
            target_strategy=target,
            eligible=eligible,
            required=required,
            reason=reason,
            cooldown_remaining=cooldown_remaining,
            exploration_count=self._exploration_count,
            exploration_budget=self.exploration_budget,
            novelty_distance=novelty_distance,
        )

    def record_exploration(self, *, experience_version: int) -> int:
        """Record one completed exploration and advance its bounded state."""
        if experience_version < 0:
            raise ValueError("experience_version must be non-negative")
        if not self._budget_available():
            raise RuntimeError("exploration budget exhausted")
        if self._cooldown_remaining(experience_version) > 0:
            raise RuntimeError("exploration cooldown is still active")
        if (
            self._last_exploration_experience_version is not None
            and experience_version < self._last_exploration_experience_version
        ):
            raise ValueError(
                "exploration experience version cannot move backward"
            )

        self._last_exploration_experience_version = experience_version
        self._exploration_count += 1
        return self._exploration_count

    def to_dict(self) -> dict:
        return {
            "novelty_threshold": self.novelty_threshold,
            "min_interval": self.min_interval,
            "exploration_budget": self.exploration_budget,
            "last_exploration_experience_version": (
                self._last_exploration_experience_version
            ),
            "exploration_count": self._exploration_count,
        }


__all__ = [
    "ContextExplorationDecision",
    "ContextExplorationController",
]
