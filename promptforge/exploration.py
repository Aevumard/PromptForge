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


@dataclass(frozen=True)
class ContextExplorationStrategyEvidence:
    """Aggregated within-episode evidence for one explored challenger."""

    strategy: str
    baseline_strategy: str | None
    comparisons: int
    families: int
    wins: int
    losses: int
    mean_relative_gain: float
    mean_absolute_gain: float
    unique_episodes: int
    ties: int
    baseline_strategies: tuple[str, ...] = ()

    @property
    def win_rate(self) -> float:
        if self.comparisons <= 0:
            return 0.0
        return self.wins / self.comparisons

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["win_rate"] = self.win_rate
        return payload


@dataclass(frozen=True)
class ContextExplorationAdoptionDecision:
    """Conservative decision about whether a challenger has enough evidence."""

    strategy: str | None
    baseline_strategy: str | None
    eligible: bool
    comparisons: int
    families: int
    win_rate: float
    mean_relative_gain: float
    mean_absolute_gain: float
    min_comparisons: int
    min_families: int
    min_win_rate: float
    min_relative_gain: float
    reason: str
    unique_episodes: int = 0
    ties: int = 0
    baseline_strategies: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["baseline_strategies"] = list(self.baseline_strategies)
        return payload


class ContextExplorationAdjudicator:
    """Compare explored challengers against observed incumbents.

    Only within-episode comparisons are used. A probe with no observed
    incumbent in the same episode cannot establish a challenger advantage.
    The adjudicator is descriptive and does not mutate memory or policy.
    """

    def __init__(
        self,
        *,
        min_comparisons: int = 2,
        min_families: int = 2,
        min_win_rate: float = 0.60,
        min_relative_gain: float = 0.01,
    ) -> None:
        if min_comparisons < 1:
            raise ValueError("min_comparisons must be at least 1")
        if min_families < 1:
            raise ValueError("min_families must be at least 1")
        if not 0.0 <= min_win_rate <= 1.0:
            raise ValueError("min_win_rate must be between 0.0 and 1.0")
        if min_relative_gain < 0.0:
            raise ValueError("min_relative_gain must be non-negative")

        self.min_comparisons = int(min_comparisons)
        self.min_families = int(min_families)
        self.min_win_rate = float(min_win_rate)
        self.min_relative_gain = float(min_relative_gain)

    @staticmethod
    def _group(
        episodes: Sequence["ContextEpisode"],
    ) -> dict[str, list["ContextEpisode"]]:
        from .memory import ContextEpisode

        grouped: dict[str, list[ContextEpisode]] = {}
        for episode in episodes:
            grouped.setdefault(episode.episode_id, []).append(episode)
        return grouped

    def assess(
        self,
        episodes: Sequence["ContextEpisode"],
    ) -> tuple[ContextExplorationStrategyEvidence, ...]:
        """Aggregate conservative probe evidence from an immutable episode set.

        Repeated probes for the same challenger inside one episode are averaged
        into one comparison. This prevents repeated measurements from inflating
        the effective sample size or allowing a best-of-many observation to win
        by cherry-picking. Ties remain ties rather than wins.
        """
        grouped = self._group(episodes)
        evidence: dict[str, list[tuple[str, str, float, float, int]]] = {}

        for group in grouped.values():
            explored_by_strategy: dict[str, list[float]] = {}
            for episode in group:
                if (
                    episode.action == "probe"
                    or episode.source == "bounded_exploration"
                ):
                    explored_by_strategy.setdefault(
                        episode.strategy,
                        [],
                    ).append(float(episode.cost))

            if not explored_by_strategy:
                continue

            incumbents = [
                episode
                for episode in group
                if not (
                    episode.action == "probe"
                    or episode.source == "bounded_exploration"
                )
            ]
            if not incumbents:
                continue

            baseline = min(
                incumbents,
                key=lambda episode: (float(episode.cost), episode.strategy),
            )
            baseline_cost = float(baseline.cost)
            baseline_strategies = tuple(
                sorted(
                    {
                        episode.strategy
                        for episode in incumbents
                        if float(episode.cost) == baseline_cost
                    }
                )
            )
            for strategy, costs in sorted(explored_by_strategy.items()):
                challenger_cost = sum(costs) / len(costs)
                absolute_gain = baseline_cost - challenger_cost
                relative_gain = (
                    absolute_gain / baseline_cost
                    if baseline_cost > 0.0
                    else 0.0
                )
                if absolute_gain > 0.0:
                    outcome = 1
                elif absolute_gain < 0.0:
                    outcome = -1
                else:
                    outcome = 0
                evidence.setdefault(strategy, []).append(
                    (
                        group[0].family_id,
                        group[0].episode_id,
                        absolute_gain,
                        relative_gain,
                        outcome,
                    )
                )
                # The family/episode identifiers above are descriptive only;
                # the row remains exactly one comparison for this episode.

        results: list[ContextExplorationStrategyEvidence] = []
        for strategy, rows in sorted(evidence.items()):
            families = {row[0] for row in rows}
            episodes = {row[1] for row in rows}
            wins = sum(row[4] > 0 for row in rows)
            losses = sum(row[4] < 0 for row in rows)
            ties = sum(row[4] == 0 for row in rows)
            baseline_strategies: set[str] = set()

            # Reconstruct the baseline strategy for each retained comparison.
            for group in grouped.values():
                if not any(
                    challenger.strategy == strategy
                    and (
                        challenger.action == "probe"
                        or challenger.source == "bounded_exploration"
                    )
                    for challenger in group
                ):
                    continue
                incumbents = [
                    episode
                    for episode in group
                    if not (
                        episode.action == "probe"
                        or episode.source == "bounded_exploration"
                    )
                ]
                if not incumbents:
                    continue
                baseline_cost = min(float(item.cost) for item in incumbents)
                baseline_strategies.update(
                    item.strategy
                    for item in incumbents
                    if float(item.cost) == baseline_cost
                )

            ordered_baselines = tuple(sorted(baseline_strategies))
            baseline_strategy = (
                ordered_baselines[0]
                if len(ordered_baselines) == 1
                else None
            )
            results.append(
                ContextExplorationStrategyEvidence(
                    strategy=strategy,
                    baseline_strategy=baseline_strategy,
                    comparisons=len(rows),
                    families=len(families),
                    wins=wins,
                    losses=losses,
                    mean_relative_gain=(
                        sum(row[3] for row in rows) / len(rows)
                        if rows
                        else 0.0
                    ),
                    mean_absolute_gain=(
                        sum(row[2] for row in rows) / len(rows)
                        if rows
                        else 0.0
                    ),
                    unique_episodes=len(episodes),
                    ties=ties,
                    baseline_strategies=ordered_baselines,
                )
            )

        return tuple(results)

    def decide(
        self,
        episodes: Sequence["ContextEpisode"],
        *,
        preferred_strategy: str | None = None,
    ) -> ContextExplorationAdoptionDecision:
        evidence = self.assess(episodes)
        eligible = [
            item
            for item in evidence
            if item.unique_episodes >= self.min_comparisons
            and item.families >= self.min_families
            and item.baseline_strategy is not None
            and item.win_rate >= self.min_win_rate
            and item.mean_relative_gain >= self.min_relative_gain
            and (
                preferred_strategy is None
                or item.strategy != preferred_strategy
            )
        ]

        if eligible:
            selected = max(
                eligible,
                key=lambda item: (
                    item.win_rate,
                    item.mean_relative_gain,
                    item.comparisons,
                    item.strategy,
                ),
            )
            return ContextExplorationAdoptionDecision(
                strategy=selected.strategy,
                baseline_strategy=selected.baseline_strategy,
                eligible=True,
                comparisons=selected.comparisons,
                families=selected.families,
                win_rate=selected.win_rate,
                mean_relative_gain=selected.mean_relative_gain,
                mean_absolute_gain=selected.mean_absolute_gain,
                min_comparisons=self.min_comparisons,
                min_families=self.min_families,
                min_win_rate=self.min_win_rate,
                min_relative_gain=self.min_relative_gain,
                reason="challenger_meets_evidence_gate",
            )

        best = (
            max(
                evidence,
                key=lambda item: (
                    item.win_rate,
                    item.mean_relative_gain,
                    item.comparisons,
                    item.strategy,
                ),
            )
            if evidence
            else None
        )
        if best is None:
            reason = "no_comparable_probe_evidence"
        elif best.unique_episodes < self.min_comparisons:
            reason = "insufficient_unique_episodes"
        elif best.families < self.min_families:
            reason = "insufficient_families"
        elif best.baseline_strategy is None:
            reason = "baseline_strategy_mismatch"
        elif best.win_rate < self.min_win_rate:
            reason = "win_rate_below_threshold"
        else:
            reason = "relative_gain_below_threshold"

        return ContextExplorationAdoptionDecision(
            strategy=(best.strategy if best is not None else None),
            baseline_strategy=(best.baseline_strategy if best is not None else None),
            eligible=False,
            comparisons=(best.comparisons if best is not None else 0),
            families=(best.families if best is not None else 0),
            win_rate=(best.win_rate if best is not None else 0.0),
            mean_relative_gain=(
                best.mean_relative_gain if best is not None else 0.0
            ),
            mean_absolute_gain=(
                best.mean_absolute_gain if best is not None else 0.0
            ),
            min_comparisons=self.min_comparisons,
            min_families=self.min_families,
            min_win_rate=self.min_win_rate,
            min_relative_gain=self.min_relative_gain,
            reason=reason,
            unique_episodes=(best.unique_episodes if best is not None else 0),
            ties=(best.ties if best is not None else 0),
            baseline_strategies=(
                best.baseline_strategies if best is not None else ()
            ),
        )


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
    "ContextExplorationStrategyEvidence",
    "ContextExplorationAdoptionDecision",
    "ContextExplorationAdjudicator",
    "ContextExplorationController",
]
