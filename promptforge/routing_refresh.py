from __future__ import annotations

from dataclasses import asdict, dataclass

from .routing_history import (
    ContextRoutingPolicyHealth,
    ContextRoutingPolicyHistorySnapshot,
)


@dataclass(frozen=True)
class ContextRoutingPolicyRefreshDecision:
    """Auditable gate for deciding whether policy evidence may be refreshed."""

    history_version: int
    experience_version: int
    latest_policy_version: int | None
    observations: int
    stability_rate: float
    freshness_age: int | None
    healthy: bool
    refresh_required: bool
    eligible: bool
    reason: str
    cooldown_remaining: int
    refresh_count: int
    refresh_budget: int | None

    def to_dict(self) -> dict:
        return asdict(self)


class ContextRoutingPolicyRefreshController:
    """Bounded refresh gate for routing-policy meta-memory.

    The controller decides whether an existing policy should be reevaluated.
    It never executes a model and never manufactures policy evidence. A caller
    must explicitly evaluate and record new evidence, then call record_refresh()
    to advance the controller state.

    Refresh is therefore:
      observe -> assess health -> gate -> evaluate -> record -> cool down
    """

    def __init__(
        self,
        *,
        min_observations: int = 2,
        min_stability: float = 1.0,
        max_age: int | None = None,
        min_interval: int = 0,
        refresh_budget: int | None = None,
    ) -> None:
        if min_observations < 1:
            raise ValueError("min_observations must be at least 1")
        if not 0.0 <= min_stability <= 1.0:
            raise ValueError("min_stability must be between 0.0 and 1.0")
        if max_age is not None and max_age < 0:
            raise ValueError("max_age must be non-negative")
        if min_interval < 0:
            raise ValueError("min_interval must be non-negative")
        if refresh_budget is not None and refresh_budget < 1:
            raise ValueError("refresh_budget must be at least 1 when provided")

        self.min_observations = int(min_observations)
        self.min_stability = float(min_stability)
        self.max_age = max_age
        self.min_interval = int(min_interval)
        self.refresh_budget = (
            None if refresh_budget is None else int(refresh_budget)
        )
        self._last_refresh_experience_version: int | None = None
        self._refresh_count = 0

    @property
    def last_refresh_experience_version(self) -> int | None:
        return self._last_refresh_experience_version

    @property
    def refresh_count(self) -> int:
        return self._refresh_count

    def _cooldown_remaining(self, experience_version: int) -> int:
        if self._last_refresh_experience_version is None:
            return 0
        elapsed = max(
            0,
            experience_version - self._last_refresh_experience_version,
        )
        return max(0, self.min_interval - elapsed)

    def _budget_available(self) -> bool:
        return (
            self.refresh_budget is None
            or self._refresh_count < self.refresh_budget
        )

    def _decision(
        self,
        *,
        health: ContextRoutingPolicyHealth,
        experience_version: int,
        history_version: int,
        latest_policy_version: int | None,
    ) -> ContextRoutingPolicyRefreshDecision:
        cooldown_remaining = self._cooldown_remaining(experience_version)
        budget_available = self._budget_available()

        if health.observations < self.min_observations:
            reason = "insufficient_evidence"
        elif not health.stable:
            reason = "unstable_policy"
        elif not health.fresh:
            reason = "stale_policy"
        elif cooldown_remaining > 0:
            reason = "refresh_cooldown"
        elif not budget_available:
            reason = "refresh_budget_exhausted"
        else:
            reason = "healthy"

        refresh_required = bool(health.refresh_recommended)
        eligible = (
            refresh_required
            and health.observations >= self.min_observations
            and cooldown_remaining == 0
            and budget_available
        )

        return ContextRoutingPolicyRefreshDecision(
            history_version=history_version,
            experience_version=experience_version,
            latest_policy_version=latest_policy_version,
            observations=health.observations,
            stability_rate=health.stability_rate,
            freshness_age=health.freshness_age,
            healthy=health.healthy,
            refresh_required=refresh_required,
            eligible=eligible,
            reason=reason,
            cooldown_remaining=cooldown_remaining,
            refresh_count=self._refresh_count,
            refresh_budget=self.refresh_budget,
        )

    def decide(
        self,
        snapshot: ContextRoutingPolicyHistorySnapshot,
        *,
        experience_version: int,
    ) -> ContextRoutingPolicyRefreshDecision:
        """Assess bounded policy history against the current experience version."""
        if not isinstance(snapshot, ContextRoutingPolicyHistorySnapshot):
            raise TypeError(
                "snapshot must be a ContextRoutingPolicyHistorySnapshot"
            )
        if experience_version < 0:
            raise ValueError("experience_version must be non-negative")

        health = snapshot.health(
            min_observations=self.min_observations,
            min_stability=self.min_stability,
            current_experience_version=experience_version,
            max_age=self.max_age,
        )
        return self._decision(
            health=health,
            experience_version=experience_version,
            history_version=snapshot.version,
            latest_policy_version=(
                snapshot.latest.version
                if snapshot.latest is not None
                else None
            ),
        )

    def decide_for_policy(
        self,
        *,
        policy_version: int,
        experience_version: int,
    ) -> ContextRoutingPolicyRefreshDecision:
        """Assess freshness for a single explicit policy-evidence snapshot."""
        if policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if experience_version < 0:
            raise ValueError("experience_version must be non-negative")

        freshness_age = max(0, experience_version - policy_version)
        fresh = (
            True
            if self.max_age is None
            else freshness_age <= self.max_age
        )
        health = ContextRoutingPolicyHealth(
            history_version=0,
            observations=1,
            mode="nearest",
            stability_rate=1.0,
            switch_count=0,
            min_stability=self.min_stability,
            freshness_age=freshness_age,
            max_age=self.max_age,
            stable=True,
            fresh=fresh,
            healthy=fresh,
            refresh_recommended=not fresh,
        )
        return self._decision(
            health=health,
            experience_version=experience_version,
            history_version=0,
            latest_policy_version=policy_version,
        )

    def record_refresh(self, *, experience_version: int) -> int:
        """Record one successfully completed refresh and advance its gate state."""
        if experience_version < 0:
            raise ValueError("experience_version must be non-negative")
        if self._last_refresh_experience_version is not None:
            if experience_version < self._last_refresh_experience_version:
                raise ValueError(
                    "refresh experience version cannot move backward"
                )
        if not self._budget_available():
            raise RuntimeError("refresh budget exhausted")
        if self._cooldown_remaining(experience_version) > 0:
            raise RuntimeError("refresh cooldown is still active")

        self._last_refresh_experience_version = experience_version
        self._refresh_count += 1
        return self._refresh_count

    def to_dict(self) -> dict:
        return {
            "min_observations": self.min_observations,
            "min_stability": self.min_stability,
            "max_age": self.max_age,
            "min_interval": self.min_interval,
            "refresh_budget": self.refresh_budget,
            "last_refresh_experience_version": self._last_refresh_experience_version,
            "refresh_count": self._refresh_count,
        }


__all__ = [
    "ContextRoutingPolicyRefreshDecision",
    "ContextRoutingPolicyRefreshController",
]
