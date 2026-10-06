from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .routing_policy import ContextRoutingPolicyEvidence


@dataclass(frozen=True)
class ContextRoutingPolicyStability:
    """Descriptive stability state for a bounded policy-evidence history.

    Stability describes how often the selected routing mode changes across
    recorded policy evaluations. It is not a quality score and does not infer
    causality, confidence, or model performance.
    """

    history_version: int
    observations: int
    latest_evidence_version: int | None
    latest_mode: str | None
    mode_counts: tuple[tuple[str, int], ...]
    switch_count: int
    stability_rate: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["mode_counts"] = {
            mode: count for mode, count in self.mode_counts
        }
        return payload


@dataclass(frozen=True)
class ContextRoutingPolicyHealth:
    """Descriptive health state for a bounded routing-policy history."""

    history_version: int
    observations: int
    mode: str
    stability_rate: float
    switch_count: int
    min_stability: float
    stable: bool
    refresh_recommended: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ContextRoutingPolicyHistorySnapshot:
    """Immutable view of previously recorded routing-policy evidence."""

    version: int
    evidences: tuple[ContextRoutingPolicyEvidence, ...]

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("version must be non-negative")
        if not isinstance(self.evidences, tuple):
            raise TypeError("evidences must be a tuple")
        previous: int | None = None
        for evidence in self.evidences:
            if not isinstance(evidence, ContextRoutingPolicyEvidence):
                raise TypeError(
                    "evidences must contain ContextRoutingPolicyEvidence"
                )
            if previous is not None and evidence.version < previous:
                raise ValueError(
                    "evidence versions must be non-decreasing"
                )
            previous = evidence.version

    @property
    def latest(self) -> ContextRoutingPolicyEvidence | None:
        return self.evidences[-1] if self.evidences else None

    def _window(self, window: int | None) -> tuple[ContextRoutingPolicyEvidence, ...]:
        if window is None:
            return self.evidences
        if window < 1:
            raise ValueError("window must be at least 1")
        return self.evidences[-int(window):]

    def select_mode(
        self,
        *,
        window: int | None = None,
        min_observations: int = 1,
    ) -> str:
        """Select a concrete mode by bounded historical consensus.

        Ties are resolved by the most recently observed mode. With insufficient
        evidence the conservative nearest baseline is returned.
        """
        if min_observations < 1:
            raise ValueError("min_observations must be at least 1")
        values = self._window(window)
        if len(values) < min_observations:
            return "nearest"

        counts: dict[str, int] = {"nearest": 0, "credit": 0}
        latest_index: dict[str, int] = {}
        for index, evidence in enumerate(values):
            counts[evidence.selected_mode] += 1
            latest_index[evidence.selected_mode] = index

        return min(
            (mode for mode, count in counts.items() if count > 0),
            key=lambda mode: (-counts[mode], -latest_index[mode], 0 if mode == "nearest" else 1),
        )

    def stability(self, *, window: int | None = None) -> ContextRoutingPolicyStability:
        values = self._window(window)
        latest = values[-1] if values else None
        counts = {
            "nearest": 0,
            "credit": 0,
        }
        switches = 0
        previous_mode: str | None = None
        for evidence in values:
            mode = evidence.selected_mode
            counts[mode] += 1
            if previous_mode is not None and previous_mode != mode:
                switches += 1
            previous_mode = mode

        observations = len(values)
        stability_rate = (
            1.0
            if observations <= 1
            else 1.0 - (switches / (observations - 1))
        )
        return ContextRoutingPolicyStability(
            history_version=self.version,
            observations=observations,
            latest_evidence_version=(
                latest.version if latest is not None else None
            ),
            latest_mode=(
                latest.selected_mode if latest is not None else None
            ),
            mode_counts=tuple(
                (mode, counts[mode])
                for mode in ("nearest", "credit")
                if counts[mode] > 0
            ),
            switch_count=switches,
            stability_rate=stability_rate,
        )

    def health(
        self,
        *,
        window: int | None = None,
        min_observations: int = 2,
        min_stability: float = 1.0,
    ) -> ContextRoutingPolicyHealth:
        """Assess whether historical policy selection is stable enough to use."""
        if min_observations < 1:
            raise ValueError("min_observations must be at least 1")
        if not 0.0 <= min_stability <= 1.0:
            raise ValueError("min_stability must be between 0.0 and 1.0")

        values = self._window(window)
        stability = self.stability(window=window)
        stable = (
            len(values) >= min_observations
            and stability.stability_rate >= min_stability
        )
        selected_mode = self.select_mode(
            window=window,
            min_observations=min_observations,
        )
        return ContextRoutingPolicyHealth(
            history_version=self.version,
            observations=len(values),
            mode=selected_mode,
            stability_rate=stability.stability_rate,
            switch_count=stability.switch_count,
            min_stability=min_stability,
            stable=stable,
            refresh_recommended=not stable,
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "evidences": [evidence.to_dict() for evidence in self.evidences],
        }


class ContextRoutingPolicyHistory:
    """Bounded meta-memory for previously evaluated routing policies.

    The history stores policy evidence separately from ordinary episodes. Its
    version is an internal history clock, while each evidence item retains the
    episodic snapshot version on which that evaluation was based.
    """

    def __init__(self, *, max_entries: int = 32) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = int(max_entries)
        self._version = 0
        self._evidences: tuple[ContextRoutingPolicyEvidence, ...] = ()

    @property
    def version(self) -> int:
        return self._version

    @property
    def latest(self) -> ContextRoutingPolicyEvidence | None:
        return self._evidences[-1] if self._evidences else None

    def record(self, evidence: ContextRoutingPolicyEvidence) -> int:
        """Append policy evidence and return the new history version."""
        if not isinstance(evidence, ContextRoutingPolicyEvidence):
            raise TypeError(
                "evidence must be a ContextRoutingPolicyEvidence"
            )
        if self._evidences and evidence.version < self._evidences[-1].version:
            raise ValueError(
                "policy evidence version cannot move backward"
            )

        self._evidences = (
            *self._evidences,
            evidence,
        )[-self.max_entries:]
        self._version += 1
        return self._version

    def snapshot(self) -> ContextRoutingPolicyHistorySnapshot:
        return ContextRoutingPolicyHistorySnapshot(
            version=self._version,
            evidences=self._evidences,
        )

    def select_mode(
        self,
        *,
        window: int | None = None,
        min_observations: int = 1,
    ) -> str:
        return self.snapshot().select_mode(
            window=window,
            min_observations=min_observations,
        )

    def stability(
        self,
        *,
        window: int | None = None,
    ) -> ContextRoutingPolicyStability:
        return self.snapshot().stability(window=window)

    def health(
        self,
        *,
        window: int | None = None,
        min_observations: int = 2,
        min_stability: float = 1.0,
    ) -> ContextRoutingPolicyHealth:
        return self.snapshot().health(
            window=window,
            min_observations=min_observations,
            min_stability=min_stability,
        )

    def to_dict(self) -> dict:
        snapshot = self.snapshot()
        payload = snapshot.to_dict()
        payload["max_entries"] = self.max_entries
        payload["stability"] = snapshot.stability().to_dict()
        payload["health"] = snapshot.health().to_dict()
        return payload


__all__ = [
    "ContextRoutingPolicyHealth",
    "ContextRoutingPolicyStability",
    "ContextRoutingPolicyHistorySnapshot",
    "ContextRoutingPolicyHistory",
]
