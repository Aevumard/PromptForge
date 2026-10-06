from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Sequence

from .memory import ContextEpisode


@dataclass(frozen=True)
class ContextMemoryTemporalEligibility:
    """Temporal admissibility for one stored memory item."""

    episode_id: str
    sequence_index: int
    age: int
    freshness_weight: float
    eligible: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ContextMemoryTemporalResult:
    """Audited temporal boundary over one ordered memory snapshot."""

    schema_version: str
    as_of_index: int
    included_ids: tuple[str, ...]
    future_excluded_ids: tuple[str, ...]
    stale_excluded_ids: tuple[str, ...]
    source_excluded_ids: tuple[str, ...]
    eligibilities: tuple[ContextMemoryTemporalEligibility, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["included_ids"] = list(self.included_ids)
        payload["future_excluded_ids"] = list(self.future_excluded_ids)
        payload["stale_excluded_ids"] = list(self.stale_excluded_ids)
        payload["source_excluded_ids"] = list(self.source_excluded_ids)
        payload["eligibilities"] = [
            item.to_dict() for item in self.eligibilities
        ]
        return payload


class ContextMemoryTemporalPolicy:
    """Optional hard temporal/source gate for episodic memory.

    Episode order is the explicit temporal clock used by the experience store.
    as_of_index is an inclusive point-in-time boundary for historical
    evaluation. Entries after it are treated as future evidence.

    This layer never infers source trust. Allowed/blocked sources are explicit
    caller configuration. A blocked or future memory item is excluded rather
    than silently reinterpreted.
    """

    def __init__(
        self,
        *,
        max_age: int | None = None,
        half_life: float = 8.0,
        min_freshness_weight: float = 0.0,
        allowed_sources: Sequence[str] = (),
        blocked_sources: Sequence[str] = (),
    ) -> None:
        if max_age is not None and (
            not isinstance(max_age, int)
            or isinstance(max_age, bool)
            or max_age < 0
        ):
            raise ValueError("max_age must be a non-negative integer or None")
        if not isfinite(float(half_life)) or half_life <= 0.0:
            raise ValueError("half_life must be finite and positive")
        if (
            not isinstance(min_freshness_weight, (int, float))
            or isinstance(min_freshness_weight, bool)
            or not isfinite(float(min_freshness_weight))
            or not 0.0 <= float(min_freshness_weight) <= 1.0
        ):
            raise ValueError("min_freshness_weight must be between 0.0 and 1.0")

        allowed = self._normalize_sources(allowed_sources, "allowed_sources")
        blocked = self._normalize_sources(blocked_sources, "blocked_sources")
        overlap = sorted(set(allowed).intersection(blocked))
        if overlap:
            raise ValueError(
                "a source cannot be both allowed and blocked: "
                + ",".join(overlap)
            )

        self.max_age = max_age
        self.half_life = float(half_life)
        self.min_freshness_weight = float(min_freshness_weight)
        self.allowed_sources = allowed
        self.blocked_sources = blocked

    @staticmethod
    def _normalize_sources(values: Sequence[str], field: str) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = str(value).strip()
            if not item:
                raise ValueError(f"{field} must not contain empty values")
            if item not in seen:
                normalized.append(item)
                seen.add(item)
        return tuple(normalized)

    def assess(
        self,
        episodes: Sequence[ContextEpisode],
        *,
        as_of_index: int | None = None,
    ) -> ContextMemoryTemporalResult:
        values = tuple(episodes)
        if any(not isinstance(item, ContextEpisode) for item in values):
            raise TypeError("episodes must contain only ContextEpisode values")
        if not values:
            raise ValueError("episodes must not be empty")

        resolved_as_of = (
            len(values) - 1 if as_of_index is None else int(as_of_index)
        )
        if resolved_as_of < 0 or resolved_as_of >= len(values):
            raise ValueError("as_of_index must be within the episode sequence")

        included: list[str] = []
        future_excluded: list[str] = []
        stale_excluded: list[str] = []
        source_excluded: list[str] = []
        eligibilities: list[ContextMemoryTemporalEligibility] = []

        for index, episode in enumerate(values):
            age = resolved_as_of - index
            freshness_weight = 2.0 ** (-max(age, 0) / self.half_life)

            if index > resolved_as_of:
                future_excluded.append(episode.episode_id)
                eligibilities.append(
                    ContextMemoryTemporalEligibility(
                        episode_id=episode.episode_id,
                        sequence_index=index,
                        age=age,
                        freshness_weight=freshness_weight,
                        eligible=False,
                        reason="future_memory",
                    )
                )
                continue

            if (
                episode.source in self.blocked_sources
                or (
                    self.allowed_sources
                    and episode.source not in self.allowed_sources
                )
            ):
                source_excluded.append(episode.episode_id)
                eligibilities.append(
                    ContextMemoryTemporalEligibility(
                        episode_id=episode.episode_id,
                        sequence_index=index,
                        age=age,
                        freshness_weight=freshness_weight,
                        eligible=False,
                        reason="source_not_admissible",
                    )
                )
                continue

            if self.max_age is not None and age > self.max_age:
                stale_excluded.append(episode.episode_id)
                eligibilities.append(
                    ContextMemoryTemporalEligibility(
                        episode_id=episode.episode_id,
                        sequence_index=index,
                        age=age,
                        freshness_weight=freshness_weight,
                        eligible=False,
                        reason="memory_too_old",
                    )
                )
                continue

            if freshness_weight < self.min_freshness_weight:
                stale_excluded.append(episode.episode_id)
                eligibilities.append(
                    ContextMemoryTemporalEligibility(
                        episode_id=episode.episode_id,
                        sequence_index=index,
                        age=age,
                        freshness_weight=freshness_weight,
                        eligible=False,
                        reason="freshness_weight_below_threshold",
                    )
                )
                continue

            included.append(episode.episode_id)
            eligibilities.append(
                ContextMemoryTemporalEligibility(
                    episode_id=episode.episode_id,
                    sequence_index=index,
                    age=age,
                    freshness_weight=freshness_weight,
                    eligible=True,
                    reason="eligible",
                )
            )

        return ContextMemoryTemporalResult(
            schema_version="memory-temporal.v1",
            as_of_index=resolved_as_of,
            included_ids=tuple(included),
            future_excluded_ids=tuple(future_excluded),
            stale_excluded_ids=tuple(stale_excluded),
            source_excluded_ids=tuple(source_excluded),
            eligibilities=tuple(eligibilities),
        )


__all__ = [
    "ContextMemoryTemporalEligibility",
    "ContextMemoryTemporalPolicy",
    "ContextMemoryTemporalResult",
]
