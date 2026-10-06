from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ConfidenceObservation:
    """Externally labeled historical confidence/outcome pair.

    confidence is a caller-supplied probability-like score. outcome is an
    externally measured binary result. PromptForge never derives either
    value from the prediction text.
    """

    observation_id: str
    confidence: float
    outcome: bool
    timestamp: float | None = None
    family: str = "default"
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.observation_id.strip():
            raise ValueError("observation_id must not be empty")
        if (
            not isinstance(self.confidence, (int, float))
            or isinstance(self.confidence, bool)
            or not isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("confidence must be a finite number between 0.0 and 1.0")
        if not isinstance(self.outcome, bool):
            raise ValueError("outcome must be a bool")
        if self.timestamp is not None and (
            not isinstance(self.timestamp, (int, float))
            or isinstance(self.timestamp, bool)
            or not isfinite(float(self.timestamp))
        ):
            raise ValueError("timestamp must be a finite number or None")
        if not self.family.strip():
            raise ValueError("family must not be empty")
        normalized = []
        seen = set()
        for value in self.tags:
            item = str(value).strip()
            if not item:
                raise ValueError("tags must not contain empty values")
            if item not in seen:
                normalized.append(item)
                seen.add(item)
        object.__setattr__(self, "tags", tuple(normalized))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConfidenceObservation":
        return cls(
            observation_id=str(value["observation_id"]),
            confidence=float(value["confidence"]),
            outcome=bool(value["outcome"]),
            timestamp=(
                None
                if value.get("timestamp") is None
                else float(value["timestamp"])
            ),
            family=str(value.get("family", "default")),
            tags=tuple(value.get("tags", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConfidenceCalibrationPolicy:
    """Deterministic, conservative calibration controls.

    Bins use a uniform [0, 1] partition. A bin must have enough observations
    before its empirical success rate is allowed to modify a live confidence.
    Centered smoothing prevents tiny bins from producing extreme rates.
    max_adjustment bounds how far calibration may move a score.
    """

    bins: int = 10
    min_bin_observations: int = 5
    min_total_observations: int = 20
    smoothing_strength: float = 2.0
    max_adjustment: float = 0.25
    cutoff: float | None = None
    allow_unknown_time: bool = False
    required_observation_ids: tuple[str, ...] = ()
    target_family: str | None = None
    min_family_observations: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.bins, int) or isinstance(self.bins, bool) or self.bins < 2:
            raise ValueError("bins must be an integer >= 2")
        if (
            not isinstance(self.min_bin_observations, int)
            or isinstance(self.min_bin_observations, bool)
            or self.min_bin_observations < 1
        ):
            raise ValueError("min_bin_observations must be an integer >= 1")
        if (
            not isinstance(self.min_total_observations, int)
            or isinstance(self.min_total_observations, bool)
            or self.min_total_observations < 1
        ):
            raise ValueError("min_total_observations must be an integer >= 1")
        for name, value in (
            ("smoothing_strength", self.smoothing_strength),
            ("max_adjustment", self.max_adjustment),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_adjustment > 1.0:
            raise ValueError("max_adjustment must be <= 1.0")
        if self.cutoff is not None and (
            not isinstance(self.cutoff, (int, float))
            or isinstance(self.cutoff, bool)
            or not isfinite(float(self.cutoff))
        ):
            raise ValueError("cutoff must be a finite number or None")
        if len(set(self.required_observation_ids)) != len(self.required_observation_ids):
            raise ValueError("required_observation_ids must be unique")
        if self.target_family is not None and not self.target_family.strip():
            raise ValueError("target_family must not be empty")
        if (
            not isinstance(self.min_family_observations, int)
            or isinstance(self.min_family_observations, bool)
            or self.min_family_observations < 1
        ):
            raise ValueError("min_family_observations must be an integer >= 1")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_observation_ids"] = list(self.required_observation_ids)
        return payload


@dataclass(frozen=True)
class ConfidenceCalibrationMetrics:
    """Descriptive metrics on the supplied historical calibration sample."""

    observation_count: int
    mean_confidence: float
    empirical_success_rate: float
    brier_score: float
    expected_calibration_error: float
    maximum_calibration_error: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Audited calibration result for one current confidence value."""

    raw_confidence: float
    calibrated_confidence: float
    adjustment: float
    status: str
    bin_index: int
    bin_count: int
    bin_empirical_rate: float | None
    global_empirical_rate: float | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class ConfidenceCalibrationModel:
    """Frozen calibration model fitted only from an explicit historical set."""

    schema_version: str
    policy: ConfidenceCalibrationPolicy
    included_observation_ids: tuple[str, ...]
    future_excluded_ids: tuple[str, ...]
    unknown_time_excluded_ids: tuple[str, ...]
    family_excluded_ids: tuple[str, ...]
    bin_counts: tuple[int, ...]
    bin_successes: tuple[int, ...]
    calibrated_bin_rates: tuple[float | None, ...]
    global_empirical_rate: float | None
    metrics: ConfidenceCalibrationMetrics

    def __post_init__(self) -> None:
        if len(self.bin_counts) != self.policy.bins:
            raise ValueError("bin_counts length must equal policy.bins")
        if len(self.bin_successes) != self.policy.bins:
            raise ValueError("bin_successes length must equal policy.bins")
        if len(self.calibrated_bin_rates) != self.policy.bins:
            raise ValueError("calibrated_bin_rates length must equal policy.bins")

    def _bin_index(self, confidence: float) -> int:
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise ValueError("confidence must be numeric")
        confidence = float(confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be a finite number between 0.0 and 1.0")
        return min(self.policy.bins - 1, int(confidence * self.policy.bins))

    def assess(self, confidence: float) -> ConfidenceAssessment:
        index = self._bin_index(confidence)
        raw = float(confidence)
        count = self.bin_counts[index]
        rate = self.calibrated_bin_rates[index]
        reasons: list[str] = []

        if self.metrics.observation_count < self.policy.min_total_observations:
            reasons.append("insufficient total calibration observations")
            return ConfidenceAssessment(
                raw_confidence=raw,
                calibrated_confidence=raw,
                adjustment=0.0,
                status="insufficient_total_data",
                bin_index=index,
                bin_count=count,
                bin_empirical_rate=rate,
                global_empirical_rate=self.global_empirical_rate,
                reasons=tuple(reasons),
            )

        if rate is None:
            if self.global_empirical_rate is None:
                reasons.append("no supported calibration evidence")
                status = "insufficient_bin_data"
                calibrated = raw
            else:
                reasons.append("bin lacks minimum observations; using global fallback")
                status = "global_fallback"
                calibrated = self.global_empirical_rate
        else:
            reasons.append("supported bin calibration applied")
            status = "calibrated"
            calibrated = rate

        delta = calibrated - raw
        bounded_delta = max(
            -self.policy.max_adjustment,
            min(self.policy.max_adjustment, delta),
        )
        calibrated = raw + bounded_delta
        if bounded_delta != delta:
            reasons.append("adjustment capped by max_adjustment")

        return ConfidenceAssessment(
            raw_confidence=raw,
            calibrated_confidence=float(calibrated),
            adjustment=float(bounded_delta),
            status=status,
            bin_index=index,
            bin_count=count,
            bin_empirical_rate=rate,
            global_empirical_rate=self.global_empirical_rate,
            reasons=tuple(reasons),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy.to_dict(),
            "included_observation_ids": list(self.included_observation_ids),
            "future_excluded_ids": list(self.future_excluded_ids),
            "unknown_time_excluded_ids": list(self.unknown_time_excluded_ids),
            "family_excluded_ids": list(self.family_excluded_ids),
            "bin_counts": list(self.bin_counts),
            "bin_successes": list(self.bin_successes),
            "calibrated_bin_rates": list(self.calibrated_bin_rates),
            "global_empirical_rate": self.global_empirical_rate,
            "metrics": self.metrics.to_dict(),
        }


class ConfidenceCalibrator:
    """Fit and apply a leakage-aware, conservative confidence calibrator.

    The calibrator is intentionally separate from action selection. It uses
    only caller-supplied historical labels and never uses the outcome of the
    confidence value being assessed.
    """

    def fit(
        self,
        observations: Sequence[ConfidenceObservation | Mapping[str, Any]],
        *,
        policy: ConfidenceCalibrationPolicy | None = None,
    ) -> ConfidenceCalibrationModel:
        active_policy = policy or ConfidenceCalibrationPolicy()
        normalized = tuple(
            item
            if isinstance(item, ConfidenceObservation)
            else ConfidenceObservation.from_mapping(item)
            for item in observations
        )
        if not normalized:
            raise ValueError("observations must contain at least one item")

        seen: set[str] = set()
        included: list[ConfidenceObservation] = []
        future_excluded: list[str] = []
        unknown_excluded: list[str] = []
        family_excluded: list[str] = []

        for item in normalized:
            if item.observation_id in seen:
                raise ValueError("duplicate observation_id: " + item.observation_id)
            seen.add(item.observation_id)

            if active_policy.target_family is not None and item.family != active_policy.target_family:
                family_excluded.append(item.observation_id)
                continue

            if active_policy.cutoff is not None:
                if item.timestamp is None:
                    if not active_policy.allow_unknown_time:
                        unknown_excluded.append(item.observation_id)
                        continue
                elif item.timestamp > active_policy.cutoff:
                    future_excluded.append(item.observation_id)
                    continue

            included.append(item)

        included_ids = {item.observation_id for item in included}
        if (
            len(included) < active_policy.min_family_observations
            and active_policy.target_family is not None
        ):
            raise ValueError(
                "target family does not have enough calibration observations: "
                + active_policy.target_family
            )
        missing_required = sorted(
            set(active_policy.required_observation_ids) - seen
        )
        if missing_required:
            raise ValueError(
                "required calibration observations are missing: "
                + ",".join(missing_required)
            )

        excluded_required = sorted(
            set(active_policy.required_observation_ids) - included_ids
        )
        if excluded_required:
            raise ValueError(
                "required calibration observations cross the active boundary: "
                + ",".join(excluded_required)
            )

        ordered = tuple(sorted(included, key=lambda item: item.observation_id))
        counts = [0 for _ in range(active_policy.bins)]
        successes = [0 for _ in range(active_policy.bins)]

        for item in ordered:
            index = min(
                active_policy.bins - 1,
                int(item.confidence * active_policy.bins),
            )
            counts[index] += 1
            successes[index] += int(item.outcome)

        supported = [
            count >= active_policy.min_bin_observations
            for count in counts
        ]
        raw_rates: list[float | None] = []
        for index, count in enumerate(counts):
            if not supported[index]:
                raw_rates.append(None)
                continue
            effective = count + active_policy.smoothing_strength
            success = (
                successes[index]
                + active_policy.smoothing_strength * 0.5
            )
            raw_rates.append(success / effective)

        calibrated = list(raw_rates)
        blocks: list[dict[str, float | int]] = []
        for index, rate in enumerate(raw_rates):
            if rate is None:
                continue
            weight = counts[index] + active_policy.smoothing_strength
            blocks.append(
                {"start": index, "end": index, "value": rate, "weight": weight}
            )
            while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
                right = blocks.pop()
                left = blocks.pop()
                weight_total = left["weight"] + right["weight"]
                merged = {
                    "start": left["start"],
                    "end": right["end"],
                    "weight": weight_total,
                    "value": (
                        left["value"] * left["weight"]
                        + right["value"] * right["weight"]
                    ) / weight_total,
                }
                blocks.append(merged)

        for block in blocks:
            value = float(block["value"])
            for index in range(int(block["start"]), int(block["end"]) + 1):
                calibrated[index] = value

        total = len(ordered)
        global_rate = (
            sum(int(item.outcome) for item in ordered) / total
            if total
            else None
        )
        mean_confidence = (
            sum(item.confidence for item in ordered) / total
            if total
            else 0.0
        )
        brier = (
            sum((item.confidence - int(item.outcome)) ** 2 for item in ordered) / total
            if total
            else 0.0
        )

        bin_groups: list[list[ConfidenceObservation]] = [
            [] for _ in range(active_policy.bins)
        ]
        for item in ordered:
            index = min(
                active_policy.bins - 1,
                int(item.confidence * active_policy.bins),
            )
            bin_groups[index].append(item)

        ece = 0.0
        mce = 0.0
        for group in bin_groups:
            if not group:
                continue
            avg_conf = sum(item.confidence for item in group) / len(group)
            avg_outcome = sum(int(item.outcome) for item in group) / len(group)
            gap = abs(avg_conf - avg_outcome)
            ece += (len(group) / total) * gap
            mce = max(mce, gap)

        metrics = ConfidenceCalibrationMetrics(
            observation_count=total,
            mean_confidence=float(mean_confidence),
            empirical_success_rate=float(global_rate or 0.0),
            brier_score=float(brier),
            expected_calibration_error=float(ece),
            maximum_calibration_error=float(mce),
        )

        return ConfidenceCalibrationModel(
            schema_version="confidence-calibration.v1",
            policy=active_policy,
            included_observation_ids=tuple(item.observation_id for item in ordered),
            future_excluded_ids=tuple(sorted(future_excluded)),
            unknown_time_excluded_ids=tuple(sorted(unknown_excluded)),
            family_excluded_ids=tuple(sorted(family_excluded)),
            bin_counts=tuple(counts),
            bin_successes=tuple(successes),
            calibrated_bin_rates=tuple(calibrated),
            global_empirical_rate=(
                float(global_rate) if global_rate is not None else None
            ),
            metrics=metrics,
        )

    def assess(
        self,
        confidence: float,
        observations: Sequence[ConfidenceObservation | Mapping[str, Any]],
        *,
        policy: ConfidenceCalibrationPolicy | None = None,
    ) -> ConfidenceAssessment:
        return self.fit(observations, policy=policy).assess(confidence)


__all__ = [
    "ConfidenceObservation",
    "ConfidenceCalibrationPolicy",
    "ConfidenceCalibrationMetrics",
    "ConfidenceAssessment",
    "ConfidenceCalibrationModel",
    "ConfidenceCalibrator",
]
