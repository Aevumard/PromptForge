from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Any

from .core import MISSING, estimate_tokens, get_path, normalize_required_paths, serialize_context


REGIME_HUB_DOMINATED = "hub_dominated"
REGIME_DEEP_HIERARCHICAL = "deep_hierarchical"
REGIME_FRAGMENTED = "fragmented"
REGIME_WIDE_SPARSE = "wide_sparse"
REGIME_COMPACT = "compact"
REGIME_MIXED = "mixed"


@dataclass(frozen=True)
class ContextTopologyProfile:
    """Interpretable structural signature for arbitrary nested context.

    The profile describes the shape of the context tree and its relationship
    to the required fields. It is descriptive; it is not a model-quality
    predictor.
    """

    schema_version: str
    top_level_fields: int
    node_count: int
    edge_count: int
    leaf_fields: int
    max_depth: int
    avg_depth: float
    depth_std: float
    max_branching: int
    avg_branching: float
    hub_ratio: float
    required_count: int
    required_present: int
    required_missing: tuple[str, ...]
    required_node_count: int
    required_node_ratio: float
    required_leaf_count: int
    required_leaf_ratio: float
    active_root_count: int
    active_root_ratio: float
    boundary_edges: int
    boundary_pressure: float
    serialized_chars: int
    estimated_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContextTopologyProfiler:
    """Compute a cheap, deterministic topology signature for nested mappings."""

    def profile(
        self,
        data: Mapping[str, Any],
        required: Iterable[str] | None = None,
    ) -> ContextTopologyProfile:
        if not isinstance(data, Mapping):
            raise TypeError("data must be a mapping")

        required_paths = normalize_required_paths(required or [])
        serialized = serialize_context(data)

        nodes: dict[str, Any] = {}
        parents: dict[str, str | None] = {}
        children: dict[str, list[str]] = {}
        leaves: set[str] = set()
        depths: list[int] = []

        def visit(value: Any, prefix: str, parent: str | None) -> None:
            if not prefix:
                return

            nodes[prefix] = value
            parents[prefix] = parent
            depths.append(prefix.count(".") + 1)

            if not isinstance(value, Mapping) or not value:
                leaves.add(prefix)
                children.setdefault(prefix, [])
                return

            child_paths: list[str] = []
            for key, child in value.items():
                path = f"{prefix}.{key}"
                child_paths.append(path)
                visit(child, path, prefix)
            children[prefix] = child_paths

        top_level = list(data.keys())
        for key, value in data.items():
            visit(value, str(key), None)

        branching_values = [
            len(items)
            for path, items in children.items()
            if path in nodes and items
        ]

        node_count = len(nodes)
        edge_count = sum(len(items) for items in children.values())
        leaf_count = len(leaves)

        if depths:
            avg_depth = sum(depths) / len(depths)
            depth_std = sqrt(
                sum((depth - avg_depth) ** 2 for depth in depths) / len(depths)
            )
        else:
            avg_depth = 0.0
            depth_std = 0.0

        max_branching = max(branching_values, default=0)
        avg_branching = (
            sum(branching_values) / len(branching_values)
            if branching_values
            else 0.0
        )
        hub_ratio = (
            max_branching / avg_branching
            if avg_branching > 0.0
            else 0.0
        )

        missing = tuple(
            path for path in required_paths if get_path(data, path) is MISSING
        )
        present_required = [path for path in required_paths if path not in missing]

        closure: set[str] = set()
        for path in present_required:
            current = path
            value = get_path(data, path)
            if isinstance(value, Mapping) and value:
                prefix = path + "."
                closure.update(
                    node for node in nodes
                    if node == path or node.startswith(prefix)
                )
            else:
                closure.add(path)

            while current in parents and parents[current] is not None:
                parent = parents[current]
                if parent is None:
                    break
                closure.add(parent)
                current = parent

        required_leaf_count = sum(
            1
            for leaf in leaves
            if any(leaf == path or leaf.startswith(path + ".") for path in present_required)
        )

        boundary_edges = 0
        for node, child_paths in children.items():
            node_in = node in closure
            for child in child_paths:
                if node_in != (child in closure):
                    boundary_edges += 1

        active_roots = {
            path.split(".", 1)[0]
            for path in present_required
            if path
        }

        required_node_ratio = (
            len(closure) / node_count if node_count else 0.0
        )
        required_leaf_ratio = (
            required_leaf_count / leaf_count if leaf_count else 0.0
        )
        active_root_ratio = (
            len(active_roots) / len(top_level) if top_level else 0.0
        )
        boundary_pressure = (
            boundary_edges / edge_count if edge_count else 0.0
        )

        return ContextTopologyProfile(
            schema_version="context-topology.v1",
            top_level_fields=len(top_level),
            node_count=node_count,
            edge_count=edge_count,
            leaf_fields=leaf_count,
            max_depth=max(depths, default=0),
            avg_depth=avg_depth,
            depth_std=depth_std,
            max_branching=max_branching,
            avg_branching=avg_branching,
            hub_ratio=hub_ratio,
            required_count=len(required_paths),
            required_present=len(present_required),
            required_missing=missing,
            required_node_count=len(closure),
            required_node_ratio=required_node_ratio,
            required_leaf_count=required_leaf_count,
            required_leaf_ratio=required_leaf_ratio,
            active_root_count=len(active_roots),
            active_root_ratio=active_root_ratio,
            boundary_edges=boundary_edges,
            boundary_pressure=boundary_pressure,
            serialized_chars=len(serialized),
            estimated_tokens=estimate_tokens(serialized),
        )


@dataclass(frozen=True)
class ContextRegimeSignature:
    """Transparent regime flags derived from a topology profile."""

    primary_regime: str
    flags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HeuristicContextRegimeSelector:
    """Descriptive routing baseline.

    Thresholds are explicit heuristics. They are not presented as validated
    universal classifiers for model behavior.
    """

    def __init__(
        self,
        *,
        hub_ratio_threshold: float = 3.0,
        max_branching_threshold: int = 8,
        deep_depth_threshold: int = 4,
        avg_depth_threshold: float = 2.5,
        fragmented_boundary_threshold: float = 0.25,
        fragmented_root_ratio: float = 0.50,
        wide_field_threshold: int = 8,
        wide_required_ratio: float = 0.25,
        compact_node_threshold: int = 12,
        compact_depth_threshold: int = 2,
    ) -> None:
        self.hub_ratio_threshold = hub_ratio_threshold
        self.max_branching_threshold = max_branching_threshold
        self.deep_depth_threshold = deep_depth_threshold
        self.avg_depth_threshold = avg_depth_threshold
        self.fragmented_boundary_threshold = fragmented_boundary_threshold
        self.fragmented_root_ratio = fragmented_root_ratio
        self.wide_field_threshold = wide_field_threshold
        self.wide_required_ratio = wide_required_ratio
        self.compact_node_threshold = compact_node_threshold
        self.compact_depth_threshold = compact_depth_threshold

    def classify(self, profile: ContextTopologyProfile) -> ContextRegimeSignature:
        flags: list[str] = []

        if (
            profile.hub_ratio >= self.hub_ratio_threshold
            or profile.max_branching >= self.max_branching_threshold
        ):
            flags.append(REGIME_HUB_DOMINATED)

        if (
            profile.max_depth >= self.deep_depth_threshold
            or profile.avg_depth >= self.avg_depth_threshold
        ):
            flags.append(REGIME_DEEP_HIERARCHICAL)

        if (
            profile.boundary_pressure >= self.fragmented_boundary_threshold
            and profile.active_root_ratio >= self.fragmented_root_ratio
        ):
            flags.append(REGIME_FRAGMENTED)

        if (
            profile.top_level_fields >= self.wide_field_threshold
            and profile.required_leaf_ratio <= self.wide_required_ratio
        ):
            flags.append(REGIME_WIDE_SPARSE)

        if (
            profile.node_count <= self.compact_node_threshold
            and profile.max_depth <= self.compact_depth_threshold
        ):
            flags.append(REGIME_COMPACT)

        if not flags:
            flags.append(REGIME_MIXED)

        priority = (
            REGIME_HUB_DOMINATED,
            REGIME_DEEP_HIERARCHICAL,
            REGIME_FRAGMENTED,
            REGIME_WIDE_SPARSE,
            REGIME_COMPACT,
            REGIME_MIXED,
        )
        primary = next(item for item in priority if item in flags)
        return ContextRegimeSignature(primary_regime=primary, flags=tuple(flags))

    def recommend(
        self,
        profile: ContextTopologyProfile,
    ) -> "ContextStrategyRecommendation":
        signature = self.classify(profile)
        recommendations = {
            REGIME_HUB_DOMINATED: (
                ("selection_only", "selection_representation_B", "selection_representation"),
                "Dense local branching makes bounded comparison of compact structural representations useful.",
            ),
            REGIME_DEEP_HIERARCHICAL: (
                ("selection_only", "selection_representation_B", "selection_representation"),
                "Deep hierarchy increases structural depth; keep compact selection first and probe alternate encodings explicitly.",
            ),
            REGIME_FRAGMENTED: (
                ("selection_only", "selection_representation_B", "noop"),
                "Required fields are structurally dispersed; preserve the hard semantic boundary and compare locality-preserving alternatives by observation.",
            ),
            REGIME_WIDE_SPARSE: (
                ("selection_only", "selection_representation_B"),
                "A wide sparse context favors aggressive semantic selection before spending budget on alternate representations.",
            ),
            REGIME_COMPACT: (
                ("selection_only", "noop"),
                "The context is structurally small; additional adaptation is unlikely to be justified by topology alone.",
            ),
            REGIME_MIXED: (
                ("selection_only", "selection_representation_B", "noop"),
                "No dominant structural signature was detected; keep a small, explicit candidate portfolio.",
            ),
        }
        preferred_arms, rationale = recommendations[signature.primary_regime]
        return ContextStrategyRecommendation(
            schema_version="context-strategy.v1",
            regime=signature.primary_regime,
            regime_flags=signature.flags,
            preferred_arms=preferred_arms,
            rationale=rationale,
        )


@dataclass(frozen=True)
class ContextStrategyRecommendation:
    schema_version: str
    regime: str
    regime_flags: tuple[str, ...]
    preferred_arms: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextTrajectoryState:
    """Online search-state signal from an externally measured trace."""

    state: str
    iteration: int
    cost: float
    stagnation_length: int
    trailing_inactive_iterations: int
    acceptance_rate: float
    recent_gain: float
    gain_decay: float | None
    recent_accepted: int
    recent_rejected: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContextTrajectoryMonitor:
    """State monitor using only observations available up to the current step."""

    def __init__(
        self,
        *,
        window: int = 3,
        stagnation_patience: int = 3,
        extinction_patience: int = 4,
        exploration_acceptance_rate: float = 0.35,
        exploration_gain_floor: float = 1e-3,
        premature_window: int = 3,
        premature_min_accepts: int = 1,
        tolerance: float = 1e-12,
    ) -> None:
        if window < 1:
            raise ValueError("window must be at least 1")
        if stagnation_patience < 1:
            raise ValueError("stagnation_patience must be at least 1")
        if extinction_patience < 1:
            raise ValueError("extinction_patience must be at least 1")
        if premature_window < 1:
            raise ValueError("premature_window must be at least 1")
        if premature_min_accepts < 0:
            raise ValueError("premature_min_accepts must be non-negative")
        if not 0.0 <= exploration_acceptance_rate <= 1.0:
            raise ValueError("exploration_acceptance_rate must be in [0,1]")
        if exploration_gain_floor < 0.0:
            raise ValueError("exploration_gain_floor must be non-negative")
        self.window = window
        self.stagnation_patience = stagnation_patience
        self.extinction_patience = extinction_patience
        self.exploration_acceptance_rate = exploration_acceptance_rate
        self.exploration_gain_floor = exploration_gain_floor
        self.premature_window = premature_window
        self.premature_min_accepts = premature_min_accepts
        self.tolerance = tolerance

    def observe(
        self,
        trace: Sequence[Mapping[str, float | int]],
    ) -> ContextTrajectoryState:
        if not trace:
            raise ValueError("trace must contain at least one observation")

        required = {"cost", "accepted", "rejected"}
        missing = required.difference(trace[0])
        if missing:
            raise ValueError(f"trace is missing required fields: {sorted(missing)}")

        rows = list(trace)
        costs = [float(row["cost"]) for row in rows]
        accepted = [int(row["accepted"]) for row in rows]
        rejected = [int(row["rejected"]) for row in rows]

        current_cost = costs[-1]
        changes = [costs[i - 1] - costs[i] for i in range(1, len(costs))]
        recent_changes = changes[-self.window:]
        recent_gain = sum(recent_changes)
        recent_accepted = sum(accepted[-self.window:])
        recent_rejected = sum(rejected[-self.window:])
        decisions = recent_accepted + recent_rejected
        acceptance_rate = recent_accepted / decisions if decisions else 0.0

        stagnation = 0
        for gain in reversed(changes):
            if gain <= self.tolerance:
                stagnation += 1
            else:
                break

        trailing_inactive = 0
        for count in reversed(accepted):
            if count == 0:
                trailing_inactive += 1
            else:
                break

        gain_decay: float | None = None
        if len(changes) >= 2 * self.window:
            previous = sum(changes[-2 * self.window : -self.window])
            if previous > self.tolerance:
                ratio = max(0.0, min(recent_gain / previous, 1.0))
                gain_decay = 1.0 - ratio

        early = accepted[: self.premature_window]
        premature = (
            len(rows) >= self.premature_window
            and sum(early) < self.premature_min_accepts
        )
        exploratory_excess = (
            recent_accepted > 0
            and acceptance_rate >= self.exploration_acceptance_rate
            and recent_gain <= self.exploration_gain_floor
        )
        extinct = trailing_inactive >= self.extinction_patience
        stagnating = stagnation >= self.stagnation_patience

        if premature:
            state = "premature_collapse"
        elif extinct:
            state = "extinct"
        elif exploratory_excess:
            state = "exploratory_excess"
        elif stagnating:
            state = "stagnating"
        else:
            state = "active"

        return ContextTrajectoryState(
            state=state,
            iteration=len(rows) - 1,
            cost=current_cost,
            stagnation_length=stagnation,
            trailing_inactive_iterations=trailing_inactive,
            acceptance_rate=acceptance_rate,
            recent_gain=recent_gain,
            gain_decay=gain_decay,
            recent_accepted=recent_accepted,
            recent_rejected=recent_rejected,
        )


@dataclass(frozen=True)
class ContextControllerDecision:
    action: str
    target_strategy: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContextPortfolioController:
    """Budget-aware continue/intensify/switch/stop policy.

    Scores are externally observed task-level costs. The controller never
    accesses a hidden oracle and never assumes that smaller context is better.
    """

    def __init__(
        self,
        *,
        switch_min_relative_gain: float = 0.01,
        stop_budget_fraction: float = 0.05,
        switch_budget_fraction: float = 0.15,
    ) -> None:
        if switch_min_relative_gain < 0.0:
            raise ValueError("switch_min_relative_gain must be non-negative")
        if not 0.0 <= stop_budget_fraction <= 1.0:
            raise ValueError("stop_budget_fraction must be in [0,1]")
        if not 0.0 <= switch_budget_fraction <= 1.0:
            raise ValueError("switch_budget_fraction must be in [0,1]")
        self.switch_min_relative_gain = switch_min_relative_gain
        self.stop_budget_fraction = stop_budget_fraction
        self.switch_budget_fraction = switch_budget_fraction

    def decide(
        self,
        *,
        trajectory: ContextTrajectoryState,
        current_strategy: str,
        current_cost: float,
        probe_scores: Mapping[str, float] | None = None,
        remaining_budget_fraction: float = 1.0,
    ) -> ContextControllerDecision:
        if remaining_budget_fraction < 0.0:
            raise ValueError("remaining_budget_fraction must be non-negative")
        if current_cost < 0.0:
            raise ValueError("current_cost must be non-negative")

        probes = {
            str(strategy): float(score)
            for strategy, score in (probe_scores or {}).items()
            if isfinite(float(score)) and float(score) >= 0.0
        }
        better = {
            strategy: score
            for strategy, score in probes.items()
            if strategy != current_strategy
            and score <= current_cost * (1.0 - self.switch_min_relative_gain)
        }

        if better and remaining_budget_fraction >= self.switch_budget_fraction:
            target = min(better, key=lambda item: (better[item], item))
            return ContextControllerDecision(
                action="switch",
                target_strategy=target,
                reason="bounded probe observed a materially lower task-level cost",
            )

        if trajectory.state == "active":
            return ContextControllerDecision(
                action="continue",
                target_strategy=current_strategy,
                reason="recent prefix still shows productive progress",
            )

        if trajectory.state == "premature_collapse":
            if probes and remaining_budget_fraction >= self.switch_budget_fraction:
                target = min(probes, key=lambda item: (probes[item], item))
                if target != current_strategy:
                    return ContextControllerDecision(
                        action="switch",
                        target_strategy=target,
                        reason="early collapse signal plus an available alternate probe",
                    )
            if remaining_budget_fraction > self.stop_budget_fraction:
                return ContextControllerDecision(
                    action="intensify",
                    target_strategy=current_strategy,
                    reason="early collapse signal without a sufficiently strong switch witness",
                )

        if trajectory.state == "stagnating":
            if remaining_budget_fraction > self.stop_budget_fraction:
                return ContextControllerDecision(
                    action="intensify",
                    target_strategy=current_strategy,
                    reason="stagnation threshold reached and budget remains",
                )

        if trajectory.state == "exploratory_excess":
            if better:
                target = min(better, key=lambda item: (better[item], item))
                return ContextControllerDecision(
                    action="switch",
                    target_strategy=target,
                    reason="high acceptance with weak gain and a better bounded probe",
                )
            return ContextControllerDecision(
                action="stop",
                target_strategy=current_strategy,
                reason="high search activity is not producing sufficient gain",
            )

        if trajectory.state == "extinct":
            return ContextControllerDecision(
                action="stop",
                target_strategy=current_strategy,
                reason="accepted changes have been absent for the configured extinction window",
            )

        if remaining_budget_fraction <= self.stop_budget_fraction:
            return ContextControllerDecision(
                action="stop",
                target_strategy=current_strategy,
                reason="remaining budget is below the stop threshold",
            )

        return ContextControllerDecision(
            action="continue",
            target_strategy=current_strategy,
            reason="no stronger transition signal is available",
        )


def rank_context_candidates(
    candidates: Sequence[Mapping[str, Any]],
    recommendation: ContextStrategyRecommendation,
    *,
    budget_tokens: int | None = None,
) -> tuple[str, ...]:
    """Rank already-evaluated candidates for an explicit bounded probe.

    The function does not claim that the preferred arm is globally better.
    It simply orders feasible candidates by structural recommendation, then by
    measured serialized size and deterministic arm order.
    """
    preference = {
        arm_id: index
        for index, arm_id in enumerate(recommendation.preferred_arms)
    }

    feasible = []
    for index, candidate in enumerate(candidates):
        arm_id = str(candidate.get("arm_id", ""))
        if not candidate.get("required_values_preserved", False):
            continue
        if budget_tokens is not None:
            try:
                if int(candidate.get("estimated_tokens", 10**12)) > budget_tokens:
                    continue
            except (TypeError, ValueError):
                continue

        feasible.append(
            (
                preference.get(arm_id, len(preference)),
                int(candidate.get("context_chars", 10**12)),
                index,
                arm_id,
            )
        )

    return tuple(
        arm_id
        for _, _, _, arm_id in sorted(
            feasible,
            key=lambda item: (item[0], item[1], item[2], item[3]),
        )
    )


__all__ = [
    "REGIME_COMPACT",
    "REGIME_DEEP_HIERARCHICAL",
    "REGIME_FRAGMENTED",
    "REGIME_HUB_DOMINATED",
    "REGIME_MIXED",
    "REGIME_WIDE_SPARSE",
    "ContextControllerDecision",
    "ContextPortfolioController",
    "ContextRegimeSignature",
    "ContextStrategyRecommendation",
    "ContextTopologyProfile",
    "ContextTopologyProfiler",
    "ContextTrajectoryMonitor",
    "ContextTrajectoryState",
    "HeuristicContextRegimeSelector",
    "rank_context_candidates",
]
