"""Public PromptForge core API."""

__version__ = "0.1.0"

from .core import (
    POLICY_BUDGET_CONSTRAINED,
    POLICY_MINIMAL,
    POLICY_MINIMAL_SERIALIZED_CONTEXT,
    inspect,
    prepare_context,
)


from .adaptive import (
    REGIME_COMPACT,
    REGIME_DEEP_HIERARCHICAL,
    REGIME_FRAGMENTED,
    REGIME_HUB_DOMINATED,
    REGIME_MIXED,
    REGIME_WIDE_SPARSE,
    ContextControllerDecision,
    ContextPortfolioController,
    ContextRegimeSignature,
    ContextStrategyRecommendation,
    ContextTopologyProfile,
    ContextTopologyProfiler,
    ContextTrajectoryMonitor,
    ContextTrajectoryState,
    HeuristicContextRegimeSelector,
    rank_context_candidates,
)

__all__ = [
    "POLICY_BUDGET_CONSTRAINED",
    "POLICY_MINIMAL",
    "POLICY_MINIMAL_SERIALIZED_CONTEXT",
    "inspect",
    "prepare_context",
    "ContextTopologyProfiler",
    "ContextTopologyProfile",
    "ContextRegimeSignature",
    "ContextStrategyRecommendation",
    "HeuristicContextRegimeSelector",
    "rank_context_candidates",
    "ContextTrajectoryMonitor",
    "ContextTrajectoryState",
    "ContextPortfolioController",
    "ContextControllerDecision",
    "REGIME_COMPACT",
    "REGIME_DEEP_HIERARCHICAL",
    "REGIME_FRAGMENTED",
    "REGIME_HUB_DOMINATED",
    "REGIME_MIXED",
    "REGIME_WIDE_SPARSE",
]
