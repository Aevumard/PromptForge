from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Mapping, Sequence

from promptforge import (
    ActionCandidate,
    ActionExecutionGuard,
    ActionExecutionRecord,
    ActionPolicy,
    ContextBlock,
    ContextBudgetPolicy,
    EpistemicContextCompiler,
    EpistemicContextPolicy,
    EvidenceRecord,
    HumanReviewRecord,
    PriorityAssessment,
    TriageState,
    ToolOutputItem,
    ToolOutputTrimmer,
    ToolOutputTrimPolicy,
    UncertaintyActionGate,
    plan_context,
)


@dataclass(frozen=True)
class BenchmarkResult:
    case_id: str
    category: str
    passed: bool
    checks: Mapping[str, bool]
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "passed": self.passed,
            "checks": dict(self.checks),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class BenchmarkReport:
    schema_version: str
    results: Sequence[BenchmarkResult]

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def passed_cases(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def failed_cases(self) -> int:
        return self.total_cases - self.passed_cases

    @property
    def pass_rate(self) -> float:
        return (
            self.passed_cases / self.total_cases
            if self.total_cases
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "summary": {
                "total_cases": self.total_cases,
                "passed_cases": self.passed_cases,
                "failed_cases": self.failed_cases,
                "pass_rate": self.pass_rate,
            },
            "results": [result.to_dict() for result in self.results],
        }


def _result(
    case_id: str,
    category: str,
    checks: Mapping[str, bool],
    details: Mapping[str, Any],
) -> BenchmarkResult:
    return BenchmarkResult(
        case_id=case_id,
        category=category,
        passed=all(checks.values()),
        checks=checks,
        details=details,
    )


def _action_policy() -> ActionPolicy:
    return ActionPolicy(
        require_support_anchors=True,
        require_support_stance=True,
        require_support_tag_match=True,
        require_support_quality=True,
        require_support_provenance_diversity=True,
        min_distinct_support_sources=2,
    )


def _case_priority_vs_actionability() -> BenchmarkResult:
    compiler = EpistemicContextCompiler()
    evidence = compiler.compile(
        [
            EvidenceRecord(
                "gateway",
                "Gateway confirms transaction exists.",
                stance="supports",
                source="gateway",
                timestamp=10,
                relevance=1.0,
                reliability=0.95,
                tags=("payment",),
            ),
            EvidenceRecord(
                "ledger",
                "Ledger contradicts the requested refund amount.",
                stance="contradicts",
                source="ledger",
                timestamp=11,
                relevance=1.0,
                reliability=0.95,
                tags=("payment",),
            ),
        ]
    )
    actions = [
        ActionCandidate(
            "refund",
            "Issue the refund now.",
            evidence_support=0.99,
            reversibility=0.4,
            downside=0.8,
            support_evidence_ids=("gateway", "ledger"),
            support_evidence_tags=("payment",),
        ),
        ActionCandidate(
            "human_review",
            "Escalate the contradiction for human review.",
            evidence_support=0.70,
            reversibility=1.0,
            downside=0.1,
            support_evidence_ids=("gateway",),
            support_evidence_tags=("payment",),
        ),
    ]
    decision = UncertaintyActionGate(_action_policy()).decide(
        actions,
        evidence=evidence,
    )
    priority = PriorityAssessment(
        urgency=1.0,
        importance=0.3,
        priority_band="P0",
        rationale=("SLA is near breach.",),
    )

    checks = {
        "priority_remains_p0": priority.priority_band == "P0",
        "refund_is_blocked": "refund" in decision.blocked_action_ids,
        "review_is_selected": decision.selected_action_id == "human_review",
    }
    return _result(
        "priority-actionability-separation",
        "decision-separation",
        checks,
        {
            "priority_band": priority.priority_band,
            "selected_action_id": decision.selected_action_id,
            "blocked_action_ids": list(decision.blocked_action_ids),
            "reasons": {
                key: list(value) for key, value in decision.reasons.items()
            },
        },
    )


def _case_future_evidence() -> BenchmarkResult:
    records = [
        EvidenceRecord(
            "past",
            "Past evidence supports a reviewable hold.",
            stance="supports",
            source="gateway",
            timestamp=10,
            relevance=1.0,
            reliability=0.9,
            tags=("payment",),
        ),
        EvidenceRecord(
            "future",
            "Future outcome would justify approval.",
            stance="supports",
            source="ledger",
            timestamp=30,
            relevance=1.0,
            reliability=1.0,
            tags=("payment",),
        ),
    ]
    evidence = EpistemicContextCompiler().compile(
        records,
        policy=EpistemicContextPolicy(cutoff=20),
    )
    actions = [
        ActionCandidate(
            "approve",
            "Approve based on the future outcome.",
            evidence_support=1.0,
            reversibility=0.5,
            downside=0.8,
            required_evidence_ids=("future",),
            support_evidence_ids=("future",),
            support_evidence_tags=("payment",),
        ),
        ActionCandidate(
            "review",
            "Route the case using currently admissible evidence.",
            evidence_support=0.7,
            reversibility=1.0,
            downside=0.1,
            support_evidence_ids=("past",),
            support_evidence_tags=("payment",),
        ),
    ]
    # Use a lighter policy here so the deliberately single-source fallback
    # can remain admissible; the temporal boundary is the feature under test.
    policy = ActionPolicy(
        require_support_anchors=True,
        require_support_stance=True,
        require_support_tag_match=True,
        require_support_quality=True,
    )
    decision = UncertaintyActionGate(policy).decide(
        actions,
        evidence=evidence,
    )
    checks = {
        "future_evidence_excluded": evidence.future_excluded_ids == ("future",),
        "future_action_blocked": "approve" in decision.blocked_action_ids,
        "admissible_fallback_selected": decision.selected_action_id == "review",
    }
    return _result(
        "future-evidence-boundary",
        "epistemic-control",
        checks,
        {
            "included_ids": list(evidence.included_ids),
            "future_excluded_ids": list(evidence.future_excluded_ids),
            "selected_action_id": decision.selected_action_id,
        },
    )


def _case_provenance_diversity() -> BenchmarkResult:
    evidence = EpistemicContextCompiler().compile(
        [
            EvidenceRecord(
                "a1",
                "Source A, observation one.",
                stance="supports",
                source="source_a",
                timestamp=10,
                relevance=1.0,
                reliability=0.9,
                tags=("access",),
            ),
            EvidenceRecord(
                "a2",
                "Source A, observation two.",
                stance="supports",
                source="source_a",
                timestamp=11,
                relevance=1.0,
                reliability=0.9,
                tags=("access",),
            ),
            EvidenceRecord(
                "b1",
                "Source B corroboration.",
                stance="supports",
                source="source_b",
                timestamp=12,
                relevance=0.8,
                reliability=0.8,
                tags=("access",),
            ),
        ]
    )
    actions = [
        ActionCandidate(
            "concentrated",
            "Use two anchors from one source.",
            evidence_support=0.99,
            reversibility=1.0,
            downside=0.1,
            support_evidence_ids=("a1", "a2"),
            support_evidence_tags=("access",),
        ),
        ActionCandidate(
            "diverse",
            "Use anchors from two declared sources.",
            evidence_support=0.80,
            reversibility=1.0,
            downside=0.2,
            support_evidence_ids=("a1", "b1"),
            support_evidence_tags=("access",),
        ),
    ]
    decision = UncertaintyActionGate(_action_policy()).decide(
        actions,
        evidence=evidence,
    )
    checks = {
        "concentrated_blocked": "concentrated" in decision.blocked_action_ids,
        "diverse_selected": decision.selected_action_id == "diverse",
        "provenance_audited": decision.support_evidence_provenance["diverse"]
        == {"a1": "source_a", "b1": "source_b"},
    }
    return _result(
        "provenance-diversity",
        "evidence-admissibility",
        checks,
        {
            "selected_action_id": decision.selected_action_id,
            "blocked_action_ids": list(decision.blocked_action_ids),
            "provenance": decision.support_evidence_provenance,
        },
    )


def _case_context_budget() -> BenchmarkResult:
    blocks = (
        ContextBlock(
            "case",
            {"ticket_id": "T-1001", "sla": "P0"},
            required=True,
            path="case",
            token_estimate=3,
        ),
        ContextBlock(
            "recent_history",
            {"customer": "C-1", "recent": ["r1", "r2"]},
            utility=2.0,
            path="history.recent",
            token_estimate=4,
        ),
        ContextBlock(
            "old_history",
            {"orders": ["x"] * 100},
            utility=0.2,
            path="history.old",
            token_estimate=10,
        ),
    )
    plan = plan_context(
        blocks,
        budget_tokens=8,
        reserve_tokens=1,
    )
    materialized = plan.materialize(blocks)
    checks = {
        "required_context_kept": "case" in plan.included_ids,
        "budget_respected": plan.selected_tokens <= plan.usable_tokens,
        "low_utility_context_omitted": "old_history" in plan.excluded_ids,
        "omitted_content_absent": "orders" not in str(materialized),
    }
    return _result(
        "context-budget-adversarial",
        "context-efficiency",
        checks,
        {
            "budget_tokens": plan.budget_tokens,
            "usable_tokens": plan.usable_tokens,
            "selected_tokens": plan.selected_tokens,
            "included_ids": list(plan.included_ids),
            "excluded_ids": list(plan.excluded_ids),
        },
    )


def _case_deferred_history() -> BenchmarkResult:
    blocks = (
        ContextBlock(
            "task",
            {"ticket_id": "T-2001"},
            required=True,
            path="task",
            token_estimate=2,
        ),
        ContextBlock(
            "history",
            {"private": "very large customer history"},
            utility=0.5,
            path="history",
            token_estimate=8,
        ),
    )
    plan = plan_context(blocks, budget_tokens=3)
    deferred = [item["block_id"] for item in (
        __import__("promptforge").DeferredContextCatalog.from_budget_plan(
            plan,
            blocks,
        ).catalog()["items"]
    )]
    checks = {
        "history_is_omitted": "history" in plan.excluded_ids,
        "history_is_deferred": deferred == ["history"],
        "history_value_not_materialized": "very large" not in str(plan.materialize(blocks)),
    }
    return _result(
        "deferred-history",
        "progressive-disclosure",
        checks,
        {
            "included_ids": list(plan.included_ids),
            "excluded_ids": list(plan.excluded_ids),
            "deferred_ids": deferred,
        },
    )


def _case_tool_output() -> BenchmarkResult:
    content = "HEAD-CRITICAL " + ("x" * 1200) + " TAIL-CRITICAL"
    item = ToolOutputItem(
        "old-log",
        "search",
        content,
        turn_index=5,
    )
    result = ToolOutputTrimmer(
        ToolOutputTrimPolicy(
            recent_turns=2,
            max_output_chars=200,
            preview_chars=80,
        )
    ).trim([item])
    trimmed = result.items[0]
    checks = {
        "old_output_trimmed": result.trimmed_ids == ("old-log",),
        "head_preserved": "HEAD-CRITICAL" in trimmed.content,
        "tail_preserved": "TAIL-CRITICAL" in trimmed.content,
        "digest_recorded": len(trimmed.content_sha256) == 64,
        "output_reduced": trimmed.retained_chars < trimmed.original_chars,
    }
    return _result(
        "tool-output-head-tail",
        "tool-control",
        checks,
        {
            "original_chars": trimmed.original_chars,
            "retained_chars": trimmed.retained_chars,
            "chars_saved": result.chars_saved,
            "estimated_tokens_saved": result.estimated_tokens_saved,
        },
    )


def _case_execution_idempotency() -> BenchmarkResult:
    guard = ActionExecutionGuard()
    first = guard.assess("refund", "ticket:T-3001:refund")
    success = ActionExecutionRecord(
        schema_version="action-execution.v1",
        execution_id="EXEC-1",
        action_id="refund",
        idempotency_key="ticket:T-3001:refund",
        status="succeeded",
        attempt=1,
        recorded_at="2026-10-06T10:00:00Z",
        external_reference="provider-1",
    )
    second = guard.assess(
        "refund",
        "ticket:T-3001:refund",
        prior_records=(success,),
    )
    checks = {
        "first_attempt_execute": first.decision == "execute",
        "second_attempt_duplicate": second.decision == "duplicate",
        "duplicate_has_no_next_attempt": second.next_attempt is None,
    }
    return _result(
        "execution-idempotency",
        "execution-safety",
        checks,
        {
            "first_decision": first.decision,
            "second_decision": second.decision,
            "second_reasons": list(second.reasons),
        },
    )


def _case_human_review_fresh_decision() -> BenchmarkResult:
    action = ActionCandidate(
        "reviewable",
        "Safe action after human review.",
        evidence_support=0.8,
        reversibility=1.0,
        downside=0.1,
    )
    evidence = EpistemicContextCompiler().compile(
        [
            EvidenceRecord(
                "e1",
                "review evidence",
                stance="supports",
                source="review-system",
                timestamp=10,
                tags=("support",),
            )
        ]
    )
    decision_a = UncertaintyActionGate().decide([action], evidence=evidence)
    state = TriageState.admitted(
        case_id="T-4001",
        decision_id="D-1",
        policy_version="policy-v1",
        evidence_snapshot_id="E-SNAP-1",
    )
    priority = PriorityAssessment(
        urgency=1.0,
        importance=0.8,
        priority_band="P0",
        rationale=("Critical SLA.",),
    )
    prioritized = state.transition(
        "prioritized",
        priority=priority,
        decision_id="D-2",
    )
    gated = prioritized.transition(
        "action_gated",
        action_decision=decision_a,
        decision_id="D-3",
    )
    waiting = gated.transition(
        "waiting_human",
        human_review_reason="manual adjudication required",
        decision_id="D-4",
    )
    review = HumanReviewRecord(
        schema_version="human-review.v1",
        review_id="HR-4001",
        reviewer_id="reviewer-1",
        reviewed_at="2026-10-06T10:05:00Z",
        evidence_snapshot_id="E-SNAP-1",
        decision="conditional",
        rationale=("Contradiction requires a fresh decision.",),
        changes_made=("Tightened action policy.",),
        resulting_policy_version="policy-v2",
    )
    fresh_action = ActionCandidate(
        "reviewable",
        "Safe action after human review.",
        evidence_support=0.7,
        reversibility=1.0,
        downside=0.1,
    )
    decision_b = UncertaintyActionGate().decide(
        [fresh_action],
        evidence=evidence,
    )
    resumed = waiting.transition(
        "action_gated",
        human_review=review,
        action_decision=decision_b,
        decision_id="D-5",
    )
    stale_rejected = False
    try:
        waiting.transition(
            "action_gated",
            human_review=review,
            decision_id="D-6",
        )
    except ValueError:
        stale_rejected = True

    checks = {
        "priority_survives_review_gate": resumed.priority.priority_band == "P0",
        "fresh_decision_is_required": resumed.action_decision is decision_b,
        "stale_resume_is_rejected": stale_rejected,
        "policy_version_updated": resumed.policy_version == "policy-v2",
    }
    return _result(
        "human-review-fresh-decision",
        "human-governance",
        checks,
        {
            "stage": resumed.stage,
            "policy_version": resumed.policy_version,
            "review_id": resumed.human_review.review_id,
            "action_score": resumed.action_decision.scores["reviewable"],
        },
    )


CASES: tuple[Callable[[], BenchmarkResult], ...] = (
    _case_priority_vs_actionability,
    _case_future_evidence,
    _case_provenance_diversity,
    _case_context_budget,
    _case_deferred_history,
    _case_tool_output,
    _case_execution_idempotency,
    _case_human_review_fresh_decision,
)


def run_benchmark() -> BenchmarkReport:
    return BenchmarkReport(
        schema_version="promptforge-v27.1-adversarial.v1",
        results=tuple(case() for case in CASES),
    )


def main() -> int:
    report = run_benchmark()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.failed_cases == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
