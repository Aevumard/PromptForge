import unittest

from promptforge.decision import ActionDecision
from promptforge.human_review import HumanReviewRecord
from promptforge.triage import (
    PriorityAssessment,
    TriageEnvelope,
    TriageState,
)


def _action_decision(score: float = 0.8) -> ActionDecision:
    return ActionDecision(
        schema_version="uncertainty-action.v6",
        selected_action_id="act-1",
        ranked_action_ids=("act-1",),
        blocked_action_ids=(),
        scores={"act-1": score},
        reasons={"act-1": ("supported",)},
        evidence_ids_available=("e-1",),
        support_evidence_ids={"act-1": ("e-1",)},
        support_evidence_stances={"act-1": {"e-1": "supports"}},
        support_evidence_tag_matches={"act-1": {"e-1": ("payment",)}},
        support_evidence_quality={
            "act-1": {"e-1": {"relevance": 0.9, "reliability": 0.9}}
        },
        support_evidence_provenance={"act-1": {"e-1": "gateway"}},
    )


def _human_review(
    *,
    decision: str = "allow",
    policy_version: str = "policy-v2",
) -> HumanReviewRecord:
    return HumanReviewRecord(
        schema_version="human-review.v1",
        review_id="HR-1",
        reviewer_id="reviewer-7",
        reviewed_at="2026-10-06T09:00:00Z",
        evidence_snapshot_id="E-SNAP-1",
        decision=decision,
        rationale=("contradiction manually adjudicated",),
        changes_made=("require explicit gateway evidence",),
        resulting_policy_version=policy_version,
    )


class TriageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.priority = PriorityAssessment(
            urgency=1.0,
            importance=0.2,
            priority_band="P0",
            rationale=("SLA is near breach",),
        )

    def test_priority_is_separate_from_actionability(self) -> None:
        action = _action_decision()
        envelope = TriageEnvelope(
            schema_version="triage-envelope.v1",
            priority=self.priority,
            action_decision=action,
        )
        payload = envelope.to_dict()
        self.assertEqual(payload["priority"]["urgency"], 1.0)
        self.assertEqual(payload["priority"]["importance"], 0.2)
        self.assertEqual(
            payload["action_decision"]["selected_action_id"],
            "act-1",
        )

    def test_high_priority_can_transition_to_human_block(self) -> None:
        state = TriageState.admitted(
            case_id="T-1",
            decision_id="D-1",
            policy_version="policy-v1",
            evidence_snapshot_id="E-SNAP-1",
        )
        prioritized = state.transition(
            "prioritized",
            priority=self.priority,
            decision_id="D-2",
        )
        gated = prioritized.transition(
            "action_gated",
            action_decision=_action_decision(),
            decision_id="D-3",
        )
        waiting = gated.transition(
            "waiting_human",
            human_review_reason="critical contradiction",
            decision_id="D-4",
        )
        self.assertEqual(waiting.priority.priority_band, "P0")
        self.assertEqual(waiting.stage, "waiting_human")

    def test_human_review_is_durable_and_resume_requires_fresh_action_decision(self) -> None:
        state = TriageState.admitted(
            case_id="T-1",
            decision_id="D-1",
            policy_version="policy-v1",
            evidence_snapshot_id="E-SNAP-1",
        )
        prioritized = state.transition(
            "prioritized",
            priority=self.priority,
            decision_id="D-2",
        )
        gated = prioritized.transition(
            "action_gated",
            action_decision=_action_decision(),
            decision_id="D-3",
        )
        waiting = gated.transition(
            "waiting_human",
            human_review_reason="critical contradiction",
            decision_id="D-4",
        )
        review = _human_review()
        resumed = waiting.transition(
            "action_gated",
            human_review=review,
            action_decision=_action_decision(score=0.6),
            decision_id="D-5",
        )

        self.assertEqual(resumed.human_review.review_id, "HR-1")
        self.assertEqual(resumed.policy_version, "policy-v2")
        self.assertEqual(resumed.action_decision.scores["act-1"], 0.6)

        with self.assertRaises(ValueError):
            waiting.transition(
                "action_gated",
                human_review=review,
                decision_id="D-6",
            )

        with self.assertRaises(ValueError):
            waiting.transition(
                "action_gated",
                human_review=HumanReviewRecord(
                    schema_version="human-review.v1",
                    review_id="HR-2",
                    reviewer_id="reviewer-8",
                    reviewed_at="2026-10-06T09:05:00Z",
                    evidence_snapshot_id="OTHER-SNAPSHOT",
                    decision="allow",
                    rationale=("different snapshot",),
                    changes_made=(),
                    resulting_policy_version="policy-v3",
                ),
                action_decision=_action_decision(),
                decision_id="D-7",
            )

    def test_invalid_transition_is_rejected(self) -> None:
        state = TriageState.admitted(
            case_id="T-1",
            decision_id="D-1",
            policy_version="policy-v1",
        )
        with self.assertRaises(ValueError):
            state.transition("executed")

    def test_historical_state_is_immutable(self) -> None:
        state = TriageState.admitted(
            case_id="T-1",
            decision_id="D-1",
            policy_version="policy-v1",
        )
        next_state = state.transition(
            "prioritized",
            priority=self.priority,
            decision_id="D-2",
        )
        self.assertEqual(state.stage, "admitted")
        self.assertIsNone(state.priority)
        self.assertEqual(next_state.parent_decision_id, "D-1")


if __name__ == "__main__":
    unittest.main()
