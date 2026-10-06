import unittest

from promptforge import (
    ActionCandidate,
    ActionPolicy,
    EvidenceRecord,
    EpistemicContextCompiler,
    EpistemicContextPolicy,
    UncertaintyActionGate,
)


class UncertaintyActionGateTests(unittest.TestCase):
    def test_prefers_reversible_low_downside_action_when_support_is_close(self):
        actions = [
            ActionCandidate(
                "rollback",
                "rollback version",
                evidence_support=0.92,
                reversibility=0.25,
                downside=0.80,
                operational_cost=0.60,
            ),
            ActionCandidate(
                "reduce_ingress",
                "reduce ingress",
                evidence_support=0.84,
                reversibility=0.95,
                downside=0.15,
                operational_cost=0.20,
            ),
        ]

        decision = UncertaintyActionGate().decide(actions)

        self.assertEqual(decision.selected_action_id, "reduce_ingress")
        self.assertEqual(decision.ranked_action_ids[0], "reduce_ingress")

    def test_action_requiring_future_evidence_is_blocked(self):
        evidence = EpistemicContextCompiler().compile(
            [
                EvidenceRecord("past", "observed", timestamp=10),
                EvidenceRecord("future", "late observation", timestamp=30),
            ],
            policy=EpistemicContextPolicy(cutoff=20),
        )

        actions = [
            ActionCandidate(
                "future_based",
                "requires future fact",
                evidence_support=0.95,
                reversibility=0.9,
                downside=0.1,
                required_evidence_ids=("future",),
            ),
            ActionCandidate(
                "past_based",
                "uses past fact",
                evidence_support=0.8,
                reversibility=0.9,
                downside=0.1,
                required_evidence_ids=("past",),
            ),
        ]

        decision = UncertaintyActionGate().decide(actions, evidence=evidence)

        self.assertEqual(decision.selected_action_id, "past_based")
        self.assertIn("future_based", decision.blocked_action_ids)

    def test_strict_mode_blocks_high_numeric_support_without_anchors(self):
        evidence = EpistemicContextCompiler().compile(
            [EvidenceRecord("past", "observed", timestamp=10)]
        )
        action = ActionCandidate(
            "unanchored",
            "highly rated but unanchored",
            evidence_support=1.0,
            reversibility=1.0,
            downside=0.1,
        )

        fallback = ActionCandidate(
            "fallback",
            "safe fallback",
            evidence_support=0.70,
            reversibility=1.0,
            downside=0.1,
            support_evidence_ids=("past",),
        )
        decision = UncertaintyActionGate(
            ActionPolicy(require_support_anchors=True)
        ).decide([action, fallback], evidence=evidence)

        self.assertEqual(
            decision.blocked_action_ids,
            ("unanchored",),
        )
        self.assertIn("evidence support is unanchored", decision.reasons["unanchored"])

    def test_strict_mode_blocks_future_support_anchor(self):
        evidence = EpistemicContextCompiler().compile(
            [
                EvidenceRecord("past", "past", timestamp=10),
                EvidenceRecord("future", "future", timestamp=30),
            ],
            policy=EpistemicContextPolicy(cutoff=20),
        )
        action = ActionCandidate(
            "future_anchor",
            "depends on future support",
            evidence_support=0.99,
            reversibility=1.0,
            downside=0.1,
            support_evidence_ids=("future",),
        )

        fallback = ActionCandidate(
            "fallback",
            "safe fallback",
            evidence_support=0.70,
            reversibility=1.0,
            downside=0.1,
            support_evidence_ids=("past",),
        )
        decision = UncertaintyActionGate(
            ActionPolicy(require_support_anchors=True)
        ).decide([action, fallback], evidence=evidence)

        self.assertIn("future_anchor", decision.blocked_action_ids)
        self.assertIn("support evidence unavailable", decision.reasons["future_anchor"])

    def test_anchored_support_passes_and_is_audited(self):
        evidence = EpistemicContextCompiler().compile(
            [
                EvidenceRecord("e1", "support one", timestamp=10),
                EvidenceRecord("e2", "support two", timestamp=11),
            ]
        )
        action = ActionCandidate(
            "anchored",
            "supported action",
            evidence_support=0.85,
            reversibility=0.9,
            downside=0.2,
            support_evidence_ids=("e1", "e2"),
        )

        decision = UncertaintyActionGate(
            ActionPolicy(
                require_support_anchors=True,
                min_support_anchors=2,
            )
        ).decide([action], evidence=evidence)

        self.assertEqual(decision.selected_action_id, "anchored")
        self.assertEqual(
            decision.support_evidence_ids["anchored"],
            ("e1", "e2"),
        )
        self.assertEqual(
            decision.to_dict()["support_evidence_ids"]["anchored"],
            ["e1", "e2"],
        )

    def test_strict_mode_enforces_minimum_anchor_count(self):
        evidence = EpistemicContextCompiler().compile(
            [
                EvidenceRecord("e1", "one support", timestamp=10),
                EvidenceRecord("e2", "second support", timestamp=11),
            ]
        )
        action = ActionCandidate(
            "thin",
            "only one anchor",
            evidence_support=0.95,
            reversibility=1.0,
            downside=0.1,
            support_evidence_ids=("e1",),
        )

        fallback = ActionCandidate(
            "fallback",
            "safe fallback",
            evidence_support=0.70,
            reversibility=1.0,
            downside=0.1,
            support_evidence_ids=("e1", "e2"),
        )
        decision = UncertaintyActionGate(
            ActionPolicy(
                require_support_anchors=True,
                min_support_anchors=2,
            )
        ).decide([action, fallback], evidence=evidence)

        self.assertIn("thin", decision.blocked_action_ids)
        self.assertIn(
            "insufficient support evidence anchors",
            decision.reasons["thin"],
        )

    def test_support_stance_must_be_compatible_with_action(self):
        evidence = EpistemicContextCompiler().compile(
            [
                EvidenceRecord(
                    "support",
                    "supports the action",
                    timestamp=10,
                    stance="supports",
                ),
                EvidenceRecord(
                    "contradiction",
                    "contradicts the action",
                    timestamp=11,
                    stance="contradicts",
                ),
                EvidenceRecord(
                    "fallback_support",
                    "supports the fallback",
                    timestamp=12,
                    stance="supports",
                ),
            ]
        )
        actions = [
            ActionCandidate(
                "supported",
                "compatible evidence",
                evidence_support=0.90,
                reversibility=0.9,
                downside=0.1,
                support_evidence_ids=("support",),
            ),
            ActionCandidate(
                "contradictory",
                "incompatible evidence",
                evidence_support=0.99,
                reversibility=0.9,
                downside=0.1,
                support_evidence_ids=("contradiction",),
            ),
            ActionCandidate(
                "fallback",
                "safe fallback",
                evidence_support=0.70,
                reversibility=1.0,
                downside=0.2,
                support_evidence_ids=("fallback_support",),
            ),
        ]

        decision = UncertaintyActionGate(
            ActionPolicy(require_support_stance=True)
        ).decide(actions, evidence=evidence)

        self.assertEqual(decision.selected_action_id, "supported")
        self.assertIn("contradictory", decision.blocked_action_ids)
        self.assertIn(
            "support evidence stance incompatible",
            decision.reasons["contradictory"],
        )
        self.assertEqual(
            decision.support_evidence_stances["supported"],
            {"support": "supports"},
        )

    def test_neutral_support_is_blocked_by_default(self):
        evidence = EpistemicContextCompiler().compile(
            [
                EvidenceRecord(
                    "neutral",
                    "does not establish support",
                    timestamp=10,
                    stance="neutral",
                ),
                EvidenceRecord(
                    "safe_support",
                    "explicit support",
                    timestamp=11,
                    stance="supports",
                ),
            ]
        )
        actions = [
            ActionCandidate(
                "neutral_anchor",
                "neutral anchor",
                evidence_support=1.0,
                reversibility=1.0,
                downside=0.1,
                support_evidence_ids=("neutral",),
            ),
            ActionCandidate(
                "supported",
                "supported action",
                evidence_support=0.8,
                reversibility=1.0,
                downside=0.1,
                support_evidence_ids=("safe_support",),
            ),
        ]

        decision = UncertaintyActionGate(
            ActionPolicy(require_support_stance=True)
        ).decide(actions, evidence=evidence)

        self.assertEqual(decision.selected_action_id, "supported")
        self.assertIn("neutral_anchor", decision.blocked_action_ids)

    def test_integrator_can_explicitly_allow_neutral_support(self):
        evidence = EpistemicContextCompiler().compile(
            [
                EvidenceRecord(
                    "neutral",
                    "neutral but admissible under explicit policy",
                    timestamp=10,
                    stance="neutral",
                ),
            ]
        )
        action = ActionCandidate(
            "neutral_allowed",
            "explicit neutral policy",
            evidence_support=0.8,
            reversibility=1.0,
            downside=0.1,
            support_evidence_ids=("neutral",),
        )

        decision = UncertaintyActionGate(
            ActionPolicy(
                require_support_stance=True,
                allowed_support_stances=("supports", "neutral"),
            )
        ).decide([action], evidence=evidence)

        self.assertEqual(decision.selected_action_id, "neutral_allowed")
        self.assertEqual(
            decision.support_evidence_stances["neutral_allowed"],
            {"neutral": "neutral"},
        )

    def test_support_stance_requires_evidence_boundary(self):
        action = ActionCandidate(
            "stance_required",
            "needs an explicit stance",
            evidence_support=0.9,
            reversibility=1.0,
            downside=0.1,
            support_evidence_ids=("missing_snapshot",),
        )
        evidence = EpistemicContextCompiler().compile(
            [
                EvidenceRecord(
                    "fallback_support",
                    "explicit fallback support",
                    timestamp=10,
                    stance="supports",
                )
            ]
        )
        fallback = ActionCandidate(
            "fallback",
            "safe fallback",
            evidence_support=0.6,
            reversibility=1.0,
            downside=0.2,
            support_evidence_ids=("fallback_support",),
        )

        decision = UncertaintyActionGate(
            ActionPolicy(require_support_stance=True)
        ).decide([action, fallback], evidence=evidence)

        self.assertEqual(decision.selected_action_id, "fallback")
        self.assertIn("stance_required", decision.blocked_action_ids)
        self.assertIn(
            "support stance unavailable",
            decision.reasons["stance_required"],
        )

    def test_invalid_support_stance_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            ActionPolicy(
                require_support_stance=True,
                allowed_support_stances=("invalid",),
            )

    def test_support_relevance_blocks_matching_stance_with_wrong_tags(self):
        evidence = EpistemicContextCompiler().compile(
            [
                EvidenceRecord(
                    "deploy",
                    "deployment evidence",
                    timestamp=10,
                    stance="supports",
                    tags=("deployment", "runtime"),
                ),
                EvidenceRecord(
                    "billing",
                    "billing evidence",
                    timestamp=11,
                    stance="supports",
                    tags=("billing", "finance"),
                ),
            ]
        )
        actions = [
            ActionCandidate(
                "runtime_action",
                "runtime action",
                evidence_support=0.99,
                reversibility=1.0,
                downside=0.1,
                support_evidence_ids=("billing",),
                support_evidence_tags=("runtime",),
            ),
            ActionCandidate(
                "safe_runtime",
                "safe runtime fallback",
                evidence_support=0.80,
                reversibility=1.0,
                downside=0.2,
                support_evidence_ids=("deploy",),
                support_evidence_tags=("runtime",),
            ),
        ]

        decision = UncertaintyActionGate(
            ActionPolicy(require_support_tag_match=True)
        ).decide(actions, evidence=evidence)

        self.assertEqual(decision.selected_action_id, "safe_runtime")
        self.assertIn("runtime_action", decision.blocked_action_ids)
        self.assertIn(
            "support evidence relevance mismatch",
            decision.reasons["runtime_action"],
        )

    def test_support_relevance_passes_and_is_audited(self):
        evidence = EpistemicContextCompiler().compile(
            [
                EvidenceRecord(
                    "e1",
                    "supports runtime action",
                    timestamp=10,
                    stance="supports",
                    tags=("runtime", "deploy"),
                ),
                EvidenceRecord(
                    "e2",
                    "supports runtime action via monitoring",
                    timestamp=11,
                    stance="supports",
                    tags=("monitoring", "runtime"),
                ),
            ]
        )
        action = ActionCandidate(
            "runtime",
            "runtime action",
            evidence_support=0.90,
            reversibility=1.0,
            downside=0.1,
            support_evidence_ids=("e1", "e2"),
            support_evidence_tags=("runtime",),
        )

        decision = UncertaintyActionGate(
            ActionPolicy(require_support_tag_match=True)
        ).decide([action], evidence=evidence)

        self.assertEqual(decision.selected_action_id, "runtime")
        self.assertEqual(
            decision.support_evidence_tag_matches["runtime"],
            {"e1": ("runtime",), "e2": ("runtime",)},
        )
        self.assertEqual(
            decision.to_dict()["support_evidence_tag_matches"]["runtime"],
            {"e1": ["runtime"], "e2": ["runtime"]},
        )

    def test_support_relevance_requires_explicit_scope(self):
        evidence = EpistemicContextCompiler().compile(
            [
                EvidenceRecord(
                    "e1",
                    "support without declared scope",
                    timestamp=10,
                    stance="supports",
                    tags=("runtime",),
                ),
            ]
        )
        action = ActionCandidate(
            "scopeless",
            "scope missing",
            evidence_support=0.9,
            reversibility=1.0,
            downside=0.1,
            support_evidence_ids=("e1",),
        )
        fallback = ActionCandidate(
            "fallback",
            "safe fallback",
            evidence_support=0.7,
            reversibility=1.0,
            downside=0.2,
            support_evidence_ids=("e1",),
            support_evidence_tags=("runtime",),
        )

        decision = UncertaintyActionGate(
            ActionPolicy(require_support_tag_match=True)
        ).decide([action, fallback], evidence=evidence)

        self.assertEqual(decision.selected_action_id, "fallback")
        self.assertIn("support relevance scope was not declared", decision.reasons["scopeless"])

    def test_invalid_support_evidence_tags_are_rejected(self):
        with self.assertRaises(ValueError):
            ActionCandidate(
                "bad_tags",
                "invalid tags",
                evidence_support=0.8,
                reversibility=1.0,
                downside=0.1,
                support_evidence_tags=("runtime", "runtime"),
            )

        with self.assertRaises(ValueError):
            ActionCandidate(
                "empty_tag",
                "empty tag",
                evidence_support=0.8,
                reversibility=1.0,
                downside=0.1,
                support_evidence_tags=("   ",),
            )

    def test_support_quality_blocks_low_relevance_even_when_stance_and_tags_match(self):
        evidence = EpistemicContextCompiler().compile(
            [
                EvidenceRecord(
                    "weak",
                    "explicitly supporting but low relevance",
                    timestamp=10,
                    stance="supports",
                    relevance=0.30,
                    reliability=0.95,
                    tags=("runtime",),
                ),
                EvidenceRecord(
                    "strong",
                    "support with adequate metadata",
                    timestamp=11,
                    stance="supports",
                    relevance=0.90,
                    reliability=0.90,
                    tags=("runtime",),
                ),
            ]
        )
        actions = [
            ActionCandidate(
                "weak_action",
                "high score but weak evidence quality",
                evidence_support=0.99,
                reversibility=1.0,
                downside=0.1,
                support_evidence_ids=("weak",),
                support_evidence_tags=("runtime",),
            ),
            ActionCandidate(
                "strong_action",
                "supported action",
                evidence_support=0.80,
                reversibility=1.0,
                downside=0.2,
                support_evidence_ids=("strong",),
                support_evidence_tags=("runtime",),
            ),
        ]

        decision = UncertaintyActionGate(
            ActionPolicy(
                require_support_quality=True,
                min_support_relevance=0.70,
                min_support_reliability=0.70,
            )
        ).decide(actions, evidence=evidence)

        self.assertEqual(decision.selected_action_id, "strong_action")
        self.assertIn("weak_action", decision.blocked_action_ids)
        self.assertIn(
            "support evidence quality below threshold",
            decision.reasons["weak_action"],
        )
        self.assertIn(
            "low_support_relevance:weak=0.3",
            decision.reasons["weak_action"],
        )

    def test_support_quality_blocks_low_reliability(self):
        evidence = EpistemicContextCompiler().compile(
            [
                EvidenceRecord(
                    "uncertain_source",
                    "support with low declared reliability",
                    timestamp=10,
                    stance="supports",
                    relevance=0.95,
                    reliability=0.20,
                    tags=("runtime",),
                ),
                EvidenceRecord(
                    "trusted_source",
                    "support with sufficient declared reliability",
                    timestamp=11,
                    stance="supports",
                    relevance=0.95,
                    reliability=0.95,
                    tags=("runtime",),
                ),
            ]
        )
        actions = [
            ActionCandidate(
                "uncertain_action",
                "low reliability support",
                evidence_support=0.99,
                reversibility=1.0,
                downside=0.1,
                support_evidence_ids=("uncertain_source",),
                support_evidence_tags=("runtime",),
            ),
            ActionCandidate(
                "trusted_action",
                "trusted support",
                evidence_support=0.80,
                reversibility=1.0,
                downside=0.2,
                support_evidence_ids=("trusted_source",),
                support_evidence_tags=("runtime",),
            ),
        ]

        decision = UncertaintyActionGate(
            ActionPolicy(
                require_support_quality=True,
                min_support_relevance=0.70,
                min_support_reliability=0.70,
            )
        ).decide(actions, evidence=evidence)

        self.assertEqual(decision.selected_action_id, "trusted_action")
        self.assertIn("uncertain_action", decision.blocked_action_ids)
        self.assertIn(
            "low_support_reliability:uncertain_source=0.2",
            decision.reasons["uncertain_action"],
        )

    def test_support_quality_passes_and_is_audited(self):
        evidence = EpistemicContextCompiler().compile(
            [
                EvidenceRecord(
                    "e1",
                    "good support",
                    timestamp=10,
                    stance="supports",
                    relevance=0.85,
                    reliability=0.90,
                    tags=("runtime",),
                ),
                EvidenceRecord(
                    "e2",
                    "second good support",
                    timestamp=11,
                    stance="supports",
                    relevance=0.75,
                    reliability=0.80,
                    tags=("runtime",),
                ),
            ]
        )
        action = ActionCandidate(
            "quality_checked",
            "quality checked support",
            evidence_support=0.90,
            reversibility=1.0,
            downside=0.1,
            support_evidence_ids=("e1", "e2"),
            support_evidence_tags=("runtime",),
        )

        decision = UncertaintyActionGate(
            ActionPolicy(
                require_support_quality=True,
                min_support_relevance=0.70,
                min_support_reliability=0.70,
            )
        ).decide([action], evidence=evidence)

        self.assertEqual(decision.selected_action_id, "quality_checked")
        self.assertEqual(
            decision.support_evidence_quality["quality_checked"],
            {
                "e1": {"relevance": 0.85, "reliability": 0.90},
                "e2": {"relevance": 0.75, "reliability": 0.80},
            },
        )
        self.assertEqual(
            decision.to_dict()["support_evidence_quality"]["quality_checked"],
            {
                "e1": {"relevance": 0.85, "reliability": 0.90},
                "e2": {"relevance": 0.75, "reliability": 0.80},
            },
        )

    def test_support_quality_requires_evidence_boundary(self):
        action = ActionCandidate(
            "quality_required",
            "needs evidence quality boundary",
            evidence_support=0.9,
            reversibility=1.0,
            downside=0.1,
            support_evidence_ids=("e1",),
        )

        with self.assertRaises(ValueError):
            UncertaintyActionGate(
                ActionPolicy(require_support_quality=True)
            ).decide([action])

    def test_invalid_support_quality_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            ActionPolicy(min_support_relevance=-0.1)
        with self.assertRaises(ValueError):
            ActionPolicy(min_support_reliability=1.1)
        with self.assertRaises(TypeError):
            ActionPolicy(require_support_quality="yes")

    def test_causal_dependence_is_a_penalty_not_a_truth_claim(self):
        actions = [
            ActionCandidate(
                "causal",
                "depends on causal interpretation",
                evidence_support=0.90,
                reversibility=0.90,
                downside=0.20,
                depends_on_causal_claim=True,
            ),
            ActionCandidate(
                "observational",
                "works without causal resolution",
                evidence_support=0.82,
                reversibility=0.90,
                downside=0.20,
                depends_on_causal_claim=False,
            ),
        ]

        decision = UncertaintyActionGate().decide(actions)

        self.assertEqual(decision.selected_action_id, "observational")
        self.assertIn("causal dependence receives a caution penalty", decision.reasons["causal"])

    def test_strict_policy_can_fail_closed_on_irreversible_or_risky_action(self):
        actions = [
            ActionCandidate(
                "dangerous",
                "irreversible high downside",
                evidence_support=0.95,
                reversibility=0.0,
                downside=0.95,
            ),
            ActionCandidate(
                "safe",
                "reversible low downside",
                evidence_support=0.70,
                reversibility=1.0,
                downside=0.10,
            ),
        ]

        decision = UncertaintyActionGate(
            ActionPolicy(require_reversible=True, max_downside=0.80)
        ).decide(actions)

        self.assertEqual(decision.selected_action_id, "safe")
        self.assertEqual(decision.blocked_action_ids, ("dangerous",))

    def test_decision_is_deterministic(self):
        action = ActionCandidate(
            "a",
            "same",
            evidence_support=0.8,
            reversibility=0.7,
            downside=0.2,
            operational_cost=0.2,
        )
        first = UncertaintyActionGate().decide([action])
        second = UncertaintyActionGate().decide([action])

        self.assertEqual(first.to_dict(), second.to_dict())


if __name__ == "__main__":
    unittest.main()
