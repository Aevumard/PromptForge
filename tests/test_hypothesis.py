import unittest

from promptforge import (
    DiscriminatingExperiment,
    EvidenceRecord,
    EpistemicContextCompiler,
    EpistemicContextPolicy,
    HypothesisEvidencePolicy,
    HypothesisLedger,
    HypothesisRecord,
)


class HypothesisLedgerTests(unittest.TestCase):
    def _evidence(self):
        return EpistemicContextCompiler().compile(
            [
                EvidenceRecord("s", "support", timestamp=10, stance="supports"),
                EvidenceRecord("c", "contradiction", timestamp=11, stance="contradicts"),
                EvidenceRecord("late", "future", timestamp=30),
            ],
            policy=EpistemicContextPolicy(cutoff=20),
        )

    def test_support_and_contradiction_are_explicit(self):
        result = HypothesisLedger().assess(
            [
                HypothesisRecord(
                    "h1",
                    "mixed mechanism",
                    support_evidence_ids=("s",),
                    contradiction_evidence_ids=("c",),
                )
            ],
            evidence=self._evidence(),
        )

        self.assertEqual(result[0].status, "contested")
        self.assertEqual(result[0].support_count, 1)
        self.assertEqual(result[0].contradiction_count, 1)
        self.assertEqual(result[0].support_ratio, 0.5)

    def test_future_evidence_does_not_support_hypothesis_at_cutoff(self):
        result = HypothesisLedger().assess(
            [
                HypothesisRecord(
                    "h1",
                    "late explanation",
                    support_evidence_ids=("late",),
                )
            ],
            evidence=self._evidence(),
        )

        self.assertEqual(result[0].status, "unresolved")
        self.assertEqual(result[0].support_count, 0)

    def test_required_missing_evidence_fails_to_insufficient_not_false(self):
        result = HypothesisLedger().assess(
            [
                HypothesisRecord(
                    "h1",
                    "needs unavailable witness",
                    required_evidence_ids=("missing",),
                )
            ],
            evidence=self._evidence(),
        )

        self.assertEqual(result[0].status, "insufficient_evidence")
        self.assertEqual(result[0].missing_required_ids, ("missing",))

    def test_source_correlated_support_is_capped_and_audited(self):
        evidence = EpistemicContextCompiler().compile(
            [
                EvidenceRecord("s1", "support 1", timestamp=10, source="sensor-a", stance="supports"),
                EvidenceRecord("s2", "support 2", timestamp=11, source="sensor-a", stance="supports"),
                EvidenceRecord("s3", "support 3", timestamp=12, source="sensor-a", stance="supports"),
                EvidenceRecord("s4", "independent support", timestamp=13, source="sensor-b", stance="supports"),
            ]
        )

        result = HypothesisLedger().assess(
            [
                HypothesisRecord(
                    "h1",
                    "source-capped hypothesis",
                    support_evidence_ids=("s1", "s2", "s3", "s4"),
                )
            ],
            evidence=evidence,
            evidence_policy=HypothesisEvidencePolicy(max_per_source=1),
        )

        self.assertEqual(result[0].support_count, 2)
        self.assertEqual(result[0].available_support_ids, ("s4", "s3"))
        self.assertEqual(
            result[0].source_excluded_ids,
            ("s2", "s1"),
        )

    def test_support_and_contradiction_each_get_their_own_source_budget(self):
        evidence = EpistemicContextCompiler().compile(
            [
                EvidenceRecord("s1", "support", timestamp=10, source="lab-a", stance="supports"),
                EvidenceRecord("s2", "support repeat", timestamp=11, source="lab-a", stance="supports"),
                EvidenceRecord("c1", "contradiction", timestamp=12, source="lab-a", stance="contradicts"),
                EvidenceRecord("c2", "contradiction repeat", timestamp=13, source="lab-a", stance="contradicts"),
            ]
        )

        result = HypothesisLedger().assess(
            [
                HypothesisRecord(
                    "h1",
                    "conflicted evidence",
                    support_evidence_ids=("s1", "s2"),
                    contradiction_evidence_ids=("c1", "c2"),
                )
            ],
            evidence=evidence,
            evidence_policy=HypothesisEvidencePolicy(max_per_source=1),
        )

        self.assertEqual(result[0].status, "contested")
        self.assertEqual(result[0].support_count, 1)
        self.assertEqual(result[0].contradiction_count, 1)
        self.assertEqual(len(result[0].source_excluded_ids), 2)

    def test_unknown_sources_can_remain_isolated(self):
        evidence = EpistemicContextCompiler().compile(
            [
                EvidenceRecord("u1", "unknown 1", timestamp=10, stance="supports"),
                EvidenceRecord("u2", "unknown 2", timestamp=11, stance="supports"),
            ]
        )

        result = HypothesisLedger().assess(
            [
                HypothesisRecord(
                    "h1",
                    "unknown-source evidence",
                    support_evidence_ids=("u1", "u2"),
                )
            ],
            evidence=evidence,
            evidence_policy=HypothesisEvidencePolicy(max_per_source=1),
        )

        self.assertEqual(result[0].support_count, 2)
        self.assertEqual(result[0].source_excluded_ids, ())

    def test_experiment_ranking_is_deterministic_and_not_called_statistical_power(self):
        experiments = [
            DiscriminatingExperiment(
                "e1",
                "low quality",
                ("h1", "h2"),
                isolation=0.4,
                control_quality=0.5,
                measurement_quality=0.5,
                operational_cost=0.5,
                operational_risk=0.2,
            ),
            DiscriminatingExperiment(
                "e2",
                "strong separation",
                ("h1", "h2", "h3"),
                isolation=0.95,
                control_quality=0.95,
                measurement_quality=0.9,
                operational_cost=0.2,
                operational_risk=0.1,
            ),
        ]

        ranked = HypothesisLedger.rank_experiments(experiments)
        self.assertEqual(ranked[0].experiment_id, "e2")
        self.assertIn("priority_score", ranked[0].to_dict())

    def test_declared_evidence_ids_require_a_snapshot(self):
        with self.assertRaises(ValueError):
            HypothesisLedger().assess(
                [
                    HypothesisRecord(
                        "h1",
                        "requires declared evidence",
                        support_evidence_ids=("e1",),
                    )
                ]
            )

    def test_no_evidence_is_unresolved_not_disproved(self):
        result = HypothesisLedger().assess(
            [HypothesisRecord("h1", "open hypothesis")]
        )

        self.assertEqual(result[0].status, "unresolved")


if __name__ == "__main__":
    unittest.main()
