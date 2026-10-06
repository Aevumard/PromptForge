import unittest

from promptforge import (
    EVIDENCE_KINDS,
    EVIDENCE_STANCES,
    EpistemicContextCompiler,
    EpistemicContextPolicy,
    EvidenceRecord,
)


class EpistemicContextTests(unittest.TestCase):
    def setUp(self):
        self.compiler = EpistemicContextCompiler()

    def test_temporal_boundary_excludes_future_and_unknown_by_default(self):
        records = [
            EvidenceRecord(
                "past",
                "queue grew",
                timestamp=100,
                relevance=0.9,
            ),
            EvidenceRecord(
                "future",
                "late dashboard",
                timestamp=130,
                relevance=1.0,
            ),
            EvidenceRecord(
                "unknown",
                "timestamp missing",
                timestamp=None,
                relevance=1.0,
            ),
        ]

        result = self.compiler.compile(
            records,
            policy=EpistemicContextPolicy(cutoff=120),
        )

        self.assertEqual(result.included_ids, ("past",))
        self.assertEqual(result.future_excluded_ids, ("future",))
        self.assertEqual(result.unknown_time_excluded_ids, ("unknown",))
        self.assertTrue(result.audit["temporal_boundary_enforced"])

    def test_required_future_evidence_is_rejected(self):
        with self.assertRaises(ValueError):
            self.compiler.compile(
                [
                    EvidenceRecord("late", "future fact", timestamp=130),
                ],
                policy=EpistemicContextPolicy(
                    cutoff=120,
                    required_ids=("late",),
                ),
            )

    def test_contradiction_and_kind_coverage_survive_selection(self):
        records = [
            EvidenceRecord("o1", "observation one", kind="observation", relevance=0.1),
            EvidenceRecord("o2", "observation two", kind="observation", relevance=0.2),
            EvidenceRecord(
                "i1",
                "inference",
                kind="inference",
                relevance=0.1,
            ),
            EvidenceRecord(
                "h1",
                "hypothesis",
                kind="hypothesis",
                relevance=0.1,
            ),
            EvidenceRecord(
                "n1",
                "negative result",
                kind="observation",
                stance="contradicts",
                relevance=0.05,
            ),
        ]

        result = self.compiler.compile(
            records,
            policy=EpistemicContextPolicy(max_records=4),
        )

        self.assertIn("n1", result.included_ids)
        self.assertIn("i1", result.included_ids)
        self.assertIn("h1", result.included_ids)
        self.assertIn("n1", result.contradiction_ids)
        self.assertTrue(result.audit["contradiction_preserved"])

    def test_causal_status_is_descriptive_not_causal_proof(self):
        isolated = EvidenceRecord(
            "x1",
            "single intervention",
            tags=("intervention",),
            intervention_factors=("ingress",),
        )
        multivariable = EvidenceRecord(
            "x2",
            "combined intervention",
            tags=("intervention", "multivariable", "placebo_absent"),
            intervention_factors=("ingress", "route_class"),
        )
        confounded = EvidenceRecord(
            "x3",
            "confounded intervention",
            tags=("intervention",),
            intervention_factors=("route_class",),
            confounders=("image_load",),
        )

        self.assertEqual(isolated.causal_status, "isolated_candidate")
        self.assertEqual(multivariable.causal_status, "multivariable_intervention")
        self.assertEqual(confounded.causal_status, "confounded_intervention")

        result = self.compiler.compile([isolated, multivariable, confounded])
        statuses = {
            item["evidence_id"]: item["causal_status"]
            for item in result.context["evidence"]
        }
        self.assertEqual(statuses["x2"], "multivariable_intervention")
        self.assertEqual(statuses["x3"], "confounded_intervention")

    def test_budget_is_deterministic(self):
        records = [
            EvidenceRecord(
                f"e{i}",
                "x" * 20,
                relevance=1.0 - i * 0.1,
                timestamp=i,
            )
            for i in range(6)
        ]

        policy = EpistemicContextPolicy(
            cutoff=10,
            max_records=3,
            budget_tokens=30,
            preserve_contradictions=False,
            preserve_each_kind=False,
        )
        a = self.compiler.compile(records, policy=policy)
        b = self.compiler.compile(records, policy=policy)

        self.assertEqual(a.included_ids, b.included_ids)
        self.assertEqual(a.serialized_context, b.serialized_context)
        self.assertLessEqual(a.estimated_tokens, 30)

    def test_mapping_inputs_and_constant_contract(self):
        result = self.compiler.compile(
            [
                {
                    "evidence_id": "m1",
                    "content": "mapped",
                    "kind": "observation",
                    "stance": "neutral",
                    "timestamp": 1,
                }
            ]
        )

        self.assertEqual(result.included_ids, ("m1",))
        self.assertIn("observation", EVIDENCE_KINDS)
        self.assertIn("contradicts", EVIDENCE_STANCES)


if __name__ == "__main__":
    unittest.main()
