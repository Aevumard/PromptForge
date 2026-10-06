import unittest

from promptforge import ContextCognitiveLoop


class CognitiveLoopTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "task": {"id": "T-1", "action": "review", "noise": "x"},
            "user": {"id": "U-1"},
        }
        self.candidates = [
            {
                "arm_id": "selection_only",
                "context_chars": 80,
                "estimated_tokens": 20,
                "required_values_preserved": True,
            },
            {
                "arm_id": "selection_representation_B",
                "context_chars": 120,
                "estimated_tokens": 30,
                "required_values_preserved": True,
            },
        ]

    def test_first_cycle_uses_structural_policy_when_memory_is_empty(self):
        loop = ContextCognitiveLoop(
            routing_features=("node_count", "max_depth", "boundary_pressure"),
        )
        proposal = loop.propose(
            cycle_id="cycle-1",
            data=self.data,
            required=["task.id", "task.action"],
            candidates=self.candidates,
        )

        self.assertEqual(proposal.schema_version, "context-cognitive-proposal.v1")
        self.assertEqual(proposal.experience_version, 0)
        self.assertIsNone(proposal.memory_route)
        self.assertEqual(proposal.decision.source, "structural_regime")
        self.assertEqual(proposal.decision.strategy, "selection_only")

    def test_observed_outcome_becomes_future_routing_evidence(self):
        loop = ContextCognitiveLoop(
            routing_features=("node_count", "max_depth", "boundary_pressure"),
        )
        first = loop.propose(
            cycle_id="cycle-1",
            data=self.data,
            required=["task.id", "task.action"],
            candidates=self.candidates,
        )
        version = loop.observe(first, family_id="family-A", cost=1.0, strategy="selection_representation_B")

        second = loop.propose(
            cycle_id="cycle-2",
            data=self.data,
            required=["task.id", "task.action"],
            candidates=self.candidates,
        )

        self.assertEqual(version, 1)
        self.assertEqual(second.experience_version, 1)
        self.assertIsNotNone(second.memory_route)
        self.assertEqual(second.memory_route.strategy, "selection_representation_B")
        self.assertEqual(second.decision.source, "episodic_memory")
        self.assertEqual(second.decision.strategy, "selection_representation_B")

    def test_proposal_keeps_original_evidence_version_after_learning(self):
        loop = ContextCognitiveLoop(
            routing_features=("node_count", "max_depth"),
        )
        proposal = loop.propose(
            cycle_id="cycle-1",
            data=self.data,
            required=["task.id"],
            candidates=self.candidates,
        )
        loop.observe(proposal, family_id="family-A", cost=2.0)

        self.assertEqual(proposal.experience_version, 0)
        self.assertEqual(loop.snapshot().version, 1)
        self.assertEqual(len(proposal.to_dict()["profile"]["required_missing"]), 0)


if __name__ == "__main__":
    unittest.main()
