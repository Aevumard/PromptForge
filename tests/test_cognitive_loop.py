import unittest

from promptforge import (
    ContextCognitiveLoop,
    ContextRoutingModeScore,
    ContextRoutingPolicyEvidence,
)


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

    def test_adaptive_mode_without_policy_falls_back_to_nearest(self):
        loop = ContextCognitiveLoop(memory_routing_mode="adaptive")
        proposal = loop.propose(
            cycle_id="cycle-adaptive-fallback",
            data=self.data,
            required=["task.id"],
            candidates=self.candidates,
        )

        self.assertEqual(proposal.memory_routing_mode, "adaptive")
        self.assertEqual(proposal.memory_routing_selected_mode, "nearest")
        self.assertIsNone(proposal.memory_policy_version)

    def test_adaptive_mode_uses_frozen_policy_evidence(self):
        evidence = ContextRoutingPolicyEvidence(
            version=11,
            scores=(
                ContextRoutingModeScore("nearest", 5, 3, 0.4, 2.0, 0.8),
                ContextRoutingModeScore("credit", 5, 3, 0.8, 0.5, 0.2),
            ),
            selected_mode="credit",
        )
        loop = ContextCognitiveLoop(
            memory_routing_mode="adaptive",
            memory_top_k=4,
            memory_routing_policy_evidence=evidence,
            routing_features=("node_count", "max_depth", "boundary_pressure"),
        )
        first = loop.propose(
            cycle_id="cycle-adaptive",
            data=self.data,
            required=["task.id", "task.action"],
            candidates=self.candidates,
        )
        loop.observe(
            first,
            family_id="family-adaptive",
            cost=1.0,
            strategy="selection_representation_B",
        )

        second = loop.propose(
            cycle_id="cycle-adaptive-2",
            data=self.data,
            required=["task.id", "task.action"],
            candidates=self.candidates,
        )

        self.assertEqual(second.memory_routing_mode, "adaptive")
        self.assertEqual(second.memory_routing_selected_mode, "credit")
        self.assertEqual(second.memory_policy_version, 11)
        self.assertEqual(second.memory_route.strategy, "selection_representation_B")


    def test_adaptive_mode_can_use_policy_meta_memory(self):
        from promptforge import ContextRoutingPolicyHistory

        history = ContextRoutingPolicyHistory()
        history.record(
            ContextRoutingPolicyEvidence(
                version=10,
                scores=(
                    ContextRoutingModeScore("nearest", 5, 3, 0.4, 2.0, 0.8),
                    ContextRoutingModeScore("credit", 5, 3, 0.8, 0.5, 0.2),
                ),
                selected_mode="nearest",
            )
        )
        history.record(
            ContextRoutingPolicyEvidence(
                version=11,
                scores=(
                    ContextRoutingModeScore("nearest", 5, 3, 0.4, 2.0, 0.8),
                    ContextRoutingModeScore("credit", 5, 3, 0.8, 0.5, 0.2),
                ),
                selected_mode="credit",
            )
        )

        loop = ContextCognitiveLoop(
            memory_routing_mode="adaptive",
            memory_routing_policy_history=history,
        )
        proposal = loop.propose(
            cycle_id="cycle-meta-memory",
            data=self.data,
            required=["task.id"],
            candidates=self.candidates,
        )

        self.assertEqual(proposal.schema_version, "context-cognitive-proposal.v5")
        self.assertEqual(proposal.memory_routing_mode, "adaptive")
        self.assertEqual(proposal.memory_routing_selected_mode, "credit")
        self.assertEqual(proposal.memory_policy_version, 11)
        self.assertEqual(proposal.memory_policy_history_version, 2)

    def test_policy_evidence_can_be_evaluated_and_recorded(self):
        from promptforge import ContextRoutingPolicyHistory

        loop = ContextCognitiveLoop(
            memory_routing_policy_history=ContextRoutingPolicyHistory()
        )
        first = loop.propose(
            cycle_id="cycle-meta-1",
            data=self.data,
            required=["task.id"],
            candidates=self.candidates,
        )
        loop.observe(
            first,
            family_id="family-a",
            cost=1.0,
            strategy="selection_only",
        )
        second = loop.propose(
            cycle_id="cycle-meta-2",
            data=self.data,
            required=["task.id"],
            candidates=self.candidates,
        )
        loop.observe(
            second,
            family_id="family-b",
            cost=2.0,
            strategy="selection_representation_B",
        )
        third = loop.propose(
            cycle_id="cycle-meta-3",
            data=self.data,
            required=["task.id"],
            candidates=self.candidates,
        )
        loop.observe(
            third,
            family_id="family-c",
            cost=1.5,
            strategy="selection_only",
        )

        evidence = loop.evaluate_and_record_memory_routing_policy()

        self.assertEqual(evidence.version, 3)
        self.assertEqual(loop.policy_history_snapshot().version, 1)
        self.assertEqual(
            loop.policy_history_snapshot().latest.version,
            3,
        )


    def test_adaptive_mode_falls_back_when_policy_history_is_unstable(self):
        from promptforge import ContextRoutingPolicyHistory

        history = ContextRoutingPolicyHistory()
        for version, mode in (
            (10, "nearest"),
            (11, "credit"),
            (12, "nearest"),
        ):
            history.record(
                ContextRoutingPolicyEvidence(
                    version=version,
                    scores=(
                        ContextRoutingModeScore("nearest", 5, 3, 0.6, 1.0, 0.4),
                        ContextRoutingModeScore("credit", 5, 3, 0.7, 0.9, 0.3),
                    ),
                    selected_mode=mode,
                )
            )

        loop = ContextCognitiveLoop(
            memory_routing_mode="adaptive",
            memory_routing_policy_history=history,
            memory_policy_history_min_observations=3,
            memory_policy_min_stability=0.5,
        )
        proposal = loop.propose(
            cycle_id="cycle-unstable-policy",
            data=self.data,
            required=["task.id"],
            candidates=self.candidates,
        )

        self.assertEqual(proposal.schema_version, "context-cognitive-proposal.v5")
        self.assertEqual(proposal.memory_routing_selected_mode, "nearest")
        self.assertEqual(proposal.memory_policy_stability_rate, 0.0)
        self.assertTrue(proposal.memory_policy_refresh_recommended)

    def test_invalid_memory_routing_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            ContextCognitiveLoop(memory_routing_mode="unknown")

    def test_default_memory_routing_mode_is_nearest(self):
        loop = ContextCognitiveLoop()
        proposal = loop.propose(
            cycle_id="cycle-default-mode",
            data=self.data,
            required=["task.id"],
            candidates=self.candidates,
        )

        self.assertEqual(proposal.memory_routing_mode, "nearest")
        self.assertEqual(proposal.memory_top_k, 5)

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

        self.assertEqual(proposal.schema_version, "context-cognitive-proposal.v4")
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

    def test_credit_memory_mode_uses_comparative_evidence(self):
        from promptforge import ContextMemoryCreditPolicy

        loop = ContextCognitiveLoop(
            memory_routing_mode="credit",
            memory_top_k=4,
            memory_credit_policy=ContextMemoryCreditPolicy(half_life=100.0),
            routing_features=("node_count", "max_depth", "boundary_pressure"),
        )
        first = loop.propose(
            cycle_id="cycle-credit",
            data=self.data,
            required=["task.id", "task.action"],
            candidates=self.candidates,
        )

        loop.observe(
            first,
            family_id="family-credit",
            cost=3.0,
            strategy="selection_only",
        )
        loop.observe(
            first,
            family_id="family-credit",
            cost=1.0,
            strategy="selection_representation_B",
        )

        second = loop.propose(
            cycle_id="cycle-credit-2",
            data=self.data,
            required=["task.id", "task.action"],
            candidates=self.candidates,
        )

        self.assertEqual(second.memory_routing_mode, "credit")
        self.assertIsNotNone(second.memory_route)
        self.assertEqual(second.memory_route.strategy, "selection_representation_B")
        self.assertEqual(second.decision.source, "episodic_memory")
        self.assertEqual(
            second.decision.strategy,
            "selection_representation_B",
        )
        payload = second.memory_route.to_dict()
        self.assertGreater(payload["selected_credit"], 0.0)
        self.assertEqual(payload["top_k"], 4)
        self.assertTrue(payload["strategy_scores"])

    def test_prepare_materializes_controller_choice_in_one_call(self):
        loop = ContextCognitiveLoop(
            routing_features=("node_count", "max_depth"),
        )
        result = loop.prepare(
            cycle_id="cycle-prepare",
            data=self.data,
            required=["task.id", "task.action"],
        )

        self.assertEqual(
            result.proposal.decision.strategy,
            result.prepared["selected_arm"],
        )
        self.assertEqual(result.prepared["task_family"], "agent_request")
        self.assertTrue(result.prepared["required_values_preserved"])

    def test_observation_persists_full_trajectory_and_decision_provenance(self):
        from promptforge import ContextTrajectoryMonitor

        trajectory = ContextTrajectoryMonitor(stagnation_patience=2).observe([
            {"cost": 10.0, "accepted": 1, "rejected": 0},
            {"cost": 9.0, "accepted": 1, "rejected": 0},
            {"cost": 9.0, "accepted": 0, "rejected": 1},
            {"cost": 9.0, "accepted": 0, "rejected": 1},
        ])
        loop = ContextCognitiveLoop()
        proposal = loop.propose(
            cycle_id="cycle-trace",
            data=self.data,
            required=["task.id", "task.action"],
            candidates=self.candidates,
            trajectory=trajectory,
            current_strategy="selection_only",
            current_cost=9.0,
            remaining_budget_fraction=0.5,
        )
        loop.observe(proposal, family_id="family-trace", cost=8.5)

        episode = loop.snapshot().episodes[0]
        self.assertEqual(episode.trajectory_state, trajectory.state)
        self.assertEqual(episode.trajectory["stagnation_length"], trajectory.stagnation_length)
        self.assertEqual(episode.regime_flags, proposal.decision.regime_flags)
        self.assertEqual(episode.decision_reason, proposal.decision.reason)
        self.assertEqual(episode.candidate_order, proposal.candidate_order)
    def test_experience_summary_is_descriptive_and_deterministic(self):
        from promptforge import experience_summary

        loop = ContextCognitiveLoop()
        proposal = loop.propose(
            cycle_id="cycle-summary",
            data=self.data,
            required=["task.id"],
            candidates=self.candidates,
        )
        loop.observe(
            proposal,
            family_id="family-summary",
            cost=2.0,
            outcome={"status": "success"},
        )

        summary = experience_summary(loop.snapshot().episodes)
        self.assertEqual(summary["episodes"], 1)
        self.assertEqual(summary["families"], 1)
        self.assertEqual(summary["strategies"], 1)
        self.assertEqual(summary["actions"][proposal.decision.action], 1)
        self.assertEqual(summary["sources"][proposal.decision.source], 1)
        self.assertEqual(summary["regimes"][proposal.decision.regime], 1)
        self.assertAlmostEqual(summary["mean_cost"], 2.0)
    def test_episode_state_is_serializable_for_audit(self):
        loop = ContextCognitiveLoop()
        proposal = loop.propose(
            cycle_id="cycle-audit",
            data=self.data,
            required=["task.id"],
            candidates=self.candidates,
        )
        loop.observe(
            proposal,
            family_id="family-audit",
            cost=1.25,
            outcome={"status": "observed"},
        )

        episode = loop.snapshot().episodes[0]
        payload = episode.to_dict()
        self.assertEqual(payload["episode_id"], "cycle-audit")
        self.assertEqual(payload["source"], proposal.decision.source)
        self.assertEqual(payload["outcome"]["status"], "observed")
    def test_observation_persists_cognitive_state_and_outcome(self):
        loop = ContextCognitiveLoop()
        proposal = loop.propose(
            cycle_id="cycle-state",
            data=self.data,
            required=["task.id", "task.action"],
            candidates=self.candidates,
        )
        loop.observe(
            proposal,
            family_id="family-state",
            cost=0.75,
            outcome={"status": "success", "quality": 0.91},
        )

        episode = loop.snapshot().episodes[0]
        self.assertEqual(episode.action, proposal.decision.action)
        self.assertEqual(episode.source, proposal.decision.source)
        self.assertEqual(episode.regime, proposal.decision.regime)
        self.assertEqual(
            episode.novelty_distance,
            proposal.decision.novelty_distance,
        )
        self.assertIsNone(episode.trajectory_state)
        self.assertEqual(episode.outcome["status"], "success")
        self.assertEqual(episode.cost, 0.75)
    def test_relational_topology_is_carried_into_future_experience(self):
        from promptforge import ContextRelation

        loop = ContextCognitiveLoop()
        proposal = loop.propose(
            cycle_id="cycle-rel",
            data=self.data,
            required=["task.id"],
            candidates=self.candidates,
            relations=[
                ContextRelation("task", "user", "owned_by"),
                ContextRelation("task", "goal", "targets"),
            ],
        )
        loop.observe(proposal, family_id="family-R", cost=1.0)
        stored = loop.snapshot().episodes[0].topology

        self.assertEqual(proposal.relational_profile.relation_count, 2)
        self.assertEqual(stored["relation_count"], 2)
        self.assertEqual(stored["relation_components"], 1)


if __name__ == "__main__":
    unittest.main()
