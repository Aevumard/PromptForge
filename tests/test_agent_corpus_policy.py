from pathlib import Path
import unittest


class AgentCorpusPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.agents = (cls.root / "AGENTS.md").read_text(encoding="utf-8")
        cls.readme = (cls.root / "README.md").read_text(encoding="utf-8")
        cls.external = (
            cls.root / "docs" / "agent_corpus_external_patterns.md"
        ).read_text(encoding="utf-8")

    def test_priority_and_actionability_remain_separate(self):
        text = self.agents + "\n" + self.readme
        required = [
            "priority",
            "actionability",
            "high-priority case can legitimately be blocked for action",
            "UncertaintyActionGate",
        ]
        for token in required:
            self.assertIn(token, text)

    def test_external_pattern_corpus_is_wired(self):
        self.assertIn("LangGraph", self.external)
        self.assertIn("DSPy", self.external)
        self.assertIn("Pydantic AI", self.external)
        self.assertIn("Guardrails", self.external)
        self.assertIn("LlamaIndex", self.external)

    def test_critical_external_boundaries_are_preserved(self):
        checks = [
            "retrieval -> candidate set -> admissibility",
            "bounded repair",
            "later outcomes may improve future policy evidence",
            "metric-first",
            "human intervention",
        ]
        for token in checks:
            self.assertIn(token, self.external)

    def test_durable_human_review_boundary_is_wired(self):
        for token in (
            "HumanReviewRecord",
            "fresh `ActionDecision`",
            "human-review.v1",
            "ActionExecutionGuard",
            "idempotency key",
            "action-execution.v1",
        ):
            self.assertIn(token, self.agents + "\n" + self.readme)

    def test_token_budget_boundary_is_wired(self):
        for token in (
            "ContextBudgetPlanner",
            "utility per token cost",
            "reserved headroom",
            "compact_manifest",
        ):
            self.assertIn(token, self.agents + "\n" + self.readme)


if __name__ == "__main__":
    unittest.main()
