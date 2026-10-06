import unittest

from promptforge.budget import (
    ContextBlock,
    ContextBudgetPlanner,
    ContextBudgetPolicy,
    plan_context,
)


class BudgetTests(unittest.TestCase):
    def test_required_blocks_are_pinned_and_optional_blocks_pack_by_density(self):
        blocks = (
            ContextBlock("system", {"role": "system"}, required=True),
            ContextBlock("critical", {"id": "A-1", "sla": "2h"}, utility=10.0),
            ContextBlock("useful", {"history": "relevant"}, utility=4.0),
            ContextBlock("noise", {"commentary": "x" * 400}, utility=0.1),
        )
        plan = plan_context(
            blocks,
            budget_tokens=12,
            reserve_tokens=2,
        )
        self.assertIn("system", plan.included_ids)
        self.assertIn("critical", plan.included_ids)
        self.assertNotIn("noise", plan.included_ids)
        self.assertLessEqual(plan.selected_tokens, plan.usable_tokens)
        self.assertGreaterEqual(plan.tokens_saved, 0)

    def test_required_overflow_fails_closed(self):
        blocks = (
            ContextBlock("required", "x" * 1000, required=True),
        )
        with self.assertRaises(ValueError):
            ContextBudgetPlanner(
                ContextBudgetPolicy(budget_tokens=10),
            ).plan(blocks)

    def test_compact_manifest_excludes_content_for_omitted_blocks(self):
        blocks = (
            ContextBlock("keep", {"id": "T-1"}, required=True, path="task.id"),
            ContextBlock("omit", {"secret": "large"}, utility=0.1, path="private"),
        )
        plan = plan_context(blocks, budget_tokens=4)
        manifest = plan.compact_manifest(blocks)
        self.assertEqual(manifest["included"][0]["block_id"], "keep")
        self.assertEqual(manifest["omitted"][0]["block_id"], "omit")
        self.assertNotIn("large", str(manifest))

    def test_materialize_rebuilds_selected_nested_context(self):
        blocks = (
            ContextBlock("user", "Ada", required=True, path="user.name"),
            ContextBlock("task", "review", required=True, path="task.action"),
        )
        plan = plan_context(blocks, budget_tokens=4)
        self.assertEqual(
            plan.materialize(blocks),
            {"user": {"name": "Ada"}, "task": {"action": "review"}},
        )

    def test_reserve_ratio_creates_headroom(self):
        policy = ContextBudgetPolicy(
            budget_tokens=100,
            reserve_ratio=0.10,
        )
        self.assertEqual(policy.effective_reserve_tokens, 10)
        self.assertEqual(policy.usable_tokens, 90)


if __name__ == "__main__":
    unittest.main()
