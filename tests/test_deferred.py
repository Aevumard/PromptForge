import unittest

from promptforge.budget import ContextBlock, plan_context
from promptforge.deferred import (
    DeferredContextCatalog,
    build_context_packet,
)


class DeferredContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.blocks = (
            ContextBlock(
                "history",
                {"customer": {"id": "C-1", "orders": 12}},
                utility=2.0,
                path="customer_history",
                token_estimate=5,
            ),
            ContextBlock(
                "logs",
                {"events": ["e1", "e2"]},
                utility=3.0,
                path="incident_logs",
                token_estimate=4,
            ),
            ContextBlock(
                "notes",
                {"text": "long notes"},
                utility=0.5,
                path="notes",
                token_estimate=6,
            ),
        )

    def test_catalog_is_metadata_only(self) -> None:
        catalog = DeferredContextCatalog.from_blocks(
            self.blocks,
            descriptions={"history": "recent customer history"},
        ).catalog()
        self.assertEqual(catalog["count"], 3)
        self.assertEqual(
            catalog["items"][0]["description"],
            "recent customer history",
        )
        self.assertNotIn("orders", str(catalog))
        self.assertNotIn("events", str(catalog))

    def test_explicit_load_returns_only_requested_blocks(self) -> None:
        catalog = DeferredContextCatalog.from_blocks(self.blocks)
        result = catalog.load(["logs"])
        self.assertEqual(result.loaded_ids, ("logs",))
        self.assertEqual(result.unknown_ids, ())
        self.assertEqual(
            result.materialized,
            {"incident_logs": {"events": ["e1", "e2"]}},
        )
        self.assertEqual(result.selected_tokens, 4)

    def test_load_budget_defers_expensive_block(self) -> None:
        catalog = DeferredContextCatalog.from_blocks(self.blocks)
        result = catalog.load(["notes", "logs"], token_budget=4)
        self.assertEqual(result.loaded_ids, ("logs",))
        self.assertEqual(result.omitted_ids, ("notes",))

    def test_delivery_packet_is_model_facing_and_omits_deferred_values(self) -> None:
        blocks = (
            ContextBlock(
                "case",
                {"id": "T-1"},
                required=True,
                path="case",
                token_estimate=2,
            ),
            ContextBlock(
                "history",
                {"secret": "large-history"},
                utility=0.5,
                path="history",
                token_estimate=5,
            ),
        )
        plan = plan_context(blocks, budget_tokens=3)
        packet = build_context_packet(
            plan,
            blocks,
            descriptions={"history": "customer history"},
        ).to_dict()

        self.assertEqual(packet["context"], {"case": {"id": "T-1"}})
        self.assertEqual(packet["context_tokens"], 2)
        self.assertEqual(
            packet["deferred_catalog"]["items"][0]["block_id"],
            "history",
        )
        self.assertNotIn("large-history", str(packet))
        self.assertEqual(
            packet["deferred_catalog"]["items"][0]["description"],
            "customer history",
        )

    def test_unknown_ids_are_auditable(self) -> None:
        catalog = DeferredContextCatalog.from_blocks(self.blocks)
        result = catalog.load(["missing-id"])
        self.assertEqual(result.unknown_ids, ("missing-id",))
        self.assertEqual(result.materialized, {})

    def test_budget_plan_can_feed_deferred_catalog(self) -> None:
        blocks = (
            ContextBlock("required", {"id": "T-1"}, required=True, token_estimate=2),
            ContextBlock("history", {"x": "y"}, utility=1.0, token_estimate=4),
            ContextBlock("noise", {"x": "z"}, utility=0.1, token_estimate=4),
        )
        plan = plan_context(blocks, budget_tokens=6)
        catalog = DeferredContextCatalog.from_budget_plan(plan, blocks)
        entries = catalog.catalog()
        self.assertEqual(entries["count"], 1)
        self.assertEqual(entries["items"][0]["block_id"], "noise")


if __name__ == "__main__":
    unittest.main()
