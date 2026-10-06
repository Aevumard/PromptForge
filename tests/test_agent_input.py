import unittest

from promptforge import (
    AgentInputPacket,
    AgentPreparation,
    ContextBlock,
    ToolOutputItem,
    prepare_agent_input,
)


class AgentInputTests(unittest.TestCase):
    def test_single_call_combines_budget_deferred_and_tool_output_trim(self):
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
                {"details": "old"},
                utility=0.5,
                path="history",
                token_estimate=5,
            ),
        )
        result = prepare_agent_input(
            blocks,
            budget_tokens=3,
            descriptions={"history": "customer history"},
            tool_outputs=(
                ToolOutputItem(
                    "tool-1",
                    "search",
                    "x" * 1000,
                    turn_index=4,
                ),
            ),
            max_tool_output_chars=100,
            tool_preview_chars=40,
        )

        self.assertIsInstance(result, AgentPreparation)
        self.assertIsInstance(result.packet, AgentInputPacket)
        self.assertEqual(result.packet.context, {"case": {"id": "T-1"}})
        self.assertIn("history", result.packet.omitted_context_ids)
        self.assertEqual(
            result.packet.deferred_catalog["items"][0]["block_id"],
            "history",
        )
        self.assertEqual(result.packet.tool_outputs[0]["trimmed"], True)
        self.assertGreater(result.packet.total_estimated_savings, 0)

    def test_common_packet_is_small_and_serializable(self):
        blocks = (
            ContextBlock(
                "case",
                {"id": "T-2"},
                required=True,
                path="case",
                token_estimate=2,
            ),
        )
        packet = prepare_agent_input(
            blocks,
            budget_tokens=10,
        ).packet.to_dict()

        self.assertEqual(packet["schema_version"], "agent-input.v1")
        self.assertEqual(packet["context_tokens"], 2)
        self.assertIsInstance(packet["deferred_catalog"], dict)
        self.assertEqual(packet["tool_outputs"], [])


if __name__ == "__main__":
    unittest.main()
