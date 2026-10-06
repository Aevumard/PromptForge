import unittest

from promptforge.tool_output import (
    ToolOutputItem,
    ToolOutputTrimPolicy,
    ToolOutputTrimmer,
    trim_tool_outputs,
)


class ToolOutputTrimTests(unittest.TestCase):
    def test_recent_output_is_preserved(self) -> None:
        item = ToolOutputItem(
            "recent",
            "search",
            "x" * 1000,
            turn_index=0,
        )
        result = trim_tool_outputs(
            [item],
            recent_turns=2,
            max_output_chars=100,
            preview_chars=40,
        )
        self.assertEqual(result.trimmed_ids, ())
        self.assertEqual(result.items[0].content, item.content)

    def test_old_output_keeps_head_and_tail_and_records_digest(self) -> None:
        content = "HEAD" + ("x" * 500) + "TAIL"
        item = ToolOutputItem("old", "execute", content, turn_index=3)
        result = ToolOutputTrimmer(
            ToolOutputTrimPolicy(
                recent_turns=2,
                max_output_chars=100,
                preview_chars=40,
            )
        ).trim([item])

        trimmed = result.items[0]
        self.assertEqual(result.trimmed_ids, ("old",))
        self.assertIn("HEAD", trimmed.content)
        self.assertIn("TAIL", trimmed.content)
        self.assertGreater(trimmed.chars_saved, 0)
        self.assertEqual(len(trimmed.content_sha256), 64)

    def test_eligible_tools_can_limit_trimming(self) -> None:
        items = (
            ToolOutputItem("a", "search", "x" * 1000, turn_index=4),
            ToolOutputItem("b", "database", "x" * 1000, turn_index=4),
        )
        result = trim_tool_outputs(
            items,
            recent_turns=2,
            max_output_chars=100,
            preview_chars=40,
            eligible_tools={"search"},
        )
        self.assertEqual(result.trimmed_ids, ("a",))
        self.assertEqual(result.items[1].content, items[1].content)

    def test_model_input_is_compact_and_provider_agnostic(self) -> None:
        item = ToolOutputItem("old", "search", "x" * 1000, turn_index=4)
        result = trim_tool_outputs(
            [item],
            max_output_chars=100,
            preview_chars=40,
        )
        payload = result.model_input[0]
        self.assertTrue(payload["trimmed"])
        self.assertLess(len(payload["content"]), len(item.content))


if __name__ == "__main__":
    unittest.main()
