import json
import os
from unittest import TestCase
from unittest.mock import patch

from benchmarks.model_loop_v27 import parse_prediction
from benchmarks.providers.openai_compatible import (
    OpenAICompatibleAgentAdapter,
    OpenAICompatibleConfig,
    _extract_prediction,
)


class _FakeResponse:
    def __init__(self, payload: str) -> None:
        self.payload = payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self.payload


class ProviderAdapterTests(TestCase):
    def test_extracts_direct_prediction_object(self) -> None:
        payload = {
            "ticket_id": "T-0001",
            "category": "payments",
            "sla": "urgent",
            "priority": "P0",
            "action": "human_review",
            "requires_human": True,
            "contradiction_detected": True,
        }
        extracted = _extract_prediction(json.dumps(payload))
        self.assertEqual(extracted["ticket_id"], "T-0001")

    def test_extracts_chat_completion_json(self) -> None:
        inner = {
            "ticket_id": "T-0001",
            "category": "payments",
            "sla": "urgent",
            "priority": "P0",
            "action": "refund",
            "requires_human": False,
            "contradiction_detected": False,
        }
        fence = chr(96) * 3
        response = {
            "choices": [
                {
                    "message": {
                        "content": fence
                        + "json\n"
                        + json.dumps(inner)
                        + "\n"
                        + fence
                    }
                }
            ]
        }
        self.assertEqual(_extract_prediction(json.dumps(response)), inner)

    def test_adapter_sends_model_and_auth(self) -> None:
        config = OpenAICompatibleConfig(
            endpoint_url="http://example.test/v1/chat/completions",
            model="test-model",
            api_key="secret",
        )
        adapter = OpenAICompatibleAgentAdapter(config)
        model_input = {"ticket_id": "T-0001", "promptforge": {"context_tokens": 7}}
        inner = {
            "ticket_id": "T-0001",
            "category": "payments",
            "sla": "urgent",
            "priority": "P0",
            "action": "refund",
            "requires_human": False,
            "contradiction_detected": False,
        }
        response = {"choices": [{"message": {"content": json.dumps(inner)}}]}

        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeResponse(json.dumps(response))

        with patch("urllib.request.urlopen", fake_urlopen):
            result = adapter.predict(model_input)

        self.assertEqual(result["action"], "refund")
        self.assertEqual(captured["timeout"], 60.0)
        self.assertEqual(
            captured["request"].headers["Authorization"],
            "Bearer secret",
        )
        sent = json.loads(captured["request"].data.decode("utf-8"))
        self.assertEqual(sent["model"], "test-model")
        self.assertEqual(sent["temperature"], 0)
        self.assertEqual(
            sent["messages"][1]["content"],
            json.dumps(model_input, ensure_ascii=False, sort_keys=True),
        )

    def test_parser_rejects_string_booleans(self) -> None:
        payload = {
            "ticket_id": "T-1",
            "category": "payments",
            "sla": "urgent",
            "priority": "P0",
            "action": "refund",
            "requires_human": "false",
            "contradiction_detected": False,
        }
        with self.assertRaises(ValueError):
            parse_prediction(payload, ticket_id="T-1")

        payload["requires_human"] = False
        payload["contradiction_detected"] = "true"
        with self.assertRaises(ValueError):
            parse_prediction(payload, ticket_id="T-1")

    def test_from_env_requires_endpoint_and_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                OpenAICompatibleConfig.from_env()


if __name__ == "__main__":
    import unittest

    unittest.main()
