import json
import os
from unittest import TestCase
from unittest.mock import MagicMock, patch

from benchmarks.model_loop_v27 import parse_prediction
from benchmarks.providers.openai_compatible import (
    OpenAICompatibleAgentAdapter,
    OpenAICompatibleConfig,
    ProviderCallTelemetry,
    _extract_prediction,
    main,
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
        response = {
            "choices": [{"message": {"content": json.dumps(inner)}}],
            "usage": {
                "prompt_tokens": 111,
                "completion_tokens": 17,
                "total_tokens": 128,
            },
        }

        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeResponse(json.dumps(response))

        with patch("urllib.request.urlopen", fake_urlopen):
            result = adapter.predict(model_input)

        self.assertEqual(result["action"], "refund")
        self.assertEqual(captured["timeout"], 60.0)
        self.assertIsInstance(adapter.last_call_telemetry, ProviderCallTelemetry)
        self.assertEqual(adapter.last_call_telemetry.attempts, 1)
        self.assertEqual(adapter.last_call_telemetry.prompt_tokens, 111)
        self.assertEqual(adapter.last_call_telemetry.completion_tokens, 17)
        self.assertEqual(adapter.last_call_telemetry.total_tokens, 128)
        self.assertGreaterEqual(adapter.last_call_telemetry.elapsed_ms, 0.0)
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


    def test_cli_passes_checkpoint_to_resumable_runner(self) -> None:
        config = OpenAICompatibleConfig(
            endpoint_url="http://example.test/v1/chat/completions",
            model="test-model",
        )
        fake_report = MagicMock()
        fake_report.to_dict.return_value = {"summary": {}}
        fake_report.raw_metrics.to_dict.return_value = {}
        fake_report.guarded_metrics.to_dict.return_value = {}

        argv = [
            "openai_compatible",
            "--count",
            "4",
            "--checkpoint",
            "run.checkpoint.jsonl",
            "--output",
            "predictions.jsonl",
            "--report",
            "report.json",
        ]
        with (
            patch("sys.argv", argv),
            patch(
                "benchmarks.providers.openai_compatible.OpenAICompatibleConfig.from_env",
                return_value=config,
            ),
            patch(
                "benchmarks.providers.openai_compatible.run_resumable_model_loop",
                return_value=fake_report,
            ) as run,
            patch(
                "benchmarks.providers.openai_compatible.write_prediction_jsonl"
            ),
            patch("benchmarks.providers.openai_compatible.write_report"),
            patch(
                "benchmarks.providers.openai_compatible.analyze_report"
            ) as analyze,
            patch(
                "benchmarks.providers.openai_compatible.render_markdown",
                return_value="# analysis",
            ),
            patch(
                "benchmarks.providers.openai_compatible.analyze_bootstrap"
            ) as bootstrap,
            patch(
                "benchmarks.providers.openai_compatible.render_bootstrap_markdown",
                return_value="# bootstrap",
            ),
        ):
            analyze.return_value.to_dict.return_value = {"summary": {}, "records": []}
            bootstrap.return_value.to_dict.return_value = {"summary": {}}
            self.assertEqual(main(), 0)

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["checkpoint_path"], "run.checkpoint.jsonl")
        self.assertTrue(kwargs["retry_failed"])
        self.assertTrue(kwargs["fsync_each_record"])
        analyze.assert_called_once()
        bootstrap.assert_called_once()

    def test_from_env_autodetects_local_ollama_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "benchmarks.providers.openai_compatible.discover_ollama_model",
                return_value="qwen3:8b",
            ):
                config = OpenAICompatibleConfig.from_env()

        self.assertEqual(
            config.endpoint_url,
            "http://localhost:11434/v1/chat/completions",
        )
        self.assertEqual(config.model, "qwen3:8b")
        self.assertEqual(config.api_key, "ollama")


if __name__ == "__main__":
    import unittest

    unittest.main()
