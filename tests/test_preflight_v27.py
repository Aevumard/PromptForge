import json
from unittest import TestCase
from unittest.mock import patch

from benchmarks.preflight_v27 import preflight_ollama, require_ready


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class PreflightTests(TestCase):
    def test_selects_requested_model_when_installed(self):
        payload = {"models": [{"name": "qwen3:8b"}, {"name": "gemma3:4b"}]}
        with patch(
            "benchmarks.preflight_v27.urllib.request.urlopen",
            return_value=_FakeResponse(payload),
        ):
            result = preflight_ollama(
                "http://localhost:11434",
                model="qwen3:8b",
            )
        self.assertTrue(result.reachable)
        self.assertTrue(result.model_available)
        self.assertEqual(result.selected_model, "qwen3:8b")
        self.assertEqual(result.endpoint_url, "http://localhost:11434/v1/chat/completions")

    def test_auto_selects_deterministically(self):
        payload = {"models": [{"name": "qwen3:8b"}, {"name": "gemma3:4b"}]}
        with patch(
            "benchmarks.preflight_v27.urllib.request.urlopen",
            return_value=_FakeResponse(payload),
        ):
            result = preflight_ollama("http://localhost:11434")
        self.assertEqual(result.selected_model, "gemma3:4b")

    def test_missing_model_is_not_ready(self):
        payload = {"models": [{"name": "qwen3:8b"}]}
        with patch(
            "benchmarks.preflight_v27.urllib.request.urlopen",
            return_value=_FakeResponse(payload),
        ):
            result = preflight_ollama(
                "http://localhost:11434",
                model="llama3:8b",
            )
        with self.assertRaises(RuntimeError):
            require_ready(result)
