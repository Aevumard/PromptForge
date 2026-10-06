import io
import json
from unittest import TestCase
from unittest.mock import patch
import urllib.error

from benchmarks.providers.openai_compatible import (
    OpenAICompatibleAgentAdapter,
    OpenAICompatibleConfig,
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


class ProviderRetryTests(TestCase):
    def _adapter(self) -> OpenAICompatibleAgentAdapter:
        return OpenAICompatibleAgentAdapter(
            OpenAICompatibleConfig(
                endpoint_url="http://example.test/v1/chat/completions",
                model="test-model",
                max_attempts=3,
                retry_backoff_seconds=0,
            )
        )

    def test_retries_transient_http_error_then_succeeds(self) -> None:
        inner = {
            "ticket_id": "T-0001",
            "category": "payments",
            "sla": "urgent",
            "priority": "P0",
            "action": "refund",
            "requires_human": False,
            "contradiction_detected": False,
        }
        transient = urllib.error.HTTPError(
            "http://example.test",
            503,
            "busy",
            {"Retry-After": "0"},
            io.BytesIO(b"busy"),
        )

        with patch(
            "urllib.request.urlopen",
            side_effect=[
                transient,
                _FakeResponse(json.dumps({"choices": [{"message": {"content": json.dumps(inner)}}]})),
            ],
        ) as mocked, patch("time.sleep") as sleep:
            result = self._adapter().predict({"ticket_id": "T-0001"})

        self.assertEqual(result["action"], "refund")
        self.assertEqual(mocked.call_count, 2)
        sleep.assert_called_once_with(0.0)

    def test_does_not_retry_non_transient_4xx(self) -> None:
        error = urllib.error.HTTPError(
            "http://example.test",
            400,
            "bad request",
            {},
            io.BytesIO(b"bad request"),
        )
        with patch("urllib.request.urlopen", side_effect=error) as mocked:
            with self.assertRaises(RuntimeError):
                self._adapter().predict({"ticket_id": "T-0001"})
        mocked.assert_called_once_with(mocked.call_args.args[0], timeout=60.0)


if __name__ == "__main__":
    import unittest

    unittest.main()
