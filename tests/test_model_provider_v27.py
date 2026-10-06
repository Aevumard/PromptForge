import json
import os
from unittest import TestCase
from unittest.mock import patch

from benchmarks.providers.openai_compatible import (
    OpenAICompatibleConfig,
    discover_ollama_model,
)

class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        return False
    def read(self) -> bytes:
        return json.dumps(self.payload).encode('utf-8')

class LocalOllamaAutoDetectTests(TestCase):
    def test_discover_ollama_model_returns_deterministic_name(self) -> None:
        payload = {'models': [{'name': 'qwen3:8b'}, {'name': 'gemma3:4b'}, {'name': 'llama3:8b'}]}
        with patch('benchmarks.providers.openai_compatible.urllib.request.urlopen', return_value=_FakeResponse(payload)):
            self.assertEqual(discover_ollama_model('http://localhost:11434'), 'gemma3:4b')

    def test_from_env_autodetects_local_ollama_without_provider_endpoint(self) -> None:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith('PROMPTFORGE_MODEL_')
            and key != 'PROMPTFORGE_OLLAMA_BASE_URL'
        }
        with patch.dict(os.environ, env, clear=True):
            with patch('benchmarks.providers.openai_compatible.discover_ollama_model', return_value='qwen3:8b') as discover:
                config = OpenAICompatibleConfig.from_env()
        self.assertEqual(config.endpoint_url, 'http://localhost:11434/v1/chat/completions')
        self.assertEqual(config.model, 'qwen3:8b')
        self.assertEqual(config.api_key, 'ollama')
        discover.assert_called_once_with('http://localhost:11434')

    def test_explicit_openai_compatible_endpoint_still_requires_model(self) -> None:
        env = {'PROMPTFORGE_MODEL_URL': 'http://example.test/v1/chat/completions'}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                OpenAICompatibleConfig.from_env()

if __name__ == '__main__':
    import unittest
    unittest.main()