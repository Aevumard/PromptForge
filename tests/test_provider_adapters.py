import unittest


from harness.providers.provider_adapter import (
    NormalizedProviderResult,
    ProviderAdapter,
    build_result,
)


class FakeAdapter(ProviderAdapter):

    provider_name = "fake"

    def __init__(self, result):
        self.result = result

    def generate(self, prompt):
        return self.result


class FakeDeepSeekProvider:

    def __init__(self, result):
        self.result = result

    def generate(self, prompt):
        return dict(self.result)


class ProviderAdapterContractTests(unittest.TestCase):

    def test_model_ok_contract(self):

        result = build_result(
            provider="openai",
            model="gpt-5-mini",
            raw_status="MODEL_OK",
            input_tokens=11,
            reasoning_tokens=0,
            output_tokens=43,
            total_tokens=54,
            latency_ms=3006.31,
            raw_text="OK",
            response_id="resp-test",
            error=None,
        )

        self.assertIsInstance(
            result,
            NormalizedProviderResult,
        )

        self.assertEqual(
            result.provider,
            "openai",
        )

        self.assertEqual(
            result.model,
            "gpt-5-mini",
        )

        self.assertEqual(
            result.provider_status,
            "MODEL_OK",
        )

        self.assertEqual(
            result.raw_text,
            "OK",
        )

        self.assertEqual(
            result.to_dict(),
            {
                "provider": "openai",
                "model": "gpt-5-mini",
                "provider_status": "MODEL_OK",
                "input_tokens": 11,
                "reasoning_tokens": 0,
                "output_tokens": 43,
                "total_tokens": 54,
                "latency_ms": 3006.31,
                "raw_text": "OK",
                "response_id": "resp-test",
                "error": None,
            },
        )

    def test_empty_output_contract(self):

        result = build_result(
            provider="google",
            model="gemini-3.6-flash",
            raw_status="MODEL_OK",
            input_tokens=8,
            reasoning_tokens=7,
            output_tokens=0,
            total_tokens=15,
            latency_ms=100.0,
            raw_text="",
            response_id="interaction-test",
            error=None,
        )

        self.assertEqual(
            result.provider_status,
            "EMPTY_OUTPUT",
        )

        self.assertEqual(
            result.raw_text,
            "",
        )

    def test_error_normalization(self):

        result = build_result(
            provider="deepseek",
            model="deepseek-reasoner",
            raw_status="API_TIMEOUT",
            error="timeout",
        )

        self.assertEqual(
            result.provider_status,
            "ERROR",
        )

        self.assertEqual(
            result.error,
            "timeout",
        )

    def test_interface_is_abstract(self):

        with self.assertRaises(TypeError):
            ProviderAdapter()

    def test_fake_adapter_contract(self):

        normalized = build_result(
            provider="fake",
            model="fake-model",
            raw_status="MODEL_OK",
            input_tokens=1,
            reasoning_tokens=2,
            output_tokens=3,
            total_tokens=6,
            latency_ms=1.5,
            raw_text="OK",
            response_id="fake-1",
        )

        adapter = FakeAdapter(
            normalized
        )

        result = adapter.generate(
            "test"
        )

        self.assertEqual(
            result.provider,
            "fake",
        )

        self.assertEqual(
            result.raw_text,
            "OK",
        )

    def test_deepseek_wrapper_normalizes_without_api(self):

        from harness.providers.deepseek_adapter import (
            DeepSeekProviderAdapter,
        )

        fake = FakeDeepSeekProvider(
            {
                "status": "MODEL_OK",
                "model": "deepseek-reasoner",
                "input_tokens": 10,
                "reasoning_tokens": 20,
                "output_tokens": 30,
                "total_tokens": 60,
                "latency_ms": 12.5,
                "text": "OK",
                "response_id": "ds-test",
                "error": None,
            }
        )

        adapter = DeepSeekProviderAdapter(
            provider=fake
        )

        result = adapter.generate(
            "test"
        )

        self.assertEqual(
            result.provider,
            "deepseek",
        )

        self.assertEqual(
            result.provider_status,
            "MODEL_OK",
        )

        self.assertEqual(
            result.input_tokens,
            10,
        )

        self.assertEqual(
            result.reasoning_tokens,
            20,
        )

        self.assertEqual(
            result.output_tokens,
            30,
        )

        self.assertEqual(
            result.total_tokens,
            60,
        )

        self.assertEqual(
            result.raw_text,
            "OK",
        )


if __name__ == "__main__":
    unittest.main()
