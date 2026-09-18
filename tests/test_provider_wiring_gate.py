import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PROVIDERS_DIR = ROOT / "harness" / "providers"
TASK_FILE = ROOT / "tasks" / "suite.json"

PROVIDER_FILES = {
    "base": PROVIDERS_DIR / "provider_adapter.py",
    "openai": PROVIDERS_DIR / "openai_adapter.py",
    "gemini": PROVIDERS_DIR / "gemini_adapter.py",
    "deepseek": PROVIDERS_DIR / "deepseek_adapter.py",
}


def read_source(path):
    return path.read_text(
        encoding="utf-8",
        errors="strict",
    )


def parse_source(path):
    source = read_source(path)
    return ast.parse(
        source,
        filename=str(path),
    )


def has_call_attribute(tree, attribute_name):
    for node in ast.walk(tree):

        if isinstance(node, ast.Call):

            func = node.func

            if isinstance(func, ast.Attribute):

                if func.attr == attribute_name:
                    return True

    return False


def has_attribute_read(tree, attribute_name):
    for node in ast.walk(tree):

        if isinstance(node, ast.Attribute):

            if node.attr == attribute_name:
                return True

    return False


def has_string_constant(tree, value):
    for node in ast.walk(tree):

        if isinstance(node, ast.Constant):

            if isinstance(node.value, str):

                if node.value == value:
                    return True

    return False


class ProviderWiringGateTests(unittest.TestCase):

    def test_provider_files_exist(self):

        for name, path in PROVIDER_FILES.items():

            self.assertTrue(
                path.exists(),
                f"missing provider file: {name}: {path}",
            )

    def test_provider_modules_compile(self):

        for name, path in PROVIDER_FILES.items():

            parse_source(path)

    def test_no_bom_in_provider_files(self):

        for name, path in PROVIDER_FILES.items():

            raw = path.read_bytes()

            self.assertFalse(
                raw.startswith(b"\xef\xbb\xbf"),
                f"UTF-8 BOM present in {name}: {path}",
            )

    def test_base_contract_exists(self):

        from harness.providers.provider_adapter import (
            NormalizedProviderResult,
            ProviderAdapter,
        )

        with self.assertRaises(TypeError):
            ProviderAdapter()

        result = NormalizedProviderResult(
            provider="test",
            model="test-model",
            provider_status="MODEL_OK",
            input_tokens=1,
            reasoning_tokens=2,
            output_tokens=3,
            total_tokens=6,
            latency_ms=1.5,
            raw_text="OK",
            response_id="test-id",
            error=None,
        )

        payload = result.to_dict()

        self.assertEqual(
            set(payload.keys()),
            {
                "provider",
                "model",
                "provider_status",
                "input_tokens",
                "reasoning_tokens",
                "output_tokens",
                "total_tokens",
                "latency_ms",
                "raw_text",
                "response_id",
                "error",
            },
        )

    def test_all_three_adapters_are_provider_adapters(self):

        from harness.providers.provider_adapter import (
            ProviderAdapter,
        )

        from harness.providers.deepseek_adapter import (
            DeepSeekProviderAdapter,
        )

        from harness.providers.openai_adapter import (
            OpenAIProviderAdapter,
        )

        from harness.providers.gemini_adapter import (
            GeminiProviderAdapter,
        )

        adapters = {
            "deepseek": DeepSeekProviderAdapter,
            "openai": OpenAIProviderAdapter,
            "google": GeminiProviderAdapter,
        }

        for name, adapter_cls in adapters.items():

            self.assertTrue(
                issubclass(
                    adapter_cls,
                    ProviderAdapter,
                ),
                f"{name} adapter is not a ProviderAdapter",
            )

            self.assertTrue(
                callable(
                    getattr(
                        adapter_cls,
                        "generate",
                        None,
                    )
                ),
                f"{name} adapter has no generate method",
            )

            self.assertEqual(
                adapter_cls.provider_name,
                name,
            )

    def test_openai_structural_wiring(self):

        path = PROVIDER_FILES["openai"]
        tree = parse_source(path)
        source = read_source(path)

        # Responses API call.
        self.assertTrue(
            has_call_attribute(
                tree,
                "create",
            )
        )

        self.assertIn(
            "responses.create",
            source,
        )

        # Current adapter uses:
        # getattr(response, "output_text", None)
        self.assertTrue(
            has_string_constant(
                tree,
                "output_text",
            )
        )

        # Reasoning configuration.
        self.assertTrue(
            has_string_constant(
                tree,
                "effort",
            )
        )

        self.assertTrue(
            has_string_constant(
                tree,
                "low",
            )
        )

        self.assertIn(
            "reasoning={",
            source,
        )

    def test_google_structural_wiring(self):

        path = PROVIDER_FILES["gemini"]
        tree = parse_source(path)
        source = read_source(path)

        # Interactions API call.
        self.assertTrue(
            has_call_attribute(
                tree,
                "create",
            )
        )

        self.assertIn(
            "interactions.create",
            source,
        )

        # Current adapter uses:
        # getattr(interaction, "output_text", None)
        self.assertTrue(
            has_string_constant(
                tree,
                "output_text",
            )
        )

        # Confirm the active model fallback.
        self.assertTrue(
            has_string_constant(
                tree,
                "gemini-3.6-flash",
            )
        )

        # Interactions usage fields.
        self.assertTrue(
            has_string_constant(
                tree,
                "total_input_tokens",
            )
        )

        self.assertTrue(
            has_string_constant(
                tree,
                "total_output_tokens",
            )
        )

        self.assertTrue(
            has_string_constant(
                tree,
                "total_thought_tokens",
            )
        )

        self.assertTrue(
            has_string_constant(
                tree,
                "total_tokens",
            )
        )

    def test_deepseek_structural_wiring(self):

        path = PROVIDER_FILES["deepseek"]
        tree = parse_source(path)
        source = read_source(path)

        self.assertTrue(
            has_call_attribute(
                tree,
                "generate",
            )
        )

        self.assertIn(
            "DeepSeekProvider",
            source,
        )

        self.assertIn(
            "self.provider.generate",
            source,
        )

    def test_deepseek_normalization_offline(self):

        from harness.providers.deepseek_adapter import (
            DeepSeekProviderAdapter,
        )

        class FakeDeepSeekProvider:

            def generate(self, prompt):

                return {
                    "status": "MODEL_OK",
                    "model": "deepseek-reasoner",
                    "input_tokens": 100,
                    "reasoning_tokens": 50,
                    "output_tokens": 25,
                    "total_tokens": 175,
                    "latency_ms": 10.0,
                    "text": '{"ok":true}',
                    "response_id": "offline-deepseek",
                    "error": None,
                }

        adapter = DeepSeekProviderAdapter(
            provider=FakeDeepSeekProvider()
        )

        result = adapter.generate(
            "offline test"
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
            result.model,
            "deepseek-reasoner",
        )

        self.assertEqual(
            result.input_tokens,
            100,
        )

        self.assertEqual(
            result.reasoning_tokens,
            50,
        )

        self.assertEqual(
            result.output_tokens,
            25,
        )

        self.assertEqual(
            result.total_tokens,
            175,
        )

        self.assertEqual(
            result.raw_text,
            '{"ok":true}',
        )

    def test_normalization_status_rules(self):

        from harness.providers.provider_adapter import (
            build_result,
        )

        ok = build_result(
            provider="openai",
            model="gpt-5-mini",
            raw_status="MODEL_OK",
            raw_text="OK",
        )

        empty = build_result(
            provider="google",
            model="gemini-3.6-flash",
            raw_status="MODEL_OK",
            raw_text="",
        )

        error = build_result(
            provider="deepseek",
            model="deepseek-reasoner",
            raw_status="API_TIMEOUT",
            raw_text="",
            error="timeout",
        )

        self.assertEqual(
            ok.provider_status,
            "MODEL_OK",
        )

        self.assertEqual(
            empty.provider_status,
            "EMPTY_OUTPUT",
        )

        self.assertEqual(
            error.provider_status,
            "ERROR",
        )

    def test_quality_is_not_provider_status(self):

        from harness.evaluators.deterministic import evaluate

        suite = json.loads(
            TASK_FILE.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            len(suite["tasks"]),
            4,
        )

        for task in suite["tasks"]:

            expected = {
                key: task["data"][key]
                for key in task["required"]
            }

            passed, verification = evaluate(
                expected,
                expected,
            )

            self.assertTrue(
                passed,
                f"deterministic exact match failed for {task['task_id']}",
            )

            self.assertTrue(
                verification["passed"],
                f"verification failed for {task['task_id']}",
            )

            wrong = dict(expected)

            first_key = task["required"][0]

            wrong[first_key] = (
                "__STATIC_GATE_WRONG_VALUE__"
            )

            wrong_passed, wrong_verification = evaluate(
                expected,
                wrong,
            )

            self.assertFalse(
                wrong_passed,
                f"false quality pass for {task['task_id']}",
            )

            self.assertFalse(
                wrong_verification["passed"],
                f"verification accepted incorrect output for {task['task_id']}",
            )

    def test_suite_contains_the_four_frozen_tasks(self):

        suite = json.loads(
            TASK_FILE.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            [
                task["task_id"]
                for task in suite["tasks"]
            ],
            [
                "T001",
                "T002",
                "T003",
                "T004",
            ],
        )

    def test_imports_do_not_execute_provider_calls(self):

        import harness.providers.provider_adapter
        import harness.providers.openai_adapter
        import harness.providers.gemini_adapter
        import harness.providers.deepseek_adapter

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
