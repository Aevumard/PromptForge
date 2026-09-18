from __future__ import annotations

from typing import Any

from .provider_adapter import (
    NormalizedProviderResult,
    ProviderAdapter,
    build_result,
)


class DeepSeekProviderAdapter(ProviderAdapter):

    provider_name = "deepseek"

    def __init__(
        self,
        *,
        provider: Any | None = None,
    ) -> None:

        if provider is None:
            from .deepseek import DeepSeekProvider

            provider = DeepSeekProvider()

        self.provider = provider

    def generate(
        self,
        prompt: str,
    ) -> NormalizedProviderResult:

        try:
            result = self.provider.generate(
                prompt
            )

            raw_status = result.get(
                "status"
            )

            text = result.get(
                "text",
                "",
            )

            if not isinstance(text, str):
                text = ""

            return build_result(
                provider=self.provider_name,
                model=result.get(
                    "model",
                    "",
                ),
                raw_status=raw_status,
                input_tokens=result.get(
                    "input_tokens",
                    0,
                ),
                reasoning_tokens=result.get(
                    "reasoning_tokens",
                    0,
                ),
                output_tokens=result.get(
                    "output_tokens",
                    0,
                ),
                total_tokens=result.get(
                    "total_tokens",
                    0,
                ),
                latency_ms=result.get(
                    "latency_ms",
                    0,
                ),
                raw_text=text,
                response_id=result.get(
                    "response_id"
                ),
                error=result.get(
                    "error"
                ),
            )

        except Exception as exc:

            return build_result(
                provider=self.provider_name,
                model="",
                raw_status="ERROR",
                error=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )
