from __future__ import annotations

import os
import time

from .provider_adapter import (
    NormalizedProviderResult,
    ProviderAdapter,
    build_result,
)


class OpenAIProviderAdapter(ProviderAdapter):

    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> None:

        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv(
                "OPENAI_API_KEY",
                "",
            ).strip()
        )

        self.model = (
            model
            if model is not None
            else (
                os.getenv(
                    "OPENAI_MODEL",
                    "gpt-5-mini",
                ).strip()
                or "gpt-5-mini"
            )
        )

        self.max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else int(
                os.getenv(
                    "OPENAI_MAX_OUTPUT_TOKENS",
                    "512",
                )
            )
        )

        self.reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else os.getenv(
                "OPENAI_REASONING_EFFORT",
                "low",
            ).strip()
        )

    def generate(
        self,
        prompt: str,
    ) -> NormalizedProviderResult:

        if not self.api_key:
            return build_result(
                provider=self.provider_name,
                model=self.model,
                raw_status="ERROR",
                error="OPENAI_API_KEY not present",
            )

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key
            )

            started = time.perf_counter()

            response = client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=self.max_output_tokens,
                reasoning={
                    "effort": self.reasoning_effort,
                },
            )

            elapsed = (
                time.perf_counter()
                - started
            ) * 1000.0

            text = getattr(
                response,
                "output_text",
                None,
            )

            if not isinstance(text, str):
                text = ""

            usage = getattr(
                response,
                "usage",
                None,
            )

            input_tokens = 0
            output_tokens = 0
            reasoning_tokens = 0
            total_tokens = 0

            if usage is not None:

                input_tokens = int(
                    getattr(
                        usage,
                        "input_tokens",
                        0,
                    )
                    or 0
                )

                output_tokens = int(
                    getattr(
                        usage,
                        "output_tokens",
                        0,
                    )
                    or 0
                )

                total_tokens = int(
                    getattr(
                        usage,
                        "total_tokens",
                        input_tokens
                        + output_tokens,
                    )
                    or 0
                )

                details = getattr(
                    usage,
                    "output_tokens_details",
                    None,
                )

                if details is not None:
                    reasoning_tokens = int(
                        getattr(
                            details,
                            "reasoning_tokens",
                            0,
                        )
                        or 0
                    )

            response_status = getattr(
                response,
                "status",
                None,
            )

            raw_status = (
                "MODEL_OK"
                if response_status == "completed"
                and text.strip()
                else (
                    "EMPTY_OUTPUT"
                    if response_status == "completed"
                    else "ERROR"
                )
            )

            error = None

            return build_result(
                provider=self.provider_name,
                model=getattr(
                    response,
                    "model",
                    self.model,
                ),
                raw_status=raw_status,
                input_tokens=input_tokens,
                reasoning_tokens=reasoning_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=elapsed,
                raw_text=text,
                response_id=getattr(
                    response,
                    "id",
                    None,
                ),
                error=error,
            )

        except Exception as exc:

            return build_result(
                provider=self.provider_name,
                model=self.model,
                raw_status="ERROR",
                error=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )
