from __future__ import annotations

import os
import time

from .provider_adapter import (
    NormalizedProviderResult,
    ProviderAdapter,
    build_result,
)


class GeminiProviderAdapter(ProviderAdapter):

    provider_name = "google"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:

        self.api_key = (
            api_key
            if api_key is not None
            else (
                os.getenv(
                    "GEMINI_API_KEY",
                    "",
                ).strip()
                or os.getenv(
                    "GOOGLE_API_KEY",
                    "",
                ).strip()
            )
        )

        self.model = (
            model
            if model is not None
            else (
                os.getenv(
                    "GEMINI_MODEL",
                    "gemini-3.6-flash",
                ).strip()
                or "gemini-3.6-flash"
            )
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
                error=(
                    "GEMINI_API_KEY / "
                    "GOOGLE_API_KEY not present"
                ),
            )

        try:
            from google import genai

            client = genai.Client(
                api_key=self.api_key
            )

            started = time.perf_counter()

            interaction = client.interactions.create(
                model=self.model,
                input=prompt,
            )

            elapsed = (
                time.perf_counter()
                - started
            ) * 1000.0

            text = getattr(
                interaction,
                "output_text",
                None,
            )

            if not isinstance(text, str):
                text = ""

            usage = getattr(
                interaction,
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
                        "total_input_tokens",
                        0,
                    )
                    or 0
                )

                output_tokens = int(
                    getattr(
                        usage,
                        "total_output_tokens",
                        0,
                    )
                    or 0
                )

                reasoning_tokens = int(
                    getattr(
                        usage,
                        "total_thought_tokens",
                        0,
                    )
                    or 0
                )

                total_tokens = int(
                    getattr(
                        usage,
                        "total_tokens",
                        (
                            input_tokens
                            + output_tokens
                            + reasoning_tokens
                        ),
                    )
                    or 0
                )

            interaction_status = getattr(
                interaction,
                "status",
                None,
            )

            raw_status = (
                "MODEL_OK"
                if interaction_status == "completed"
                and text.strip()
                else (
                    "EMPTY_OUTPUT"
                    if interaction_status == "completed"
                    else "ERROR"
                )
            )

            return build_result(
                provider=self.provider_name,
                model=getattr(
                    interaction,
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
                    interaction,
                    "id",
                    None,
                ),
                error=None,
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
