from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping
import urllib.error
import urllib.request

from benchmarks.model_loop_v27 import ModelLoopReport, run_model_loop
from benchmarks.tickets_v27 import generate_ticket_suite


DEFAULT_SYSTEM_PROMPT = (
    "You are the classification agent in a controlled support benchmark. "
    "Return exactly one JSON object and no markdown. Use only the enum values "
    "provided in output_schema. Base the decision only on the supplied ticket, "
    "PromptForge-selected context, and evidence. Never invent missing facts. "
    "Set contradiction_detected=true when material evidence conflicts. "
    "Set requires_human=true when the case cannot be safely resolved without "
    "human review. When safe_action is not justified by the supplied evidence, "
    "use action=human_review."
)


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    endpoint_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 60.0
    json_mode: bool = False
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def from_env(cls) -> "OpenAICompatibleConfig":
        endpoint_url = os.getenv("PROMPTFORGE_MODEL_URL", "").strip()
        model = os.getenv("PROMPTFORGE_MODEL_NAME", "").strip()
        if not endpoint_url:
            raise ValueError("PROMPTFORGE_MODEL_URL is required")
        if not model:
            raise ValueError("PROMPTFORGE_MODEL_NAME is required")

        timeout_text = os.getenv("PROMPTFORGE_MODEL_TIMEOUT", "60").strip()
        try:
            timeout_seconds = float(timeout_text)
        except ValueError as exc:
            raise ValueError("PROMPTFORGE_MODEL_TIMEOUT must be numeric") from exc
        if timeout_seconds <= 0:
            raise ValueError("PROMPTFORGE_MODEL_TIMEOUT must be positive")

        json_mode_text = os.getenv("PROMPTFORGE_MODEL_JSON_MODE", "0").strip().lower()
        return cls(
            endpoint_url=endpoint_url,
            model=model,
            api_key=os.getenv("PROMPTFORGE_MODEL_API_KEY") or None,
            timeout_seconds=timeout_seconds,
            json_mode=json_mode_text in {"1", "true", "yes", "on"},
            system_prompt=os.getenv(
                "PROMPTFORGE_MODEL_SYSTEM_PROMPT",
                DEFAULT_SYSTEM_PROMPT,
            ),
        )


class OpenAICompatibleAgentAdapter:
    """Call a Chat Completions-compatible endpoint without an SDK dependency."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        if not config.endpoint_url.strip():
            raise ValueError("endpoint_url must not be empty")
        if not config.model.strip():
            raise ValueError("model must not be empty")
        self.config = config

    def predict(self, model_input: Mapping[str, Any]) -> Mapping[str, Any]:
        request_payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        model_input,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
        }
        if self.config.json_mode:
            request_payload["response_format"] = {"type": "json_object"}

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        request = urllib.request.Request(
            self.config.endpoint_url,
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"model endpoint returned HTTP {exc.code}: {details[:1000]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"model endpoint request failed: {exc}") from exc

        return _extract_prediction(body)


def _extract_prediction(response_body: str) -> Mapping[str, Any]:
    """Extract a prediction from direct JSON or common chat-completion JSON."""
    try:
        response = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise ValueError("model response is not valid JSON") from exc

    if isinstance(response, Mapping) and all(
        key in response
        for key in (
            "ticket_id",
            "category",
            "sla",
            "priority",
            "action",
            "requires_human",
            "contradiction_detected",
        )
    ):
        return response

    if not isinstance(response, Mapping):
        raise ValueError("model response must be a JSON object")

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("model response does not contain choices")

    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("model response choice must be an object")

    message = first.get("message")
    content: Any = message.get("content") if isinstance(message, Mapping) else None
    if content is None:
        content = first.get("text")

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        content = "".join(text_parts)

    if not isinstance(content, str) or not content.strip():
        raise ValueError("model response does not contain textual JSON content")

    return _parse_json_object(content)


def _parse_json_object(content: str) -> Mapping[str, Any]:
    fence = chr(96) * 3
    text = content.strip()
    if text.startswith(fence):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith(fence):
            lines = lines[1:]
        if lines and lines[-1].strip() == fence:
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model content does not contain a JSON object")
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("model content does not contain valid JSON") from exc

    if not isinstance(payload, Mapping):
        raise ValueError("model content JSON must be an object")
    return payload


def write_prediction_jsonl(report: ModelLoopReport, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in report.records:
            if record.raw_prediction is None:
                continue
            handle.write(json.dumps(record.raw_prediction.to_dict(), sort_keys=True))
            handle.write("\n")


def write_report(report: ModelLoopReport, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PromptForge's ticket benchmark against an OpenAI-compatible model."
    )
    parser.add_argument("--output", default="predictions.jsonl")
    parser.add_argument("--report", default="model_report.json")
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--budget-tokens", type=int, default=500)
    parser.add_argument("--reserve-tokens", type=int, default=50)
    parser.add_argument("--no-guard", action="store_true")
    args = parser.parse_args()

    config = OpenAICompatibleConfig.from_env()
    adapter = OpenAICompatibleAgentAdapter(config)
    cases = generate_ticket_suite(count=args.count, seed=args.seed)
    report = run_model_loop(
        cases,
        adapter,
        budget_tokens=args.budget_tokens,
        reserve_tokens=args.reserve_tokens,
        apply_guard=not args.no_guard,
    )
    write_prediction_jsonl(report, args.output)
    write_report(report, args.report)
    print(json.dumps(report.to_dict()["summary"], indent=2, sort_keys=True))
    print(json.dumps({"raw": report.raw_metrics.to_dict(), "guarded": report.guarded_metrics.to_dict()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
