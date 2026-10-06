from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
import urllib.error
import urllib.request

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass(frozen=True)
class OllamaPreflight:
    base_url: str
    endpoint_url: str
    reachable: bool
    installed_models: tuple[str, ...]
    selected_model: str | None
    model_available: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "endpoint_url": self.endpoint_url,
            "reachable": self.reachable,
            "installed_models": list(self.installed_models),
            "selected_model": self.selected_model,
            "model_available": self.model_available,
            "error": self.error,
        }


def preflight_ollama(
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    *,
    model: str | None = None,
    timeout_seconds: float = 2.0,
) -> OllamaPreflight:
    base = base_url.strip().rstrip("/")
    if not base:
        raise ValueError("Ollama base URL must not be empty")
    endpoint = f"{base}/v1/chat/completions"

    request = urllib.request.Request(
        f"{base}/api/tags",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        return OllamaPreflight(
            base_url=base,
            endpoint_url=endpoint,
            reachable=False,
            installed_models=(),
            selected_model=None,
            model_available=False,
            error=f"Ollama was not reachable at {base}: {exc}",
        )
    except json.JSONDecodeError as exc:
        return OllamaPreflight(
            base_url=base,
            endpoint_url=endpoint,
            reachable=True,
            installed_models=(),
            selected_model=None,
            model_available=False,
            error="Ollama /api/tags returned invalid JSON.",
        )

    raw_models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(raw_models, list):
        return OllamaPreflight(
            base_url=base,
            endpoint_url=endpoint,
            reachable=True,
            installed_models=(),
            selected_model=None,
            model_available=False,
            error="Ollama /api/tags did not return a models list.",
        )

    names = tuple(
        sorted(
            {
                str(item.get("name", "")).strip()
                for item in raw_models
                if isinstance(item, dict) and str(item.get("name", "")).strip()
            }
        )
    )
    selected = model.strip() if model and model.strip() else (names[0] if names else None)
    available = selected in names if selected else False
    error = None
    if not names:
        error = "Ollama is running but no models are installed."
    elif selected is not None and not available:
        error = f"Requested model {selected!r} is not installed."

    return OllamaPreflight(
        base_url=base,
        endpoint_url=endpoint,
        reachable=True,
        installed_models=names,
        selected_model=selected,
        model_available=available,
        error=error,
    )


def require_ready(preflight: OllamaPreflight) -> OllamaPreflight:
    if not preflight.reachable or not preflight.model_available:
        detail = preflight.error or "Ollama preflight failed."
        raise RuntimeError(detail)
    return preflight
