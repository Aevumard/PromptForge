from __future__ import annotations

import argparse
import json
import os

from benchmarks.preflight_v27 import (
    DEFAULT_OLLAMA_BASE_URL,
    preflight_ollama,
    require_ready,
)
from benchmarks.providers.openai_compatible import (
    OpenAICompatibleAgentAdapter,
    OpenAICompatibleConfig,
)
from benchmarks.replicates_v27 import (
    _provider_metadata,
    run_replicate_experiment,
    write_experiment,
)
from benchmarks.tickets_v27 import generate_ticket_suite


def _progress(event: str, payload: dict) -> None:
    if event == "preflight":
        print(
            f"[PromptForge] Ollama OK | model={payload['model']} "
            f"| endpoint={payload['endpoint']}"
        )
    elif event == "replicate_start":
        print(
            f"[PromptForge] replicate {payload['index']}/{payload['total']} "
            f"START | {payload['run_id']}"
        )
    elif event == "replicate_complete":
        print(
            f"[PromptForge] replicate {payload['index']}/{payload['total']} "
            f"DONE | coverage={payload['coverage']:.2%} "
            f"| guarded_action_accuracy={payload['guarded_action_accuracy']:.2%} "
            f"| unsafe={payload['guarded_unsafe_action_rate']:.2%}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the PromptForge benchmark locally through Ollama."
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--model")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--budget-tokens", type=int, default=500)
    parser.add_argument("--reserve-tokens", type=int, default=50)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--output-dir", default="replicate_runs")
    parser.add_argument("--output-json", default="replicate_analysis.json")
    parser.add_argument("--output-markdown", default="replicate_analysis.md")
    parser.add_argument("--no-epistemic", action="store_true")
    parser.add_argument("--no-guard", action="store_true")
    parser.add_argument("--no-retry-failed", action="store_true")
    parser.add_argument("--no-fsync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    preflight = preflight_ollama(
        args.ollama_url,
        model=args.model,
    )
    if args.json_output:
        print(json.dumps(preflight.to_dict(), indent=2, sort_keys=True))
    else:
        print(
            f"[PromptForge] preflight | reachable={preflight.reachable} "
            f"| models={len(preflight.installed_models)} "
            f"| selected={preflight.selected_model or '-'}"
        )
        if preflight.error:
            print(f"[PromptForge] ERROR: {preflight.error}")

    require_ready(preflight)
    if args.dry_run:
        return 0

    os.environ["PROMPTFORGE_MODEL_URL"] = preflight.endpoint_url
    os.environ["PROMPTFORGE_MODEL_NAME"] = preflight.selected_model or ""
    os.environ["PROMPTFORGE_MODEL_API_KEY"] = "ollama"

    config = OpenAICompatibleConfig.from_env()
    if not args.json_output:
        _progress(
            "preflight",
            {
                "model": config.model,
                "endpoint": config.endpoint_url,
            },
        )

    cases = generate_ticket_suite(count=args.count, seed=args.seed)

    def adapter_factory() -> OpenAICompatibleAgentAdapter:
        return OpenAICompatibleAgentAdapter(config)

    experiment = run_replicate_experiment(
        cases,
        adapter_factory,
        replicates=args.replicates,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_confidence=args.bootstrap_confidence,
        output_dir=args.output_dir,
        budget_tokens=args.budget_tokens,
        reserve_tokens=args.reserve_tokens,
        include_epistemic=not args.no_epistemic,
        apply_guard=not args.no_guard,
        retry_failed=not args.no_retry_failed,
        fsync_each_record=not args.no_fsync,
        provider_metadata=_provider_metadata(config),
        progress_callback=lambda event, payload: (
            None if args.json_output else _progress(event, payload)
        ),
    )
    write_experiment(
        experiment,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )

    print(
        json.dumps(
            {
                "model": experiment.provider.get("model"),
                "replicates": experiment.replicate_count,
                "suite": experiment.suite,
                "pairwise_disagreement": experiment.pairwise_disagreement.to_dict(),
                "output_json": args.output_json,
                "output_markdown": args.output_markdown,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
