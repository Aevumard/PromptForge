from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from benchmarks.model_loop_v27 import (
    AgentAdapter,
    ModelLoopRecord,
    ModelLoopReport,
    build_model_input,
    parse_prediction,
)
from benchmarks.ticket_guard import guard_predictions
from benchmarks.tickets_v27 import (
    BenchmarkMetrics,
    Prediction,
    TicketCase,
    evaluate_predictions,
)

CHECKPOINT_SCHEMA = "promptforge-v27.6-checkpoint.v1"
REPORT_SCHEMA = "promptforge-v27.6-resumable-run.v1"


@dataclass(frozen=True)
class CheckpointStore:
    path: Path
    fsync_each_record: bool = True

    def load(self) -> dict[str, ModelLoopRecord]:
        records: dict[str, ModelLoopRecord] = {}
        if not self.path.exists():
            return records
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid checkpoint JSON at line {line_number}"
                    ) from exc
                if payload.get("schema_version") != CHECKPOINT_SCHEMA:
                    raise ValueError(
                        f"unsupported checkpoint schema at line {line_number}"
                    )
                record = _record_from_dict(payload)
                records[record.ticket_id] = record
        return records

    def append(self, record: ModelLoopRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA,
            **record.to_dict(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
            handle.flush()
            if self.fsync_each_record:
                os.fsync(handle.fileno())


def _record_from_dict(payload: Mapping[str, Any]) -> ModelLoopRecord:
    ticket_id = str(payload.get("ticket_id", "")).strip()
    if not ticket_id:
        raise ValueError("checkpoint record is missing ticket_id")

    raw_payload = payload.get("raw_prediction")
    guarded_payload = payload.get("guarded_prediction")

    raw_prediction = (
        _prediction_from_dict(raw_payload, ticket_id=ticket_id)
        if isinstance(raw_payload, Mapping)
        else None
    )
    guarded_prediction = (
        _prediction_from_dict(guarded_payload, ticket_id=ticket_id)
        if isinstance(guarded_payload, Mapping)
        else None
    )

    omitted = payload.get("omitted_context_ids", [])
    if not isinstance(omitted, list) or not all(
        isinstance(item, str) for item in omitted
    ):
        raise ValueError(f"invalid omitted_context_ids for {ticket_id}")

    return ModelLoopRecord(
        ticket_id=ticket_id,
        raw_prediction=raw_prediction,
        guarded_prediction=guarded_prediction,
        context_tokens=int(payload.get("context_tokens", 0)),
        budget_tokens=int(payload.get("budget_tokens", 0)),
        tokens_saved=int(payload.get("tokens_saved", 0)),
        omitted_context_ids=tuple(omitted),
        elapsed_ms=float(payload.get("elapsed_ms", 0.0)),
        error=None if payload.get("error") is None else str(payload["error"]),
    )


def _prediction_from_dict(
    payload: Mapping[str, Any],
    *,
    ticket_id: str,
) -> Prediction:
    return parse_prediction(payload, ticket_id=ticket_id)


def run_resumable_model_loop(
    cases: Sequence[TicketCase],
    adapter: AgentAdapter,
    *,
    checkpoint_path: str | Path,
    budget_tokens: int = 500,
    reserve_tokens: int = 50,
    apply_guard: bool = True,
    include_epistemic: bool = True,
    retry_failed: bool = True,
    fsync_each_record: bool = True,
) -> ModelLoopReport:
    """Run a model benchmark while checkpointing every ticket.

    Successful tickets are never called again on resume. Failed tickets can be
    retried on a later invocation without discarding successful work.
    """
    store = CheckpointStore(
        Path(checkpoint_path),
        fsync_each_record=fsync_each_record,
    )
    saved = store.load()
    records_by_id: dict[str, ModelLoopRecord] = dict(saved)

    for case in cases:
        existing = records_by_id.get(case.ticket_id)
        if existing is not None:
            if existing.raw_prediction is not None:
                continue
            if not retry_failed:
                continue

        started = time.perf_counter()
        try:
            model_input = build_model_input(
                case,
                budget_tokens=budget_tokens,
                reserve_tokens=reserve_tokens,
                include_epistemic=include_epistemic,
            )
            packet = model_input["promptforge"]
            payload = adapter.predict(model_input)
            prediction = parse_prediction(payload, ticket_id=case.ticket_id)
            guarded = (
                guard_predictions(cases, (prediction,))[0]
                if apply_guard
                else prediction
            )
            record = ModelLoopRecord(
                ticket_id=case.ticket_id,
                raw_prediction=prediction,
                guarded_prediction=guarded,
                context_tokens=int(packet["context_tokens"]),
                budget_tokens=int(packet["budget_tokens"]),
                tokens_saved=int(packet["tokens_saved"]),
                omitted_context_ids=tuple(packet["omitted_context_ids"]),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:
            record = ModelLoopRecord(
                ticket_id=case.ticket_id,
                raw_prediction=None,
                guarded_prediction=None,
                context_tokens=0,
                budget_tokens=budget_tokens,
                tokens_saved=0,
                omitted_context_ids=(),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )

        store.append(record)
        records_by_id[case.ticket_id] = record

    ordered_records = tuple(
        records_by_id[case.ticket_id]
        for case in cases
        if case.ticket_id in records_by_id
    )
    raw_predictions = [
        record.raw_prediction
        for record in ordered_records
        if record.raw_prediction is not None
    ]
    guarded_predictions = [
        record.guarded_prediction
        for record in ordered_records
        if record.guarded_prediction is not None
    ]

    raw_metrics: BenchmarkMetrics = evaluate_predictions(cases, raw_predictions)
    guarded_metrics: BenchmarkMetrics = evaluate_predictions(
        cases,
        guarded_predictions,
    )

    return ModelLoopReport(
        schema_version=REPORT_SCHEMA,
        raw_metrics=raw_metrics,
        guarded_metrics=guarded_metrics,
        records=ordered_records,
    )


def checkpoint_summary(path: str | Path) -> dict[str, int]:
    records = CheckpointStore(
        Path(path),
        fsync_each_record=False,
    ).load()
    return {
        "records": len(records),
        "successful": sum(
            record.raw_prediction is not None for record in records.values()
        ),
        "failed": sum(
            record.raw_prediction is None for record in records.values()
        ),
    }
