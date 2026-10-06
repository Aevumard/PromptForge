from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


EXECUTION_STATUSES = (
    "started",
    "succeeded",
    "failed",
    "skipped",
)

EXECUTION_DECISIONS = (
    "execute",
    "retry",
    "duplicate",
    "blocked",
)


@dataclass(frozen=True)
class ActionExecutionRecord:
    """Immutable receipt for an external action attempt.

    PromptForge records execution state but does not perform the side effect.
    The idempotency key is an integration-owned duplicate-action boundary.
    """

    schema_version: str
    execution_id: str
    action_id: str
    idempotency_key: str
    status: str
    attempt: int
    recorded_at: str
    external_reference: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("schema_version", self.schema_version),
            ("execution_id", self.execution_id),
            ("action_id", self.action_id),
            ("idempotency_key", self.idempotency_key),
            ("recorded_at", self.recorded_at),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.schema_version != "action-execution.v1":
            raise ValueError("schema_version must be action-execution.v1")
        if self.status not in EXECUTION_STATUSES:
            raise ValueError(f"unknown execution status: {self.status}")
        if (
            not isinstance(self.attempt, int)
            or isinstance(self.attempt, bool)
            or self.attempt < 1
        ):
            raise ValueError("attempt must be a positive integer")
        if self.external_reference is not None and not self.external_reference.strip():
            raise ValueError("external_reference must be non-empty when supplied")
        if self.error_code is not None and not self.error_code.strip():
            raise ValueError("error_code must be non-empty when supplied")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionPolicy:
    """Deterministic duplicate/retry policy for caller-supplied receipts."""

    require_idempotency_key: bool = True
    max_attempts: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.require_idempotency_key, bool):
            raise TypeError("require_idempotency_key must be a bool")
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be at least 1")


@dataclass(frozen=True)
class ExecutionDecision:
    schema_version: str
    decision: str
    action_id: str
    idempotency_key: str
    next_attempt: int | None
    prior_execution_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "execution-decision.v1":
            raise ValueError("schema_version must be execution-decision.v1")
        if self.decision not in EXECUTION_DECISIONS:
            raise ValueError(f"unknown execution decision: {self.decision}")
        if not self.action_id.strip():
            raise ValueError("action_id must not be empty")
        if not self.idempotency_key.strip():
            raise ValueError("idempotency_key must not be empty")
        if self.next_attempt is not None and self.next_attempt < 1:
            raise ValueError("next_attempt must be positive when supplied")
        if not self.reasons:
            raise ValueError("reasons must contain at least one item")
        if any(not item.strip() for item in self.reasons):
            raise ValueError("reasons must contain non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["prior_execution_ids"] = list(self.prior_execution_ids)
        payload["reasons"] = list(self.reasons)
        return payload


class ActionExecutionGuard:
    """Check an external action attempt before a side effect occurs."""

    def __init__(self, policy: ExecutionPolicy | None = None) -> None:
        self.policy = policy or ExecutionPolicy()

    def assess(
        self,
        action_id: str,
        idempotency_key: str,
        prior_records: Sequence[ActionExecutionRecord] = (),
    ) -> ExecutionDecision:
        if not action_id.strip():
            raise ValueError("action_id must not be empty")
        if not idempotency_key.strip():
            if self.policy.require_idempotency_key:
                return ExecutionDecision(
                    schema_version="execution-decision.v1",
                    decision="blocked",
                    action_id=action_id,
                    idempotency_key="missing",
                    next_attempt=None,
                    prior_execution_ids=(),
                    reasons=("idempotency key is required",),
                )
            idempotency_key = "unkeyed:" + action_id

        records = tuple(prior_records)
        if any(not isinstance(item, ActionExecutionRecord) for item in records):
            raise TypeError(
                "prior_records must contain only ActionExecutionRecord values"
            )

        relevant = tuple(
            item for item in records
            if item.action_id == action_id
            and item.idempotency_key == idempotency_key
        )
        ids = tuple(item.execution_id for item in relevant)

        if any(item.status == "succeeded" for item in relevant):
            return ExecutionDecision(
                schema_version="execution-decision.v1",
                decision="duplicate",
                action_id=action_id,
                idempotency_key=idempotency_key,
                next_attempt=None,
                prior_execution_ids=ids,
                reasons=("successful execution already recorded",),
            )

        if any(item.status == "started" for item in relevant):
            return ExecutionDecision(
                schema_version="execution-decision.v1",
                decision="duplicate",
                action_id=action_id,
                idempotency_key=idempotency_key,
                next_attempt=None,
                prior_execution_ids=ids,
                reasons=("execution with the same idempotency key is in progress",),
            )

        attempt = len(relevant) + 1
        if attempt > self.policy.max_attempts:
            return ExecutionDecision(
                schema_version="execution-decision.v1",
                decision="blocked",
                action_id=action_id,
                idempotency_key=idempotency_key,
                next_attempt=None,
                prior_execution_ids=ids,
                reasons=("execution retry budget exhausted",),
            )

        if relevant:
            return ExecutionDecision(
                schema_version="execution-decision.v1",
                decision="retry",
                action_id=action_id,
                idempotency_key=idempotency_key,
                next_attempt=attempt,
                prior_execution_ids=ids,
                reasons=("previous execution attempts failed",),
            )

        return ExecutionDecision(
            schema_version="execution-decision.v1",
            decision="execute",
            action_id=action_id,
            idempotency_key=idempotency_key,
            next_attempt=1,
            prior_execution_ids=(),
            reasons=("no prior execution recorded for idempotency key",),
        )


__all__ = [
    "EXECUTION_STATUSES",
    "EXECUTION_DECISIONS",
    "ActionExecutionRecord",
    "ExecutionPolicy",
    "ExecutionDecision",
    "ActionExecutionGuard",
]
