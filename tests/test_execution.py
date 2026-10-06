import unittest

from promptforge.execution import (
    ActionExecutionGuard,
    ActionExecutionRecord,
    ExecutionPolicy,
)


def record(status: str, attempt: int = 1) -> ActionExecutionRecord:
    return ActionExecutionRecord(
        schema_version="action-execution.v1",
        execution_id=f"X-{attempt}",
        action_id="act-1",
        idempotency_key="case:T-1:act-1",
        status=status,
        attempt=attempt,
        recorded_at="2026-10-06T09:10:00Z",
        external_reference="provider-17" if status == "succeeded" else None,
        error_code="TIMEOUT" if status == "failed" else None,
    )


class ExecutionTests(unittest.TestCase):
    def test_missing_idempotency_key_fails_closed(self) -> None:
        result = ActionExecutionGuard().assess("act-1", "")
        self.assertEqual(result.decision, "blocked")

    def test_successful_key_is_duplicate(self) -> None:
        result = ActionExecutionGuard().assess(
            "act-1",
            "case:T-1:act-1",
            prior_records=(record("succeeded"),),
        )
        self.assertEqual(result.decision, "duplicate")
        self.assertEqual(result.next_attempt, None)

    def test_failed_attempt_can_retry_when_budget_allows(self) -> None:
        result = ActionExecutionGuard(
            ExecutionPolicy(max_attempts=2)
        ).assess(
            "act-1",
            "case:T-1:act-1",
            prior_records=(record("failed"),),
        )
        self.assertEqual(result.decision, "retry")
        self.assertEqual(result.next_attempt, 2)

    def test_in_flight_key_is_duplicate(self) -> None:
        result = ActionExecutionGuard().assess(
            "act-1",
            "case:T-1:act-1",
            prior_records=(record("started"),),
        )
        self.assertEqual(result.decision, "duplicate")


if __name__ == "__main__":
    unittest.main()
