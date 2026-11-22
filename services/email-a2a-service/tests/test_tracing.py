"""Tests for tracing in the email service (app/tracing.py): the orchestrator's trace continues here.

A recording stand-in replaces Langfuse; the service runs in memory with fakes (as in test_executor.py), and the
messages carry a traceparent in their metadata, exactly as the orchestrator's A2AEmailAgent sends it.
"""

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import pytest

from app.client_demo import send_data
from app.tracing import Tracing, parse_traceparent, set_tracing
from tests.test_executor import REQUEST, Service, start

TRACE_ID, SPAN_ID = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"
TRACEPARENT = f"00-{TRACE_ID}-{SPAN_ID}-01"


class Recorded:
    """One recorded span: name, type, arguments, parent, updates."""

    def __init__(self, name: str, as_type: str, attrs: dict[str, Any], parent: "Recorded | None") -> None:
        """Keep what the span was opened with."""
        self.name, self.as_type, self.attrs, self.parent = name, as_type, attrs, parent
        self.updates: dict[str, Any] = {}

    def update(self, **kwargs: Any) -> None:
        """Record the update."""
        self.updates.update(kwargs)


_current: ContextVar[Recorded | None] = ContextVar("email_recorded_current", default=None)


class RecordingTracing(Tracing):
    """Tracing that is on but records instead of sending."""

    def __init__(self) -> None:
        """Nothing recorded yet."""
        super().__init__()
        self.spans: list[Recorded] = []

    @property
    def enabled(self) -> bool:
        """Always on."""
        return True

    @contextmanager
    def span(self, name: str, *, as_type: str = "span", **attrs: Any) -> Iterator[Recorded]:
        """Record the span under the current one and make it current while the block runs."""
        span = Recorded(name, as_type, attrs, _current.get())
        self.spans.append(span)
        token = _current.set(span)
        try:
            yield span
        finally:
            _current.reset(token)

    def named(self, name: str) -> Recorded:
        """The (single) recorded span with this name."""
        [span] = [s for s in self.spans if s.name == name]
        return span


@pytest.fixture
def tracing() -> Iterator[RecordingTracing]:
    """Tracing on (recording) for one test; off again afterwards."""
    recording = RecordingTracing()
    set_tracing(recording)
    try:
        yield recording
    finally:
        set_tracing(Tracing())


@pytest.fixture
async def service() -> AsyncIterator[Service]:
    """The in-memory service with a working fake crew and mailer."""
    async for running in start():
        yield running


def test_a_traceparent_becomes_langfuses_trace_context() -> None:
    """A valid W3C traceparent -> {trace_id, parent_span_id}; anything else -> None (a new trace)."""
    assert parse_traceparent(TRACEPARENT) == {"trace_id": TRACE_ID, "parent_span_id": SPAN_ID}
    assert parse_traceparent(TRACEPARENT.upper()) == {"trace_id": TRACE_ID, "parent_span_id": SPAN_ID}
    for bad in (None, "", "garbage", f"00-{TRACE_ID}-short-01", 42):
        assert parse_traceparent(bad) is None


async def test_a_draft_continues_the_callers_trace(service: Service, tracing: RecordingTracing) -> None:
    """Message #1 with a traceparent -> "email service: draft" opened with that remote parent; the checks nest
    under it; its output says where the task was left."""
    task = await send_data(service.client, REQUEST, metadata={"traceparent": TRACEPARENT})
    root = tracing.named("email service: draft")
    assert root.as_type == "agent" and root.parent is None
    assert root.attrs["trace_context"] == {"trace_id": TRACE_ID, "parent_span_id": SPAN_ID}
    assert root.attrs["input"]["to"] == "accounts@company.example"
    assert root.updates["output"]["state"] == "INPUT_REQUIRED" and root.updates["output"]["task_id"] == task.id
    for name in ("request checks", "draft checks"):
        check = tracing.named(name)
        assert check.parent is root and check.as_type == "guardrail" and check.updates["output"]["passed"]


async def test_a_decision_traces_the_checks_and_the_send(service: Service, tracing: RecordingTracing) -> None:
    """Message #2 -> "email service: decision" with "body checks" and "smtp send" (the receipt as output)."""
    task = await send_data(service.client, REQUEST)
    await send_data(service.client, {"decision": "approve"}, task.id, task.context_id, {"traceparent": TRACEPARENT})
    root = tracing.named("email service: decision")
    assert root.attrs["trace_context"]["trace_id"] == TRACE_ID and root.updates["output"]["state"] == "COMPLETED"
    smtp = tracing.named("smtp send")
    assert smtp.parent is root and smtp.as_type == "tool" and smtp.updates["output"]["to"] == "accounts@company.example"
    assert tracing.named("body checks").parent is root


async def test_a_refused_request_is_a_warning_on_its_check(service: Service, tracing: RecordingTracing) -> None:
    """A recipient off the allow-list: the "request checks" span says why; the task state is REJECTED."""
    await send_data(service.client, {**REQUEST, "to": "ceo@gmail.com"})
    checks = tracing.named("request checks")
    assert checks.updates["level"] == "WARNING" and "not allowed" in checks.updates["output"]["reason"]
    assert tracing.named("email service: draft").updates["output"]["state"] == "REJECTED"
