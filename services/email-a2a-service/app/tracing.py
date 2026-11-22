"""Langfuse tracing in the email service: its spans join the ORCHESTRATOR'S trace (project plan 3-G).

One chat turn that drafts an email crosses two processes. Without this, Langfuse would show two unrelated traces.
With it, the orchestrator's trace continues here:

    chat turn                                (orchestrator)
      └─ email service: draft                (orchestrator: the A2A call)  --traceparent in the A2A message-->
           └─ email service: draft           (THIS service, same trace)
                ├─ request checks            guardrail
                ├─ email crew                generation: qwen, tokens, the writer's draft and the reviewer's final
                └─ draft checks              guardrail

How: the orchestrator puts a W3C `traceparent` ("00-<trace id>-<span id>-01") in the A2A message's metadata;
parse_traceparent() turns it into Langfuse's TraceContext, and the executor opens its first span with that as the
REMOTE parent. Metadata (not an HTTP header) because the A2A SDK may run the executor in another asyncio task: the
message travels with the work, a header's context might not.

OFF unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (the same project keys as the orchestrator's): then
every helper is a no-op. Spans are sent in the background; shutdown() sends what's left.
"""

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langfuse import Langfuse
from langfuse.types import TraceContext

from app.config import Settings

logger = logging.getLogger(__name__)

TRACEPARENT = re.compile(r"00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}")


class NoopObservation:
    """What span() yields when tracing is off: accepts update() and ignores it."""

    def update(self, **kwargs: Any) -> None:
        """Ignore the update."""


NOOP = NoopObservation()


class Tracing:
    """A thin wrapper around the Langfuse client (None = tracing off)."""

    def __init__(self, client: Any = None) -> None:
        """Keep the client."""
        self.client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "Tracing":
        """Connect when both keys are set; otherwise tracing is off."""
        secret = settings.langfuse_secret_key.get_secret_value()
        if not (settings.langfuse_public_key and secret):
            return cls()
        return cls(Langfuse(public_key=settings.langfuse_public_key, secret_key=secret,
                            base_url=settings.langfuse_base_url))  # fmt: skip

    @property
    def enabled(self) -> bool:
        """True when spans are sent to Langfuse."""
        return self.client is not None

    @contextmanager
    def span(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any = None,
        model: str | None = None,
        trace_context: TraceContext | None = None,
    ) -> Iterator[Any]:
        """An observation under the current one, or under `trace_context` (a parent in ANOTHER process).

        Without a current observation and without trace_context, it starts a new trace of its own (e.g. the demo
        client calling the service directly).
        """
        if not self.enabled:
            yield NOOP
            return
        extra: dict[str, Any] = {"model": model} if model else {}
        if trace_context:
            extra["trace_context"] = trace_context
        with self.client.start_as_current_observation(name=name, as_type=as_type, input=input, **extra) as observation:
            yield observation

    def shutdown(self) -> None:
        """Send everything still buffered and stop the exporter (service shutdown)."""
        if self.enabled:
            self.client.shutdown()


def parse_traceparent(value: Any) -> TraceContext | None:
    """A W3C traceparent ("00-<32 hex trace id>-<16 hex span id>-<flags>") as Langfuse's TraceContext, or None."""
    match = TRACEPARENT.fullmatch(value.strip().lower()) if isinstance(value, str) else None
    if match is None:
        return None
    return TraceContext(trace_id=match.group(1), parent_span_id=match.group(2))


_tracing = Tracing()  # off until setup_tracing() runs at startup; tests keep it off


def setup_tracing(settings: Settings) -> Tracing:
    """Configure the service-wide tracing from the settings (startup) and return it."""
    global _tracing
    _tracing = Tracing.from_settings(settings)
    logger.info("Tracing %s", f"to Langfuse at {settings.langfuse_base_url}" if _tracing.enabled else "is off")
    return _tracing


def set_tracing(tracing: Tracing) -> None:
    """Replace the service-wide tracing (tests use a recording fake)."""
    global _tracing
    _tracing = tracing


def get_tracing() -> Tracing:
    """The service-wide tracing (a no-op one until setup_tracing() runs)."""
    return _tracing
