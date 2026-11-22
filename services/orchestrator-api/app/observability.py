"""Tracing with Langfuse: see every step of every answer (project plan Phase 2 and section 3-G).

One chat turn becomes ONE trace in Langfuse (self-hosted, docker-compose.langfuse.yml):

    chat turn                          question in, answer out; session = the chat
      ├─ input guard                   the guardrail's verdict
      │    ├─ prompt guard             a generation: the Groq model, its tokens
      │    └─ safety model             a generation: the Groq model, its tokens
      └─ orchestrator                  the LangGraph run: every node, every LLM call with its tokens
           └─ tools → the DB / Doc / Web agents' own graphs, the MCP search tool
      └─ email service: draft          the A2A call; the email service's own spans (checks, crew, SMTP) continue
                                       this trace there: traceparent() travels in the A2A message

How it fits together: the Langfuse SDK is built on OpenTelemetry. trace() and span() open an observation and
make it the CURRENT one (a contextvar, like the guard's trace), so everything started inside nests under it.
LangChain/LangGraph runs are recorded by the handler from callbacks(), passed in the orchestrator's run config;
LangChain hands it down to runnables started inside a node, so the agents' graphs nest under their tool call.

OFF unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set: then every helper here is a no-op and nothing is
sent anywhere (tests, CI, a machine without Langfuse). The SDK sends in a background thread, so tracing adds no
waiting to an answer; flush() at shutdown sends what's left.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler

from app.config import Settings

logger = logging.getLogger(__name__)


class NoopObservation:
    """What trace() and span() yield when tracing is off: accepts update() and ignores it."""

    def update(self, **kwargs: Any) -> None:
        """Ignore the update."""


NOOP = NoopObservation()


class Tracing:
    """A thin wrapper around the Langfuse client: the few things the app traces, all no-ops when tracing is off."""

    def __init__(self, client: Any = None, public_key: str = "") -> None:
        """Keep the client (None = tracing off) and its public key (the LangChain handler finds the client by it)."""
        self.client = client
        self.public_key = public_key

    @classmethod
    def from_settings(cls, settings: Settings) -> "Tracing":
        """Connect to Langfuse when both keys are set; otherwise tracing is off."""
        secret = settings.langfuse_secret_key.get_secret_value()
        if not (settings.langfuse_public_key and secret):
            return cls()
        client = Langfuse(
            public_key=settings.langfuse_public_key, secret_key=secret, base_url=settings.langfuse_base_url
        )
        return cls(client, settings.langfuse_public_key)

    @property
    def enabled(self) -> bool:
        """True when traces are sent to Langfuse."""
        return self.client is not None

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        session_id: str | None = None,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Iterator[Any]:
        """Start a NEW trace (one chat turn, one email decision...) and make it current; yields its root observation.

        propagate_attributes is entered FIRST, so the trace name, session and tags apply to every observation
        created inside, the root one included (the Langfuse docs warn that attributes set later miss earlier spans).
        session_id groups the traces of one chat in Langfuse's Sessions view.
        """
        if not self.enabled:
            yield NOOP
            return
        attributes = {"trace_name": name, "session_id": session_id, "tags": tags, "metadata": metadata}
        with propagate_attributes(**{k: v for k, v in attributes.items() if v}):
            with self.client.start_as_current_observation(name=name, as_type="agent", input=input) as root:
                yield root

    @contextmanager
    def span(self, name: str, *, as_type: str = "span", input: Any = None, model: str | None = None) -> Iterator[Any]:
        """A child observation of whatever is current (a guardrail, a generation...); yields it for update().

        For as_type="generation", pass the model; update(output=..., usage_details={"input": n, "output": m})
        records the tokens, so Langfuse can count usage per model.
        """
        if not self.enabled:
            yield NOOP
            return
        extra = {"model": model} if model else {}
        with self.client.start_as_current_observation(name=name, as_type=as_type, input=input, **extra) as observation:
            yield observation

    def traceparent(self) -> str | None:
        """The current observation as a W3C traceparent ("00-<trace id>-<span id>-01"), or None (off / none current).

        Sent to the email service inside the A2A message (plan 05 + tracing), so its spans continue THIS trace.
        """
        if not self.enabled:
            return None
        trace_id, span_id = self.client.get_current_trace_id(), self.client.get_current_observation_id()
        return f"00-{trace_id}-{span_id}-01" if trace_id and span_id else None

    def callbacks(self) -> list[Any]:
        """The LangChain callback handler for a graph run's config (empty when tracing is off)."""
        return [CallbackHandler(public_key=self.public_key)] if self.enabled else []

    def flush(self) -> None:
        """Send everything still buffered (short-lived scripts and tests; the app sends in the background)."""
        if self.enabled:
            self.client.flush()

    def shutdown(self) -> None:
        """Flush and stop the background exporter (called once when the API stops)."""
        if self.enabled:
            self.client.shutdown()


_tracing = Tracing()  # off until setup_tracing() runs (the lifespan); tests keep it off


def setup_tracing(settings: Settings) -> Tracing:
    """Configure the app-wide tracing from the settings (called once in the lifespan) and return it."""
    global _tracing
    _tracing = Tracing.from_settings(settings)
    if _tracing.enabled:
        logger.info("Tracing to Langfuse at %s", settings.langfuse_base_url)
    else:
        logger.info("Tracing is off (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set)")
    return _tracing


def set_tracing(tracing: Tracing) -> None:
    """Replace the app-wide tracing (tests use a recording fake)."""
    global _tracing
    _tracing = tracing


def get_tracing() -> Tracing:
    """The app-wide tracing (a no-op one until setup_tracing() runs)."""
    return _tracing


def usage_of(response: Any) -> dict[str, int]:
    """Token usage of a raw Groq response as Langfuse's usage_details ({} when the response has none)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {"input": usage.prompt_tokens or 0, "output": usage.completion_tokens or 0}
