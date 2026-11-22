"""A recording stand-in for the app's Langfuse tracing (app/observability.py), shared by the tests.

RecordingTracing keeps every trace and span the app opens, with its parent, so a test can check the SHAPE of a
trace (the chat turn holds the guard, the guard holds two generations...) without a Langfuse server.
Its callbacks() hand LangGraph a handler that records run names, standing in for Langfuse's CallbackHandler.
Use it through the `tracing` fixture in conftest.py, which puts the real (off) tracing back after the test.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from app.observability import Tracing


class RecordedObservation:
    """One trace or span: its kind ("trace", "span", "guardrail", "generation"...), name, arguments and updates."""

    def __init__(self, kind: str, name: str, attrs: dict[str, Any], parent: "RecordedObservation | None") -> None:
        """Keep what the app opened it with; updates start empty."""
        self.kind = kind
        self.name = name
        self.attrs = attrs
        self.parent = parent
        self.updates: dict[str, Any] = {}

    def update(self, **kwargs: Any) -> None:
        """Record the update (later keys win, like Langfuse)."""
        self.updates.update(kwargs)


class RunNames(BaseCallbackHandler):
    """Records the name of every LangChain/LangGraph run that starts (a graph, a node, a model call...)."""

    run_inline = True  # called in order, on the event loop

    def __init__(self) -> None:
        """No runs yet."""
        self.names: list[str] = []

    def on_chain_start(self, serialized: Any, inputs: Any, **kwargs: Any) -> None:
        """A graph or node started: LangChain passes its run name as `name`."""
        self.names.append(kwargs.get("name") or "")


_current: ContextVar[RecordedObservation | None] = ContextVar("recorded_current", default=None)
FAKE_TRACEPARENT = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"


class RecordingTracing(Tracing):
    """Tracing that is ON but sends nothing: it records instead."""

    def __init__(self) -> None:
        """Nothing recorded yet."""
        super().__init__()
        self.observations: list[RecordedObservation] = []
        self.handler = RunNames()

    @property
    def enabled(self) -> bool:
        """Always on."""
        return True

    @contextmanager
    def _open(self, kind: str, name: str, attrs: dict[str, Any]) -> Iterator[RecordedObservation]:
        """Record a new observation under the current one and make it current while the block runs."""
        observation = RecordedObservation(kind, name, attrs, _current.get())
        self.observations.append(observation)
        token = _current.set(observation)
        try:
            yield observation
        finally:
            _current.reset(token)

    @contextmanager
    def trace(self, name: str, **attrs: Any) -> Iterator[RecordedObservation]:
        """Record a trace (session_id, input, metadata, tags in attrs)."""
        with self._open("trace", name, attrs) as observation:
            yield observation

    @contextmanager
    def span(self, name: str, *, as_type: str = "span", **attrs: Any) -> Iterator[RecordedObservation]:
        """Record a span of the given type (input, model in attrs)."""
        with self._open(as_type, name, attrs) as observation:
            yield observation

    def traceparent(self) -> str | None:
        """A fixed traceparent: lets tests check it is passed on (e.g. to the email service)."""
        return FAKE_TRACEPARENT

    def callbacks(self) -> list[Any]:
        """The run-name recorder, where the app would put Langfuse's handler."""
        return [self.handler]

    def named(self, name: str) -> list[RecordedObservation]:
        """Every recorded observation with this name, in order."""
        return [o for o in self.observations if o.name == name]
