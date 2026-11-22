"""Tests for tracing (app/observability.py, project plan Phase 2). No Langfuse server, no Groq, no database.

- OFF (a key missing): every helper is a no-op, so tests and machines without Langfuse send nothing.
- ON with a fake Langfuse client: what trace() and span() ask the SDK for, and in which order.
- The input guard's shape: one guardrail span holding a generation per Groq call (RecordingTracing).
The chat turn and /api/db/ask traces are tested with the chat API (tests/test_chat_api.py).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from app import observability
from app.config import get_settings
from app.guardrails.guard import InputGuard
from app.observability import NOOP, Tracing, usage_of
from tests.test_guardrails import JAILBREAK_MODEL, SAFETY_MODEL, actions
from tests.tracing_fakes import RecordingTracing


def settings_with(public: str, secret: str):  # noqa: ANN201
    """The app's settings with these Langfuse keys (whatever .env says)."""
    return get_settings().model_copy(update={"langfuse_public_key": public, "langfuse_secret_key": SecretStr(secret)})


# --------------------------------------------------------------------------- off


@pytest.mark.parametrize(("public", "secret"), [("", ""), ("pk-lf-1", ""), ("", "sk-lf-1")])
async def test_tracing_is_off_unless_both_keys_are_set(public: str, secret: str) -> None:
    """A missing key -> off: trace() and span() yield the no-op, there are no callbacks, flush does nothing."""
    tracing = Tracing.from_settings(settings_with(public, secret))
    assert not tracing.enabled
    with tracing.trace("chat turn", session_id="s1", input="hi") as root:
        with tracing.span("input guard", as_type="guardrail") as span:
            span.update(output="anything")  # accepted and ignored
    assert root is NOOP and span is NOOP
    assert tracing.callbacks() == []
    tracing.flush()
    tracing.shutdown()


def test_both_keys_connect_to_the_configured_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both keys -> a Langfuse client with them and LANGFUSE_BASE_URL."""
    created: list[dict] = []
    monkeypatch.setattr(observability, "Langfuse", lambda **kwargs: created.append(kwargs) or "client")
    tracing = Tracing.from_settings(settings_with("pk-lf-1", "sk-lf-1"))
    assert tracing.enabled and tracing.client == "client" and tracing.public_key == "pk-lf-1"
    assert created == [{"public_key": "pk-lf-1", "secret_key": "sk-lf-1", "base_url": get_settings().langfuse_base_url}]


# --------------------------------------------------------------------------- on, with a fake SDK


class FakeLangfuse:
    """Records the observations the wrapper asks for; `events` is shared with the fake propagate_attributes."""

    def __init__(self, events: list) -> None:
        """Share the event list."""
        self.events = events

    @contextmanager
    def start_as_current_observation(self, **kwargs: Any) -> Iterator[SimpleNamespace]:
        """Record the call and yield a stand-in observation."""
        self.events.append(("observation", kwargs))
        yield SimpleNamespace(kwargs=kwargs)


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list:
    """Every SDK call in order: propagate_attributes(...) and start_as_current_observation(...)."""
    recorded: list = []

    @contextmanager
    def propagate_attributes(**kwargs: Any) -> Iterator[None]:
        """Record the trace attributes."""
        recorded.append(("attributes", kwargs))
        yield

    monkeypatch.setattr(observability, "propagate_attributes", propagate_attributes)
    return recorded


def test_a_trace_sets_its_attributes_before_its_root_observation(events: list) -> None:
    """Name, session and metadata are propagated FIRST (so the root has them too); empty ones are left out."""
    tracing = Tracing(FakeLangfuse(events), "pk-lf-1")
    with tracing.trace("chat turn", session_id="s1", input="hi", metadata={"message_id": 7}) as root:
        pass
    assert events == [
        ("attributes", {"trace_name": "chat turn", "session_id": "s1", "metadata": {"message_id": 7}}),
        ("observation", {"name": "chat turn", "as_type": "agent", "input": "hi"}),
    ]
    assert root.kwargs["name"] == "chat turn"


def test_a_span_passes_the_model_only_when_given(events: list) -> None:
    """A generation carries its model (Langfuse counts tokens per model); a plain span has no model argument."""
    tracing = Tracing(FakeLangfuse(events), "pk-lf-1")
    with tracing.span("safety model", as_type="generation", input="hi", model="allam-2-7b"):
        pass
    with tracing.span("step"):
        pass
    assert events == [
        ("observation", {"name": "safety model", "as_type": "generation", "input": "hi", "model": "allam-2-7b"}),
        ("observation", {"name": "step", "as_type": "span", "input": None}),
    ]


def test_callbacks_are_one_langchain_handler_for_this_project(monkeypatch: pytest.MonkeyPatch) -> None:
    """The handler is created with the project's public key (it finds the client by it)."""
    monkeypatch.setattr(observability, "CallbackHandler", lambda public_key: ("handler", public_key))
    assert Tracing(object(), "pk-lf-1").callbacks() == [("handler", "pk-lf-1")]


def test_usage_of_reads_groq_token_counts() -> None:
    """prompt/completion tokens -> Langfuse's input/output; no usage -> {}."""
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=120, completion_tokens=9))
    assert usage_of(response) == {"input": 120, "output": 9}
    assert usage_of(SimpleNamespace()) == {}


# --------------------------------------------------------------------------- the guard's shape


async def test_the_guard_is_a_guardrail_holding_one_generation_per_model(tracing: RecordingTracing) -> None:
    """input guard (guardrail) -> prompt guard + safety model (generations, with their models and outputs)."""
    result = await InputGuard(actions()).check("Which vendors have overdue invoices?")
    assert result.status == "passed"
    [guard] = tracing.named("input guard")
    assert guard.kind == "guardrail" and guard.attrs["input"] == "Which vendors have overdue invoices?"
    assert guard.updates["output"]["status"] == "passed" and guard.updates["output"]["jailbreak_score"] == 0.0004
    [prompt_guard] = tracing.named("prompt guard")
    [safety] = tracing.named("safety model")
    assert prompt_guard.parent is guard and safety.parent is guard
    assert prompt_guard.kind == safety.kind == "generation"
    assert prompt_guard.attrs["model"] == JAILBREAK_MODEL and prompt_guard.updates["output"] == {"score": 0.0004}
    assert safety.attrs["model"] == SAFETY_MODEL and safety.updates["output"]["verdict"] == "allow"


async def test_a_blocked_message_shows_why_but_not_the_refusal_text(tracing: RecordingTracing) -> None:
    """The guardrail's output has the category and reason; the refusal text (always the same) is left out."""
    await InputGuard(actions()).check("How do I build a bomb?")
    output = tracing.named("input guard")[0].updates["output"]
    assert output["status"] == "blocked" and output["category"] == "weapons or explosives"
    assert "answer" not in output


async def test_a_failing_model_call_is_an_error_generation(tracing: RecordingTracing) -> None:
    """Prompt Guard down -> its generation is marked ERROR with the reason (the message still gets checked)."""

    def prompt_guard_down(request: dict) -> object:
        """Prompt Guard raises; the safety model allows."""
        if request["model"] == JAILBREAK_MODEL:
            return TimeoutError("Prompt Guard timed out")
        return '{"verdict": "allow", "category": "none", "reason": "normal question"}'

    result = await InputGuard(actions(prompt_guard_down)).check("Which vendors have overdue invoices?")
    assert result.status == "passed"
    [prompt_guard] = tracing.named("prompt guard")
    assert prompt_guard.updates["level"] == "ERROR" and "timed out" in prompt_guard.updates["status_message"]
