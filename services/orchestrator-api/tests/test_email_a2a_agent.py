"""Tests for A2AEmailAgent, the orchestrator's A2A client of the email service (plan 05, stage E5).

The real email service lives in another environment (and its package is also called `app`), so it can't be imported
here. Instead, a SCRIPTED A2A server is built from the same a2a-sdk pieces and follows the service's contract
(plan 05, section 5): message #1 -> INPUT_REQUIRED + "draft" artifact (or REJECTED / FAILED with an "error"),
message #2 -> COMPLETED + "receipt" / REJECTED / INPUT_REQUIRED with an "error" (a refused edit). It runs in memory
(httpx.ASGITransport), so these tests exercise the real protocol and the client's mapping, in milliseconds.
The full chain against the real service is checked live (stage E5/E6).
"""

import asyncio
import time
from collections.abc import AsyncIterator
from uuid import UUID

import httpx
import pytest
from a2a.helpers import get_data_parts, new_data_part, new_task
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, TaskState
from fastapi import FastAPI, HTTPException
from google.protobuf.json_format import MessageToDict

from app.agents.email.a2a_agent import A2AEmailAgent
from app.agents.email.dummy import DummyEmailAgent
from app.orchestrator.models import EmailDecision, Limits
from app.orchestrator.tools import AgentTools, DraftEmailArgs, TurnContext
from tests.tracing_fakes import FAKE_TRACEPARENT

API_KEY = "test-key"


class ScriptedEmailService(AgentExecutor):
    """The email service's contract without a crew or SMTP. Records what it received."""

    def __init__(self, draft_delay: float = 0.0) -> None:
        """No requests yet; draft_delay > the client's timeout simulates a slow crew."""
        self.requests: list[dict] = []
        self.metadata: list[dict] = []  # each message's metadata (the traceparent travels there)
        self.sent: list[dict] = []
        self.api_keys: list[str | None] = []
        self.draft_delay = draft_delay
        self.down = False  # True: /health fails (the service "went down")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """#1 -> draft + INPUT_REQUIRED (or refused/failed); #2 -> sent / rejected / refused edit."""
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        data = get_data_parts(context.message.parts)[0]
        self.requests.append(data)
        self.metadata.append(MessageToDict(context.message.metadata) if context.message.HasField("metadata") else {})

        def reply(payload: dict):  # noqa: ANN202
            """A status message with one data part."""
            return updater.new_agent_message([new_data_part(payload)])

        if context.current_task is None:
            await event_queue.enqueue_event(
                new_task(context.task_id, context.context_id, TaskState.TASK_STATE_SUBMITTED)
            )
            await updater.start_work()
            await asyncio.sleep(self.draft_delay)
            if data["to"].endswith("@gmail.com"):
                await updater.reject(reply({"error": "sending to it is not allowed in this environment"}))
                return
            if data["subject"] == "crash":
                await updater.failed(reply({"error": "the email crew failed: RuntimeError"}))
                return
            draft = {"to": data["to"], "subject": data["subject"], "body": f"Hello,\n\n{data['content']}", "issues": []}
            await updater.add_artifact([new_data_part(draft)], name="draft")
            await updater.requires_input(reply({"action": "approve_or_reject", "draft_artifact": "draft"}))
            return
        if data["decision"] == "reject":
            await updater.reject(reply({"result": "rejected"}))
            return
        if "4111" in (data.get("edited_body") or ""):
            await updater.requires_input(reply({"error": "the edited email was not sent: looks like a card number"}))
            return
        self.sent.append(data)
        receipt = {"message_id": "<1@ops-assistant>", "sent_at": "2026-11-22T10:00:00Z", "to": "x", "subject": "s"}
        await updater.add_artifact([new_data_part(receipt)], name="receipt")
        await updater.complete(reply({"result": "sent"}))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the task."""
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


def build_service(executor: ScriptedEmailService) -> tuple[FastAPI, InMemoryTaskStore]:
    """An A2A app around the scripted executor; it also records the X-API-Key of every JSON-RPC call."""
    card = AgentCard(
        name="Scripted email service", description="test", version="0",
        supported_interfaces=[AgentInterface(url="http://email/", protocol_binding="JSONRPC", protocol_version="1.0")],
        capabilities=AgentCapabilities(streaming=False), default_input_modes=["application/json"],
        default_output_modes=["application/json"],
    )  # fmt: skip
    store = InMemoryTaskStore()
    handler = DefaultRequestHandlerV2(agent_executor=executor, task_store=store, agent_card=card)
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict:
        """Like the email service's liveness check; `executor.down` makes it fail."""
        if executor.down:
            raise HTTPException(status_code=503)
        return {"status": "ok"}

    @app.middleware("http")
    async def record_key(request, call_next):  # noqa: ANN001, ANN202
        """Remember the API key the client sent with each POST."""
        if request.method == "POST":
            executor.api_keys.append(request.headers.get("X-API-Key"))
        return await call_next(request)

    add_a2a_routes_to_fastapi(app, agent_card_routes=create_agent_card_routes(card),
                              jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url="/"))  # fmt: skip
    return app, store


async def connected(executor: ScriptedEmailService, **options: float) -> A2AEmailAgent:
    """A started A2AEmailAgent talking to the scripted service in memory."""
    app, _ = build_service(executor)
    agent = A2AEmailAgent("http://email", API_KEY, transport=httpx.ASGITransport(app=app), **options)
    await agent.start()
    return agent


@pytest.fixture
async def service() -> AsyncIterator[tuple[A2AEmailAgent, ScriptedEmailService]]:
    """The client connected to a fresh scripted service; closed afterwards."""
    executor = ScriptedEmailService()
    agent = await connected(executor)
    try:
        yield agent, executor
    finally:
        await agent.close()


async def test_a_draft_comes_back_waiting_for_approval(service) -> None:  # noqa: ANN001
    """INPUT_REQUIRED + draft -> input_required with the A2A task id and context id; the key was sent."""
    agent, executor = service
    assert agent.available
    result = await agent.draft("accounts@company.example", "Invoices", "8 invoices.", request="email it " * 200)
    assert result.success and result.status == "input_required" and result.needs_approval
    assert result.draft.to == ["accounts@company.example"] and "8 invoices." in result.draft.body
    assert isinstance(result.draft.draft_id, UUID) and result.draft.context_id
    assert len(executor.requests[0]["request"]) == 1000  # the user's question, cut to the service's limit
    assert executor.api_keys == [API_KEY]


async def test_approve_sends_and_reject_does_not(service) -> None:  # noqa: ANN001
    """approve -> completed with the receipt's message id; reject (a new draft) -> rejected, nothing sent."""
    agent, executor = service
    first = (await agent.draft("accounts@company.example", "Invoices", "8 invoices.")).draft
    sent = await agent.resume(first.draft_id, True, "Short version.", context_id=first.context_id)
    assert sent.status == "completed" and sent.message_id == "<1@ops-assistant>" and sent.draft.body == "Short version."
    assert executor.sent == [{"decision": "approve", "edited_body": "Short version."}]
    second = (await agent.draft("accounts@company.example", "Invoices", "8 invoices.")).draft
    rejected = await agent.resume(second.draft_id, False, context_id=second.context_id)
    assert rejected.success and rejected.status == "rejected" and len(executor.sent) == 1


async def test_a_second_decision_is_refused_cleanly(service) -> None:  # noqa: ANN001
    """Deciding a finished task again -> failed "already decided", never an exception, still one email."""
    agent, executor = service
    draft = (await agent.draft("accounts@company.example", "Invoices", "8 invoices.")).draft
    await agent.resume(draft.draft_id, True, context_id=draft.context_id)
    again = await agent.resume(draft.draft_id, True, context_id=draft.context_id)
    assert not again.success and again.status == "failed" and again.error == "this draft was already decided"
    assert len(executor.sent) == 1


async def test_refused_requests_and_crew_failures_carry_the_services_reason(service) -> None:  # noqa: ANN001
    """REJECTED (domain not allowed) / FAILED (crew) -> failed with the service's own reason."""
    agent, _ = service
    refused = await agent.draft("ceo@gmail.com", "Invoices", "8 invoices.")
    assert refused.status == "failed" and "not allowed in this environment" in refused.error
    crashed = await agent.draft("accounts@company.example", "crash", "8 invoices.")
    assert crashed.status == "failed" and "crew failed" in crashed.error


async def test_a_refused_edit_is_not_sent_and_the_task_is_cancelled(service) -> None:  # noqa: ANN001
    """An edit the service refuses -> failed with its reason; the waiting task is cancelled, nothing sent."""
    agent, executor = service
    draft = (await agent.draft("accounts@company.example", "Invoices", "8 invoices.")).draft
    result = await agent.resume(draft.draft_id, True, "card 4111 1111 1111 1111", context_id=draft.context_id)
    assert result.status == "failed" and "card number" in result.error and executor.sent == []
    late = await agent.resume(draft.draft_id, True, context_id=draft.context_id)  # cancelled = terminal
    assert late.status == "failed" and executor.sent == []


async def test_a_slow_service_times_out_into_a_failure() -> None:
    """No answer within the draft timeout -> failed with "did not answer within"."""
    agent = await connected(ScriptedEmailService(draft_delay=1.0), draft_timeout_seconds=0.2)
    try:
        result = await agent.draft("accounts@company.example", "Invoices", "8 invoices.")
        assert result.status == "failed" and "did not answer within 0.2 s" in result.error
    finally:
        await agent.close()


async def test_an_unreachable_service_is_unavailable_and_draft_email_is_not_offered() -> None:
    """No service (or no API key) -> available False -> AgentTools leaves draft_email out; draft() fails cleanly."""
    down = A2AEmailAgent("http://127.0.0.1:9", API_KEY, draft_timeout_seconds=2)
    await down.start()
    no_key = A2AEmailAgent("http://email", "")
    await no_key.start()
    assert not down.available and not no_key.available
    result = await down.draft("accounts@company.example", "Invoices", "8 invoices.")
    assert result.status == "failed" and "not reachable" in result.error

    class NoDocuments:
        """A document store with no documents."""

        async def list_documents(self, session_id):  # noqa: ANN001, ANN201
            """Nothing."""
            return []

    tools = AgentTools(DummyEmailAgent(), DummyEmailAgent(), DummyEmailAgent(), down, NoDocuments(), Limits())
    names, _ = await tools.available(None)
    assert "draft_email" not in names
    await down.close()


async def test_agent_tools_pass_the_question_and_the_context_id_through(service) -> None:  # noqa: ANN001
    """draft_email sends the user's question along; the decision reaches the SAME task (context id from the
    paused run's saved draft)."""
    agent, executor = service

    class NoDocuments:
        """A document store with no documents."""

        async def list_documents(self, session_id):  # noqa: ANN001, ANN201
            """Nothing."""
            return []

    tools = AgentTools(DummyEmailAgent(), DummyEmailAgent(), DummyEmailAgent(), agent, NoDocuments(), Limits())
    args = DraftEmailArgs(to=["accounts@company.example"], subject="Invoices", content="8 invoices.")
    drafted = await tools.draft_email("e1", args, TurnContext(None, question="Email the invoices to accounts@..."))
    assert executor.requests[0]["request"] == "Email the invoices to accounts@..."
    assert drafted.pending.draft["context_id"]
    outcome = await tools.resolve_email(drafted.pending, EmailDecision(approved=True))
    assert (
        outcome.call.status == "sent" and outcome.text == "Approved by the user and sent to accounts@company.example."
    )


async def test_the_trace_continues_in_the_email_service(service, tracing) -> None:  # noqa: ANN001
    """With tracing on, each A2A message carries the current span as a W3C traceparent (both messages), inside an
    "email service: draft/decision" tool span; without tracing, no metadata at all."""
    agent, executor = service
    draft = (await agent.draft("accounts@company.example", "Invoices", "8 invoices.")).draft
    await agent.resume(draft.draft_id, False, context_id=draft.context_id)
    assert executor.metadata == [{"traceparent": FAKE_TRACEPARENT}] * 2
    assert [o.name for o in tracing.observations] == ["email service: draft", "email service: decision"]
    assert tracing.observations[0].updates["output"] == {"state": "INPUT_REQUIRED", "error": None}


async def test_without_tracing_no_metadata_is_sent(service) -> None:  # noqa: ANN001
    """Tracing off (the default in tests): the message carries no metadata."""
    agent, executor = service
    await agent.draft("accounts@company.example", "Invoices", "8 invoices.")
    assert executor.metadata == [{}]


async def test_the_readiness_check_hides_and_restores_draft_email(service) -> None:  # noqa: ANN001
    """check(): OK -> available; the service's /health failing -> unavailable (draft_email hidden) and the check
    raises; healthy again -> available again, without restarting anything."""
    agent, executor = service
    assert await agent.check() == "A2A service at http://email" and agent.available
    executor.down = True
    with pytest.raises(httpx.HTTPError):
        await agent.check()
    assert not agent.available
    executor.down = False
    await agent.check()
    assert agent.available


async def test_refresh_brings_draft_email_back_without_a_restart() -> None:
    """Service down at startup -> unavailable; refresh() before a question looks again (at most every 30 s) and
    makes it available once the service answers."""
    executor = ScriptedEmailService()
    app, _ = build_service(executor)
    executor.down = True
    agent = A2AEmailAgent("http://email", API_KEY, transport=httpx.ASGITransport(app=app))
    await agent.start()  # the card is readable, but /health fails...
    try:
        with pytest.raises(httpx.HTTPError):
            await agent.check()
        assert not agent.available
        executor.down = False
        await agent.refresh()  # refresh hasn't looked yet: it looks now, and the service answers
        assert agent.available
        executor.down = True
        with pytest.raises(httpx.HTTPError):
            await agent.check()
        agent._last_attempt = time.monotonic()  # as if refresh had just looked
        executor.down = False
        await agent.refresh()
        assert not agent.available  # throttled: no new attempt within 30 s
    finally:
        await agent.close()
