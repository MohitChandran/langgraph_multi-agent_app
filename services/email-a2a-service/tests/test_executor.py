"""Tests for the approval lifecycle (app/hitl.py) through the REAL A2A routes (plan 05, stage E2).

The app runs in memory: httpx's ASGITransport sends requests straight into it (no server, no port), the A2A client is
built from the served agent card exactly as the orchestrator will build it, the crew and the mailer are fakes, and
tasks live in the SDK's InMemoryTaskStore. So each test is the real protocol, end to end, in milliseconds.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import SimpleNamespace

import httpx
import pytest
from a2a.client import Client
from a2a.helpers import new_data_part
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import Message, Role, Task

from app.client_demo import artifact, make_client, send_data, state, status_data
from app.config import Settings
from app.fakes import FakeDrafter, FakeMailer
from app.hitl import EmailExecutor
from app.server import create_app

API_KEY = "test-key"
REQUEST = {
    "to": "accounts@company.example",
    "subject": "Pending invoices due this week",
    "content": "8 invoices are pending, 73,400 USD in total.",
    "request": "Find the pending invoices due this week and email a summary to accounts@company.example",
}


def settings(**overrides: object) -> Settings:
    """Test settings: no .env, the in-memory host name, a short crew timeout."""
    base = {
        "email_service_api_key": API_KEY,
        "email_database_url": "postgresql+asyncpg://unused/db",
        "groq_api_key": "unused",
        "email_service_public_url": "http://test",
        "email_crew_timeout_seconds": 0.5,
    }
    return Settings(_env_file=None, **{**base, **overrides})


@dataclass
class Service:
    """The running in-memory service: the A2A client, the raw HTTP client, and the fakes to inspect."""

    client: Client
    http: httpx.AsyncClient
    drafter: FakeDrafter
    mailer: FakeMailer


async def start(drafter: FakeDrafter | None = None, mailer: FakeMailer | None = None) -> AsyncIterator[Service]:
    """Build the app with fakes, run its startup, and yield a client built from its agent card."""
    drafter, mailer = drafter or FakeDrafter(), mailer or FakeMailer()
    app = create_app(settings=settings(), drafter=drafter, mailer=mailer, task_store=InMemoryTaskStore())
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", headers={"X-API-Key": API_KEY}
        ) as http:
            yield Service(await make_client("http://test", http), http, drafter, mailer)


@pytest.fixture
async def service() -> AsyncIterator[Service]:
    """The service with a working fake crew and mailer."""
    async for running in start():
        yield running


async def drafted(service: Service) -> Task:
    """Message #1 with the standard request; the task as it comes back."""
    return await send_data(service.client, REQUEST)


# --------------------------------------------------------------------------- message #1: draft and pause


async def test_a_request_is_drafted_and_paused_for_approval(service: Service) -> None:
    """#1 -> INPUT_REQUIRED, a "draft" artifact (recipient from the request), a note saying what input is needed."""
    task = await drafted(service)
    assert state(task) == "INPUT_REQUIRED"
    draft = artifact(task, "draft")
    assert draft["to"] == "accounts@company.example" and "73,400 USD" in draft["body"]
    assert status_data(task) == {"action": "approve_or_reject", "draft_artifact": "draft"}
    assert service.mailer.sent == []  # nothing leaves before a decision


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"to": "ceo@gmail.com"}, "not allowed in this environment"),
        ({"to": "a@company.example, b@company.example"}, "not one valid email address"),
        ({"subject": "x" * 151}, "subject"),
        ({"cc": "boss@company.example"}, "cc"),  # an unknown field: contract mismatch
    ],
)
async def test_an_invalid_request_is_rejected_before_the_crew_runs(service: Service, change: dict, reason: str) -> None:
    """Bad recipient, size or field -> REJECTED with the reason; the crew is never called."""
    task = await send_data(service.client, {**REQUEST, **change})
    assert state(task) == "REJECTED" and reason in status_data(task)["error"]
    assert service.drafter.requests == []


async def test_a_crew_failure_or_timeout_fails_the_task() -> None:
    """The crew raising, or taking longer than EMAIL_CREW_TIMEOUT_SECONDS -> FAILED, with a reason."""
    async for service in start(drafter=FakeDrafter(fail=True)):
        task = await drafted(service)
        assert state(task) == "FAILED" and "crew failed" in status_data(task)["error"]
    async for service in start(drafter=FakeDrafter(delay_seconds=2)):
        task = await drafted(service)
        assert state(task) == "FAILED" and "took too long" in status_data(task)["error"]


# --------------------------------------------------------------------------- message #2: the decision


async def test_an_approval_sends_once_and_completes_with_a_receipt(service: Service) -> None:
    """approve -> the mailer is called exactly once with the draft -> COMPLETED + "receipt" artifact."""
    task = await drafted(service)
    done = await send_data(service.client, {"decision": "approve"}, task.id, task.context_id)
    assert state(done) == "COMPLETED"
    [sent] = service.mailer.sent
    assert sent["to"] == "accounts@company.example" and sent["task_id"] == task.id
    assert sent["body"] == artifact(task, "draft")["body"]
    receipt = artifact(done, "receipt")
    assert receipt["message_id"] == status_data(done)["message_id"] and receipt["to"] == "accounts@company.example"


async def test_a_rejection_sends_nothing(service: Service) -> None:
    """reject -> REJECTED, the mailer is never called."""
    task = await drafted(service)
    done = await send_data(service.client, {"decision": "reject"}, task.id, task.context_id)
    assert state(done) == "REJECTED" and service.mailer.sent == []


async def test_a_finished_task_refuses_another_decision(service: Service) -> None:
    """A second approval of a COMPLETED task is refused by the SDK: still exactly one email."""
    task = await drafted(service)
    await send_data(service.client, {"decision": "approve"}, task.id, task.context_id)
    with pytest.raises(Exception, match="terminal state"):
        await send_data(service.client, {"decision": "approve"}, task.id, task.context_id)
    assert len(service.mailer.sent) == 1


async def test_an_edited_body_is_sent_as_written(service: Service) -> None:
    """approve with edited_body -> exactly that text is sent (no LLM rewrite)."""
    task = await drafted(service)
    edited = "Hi team,\n\n8 invoices are pending (73,400 USD).\n\nThanks"
    await send_data(service.client, {"decision": "approve", "edited_body": edited}, task.id, task.context_id)
    assert service.mailer.sent[0]["body"] == edited


async def test_an_edit_that_fails_a_check_stays_paused_and_sends_nothing(service: Service) -> None:
    """An edited body with a card number -> still INPUT_REQUIRED, with the reason; nothing sent; a valid decision
    afterwards still works."""
    task = await drafted(service)
    bad = {"decision": "approve", "edited_body": "Pay with card 4111 1111 1111 1111"}
    paused = await send_data(service.client, bad, task.id, task.context_id)
    assert state(paused) == "INPUT_REQUIRED" and "payment card" in status_data(paused)["error"]
    assert service.mailer.sent == []
    done = await send_data(service.client, {"decision": "reject"}, task.id, task.context_id)
    assert state(done) == "REJECTED"


async def test_an_invalid_decision_stays_paused(service: Service) -> None:
    """{"decision": "maybe"} -> still INPUT_REQUIRED with the reason."""
    task = await drafted(service)
    paused = await send_data(service.client, {"decision": "maybe"}, task.id, task.context_id)
    assert state(paused) == "INPUT_REQUIRED" and "invalid decision" in status_data(paused)["error"]


async def test_an_smtp_failure_fails_the_task() -> None:
    """The mailer raising -> FAILED with "SMTP error"."""
    async for service in start(mailer=FakeMailer(fail=True)):
        task = await drafted(service)
        done = await send_data(service.client, {"decision": "approve"}, task.id, task.context_id)
        assert state(done) == "FAILED" and "SMTP error" in status_data(done)["error"]


# --------------------------------------------------------------------------- protocol details and auth


class RecordingQueue:
    """Stands in for the SDK's EventQueue: records what the executor publishes, in order."""

    def __init__(self) -> None:
        """No events yet."""
        self.events: list[object] = []

    async def enqueue_event(self, event: object) -> None:
        """Record one event."""
        self.events.append(event)


async def test_the_task_is_published_before_any_status_update() -> None:
    """A2A 1.x: the first event of a new task must be the Task itself (plan 05, gotcha 2)."""
    queue = RecordingQueue()
    message = Message(message_id="m1", role=Role.ROLE_USER, parts=[new_data_part(REQUEST)])
    context = SimpleNamespace(task_id="t1", context_id="c1", current_task=None, message=message)
    await EmailExecutor(FakeDrafter(), FakeMailer(), settings()).execute(context, queue)
    assert isinstance(queue.events[0], Task)
    assert len(queue.events) > 1  # then WORKING, the draft artifact, INPUT_REQUIRED


async def test_calls_need_the_api_key_but_the_card_and_health_are_public(service: Service) -> None:
    """POST / without or with a wrong key -> 401; the agent card and /health need no key."""
    body = {"jsonrpc": "2.0", "id": 1, "method": "GetTask", "params": {"id": "x"}}
    async with httpx.AsyncClient(transport=service.http._transport, base_url="http://test") as anonymous:
        assert (await anonymous.post("/", json=body)).status_code == 401
        assert (await anonymous.post("/", json=body, headers={"X-API-Key": "wrong"})).status_code == 401
        card = await anonymous.get("/.well-known/agent-card.json")
        assert card.status_code == 200 and card.json()["securitySchemes"]["apiKey"]["apiKeySecurityScheme"]["name"]
        assert (await anonymous.get("/health")).json() == {"status": "ok"}


async def test_readiness_reports_the_task_store_smtp_and_tracing(service: Service) -> None:
    """/health/ready is public: the in-memory store is fine, tracing is off (no keys), the SMTP server is probed
    (whatever Mailpit's state, the answer is 200: SMTP is optional)."""
    async with httpx.AsyncClient(transport=service.http._transport, base_url="http://test") as anonymous:
        response = await anonymous.get("/health/ready")
    body = response.json()
    assert response.status_code == 200 and body["status"] in ("ready", "degraded")
    assert body["checks"]["task store"] == {**body["checks"]["task store"], "status": "ok", "detail": "in memory"}
    assert body["checks"]["tracing (Langfuse)"]["status"] == "off"
    assert body["checks"]["smtp server"]["required"] is False
