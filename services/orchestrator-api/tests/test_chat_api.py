"""API tests for chat sessions (plan 01, stage S2; plan 03, stage O4): real routes + real Postgres,
the REAL orchestrator, FAKE agents and a FAKE supervisor LLM (no Groq).

How the app is wired here:
  - httpx.AsyncClient + ASGITransport sends HTTP requests straight into the FastAPI app
    in memory (no server, no port).
  - ASGITransport does NOT run the lifespan, so app.state is empty. We don't need it:
    dependency_overrides hands the endpoints our test ChatStore and an Orchestrator built from
    fakes: RuleSupervisor (below) decides like a simple, predictable LLM, and FakeAgent /
    FakeDocAgent answer instantly. So these tests exercise the whole chain:
    route -> service -> orchestrator loop -> agent -> saved message -> memory on the next turn.
  - Every test message starts with "[pytest]", so its chat title does too, and the
    fixture deletes exactly those chats afterwards (your real chats are never touched).
"""

from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.agents.db.models import DBAgentResult
from app.agents.doc.models import Citation, DocAgentResult
from app.agents.doc.store import DocumentStore
from app.agents.email.dummy import DummyEmailAgent
from app.agents.web.dummy import DummyWebSearchAgent
from app.chat.service import make_title
from app.chat.store import ChatStore
from app.dependencies import get_chat_store, get_db_agent, get_document_store, get_input_guard, get_orchestrator
from app.guardrails.guard import GuardResult
from app.guardrails.policy import REFUSE_HARMFUL
from app.main import app
from app.orchestrator.graph import Orchestrator
from app.orchestrator.models import Limits
from app.orchestrator.tools import EMAIL_PATTERN, AgentTools
from tests.orchestrator_fakes import ScriptedLLM, answer, call
from tests.tracing_fakes import RecordingTracing

TEST_PREFIX = "[pytest]"


class FakeAgent:
    """Stands in for DBAgent: answers instantly, records every call, can be told to crash."""

    name = "db"

    def __init__(self) -> None:
        """Start with no recorded calls and crashing switched off."""
        self.calls: list[dict] = []
        self.crash = False

    async def ask(
        self,
        question: str,
        history: list | None = None,
        session_id: UUID | None = None,
        callbacks: list | None = None,
    ) -> DBAgentResult:
        """Record the question (and the memory it was given), then answer or raise. callbacks: tracing, ignored."""
        self.calls.append({"question": question, "history": history, "session_id": session_id})
        if self.crash:
            raise RuntimeError("Groq is unreachable")
        return DBAgentResult(
            question=question,
            answer=f"Answer to: {question}",
            success=True,
            sql=f"SELECT '{question}' AS q LIMIT 100",
            columns=["q"],
            rows=[{"q": question}],
            attempts=1,
            steps=["write_sql", "validate_sql", "execute_sql", "answer"],
        )


class FakeDocAgent:
    """Stands in for DocAgent: answers with one citation, records calls (incl. which chat it searched)."""

    name = "doc"

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls: list[dict] = []

    async def ask(self, question: str, history: list | None = None, session_id: UUID | None = None) -> DocAgentResult:
        """Record the call and answer from a pretend source on page 2."""
        self.calls.append({"question": question, "history": history, "session_id": session_id})
        citation = Citation(
            n=1,
            chunk_id=1,
            document_id=uuid4(),
            filename="q3-report.pdf",
            page_start=2,
            page_end=2,
            heading_path="4 Cloud Vendors > 4.2 SLA Terms",
            chunk_type="table",
            snippet="AWS 99.95%",
        )
        return DocAgentResult(
            question=question,
            answer=f"From the PDF: {question} [1]",
            success=True,
            grounded=True,
            standalone_query=question,
            citations=[citation],
            steps=["rewrite_query", "answer"],
        )


class RuleSupervisor(ScriptedLLM):
    """A fake supervisor LLM with fixed rules instead of a script, so any conversation works:

      - a question with "hello" -> answered directly, no tools;
      - otherwise first ask the documents (if the chat has any, i.e. ask_documents is offered)
        or the database, with the user's question;
      - if the question contains an email address, then draft an email to it with that result;
      - then answer "Final answer for: <question>".
    `crash = True` makes it raise a bug (not an LLM error), which fails the whole turn.
    """

    def __init__(self) -> None:
        """No crash; every reply is decided by `decide`."""
        super().__init__(self.decide)
        self.crash = False

    def decide(self, messages: list[BaseMessage]) -> AIMessage | Exception:
        """Pick the next reply from what the conversation looks like so far."""
        if self.crash:
            return RuntimeError("bug in the supervisor")
        system, question = messages[0].content, messages[1].content
        results = [m for m in messages if isinstance(m, ToolMessage)]
        drafted = any(c["name"] == "draft_email" for m in messages if isinstance(m, AIMessage) for c in m.tool_calls)
        if "hello" in question.lower():
            return answer("Hi there!")
        if not results:
            return call("ask_documents" if "- ask_documents:" in system else "ask_database", question=question)
        address = EMAIL_PATTERN.search(question)
        if address and not drafted:
            return call("draft_email", to=[address.group(0)], subject="Summary", content=results[-1].content[:500])
        return answer(f"Final answer for: {question}")


class FakeGuard:
    """Stands in for the input guard (no Groq): blocks messages with "bomb", passes the rest, records each check."""

    def __init__(self) -> None:
        """No messages checked yet."""
        self.checked: list[str] = []

    async def check(self, message: str) -> GuardResult:
        """Block "bomb" as a harmful request; pass everything else with a low jailbreak score."""
        self.checked.append(message)
        if "bomb" in message:
            return GuardResult(
                status="blocked", rail="safety check", category="weapons or explosives",
                reason="asks how to build a bomb", jailbreak_score=0.0004, answer=REFUSE_HARMFUL, duration_ms=5,
            )  # fmt: skip
        return GuardResult(status="passed", jailbreak_score=0.0004, duration_ms=5)


@pytest.fixture
def guard() -> FakeGuard:
    """A fresh FakeGuard per test."""
    return FakeGuard()


@pytest.fixture
def fake_agent() -> FakeAgent:
    """A fresh FakeAgent per test."""
    return FakeAgent()


@pytest.fixture
def fake_doc_agent() -> FakeDocAgent:
    """A fresh FakeDocAgent per test."""
    return FakeDocAgent()


@pytest.fixture
def supervisor() -> RuleSupervisor:
    """A fresh rule-based fake supervisor per test."""
    return RuleSupervisor()


@pytest.fixture
def orchestrator(
    document_store: DocumentStore, fake_agent: FakeAgent, fake_doc_agent: FakeDocAgent, supervisor: RuleSupervisor
) -> Orchestrator:
    """The real Orchestrator over the fake agents, the dummy web/email agents and an in-memory checkpointer."""
    tools = AgentTools(fake_agent, fake_doc_agent, DummyWebSearchAgent(), DummyEmailAgent(), document_store, Limits())
    return Orchestrator(tools, chat_model=supervisor)


@pytest.fixture
async def client(
    chat_store: ChatStore,
    document_store: DocumentStore,
    orchestrator: Orchestrator,
    guard: FakeGuard,
    fake_agent: FakeAgent,
) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client for the app with the test stores, guard, orchestrator and DB agent injected; cleans up test chats."""
    app.dependency_overrides[get_chat_store] = lambda: chat_store
    app.dependency_overrides[get_document_store] = lambda: document_store
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    app.dependency_overrides[get_input_guard] = lambda: guard
    app.dependency_overrides[get_db_agent] = lambda: fake_agent  # POST /api/db/ask
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            yield http
    finally:
        app.dependency_overrides.clear()
        await chat_store.fetch_one(
            "WITH gone AS (DELETE FROM app.chat_sessions WHERE title LIKE %s RETURNING 1) SELECT count(*) FROM gone",
            (f"{TEST_PREFIX}%",),
        )


async def send(client: httpx.AsyncClient, content: str, session_id: str | None = None) -> httpx.Response:
    """POST one test message (prefixed so it can be cleaned up)."""
    return await client.post("/api/chat", json={"session_id": session_id, "content": f"{TEST_PREFIX} {content}"})


async def decide(client: httpx.AsyncClient, message_id: int, body: dict) -> httpx.Response:
    """POST an email decision for an answer."""
    return await client.post(f"/api/messages/{message_id}/email-approval", json=body)


# ------------------------------------------------------------------ create & continue


@pytest.mark.integration
async def test_first_message_creates_a_chat_and_the_second_continues_it(client: httpx.AsyncClient) -> None:
    """session_id null -> a new chat with 2 messages; sending with its id appends 2 more to the SAME chat."""
    first = await send(client, "Which vendors have overdue invoices?")
    assert first.status_code == 200
    body = first.json()
    session_id = body["session"]["id"]
    assert body["session"]["title"] == f"{TEST_PREFIX} Which vendors have overdue invoices?"
    assert body["session"]["message_count"] == 2 and body["session"]["is_running"] is False
    assert body["user_message"]["role"] == "user"
    assistant = body["assistant_message"]
    assert assistant["status"] == "complete" and assistant["content"].startswith("Answer to:")  # passthrough (D9)
    payload = assistant["payload"]
    assert payload["agent"] == "orchestrator" and payload["route"]["router"] == "llm"
    [db_call] = payload["result"]["calls"]
    assert db_call["agent"] == "db" and db_call["result"]["sql"].startswith("SELECT")
    assert "answer" not in payload["result"]  # the answer is the content, not duplicated
    assert assistant["latency_ms"] is not None

    second = await send(client, "only the ones in India", session_id)
    assert second.status_code == 200
    assert second.json()["session"]["id"] == session_id
    assert second.json()["session"]["message_count"] == 4

    page = (await client.get(f"/api/sessions/{session_id}/messages")).json()
    assert [m["role"] for m in page["items"]] == ["user", "assistant", "user", "assistant"]
    assert page["items"][2]["content"] == f"{TEST_PREFIX} only the ones in India"
    assert page["has_more"] is False


@pytest.mark.integration
async def test_sessions_are_listed_by_latest_activity(client: httpx.AsyncClient) -> None:
    """The chat used most recently is first in the sidebar."""
    chat_a = (await send(client, "chat A")).json()["session"]["id"]
    chat_b = (await send(client, "chat B")).json()["session"]["id"]
    await send(client, "back to A", chat_a)

    ids = [s["id"] for s in (await client.get("/api/sessions")).json()]
    assert ids.index(chat_a) < ids.index(chat_b)


# ------------------------------------------------------------------ busy guard & failures


@pytest.mark.integration
async def test_busy_chat_returns_409_but_a_stale_claim_is_taken_over(
    client: httpx.AsyncClient, chat_store: ChatStore
) -> None:
    """While a reply is running (fresh claim) -> 409. A claim older than 5 minutes is from a crash -> taken over."""
    session_id = (await send(client, "first")).json()["session"]["id"]

    await chat_store.fetch_one(
        "UPDATE app.chat_sessions SET active_run_at = now() WHERE id = %s RETURNING id", (session_id,)
    )
    busy = await send(client, "second", session_id)
    assert busy.status_code == 409 and "still being generated" in busy.json()["detail"]

    await chat_store.fetch_one(
        "UPDATE app.chat_sessions SET active_run_at = now() - interval '6 minutes' WHERE id = %s RETURNING id",
        (session_id,),
    )
    taken_over = await send(client, "second", session_id)
    assert taken_over.status_code == 200 and taken_over.json()["session"]["is_running"] is False


@pytest.mark.integration
async def test_orchestrator_crash_is_saved_as_error_and_the_chat_keeps_working(
    client: httpx.AsyncClient, supervisor: RuleSupervisor
) -> None:
    """A bug in the loop -> the answer is saved with status 'error', the claim is released, the next send works."""
    supervisor.crash = True
    crashed = await send(client, "this will crash")
    assert crashed.status_code == 200
    body = crashed.json()
    assert body["assistant_message"]["status"] == "error"
    assert body["assistant_message"]["content"] == "Sorry, something went wrong while answering. Please try again."
    assert body["session"]["is_running"] is False  # released in `finally`

    supervisor.crash = False
    retry = await send(client, "try again", body["session"]["id"])
    assert retry.status_code == 200 and retry.json()["assistant_message"]["status"] == "complete"


@pytest.mark.integration
async def test_a_crashing_agent_is_one_failed_call_not_a_failed_answer(
    client: httpx.AsyncClient, fake_agent: FakeAgent
) -> None:
    """The DB agent raises -> its call is an error, the supervisor still answers, the turn completes."""
    fake_agent.crash = True
    assistant = (await send(client, "this agent will crash")).json()["assistant_message"]
    assert assistant["status"] == "complete" and assistant["content"].startswith("Final answer for:")
    [db_call] = assistant["payload"]["result"]["calls"]
    assert db_call["status"] == "error" and db_call["error"] == "RuntimeError: Groq is unreachable"


# ------------------------------------------------------------------ 404 / 422


@pytest.mark.integration
async def test_unknown_chat_returns_404_everywhere(client: httpx.AsyncClient) -> None:
    """Every endpoint that takes a session id answers 404 for one that doesn't exist."""
    unknown = str(uuid4())
    assert (await send(client, "hello?", unknown)).status_code == 404
    assert (await client.get(f"/api/sessions/{unknown}/messages")).status_code == 404
    assert (await client.patch(f"/api/sessions/{unknown}", json={"title": "x"})).status_code == 404
    assert (await client.delete(f"/api/sessions/{unknown}")).status_code == 404
    assert (await decide(client, 2_000_000_000, {"approved": True})).status_code == 404  # unknown message


@pytest.mark.integration
async def test_invalid_input_returns_422(client: httpx.AsyncClient) -> None:
    """Pydantic and FastAPI reject bad input before our code runs."""
    assert (await client.post("/api/chat", json={"content": ""})).status_code == 422
    assert (await client.post("/api/chat", json={"content": "x" * 1001})).status_code == 422
    assert (await client.get("/api/sessions/not-a-uuid/messages")).status_code == 422
    assert (await client.post("/api/messages/abc/email-approval", json={"approved": True})).status_code == 422
    assert (await decide(client, 1, {})).status_code == 422  # `approved` is required
    assert (await decide(client, 1, {"approved": "maybe"})).status_code == 422
    assert (await decide(client, 1, {"approved": True, "edited_body": "x" * 5001})).status_code == 422


# ------------------------------------------------------------------ pagination, rename, delete


@pytest.mark.integration
async def test_messages_are_paginated_newest_page_first(client: httpx.AsyncClient, chat_store: ChatStore) -> None:
    """120 messages -> the latest 100 with has_more=true; before=<first id> returns the 20 older ones."""
    session_id = (await send(client, "pagination")).json()["session"]["id"]  # messages 1-2
    await chat_store.fetch_one(
        "WITH added AS (INSERT INTO app.chat_messages (session_id, role, content)"
        " SELECT %s, 'user', 'm' || n FROM generate_series(3, 120) AS n RETURNING 1) SELECT count(*) FROM added",
        (session_id,),
    )

    latest = (await client.get(f"/api/sessions/{session_id}/messages")).json()
    assert len(latest["items"]) == 100 and latest["has_more"] is True
    assert latest["items"][-1]["content"] == "m120"  # oldest -> newest

    first_id = latest["items"][0]["id"]
    older = (await client.get(f"/api/sessions/{session_id}/messages", params={"before": first_id})).json()
    assert len(older["items"]) == 20 and older["has_more"] is False
    assert older["items"][0]["content"] == f"{TEST_PREFIX} pagination"


@pytest.mark.integration
async def test_rename_keeps_sidebar_order_and_delete_removes_messages(
    client: httpx.AsyncClient, chat_store: ChatStore
) -> None:
    """Rename changes only the title (no jump to the top); delete removes the chat AND its messages."""
    old_chat = (await send(client, "older chat")).json()["session"]["id"]
    new_chat = (await send(client, "newer chat")).json()["session"]["id"]

    renamed = await client.patch(f"/api/sessions/{old_chat}", json={"title": f"{TEST_PREFIX} renamed"})
    assert renamed.status_code == 200 and renamed.json()["title"] == f"{TEST_PREFIX} renamed"
    ids = [s["id"] for s in (await client.get("/api/sessions")).json()]
    assert ids.index(new_chat) < ids.index(old_chat)

    assert (await client.delete(f"/api/sessions/{old_chat}")).status_code == 204
    assert (await client.get(f"/api/sessions/{old_chat}/messages")).status_code == 404
    left = await chat_store.fetch_one("SELECT count(*) AS n FROM app.chat_messages WHERE session_id = %s", (old_chat,))
    assert left["n"] == 0  # ON DELETE CASCADE


# ------------------------------------------------------------------ memory (stage S3)


def remembered_questions(fake_agent: FakeAgent) -> list[str]:
    """The questions in the history the agent received on its LAST call, without the test prefix."""
    return [turn.question.removeprefix(f"{TEST_PREFIX} ") for turn in fake_agent.calls[-1]["history"]]


@pytest.mark.integration
async def test_agent_sees_only_the_last_3_exchanges_and_never_the_new_question(
    client: httpx.AsyncClient, fake_agent: FakeAgent
) -> None:
    """After 5 turns, turn 6 gets exactly turns 3-5 (oldest first), each with its SQL; not turn 6 itself."""
    session_id = (await send(client, "q1")).json()["session"]["id"]
    assert fake_agent.calls[0]["history"] == []  # a new chat starts with no memory
    for n in range(2, 7):
        await send(client, f"q{n}", session_id)

    assert remembered_questions(fake_agent) == ["q3", "q4", "q5"]
    assert all(turn.sql and turn.sql.startswith("SELECT") for turn in fake_agent.calls[-1]["history"])


@pytest.mark.integration
async def test_failed_turns_are_not_remembered(
    client: httpx.AsyncClient, fake_agent: FakeAgent, supervisor: RuleSupervisor
) -> None:
    """A turn whose answer failed (status 'error') is left out of memory."""
    session_id = (await send(client, "good question")).json()["session"]["id"]
    supervisor.crash = True
    await send(client, "crashing question", session_id)
    supervisor.crash = False
    await send(client, "follow-up", session_id)

    assert remembered_questions(fake_agent) == ["good question"]


@pytest.mark.integration
async def test_memory_is_isolated_per_chat(client: httpx.AsyncClient, fake_agent: FakeAgent) -> None:
    """Chat B never sees chat A's turns, even right after them."""
    chat_a = (await send(client, "invoices question")).json()["session"]["id"]
    await send(client, "invoices follow-up", chat_a)

    chat_b = (await send(client, "contracts question")).json()["session"]["id"]
    assert fake_agent.calls[-1]["history"] == []
    await send(client, "contracts follow-up", chat_b)
    assert remembered_questions(fake_agent) == ["contracts question"]


# ------------------------------------------------------------------ which agents the orchestrator may use (plan 03)


async def add_document(document_store: DocumentStore, session_id: str) -> None:
    """Attach a (fake) document row to a chat: from then on, the orchestrator can ask its documents."""
    await document_store.create_document(uuid4(), UUID(session_id), "q3-report.pdf", "application/pdf", 1, uuid4().hex)


@pytest.mark.integration
async def test_ask_documents_is_offered_only_in_a_chat_with_documents(
    client: httpx.AsyncClient,
    fake_agent: FakeAgent,
    fake_doc_agent: FakeDocAgent,
    document_store: DocumentStore,
    supervisor: RuleSupervisor,
) -> None:
    """No documents -> no ask_documents tool (the DB agent answers); after an upload -> the Doc agent, THIS chat."""
    first = (await send(client, "How many vendors do we have?")).json()
    session_id = first["session"]["id"]
    assert "ask_documents" not in supervisor.calls[0].tool_names and len(fake_agent.calls) == 1
    route = first["assistant_message"]["payload"]["route"]
    assert route == {"router": "llm", "agent": "db", "reason": route["reason"]}
    assert route["reason"].startswith('Asked the database agent: "[pytest] How many vendors')  # saved with it

    await add_document(document_store, session_id)
    second = (await send(client, "What credit applies if AWS breaches its SLA?", session_id)).json()
    assert "ask_documents" in supervisor.calls[-1].tool_names
    payload = second["assistant_message"]["payload"]
    assert payload["route"]["agent"] == "doc" and payload["result"]["passthrough"] is True
    assert second["assistant_message"]["content"].endswith("[1]")  # the Doc agent's own citation (D9)
    assert payload["result"]["calls"][0]["result"]["citations"][0]["filename"] == "q3-report.pdf"
    assert fake_doc_agent.calls[0]["session_id"] == UUID(session_id)  # it searches THIS chat's documents
    assert len(fake_agent.calls) == 1  # the DB agent wasn't asked again


@pytest.mark.integration
async def test_memory_of_doc_answers_carries_their_sources(
    client: httpx.AsyncClient, fake_doc_agent: FakeDocAgent, document_store: DocumentStore
) -> None:
    """A doc answer is remembered with the file and page it cited, so follow-ups can refer to it."""
    session_id = (await send(client, "start")).json()["session"]["id"]
    await add_document(document_store, session_id)
    await send(client, "What does the SLA table say?", session_id)
    await send(client, "and for Azure?", session_id)

    remembered = fake_doc_agent.calls[-1]["history"][-1]
    assert remembered.sources == "q3-report.pdf p.2" and remembered.sql is None


# ------------------------------------------------------------------ email approval (plan 03, S13-S15)

EMAIL_REQUEST = "Which invoices are overdue? Email them to accounts@company.example"


@pytest.mark.integration
async def test_an_email_waits_for_approval_and_the_chat_is_locked_until_then(client: httpx.AsyncClient) -> None:
    """Draft -> awaiting_approval; a new message -> 409; approve (edited) -> the SAME message completes, sent."""
    first = (await send(client, EMAIL_REQUEST)).json()
    session_id, waiting = first["session"]["id"], first["assistant_message"]
    assert waiting["status"] == "awaiting_approval" and first["session"]["is_running"] is False
    assert waiting["content"].startswith("I drafted an email to accounts@company.example.")
    assert waiting["payload"]["route"]["agent"] == "db → email"
    draft = waiting["payload"]["result"]["email"]
    assert draft["status"] == "awaiting_approval" and draft["to"] == ["accounts@company.example"]

    blocked = await send(client, "and another question", session_id)
    assert blocked.status_code == 409 and blocked.json()["detail"] == "Approve or reject the pending email first."

    approved = await decide(client, waiting["id"], {"approved": True, "edited_body": "Shorter body."})
    assert approved.status_code == 200
    final = approved.json()["assistant_message"]
    assert final["id"] == waiting["id"] and final["status"] == "complete"
    assert final["content"] == f"Final answer for: {TEST_PREFIX} {EMAIL_REQUEST}"
    email = final["payload"]["result"]["email"]
    assert email["status"] == "sent" and email["body"] == "Shorter body." and email["message_id"]
    assert final["latency_ms"] >= waiting["latency_ms"]

    assert (await decide(client, waiting["id"], {"approved": True})).status_code == 409  # decided already
    assert (await send(client, "thanks", session_id)).status_code == 200  # the chat is unlocked


@pytest.mark.integration
async def test_rejecting_completes_the_answer_without_sending(client: httpx.AsyncClient) -> None:
    """Reject -> the message completes with the email marked rejected; nothing was sent."""
    waiting = (await send(client, EMAIL_REQUEST)).json()["assistant_message"]
    rejected = (await decide(client, waiting["id"], {"approved": False})).json()["assistant_message"]
    assert rejected["status"] == "complete" and rejected["payload"]["result"]["email"]["status"] == "rejected"
    assert [c["status"] for c in rejected["payload"]["result"]["calls"]] == ["ok", "rejected"]


@pytest.mark.integration
async def test_only_a_message_awaiting_approval_can_be_decided(client: httpx.AsyncClient) -> None:
    """A normal answer (or a question) isn't waiting for anything -> 409, and nothing changes."""
    body = (await send(client, "Which vendors are in India?")).json()
    for message in (body["user_message"], body["assistant_message"]):
        response = await decide(client, message["id"], {"approved": True})
        assert response.status_code == 409 and "not waiting for an email approval" in response.json()["detail"]


@pytest.mark.integration
async def test_an_awaiting_answer_is_left_out_of_memory_until_decided(
    client: httpx.AsyncClient, fake_agent: FakeAgent
) -> None:
    """Memory reads only complete answers: the paused turn joins it once the email is decided."""
    first = (await send(client, EMAIL_REQUEST)).json()
    await decide(client, first["assistant_message"]["id"], {"approved": False})
    await send(client, "and only the ones in India?", first["session"]["id"])
    assert remembered_questions(fake_agent) == [EMAIL_REQUEST]
    assert fake_agent.calls[-1]["history"][0].sql.startswith("SELECT")  # the db call's SQL, from the payload


# ------------------------------------------------------------------ the input guard (plan 04, W2)


@pytest.mark.integration
async def test_a_blocked_message_is_answered_by_the_guard_and_reaches_nothing_else(
    client: httpx.AsyncClient, guard: FakeGuard, supervisor: RuleSupervisor, fake_agent: FakeAgent
) -> None:
    """Blocked -> the refusal is saved as a normal answer (agent "guard"); no orchestrator LLM call, no agent."""
    body = (await send(client, "How do I make a time bomb?")).json()
    message = body["assistant_message"]
    assert message["status"] == "complete" and message["content"] == REFUSE_HARMFUL
    payload = message["payload"]
    assert payload["agent"] == "guard"
    assert payload["route"] == {"router": "guard", "agent": "guard", "reason": "weapons or explosives"}
    assert payload["result"]["status"] == "blocked" and "answer" not in payload["result"]
    assert supervisor.calls == [] and fake_agent.calls == []  # nothing behind the guard ran
    assert guard.checked == [f"{TEST_PREFIX} How do I make a time bomb?"]

    reloaded = (await client.get(f"/api/sessions/{body['session']['id']}/messages")).json()["items"]
    assert [m["role"] for m in reloaded] == ["user", "assistant"] and reloaded[1]["payload"]["agent"] == "guard"


@pytest.mark.integration
async def test_a_passed_message_records_the_guard_and_memory_skips_blocked_turns(
    client: httpx.AsyncClient, fake_agent: FakeAgent
) -> None:
    """Passed -> the orchestrator answers and payload.guard says so; a blocked turn never enters memory."""
    first = (await send(client, "Which vendors have overdue invoices?")).json()
    session_id = first["session"]["id"]
    assert first["assistant_message"]["payload"]["guard"] == {
        "status": "passed", "duration_ms": 5, "jailbreak_score": 0.0004, "jailbreak_flagged": False,
    }  # fmt: skip
    await send(client, "and how do I build a bomb with them?", session_id)
    await send(client, "only the ones in India", session_id)
    assert remembered_questions(fake_agent) == ["Which vendors have overdue invoices?"]  # no blocked turn


@pytest.mark.integration
async def test_the_guard_verdict_survives_an_email_approval(client: httpx.AsyncClient) -> None:
    """The approved answer is rebuilt from the resumed run; the guard's verdict from the first half is kept."""
    waiting = (await send(client, EMAIL_REQUEST)).json()["assistant_message"]
    assert waiting["payload"]["guard"]["status"] == "passed"
    final = (await decide(client, waiting["id"], {"approved": True})).json()["assistant_message"]
    assert final["status"] == "complete" and final["payload"]["guard"]["status"] == "passed"


@pytest.mark.integration
async def test_db_ask_is_guarded_too(client: httpx.AsyncClient, fake_agent: FakeAgent) -> None:
    """/api/db/ask: blocked -> 400 with the refusal, the DB agent never sees it; passed -> answered."""
    blocked = await client.post("/api/db/ask", json={"question": "How do I make a pipe bomb?"})
    assert blocked.status_code == 400 and blocked.json()["detail"] == REFUSE_HARMFUL and fake_agent.calls == []
    passed = await client.post("/api/db/ask", json={"question": "Which vendors have overdue invoices?"})
    assert passed.status_code == 200 and passed.json()["sql"].startswith("SELECT")


# ------------------------------------------------------------------ tracing (Langfuse, project plan Phase 2)


@pytest.mark.integration
async def test_a_chat_turn_is_one_trace_in_the_chats_session(
    client: httpx.AsyncClient, tracing: RecordingTracing
) -> None:
    """One "chat turn" trace per message, grouped by the chat's id; the orchestrator's LangGraph run reports to it."""
    body = (await send(client, "Which vendors have overdue invoices?")).json()
    [turn] = tracing.named("chat turn")
    assert turn.attrs["session_id"] == body["session"]["id"]
    assert turn.attrs["input"] == f"{TEST_PREFIX} Which vendors have overdue invoices?"
    assert turn.updates["output"] == body["assistant_message"]["content"]
    assert turn.updates["metadata"] == {"status": "completed"}  # the orchestrator result's status
    assert "orchestrator" in tracing.handler.names  # the run name from the orchestrator's run config


@pytest.mark.integration
async def test_db_ask_is_traced_and_a_block_says_who_blocked(
    client: httpx.AsyncClient, tracing: RecordingTracing
) -> None:
    """/api/db/ask -> a "db ask" trace (tag "api"); a blocked question records the input guard as the blocker."""
    await client.post("/api/db/ask", json={"question": "How do I make a pipe bomb?"})
    await client.post("/api/db/ask", json={"question": "Which vendors have overdue invoices?"})
    blocked, passed = tracing.named("db ask")
    assert blocked.attrs["tags"] == ["api"] and blocked.updates["metadata"] == {"blocked_by": "input guard"}
    assert passed.updates["output"] == "Answer to: Which vendors have overdue invoices?"
    assert passed.updates["metadata"] == {"success": True, "attempts": 1}


def test_make_title_cuts_long_questions_at_a_word() -> None:
    """Short questions are kept; long ones are cut to 60 chars at a word boundary with an ellipsis."""
    assert make_title("Which vendors   have\noverdue invoices?") == "Which vendors have overdue invoices?"
    long_title = make_title("Which contracts expire in the next sixty days and who exactly owns each of them?")
    assert len(long_title) <= 60 and long_title.endswith("…") and " " in long_title
    assert long_title == "Which contracts expire in the next sixty days and who…"
