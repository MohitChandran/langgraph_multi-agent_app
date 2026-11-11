"""API tests for chat sessions (plan 01, stage S2): real routes + real Postgres, FAKE agent.

How the app is wired here:
  - httpx.AsyncClient + ASGITransport sends HTTP requests straight into the FastAPI app
    in memory (no server, no port).
  - ASGITransport does NOT run the lifespan, so app.state is empty. We don't need it:
    dependency_overrides hands the endpoints our test ChatStore and a FakeAgent instead.
  - Every test message starts with "[pytest]", so its chat title does too, and the
    fixture deletes exactly those chats afterwards (your real chats are never touched).
"""

from collections.abc import AsyncIterator
from uuid import uuid4

import httpx
import pytest

from app.agents.db.models import DBAgentResult
from app.chat.service import make_title
from app.chat.store import ChatStore
from app.dependencies import get_chat_store, get_db_agent
from app.main import app

TEST_PREFIX = "[pytest]"


class FakeAgent:
    """Stands in for DBAgent: answers instantly, records every call, can be told to crash."""

    def __init__(self) -> None:
        """Start with no recorded calls and crashing switched off."""
        self.calls: list[dict] = []
        self.crash = False

    async def ask(self, question: str, history: list | None = None) -> DBAgentResult:
        """Record the question (and the memory it was given), then answer or raise."""
        self.calls.append({"question": question, "history": history})
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


@pytest.fixture
def fake_agent() -> FakeAgent:
    """A fresh FakeAgent per test."""
    return FakeAgent()


@pytest.fixture
async def client(chat_store: ChatStore, fake_agent: FakeAgent) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client for the app, with the test store and fake agent injected; cleans up test chats."""
    app.dependency_overrides[get_chat_store] = lambda: chat_store
    app.dependency_overrides[get_db_agent] = lambda: fake_agent
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
    assert assistant["status"] == "complete" and assistant["content"].startswith("Answer to:")
    assert assistant["payload"]["agent"] == "db" and assistant["payload"]["result"]["sql"].startswith("SELECT")
    assert "answer" not in assistant["payload"]["result"]  # the answer is the content, not duplicated
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
async def test_agent_crash_is_saved_as_error_and_the_chat_keeps_working(
    client: httpx.AsyncClient, fake_agent: FakeAgent
) -> None:
    """A crashing agent -> the answer is saved with status 'error', the claim is released, the next send works."""
    fake_agent.crash = True
    crashed = await send(client, "this will crash")
    assert crashed.status_code == 200
    body = crashed.json()
    assert body["assistant_message"]["status"] == "error"
    assert "something went wrong" in body["assistant_message"]["content"]
    assert body["session"]["is_running"] is False  # released in `finally`

    fake_agent.crash = False
    retry = await send(client, "try again", body["session"]["id"])
    assert retry.status_code == 200 and retry.json()["assistant_message"]["status"] == "complete"


# ------------------------------------------------------------------ 404 / 422


@pytest.mark.integration
async def test_unknown_chat_returns_404_everywhere(client: httpx.AsyncClient) -> None:
    """Every endpoint that takes a session id answers 404 for one that doesn't exist."""
    unknown = str(uuid4())
    assert (await send(client, "hello?", unknown)).status_code == 404
    assert (await client.get(f"/api/sessions/{unknown}/messages")).status_code == 404
    assert (await client.patch(f"/api/sessions/{unknown}", json={"title": "x"})).status_code == 404
    assert (await client.delete(f"/api/sessions/{unknown}")).status_code == 404


@pytest.mark.integration
async def test_invalid_input_returns_422(client: httpx.AsyncClient) -> None:
    """Pydantic and FastAPI reject bad input before our code runs."""
    assert (await client.post("/api/chat", json={"content": ""})).status_code == 422
    assert (await client.post("/api/chat", json={"content": "x" * 1001})).status_code == 422
    assert (await client.get("/api/sessions/not-a-uuid/messages")).status_code == 422


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
async def test_failed_turns_are_not_remembered(client: httpx.AsyncClient, fake_agent: FakeAgent) -> None:
    """A turn whose answer failed (status 'error') is left out of memory."""
    session_id = (await send(client, "good question")).json()["session"]["id"]
    fake_agent.crash = True
    await send(client, "crashing question", session_id)
    fake_agent.crash = False
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


def test_make_title_cuts_long_questions_at_a_word() -> None:
    """Short questions are kept; long ones are cut to 60 chars at a word boundary with an ellipsis."""
    assert make_title("Which vendors   have\noverdue invoices?") == "Which vendors have overdue invoices?"
    long_title = make_title("Which contracts expire in the next sixty days and who exactly owns each of them?")
    assert len(long_title) <= 60 and long_title.endswith("…") and " " in long_title
    assert long_title == "Which contracts expire in the next sixty days and who…"
