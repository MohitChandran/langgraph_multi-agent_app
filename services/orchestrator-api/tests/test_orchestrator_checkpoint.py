"""Integration tests for the Postgres checkpointer (app/orchestrator/checkpoint.py), plan 03 sections 10 and 17.

The promise: a question paused for email approval survives an API restart. These tests "restart" by
building a NEW CheckpointStore + Orchestrator on the same database and resuming there.
Real Postgres (skipped when it isn't running); a scripted fake LLM; no Groq.
"""

from uuid import uuid4

import langgraph.checkpoint.serde._msgpack as lg_msgpack
import psycopg
import pytest

from app.agents.db.database import ReadOnlyDatabase
from app.agents.email.dummy import DummyEmailAgent
from app.chat.store import ChatStore
from app.config import get_settings
from app.orchestrator.checkpoint import CheckpointStore, message_id_of
from app.orchestrator.graph import Orchestrator, thread_id_for
from app.orchestrator.models import EmailDecision
from tests.orchestrator_fakes import ScriptedLLM, answer, call
from tests.test_orchestrator_tools import make_tools

QUESTION = "Which vendors have overdue invoices? Email them to accounts@company.example"


def email_script() -> ScriptedLLM:
    """Data, then a draft to the user's address, then the final answer."""
    return ScriptedLLM(
        call("ask_database", question="Which vendors have overdue invoices?"),
        call("draft_email", to=["accounts@company.example"], subject="Overdue", content="3 vendors are late."),
        answer("Finished."),
    )


async def thread_exists(store: CheckpointStore, thread_id: str) -> bool:
    """True if the thread has a row in app.checkpoints."""
    return thread_id in await store.thread_ids()


@pytest.mark.integration
async def test_checkpoint_tables_live_in_schema_app(chat_store: ChatStore, checkpoint_store: CheckpointStore) -> None:
    """setup() is idempotent and creates the tables in schema app (app_rw's search_path), not in public."""
    await checkpoint_store.setup()
    row = await chat_store.fetch_one(
        "SELECT to_regclass('app.checkpoints') IS NOT NULL AS in_app,"
        "       to_regclass('public.checkpoints') IS NULL AS not_in_public"
    )
    assert row == {"in_app": True, "not_in_public": True}


@pytest.mark.integration
async def test_db_agent_user_cannot_read_checkpoints(checkpoint_store: CheckpointStore, db: ReadOnlyDatabase) -> None:
    """LLM-written SQL runs as readonly_agent, which has no access to schema app (where drafts are saved)."""
    with pytest.raises(psycopg.errors.InsufficientPrivilege, match="permission denied for schema app"):
        await db.fetch_trusted("SELECT thread_id FROM app.checkpoints")


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.integration
async def test_a_paused_turn_survives_a_restart(
    checkpoint_store: CheckpointStore, monkeypatch: pytest.MonkeyPatch, strict: bool
) -> None:
    """Pause with one Orchestrator, resume with a NEW Orchestrator and a NEW saver (= the API restarted).

    The email agent is shared, as a real email service would survive our restart, so the draft is sent.
    strict=True: LangGraph's msgpack allow-list blocks unknown classes; it still loads, because the
    state holds only plain JSON and LangChain messages (plan section 7).
    """
    monkeypatch.setattr(lg_msgpack, "STRICT_MSGPACK_ENABLED", strict)
    session_id, message_id = uuid4(), 900_000_000 + int(strict)
    thread_id = thread_id_for(session_id, message_id)
    tools = make_tools()
    email_agent = tools.email_agent

    paused = await Orchestrator(tools, checkpoint_store.saver, chat_model=email_script()).ask(
        QUESTION, session_id=session_id, message_id=message_id
    )
    assert paused.status == "awaiting_approval" and await thread_exists(checkpoint_store, thread_id)

    async with CheckpointStore(get_settings().app_database_url) as restarted:
        tools_after_restart = make_tools()
        tools_after_restart.email_agent = email_agent
        orchestrator = Orchestrator(tools_after_restart, restarted.saver, chat_model=ScriptedLLM(answer("Finished.")))
        done = await orchestrator.resume_email(session_id, message_id, EmailDecision(approved=True))

    assert done.status == "completed" and done.answer == "Finished."
    assert [c.status for c in done.calls] == ["ok", "sent"] and done.email["status"] == "sent"
    assert done.calls[0].result["rows"]  # the DB result came back from Postgres too
    assert not await thread_exists(checkpoint_store, thread_id)  # deleted once the turn completed


@pytest.mark.integration
async def test_after_a_restart_the_dummy_has_forgotten_the_draft_and_says_nothing_was_sent(
    checkpoint_store: CheckpointStore,
) -> None:
    """The dummy email agent keeps drafts in memory: after a restart, approving resumes the graph,
    but the draft is gone, so the call is an error and the supervisor is told nothing was sent."""
    session_id, message_id = uuid4(), 900_000_010
    await Orchestrator(make_tools(), checkpoint_store.saver, chat_model=email_script()).ask(
        QUESTION, session_id=session_id, message_id=message_id
    )

    restarted_tools = make_tools()
    restarted_tools.email_agent = DummyEmailAgent()  # a fresh dummy: no drafts
    llm = ScriptedLLM(answer("The email could not be sent."))
    done = await Orchestrator(restarted_tools, checkpoint_store.saver, chat_model=llm).resume_email(
        session_id, message_id, EmailDecision(approved=True)
    )
    assert done.status == "completed" and done.calls[-1].status == "error" and done.email["status"] == "error"
    assert llm.calls[0].messages[-1].content.startswith("NOT SENT: the email could not be sent (Unknown or already")


@pytest.mark.integration
async def test_orphan_cleanup_keeps_only_threads_of_messages_awaiting_approval(
    chat_store: ChatStore, checkpoint_store: CheckpointStore
) -> None:
    """At startup, threads whose message isn't 'awaiting_approval' any more are deleted; waiting ones stay."""
    session = await chat_store.create_session("[pytest] orphan cleanup")
    try:
        _, waiting = await chat_store.add_turn(session.id, QUESTION)
        await chat_store.mark_awaiting_approval(waiting.id, "Waiting.", {"agent": "orchestrator"}, 1)
        orphan_id = 999_999_999  # no such message
        for message_id in (waiting.id, orphan_id):
            await Orchestrator(make_tools(), checkpoint_store.saver, chat_model=email_script()).ask(
                QUESTION, session_id=session.id, message_id=message_id
            )
        assert await thread_exists(checkpoint_store, thread_id_for(session.id, orphan_id))

        deleted = await checkpoint_store.delete_orphans(await chat_store.awaiting_approval_message_ids())

        assert deleted >= 1
        assert await thread_exists(checkpoint_store, thread_id_for(session.id, waiting.id))
        assert not await thread_exists(checkpoint_store, thread_id_for(session.id, orphan_id))
    finally:
        await checkpoint_store.saver.adelete_thread(thread_id_for(session.id, waiting.id))
        await chat_store.delete_session(session.id)


def test_message_id_of_reads_the_number_after_the_last_colon() -> None:
    """Thread ids are "<session_id>:<message_id>"; anything else has no message id."""
    assert message_id_of("9f1c0e2a-0000-0000-0000-000000000000:42") == 42
    assert message_id_of("None:cli-1a2b3c4d") is None and message_id_of("no-colon") is None
