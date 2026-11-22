"""Tests for the chat storage foundations (app/chat/store.py) - plan 01, stage S1.

Most important: the SECURITY boundary between the two database users.
  - readonly_agent (runs LLM-written SQL) must not be able to read chat history;
  - app_rw (runs the chat code) must not be able to read the business tables.
Each test that creates rows deletes them again.
"""

import psycopg
import pytest

from app.agents.db.database import ReadOnlyDatabase
from app.agents.db.safety import SQLValidationError, validate_sql
from app.chat.store import ChatStore


@pytest.mark.integration
async def test_setup_is_idempotent_and_tables_live_in_app_schema(chat_store: ChatStore) -> None:
    """Running setup() again is harmless, and the tables exist in schema app, not in public."""
    await chat_store.setup()  # the fixture already ran it once

    row = await chat_store.fetch_one(
        "SELECT to_regclass('app.chat_sessions') IS NOT NULL AS sessions,"
        "       to_regclass('app.chat_messages') IS NOT NULL AS messages,"
        "       to_regclass('public.chat_sessions') IS NULL AND to_regclass('public.chat_messages') IS NULL"
        "       AS none_in_public"
    )
    assert row == {"sessions": True, "messages": True, "none_in_public": True}


@pytest.mark.integration
async def test_db_agent_user_cannot_read_chat_tables(chat_store: ChatStore, db: ReadOnlyDatabase) -> None:
    """Layer 1: even SQL that skips the safety gate can't read chats as readonly_agent."""
    with pytest.raises(psycopg.errors.InsufficientPrivilege, match="permission denied for schema app"):
        await db.fetch_trusted("SELECT content FROM app.chat_messages")


def test_safety_gate_rejects_chat_tables() -> None:
    """Layer 2: the safety gate refuses schema app before the query reaches Postgres."""
    with pytest.raises(SQLValidationError, match="Schema 'app' is not allowed"):
        validate_sql("SELECT content FROM app.chat_messages")


@pytest.mark.integration
async def test_chat_user_cannot_read_business_tables(chat_store: ChatStore) -> None:
    """The other direction: app_rw has no rights on the business data in public."""
    with pytest.raises(psycopg.errors.InsufficientPrivilege, match="permission denied for table invoices"):
        await chat_store.fetch_one("SELECT invoice_number FROM public.invoices")


@pytest.mark.integration
async def test_fail_stale_pending_only_touches_old_pending_replies(chat_store: ChatStore) -> None:
    """A reply pending for over 10 minutes becomes 'error'; a fresh pending reply is left alone."""
    session = await chat_store.fetch_one("INSERT INTO app.chat_sessions (title) VALUES ('test: stale') RETURNING id")
    try:
        insert_reply = (
            "INSERT INTO app.chat_messages (session_id, role, status, created_at)"
            " VALUES (%s, 'assistant', 'pending', now() - make_interval(mins => %s)) RETURNING id"
        )
        old = await chat_store.fetch_one(insert_reply, (session["id"], 11))
        fresh = await chat_store.fetch_one(insert_reply, (session["id"], 1))

        assert await chat_store.fail_stale_pending() >= 1

        read = "SELECT status, content FROM app.chat_messages WHERE id = %s"
        assert await chat_store.fetch_one(read, (old["id"],)) == {
            "status": "error",
            "content": "The reply was interrupted.",
        }
        assert (await chat_store.fetch_one(read, (fresh["id"],)))["status"] == "pending"
    finally:
        await chat_store.fetch_one("DELETE FROM app.chat_sessions WHERE id = %s RETURNING id", (session["id"],))


# ------------------------------------------------------------------ email approval (plan 03, 11.3)


@pytest.mark.integration
async def test_awaiting_approval_status_and_its_queries(chat_store: ChatStore) -> None:
    """The migrated constraint accepts 'awaiting_approval' (setup stays idempotent), and the queries find it."""
    await chat_store.setup()  # the constraint migration is a no-op the second time
    session = await chat_store.create_session("[pytest] awaiting approval")
    try:
        _, answer = await chat_store.add_turn(session.id, "Email the list to accounts@company.example")
        assert not await chat_store.has_awaiting_approval(session.id)

        waiting = await chat_store.mark_awaiting_approval(answer.id, "I drafted an email.", {"agent": "x"}, 12)
        assert waiting.status == "awaiting_approval" and waiting.content == "I drafted an email."
        assert await chat_store.has_awaiting_approval(session.id)
        assert answer.id in await chat_store.awaiting_approval_message_ids()
        assert await chat_store.get_message(answer.id) == (session.id, waiting)
        assert await chat_store.get_message(-1) is None

        with pytest.raises(psycopg.errors.CheckViolation):  # other statuses are still refused
            await chat_store.fetch_one("UPDATE app.chat_messages SET status = 'paused' WHERE id = %s", (answer.id,))
    finally:
        await chat_store.delete_session(session.id)


@pytest.mark.integration
async def test_fail_stale_pending_never_touches_answers_awaiting_approval(chat_store: ChatStore) -> None:
    """An answer waiting hours for an email decision isn't 'interrupted': the user can still approve it."""
    session = await chat_store.create_session("[pytest] stale approval")
    try:
        _, answer = await chat_store.add_turn(session.id, "q")
        await chat_store.mark_awaiting_approval(answer.id, "I drafted an email.", {}, 1)
        await chat_store.fetch_one(
            "UPDATE app.chat_messages SET created_at = now() - interval '2 hours' WHERE id = %s RETURNING id",
            (answer.id,),
        )
        await chat_store.fail_stale_pending()
        assert (await chat_store.get_message(answer.id))[1].status == "awaiting_approval"
    finally:
        await chat_store.delete_session(session.id)


def test_nul_characters_are_removed_before_saving() -> None:
    """Postgres can't store NUL in TEXT/JSONB: without_nul cleans strings anywhere in a payload (plan 05 eval find)."""
    from app.chat.store import without_nul

    payload = {"email": {"subject": "Due (Nov 16\x0021)", "to": ["a@b.example"]}, "n": 3, "list": ["x\x00y", None]}
    assert without_nul(payload) == {"email": {"subject": "Due (Nov 1621)", "to": ["a@b.example"]}, "n": 3,
                                    "list": ["xy", None]}  # fmt: skip
    assert without_nul("a\x00b") == "ab" and without_nul(None) is None


@pytest.mark.integration
async def test_an_answer_with_a_nul_character_is_saved(chat_store) -> None:  # noqa: ANN001
    """The bug the eval found: a NUL inside the payload no longer fails complete_message."""
    session = await chat_store.create_session("[pytest] nul")
    try:
        _, pending = await chat_store.add_turn(session.id, "question\x00")
        saved = await chat_store.complete_message(pending.id, "answer\x00", {"subject": "a\x00b"}, 5)
        assert saved.content == "answer" and saved.payload == {"subject": "ab"}
    finally:
        await chat_store.delete_session(session.id)
