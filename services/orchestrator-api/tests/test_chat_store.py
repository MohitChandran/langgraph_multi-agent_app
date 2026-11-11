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
