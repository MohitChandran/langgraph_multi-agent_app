"""Postgres storage for chat sessions and messages (schema `app`, user `app_rw`).

Every chat platform keeps conversations on the SERVER, one row per message: the browser
is only a view, so a reload or another tab shows the same chats, and the agent's memory
is read from the same rows the UI shows (one source of truth).

This pool is deliberately different from the DB agent's ReadOnlyDatabase:
  - it logs in as app_rw, which owns schema `app` and has no rights on the business tables;
  - autocommit is ON: each single statement commits immediately (no lingering transaction),
    and multi-statement writes are grouped explicitly with `async with conn.transaction()`.
    (The read-only pool keeps autocommit OFF, because read-only mode only works inside
    the `BEGIN READ ONLY` transactions psycopg starts. See database.py.)

Tables are created here at startup with CREATE TABLE IF NOT EXISTS: running it on every
start is harmless (idempotent), so we need no migration framework yet.

Every query below reads through one of the two indexes, so each costs a handful of row
reads whether there are 10 chats or 10,000 (plan 01, section 9).
"""

from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from app.chat.models import Message, MessagePage, Session

# Table names are always written schema-qualified (app.chat_sessions), even though app_rw's
# search_path is `app`: if the code were ever pointed at the wrong user, tables could then
# never be created silently in `public`, where the DB agent could read them.
SCHEMA_STATEMENTS = (
    # One row per conversation (one sidebar item). Its id is used everywhere: URL, API,
    # and later as the LangGraph thread_id.
    """
    CREATE TABLE IF NOT EXISTS app.chat_sessions (
        id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        title          TEXT NOT NULL,
        message_count  INTEGER NOT NULL DEFAULT 0,
        active_run_at  TIMESTAMPTZ,
        created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "COMMENT ON COLUMN app.chat_sessions.active_run_at IS 'Set while a reply is being generated, else NULL.'",
    "COMMENT ON COLUMN app.chat_sessions.updated_at IS 'Last activity; the sidebar is sorted by it.'",
    # The sidebar query (ORDER BY updated_at DESC LIMIT 100) reads this index in order: no sort needed.
    "CREATE INDEX IF NOT EXISTS chat_sessions_recent_idx ON app.chat_sessions (updated_at DESC)",
    # Append-only transcript. A BIGINT identity gives a strict order; two timestamps can tie.
    """
    CREATE TABLE IF NOT EXISTS app.chat_messages (
        id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        session_id  UUID NOT NULL REFERENCES app.chat_sessions(id) ON DELETE CASCADE,
        role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content     TEXT NOT NULL DEFAULT '',
        status      TEXT NOT NULL DEFAULT 'complete' CHECK (status IN ('pending', 'complete', 'error')),
        payload     JSONB NOT NULL DEFAULT '{}',
        latency_ms  INTEGER,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "COMMENT ON COLUMN app.chat_messages.content IS 'The question, or the final answer (markdown).'",
    "COMMENT ON COLUMN app.chat_messages.payload IS 'Assistant extras: agent name and its result (SQL, rows, steps).'",
    # Opening a chat and loading memory both read "messages of ONE session, by id": this index serves both.
    "CREATE INDEX IF NOT EXISTS chat_messages_session_idx ON app.chat_messages (session_id, id)",
)

STALE_PENDING_MINUTES = 10  # a reply still 'pending' after this long was interrupted (API crash/restart)
CLAIM_TIMEOUT_MINUTES = 5  # a "busy" flag older than this belongs to a crashed run and can be taken over
MAX_SESSIONS_LISTED = 100

# Column lists shared by the queries below, so every query returns exactly the model's fields.
SESSION_COLUMNS = "id, title, message_count, created_at, updated_at, active_run_at IS NOT NULL AS is_running"
MESSAGE_COLUMNS = "id, role, content, status, payload, latency_ms, created_at"


class ChatStore:
    """Pool of connections to schema `app`, plus the queries for chat sessions and messages.

    Use it like ReadOnlyDatabase:

        async with ChatStore(url) as store:
            await store.setup()

    It's a class (not module-level functions with a global pool) so its owner controls
    its lifetime: the FastAPI lifespan opens one at startup, each test opens its own.
    """

    def __init__(self, url: str, max_connections: int = 5) -> None:
        """Prepare the pool without connecting yet (connecting happens in open())."""
        self._pool = AsyncConnectionPool(
            conninfo=url,
            min_size=1,
            max_size=max_connections,
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=False,
        )

    async def open(self) -> None:
        """Open the pool and wait until a connection works (fails fast on a wrong URL/password)."""
        await self._pool.open(wait=True, timeout=10)

    async def close(self) -> None:
        """Close all connections in the pool."""
        await self._pool.close()

    async def __aenter__(self) -> "ChatStore":
        """Open the pool when entering an `async with` block."""
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the pool when leaving the `async with` block, even after an error."""
        await self.close()

    # ------------------------------------------------------------ setup

    async def setup(self) -> None:
        """Create the chat tables and indexes if they don't exist yet. Safe to run on every startup.

        All statements run in ONE transaction: either everything is created or nothing is.
        If schema `app` is missing, the database was created before 03_app_schema.sh
        existed, and the error says how to fix it.
        """
        try:
            async with self._pool.connection() as conn, conn.transaction():
                for statement in SCHEMA_STATEMENTS:
                    await conn.execute(statement)
        except psycopg.errors.InvalidSchemaName as error:
            raise RuntimeError(
                "Schema 'app' does not exist. Recreate the database once with ./scripts/reset_db.sh "
                "(it runs scripts/db/03_app_schema.sh)."
            ) from error

    async def fail_stale_pending(self, older_than_minutes: int = STALE_PENDING_MINUTES) -> int:
        """Mark assistant replies stuck in 'pending' as 'error'. Returns how many were fixed.

        A reply is saved as 'pending' BEFORE the agent runs (so a reload shows "thinking").
        If the API crashed or restarted mid-run, that row would stay pending forever and the
        UI would wait for nothing. Called once at startup.
        """
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                UPDATE app.chat_messages
                SET status = 'error', content = 'The reply was interrupted.'
                WHERE status = 'pending' AND created_at < now() - make_interval(mins => %s)
                """,
                (older_than_minutes,),
            )
            return cursor.rowcount

    # --------------------------------------------------------- sessions

    async def create_session(self, title: str) -> Session:
        """Insert a new, empty conversation and return it. Postgres generates the UUID."""
        row = await self.fetch_one(
            f"INSERT INTO app.chat_sessions (title) VALUES (%s) RETURNING {SESSION_COLUMNS}", (title,)
        )
        return Session(**row)

    async def list_sessions(self, limit: int = MAX_SESSIONS_LISTED) -> list[Session]:
        """Return conversations, most recently active first (the sidebar)."""
        rows = await self._fetch_all(
            f"SELECT {SESSION_COLUMNS} FROM app.chat_sessions ORDER BY updated_at DESC LIMIT %s", (limit,)
        )
        return [Session(**row) for row in rows]

    async def get_session(self, session_id: UUID) -> Session | None:
        """Return one conversation, or None if it doesn't exist."""
        row = await self.fetch_one(f"SELECT {SESSION_COLUMNS} FROM app.chat_sessions WHERE id = %s", (session_id,))
        return Session(**row) if row else None

    async def rename_session(self, session_id: UUID, title: str) -> Session | None:
        """Change a conversation's title. Returns None if it doesn't exist.

        Deliberately doesn't touch updated_at: renaming isn't "activity", so the
        chat shouldn't jump to the top of the sidebar.
        """
        row = await self.fetch_one(
            f"UPDATE app.chat_sessions SET title = %s WHERE id = %s RETURNING {SESSION_COLUMNS}", (title, session_id)
        )
        return Session(**row) if row else None

    async def delete_session(self, session_id: UUID) -> bool:
        """Delete a conversation; its messages go with it (ON DELETE CASCADE). False if it didn't exist."""
        row = await self.fetch_one("DELETE FROM app.chat_sessions WHERE id = %s RETURNING id", (session_id,))
        return row is not None

    async def claim_session(self, session_id: UUID) -> Session | None:
        """Mark the conversation busy, so only one reply is generated at a time. None = couldn't claim.

        ONE conditional UPDATE does the check and the change together (atomically): if two
        requests race, Postgres lets exactly one of them match the WHERE clause. A separate
        "SELECT, then UPDATE" would let both see "not busy" and both proceed.
        A claim older than CLAIM_TIMEOUT_MINUTES belongs to a crashed run, so it may be taken over.
        """
        row = await self.fetch_one(
            f"""
            UPDATE app.chat_sessions SET active_run_at = now()
            WHERE id = %s
              AND (active_run_at IS NULL OR active_run_at < now() - make_interval(mins => %s))
            RETURNING {SESSION_COLUMNS}
            """,
            (session_id, CLAIM_TIMEOUT_MINUTES),
        )
        return Session(**row) if row else None

    async def release_session(self, session_id: UUID) -> Session | None:
        """Clear the busy flag and bump updated_at (the chat moves to the top of the sidebar)."""
        row = await self.fetch_one(
            f"UPDATE app.chat_sessions SET active_run_at = NULL, updated_at = now() WHERE id = %s"
            f" RETURNING {SESSION_COLUMNS}",
            (session_id,),
        )
        return Session(**row) if row else None

    # --------------------------------------------------------- messages

    async def add_turn(self, session_id: UUID, question: str) -> tuple[Message, Message]:
        """Save the question and an empty 'pending' answer, in ONE transaction. Returns both messages.

        Saving BEFORE the agent runs means a closed tab or a crash never loses the question,
        and the pending row tells a reloaded UI that the answer is still coming.
        The transaction makes the three writes all-or-nothing: no question without its
        answer slot, and message_count always matches the real number of messages.
        """
        async with self._pool.connection() as conn, conn.transaction():
            user = await (
                await conn.execute(
                    f"INSERT INTO app.chat_messages (session_id, role, content) VALUES (%s, 'user', %s)"
                    f" RETURNING {MESSAGE_COLUMNS}",
                    (session_id, question),
                )
            ).fetchone()
            assistant = await (
                await conn.execute(
                    f"INSERT INTO app.chat_messages (session_id, role, status) VALUES (%s, 'assistant', 'pending')"
                    f" RETURNING {MESSAGE_COLUMNS}",
                    (session_id,),
                )
            ).fetchone()
            await conn.execute(
                "UPDATE app.chat_sessions SET message_count = message_count + 2, updated_at = now() WHERE id = %s",
                (session_id,),
            )
        return Message(**user), Message(**assistant)

    async def complete_message(
        self, message_id: int, content: str, payload: dict[str, Any], latency_ms: int
    ) -> Message | None:
        """Fill in a pending answer: its text, the agent's result (payload) and how long it took.

        Returns None if the message no longer exists (its chat was deleted while the agent ran).
        """
        row = await self.fetch_one(
            f"UPDATE app.chat_messages SET content = %s, payload = %s, latency_ms = %s, status = 'complete'"
            f" WHERE id = %s RETURNING {MESSAGE_COLUMNS}",
            (content, Jsonb(payload), latency_ms, message_id),
        )
        return Message(**row) if row else None

    async def fail_message(self, message_id: int, content: str, latency_ms: int) -> Message | None:
        """Mark a pending answer as failed, with a short readable error as its text. None if it's gone."""
        row = await self.fetch_one(
            f"UPDATE app.chat_messages SET content = %s, latency_ms = %s, status = 'error'"
            f" WHERE id = %s RETURNING {MESSAGE_COLUMNS}",
            (content, latency_ms, message_id),
        )
        return Message(**row) if row else None

    async def list_messages(self, session_id: UUID, limit: int = 100, before: int | None = None) -> MessagePage:
        """Return up to `limit` messages, oldest to newest. With `before`, only messages older than that id.

        The "LIMIT n + 1" trick: we fetch one extra row. If it comes back, there are older
        messages (has_more=True) and we drop it; no COUNT(*) query needed.
        Rows are read newest-first (fast, via the index) and reversed for display.
        """
        if before is None:
            query = f"SELECT {MESSAGE_COLUMNS} FROM app.chat_messages WHERE session_id = %s ORDER BY id DESC LIMIT %s"
            params: tuple[Any, ...] = (session_id, limit + 1)
        else:
            query = (
                f"SELECT {MESSAGE_COLUMNS} FROM app.chat_messages WHERE session_id = %s AND id < %s"
                f" ORDER BY id DESC LIMIT %s"
            )
            params = (session_id, before, limit + 1)
        rows = await self._fetch_all(query, params)
        has_more = len(rows) > limit
        return MessagePage(items=[Message(**row) for row in reversed(rows[:limit])], has_more=has_more)

    async def recent_complete_messages(self, session_id: UUID, limit: int) -> list[dict[str, Any]]:
        """The newest `limit` COMPLETE messages of a session, newest first: the raw material for memory.

        Returns only what memory needs (role, content, the SQL from the payload), never rows.
        Failed or pending answers are excluded so they don't pollute the agent's memory.
        """
        return await self._fetch_all(
            """
            SELECT role, content, payload->'result'->>'sql' AS sql
            FROM app.chat_messages
            WHERE session_id = %s AND status = 'complete'
            ORDER BY id DESC
            LIMIT %s
            """,
            (session_id, limit),
        )

    # ---------------------------------------------------------- helpers

    async def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        """Run one statement and return its first row as a dict (or None). Also used by tests."""
        async with self._pool.connection() as conn:
            cursor = await conn.execute(query, params)
            return await cursor.fetchone()

    async def _fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Run one statement and return all rows as dicts."""
        async with self._pool.connection() as conn:
            cursor = await conn.execute(query, params)
            return await cursor.fetchall()
