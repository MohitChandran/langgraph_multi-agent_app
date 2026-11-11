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
"""

from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

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

    async def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        """Run one query and return its first row (or None). Used by tests and later stages."""
        async with self._pool.connection() as conn:
            cursor = await conn.execute(query, params)
            return await cursor.fetchone()
