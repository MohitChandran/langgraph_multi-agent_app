"""Where a paused question waits for the human: the Postgres checkpointer (plan 03, section 10).

When the graph pauses for email approval, LangGraph saves its whole state (messages, calls, budget,
the draft) as a "checkpoint". Kept in memory, it would be lost when the API restarts, and the user's
Approve button would hit nothing. AsyncPostgresSaver writes it to Postgres, so the pause survives.

    async with CheckpointStore(settings.app_database_url) as checkpoints:
        await checkpoints.setup()                     # creates/migrates its tables (idempotent)
        orchestrator = Orchestrator(tools, checkpoints.saver)

Design choices (plan D7):
  - ONE thread per turn, "<session_id>:<message_id>", written only when a run pauses or stops
    (durability="exit") and deleted when the turn finishes. Memory lives in chat_messages, so
    checkpoints only ever hold paused turns.
  - The tables land in schema `app`: app_rw's search_path is `app, public`, and readonly_agent (the
    DB agent's LLM-written SQL) has no access to schema app, so it can never read a checkpoint.
  - Its OWN pool: the saver needs autocommit=True (its setup runs CREATE INDEX CONCURRENTLY, which
    can't run inside a transaction) and prepare_threshold=0 (no server-side prepared statements).
  - delete_orphans() at startup removes threads that no longer belong to a message waiting for
    approval (a chat deleted while paused, a crash right after a turn finished).
"""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


class CheckpointStore:
    """A connection pool to schema `app` plus the LangGraph Postgres checkpointer that uses it.

    Opened and closed like ChatStore / DocumentStore (`async with`), so its owner (the FastAPI
    lifespan, a test, the CLI) controls its lifetime.
    """

    def __init__(self, url: str, max_connections: int = 3) -> None:
        """Prepare the pool (not connected yet) and the saver on top of it."""
        self._pool = AsyncConnectionPool(
            conninfo=url,
            min_size=1,
            max_size=max_connections,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=False,
        )
        self.saver = AsyncPostgresSaver(self._pool)

    async def open(self) -> None:
        """Open the pool and wait until a connection works (fails fast on a wrong URL/password)."""
        await self._pool.open(wait=True, timeout=10)

    async def close(self) -> None:
        """Close all connections in the pool."""
        await self._pool.close()

    async def __aenter__(self) -> "CheckpointStore":
        """Open the pool when entering an `async with` block."""
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the pool when leaving the `async with` block, even after an error."""
        await self.close()

    async def setup(self) -> None:
        """Create or migrate the checkpoint tables (checkpoints, checkpoint_blobs, checkpoint_writes,
        checkpoint_migrations). Safe on every startup: LangGraph records which migrations already ran."""
        await self.saver.setup()

    async def thread_ids(self) -> list[str]:
        """Every checkpoint thread currently saved (= every turn paused, or left behind by a crash)."""
        async with self._pool.connection() as conn:
            rows = await (await conn.execute("SELECT DISTINCT thread_id FROM app.checkpoints")).fetchall()
        return [row["thread_id"] for row in rows]

    async def delete_orphans(self, awaiting_message_ids: set[int]) -> int:
        """Delete every thread whose message isn't waiting for approval any more. Returns how many.

        `awaiting_message_ids` = ids of the chat messages with status 'awaiting_approval'
        (ChatStore.awaiting_approval_message_ids). A thread id that doesn't end in such an id
        belongs to nothing the user can still approve.
        """
        deleted = 0
        for thread_id in await self.thread_ids():
            if message_id_of(thread_id) not in awaiting_message_ids:
                await self.saver.adelete_thread(thread_id)
                deleted += 1
        return deleted


def message_id_of(thread_id: str) -> int | None:
    """The chat message id at the end of a thread id: "9f1c...:42" -> 42. None if it isn't a number."""
    tail = thread_id.rpartition(":")[2]
    return int(tail) if tail.isdigit() else None
