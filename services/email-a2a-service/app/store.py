"""Where the A2A tasks live: Postgres schema `email`, table `email.tasks` (plan 05, section 6.6).

An A2A task is the email service's memory of one email: the request, the draft (an artifact) and its state
(WORKING -> INPUT_REQUIRED -> COMPLETED / REJECTED / FAILED). The draft waits for a human for minutes or hours, and
the service may restart meanwhile, so the task must be in a database, not in memory: a2a-sdk's DatabaseTaskStore
saves every task as one row (SQLAlchemy, asyncpg driver).

The role email_rw (scripts/db/04_email_schema.sh) only has schema `email`, and its search_path is `email`, so the
SDK's plain table name "tasks" becomes email.tasks.

Check the setup (from services/email-a2a-service):
    uv run python -m app.store
"""

import asyncio

from a2a.server.tasks import DatabaseTaskStore
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import Settings, get_settings


def create_engine(settings: Settings) -> AsyncEngine:
    """The async connection pool for email_rw. pool_pre_ping replaces connections Postgres dropped (e.g. a restart)."""
    return create_async_engine(settings.email_database_url.get_secret_value(), pool_size=5, pool_pre_ping=True)


def build_task_store(engine: AsyncEngine) -> DatabaseTaskStore:
    """The SDK's task store on our engine (no connection yet: the engine connects on first use)."""
    return DatabaseTaskStore(engine, create_table=True, table_name="tasks")


async def open_task_store(engine: AsyncEngine) -> DatabaseTaskStore:
    """The task store with email.tasks created NOW.

    The SDK would create the table lazily on the first request; doing it at startup means a missing schema or a wrong
    password stops the service immediately instead of failing the first email.
    """
    store = build_task_store(engine)
    await store.initialize()
    return store


async def describe(engine: AsyncEngine) -> dict[str, object]:
    """Who we are connected as, where unqualified tables go, and how many tasks are stored (for the CLI)."""
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT current_user, current_schema(), to_regclass('email.tasks')::text, count(*) FROM tasks")
            )
        ).one()
    return {"user": row[0], "schema": row[1], "table": row[2], "tasks": row[3]}


async def _main() -> None:
    """CLI: create the table if needed and show where it is."""
    engine = create_engine(get_settings())
    try:
        await open_task_store(engine)
        print(await describe(engine))
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
