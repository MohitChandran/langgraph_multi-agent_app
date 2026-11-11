"""Shared pytest fixtures.

pytest automatically loads this file and makes its fixtures available to every
test in this folder: a test just names `db` or `chat_store` as a parameter to receive one.

Both fixtures need the local Postgres. If there is no .env or Postgres isn't running
(e.g. in CI), the test is SKIPPED instead of failing: run ./scripts/reset_db.sh first.
"""

from collections.abc import AsyncIterator

import pytest

from app.agents.db.database import ReadOnlyDatabase
from app.chat.store import ChatStore
from app.config import Settings, get_settings


def settings_or_skip() -> Settings:
    """Return the settings, or skip the test if .env is missing/incomplete."""
    try:
        return get_settings()
    except Exception as error:  # pydantic ValidationError: required settings missing
        pytest.skip(f"Settings not available: {type(error).__name__}")


async def open_or_skip(resource: ReadOnlyDatabase | ChatStore) -> None:
    """Open a connection pool, or skip the test if Postgres can't be reached."""
    try:
        await resource.open()
    except Exception as error:  # connection refused, wrong password, timeout, ...
        pytest.skip(f"Postgres not available: {type(error).__name__}")


@pytest.fixture
async def db() -> AsyncIterator[ReadOnlyDatabase]:
    """An open read-only connection pool (user readonly_agent), closed after the test."""
    database = ReadOnlyDatabase(settings_or_skip().readonly_database_url)
    await open_or_skip(database)
    try:
        yield database
    finally:
        await database.close()


@pytest.fixture
async def chat_store() -> AsyncIterator[ChatStore]:
    """An open ChatStore (user app_rw) with the chat tables created, closed after the test.

    setup() is NOT wrapped in a skip: if Postgres is up but schema `app` is missing,
    that's a real problem to fix (run ./scripts/reset_db.sh), so the test should fail.
    """
    store = ChatStore(settings_or_skip().app_database_url)
    await open_or_skip(store)
    try:
        await store.setup()
        yield store
    finally:
        await store.close()
