"""Shared pytest fixtures.

pytest automatically loads this file and makes its fixtures available to every
test in this folder: a test just names `db` as a parameter to receive one.
"""

from collections.abc import AsyncIterator

import pytest

from app.agents.db.database import ReadOnlyDatabase
from app.config import get_settings


@pytest.fixture
async def db() -> AsyncIterator[ReadOnlyDatabase]:
    """An open read-only connection pool to the local Postgres, closed after the test.

    If there is no .env or Postgres isn't running (e.g. in CI), the test is SKIPPED
    instead of failing: integration tests need `./scripts/reset_db.sh` first.
    """
    try:
        url = get_settings().readonly_database_url
        database = ReadOnlyDatabase(url)
        await database.open()
    except Exception as error:  # missing settings, connection refused, timeout, ...
        pytest.skip(f"Postgres not available: {type(error).__name__}")
    try:
        yield database
    finally:
        await database.close()
