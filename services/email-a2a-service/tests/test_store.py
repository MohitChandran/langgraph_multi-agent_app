"""Tests for the task storage (plan 05, stage E1): schema `email`, table `email.tasks`, nobody else can read it."""

import asyncpg
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.store import describe, open_task_store
from tests.conftest import root_env


@pytest.mark.integration
async def test_the_task_table_is_created_in_schema_email(engine: AsyncEngine) -> None:
    """open_task_store creates the SDK's table; as email_rw, unqualified names land in schema `email`."""
    await open_task_store(engine)
    info = await describe(engine)
    assert info["user"] == "email_rw" and info["schema"] == "email"
    async with engine.connect() as conn:
        schema = (await conn.execute(text("SELECT table_schema FROM information_schema.tables "
                                          "WHERE table_name = 'tasks'"))).scalars().all()  # fmt: skip
    assert schema == ["email"]


@pytest.mark.integration
@pytest.mark.parametrize("url_setting", ["READONLY_DATABASE_URL", "APP_DATABASE_URL"])
async def test_the_other_roles_cannot_read_the_drafts(engine: AsyncEngine, url_setting: str) -> None:
    """readonly_agent (LLM-written SQL) and app_rw (chats) get "permission denied" on email.tasks."""
    await open_task_store(engine)
    conn = await asyncpg.connect(root_env(url_setting))
    try:
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.fetch("SELECT * FROM email.tasks")
    finally:
        await conn.close()
