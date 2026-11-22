"""Shared pytest fixtures for the email service.

Integration fixtures need the local Postgres (docker compose) and this service's .env; when either is missing the
test is SKIPPED, not failed (e.g. in CI).
"""

import os

# CrewAI calls load_dotenv() when it is imported (crewai/llm.py): this service's .env lands in os.environ, where every
# Settings(...) sees it, even Settings(_env_file=None). Tests must never send spans to Langfuse, so the tracing keys
# are pinned to empty FIRST (load_dotenv never overrides a variable that already exists).
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

from collections.abc import AsyncIterator  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncEngine  # noqa: E402

from app.config import Settings, get_settings  # noqa: E402
from app.store import create_engine  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]


def settings_or_skip() -> Settings:
    """The settings, or skip the test when .env is missing or incomplete."""
    try:
        return get_settings()
    except Exception as error:  # pydantic ValidationError: required settings missing
        pytest.skip(f"Settings not available: {type(error).__name__}")


def root_env(name: str) -> str:
    """A value from the repo-root .env (the OTHER roles' URLs, for the isolation test); skip if absent."""
    path = REPO_ROOT / ".env"
    for line in path.read_text().splitlines() if path.exists() else []:
        key, _, value = line.partition("=")
        if key.strip() == name:
            return value.strip()
    pytest.skip(f"{name} not in the repo-root .env")


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    """An async engine as email_rw, disposed after the test; skips when Postgres can't be reached."""
    engine = create_engine(settings_or_skip())
    try:
        async with engine.connect():
            pass
    except Exception as error:  # connection refused, wrong password, missing role...
        await engine.dispose()
        pytest.skip(f"Postgres not available: {type(error).__name__}")
    try:
        yield engine
    finally:
        await engine.dispose()
