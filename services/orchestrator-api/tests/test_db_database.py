"""Tests for the DB agent's database access (app/agents/db/database.py).

Integration tests run against the local Postgres (skipped if it's not running).
The most important ones deliberately BYPASS the safety gate by building a
ValidatedQuery by hand, to prove the database layer blocks attacks on its own.
That's what defense in depth means: each layer works even if another one fails.
"""

from datetime import date
from decimal import Decimal

import pytest

from app.agents.db.database import QueryExecutionError, ReadOnlyDatabase, _to_json_value
from app.agents.db.safety import MAX_ROWS, ValidatedQuery, validate_sql


def unchecked(sql: str) -> ValidatedQuery:
    """Build a ValidatedQuery WITHOUT running the safety gate (only for tests that attack the DB layer)."""
    return ValidatedQuery(sql=sql, tables=frozenset())


def test_to_json_value_converts_database_types() -> None:
    """Decimal and date values become plain float / ISO strings; others are unchanged."""
    assert _to_json_value(Decimal("1520.50")) == 1520.5
    assert _to_json_value(date(2026, 11, 5)) == "2026-11-05"
    assert _to_json_value("pending") == "pending"
    assert _to_json_value(None) is None


@pytest.mark.integration
async def test_run_query_returns_json_friendly_rows(db: ReadOnlyDatabase) -> None:
    """A valid query returns columns in order and rows with JSON-friendly values."""
    result = await db.run_query(validate_sql("SELECT invoice_number, amount, due_date FROM invoices LIMIT 3"))
    assert result.columns == ["invoice_number", "amount", "due_date"]
    assert result.row_count == 3
    assert isinstance(result.rows[0]["amount"], float)
    assert isinstance(result.rows[0]["due_date"], str)
    assert result.hit_row_limit is False


@pytest.mark.integration
async def test_hit_row_limit_is_reported(db: ReadOnlyDatabase) -> None:
    """When the LIMIT cuts results off, hit_row_limit is True so the answer can say 'first 100'."""
    result = await db.run_query(validate_sql("SELECT invoice_number FROM invoices"))  # 300 rows exist
    assert result.row_count == MAX_ROWS
    assert result.hit_row_limit is True


@pytest.mark.integration
@pytest.mark.parametrize(
    "attack",
    ["DELETE FROM invoices", "UPDATE invoices SET status = 'paid'", "CREATE TABLE hack (x int)"],
)
async def test_database_blocks_writes_even_without_safety_gate(db: ReadOnlyDatabase, attack: str) -> None:
    """Layer 1 alone: even if validate_sql() were bypassed, Postgres refuses to write."""
    with pytest.raises(QueryExecutionError, match="read-only transaction"):
        await db.run_query(unchecked(attack))


@pytest.mark.integration
async def test_slow_query_is_cancelled_by_timeout(db: ReadOnlyDatabase) -> None:
    """Layer 3: the readonly_agent user's 5s statement_timeout kills runaway queries."""
    with pytest.raises(QueryExecutionError, match="statement timeout"):
        await db.run_query(unchecked("SELECT pg_sleep(6)"))


@pytest.mark.integration
async def test_postgres_hint_is_kept_for_repair(db: ReadOnlyDatabase) -> None:
    """A typo returns Postgres's message AND its hint, which the LLM repair step will use."""
    with pytest.raises(QueryExecutionError, match="Perhaps you meant"):
        await db.run_query(validate_sql("SELECT amout FROM invoices"))
