"""Tests for schema reflection (app/agents/db/schema.py).

The to_prompt() test is a pure unit test on hand-built objects; the load_schema()
tests read the real local Postgres (skipped if it's not running).
"""

import pytest

from app.agents.db.database import ReadOnlyDatabase
from app.agents.db.safety import ALLOWED_TABLES
from app.agents.db.schema import Column, DatabaseSchema, Table, load_schema


def test_to_prompt_format() -> None:
    """Types, NOT NULL, foreign keys, comments and example values all appear in the prompt text."""
    schema = DatabaseSchema(
        tables=[
            Table(
                name="invoices",
                comment="Bills received from vendors.",
                columns=[
                    Column("vendor_id", "integer", nullable=False, references="vendors.id"),
                    Column("status", "text", False, comment="Payment state.", example_values=["paid", "pending"]),
                    Column("paid_date", "date", nullable=True),
                ],
            )
        ]
    )
    assert schema.to_prompt() == (
        "TABLE invoices  -- Bills received from vendors.\n"
        "  vendor_id integer NOT NULL -> vendors.id\n"
        "  status text NOT NULL  -- Payment state; values: 'paid', 'pending'\n"
        "  paid_date date"
    )


@pytest.mark.integration
async def test_load_schema_describes_exactly_the_allowed_tables(db: ReadOnlyDatabase) -> None:
    """Only business tables are described, so the LLM never learns other tables exist."""
    schema = await load_schema(db)
    assert {table.name for table in schema.tables} == ALLOWED_TABLES


@pytest.mark.integration
async def test_load_schema_finds_keys_comments_and_values(db: ReadOnlyDatabase) -> None:
    """Foreign keys, COMMENTs and example values are reflected from the real database."""
    schema = await load_schema(db)
    columns = {(t.name, c.name): c for t in schema.tables for c in t.columns}

    assert columns[("invoices", "vendor_id")].references == "vendors.id"
    assert "pending" in columns[("invoices", "status")].comment
    assert "Amazon Web Services" in columns[("vendors", "name")].example_values
    assert columns[("users", "email")].example_values == []  # emails are skipped
    assert columns[("users", "full_name")].example_values == []  # 25 names > MAX_EXAMPLE_VALUES
