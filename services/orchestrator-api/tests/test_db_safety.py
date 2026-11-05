"""Unit tests for the DB agent's SQL safety gate (app/agents/db/safety.py).

Three groups:
  - legitimate business queries that MUST pass,
  - attack / mistake queries that MUST be rejected,
  - the LIMIT rewriting rules.
No database or LLM is needed, so these run in milliseconds and in CI.
Run: cd services/orchestrator-api && uv run pytest -v
"""

import pytest

from app.agents.db.safety import MAX_ROWS, SQLValidationError, validate_sql

ALLOWED_QUERIES = [
    "SELECT name, category FROM vendors",
    "SELECT v.name, sum(i.amount) AS total FROM invoices i JOIN vendors v ON v.id = i.vendor_id GROUP BY v.name",
    "SELECT count(*) FROM invoices WHERE status = 'pending'",
    "WITH due AS (SELECT vendor_id, amount FROM invoices WHERE status = 'pending') SELECT vendor_id FROM due",
    "SELECT name FROM vendors WHERE category = 'cloud' UNION SELECT full_name FROM users",
    "SELECT name FROM public.vendors",
    "SELECT invoice_number FROM invoices WHERE vendor_id IN (SELECT id FROM vendors WHERE country = 'India')",
    "SELECT name FROM vendors WHERE name = 'Drop Shipping Ltd'",  # keyword inside a string is just data
    "SELECT name FROM vendors -- DROP TABLE vendors",  # keyword inside a comment is ignored
    "SELECT name FROM vendors;",  # one trailing semicolon is still one statement
]

REJECTED_QUERIES = {
    "insert": "INSERT INTO vendors (name) VALUES ('x')",
    "update": "UPDATE invoices SET status = 'paid'",
    "delete": "DELETE FROM invoices",
    "drop": "DROP TABLE invoices",
    "truncate": "TRUNCATE invoices",
    "create": "CREATE TABLE hack (x int)",
    "alter": "ALTER TABLE vendors ADD COLUMN x int",
    "grant": "GRANT ALL ON vendors TO public",
    "stacked queries": "SELECT name FROM vendors; DROP TABLE vendors",
    "delete hidden in CTE": "WITH d AS (DELETE FROM invoices RETURNING id) SELECT id FROM d",
    "select into new table": "SELECT name INTO stolen FROM vendors",
    "row lock": "SELECT name FROM vendors FOR UPDATE",
    "system catalog": "SELECT usename FROM pg_catalog.pg_user",
    "information schema": "SELECT table_name FROM information_schema.tables",
    "unknown table": "SELECT secret FROM salaries",
    "pg_sleep": "SELECT pg_sleep(10), name FROM vendors",
    "pg_sleep uppercase": "SELECT PG_SLEEP(10), name FROM vendors",
    "read server file": "SELECT pg_read_file('/etc/passwd'), name FROM vendors",
    "dblink": "SELECT dblink('host=evil', 'SELECT 1'), name FROM vendors",
    "change setting": "SELECT set_config('statement_timeout', '0', false), name FROM vendors",
    "read setting": "SELECT current_setting('data_directory'), name FROM vendors",
    "select star": "SELECT * FROM vendors",
    "table star": "SELECT v.* FROM vendors v",
    "function in FROM": "SELECT x FROM generate_series(1, 1000000) AS x",
    "no table": "SELECT 1",
    "vacuum": "VACUUM",
    "set statement": "SET statement_timeout = 0",
    "transaction": "BEGIN",
    "empty": "   ",
    "not sql": "show me the invoices please",
}


@pytest.mark.parametrize("sql", ALLOWED_QUERIES)
def test_allows_read_only_business_queries(sql: str) -> None:
    """Normal analytical queries pass and come back with a row limit applied."""
    result = validate_sql(sql)
    assert f"LIMIT {MAX_ROWS}" in result.sql


@pytest.mark.parametrize("sql", REJECTED_QUERIES.values(), ids=REJECTED_QUERIES.keys())
def test_rejects_unsafe_queries(sql: str) -> None:
    """Writes, DDL, system access, dangerous functions and SELECT * are all refused."""
    with pytest.raises(SQLValidationError):
        validate_sql(sql)


def test_rejects_too_long_query() -> None:
    """Absurdly long queries are refused before parsing."""
    with pytest.raises(SQLValidationError, match="too long"):
        validate_sql("SELECT name FROM vendors WHERE " + " OR ".join(["id = 1"] * 2_000))


def test_error_message_lists_allowed_tables() -> None:
    """The error tells the LLM what IS allowed, so the repair step can fix the query."""
    with pytest.raises(SQLValidationError, match="vendors"):
        validate_sql("SELECT secret FROM salaries")


def test_reports_tables_used() -> None:
    """ValidatedQuery.tables lists the real tables read, without CTE names."""
    result = validate_sql(
        "WITH due AS (SELECT vendor_id FROM invoices) SELECT v.name FROM due JOIN vendors v ON v.id = due.vendor_id"
    )
    assert result.tables == {"invoices", "vendors"}


@pytest.mark.parametrize(
    ("sql", "expected_limit"),
    [
        ("SELECT name FROM vendors", MAX_ROWS),  # no limit -> added
        ("SELECT name FROM vendors LIMIT 5", 5),  # small limit -> kept
        ("SELECT name FROM vendors LIMIT 5000", MAX_ROWS),  # too big -> lowered
        ("SELECT name FROM vendors LIMIT ALL", MAX_ROWS),  # unlimited -> capped
        ("SELECT name FROM vendors FETCH FIRST 500 ROWS ONLY", MAX_ROWS),  # FETCH syntax -> capped
        ("SELECT name FROM vendors UNION SELECT full_name FROM users LIMIT 1000", MAX_ROWS),  # union -> capped
    ],
)
def test_row_limit_is_enforced(sql: str, expected_limit: int) -> None:
    """The outermost query never returns more than MAX_ROWS rows."""
    result = validate_sql(sql)
    assert result.sql.endswith(f"LIMIT {expected_limit}")
