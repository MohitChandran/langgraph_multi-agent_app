"""Read-only database access for the DB agent.

All queries run through a pool of async connections logged in as `readonly_agent`.

Why a connection POOL? Opening a Postgres connection takes tens of milliseconds
(network round trips + authentication). A pool opens a few connections once and
lends them out per query, so many concurrent users (FastAPI in Phase 2) don't each
pay that cost or overload Postgres with connections.

Why ASYNC? While Postgres works on a query, the Python process isn't blocked:
FastAPI can serve other requests and LangGraph can run other agents in parallel.

Safety: this module is layers 1 and 3 of the defense in depth.
  - it only ever uses READONLY_DATABASE_URL (the read-only user),
  - every connection is additionally put in read-only mode by our code,
  - the user's 5s statement_timeout stops runaway queries.
`run_query()` accepts only a `ValidatedQuery`, so LLM-written SQL can't reach the
database without first passing safety.validate_sql() (layer 2).

Try it:  uv run python -m app.agents.db.database "SELECT name, category FROM vendors"
"""

import asyncio
import sys
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.agents.db.safety import MAX_ROWS, SQLValidationError, ValidatedQuery, validate_sql
from app.config import get_settings


class QueryExecutionError(RuntimeError):
    """Raised when Postgres rejects or cancels a query.

    Holds Postgres's own error message (e.g. 'column "amout" does not exist'),
    which the repair step sends back to the LLM so it can fix the SQL.
    """


@dataclass(frozen=True)
class QueryResult:
    """Rows returned by a query, converted to JSON-friendly Python values.

    Attributes:
        columns: column names, in SELECT order.
        rows: one dict per row, e.g. {"name": "Dell Technologies", "total": 1520.5}.
        hit_row_limit: True if we got exactly MAX_ROWS rows, i.e. there may be more
            that the LIMIT cut off. The answer step tells the user so.
    """

    columns: list[str]
    rows: list[dict[str, Any]]
    hit_row_limit: bool

    @property
    def row_count(self) -> int:
        """Number of rows returned."""
        return len(self.rows)


class ReadOnlyDatabase:
    """A pool of read-only connections to the business database.

    Use it as an async context manager so the pool is always closed:

        async with ReadOnlyDatabase(url) as db:
            result = await db.run_query(validate_sql("SELECT name FROM vendors"))

    It's a class (not a global pool) so that each owner controls its lifetime:
    the FastAPI app opens one at startup, each test opens its own.
    """

    def __init__(self, url: str, max_connections: int = 5) -> None:
        """Prepare the pool without connecting yet (connecting happens in open())."""
        self._pool = AsyncConnectionPool(
            conninfo=url,
            min_size=1,
            max_size=max_connections,
            kwargs={"row_factory": dict_row},  # rows come back as {"column": value} dicts
            configure=self._configure_connection,
            open=False,
        )

    @staticmethod
    async def _configure_connection(conn: psycopg.AsyncConnection) -> None:
        """Run once on every new connection: force read-only transactions.

        The `readonly_agent` user is already read-only in Postgres. Setting it here
        too means that even if someone misconfigures the user or the URL, our code
        still can't write.

        Don't turn on autocommit: read-only mode is applied through the `BEGIN READ ONLY`
        that psycopg sends when it starts a transaction. In autocommit mode psycopg never
        sends BEGIN, so the setting is silently ignored (we tested this: with autocommit,
        a CREATE TABLE went through). The pool commits/rolls back when a connection is returned.
        """
        await conn.set_read_only(True)

    async def open(self) -> None:
        """Open the pool and wait until a connection works (fails fast on a wrong URL/password)."""
        await self._pool.open(wait=True, timeout=10)

    async def close(self) -> None:
        """Close all connections in the pool."""
        await self._pool.close()

    async def __aenter__(self) -> "ReadOnlyDatabase":
        """Open the pool when entering an `async with` block."""
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the pool when leaving the `async with` block, even after an error."""
        await self.close()

    async def run_query(self, query: ValidatedQuery) -> QueryResult:
        """Execute a validated SELECT and return its rows.

        Takes a ValidatedQuery, not a str: the type itself proves the SQL went
        through the safety gate. Any Postgres error (bad column, timeout, permission
        denied) becomes a QueryExecutionError carrying Postgres's message.
        """
        try:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(query.sql)
                rows = await cur.fetchall()
                columns = [column.name for column in cur.description or []]
        except psycopg.Error as error:
            raise QueryExecutionError(_postgres_message(error)) from error

        clean_rows = [{key: _to_json_value(value) for key, value in row.items()} for row in rows]
        return QueryResult(columns=columns, rows=clean_rows, hit_row_limit=len(rows) >= MAX_ROWS)

    async def fetch_trusted(self, query: str | sql.Composable, params: tuple | None = None) -> list[dict[str, Any]]:
        """Run SQL written by US (never by the LLM), e.g. schema reflection queries.

        Skips the safety gate because our own code needs things the LLM may not use,
        like information_schema. Still runs as the read-only user.
        """
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchall()


def _postgres_message(error: psycopg.Error) -> str:
    """Extract the useful one-line message from a psycopg error.

    Postgres errors carry a primary message plus an optional hint, e.g.
    'column "amout" does not exist' + 'Perhaps you meant to reference the column "invoices.amount".'
    The hint is gold for the LLM's repair step, so we keep it.
    """
    diag = error.diag
    message = diag.message_primary or str(error).strip()
    return f"{message} (hint: {diag.message_hint})" if diag.message_hint else message


def _to_json_value(value: Any) -> Any:
    """Convert database types to plain JSON types (for the LLM prompt, API responses and logs).

    NUMERIC comes back as Decimal and dates as date/datetime objects, which
    json.dumps can't handle. Money as float is fine for display (2 decimals).
    """
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value


async def _main(raw_sql: str) -> None:
    """CLI helper: validate a query, run it, and print the rows."""
    try:
        query = validate_sql(raw_sql)
    except SQLValidationError as error:
        print(f"Rejected by safety gate: {error}")
        return
    print(f"Running: {query.sql}\n")
    async with ReadOnlyDatabase(get_settings().readonly_database_url) as db:
        try:
            result = await db.run_query(query)
        except QueryExecutionError as error:
            print(f"Postgres error: {error}")
            return
    for row in result.rows:
        print(row)
    print(f"\n{result.row_count} rows" + (" (row limit reached, there may be more)" if result.hit_row_limit else ""))


if __name__ == "__main__":
    asyncio.run(_main(" ".join(sys.argv[1:]) or "SELECT name, category FROM vendors"))
