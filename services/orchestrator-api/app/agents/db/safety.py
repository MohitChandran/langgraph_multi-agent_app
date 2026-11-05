"""SQL safety gate for the DB agent.

Every SQL query the LLM writes passes through `validate_sql()` before it can reach
Postgres. This is safety layer 2 of our defense in depth:

    layer 1: Postgres read-only user (scripts/db/02_readonly_role.sh)  - the database refuses writes
    layer 2: this module                                               - we refuse anything suspicious
    layer 3: row LIMIT + statement_timeout                             - bounds on what a valid query costs

Why parse instead of searching the text for words like "DROP"?  String checks are easy
to fool ("dRoP", comments, a DELETE hidden inside a WITH clause) and they also block
innocent queries (a vendor named "Drop Shipping Ltd"). sqlglot turns the SQL into a
syntax tree, so we check what the query actually DOES, not what characters it contains.

This module has no LLM and no database in it: it's pure Python, so it is fast and
fully unit-testable (see tests/test_db_safety.py).
"""

import logging
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

# sqlglot logs a warning for text it can't model ("Falling back to parsing as a 'Command'").
# We reject those queries ourselves with a clear message, so the warning is just noise.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

# The only tables the agent may read. Anything else (pg_catalog, information_schema,
# the RAG tables added in Phase 3, ...) is rejected even if Postgres would allow it.
ALLOWED_TABLES = frozenset({"vendors", "users", "contracts", "invoices", "tickets", "audit_logs"})

MAX_ROWS = 100  # results go into an LLM prompt and a UI table; 100 rows is plenty
MAX_SQL_CHARS = 5_000  # a real business question never needs a longer query

# Syntax-tree node types that change data, change the schema, or take locks.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Copy,
    exp.Grant,
    exp.Revoke,
    exp.Set,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Lock,
    exp.Into,  # SELECT ... INTO new_table creates a table
    exp.Command,  # sqlglot's fallback for statements it doesn't model (VACUUM, DO, CALL, ...)
)

# Functions that can sleep, read server files, open connections, or read/change settings.
BLOCKED_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink")
BLOCKED_FUNCTIONS = frozenset(
    {
        "set_config",
        "current_setting",
        "query_to_xml",
        "query_to_xml_and_xmlschema",
        "table_to_xml",
        "cursor_to_xml",
    }
)


class SQLValidationError(ValueError):
    """Raised when a query is rejected.

    The message is written for the LLM: in the repair step it is sent back to the
    model so it can fix its query, so it says what is wrong AND what is allowed.
    """


@dataclass(frozen=True)
class ValidatedQuery:
    """A query that passed every check and is safe to execute.

    Attributes:
        sql: the final SQL to run. It may differ from the LLM's text because we
             add or lower the LIMIT, and sqlglot re-prints it in a normalized form.
        tables: the business tables the query reads (useful for logging and tracing).
    """

    sql: str
    tables: frozenset[str]


def validate_sql(sql: str) -> ValidatedQuery:
    """Check LLM-generated SQL and return a safe, row-limited version of it.

    Runs every check in order and stops at the first problem.

    Raises:
        SQLValidationError: if the query is empty, unparsable, not a single read-only
            SELECT, touches tables outside ALLOWED_TABLES, uses a blocked function,
            or uses SELECT *.
    """
    statement = _parse_single_statement(sql)
    _check_is_select(statement)
    _check_no_forbidden_nodes(statement)
    tables = _check_tables(statement)
    _check_functions(statement)
    _check_no_select_star(statement)
    _enforce_row_limit(statement)
    return ValidatedQuery(sql=statement.sql(dialect="postgres"), tables=tables)


def _parse_single_statement(sql: str) -> exp.Expression:
    """Parse the SQL with the Postgres dialect and make sure it is exactly one statement.

    One statement only blocks "stacked queries" such as `SELECT 1; DROP TABLE invoices`.
    """
    if not sql or not sql.strip():
        raise SQLValidationError("The query is empty.")
    if len(sql) > MAX_SQL_CHARS:
        raise SQLValidationError(f"The query is too long (max {MAX_SQL_CHARS} characters). Simplify it.")
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except ParseError as error:
        raise SQLValidationError(f"The query is not valid PostgreSQL: {error}") from error
    if len(statements) != 1:
        raise SQLValidationError(f"Exactly one SQL statement is allowed, got {len(statements)}.")
    return statements[0]


def _check_is_select(statement: exp.Expression) -> None:
    """The top-level statement must be a SELECT, or a UNION / INTERSECT / EXCEPT of SELECTs.

    A WITH clause (CTE) is fine: sqlglot attaches it to the SELECT it belongs to.
    """
    if not isinstance(statement, exp.Select | exp.SetOperation):
        raise SQLValidationError(f"Only SELECT queries are allowed, got {statement.key.upper()}.")


def _check_no_forbidden_nodes(statement: exp.Expression) -> None:
    """Walk the whole syntax tree and reject any write, DDL, lock or unknown command.

    Checking the whole tree (not just the top) matters: Postgres allows writes hidden
    inside a SELECT, e.g. `WITH d AS (DELETE FROM invoices RETURNING id) SELECT id FROM d`.
    """
    for node in statement.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise SQLValidationError(f"Only read-only queries are allowed; found {node.key.upper()}.")
    if statement.args.get("locks"):  # SELECT ... FOR UPDATE / FOR SHARE
        raise SQLValidationError("Row locks (FOR UPDATE / FOR SHARE) are not allowed.")


def _check_tables(statement: exp.Expression) -> frozenset[str]:
    """Make sure every table the query reads is one of ALLOWED_TABLES and return them.

    Names defined by the query's own WITH clauses (CTEs) are allowed too, since
    they are temporary results built from allowed tables. A schema prefix is only
    allowed if it is `public` (blocks pg_catalog.pg_user, information_schema.columns, ...).
    """
    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    tables = set()
    for table in statement.find_all(exp.Table):
        name = table.name.lower()
        schema = table.db.lower()
        if not name:
            raise SQLValidationError("Functions in the FROM clause are not allowed; select from tables only.")
        if schema and schema != "public":
            raise SQLValidationError(f"Schema '{schema}' is not allowed. Use only: {_allowed_list()}.")
        if name in cte_names and not schema:
            continue
        if name not in ALLOWED_TABLES:
            raise SQLValidationError(f"Table '{name}' is not allowed. Use only: {_allowed_list()}.")
        tables.add(name)
    if not tables:
        raise SQLValidationError(f"The query must read from at least one table: {_allowed_list()}.")
    return frozenset(tables)


def _check_functions(statement: exp.Expression) -> None:
    """Reject functions that can sleep, read files, open connections or touch server settings.

    Postgres-specific functions like pg_sleep are unknown to sqlglot, so they appear
    as `Anonymous` nodes holding the raw name; known functions report their SQL name.
    """
    for func in statement.find_all(exp.Func):
        name = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()
        if name.startswith(BLOCKED_FUNCTION_PREFIXES) or name in BLOCKED_FUNCTIONS:
            raise SQLValidationError(f"Function '{name}' is not allowed.")


def _check_no_select_star(statement: exp.Expression) -> None:
    """Reject `SELECT *` and `table.*`; columns must be named explicitly.

    Naming columns keeps results small and readable, and makes the LLM think about
    which data the question really needs. `COUNT(*)` is fine: that star doesn't
    return columns.
    """
    for star in statement.find_all(exp.Star):
        if not isinstance(star.parent, exp.Count):
            raise SQLValidationError("SELECT * is not allowed; list the columns you need explicitly.")


def _enforce_row_limit(statement: exp.Expression) -> None:
    """Make sure the outermost query returns at most MAX_ROWS rows (modifies the tree in place).

    - no LIMIT                         -> add LIMIT 100
    - LIMIT 20                         -> keep it
    - LIMIT 5000 / LIMIT ALL / FETCH   -> replace with LIMIT 100
    Only the outermost LIMIT matters: it caps what comes back, whatever happens inside.
    """
    limit = statement.args.get("limit")
    if isinstance(limit, exp.Limit):
        value = limit.expression
        if isinstance(value, exp.Literal) and value.is_int and int(value.name) <= MAX_ROWS:
            return
    statement.set("limit", exp.Limit(expression=exp.Literal.number(MAX_ROWS)))


def _allowed_list() -> str:
    """Return the allowed table names as a sorted, comma-separated string for error messages."""
    return ", ".join(sorted(ALLOWED_TABLES))
