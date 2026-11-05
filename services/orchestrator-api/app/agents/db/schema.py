"""Schema reflection: read the database structure and describe it to the LLM.

The LLM can only write correct SQL if it knows the tables, columns, how tables
join, and which values exist (is it status = 'pending' or 'Pending'?). Instead of
hand-writing that into the prompt (which goes stale the moment the schema changes),
we READ it from Postgres at startup:

  - columns + types + NULL-ability   from information_schema.columns
  - table / column descriptions      from the COMMENTs in scripts/db/01_schema.sql
  - foreign keys (how to JOIN)       from pg_constraint
  - real example values              from the data itself, for low-cardinality text
                                     columns (statuses, categories, countries, vendor names)

The result is turned into compact text by DatabaseSchema.to_prompt().

These queries are written by us and read system views, so they use
ReadOnlyDatabase.fetch_trusted(). The LLM's own SQL can't touch system views
(the safety gate blocks them) - only this trusted code can.

Try it:  uv run python -m app.agents.db.schema
"""

import asyncio
from dataclasses import dataclass, field

from psycopg import sql

from app.agents.db.database import ReadOnlyDatabase
from app.agents.db.safety import ALLOWED_TABLES
from app.config import get_settings

# A text column with at most this many distinct values gets its values listed in the
# prompt. 20 includes statuses, categories, countries and the 18 vendor names
# (so "AWS" can be matched to 'Amazon Web Services'), but skips names, emails, numbers.
MAX_EXAMPLE_VALUES = 20
TEXT_TYPES = frozenset({"text", "character", "character varying"})

COLUMNS_QUERY = """
    SELECT c.table_name,
           c.column_name,
           c.data_type,
           c.is_nullable = 'YES' AS nullable,
           col_description(format('%%I.%%I', c.table_schema, c.table_name)::regclass, c.ordinal_position) AS comment
    FROM information_schema.columns AS c
    WHERE c.table_schema = 'public' AND c.table_name = ANY(%s)
    ORDER BY c.table_name, c.ordinal_position
"""

TABLE_COMMENTS_QUERY = """
    SELECT c.relname AS table_name, obj_description(c.oid, 'pg_class') AS comment
    FROM pg_class AS c
    WHERE c.relnamespace = 'public'::regnamespace AND c.relname = ANY(%s)
"""

# One row per foreign-key column, e.g. invoices.vendor_id -> vendors.id
FOREIGN_KEYS_QUERY = """
    SELECT src.relname AS table_name, src_col.attname AS column_name,
           dst.relname AS ref_table, dst_col.attname AS ref_column
    FROM pg_constraint AS con
    JOIN pg_class AS src ON src.oid = con.conrelid
    JOIN pg_class AS dst ON dst.oid = con.confrelid
    JOIN pg_attribute AS src_col ON src_col.attrelid = con.conrelid AND src_col.attnum = con.conkey[1]
    JOIN pg_attribute AS dst_col ON dst_col.attrelid = con.confrelid AND dst_col.attnum = con.confkey[1]
    WHERE con.contype = 'f' AND con.connamespace = 'public'::regnamespace AND src.relname = ANY(%s)
"""


@dataclass
class Column:
    """One column, with everything the LLM needs to use it correctly."""

    name: str
    data_type: str
    nullable: bool
    comment: str | None = None
    references: str | None = None  # "vendors.id" if this column is a foreign key
    example_values: list[str] = field(default_factory=list)


@dataclass
class Table:
    """One table: its description and its columns in database order."""

    name: str
    comment: str | None
    columns: list[Column]


@dataclass
class DatabaseSchema:
    """The whole reflected schema, ready to be put into the SQL-generation prompt."""

    tables: list[Table]

    def to_prompt(self) -> str:
        """Render the schema as compact, SQL-like text for the LLM.

        Example output:
            TABLE invoices  -- Bills received from vendors.
              vendor_id integer NOT NULL -> vendors.id
              status text NOT NULL  -- pending = ...; values: 'cancelled', 'overdue', 'paid', 'pending'

        This format is short (saves tokens) and close to SQL, which LLMs read well.
        """
        blocks = []
        for table in self.tables:
            lines = [f"TABLE {table.name}" + (f"  -- {table.comment}" if table.comment else "")]
            for column in table.columns:
                line = f"  {column.name} {column.data_type}" + ("" if column.nullable else " NOT NULL")
                if column.references:
                    line += f" -> {column.references}"
                notes = [column.comment.rstrip(".")] if column.comment else []
                if column.example_values:
                    notes.append("values: " + ", ".join(f"'{value}'" for value in column.example_values))
                if notes:
                    line += "  -- " + "; ".join(notes)
                lines.append(line)
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


async def load_schema(db: ReadOnlyDatabase) -> DatabaseSchema:
    """Read tables, columns, comments, foreign keys and example values from Postgres.

    Only ALLOWED_TABLES are described: the LLM shouldn't even learn that other
    tables exist. Call this once at startup and reuse the result; the schema
    doesn't change while the app runs.
    """
    table_names = sorted(ALLOWED_TABLES)
    column_rows = await db.fetch_trusted(COLUMNS_QUERY, (table_names,))
    comment_rows = await db.fetch_trusted(TABLE_COMMENTS_QUERY, (table_names,))
    table_comments = {row["table_name"]: row["comment"] for row in comment_rows}
    foreign_keys = {
        (row["table_name"], row["column_name"]): f"{row['ref_table']}.{row['ref_column']}"
        for row in await db.fetch_trusted(FOREIGN_KEYS_QUERY, (table_names,))
    }

    tables: dict[str, Table] = {}
    for row in column_rows:
        name = row["table_name"]
        table = tables.setdefault(name, Table(name, table_comments.get(name), []))
        column = Column(
            name=row["column_name"],
            data_type=row["data_type"],
            nullable=row["nullable"],
            comment=row["comment"],
            references=foreign_keys.get((row["table_name"], row["column_name"])),
        )
        if column.data_type in TEXT_TYPES and not column.name.endswith("email"):
            column.example_values = await _example_values(db, table.name, column.name)
        table.columns.append(column)
    return DatabaseSchema(tables=list(tables.values()))


async def _example_values(db: ReadOnlyDatabase, table: str, column: str) -> list[str]:
    """Return the column's distinct values if there are at most MAX_EXAMPLE_VALUES, else [].

    Fetches MAX_EXAMPLE_VALUES + 1 values: if we get that many, the column has too
    many distinct values (names, numbers, free text) and listing them would waste tokens.
    Table/column names go through sql.Identifier, which quotes them safely.
    """
    query = sql.SQL(
        "SELECT DISTINCT {column} AS value FROM {table} WHERE {column} IS NOT NULL ORDER BY 1 LIMIT {n}"
    ).format(column=sql.Identifier(column), table=sql.Identifier(table), n=sql.Literal(MAX_EXAMPLE_VALUES + 1))
    values = [str(row["value"]) for row in await db.fetch_trusted(query)]
    return values if len(values) <= MAX_EXAMPLE_VALUES else []


async def _main() -> None:
    """CLI helper: print the schema exactly as the LLM will see it, plus its size."""
    async with ReadOnlyDatabase(get_settings().readonly_database_url) as db:
        text = (await load_schema(db)).to_prompt()
    print(text)
    print(f"\n[{len(text)} characters, roughly {len(text) // 4} tokens]")


if __name__ == "__main__":
    asyncio.run(_main())
