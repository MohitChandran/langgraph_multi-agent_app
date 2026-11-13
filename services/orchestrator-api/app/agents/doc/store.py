"""Postgres storage for uploaded documents and their searchable chunks (schema `app`, user `app_rw`).

Why the vectors live in Postgres (pgvector) instead of a separate vector database:
one database and ONE transaction hold the document row, its chunks, their keyword index
and their vectors. A document is therefore searchable completely or not at all, and a
single SQL query can filter by chat, search by meaning (vectors) and search by words
(full-text) together. Plan 02, decision D2.

Two indexes do the searching (plan 02, section 4.2):
  - HNSW on `embedding`: approximate nearest-neighbour search by meaning (cosine distance `<=>`).
  - GIN on `tsv`: keyword search. `tsv` is a GENERATED column: Postgres computes it from
    `embed_text` on every insert, so we never write it ourselves.

Same pool style as ChatStore: autocommit ON, multi-statement writes inside
`async with conn.transaction()`. This module only grows in later stages:
R1 = tables; R5 = documents, job queue, saving chunks; R6 = search.
"""

from typing import Any

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

# The embedding model (granite-embedding-small-english-r2) outputs 384 numbers per text.
# The column type must match exactly; embeddings.py checks the model at startup.
EMBEDDING_DIM = 384

# Schema-qualified table names, as in ChatStore. The unqualified `vector` type and the
# `vector_cosine_ops` operator class come from schema public (app_rw's search_path = app, public).
SCHEMA_STATEMENTS = (
    # One row per uploaded file. A document belongs to ONE chat (decision D9):
    # deleting the chat deletes its documents, and their chunks with them.
    """
    CREATE TABLE IF NOT EXISTS app.documents (
        id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id        UUID NOT NULL REFERENCES app.chat_sessions(id) ON DELETE CASCADE,
        filename          TEXT NOT NULL,
        mime_type         TEXT NOT NULL,
        size_bytes        BIGINT NOT NULL,
        sha256            TEXT NOT NULL,
        title             TEXT,
        status            TEXT NOT NULL DEFAULT 'queued'
                          CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
        stage             TEXT,
        progress          TEXT,
        error             TEXT,
        attempts          INTEGER NOT NULL DEFAULT 0,
        page_count        INTEGER,
        chunk_count       INTEGER,
        table_count       INTEGER,
        picture_count     INTEGER,
        pictures_skipped  INTEGER,
        embedding_model   TEXT,
        created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
        started_at        TIMESTAMPTZ,
        finished_at       TIMESTAMPTZ,
        UNIQUE (session_id, sha256)
    )
    """,
    "COMMENT ON COLUMN app.documents.filename IS 'Display only. Never used to build a file path (untrusted input).'",
    "COMMENT ON COLUMN app.documents.stage IS 'While processing: parsing, describing_pictures, chunking or embedding.'",
    "COMMENT ON COLUMN app.documents.progress IS 'Human-readable progress within the stage, e.g. pictures 3/12.'",
    "COMMENT ON COLUMN app.documents.sha256 IS 'File hash: the same file twice in one chat is processed once.'",
    "COMMENT ON COLUMN app.documents.embedding_model IS 'Model that made the vectors (two models never mix).'",
    "CREATE INDEX IF NOT EXISTS documents_session_idx ON app.documents (session_id, created_at)",
    # Partial index: contains ONLY queued rows, so the worker finds the next job instantly
    # no matter how many finished documents exist.
    "CREATE INDEX IF NOT EXISTS documents_queue_idx ON app.documents (created_at) WHERE status = 'queued'",
    # One row per searchable piece of a document.
    f"""
    CREATE TABLE IF NOT EXISTS app.document_chunks (
        id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        document_id   UUID NOT NULL REFERENCES app.documents(id) ON DELETE CASCADE,
        session_id    UUID NOT NULL,
        chunk_index   INTEGER NOT NULL,
        chunk_type    TEXT NOT NULL CHECK (chunk_type IN ('text', 'table', 'picture')),
        heading_path  TEXT NOT NULL DEFAULT '',
        page_start    INTEGER NOT NULL,
        page_end      INTEGER NOT NULL,
        content       TEXT NOT NULL,
        embed_text    TEXT NOT NULL,
        image_path    TEXT,
        token_count   INTEGER NOT NULL,
        embedding     vector({EMBEDDING_DIM}) NOT NULL,
        tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', embed_text)) STORED,
        UNIQUE (document_id, chunk_index)
    )
    """,
    "COMMENT ON COLUMN app.document_chunks.session_id IS 'Copied from documents: filter by chat without a JOIN.'",
    "COMMENT ON COLUMN app.document_chunks.content IS 'What the LLM reads and the UI shows.'",
    "COMMENT ON COLUMN app.document_chunks.embed_text IS 'Title + heading path + content: embedded + indexed.'",
    "COMMENT ON COLUMN app.document_chunks.tsv IS 'Keyword index entry, computed by Postgres from embed_text.'",
    "CREATE INDEX IF NOT EXISTS chunks_session_idx ON app.document_chunks (session_id)",
    "CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON app.document_chunks USING gin (tsv)",
    "CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON app.document_chunks USING hnsw (embedding vector_cosine_ops)",
)


class DocumentStore:
    """Pool of connections to schema `app` for documents, chunks and vector search.

    Usage, like ChatStore:

        async with DocumentStore(url) as store:
            await store.setup()

    setup() needs ChatStore.setup() to have run first: documents reference chat_sessions.
    """

    def __init__(self, url: str, max_connections: int = 5) -> None:
        """Prepare the pool without connecting yet (connecting happens in open())."""
        self._pool = AsyncConnectionPool(
            conninfo=url,
            min_size=1,
            max_size=max_connections,
            kwargs={"autocommit": True, "row_factory": dict_row},
            configure=self._configure_connection,
            open=False,
        )

    @staticmethod
    async def _configure_connection(conn: psycopg.AsyncConnection) -> None:
        """Run once on every new connection in the pool.

        1. register_vector_async: teach this connection the `vector` type, so numpy arrays are
           sent as real vectors, and vectors come back as `pgvector.Vector` objects (call
           `.to_numpy()`) instead of the string "[0.1,0.2,...]".
        2. hnsw.iterative_scan = relaxed_order (pgvector >= 0.8): the HNSW index returns the
           nearest vectors first and the `WHERE session_id = ...` filter is applied afterwards.
           Without this, a chat with few chunks could get fewer results than asked for, because
           most of the nearest vectors belong to OTHER chats. Iterative scan keeps going until
           enough rows pass the filter.
        """
        await register_vector_async(conn)
        await conn.execute("SET hnsw.iterative_scan = relaxed_order")

    async def open(self) -> None:
        """Open the pool and wait until a connection works (fails fast on a wrong URL/password)."""
        await self._pool.open(wait=True, timeout=10)

    async def close(self) -> None:
        """Close all connections in the pool."""
        await self._pool.close()

    async def __aenter__(self) -> "DocumentStore":
        """Open the pool when entering an `async with` block."""
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the pool when leaving the `async with` block, even after an error."""
        await self.close()

    async def setup(self) -> None:
        """Create the document tables and indexes if they don't exist yet. Safe on every startup.

        One transaction: everything is created, or nothing. Needs schema app (03_app_schema.sh),
        the chat tables (ChatStore.setup) and pgvector in app_rw's search_path.
        """
        async with self._pool.connection() as conn, conn.transaction():
            for statement in SCHEMA_STATEMENTS:
                await conn.execute(statement)

    async def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        """Run one statement and return its first row as a dict (or None). Used by tests and later stages."""
        async with self._pool.connection() as conn:
            cursor = await conn.execute(query, params)
            return await cursor.fetchone()
