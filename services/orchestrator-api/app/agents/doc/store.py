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

from collections.abc import Sequence
from typing import Any
from uuid import UUID

import numpy as np
import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.agents.doc.models import Chunk, Document

# Columns of the API's Document model, shared by every query that returns documents.
DOCUMENT_COLUMNS = (
    "id, session_id, filename, mime_type, size_bytes, title, status, stage, progress, error, page_count,"
    " chunk_count, table_count, picture_count, pictures_skipped, created_at, finished_at"
)
MAX_ATTEMPTS = 3  # a document whose processing was interrupted this many times is marked failed

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

    # ---------------------------------------------------------------- documents (R5)

    async def create_document(
        self, document_id: UUID, session_id: UUID, filename: str, mime_type: str, size_bytes: int, sha256: str
    ) -> Document:
        """Insert a new document with status 'queued'; the worker picks it up from there."""
        row = await self.fetch_one(
            f"INSERT INTO app.documents (id, session_id, filename, mime_type, size_bytes, sha256)"
            f" VALUES (%s, %s, %s, %s, %s, %s) RETURNING {DOCUMENT_COLUMNS}",
            (document_id, session_id, filename, mime_type, size_bytes, sha256),
        )
        return Document(**row)

    async def find_by_hash(self, session_id: UUID, sha256: str) -> Document | None:
        """The document with these exact bytes in this chat, if it was uploaded before."""
        row = await self.fetch_one(
            f"SELECT {DOCUMENT_COLUMNS} FROM app.documents WHERE session_id = %s AND sha256 = %s", (session_id, sha256)
        )
        return Document(**row) if row else None

    async def requeue(self, document_id: UUID) -> Document:
        """Put a failed document back in the queue (the user uploaded the same file again)."""
        row = await self.fetch_one(
            f"UPDATE app.documents SET status = 'queued', stage = NULL, progress = NULL, error = NULL, attempts = 0,"
            f" started_at = NULL, finished_at = NULL WHERE id = %s RETURNING {DOCUMENT_COLUMNS}",
            (document_id,),
        )
        return Document(**row)

    async def get_document(self, document_id: UUID) -> Document | None:
        """One document, or None if it doesn't exist."""
        row = await self.fetch_one(f"SELECT {DOCUMENT_COLUMNS} FROM app.documents WHERE id = %s", (document_id,))
        return Document(**row) if row else None

    async def list_documents(self, session_id: UUID) -> list[Document]:
        """A chat's documents, oldest first (served by documents_session_idx)."""
        rows = await self._fetch_all(
            f"SELECT {DOCUMENT_COLUMNS} FROM app.documents WHERE session_id = %s ORDER BY created_at", (session_id,)
        )
        return [Document(**row) for row in rows]

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document; its chunks go with it (ON DELETE CASCADE). False if it didn't exist.

        The caller deletes the folder AFTERWARDS (rows first, then files: plan gotcha 12), so a
        crash in between leaves an orphan folder, which the startup cleanup removes.
        """
        return await self.fetch_one("DELETE FROM app.documents WHERE id = %s RETURNING id", (document_id,)) is not None

    async def all_document_ids(self) -> set[UUID]:
        """Every document id, for the orphan-folder cleanup at startup."""
        return {row["id"] for row in await self._fetch_all("SELECT id FROM app.documents")}

    async def picture_path(self, document_id: UUID, chunk_id: int) -> str | None:
        """The saved crop of a picture chunk, or None if that chunk isn't a picture of this document."""
        row = await self.fetch_one(
            "SELECT image_path FROM app.document_chunks WHERE id = %s AND document_id = %s AND chunk_type = 'picture'",
            (chunk_id, document_id),
        )
        return row["image_path"] if row else None

    # ------------------------------------------------------------- job queue (R5)

    async def claim_next_document(self, session_id: UUID | None = None) -> Document | None:
        """Take the oldest queued document and mark it 'processing'. None if the queue is empty.

        Postgres as a job queue (decision D10): the inner SELECT locks the row it picks
        FOR UPDATE, and SKIP LOCKED makes a second worker (or a second API process) skip rows
        that are already locked instead of waiting, so two workers can never claim the same
        document. One statement = one transaction, so the claim is atomic.
        `session_id` restricts the queue to one chat: tests use it so they never take real documents.
        """
        chat_filter = "AND session_id = %s" if session_id else ""
        row = await self.fetch_one(
            f"""
            UPDATE app.documents
            SET status = 'processing', stage = 'parsing', progress = NULL, started_at = now(),
                attempts = attempts + 1, error = NULL
            WHERE id = (
                SELECT id FROM app.documents WHERE status = 'queued' {chat_filter}
                ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1
            )
            RETURNING {DOCUMENT_COLUMNS}
            """,
            (session_id,) if session_id else (),
        )
        return Document(**row) if row else None

    async def recover_interrupted(self, session_id: UUID | None = None) -> tuple[int, int]:
        """At startup: documents left 'processing' by a crash go back to the queue, or fail after MAX_ATTEMPTS.

        Returns (requeued, failed). Only safe when no worker is running yet (it runs before the
        worker starts). `session_id` restricts it to one chat (tests).
        """
        chat_filter = "AND session_id = %(session_id)s" if session_id else ""
        params = {"max": MAX_ATTEMPTS, "session_id": session_id}
        async with self._pool.connection() as conn, conn.transaction():
            failed = await conn.execute(
                f"UPDATE app.documents SET status = 'failed', stage = NULL, progress = NULL, finished_at = now(),"
                f" error = 'Processing was interrupted ' || attempts || ' times.'"
                f" WHERE status = 'processing' AND attempts >= %(max)s {chat_filter}",
                params,
            )
            requeued = await conn.execute(
                f"UPDATE app.documents SET status = 'queued', stage = NULL, progress = NULL"
                f" WHERE status = 'processing' {chat_filter}",
                params,
            )
        return requeued.rowcount, failed.rowcount

    async def set_stage(self, document_id: UUID, stage: str, progress: str | None = None) -> None:
        """Record the current processing stage (and progress within it) for the UI's status chip."""
        await self._execute(
            "UPDATE app.documents SET stage = %s, progress = %s WHERE id = %s", (stage, progress, document_id)
        )

    async def set_parsed(self, document_id: UUID, title: str, page_count: int) -> None:
        """Store what parsing learned (title and pages) as soon as it's known."""
        await self._execute(
            "UPDATE app.documents SET title = %s, page_count = %s WHERE id = %s", (title, page_count, document_id)
        )

    async def mark_failed(self, document_id: UUID, error: str) -> None:
        """Processing failed: store a short, readable reason (the full traceback goes to the log)."""
        await self._execute(
            "UPDATE app.documents SET status = 'failed', stage = NULL, progress = NULL, error = %s, finished_at = now()"
            " WHERE id = %s",
            (error, document_id),
        )

    async def save_chunks_and_mark_ready(
        self,
        document_id: UUID,
        session_id: UUID,
        chunks: Sequence[Chunk],
        vectors: Sequence[np.ndarray],
        *,
        page_count: int,
        table_count: int,
        picture_count: int,
        pictures_skipped: int,
        embedding_model: str,
    ) -> None:
        """Store all chunks with their vectors and mark the document 'ready', in ONE transaction (plan 5.8).

        All-or-nothing: a document is fully searchable or not at all. The first DELETE makes a
        re-run safe (a previous attempt that died halfway leaves nothing behind). Postgres
        computes each chunk's `tsv` (generated column) and adds it to the HNSW and GIN indexes.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        rows = [
            (document_id, session_id, c.chunk_index, c.chunk_type, c.heading_path, c.page_start, c.page_end,
             c.content, c.embed_text, c.image_path, c.token_count, vector)
            for c, vector in zip(chunks, vectors, strict=True)
        ]  # fmt: skip
        async with self._pool.connection() as conn, conn.transaction():
            await conn.execute("DELETE FROM app.document_chunks WHERE document_id = %s", (document_id,))
            async with conn.cursor() as cursor:
                await cursor.executemany(
                    "INSERT INTO app.document_chunks (document_id, session_id, chunk_index, chunk_type, heading_path,"
                    " page_start, page_end, content, embed_text, image_path, token_count, embedding)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    rows,
                )
            await conn.execute(
                "UPDATE app.documents SET status = 'ready', stage = NULL, progress = NULL, error = NULL,"
                " finished_at = now(), chunk_count = %s, page_count = %s, table_count = %s, picture_count = %s,"
                " pictures_skipped = %s, embedding_model = %s WHERE id = %s",
                (len(chunks), page_count, table_count, picture_count, pictures_skipped, embedding_model, document_id),
            )

    # ---------------------------------------------------------------- helpers

    async def _fetch_all(self, query: str, params: Any = ()) -> list[dict[str, Any]]:
        """Run one statement and return all rows as dicts."""
        async with self._pool.connection() as conn:
            cursor = await conn.execute(query, params)
            return await cursor.fetchall()

    async def _execute(self, query: str, params: Any = ()) -> None:
        """Run one statement that returns no rows."""
        async with self._pool.connection() as conn:
            await conn.execute(query, params)
