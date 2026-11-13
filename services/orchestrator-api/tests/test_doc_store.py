"""Tests for the document storage foundations (app/agents/doc/store.py) - plan 02, stage R1.

Covers the R1 acceptance criteria (tables in schema app, the DB agent can't read them)
plus the two pieces of pgvector plumbing everything later depends on: numpy vectors are
stored as real vectors, and Postgres fills the keyword column `tsv` by itself.
"""

import numpy as np
import psycopg
import pytest
from pgvector import Vector

from app.agents.db.database import ReadOnlyDatabase
from app.agents.db.safety import SQLValidationError, validate_sql
from app.agents.doc.store import EMBEDDING_DIM, DocumentStore


@pytest.mark.integration
async def test_setup_is_idempotent_and_tables_and_indexes_live_in_app(document_store: DocumentStore) -> None:
    """setup() twice is harmless; both tables and the HNSW + GIN indexes exist in schema app only."""
    await document_store.setup()

    row = await document_store.fetch_one(
        "SELECT to_regclass('app.documents') IS NOT NULL AS documents,"
        "       to_regclass('app.document_chunks') IS NOT NULL AS chunks,"
        "       to_regclass('public.documents') IS NULL"
        "           AND to_regclass('public.document_chunks') IS NULL AS none_in_public,"
        "       (SELECT indexdef FROM pg_indexes"
        "        WHERE schemaname = 'app' AND indexname = 'chunks_embedding_idx') AS hnsw,"
        "       (SELECT indexdef FROM pg_indexes WHERE schemaname = 'app' AND indexname = 'chunks_tsv_idx') AS gin"
    )
    assert row["documents"] and row["chunks"] and row["none_in_public"]
    assert "USING hnsw (embedding vector_cosine_ops)" in row["hnsw"]
    assert "USING gin (tsv)" in row["gin"]


@pytest.mark.integration
@pytest.mark.parametrize("table", ["app.documents", "app.document_chunks"])
async def test_db_agent_user_cannot_read_document_tables(
    document_store: DocumentStore, db: ReadOnlyDatabase, table: str
) -> None:
    """Layer 1: readonly_agent (runs LLM-written SQL) is refused by Postgres itself."""
    with pytest.raises(psycopg.errors.InsufficientPrivilege, match="permission denied for schema app"):
        await db.fetch_trusted(f"SELECT 1 FROM {table}")


def test_safety_gate_rejects_document_tables() -> None:
    """Layer 2: the safety gate refuses schema app before the query reaches Postgres."""
    with pytest.raises(SQLValidationError, match="Schema 'app' is not allowed"):
        validate_sql("SELECT content FROM app.document_chunks")


@pytest.mark.integration
async def test_connections_are_configured_for_pgvector(document_store: DocumentStore) -> None:
    """The pool's configure hook ran: vectors come back typed (not as a string), and iterative HNSW scans are on.

    pgvector-python 0.5 returns a `Vector` object; `.to_numpy()` gives the numpy array.
    """
    row = await document_store.fetch_one(
        "SELECT '[1, 2, 3]'::vector AS v, current_setting('hnsw.iterative_scan') AS iterative_scan"
    )
    assert isinstance(row["v"], Vector) and row["v"].to_numpy().tolist() == [1.0, 2.0, 3.0]
    assert row["iterative_scan"] == "relaxed_order"


@pytest.mark.integration
async def test_chunk_round_trip_fills_tsv_and_deleting_the_chat_cascades(document_store: DocumentStore) -> None:
    """A chunk stores a 384-d numpy vector, Postgres computes tsv from embed_text, and deleting
    the chat removes its documents and their chunks (two levels of ON DELETE CASCADE)."""
    session = await document_store.fetch_one(
        "INSERT INTO app.chat_sessions (title) VALUES ('[pytest] doc store') RETURNING id"
    )
    try:
        document = await document_store.fetch_one(
            "INSERT INTO app.documents (session_id, filename, mime_type, size_bytes, sha256)"
            " VALUES (%s, 'report.pdf', 'application/pdf', 10, 'abc') RETURNING id",
            (session["id"],),
        )
        embed_text = "Q3 Report\nSLA Terms\n\nService credits if Amazon breaches uptime"
        vector = np.random.default_rng(0).random(EMBEDDING_DIM, dtype=np.float32)
        vector /= np.linalg.norm(vector)
        chunk = await document_store.fetch_one(
            "INSERT INTO app.document_chunks (document_id, session_id, chunk_index, chunk_type, page_start,"
            " page_end, content, embed_text, token_count, embedding)"
            " VALUES (%s, %s, 0, 'table', 4, 4, 'AWS | 99.95%%', %s, 12, %s)"
            " RETURNING id, tsv::text AS tsv, vector_dims(embedding) AS dims,"
            "           round(vector_norm(embedding)::numeric, 3) AS norm",
            (document["id"], session["id"], embed_text, vector),
        )
        assert chunk["dims"] == EMBEDDING_DIM and float(chunk["norm"]) == 1.0
        assert "'credit'" in chunk["tsv"] and "'breach'" in chunk["tsv"]  # stemmed by the 'english' config

        await document_store.fetch_one("DELETE FROM app.chat_sessions WHERE id = %s RETURNING id", (session["id"],))
        left = await document_store.fetch_one(
            "SELECT (SELECT count(*) FROM app.documents WHERE id = %s) AS documents,"
            "       (SELECT count(*) FROM app.document_chunks WHERE id = %s) AS chunks",
            (document["id"], chunk["id"]),
        )
        assert left == {"documents": 0, "chunks": 0}
    finally:
        await document_store.fetch_one("DELETE FROM app.chat_sessions WHERE id = %s RETURNING id", (session["id"],))
