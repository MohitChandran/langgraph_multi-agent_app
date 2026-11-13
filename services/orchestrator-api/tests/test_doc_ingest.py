"""Tests for ingestion: storing chunks, the Postgres job queue, the pipeline and the worker - plan 02, stage R5.

No Docling, no Groq, no embedding model: parse/describe/embed are fakes, so the whole pipeline
runs in milliseconds. Store tests use the real Postgres (skipped without it). Queue and recovery
calls are limited to the test's own chat, so a running dev server's documents are never touched.
"""

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import numpy as np
import psycopg
import pytest

from app.agents.doc.embeddings import normalize
from app.agents.doc.ingest import IngestDeps, ingest_document
from app.agents.doc.models import Chunk, Document, Element, ParsedDocument
from app.agents.doc.store import EMBEDDING_DIM, DocumentStore
from app.agents.doc.worker import IngestionWorker
from app.documents.files import MARKDOWN, document_dir, remove_orphan_folders, save_original


def fake_vector(text: str) -> np.ndarray:
    """A deterministic, normalized 384-d vector derived from the text's hash (same text -> same vector)."""
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    return normalize(np.random.default_rng(seed).normal(size=EMBEDDING_DIM))


class FakeEmbedder:
    """Stands in for Embedder: hash-based vectors, no model download."""

    model_name = "fake-embedder"

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        """One fake vector per text."""
        return [fake_vector(text) for text in texts]


class WordCounter:
    """Stands in for TokenCounter: 1 token = 1 word."""

    def count(self, text: str) -> int:
        """Number of words."""
        return len(text.split())


SAMPLE = ParsedDocument(
    "Q3 Report",
    3,
    [
        Element("heading", "4 Cloud Vendors", 1, 1, level=1),
        Element("heading", "4.2 SLA Terms", 1, 1, level=2),
        Element("text", "Service credits are claimed within 30 days.", 1, 1),
        Element("table", "Cloud SLAs", 2, 2, table_rows=[["Vendor", "SLA"], ["AWS", "99.95%"]]),
        Element("picture", "Figure 1: Incidents", 3, 3, image_path="chart.png", picture_label="bar_chart",
                area_fraction=0.15, width_px=800, height_px=400),
        Element("picture", "", 1, 1, image_path="logo.png", picture_label="logo", area_fraction=0.01,
                width_px=90, height_px=90),
    ],
)  # fmt: skip


async def new_document(store: DocumentStore, session_id: UUID, data_dir: Path, name: str = "report.md") -> Document:
    """A queued document with a real file on disk (as the upload endpoint would create it)."""
    document_id = uuid4()
    save_original(data_dir, document_id, MARKDOWN, b"# Report")
    return await store.create_document(document_id, session_id, name, MARKDOWN.mime_type, 8, uuid4().hex)


def deps_for(store: DocumentStore, data_dir: Path, parse=None, describe=None, max_pages: int = 150) -> IngestDeps:  # noqa: ANN001
    """IngestDeps wired to fakes; parse/describe can be replaced per test."""

    async def default_describe(picture: Element, heading_path: str) -> str:
        """Fake vision: a fixed description that mentions where the picture sits."""
        return f"Bar chart in {heading_path}. Q3: 14"

    return IngestDeps(
        store=store,
        data_dir=data_dir,
        parse=parse or (lambda path, pictures_dir, title: SAMPLE),
        describe=describe or default_describe,
        embedder=FakeEmbedder(),
        counter=WordCounter(),
        chunk_tokens=350,
        max_pages=max_pages,
        max_pictures=20,
        min_picture_area=0.02,
    )


async def chunk_rows(store: DocumentStore, document_id: UUID) -> list[dict]:
    """The stored chunks of a document with their vector size/length and keyword column."""
    async with store._pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT chunk_index, chunk_type, heading_path, content, tsv::text AS tsv, vector_dims(embedding) AS dims,"
            " round(vector_norm(embedding)::numeric, 3) AS norm FROM app.document_chunks"
            " WHERE document_id = %s ORDER BY chunk_index",
            (document_id,),
        )
        return await cursor.fetchall()


# ------------------------------------------------------------------ the pipeline


@pytest.mark.integration
async def test_ingest_turns_a_queued_document_into_ready_chunks(
    document_store: DocumentStore, test_session: UUID, tmp_path: Path
) -> None:
    """Claim -> parse -> describe -> chunk -> embed -> stored: status ready, counts, 384-d unit vectors, tsv filled."""
    titles = []
    parse = lambda path, pictures_dir, title: titles.append(title) or SAMPLE  # noqa: E731
    await new_document(document_store, test_session, tmp_path, name="q3-report.pdf")
    claimed = await document_store.claim_next_document(test_session)
    assert claimed.status == "processing" and claimed.stage == "parsing"

    await ingest_document(claimed, deps_for(document_store, tmp_path, parse=parse))

    document = await document_store.get_document(claimed.id)
    assert document.status == "ready" and document.stage is None and document.error is None
    assert (document.chunk_count, document.table_count, document.picture_count, document.pictures_skipped) == (
        3,
        1,
        1,
        1,
    )
    assert document.title == "Q3 Report" and titles == ["q3-report"]  # fallback title = uploaded name, not "original"

    rows = await chunk_rows(document_store, claimed.id)
    assert [r["chunk_type"] for r in rows] == ["text", "table", "picture"]
    assert all(r["dims"] == EMBEDDING_DIM and float(r["norm"]) == 1.0 for r in rows)
    assert rows[0]["heading_path"] == "4 Cloud Vendors > 4.2 SLA Terms" and "'credit'" in rows[0]["tsv"]
    assert "Bar chart in 4 Cloud Vendors > 4.2 SLA Terms. Q3: 14" in rows[2]["content"]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("parse", "error_part"),
    [
        (lambda path, pictures, title: (_ for _ in ()).throw(RuntimeError("boom")), "Processing failed (RuntimeError)"),
        (lambda path, pictures, title: ParsedDocument("Big", 200, SAMPLE.elements), "200 pages; the limit is 150"),
        (lambda path, pictures, title: ParsedDocument("Empty", 1, []), "No text could be extracted"),
    ],
    ids=["crash", "too-many-pages", "no-text"],
)
async def test_failures_end_as_status_failed_with_a_readable_reason(
    document_store: DocumentStore,
    test_session: UUID,
    tmp_path: Path,
    parse,
    error_part: str,  # noqa: ANN001
) -> None:
    """A crash, a too-long document and an empty one all end as 'failed' with a short reason; nothing is raised."""
    await new_document(document_store, test_session, tmp_path)
    claimed = await document_store.claim_next_document(test_session)
    await ingest_document(claimed, deps_for(document_store, tmp_path, parse=parse))
    document = await document_store.get_document(claimed.id)
    assert document.status == "failed" and error_part in document.error
    assert await chunk_rows(document_store, claimed.id) == []


@pytest.mark.integration
async def test_a_failed_picture_description_does_not_fail_the_document(
    document_store: DocumentStore, test_session: UUID, tmp_path: Path
) -> None:
    """Vision returning None (Groq down) -> the picture keeps its caption; the document is still ready."""

    async def broken_describe(picture: Element, heading_path: str) -> None:
        """Simulates describe_picture after Groq failed."""
        return None

    await new_document(document_store, test_session, tmp_path)
    claimed = await document_store.claim_next_document(test_session)
    await ingest_document(claimed, deps_for(document_store, tmp_path, describe=broken_describe))
    document = await document_store.get_document(claimed.id)
    assert document.status == "ready" and document.picture_count == 0
    assert (await chunk_rows(document_store, claimed.id))[2]["content"] == "Figure (page 3): Figure 1: Incidents"


# ------------------------------------------------------------------ saving chunks


def chunks(n: int, word: str) -> list[Chunk]:
    """n simple text chunks."""
    return [Chunk(i, "text", "", 1, 1, f"{word} {i}", f"{word} {i}", 2) for i in range(n)]


@pytest.mark.integration
async def test_saving_chunks_is_idempotent_and_all_or_nothing(
    document_store: DocumentStore, test_session: UUID, tmp_path: Path
) -> None:
    """A second save REPLACES the chunks; a save that fails halfway leaves no chunks and the status unchanged."""
    document = await new_document(document_store, test_session, tmp_path)
    await document_store.claim_next_document(test_session)
    counts = {"page_count": 1, "table_count": 0, "picture_count": 0, "pictures_skipped": 0, "embedding_model": "m"}

    first, second = chunks(3, "alpha"), chunks(2, "beta")
    await document_store.save_chunks_and_mark_ready(
        document.id, test_session, first, FakeEmbedder().embed_documents(["x"] * 3), **counts
    )
    await document_store.save_chunks_and_mark_ready(
        document.id, test_session, second, FakeEmbedder().embed_documents(["y"] * 2), **counts
    )
    assert [r["content"] for r in await chunk_rows(document_store, document.id)] == ["beta 0", "beta 1"]

    other = await new_document(document_store, test_session, tmp_path, name="other.md")
    await document_store.claim_next_document(test_session)
    bad_vectors = [fake_vector("ok"), np.ones(EMBEDDING_DIM - 1, dtype=np.float32)]  # 2nd has the wrong size
    with pytest.raises(psycopg.errors.DataException):
        await document_store.save_chunks_and_mark_ready(
            other.id, test_session, chunks(2, "gamma"), bad_vectors, **counts
        )
    assert await chunk_rows(document_store, other.id) == []  # the first, valid insert was rolled back too
    assert (await document_store.get_document(other.id)).status == "processing"


# ------------------------------------------------------------------ the queue


@pytest.mark.integration
async def test_two_simultaneous_claims_get_different_documents(
    document_store: DocumentStore, test_session: UUID, tmp_path: Path
) -> None:
    """FOR UPDATE SKIP LOCKED: two workers claiming at the same moment never get the same document."""
    created = {(await new_document(document_store, test_session, tmp_path, name=f"{n}.md")).id for n in range(2)}
    first, second = await asyncio.gather(
        document_store.claim_next_document(test_session), document_store.claim_next_document(test_session)
    )
    assert {first.id, second.id} == created
    assert await document_store.claim_next_document(test_session) is None  # queue now empty


@pytest.mark.integration
async def test_recovery_requeues_interrupted_documents_and_fails_them_after_3_attempts(
    document_store: DocumentStore, test_session: UUID, tmp_path: Path
) -> None:
    """'processing' at startup = interrupted: back to 'queued', unless it already had 3 attempts."""
    retry = await new_document(document_store, test_session, tmp_path, name="retry.md")
    give_up = await new_document(document_store, test_session, tmp_path, name="give-up.md")
    for document, attempts in ((retry, 1), (give_up, 3)):
        await document_store.fetch_one(
            "UPDATE app.documents SET status = 'processing', attempts = %s WHERE id = %s RETURNING id",
            (attempts, document.id),
        )
    assert await document_store.recover_interrupted(test_session) == (1, 1)
    assert (await document_store.get_document(retry.id)).status == "queued"
    failed = await document_store.get_document(give_up.id)
    assert failed.status == "failed" and failed.error == "Processing was interrupted 3 times."


# ------------------------------------------------------------------ worker and files (no database)


class FakeQueue:
    """Stands in for DocumentStore in worker tests: hands out documents from a list."""

    def __init__(self, names: list[str]) -> None:
        """Documents waiting in the 'queue': just objects with an id and a filename."""
        self.documents = [SimpleNamespace(id=uuid4(), filename=name) for name in names]

    async def claim_next_document(self) -> SimpleNamespace | None:
        """Pop the next document, or None when empty."""
        return self.documents.pop(0) if self.documents else None


async def test_worker_processes_every_queued_document_then_idles_and_stops_cleanly(tmp_path: Path) -> None:
    """run_once returns True per document and False on an empty queue; the loop survives an ingest error."""
    done = []

    async def ingest(document: SimpleNamespace) -> None:
        """Record the document; 'broken' raises like a bug would."""
        if document.filename == "broken":
            raise RuntimeError("bug")
        done.append(document.filename)

    worker = IngestionWorker(FakeQueue(["a", "broken", "b"]), ingest, tmp_path, idle_seconds=0.01)
    worker.start()
    for _ in range(100):
        if done == ["a", "b"]:
            break
        await asyncio.sleep(0.01)
    await worker.stop()
    assert done == ["a", "b"]  # the error in "broken" didn't stop the loop
    assert await worker.run_once() is False


def test_orphan_cleanup_removes_only_unknown_document_folders(tmp_path: Path) -> None:
    """A folder named by an unknown UUID is removed; known documents and non-UUID folders are left alone."""
    known, orphan = uuid4(), uuid4()
    for folder_id in (known, orphan):
        document_dir(tmp_path, folder_id).mkdir(parents=True)
    (tmp_path / "documents" / "notes").mkdir()

    assert remove_orphan_folders(tmp_path, {known}) == 1
    assert document_dir(tmp_path, known).exists() and not document_dir(tmp_path, orphan).exists()
    assert (tmp_path / "documents" / "notes").exists()
