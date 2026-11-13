"""Ingest one document: parse -> describe pictures -> chunk -> embed -> store (plan 02, section 5.3).

The worker (worker.py) calls ingest_document() for each document it claims. Every stage is
written to the database as it starts, so the UI's status chip can show "Describing pictures
3/12" live. The last step stores all chunks and marks the document 'ready' in ONE transaction.

Any failure marks the document 'failed' with a short, readable reason; the full traceback goes
to the log. Nothing is re-raised: one bad document must not stop the worker.

The heavy parts (Docling, the vision model, the embedding model) are passed in as IngestDeps,
so tests can run the whole pipeline with fakes in milliseconds.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pypdfium2

from app.agents.doc.chunking import chunk_document, heading_paths
from app.agents.doc.models import Document, Element, ParsedDocument
from app.agents.doc.store import DocumentStore
from app.agents.doc.vision import select_pictures
from app.documents.files import document_dir, find_original

logger = logging.getLogger(__name__)


class IngestError(Exception):
    """A failure with a message that is safe and useful to show the user (e.g. "too many pages")."""


class EmbedsDocuments(Protocol):
    """What ingestion needs from an embedder: batch embedding plus its model name."""

    model_name: str

    def embed_documents(self, texts: list[str]) -> Sequence[np.ndarray]:
        """One vector per text."""
        ...


class CountsTokens(Protocol):
    """What chunking needs from a token counter."""

    def count(self, text: str) -> int:
        """Number of tokens in text."""
        ...


@dataclass
class IngestDeps:
    """Everything ingest_document needs, injected so tests can swap in fakes.

    parse:    (source file, pictures folder, fallback title) -> ParsedDocument. Blocking (Docling).
    describe: (picture, heading path) -> description or None. Async (Groq vision).
    """

    store: DocumentStore
    data_dir: Path
    parse: Callable[[Path, Path, str], ParsedDocument]
    describe: Callable[[Element, str], Awaitable[str | None]]
    embedder: EmbedsDocuments
    counter: CountsTokens
    chunk_tokens: int
    max_pages: int
    max_pictures: int
    min_picture_area: float


async def ingest_document(document: Document, deps: IngestDeps) -> None:
    """Run every stage for one claimed document. Never raises: failures end as status 'failed'."""
    try:
        await _ingest(document, deps)
    except IngestError as error:
        logger.warning("Document %s rejected: %s", document.id, error)
        await deps.store.mark_failed(document.id, str(error))
    except Exception as error:  # a bug or an unexpected file: log it all, show a short reason
        logger.exception("Document %s failed", document.id)
        await deps.store.mark_failed(document.id, f"Processing failed ({type(error).__name__}).")


async def _ingest(document: Document, deps: IngestDeps) -> None:
    """The stages themselves (plan table 5.3). Raises IngestError for user-facing failures."""
    store = deps.store
    source = find_original(deps.data_dir, document.id)
    if source is None:
        raise IngestError("The uploaded file is missing on the server.")
    _check_pdf_pages(source, deps.max_pages)

    # 1. parsing (stage already set by the claim). CPU-heavy: in a thread, so the API stays responsive.
    pictures_dir = document_dir(deps.data_dir, document.id) / "pictures"
    parsed = await asyncio.to_thread(deps.parse, source, pictures_dir, Path(document.filename).stem)
    if parsed.page_count > deps.max_pages:
        raise IngestError(f"The document has {parsed.page_count} pages; the limit is {deps.max_pages}.")
    await store.set_parsed(document.id, parsed.title, parsed.page_count)

    # 2. describing_pictures: one Groq call per kept picture; a failed one keeps its caption only.
    to_describe, skipped = select_pictures(parsed.elements, deps.max_pictures, deps.min_picture_area)
    paths = dict(zip(map(id, parsed.elements), heading_paths(parsed.elements), strict=True))
    described = 0
    for number, picture in enumerate(to_describe, start=1):
        await store.set_stage(document.id, "describing_pictures", f"pictures {number}/{len(to_describe)}")
        picture.description = await deps.describe(picture, paths[id(picture)])
        described += picture.description is not None

    # 3. chunking
    await store.set_stage(document.id, "chunking")
    chunks = await asyncio.to_thread(chunk_document, parsed, deps.counter, deps.chunk_tokens)
    if not chunks:
        raise IngestError("No text could be extracted from this document.")

    # 4. embedding, then store everything + mark ready in one transaction
    await store.set_stage(document.id, "embedding", f"{len(chunks)} chunks")
    vectors = await asyncio.to_thread(deps.embedder.embed_documents, [chunk.embed_text for chunk in chunks])
    await store.save_chunks_and_mark_ready(
        document.id,
        document.session_id,
        chunks,
        vectors,
        page_count=parsed.page_count,
        table_count=sum(element.kind == "table" for element in parsed.elements),
        picture_count=described,
        pictures_skipped=len(skipped),
        embedding_model=deps.embedder.model_name,
    )
    logger.info("Document %s ready: %d chunks, %d pictures described", document.id, len(chunks), described)


def _check_pdf_pages(source: Path, max_pages: int) -> None:
    """Reject an oversized PDF BEFORE Docling spends minutes on it (counting pages takes milliseconds)."""
    if source.suffix != ".pdf":
        return
    try:
        pdf = pypdfium2.PdfDocument(source)
    except pypdfium2.PdfiumError as error:
        raise IngestError("The PDF could not be opened (damaged or password-protected).") from error
    try:
        pages = len(pdf)
    finally:
        pdf.close()
    if pages > max_pages:
        raise IngestError(f"The document has {pages} pages; the limit is {max_pages}.")
