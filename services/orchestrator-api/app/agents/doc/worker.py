"""Background ingestion worker: Postgres as the job queue (plan 02, section 5.2, decision D10).

One asyncio task inside the API process:

    loop: claim the oldest 'queued' document -> ingest it -> repeat
          queue empty -> sleep 2 s and look again

Why Postgres instead of Redis/Celery: the queue IS the documents table. A job survives a
restart (it's a row), the UI reads status from the same row, and there is nothing extra to
run. `FOR UPDATE SKIP LOCKED` (store.claim_next_document) makes claiming safe even if several
workers ever run. One document at a time: parsing already uses all CPU cores.

Crash recovery happens once at startup, BEFORE the loop starts (recover()):
  - documents stuck in 'processing' go back to 'queued' (or 'failed' after 3 attempts);
  - folders on disk without a database row are deleted.
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from app.agents.doc.models import Document
from app.agents.doc.store import DocumentStore
from app.documents.files import remove_orphan_folders

logger = logging.getLogger(__name__)

IDLE_SECONDS = 2.0


class IngestionWorker:
    """Runs ingest(document) for every queued document, one at a time, until stopped."""

    def __init__(
        self,
        store: DocumentStore,
        ingest: Callable[[Document], Awaitable[None]],
        data_dir: Path,
        idle_seconds: float = IDLE_SECONDS,
    ) -> None:
        """Keep the dependencies; nothing runs until start()."""
        self._store = store
        self._ingest = ingest
        self._data_dir = data_dir
        self._idle_seconds = idle_seconds
        self._task: asyncio.Task | None = None

    async def recover(self) -> None:
        """Startup clean-up after a crash or restart. Must run before start(), while no worker is active."""
        requeued, failed = await self._store.recover_interrupted()
        orphans = remove_orphan_folders(self._data_dir, await self._store.all_document_ids())
        logger.info("Ingestion recovery: %d re-queued, %d failed, %d orphan folders removed", requeued, failed, orphans)

    def start(self) -> None:
        """Start the loop as a background task (returns immediately)."""
        self._task = asyncio.create_task(self._run(), name="ingestion-worker")

    async def stop(self) -> None:
        """Cancel the loop and wait for it to end. A document being processed stays 'processing'
        and is re-queued by recover() at the next startup."""
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def run_once(self) -> bool:
        """Claim and ingest one document. Returns False if the queue was empty. (Tests call this directly.)"""
        document = await self._store.claim_next_document()
        if document is None:
            return False
        logger.info("Ingesting document %s (%s), attempt started", document.id, document.filename)
        await self._ingest(document)
        return True

    async def _run(self) -> None:
        """The loop. An unexpected error (e.g. Postgres briefly down) is logged and the loop keeps going."""
        while True:
            try:
                worked = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Ingestion worker error; retrying in %.0fs", self._idle_seconds)
                worked = False
            if not worked:
                await asyncio.sleep(self._idle_seconds)
