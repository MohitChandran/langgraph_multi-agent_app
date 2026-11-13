"""Document endpoints (plan 02, section 7).

    POST   /api/documents                          upload (multipart: file, optional session_id) -> 202
    GET    /api/sessions/{id}/documents            a chat's documents (the UI polls this for status)
    GET    /api/documents/{id}                     one document
    DELETE /api/documents/{id}                     delete rows + files
    GET    /api/documents/{id}/file                the original file (opens inline, e.g. PDF at #page=4)
    GET    /api/documents/{id}/pictures/{chunk_id} a picture crop shown next to a citation

Upload only VALIDATES and QUEUES (202 Accepted = "received, not done yet"): parsing takes
minutes, so it happens in the background worker. The response tells the UI which chat and
document to poll.
"""

import asyncio
import hashlib
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import psycopg
from fastapi import APIRouter, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.agents.doc.models import Document
from app.agents.doc.store import DocumentStore
from app.chat.models import Session
from app.chat.service import make_title
from app.config import get_settings
from app.dependencies import ChatStoreDep, DocumentStoreDep
from app.documents.files import (
    FileType,
    delete_document_files,
    detect_file_type,
    document_dir,
    find_original,
    save_original,
)

router = APIRouter(prefix="/api", tags=["documents"])

# Files are served with this header so a browser never "sniffs" an uploaded file into a
# different type (e.g. treating a crafted .txt as HTML and running scripts in it).
NO_SNIFF = {"X-Content-Type-Options": "nosniff"}


class UploadResponse(BaseModel):
    """The chat the document belongs to (new or existing) and the document itself."""

    session: Session
    document: Document


@router.post("/documents", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile,
    chat_store: ChatStoreDep,
    store: DocumentStoreDep,
    session_id: Annotated[UUID | None, Form()] = None,
) -> UploadResponse:
    """Validate an upload, store it and queue it for processing (plan 5.1).

    No session_id -> a new chat titled with the filename (chats are created lazily, like
    POST /api/chat). The same bytes uploaded to the same chat again return the existing
    document instead of processing it twice; a FAILED one is re-queued.
    413 too big, 415 unsupported type, 404 unknown chat.
    """
    settings = get_settings()
    max_bytes = settings.rag_max_file_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)  # never read more than the limit + 1 byte into memory
    if len(data) > max_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"The file is over {settings.rag_max_file_mb} MB.")
    filename = Path(file.filename or "").name.strip()[:255] or "upload"  # display only, never a path
    file_type = detect_file_type(data, filename) if data else None
    if file_type is None:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "Unsupported or empty file. Upload a PDF, DOCX, PPTX, Markdown, TXT, PNG or JPG file.",
        )

    session = (
        await chat_store.create_session(make_title(filename))
        if session_id is None
        else await chat_store.get_session(session_id)
    )
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Chat {session_id} does not exist.")

    sha256 = hashlib.sha256(data).hexdigest()
    document = await _store_upload(store, settings.data_dir, session.id, filename, file_type, data, sha256)
    session = await chat_store.touch_session(session.id) or session
    return UploadResponse(session=session, document=document)


async def _store_upload(
    store: DocumentStore, data_dir: Path, session_id: UUID, filename: str, file_type: FileType, data: bytes, sha256: str
) -> Document:
    """Save the bytes and the row for a new upload, or reuse/re-queue a previous upload of the same file.

    Files are written first, then the row, so a document row always has its file. If the
    insert fails, the files are removed again. Two simultaneous uploads of the same file both
    pass the duplicate check; the UNIQUE (session_id, sha256) constraint lets only one insert
    win, and the other returns the winner's document.
    """
    existing = await store.find_by_hash(session_id, sha256)
    if existing is not None and existing.status != "failed":
        return existing
    if existing is not None:  # same file failed before: try again
        await asyncio.to_thread(save_original, data_dir, existing.id, file_type, data)
        return await store.requeue(existing.id)

    document_id = uuid4()
    await asyncio.to_thread(save_original, data_dir, document_id, file_type, data)
    try:
        return await store.create_document(document_id, session_id, filename, file_type.mime_type, len(data), sha256)
    except psycopg.errors.UniqueViolation:
        delete_document_files(data_dir, document_id)
        return await store.find_by_hash(session_id, sha256)
    except Exception:
        delete_document_files(data_dir, document_id)
        raise


@router.get("/sessions/{session_id}/documents")
async def list_documents(session_id: UUID, chat_store: ChatStoreDep, store: DocumentStoreDep) -> list[Document]:
    """A chat's documents with their live status, oldest first."""
    if await chat_store.get_session(session_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Chat {session_id} does not exist.")
    return await store.list_documents(session_id)


@router.get("/documents/{document_id}")
async def get_document(document_id: UUID, store: DocumentStoreDep) -> Document:
    """One document."""
    return await _document_or_404(store, document_id)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: UUID, store: DocumentStoreDep) -> Response:
    """Delete a document: rows first (chunks cascade), then its folder (plan gotcha 12)."""
    if not await store.delete_document(document_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document {document_id} does not exist.")
    await asyncio.to_thread(delete_document_files, get_settings().data_dir, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/documents/{document_id}/file")
async def download_original(document_id: UUID, store: DocumentStoreDep) -> FileResponse:
    """The original file, shown inline: a PDF link with #page=4 opens the browser's viewer at page 4."""
    document = await _document_or_404(store, document_id)
    path = find_original(get_settings().data_dir, document_id)
    if path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "The file is missing on the server.")
    return FileResponse(
        path, media_type=document.mime_type, filename=document.filename, content_disposition_type="inline",
        headers=NO_SNIFF,
    )  # fmt: skip


@router.get("/documents/{document_id}/pictures/{chunk_id}")
async def picture(document_id: UUID, chunk_id: int, store: DocumentStoreDep) -> FileResponse:
    """The crop of a picture chunk (for thumbnails next to citations).

    The stored path must lie inside this document's folder: even a corrupted row can't make
    this endpoint serve another file from the server.
    """
    stored = await store.picture_path(document_id, chunk_id)
    folder = document_dir(get_settings().data_dir, document_id).resolve()
    if stored is None or not Path(stored).resolve().is_relative_to(folder) or not Path(stored).is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such picture.")
    return FileResponse(stored, media_type="image/png", headers=NO_SNIFF)


async def _document_or_404(store: DocumentStore, document_id: UUID) -> Document:
    """Load a document or answer 404."""
    document = await store.get_document(document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document {document_id} does not exist.")
    return document
