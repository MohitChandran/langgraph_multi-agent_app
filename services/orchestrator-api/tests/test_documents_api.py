"""Tests for the document endpoints and upload rules (app/documents/) - plan 02, stage R5.

Real routes + real Postgres; files go to a temporary DATA_DIR. Chats are created with the
"[pytest]" prefix (directly, or as the filename of an upload without session_id) and deleted
afterwards, which also deletes their documents.
"""

import io
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest

from app.agents.doc.store import DocumentStore
from app.chat.store import ChatStore
from app.config import get_settings
from app.dependencies import get_chat_store, get_document_store
from app.documents.files import DOCX, JPEG, MARKDOWN, PDF, PNG, PPTX, TEXT, detect_file_type, document_dir
from app.main import app

MD_BYTES = b"# Q3 Report\n\nService credits are claimed within 30 days.\n"


def office_zip(folder: str) -> bytes:
    """A minimal Office-like ZIP: [Content_Types].xml plus one part in `folder` (word/ or ppt/)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr(f"{folder}/document.xml", "<doc/>")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("data", "name", "expected"),
    [
        (b"%PDF-1.7 ...", "anything.bin", PDF),
        (b"\x89PNG\r\n\x1a\n...", "x", PNG),
        (b"\xff\xd8\xff\xe0...", "x", JPEG),
        (office_zip("word"), "report.docx", DOCX),
        (office_zip("ppt"), "slides.pptx", PPTX),
        (MD_BYTES, "notes.md", MARKDOWN),
        (b"plain text", "notes.txt", TEXT),
        (b"MZ\x90\x00 windows exe", "invoice.pdf", None),  # a .pdf name proves nothing
        (office_zip("xl"), "sheet.xlsx", None),  # Excel is not supported
        (b"\xff\xfe not utf-8", "notes.md", None),
        (b"text with \x00 nul", "notes.txt", None),
    ],
    ids=["pdf", "png", "jpeg", "docx", "pptx", "md", "txt", "exe-named-pdf", "xlsx", "bad-utf8", "nul-byte"],
)
def test_file_type_comes_from_the_bytes_not_the_name(data: bytes, name: str, expected) -> None:  # noqa: ANN001
    """Magic bytes decide; the extension only matters for text formats, which have no signature."""
    assert detect_file_type(data, name) == expected


@pytest.fixture
async def client(
    chat_store: ChatStore, document_store: DocumentStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client with the test stores injected and DATA_DIR pointing at a temporary folder."""
    monkeypatch.setattr(get_settings(), "data_dir", tmp_path)
    app.dependency_overrides[get_chat_store] = lambda: chat_store
    app.dependency_overrides[get_document_store] = lambda: document_store
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            yield http
    finally:
        app.dependency_overrides.clear()
        await chat_store.fetch_one(
            "WITH gone AS (DELETE FROM app.chat_sessions WHERE title LIKE '[pytest]%%' RETURNING 1)"
            " SELECT count(*) FROM gone"
        )


async def upload(client: httpx.AsyncClient, name: str, data: bytes, session_id: UUID | None = None) -> httpx.Response:
    """POST a file as multipart form data, optionally into an existing chat."""
    form = {"session_id": str(session_id)} if session_id else {}
    return await client.post("/api/documents", files={"file": (name, data, "application/octet-stream")}, data=form)


@pytest.mark.integration
async def test_upload_without_a_chat_creates_one_and_queues_the_document(
    client: httpx.AsyncClient, tmp_path: Path
) -> None:
    """202: a new chat titled with the filename; the document is queued and its bytes are on disk."""
    response = await upload(client, "[pytest] q3 report.md", MD_BYTES)
    assert response.status_code == 202
    body = response.json()
    assert body["session"]["title"] == "[pytest] q3 report.md"
    document = body["document"]
    assert document["status"] == "queued" and document["mime_type"] == "text/markdown"
    assert document["size_bytes"] == len(MD_BYTES) and document["session_id"] == body["session"]["id"]
    assert (document_dir(tmp_path, UUID(document["id"])) / "original.md").read_bytes() == MD_BYTES


@pytest.mark.integration
async def test_rejected_uploads(client: httpx.AsyncClient, test_session: UUID, monkeypatch: pytest.MonkeyPatch) -> None:
    """415 for a disguised executable or an empty file, 404 for an unknown chat, 413 over the size limit."""
    assert (await upload(client, "invoice.pdf", b"MZ\x90\x00 not a pdf", test_session)).status_code == 415
    assert (await upload(client, "empty.md", b"", test_session)).status_code == 415
    assert (await upload(client, "notes.md", MD_BYTES, uuid4())).status_code == 404
    monkeypatch.setattr(get_settings(), "rag_max_file_mb", 0)
    assert (await upload(client, "notes.md", MD_BYTES, test_session)).status_code == 413


@pytest.mark.integration
async def test_same_file_twice_in_one_chat_is_processed_once_but_another_chat_gets_its_own(
    client: httpx.AsyncClient, test_session: UUID, chat_store: ChatStore
) -> None:
    """Duplicate in the same chat -> the same document; same bytes in another chat -> a new document."""
    first = (await upload(client, "notes.md", MD_BYTES, test_session)).json()["document"]
    again = (await upload(client, "renamed.md", MD_BYTES, test_session)).json()["document"]
    assert again["id"] == first["id"]

    other_chat = await chat_store.create_session("[pytest] other chat")
    elsewhere = (await upload(client, "notes.md", MD_BYTES, other_chat.id)).json()["document"]
    assert elsewhere["id"] != first["id"]


@pytest.mark.integration
async def test_a_failed_document_uploaded_again_is_requeued(
    client: httpx.AsyncClient, test_session: UUID, document_store: DocumentStore
) -> None:
    """Re-uploading a file that failed gives it another chance instead of returning the failure."""
    first = (await upload(client, "notes.md", MD_BYTES, test_session)).json()["document"]
    await document_store.mark_failed(UUID(first["id"]), "Processing failed (RuntimeError).")
    again = (await upload(client, "notes.md", MD_BYTES, test_session)).json()["document"]
    assert again["id"] == first["id"] and again["status"] == "queued" and again["error"] is None


@pytest.mark.integration
async def test_a_path_in_the_filename_never_reaches_the_disk(
    client: httpx.AsyncClient, test_session: UUID, tmp_path: Path
) -> None:
    """'../../etc/passwd.md' is stored under our own UUID folder; only the base name is kept for display."""
    document = (await upload(client, "../../etc/passwd.md", MD_BYTES, test_session)).json()["document"]
    assert document["filename"] == "passwd.md"
    stored = list(tmp_path.rglob("*.md"))
    assert stored == [document_dir(tmp_path, UUID(document["id"])) / "original.md"]


@pytest.mark.integration
async def test_list_download_and_delete(client: httpx.AsyncClient, test_session: UUID, tmp_path: Path) -> None:
    """List the chat's documents, download the original (inline, nosniff), then delete rows + folder."""
    document_id = (await upload(client, "notes.md", MD_BYTES, test_session)).json()["document"]["id"]

    listed = (await client.get(f"/api/sessions/{test_session}/documents")).json()
    assert [d["id"] for d in listed] == [document_id]
    assert (await client.get(f"/api/sessions/{uuid4()}/documents")).status_code == 404

    file = await client.get(f"/api/documents/{document_id}/file")
    assert file.status_code == 200 and file.content == MD_BYTES
    assert file.headers["x-content-type-options"] == "nosniff"
    assert file.headers["content-disposition"].startswith("inline")

    assert (await client.delete(f"/api/documents/{document_id}")).status_code == 204
    assert not document_dir(tmp_path, UUID(document_id)).exists()
    assert (await client.get(f"/api/documents/{document_id}")).status_code == 404
    assert (await client.delete(f"/api/documents/{document_id}")).status_code == 404


@pytest.mark.integration
async def test_picture_endpoint_only_serves_files_inside_the_documents_folder(
    client: httpx.AsyncClient, test_session: UUID, tmp_path: Path, document_store: DocumentStore
) -> None:
    """A picture chunk's crop is served as PNG; a stored path pointing outside the folder gives 404."""
    document_id = UUID((await upload(client, "notes.md", MD_BYTES, test_session)).json()["document"]["id"])
    crop = document_dir(tmp_path, document_id) / "pictures" / "p001-01.png"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    vector = "[" + ",".join(["0.05"] * 384) + "]"
    ids = []
    for image_path in (str(crop), "/etc/hostname"):
        row = await document_store.fetch_one(
            "INSERT INTO app.document_chunks (document_id, session_id, chunk_index, chunk_type, page_start, page_end,"
            " content, embed_text, image_path, token_count, embedding)"
            " VALUES (%s, %s, %s, 'picture', 1, 1, 'Figure', 'Figure', %s, 1, %s::vector) RETURNING id",
            (document_id, test_session, len(ids), image_path, vector),
        )
        ids.append(row["id"])

    good = await client.get(f"/api/documents/{document_id}/pictures/{ids[0]}")
    assert good.status_code == 200 and good.headers["content-type"] == "image/png"
    assert (await client.get(f"/api/documents/{document_id}/pictures/{ids[1]}")).status_code == 404
