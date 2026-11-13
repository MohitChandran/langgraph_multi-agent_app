"""Uploaded files on disk: what type a file really is, and where it's stored (plan 02, sections 4.3 and 5.1).

Layout (DATA_DIR, git-ignored):
    data/documents/<document_id>/original.<ext>
    data/documents/<document_id>/pictures/p006-01.png

Security rules (plan gotcha 11):
  - The type comes from the file's first bytes ("magic bytes"), never from its name or the
    browser's Content-Type: both are chosen by the uploader and prove nothing.
  - The uploaded filename is stored for display only and NEVER used in a path. Paths are built
    from our own UUID, so "../../etc/passwd" can't escape the data folder.
"""

import io
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID


@dataclass(frozen=True)
class FileType:
    """A supported upload type: the MIME type we store and serve, and the extension Docling reads it by."""

    mime_type: str
    extension: str


PDF = FileType("application/pdf", "pdf")
DOCX = FileType("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx")
PPTX = FileType("application/vnd.openxmlformats-officedocument.presentationml.presentation", "pptx")
PNG = FileType("image/png", "png")
JPEG = FileType("image/jpeg", "jpg")
MARKDOWN = FileType("text/markdown", "md")
# Plain text is saved with the .md extension: Docling has no plain-text reader, and plain
# paragraphs are valid Markdown. It's still served back as text/plain.
TEXT = FileType("text/plain", "md")


def detect_file_type(data: bytes, filename: str) -> FileType | None:
    """The real type of an uploaded file, or None if it isn't one we accept (-> HTTP 415).

    Binary formats are recognised by their signature. DOCX and PPTX are both ZIP files, told
    apart by the folders inside (word/ or ppt/). Markdown and text have no signature, so for
    those the extension is a hint, and the bytes must decode as UTF-8 with no NUL bytes.
    """
    if data.startswith(b"%PDF-"):
        return PDF
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return PNG
    if data.startswith(b"\xff\xd8\xff"):
        return JPEG
    if data.startswith(b"PK\x03\x04"):
        return _office_type(data)
    suffix = Path(filename).suffix.lower()
    if suffix in {".md", ".markdown", ".txt"} and b"\x00" not in data:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            return None
        return TEXT if suffix == ".txt" else MARKDOWN
    return None


def _office_type(data: bytes) -> FileType | None:
    """DOCX or PPTX if the ZIP has Office's [Content_Types].xml and a word/ or ppt/ folder; else None.

    Only the ZIP's table of contents is read (namelist), nothing is extracted, so a "zip bomb"
    can't blow up memory here.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile:
        return None
    if "[Content_Types].xml" not in names:
        return None
    if any(name.startswith("word/") for name in names):
        return DOCX
    if any(name.startswith("ppt/") for name in names):
        return PPTX
    return None


def document_dir(data_dir: Path, document_id: UUID) -> Path:
    """The folder holding one document's files, named by OUR id only."""
    return data_dir / "documents" / str(document_id)


def original_path(data_dir: Path, document_id: UUID, file_type: FileType) -> Path:
    """Where the uploaded bytes are stored: .../<id>/original.<ext>."""
    return document_dir(data_dir, document_id) / f"original.{file_type.extension}"


def find_original(data_dir: Path, document_id: UUID) -> Path | None:
    """The stored original file of a document (whatever its extension), or None if it's missing."""
    matches = sorted(document_dir(data_dir, document_id).glob("original.*"))
    return matches[0] if matches else None


def save_original(data_dir: Path, document_id: UUID, file_type: FileType, data: bytes) -> Path:
    """Write the uploaded bytes to the document's folder (created if needed) and return the path."""
    path = original_path(data_dir, document_id, file_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def delete_document_files(data_dir: Path, document_id: UUID) -> None:
    """Remove a document's folder (original + picture crops). Missing folder = nothing to do."""
    shutil.rmtree(document_dir(data_dir, document_id), ignore_errors=True)


def remove_orphan_folders(data_dir: Path, known_ids: set[UUID]) -> int:
    """Delete document folders that have no database row (left by a crash or a deleted chat). Returns the count.

    Only folders whose name IS a UUID are touched; anything else in data/documents is left alone.
    """
    root = data_dir / "documents"
    if not root.is_dir():
        return 0
    removed = 0
    for folder in root.iterdir():
        try:
            folder_id = UUID(folder.name)
        except ValueError:
            continue
        if folder.is_dir() and folder_id not in known_ids:
            shutil.rmtree(folder, ignore_errors=True)
            removed += 1
    return removed
