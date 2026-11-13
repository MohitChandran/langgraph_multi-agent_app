"""Data shapes of the Doc QA agent (plan 02, sections 5.4, 5.6, 6.4).

Our OWN types: only parsing.py ever touches Docling's objects. Everything after it (picture
selection, chunking, tests, the eval) works with these plain dataclasses, so:
  - tests can build a document by hand in a few lines, without running Docling;
  - switching to another parser later means rewriting one adapter, not the whole pipeline.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

ElementKind = Literal["heading", "text", "table", "picture"]


@dataclass
class Element:
    """One piece of a parsed document, in reading order.

    heading -> text = the heading, level = 1 (top section), 2 (subsection), ...
    text    -> a paragraph, list item, footnote or code block
    table   -> table_rows (header row first), text = caption
    picture -> image_path (saved PNG crop), text = caption, plus what the picture filter needs
    """

    kind: ElementKind
    text: str
    page_start: int  # 1-based
    page_end: int
    level: int = 0  # headings only
    table_rows: list[list[str]] | None = None  # tables only; first row = header
    image_path: str | None = None  # pictures only
    picture_label: str | None = None  # Docling's classification, e.g. "bar_chart", "logo"
    area_fraction: float | None = None  # picture area / page area (None when the format has no pages)
    width_px: int | None = None
    height_px: int | None = None
    description: str | None = None  # filled by vision.py (stage R3)


@dataclass
class ParsedDocument:
    """A whole document after parsing: its title, page count and elements in reading order."""

    title: str
    page_count: int
    elements: list[Element]


@dataclass
class Chunk:
    """One searchable piece of a document (plan 02, section 5.6).

    `content` is what the LLM reads and the UI shows. `embed_text` adds the document title and
    the heading path in front ("contextual chunk header"), so a chunk that only says "the
    credit is 10%" is still found for "AWS SLA credit": it's embedded and keyword-indexed.
    """

    chunk_index: int  # 0..n-1 in document order
    chunk_type: Literal["text", "table", "picture"]
    heading_path: str  # "4 Cloud Vendors > 4.2 SLA Terms" ("" before the first heading)
    page_start: int
    page_end: int
    content: str
    embed_text: str
    token_count: int  # tokens of embed_text, counted with the embedding model's tokenizer
    image_path: str | None = None  # picture chunks only


DocumentStatus = Literal["queued", "processing", "ready", "failed"]


class Document(BaseModel):
    """An uploaded document as the API returns it (plan 02, section 7).

    `stage` and `progress` describe live processing ("describing_pictures", "pictures 3/12"):
    the UI polls this while status is queued/processing.
    """

    id: UUID
    session_id: UUID
    filename: str
    mime_type: str
    size_bytes: int
    title: str | None
    status: DocumentStatus
    stage: str | None
    progress: str | None
    error: str | None
    page_count: int | None
    chunk_count: int | None
    table_count: int | None
    picture_count: int | None
    pictures_skipped: int | None
    created_at: datetime
    finished_at: datetime | None
