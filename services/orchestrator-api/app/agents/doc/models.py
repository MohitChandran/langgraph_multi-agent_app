"""Data shapes of the Doc QA agent (plan 02, sections 5.4, 5.6, 6.4).

Our OWN types: only parsing.py ever touches Docling's objects. Everything after it (picture
selection, chunking, tests, the eval) works with these plain dataclasses, so:
  - tests can build a document by hand in a few lines, without running Docling;
  - switching to another parser later means rewriting one adapter, not the whole pipeline.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypedDict
from uuid import UUID

from pydantic import BaseModel, Field

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


@dataclass
class Hit:
    """A chunk found by retrieval, with how each step ranked it (plan 02, section 6.2).

    dense_rank / keyword_rank are 1-based positions in each search's list (None = that search
    didn't find it). rrf_score comes from fusing both lists; rerank_score from the cross-encoder.
    """

    chunk_id: int
    document_id: UUID
    filename: str
    chunk_type: str
    heading_path: str
    page_start: int
    page_end: int
    content: str
    token_count: int
    image_path: str | None
    dense_rank: int | None = None
    keyword_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None


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


# ---------------------------------------------------------------- question answering (R7)


class DocAnswer(BaseModel):
    """What the answer LLM must return (structured output, like the DB agent's SQLDraft).

    The Field descriptions are sent to the model inside the JSON schema, so they are instructions too.
    """

    answer_found: bool = Field(description="false if the numbered sources do not contain the answer.")
    answer: str = Field(description="The answer in markdown, with [n] after every sentence that uses source n.")
    citations: list[int] = Field(description="Numbers of the sources the answer used, e.g. [1, 3].")


class Citation(BaseModel):
    """One source the answer cites, with everything the UI needs to show and open it."""

    n: int  # the [n] used in the answer text
    chunk_id: int
    document_id: UUID
    filename: str
    page_start: int
    page_end: int
    heading_path: str
    chunk_type: str
    snippet: str  # the first ~200 characters of the chunk
    picture_url: str | None = None  # picture chunks: the crop, shown as a thumbnail


class RetrievalTrace(BaseModel):
    """How one candidate chunk was ranked at each step ("Show retrieval" panel, eval, debugging)."""

    chunk_id: int
    chunk_type: str
    page_start: int
    dense_rank: int | None
    keyword_rank: int | None
    rrf_score: float
    rerank_score: float | None
    used: bool  # became one of the answer's numbered sources


class DocAgentResult(BaseModel):
    """The Doc agent's output; saved as chat_messages.payload = {"agent": "doc", "result": ...} (plan 6.4)."""

    question: str
    answer: str
    success: bool = Field(description="True when the answer is grounded in at least one cited source.")
    grounded: bool
    standalone_query: str = Field(description="The question rewritten to stand alone (= the question without history).")
    citations: list[Citation] = Field(default_factory=list)
    retrieval: list[RetrievalTrace] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list, description="Short sentences on what the agent did and why.")
    error: str | None = None


class DocAgentState(TypedDict, total=False):
    """What the DocAgent's workflow knows at each step (LangGraph merges what each node returns)."""

    question: str
    history: str  # formatted conversation ("" = none)
    session_id: UUID
    standalone_query: str
    dense_ids: list[int]  # written by dense_search
    keyword_ids: list[int]  # written by keyword_search (in parallel: separate keys, so no merge conflict)
    candidates: list[Hit]  # fused, then reranked
    sources: list[Hit]  # the numbered sources given to the answer LLM
    answer: DocAnswer
    error: str | None
