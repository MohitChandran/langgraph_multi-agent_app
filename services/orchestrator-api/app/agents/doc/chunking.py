"""Structure-aware chunking with a token budget and a heading-path prefix (plan 02, section 5.6).

Pure Python: no models (except the tokenizer), no database, so it's fast and fully unit-testable.

Why not "split every 1,000 characters with overlap"? A fixed-size cut lands mid-sentence or
mid-table, mixes two sections into one chunk, and loses which section a chunk belongs to.
Three rules instead (decision D4):
  1. STRUCTURE-AWARE: cut only where the document has a boundary (heading, paragraph, sentence,
     table row). Tables and pictures are always their own chunks.
  2. TOKEN BUDGET: a chunk's content holds at most `max_tokens` (350) tokens, counted with the
     EMBEDDING MODEL's own tokenizer (the model's real limit is in tokens, not characters).
     Paragraphs of one section are packed together up to that budget: one idea per chunk.
  3. HEADING PREFIX: every chunk is embedded with "title + heading path" in front, so a chunk
     saying "the credit is 10%" still says WHICH credit ("4 Cloud Vendors > 4.2 SLA Terms").
No overlap: we never cut blindly, so there is nothing to patch up.

Note: plan 02's rule 4 ("merge a tiny tail into the previous chunk if both fit") is not
implemented: rule 2's greedy packing already uses the same test, so two pieces only end up
separate BECAUSE they don't fit together, and the merge could never happen. Short sections
stay as they are (a short section is still a real section).
"""

import re

from tokenizers import Tokenizer

from app.agents.doc.models import Chunk, Element, ParsedDocument

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
PLACEHOLDER_HEADER = re.compile(r"^\d+$")  # pandas names header-less columns 0, 1, 2, ...
CAPTION_PREFIX = re.compile(r"^(table|figure)\b", re.IGNORECASE)


class TokenCounter:
    """Counts tokens exactly like the embedding model does (its own tokenizer.json, downloaded once)."""

    def __init__(self, model_name: str) -> None:
        """Load the model's tokenizer from Hugging Face (cached locally after the first download)."""
        self._tokenizer = Tokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        """Number of tokens in `text`, without the [CLS]/[SEP] special tokens the model adds itself."""
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


class HeadingStack:
    """Tracks which sections we are inside, to build paths like "4 Cloud Vendors > 4.2 SLA Terms"."""

    def __init__(self) -> None:
        """Start outside any section."""
        self._stack: list[tuple[int, str]] = []

    def push(self, level: int, text: str) -> None:
        """Enter a heading of `level`: leave every open section at the same or a deeper level first.

        A new level-1 heading closes "4 ..." AND "4.2 ..."; a level-2 heading only closes the previous level 2.
        """
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, text))

    @property
    def path(self) -> str:
        """The current heading path, outermost first ("" before the first heading)."""
        return " > ".join(text for _, text in self._stack)


def heading_paths(elements: list[Element]) -> list[str]:
    """The heading path in effect at each element (same order). A heading's own path includes itself.

    Used by the picture step to tell the vision model which section a figure belongs to.
    """
    stack, paths = HeadingStack(), []
    for element in elements:
        if element.kind == "heading":
            stack.push(element.level, element.text)
        paths.append(stack.path)
    return paths


class _Chunker:
    """Walks the elements once, keeping a heading stack and a buffer of text waiting to become a chunk."""

    def __init__(self, title: str, counter: TokenCounter, max_tokens: int) -> None:
        """Set the budget and start with no chunks, no open sections and an empty buffer."""
        self.title, self.counter = title, counter
        self.max_tokens = max_tokens
        self.chunks: list[Chunk] = []
        self.headings = HeadingStack()
        self.buffer: list[str] = []
        self.buffer_pages: tuple[int, int] | None = None

    # ---- rule 2: pack paragraphs of one section up to the budget

    def add_text(self, text: str, page_start: int, page_end: int) -> None:
        """Append text to the buffer if it still fits the budget; otherwise flush and start a new buffer.

        An element too big on its own (rule 3) is split at sentence ends, and a sentence too big
        on its own is split by words, so no chunk ever exceeds the budget.
        """
        if self.counter.count(text) > self.max_tokens:
            pieces = SENTENCE_END.split(text)
            if len(pieces) == 1:  # one giant "sentence": fall back to words
                pieces = self._split_words(text)
            for piece in pieces:
                self.add_text(piece, page_start, page_end)
            return

        candidate = "\n\n".join([*self.buffer, text]) if self.buffer else text
        if self.buffer and self.counter.count(candidate) > self.max_tokens:
            self.flush()
        self.buffer.append(text)
        if self.buffer_pages is None:
            self.buffer_pages = (page_start, page_end)
        else:
            self.buffer_pages = (min(self.buffer_pages[0], page_start), max(self.buffer_pages[1], page_end))

    def _split_words(self, text: str) -> list[str]:
        """Split a text with no sentence ends into pieces of at most max_tokens tokens, at word boundaries."""
        pieces, current = [], []
        for word in text.split():
            if current and self.counter.count(" ".join([*current, word])) > self.max_tokens:
                pieces.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            pieces.append(" ".join(current))
        return pieces

    def flush(self) -> None:
        """Turn the buffered paragraphs into one text chunk and empty the buffer."""
        if not self.buffer:
            return
        content = "\n\n".join(self.buffer)
        page_start, page_end = self.buffer_pages
        self.buffer, self.buffer_pages = [], None
        self.emit("text", content, page_start, page_end)

    # ---- rule 5: tables, split by whole rows with the header in every piece

    def add_table(self, element: Element) -> None:
        """Emit a table as markdown chunk(s); a table over budget is split by rows, header repeated."""
        self.flush()
        rows = [[_clean_cell(cell) for cell in row] for row in (element.table_rows or [])]
        if not rows:
            return
        header, data = rows[0], rows[1:]
        if data and all(PLACEHOLDER_HEADER.match(cell) for cell in header):  # no real header: use the first row
            header, data = data[0], data[1:]

        caption = element.text
        caption_line = caption if not caption or CAPTION_PREFIX.match(caption) else f"Table: {caption}"
        whole = _render_table(caption_line, header, data)
        if self.counter.count(whole) <= self.max_tokens or len(data) <= 1:
            self.emit("table", whole, element.page_start, element.page_end)
            return

        start, total = 0, len(data)
        while start < total:
            end = start + 1  # a piece always gets at least one row, even if that row alone is too long
            while (
                end < total
                and self.counter.count(self._table_piece(caption_line, header, data, start, end + 1)) <= self.max_tokens
            ):
                end += 1
            self.emit(
                "table", self._table_piece(caption_line, header, data, start, end), element.page_start, element.page_end
            )
            start = end

    @staticmethod
    def _table_piece(caption_line: str, header: list[str], data: list[list[str]], start: int, end: int) -> str:
        """Markdown for rows [start, end) with a "(rows a–b of N)" note on the caption line."""
        note = f"(rows {start + 1}–{end} of {len(data)})"
        line = f"{caption_line} {note}" if caption_line else f"Table {note}"
        return _render_table(line, header, data[start:end])

    # ---- rule 6: pictures

    def add_picture(self, element: Element) -> None:
        """Emit a picture chunk from its caption and/or vision description; nothing if it has neither."""
        self.flush()
        if not element.description and not element.text:
            return
        first_line = f"Figure (page {element.page_start})" + (f": {element.text}" if element.text else "")
        content = first_line + (f"\n{element.description}" if element.description else "")
        self.emit("picture", content, element.page_start, element.page_end, image_path=element.image_path)

    # ---- rule 8: embed_text = title + heading path + content

    def emit(
        self, chunk_type: str, content: str, page_start: int, page_end: int, image_path: str | None = None
    ) -> None:
        """Append a chunk; its embed_text and token_count are filled in by finish()."""
        self.chunks.append(
            Chunk(
                chunk_index=len(self.chunks),
                chunk_type=chunk_type,
                heading_path=self.headings.path,
                page_start=page_start,
                page_end=page_end,
                content=content,
                embed_text="",
                token_count=0,
                image_path=image_path,
            )
        )

    def finish(self) -> list[Chunk]:
        """Flush the last buffer and fill in every chunk's embed_text and token_count."""
        self.flush()
        for chunk in self.chunks:
            context = "\n".join(part for part in (self.title, chunk.heading_path) if part)
            chunk.embed_text = f"{context}\n\n{chunk.content}" if context else chunk.content
            chunk.token_count = self.counter.count(chunk.embed_text)
        return self.chunks


def chunk_document(doc: ParsedDocument, counter: TokenCounter, max_tokens: int = 350) -> list[Chunk]:
    """Split a parsed document into chunks, walking its elements once in reading order (rules 1-8).

    Every chunk's `content` is at most `max_tokens` tokens (one table row too long on its own is
    the only exception: rows are never cut).
    """
    chunker = _Chunker(doc.title, counter, max_tokens)
    for element in doc.elements:
        if element.kind == "heading":
            chunker.flush()  # never pack text across a heading (rule 1)
            chunker.headings.push(element.level, element.text)
        elif element.kind == "text":
            chunker.add_text(element.text, element.page_start, element.page_end)
        elif element.kind == "table":
            chunker.add_table(element)
        elif element.kind == "picture":
            chunker.add_picture(element)
    return chunker.finish()


def _clean_cell(value: str) -> str:
    """Make a cell safe for a one-line markdown table: no newlines, escaped pipes, empty for None/nan."""
    text = " ".join(str(value).split())
    return "" if text in ("None", "nan", "NaN") else text.replace("|", "\\|")


def _render_table(caption_line: str, header: list[str], rows: list[list[str]]) -> str:
    """Markdown table: optional caption line, header row, separator, one line per row."""
    lines = [caption_line] if caption_line else []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
