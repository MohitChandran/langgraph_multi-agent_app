"""Unit tests for structure-aware chunking (app/agents/doc/chunking.py) - plan 02, stage R4.

Documents are built by hand from Element objects, so no Docling is needed. A fake counter
that counts WORDS makes the budget arithmetic easy to check by eye; one test at the end uses
the embedding model's real tokenizer.
"""

import pytest

from app.agents.doc.chunking import TokenCounter, chunk_document, heading_paths
from app.agents.doc.models import Element, ParsedDocument
from app.config import get_settings


class WordCounter:
    """Stands in for TokenCounter: 1 token = 1 whitespace-separated word."""

    def count(self, text: str) -> int:
        """Number of words in text."""
        return len(text.split())


def heading(level: int, text: str, page: int = 1) -> Element:
    """A heading element."""
    return Element("heading", text, page, page, level=level)


def text(words: int, word: str = "word", page: int = 1, end: str = "") -> Element:
    """A paragraph of `words` words (optionally ending with punctuation)."""
    return Element("text", " ".join([word] * words) + end, page, page)


def chunks_of(*elements: Element, max_tokens: int = 10, title: str = "Report"):
    """Chunk a hand-built document with the word counter."""
    return chunk_document(ParsedDocument(title, 9, list(elements)), WordCounter(), max_tokens=max_tokens)


# ------------------------------------------------------------------ headings


def test_heading_paths_nest_and_a_new_top_level_heading_pops_both() -> None:
    """L2 nests under L1; a new L1 closes both; the document title is never part of the path."""
    elements = [heading(1, "4 Cloud Vendors"), heading(2, "4.2 SLA Terms"), text(3), heading(1, "5 Incidents"), text(3)]
    chunks = chunks_of(*elements)
    assert [c.heading_path for c in chunks] == ["4 Cloud Vendors > 4.2 SLA Terms", "5 Incidents"]
    assert heading_paths(elements)[2] == "4 Cloud Vendors > 4.2 SLA Terms"


def test_same_level_heading_replaces_its_sibling() -> None:
    """4.2 replaces 4.1 but keeps their parent 4."""
    chunks = chunks_of(heading(1, "4"), heading(2, "4.1"), text(2), heading(2, "4.2"), text(2))
    assert [c.heading_path for c in chunks] == ["4 > 4.1", "4 > 4.2"]


# ------------------------------------------------------------------ packing text


def test_paragraphs_are_packed_up_to_the_budget_but_never_across_a_heading() -> None:
    """4+4 words pack into one chunk (<= 10); the third paragraph starts a new one; a heading always cuts."""
    chunks = chunks_of(text(4, "a"), text(4, "b"), text(4, "c"), heading(1, "Next"), text(1, "d"))
    assert [c.content.split()[0] for c in chunks] == ["a", "c", "d"]
    assert chunks[0].content.count("\n\n") == 1  # two paragraphs, joined by a blank line
    assert all(WordCounter().count(c.content) <= 10 for c in chunks)


def test_a_long_paragraph_is_split_at_sentence_ends() -> None:
    """3 sentences of 6 words with a budget of 10: each chunk holds whole sentences only."""
    paragraph = Element("text", " ".join(["One two three four five six."] * 3), 1, 1)
    chunks = chunks_of(paragraph)
    assert len(chunks) == 3
    assert all(c.content.endswith(".") and WordCounter().count(c.content) <= 10 for c in chunks)


def test_a_giant_sentence_is_split_by_words() -> None:
    """25 words with no sentence end, budget 10 -> pieces of at most 10 words, no word lost."""
    chunks = chunks_of(text(25))
    assert [WordCounter().count(c.content) for c in chunks] == [10, 10, 5]


# ------------------------------------------------------------------ tables


def table(rows: list[list[str]], caption: str = "") -> Element:
    """A table element (first row = header)."""
    return Element("table", caption, 2, 2, table_rows=rows)


def test_table_is_its_own_chunk_with_caption_and_markdown() -> None:
    """A table never shares a chunk with text; cells are cleaned and pipes escaped."""
    rows = [["Vendor", "SLA"], ["AWS", "99.95%"], ["Odd | name", "None"]]
    chunks = chunks_of(text(2), table(rows, "Cloud SLAs"), text(2), max_tokens=50)
    assert [c.chunk_type for c in chunks] == ["text", "table", "text"]
    assert chunks[1].content == (
        "Table: Cloud SLAs\n| Vendor | SLA |\n|---|---|\n| AWS | 99.95% |\n| Odd \\| name |  |"
    )


def test_caption_that_already_says_table_is_not_prefixed_twice() -> None:
    """'Table 1: ...' stays as is (no 'Table: Table 1: ...')."""
    chunk = chunks_of(table([["a"], ["b"]], "Table 1: Things"), max_tokens=50)[0]
    assert chunk.content.startswith("Table 1: Things\n")


def test_table_without_a_header_row_uses_its_first_row() -> None:
    """Docling/pandas name header-less columns 0, 1, 2...: the first data row becomes the header."""
    chunk = chunks_of(table([["0", "1"], ["Vendor", "SLA"], ["AWS", "99.95%"]]), max_tokens=50)[0]
    assert chunk.content.splitlines()[0] == "| Vendor | SLA |"


def test_big_table_is_split_by_rows_with_the_header_in_every_piece() -> None:
    """Over budget -> pieces of whole rows; each repeats the header and says which rows it holds."""
    rows = [["Vendor", "Service"]] + [[f"V{i}", f"S{i}"] for i in range(1, 13)]
    chunks = chunks_of(table(rows, "Register"), max_tokens=30)
    assert len(chunks) > 1
    for chunk in chunks:
        lines = chunk.content.splitlines()
        assert lines[1] == "| Vendor | Service |" and lines[2] == "|---|---|"
        assert WordCounter().count(chunk.content) <= 30
    assert chunks[0].content.startswith("Table: Register (rows 1–")
    assert chunks[-1].content.splitlines()[0].endswith("of 12)")
    all_rows = [line for c in chunks for line in c.content.splitlines()[3:]]
    assert all_rows == [f"| V{i} | S{i} |" for i in range(1, 13)]  # every row exactly once, none cut


# ------------------------------------------------------------------ pictures and embed_text


def test_picture_chunk_combines_caption_and_description() -> None:
    """caption + description; caption only when not described; nothing when it has neither."""
    described = Element("picture", "Figure 2: Incidents", 6, 6, image_path="p.png", description="Bar chart. Q3: 14")
    caption_only = Element("picture", "Figure 3: Logo", 7, 7)
    nothing = Element("picture", "", 8, 8)
    chunks = chunks_of(described, caption_only, nothing, max_tokens=50)
    assert [c.content for c in chunks] == [
        "Figure (page 6): Figure 2: Incidents\nBar chart. Q3: 14",
        "Figure (page 7): Figure 3: Logo",
    ]
    assert chunks[0].chunk_type == "picture" and chunks[0].image_path == "p.png"


def test_embed_text_puts_title_and_heading_path_before_the_content() -> None:
    """The 'contextual chunk header': title, then path, a blank line, then the content."""
    chunk = chunks_of(heading(1, "4 Cloud"), heading(2, "4.2 SLA"), text(2, "credit"), title="Q3 Report")[0]
    assert chunk.embed_text == "Q3 Report\n4 Cloud > 4.2 SLA\n\ncredit credit"
    assert chunk.token_count == WordCounter().count(chunk.embed_text)


def test_text_before_any_heading_has_an_empty_path_line_left_out() -> None:
    """No heading yet -> embed_text is just title + content."""
    chunk = chunks_of(text(2, "intro"), title="Q3 Report")[0]
    assert chunk.heading_path == "" and chunk.embed_text == "Q3 Report\n\nintro intro"


def test_chunks_are_numbered_in_document_order() -> None:
    """chunk_index runs 0..n-1."""
    chunks = chunks_of(text(8), text(8), table([["a"], ["b"]]), text(8), max_tokens=10)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ------------------------------------------------------------------ real tokenizer


@pytest.fixture(scope="module")
def real_counter() -> TokenCounter:
    """The embedding model's real tokenizer (downloaded once from Hugging Face, then cached)."""
    try:
        return TokenCounter(get_settings().rag_embedding_model)
    except Exception as error:  # offline, or no .env
        pytest.skip(f"tokenizer not available: {type(error).__name__}")


def test_every_chunk_fits_the_budget_with_the_real_tokenizer(real_counter: TokenCounter) -> None:
    """With the real tokenizer and the real 350 budget, no chunk's content exceeds 350 tokens."""
    long_section = [text(120, "Uptime credits are reviewed quarterly.", end=".") for _ in range(4)]
    big_table = table(
        [["Vendor", "Service", "Owner"]] + [[f"Vendor {i}", "Compute hosting", "A. Rao"] for i in range(80)]
    )
    doc = ParsedDocument("Q3 Report", 3, [heading(1, "4 Cloud Vendors"), *long_section, big_table])
    chunks = chunk_document(doc, real_counter, max_tokens=350)
    assert len(chunks) > 3
    assert all(real_counter.count(c.content) <= 350 for c in chunks)
