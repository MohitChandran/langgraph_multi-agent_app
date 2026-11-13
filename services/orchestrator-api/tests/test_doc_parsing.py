"""Slow end-to-end test of parsing + chunking with REAL Docling on the sample PDF - plan 02, stages R2/R4.

Skipped by default (takes ~1 minute). Run with:
    uv run scripts/make_sample_docs.py          # once, creates data/samples/q3-compliance-report.pdf
    cd services/orchestrator-api && uv run pytest -m slow -v
No vision calls: pictures are chunked by caption only, so the test needs no Groq key or quota.
"""

from pathlib import Path

import pytest

from app.agents.doc.chunking import TokenCounter, chunk_document
from app.agents.doc.models import ParsedDocument
from app.agents.doc.parsing import build_converter, parse_document
from app.agents.doc.vision import select_pictures
from app.config import REPO_ROOT, get_settings

SAMPLE = REPO_ROOT / "data" / "samples" / "q3-compliance-report.pdf"
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def parsed(tmp_path_factory: pytest.TempPathFactory) -> ParsedDocument:
    """Parse the sample PDF once for all tests in this module."""
    if not SAMPLE.exists():
        pytest.skip("sample PDF missing: run `uv run scripts/make_sample_docs.py` from the repo root")
    return parse_document(SAMPLE, tmp_path_factory.mktemp("pictures"), build_converter())


def test_title_pages_and_nested_headings(parsed: ParsedDocument) -> None:
    """The big title is the title (not a heading), and 4.2 is a level-2 heading under 4."""
    assert parsed.title == "Q3 2026 Vendor Compliance Report"
    assert parsed.page_count == 5
    levels = {e.text: e.level for e in parsed.elements if e.kind == "heading"}
    assert levels["4 Cloud Vendors"] == 1 and levels["4.2 SLA Terms"] == 2 and levels["3.1 Approval Process"] == 2


def test_tables_have_real_rows_and_columns(parsed: ParsedDocument) -> None:
    """SLA table: 5 vendors x 4 columns with its caption. Register: 34 rows, last row's cells not merged."""
    sla, register = [e for e in parsed.elements if e.kind == "table"]
    assert sla.page_start == 2 and sla.text == "Table 1: Cloud vendor SLA commitments"
    assert sla.table_rows[0] == ["Vendor", "Uptime SLA", "Credit if breached", "Critical response time"]
    assert ["Amazon Web Services", "99.95%", "10% of monthly fee", "15 minutes"] in sla.table_rows
    assert len(register.table_rows) == 35 and register.text.startswith("Table 2:")
    assert register.table_rows[-1][2] == "Regulatory compliance advisory"


def test_pictures_are_labelled_and_the_logo_is_skipped(parsed: ParsedDocument) -> None:
    """2 bar charts + 1 flow chart are kept for description; the logo is skipped by its label."""
    kept, skipped = select_pictures(parsed.elements, max_n=20, min_area=0.02)
    assert sorted(p.picture_label for p in kept) == ["bar_chart", "bar_chart", "flow_chart"]
    assert [reason for _, reason in skipped] == ["label 'logo'"]
    assert all(Path(p.image_path).exists() for p in kept)


def test_scanned_page_is_read_by_ocr(parsed: ParsedDocument) -> None:
    """Page 4 is an image: its text is only there because OCR read it."""
    page4 = " ".join(e.text for e in parsed.elements if e.page_start == 4)
    assert "Deloitte invoices were approved" in page4 and "30 September 2026" in page4


def test_chunks_follow_the_plan_shape(parsed: ParsedDocument) -> None:
    """With the real tokenizer: content <= 350 tokens, the SLA table is its own chunk with its section's
    path, and the big register table is split with the header repeated in every piece."""
    settings = get_settings()
    counter = TokenCounter(settings.rag_embedding_model)
    chunks = chunk_document(parsed, counter, settings.rag_chunk_tokens)
    assert all(counter.count(c.content) <= settings.rag_chunk_tokens for c in chunks)

    sla = next(c for c in chunks if c.chunk_type == "table" and "SLA commitments" in c.content)
    assert sla.heading_path == "4 Cloud Vendors > 4.2 SLA Terms"
    register = [c for c in chunks if c.chunk_type == "table" and "Vendor services" in c.content]
    assert len(register) >= 2
    assert all("| Vendor | Category | Service | Contract owner | Next review |" in c.content for c in register)
