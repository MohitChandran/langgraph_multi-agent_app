"""Tests for hybrid retrieval (app/agents/doc/retrieval.py + store search) - plan 02, stage R6.

RRF and source selection are pure functions (hand-computed numbers). The search tests store
chunks with hash-based fake vectors in two chats and check what each search returns.
"""

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.agents.doc.models import Chunk, Hit
from app.agents.doc.retrieval import RRF_K, rrf_fuse, select_sources
from app.agents.doc.store import DocumentStore
from app.documents.files import MARKDOWN, save_original
from tests.test_doc_ingest import fake_vector

# ------------------------------------------------------------------ RRF (pure)


def test_rrf_scores_are_the_sum_of_reciprocal_ranks() -> None:
    """Hand-computed with k = 60: an id at rank 1 in one list and rank 3 in the other gets 1/61 + 1/63."""
    fused = dict(rrf_fuse([[10, 20, 30], [40, 50, 10]]))
    assert fused[10] == pytest.approx(1 / 61 + 1 / 63)
    assert fused[20] == pytest.approx(1 / 62)
    assert fused[40] == pytest.approx(1 / 61)
    assert RRF_K == 60


def test_a_chunk_found_by_both_searches_beats_single_list_winners() -> None:
    """Rank 3 in both lists (2/63) outranks rank 1 in only one list (1/61)."""
    order = [chunk_id for chunk_id, _ in rrf_fuse([[1, 2, 99], [3, 4, 99]])]
    assert order[0] == 99


def test_ties_are_broken_by_the_dense_list_and_empty_lists_are_fine() -> None:
    """Equal scores (both rank 1 of a different list): the dense list's id wins. No input -> no output."""
    assert [chunk_id for chunk_id, _ in rrf_fuse([[7], [8]])] == [7, 8]
    assert rrf_fuse([[], []]) == []
    assert [chunk_id for chunk_id, _ in rrf_fuse([[5, 6], []])] == [5, 6]


# ------------------------------------------------------------------ source selection (pure)


def hit(chunk_id: int, tokens: int) -> Hit:
    """A minimal Hit with a given size."""
    return Hit(chunk_id, uuid4(), "f.pdf", "text", "", 1, 1, "x", tokens, None)


def test_sources_respect_top_k_and_the_token_budget() -> None:
    """At most top_k sources; a hit that doesn't fit the remaining budget is skipped, a smaller later one is taken."""
    hits = [hit(1, 1000), hit(2, 1500), hit(3, 800), hit(4, 300), hit(5, 100)]
    assert [h.chunk_id for h in select_sources(hits, top_k=6, token_budget=3000)] == [1, 2, 4, 5]
    assert [h.chunk_id for h in select_sources(hits, top_k=2, token_budget=3000)] == [1, 2]


# ------------------------------------------------------------------ search (integration)


async def store_chunks(store: DocumentStore, session_id: UUID, data_dir: Path, texts: list[str]) -> list[int]:
    """A ready document in this chat whose chunks are `texts` (fake vectors); returns the chunk ids in order."""
    document_id = uuid4()
    save_original(data_dir, document_id, MARKDOWN, b"# x")
    await store.create_document(document_id, session_id, "doc.md", MARKDOWN.mime_type, 3, uuid4().hex)
    chunks = [Chunk(i, "text", "", 1, 1, text, text, len(text.split())) for i, text in enumerate(texts)]
    await store.save_chunks_and_mark_ready(
        document_id, session_id, chunks, [fake_vector(text) for text in texts],
        page_count=1, table_count=0, picture_count=0, pictures_skipped=0, embedding_model="fake",
    )  # fmt: skip
    rows = await store._fetch_all(
        "SELECT id FROM app.document_chunks WHERE document_id = %s ORDER BY chunk_index", (document_id,)
    )
    return [row["id"] for row in rows]


TEXTS = [
    "Service credits are 10% of the monthly fee when the uptime SLA is breached.",
    "Invoices above USD 25,000 need a Finance Director's approval.",
    "Security incidents rose to 14 in the third quarter.",
]


@pytest.mark.integration
async def test_keyword_search_matches_a_full_question_with_filler_words(
    document_store: DocumentStore, test_session: UUID, tmp_path: Path
) -> None:
    """An AND query would need every word ('what', 'do', 'we', 'get'...) in one chunk; the OR query finds it."""
    ids = await store_chunks(document_store, test_session, tmp_path, TEXTS)
    found = await document_store.keyword_search(test_session, "What credit do we get if the SLA is breached?", 20)
    assert found[0] == ids[0]
    assert await document_store.keyword_search(test_session, "the a of", 20) == []  # only stop words -> nothing


@pytest.mark.integration
async def test_dense_search_returns_the_nearest_chunk_first(
    document_store: DocumentStore, test_session: UUID, tmp_path: Path
) -> None:
    """The query with the identical text has distance 0, so its chunk comes first."""
    ids = await store_chunks(document_store, test_session, tmp_path, TEXTS)
    found = await document_store.dense_search(test_session, fake_vector(TEXTS[2]), 20)
    assert found[0] == ids[2] and sorted(found) == sorted(ids)


@pytest.mark.integration
async def test_searches_never_return_another_chats_chunks(
    document_store: DocumentStore,
    chat_store,
    test_session: UUID,
    tmp_path: Path,  # noqa: ANN001
) -> None:
    """Chat B's searches only ever see chat B's chunks, even for chat A's exact text."""
    chat_b = (await chat_store.create_session("[pytest] chat B")).id
    try:
        ids_a = await store_chunks(document_store, test_session, tmp_path, TEXTS)
        ids_b = await store_chunks(document_store, chat_b, tmp_path, ["Lenovo laptops arrive in March."])
        assert await document_store.dense_search(chat_b, fake_vector(TEXTS[0]), 20) == ids_b
        assert await document_store.keyword_search(chat_b, "credits uptime breached", 20) == []
        assert set(await document_store.dense_search(test_session, fake_vector("x"), 20)) == set(ids_a)
    finally:
        await chat_store.delete_session(chat_b)
