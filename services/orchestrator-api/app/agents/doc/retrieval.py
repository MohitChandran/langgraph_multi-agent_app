"""Hybrid retrieval: vector search + keyword search -> Reciprocal Rank Fusion -> cross-encoder rerank (plan 02, 6.2).

Why two searches (decisions D6, D7):
  - dense (vectors) finds MEANING: "compensation for downtime" ~ "service credit if breached";
  - keyword (full-text) finds EXACT tokens embeddings blur: "99.95%", "INV-2026-0273", "Q3".
Each misses what the other catches, so both run and their rankings are fused.

Reciprocal Rank Fusion: score(chunk) = sum over lists of 1 / (k + rank), k = 60.
It uses only RANKS, so it doesn't matter that a cosine distance and a ts_rank_cd score are on
completely different scales. A chunk found by both searches collects two shares and rises.

Rerank (decision D8): a cross-encoder reads the query and each candidate TOGETHER, which is
far more precise than comparing two separate vectors, but slower, so it only re-orders the
~20 fused candidates. The best RAG_TOP_K that fit RAG_CONTEXT_TOKENS become the answer's sources.

CLI (prints every stage side by side):
    uv run python -m app.agents.doc.retrieval [--session <chat id>] "question"
"""

import argparse
import asyncio
from collections.abc import Sequence
from uuid import UUID

from app.agents.doc.embeddings import Embedder, Reranker
from app.agents.doc.models import Hit
from app.agents.doc.store import DocumentStore

RRF_K = 60  # the constant from the original RRF paper; dampens the gap between rank 1 and rank 2


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Fuse ranked id lists into one: [(id, score)], best first.

    score(id) = sum of 1 / (k + rank) over the lists that contain it (ranks are 1-based).
    Ties are broken by the position in the first list (dense), then the second (keyword).
    """
    scores: dict[int, float] = {}
    positions: list[dict[int, int]] = []
    for ranking in rankings:
        position = {chunk_id: rank for rank, chunk_id in enumerate(ranking, start=1)}
        positions.append(position)
        for chunk_id, rank in position.items():
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    def order(item: tuple[int, float]) -> tuple:
        """Higher score first, then earlier in the dense list, then in the keyword list."""
        chunk_id, score = item
        return (-score, *(pos.get(chunk_id, float("inf")) for pos in positions))

    return sorted(scores.items(), key=order)


def select_sources(hits: Sequence[Hit], top_k: int, token_budget: int) -> list[Hit]:
    """The best hits, in order, up to top_k of them and within the token budget of the answer prompt.

    A hit that doesn't fit the remaining budget is skipped (a smaller one after it may still fit).
    """
    selected, used = [], 0
    for hit in hits:
        if len(selected) == top_k:
            break
        if used + hit.token_count > token_budget:
            continue
        selected.append(hit)
        used += hit.token_count
    return selected


class Retriever:
    """The retrieval steps as separate methods, so the DocAgent's graph runs them as separate nodes
    (dense and keyword in parallel) and the eval can switch steps on and off."""

    def __init__(self, store: DocumentStore, embedder: Embedder, reranker: Reranker | None, candidates: int) -> None:
        """reranker=None keeps the fused order (RAG_RERANK=false)."""
        self.store, self.embedder, self.reranker, self.candidates = store, embedder, reranker, candidates

    async def dense(self, session_id: UUID, query: str) -> list[int]:
        """Top candidates by meaning. Embedding the query is CPU work, so it runs in a thread."""
        vector = await asyncio.to_thread(self.embedder.embed_query, query)
        return await self.store.dense_search(session_id, vector, self.candidates)

    async def keyword(self, session_id: UUID, query: str) -> list[int]:
        """Top candidates by exact words (any word of the query counts)."""
        return await self.store.keyword_search(session_id, query, self.candidates)

    async def fuse(self, dense_ids: list[int], keyword_ids: list[int]) -> list[Hit]:
        """RRF-fuse both lists and load the chunks, each annotated with its ranks and fused score."""
        fused = rrf_fuse([dense_ids, keyword_ids])[: self.candidates]
        loaded = await self.store.load_hits([chunk_id for chunk_id, _ in fused])
        dense_rank = {chunk_id: rank for rank, chunk_id in enumerate(dense_ids, start=1)}
        keyword_rank = {chunk_id: rank for rank, chunk_id in enumerate(keyword_ids, start=1)}
        hits = []
        for chunk_id, score in fused:
            if chunk_id in loaded:  # a chunk deleted a moment ago is simply skipped
                hit = loaded[chunk_id]
                hit.dense_rank = dense_rank.get(chunk_id)
                hit.keyword_rank = keyword_rank.get(chunk_id)
                hit.rrf_score = score
                hits.append(hit)
        return hits

    async def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Score every (query, chunk content) pair with the cross-encoder and sort by it. CPU work: in a thread."""
        if self.reranker is None or not hits:
            return hits
        scores = await asyncio.to_thread(self.reranker.score, query, [hit.content for hit in hits])
        for hit, score in zip(hits, scores, strict=True):
            hit.rerank_score = score
        return sorted(hits, key=lambda hit: hit.rerank_score, reverse=True)


# ---------------------------------------------------------------- CLI


async def _main(question: str, session_id: UUID | None) -> None:
    """Run every retrieval stage for one question and print the lists side by side."""
    from app.config import get_settings  # CLI-only imports

    settings = get_settings()
    async with DocumentStore(settings.app_database_url) as store:
        if session_id is None:  # default: the chat of the most recently finished document
            row = await store.fetch_one(
                "SELECT session_id FROM app.documents WHERE status = 'ready' ORDER BY finished_at DESC LIMIT 1"
            )
            if row is None:
                raise SystemExit("No ready document found. Upload one first.")
            session_id = row["session_id"]
        models = settings.data_dir / "models"
        embedder = await asyncio.to_thread(Embedder, settings.rag_embedding_model, models)
        reranker = await asyncio.to_thread(Reranker, settings.rag_rerank_model, models)
        retriever = Retriever(store, embedder, reranker, settings.rag_candidates)

        dense_ids, keyword_ids = await asyncio.gather(
            retriever.dense(session_id, question), retriever.keyword(session_id, question)
        )
        fused = await retriever.fuse(dense_ids, keyword_ids)
        reranked = await retriever.rerank(question, list(fused))
        sources = select_sources(reranked, settings.rag_top_k, settings.rag_context_tokens)

    def label(hit: Hit) -> str:
        """Short description of a chunk: type, page, first words."""
        first_line = hit.content.splitlines()[0][:44]
        return f"#{hit.chunk_id} {hit.chunk_type[:3]} p{hit.page_start} {first_line}"

    print(f"Chat {session_id}\nQuestion: {question}\n")
    print(f"{'':>3}  {'DENSE (by meaning)':<14}{'KEYWORD (by words)':<16}  {'FUSED (RRF)':<58}RERANKED (cross-encoder)")
    by_id = {hit.chunk_id: hit for hit in fused}
    for rank in range(min(10, max(len(fused), 1))):
        dense = f"#{dense_ids[rank]}" if rank < len(dense_ids) else ""
        keyword = f"#{keyword_ids[rank]}" if rank < len(keyword_ids) else ""
        fused_cell = f"{label(fused[rank])} ({fused[rank].rrf_score:.4f})" if rank < len(fused) else ""
        rerank_cell = f"{label(reranked[rank])} ({reranked[rank].rerank_score:+.2f})" if rank < len(reranked) else ""
        print(f"{rank + 1:>3}. {dense:<14}{keyword:<16}  {fused_cell:<58.58}{rerank_cell}")
    print(f"\nSources for the answer ({len(sources)}): " + ", ".join(f"#{h.chunk_id}" for h in sources))
    print(f"Found by both searches: {sorted(set(dense_ids) & set(keyword_ids) & set(by_id))}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Show every retrieval stage for one question.")
    parser.add_argument("question")
    parser.add_argument("--session", type=UUID, help="chat id (default: chat of the newest ready document)")
    args = parser.parse_args()
    asyncio.run(_main(args.question, args.session))


if __name__ == "__main__":
    main()
