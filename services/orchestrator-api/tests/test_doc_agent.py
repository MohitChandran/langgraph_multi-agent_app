"""Tests for the DocAgent workflow (app/agents/doc/graph.py) with fakes - plan 02, stage R7.

A fake retriever returns scripted chunks and records which queries it was asked; fake LLMs
return scripted replies. So these tests check OUR logic (rewrite only with history, parallel
searches, not_found without an LLM call, citation clean-up), not the models' quality.
"""

from uuid import UUID, uuid4

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.agents.doc.graph import DocAgent, clean_citations
from app.agents.doc.models import DocAnswer, Hit
from app.agents.doc.prompts import NOT_FOUND_TEXT
from app.chat.memory import ConversationTurn

SESSION = uuid4()


def make_hit(chunk_id: int, content: str, tokens: int = 100, chunk_type: str = "text") -> Hit:
    """A retrieved chunk."""
    path = "4 Cloud Vendors > 4.2 SLA Terms"
    return Hit(chunk_id, uuid4(), "q3-report.pdf", chunk_type, path, 2, 2, content, tokens, None)


class FakeRetriever:
    """Scripted retrieval: records the queries each search got, returns the given hits."""

    def __init__(self, hits: list[Hit]) -> None:
        """hits = what fusion and reranking will return."""
        self.hits = hits
        self.queries: dict[str, str] = {}

    async def dense(self, session_id: UUID, query: str) -> list[int]:
        """Record the query; return the hit ids."""
        self.queries["dense"] = query
        return [hit.chunk_id for hit in self.hits]

    async def keyword(self, session_id: UUID, query: str) -> list[int]:
        """Record the query; return the hit ids."""
        self.queries["keyword"] = query
        return [hit.chunk_id for hit in self.hits]

    async def fuse(self, dense_ids: list[int], keyword_ids: list[int]) -> list[Hit]:
        """Return the hits as if fused."""
        return list(self.hits)

    async def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Record the query and give descending scores."""
        self.queries["rerank"] = query
        for score, hit in zip(range(len(hits), 0, -1), hits, strict=False):
            hit.rerank_score = float(score)
        return hits


class FakeStore:
    """Only what DocAgent.ask needs before searching: the documents' (status, stage)."""

    def __init__(self, statuses: list[tuple[str, str | None]]) -> None:
        """statuses of the chat's documents."""
        self.statuses = statuses

    async def session_document_status(self, session_id: UUID) -> list[tuple[str, str | None]]:
        """The scripted statuses."""
        return self.statuses


def make_agent(hits: list[Hit], reply: DocAnswer | None = None, statuses=None, top_k: int = 6):  # noqa: ANN001
    """A DocAgent wired to fakes; returns (agent, retriever, rewriter calls, answerer calls)."""
    rewrites, answers = [], []

    def rewriter(messages):  # noqa: ANN001
        """Fake fast LLM: rewrites any follow-up to a fixed standalone query."""
        rewrites.append(messages)
        return AIMessage(content="service credit if Microsoft Azure breaches its uptime SLA")

    def answerer(messages):  # noqa: ANN001
        """Fake strong LLM: the scripted structured answer."""
        answers.append(messages)
        return reply or DocAnswer(answer_found=True, answer="The credit is 10% of the monthly fee [1].", citations=[1])

    retriever = FakeRetriever(hits)
    agent = DocAgent(
        FakeStore(statuses if statuses is not None else [("ready", None)]), retriever, top_k=top_k,
        rewriter=RunnableLambda(rewriter), answerer=RunnableLambda(answerer),
    )  # fmt: skip
    return agent, retriever, rewrites, answers


async def test_without_history_the_question_is_searched_as_is_and_answered_with_citations() -> None:
    """No history -> no rewrite call; both searches and the reranker get the question; the answer is grounded."""
    agent, retriever, rewrites, answers = make_agent([make_hit(1, "AWS | 99.95% | 10% of monthly fee")])
    result = await agent.ask("What credit applies if AWS breaches its SLA?", session_id=SESSION)

    assert rewrites == []
    assert set(retriever.queries.values()) == {"What credit applies if AWS breaches its SLA?"}
    assert result.grounded and result.success and result.citations[0].n == 1
    assert result.citations[0].filename == "q3-report.pdf" and result.citations[0].snippet.startswith("AWS")
    assert "[1] (q3-report.pdf, page 2, section: 4 Cloud Vendors > 4.2 SLA Terms, text)" in answers[0][-1].content
    assert set(result.steps) == {"rewrite_query", "dense_search", "keyword_search", "fuse", "rerank", "answer"}
    assert result.retrieval[0].used and result.retrieval[0].rerank_score == 1.0


async def test_a_follow_up_is_rewritten_and_both_searches_use_the_rewrite() -> None:
    """With history, the standalone query (not 'and for Azure?') goes to both searches and the reranker."""
    agent, retriever, rewrites, _ = make_agent([make_hit(1, "Microsoft Azure | 99.9% | 10% of monthly fee")])
    history = [ConversationTurn("What credit applies if AWS breaches its SLA?", "10% of the monthly fee.", None)]

    result = await agent.ask("and for Azure?", history=history, session_id=SESSION)

    assert len(rewrites) == 1 and "Q1: What credit applies if AWS breaches its SLA?" in rewrites[0][-1].content
    standalone = "service credit if Microsoft Azure breaches its uptime SLA"
    assert retriever.queries == {"dense": standalone, "keyword": standalone, "rerank": standalone}
    assert result.standalone_query == standalone


async def test_no_candidates_means_not_found_without_any_llm_call() -> None:
    """Nothing retrieved -> the fixed not-found text; the answer LLM is never called."""
    agent, _, _, answers = make_agent([])
    result = await agent.ask("What is our parental leave policy?", session_id=SESSION)
    assert result.answer == NOT_FOUND_TEXT and not result.grounded and answers == []
    assert "not_found" in result.steps and "answer" not in result.steps


async def test_sources_are_limited_by_top_k() -> None:
    """Only the best top_k candidates become numbered sources; the rest appear in the trace as unused."""
    hits = [make_hit(i, f"chunk {i}") for i in range(1, 9)]
    agent, _, _, answers = make_agent(hits, top_k=3)
    result = await agent.ask("question", session_id=SESSION)
    prompt = answers[0][-1].content
    assert "[3] (" in prompt and "[4] (" not in prompt
    assert [t.used for t in result.retrieval] == [True] * 3 + [False] * 5


async def test_documents_not_ready_yet_get_a_status_reply_instead_of_a_search() -> None:
    """A chat whose only document is still parsing: no search, a clear 'still being processed' answer."""
    agent, retriever, _, answers = make_agent([make_hit(1, "x")], statuses=[("processing", "describing_pictures")])
    result = await agent.ask("question", session_id=SESSION)
    assert "still being processed (stage: describing pictures)" in result.answer
    assert retriever.queries == {} and answers == [] and not result.success


async def test_an_answer_citing_nothing_valid_is_not_grounded() -> None:
    """The model cites [7] but there is 1 source: the citation is dropped and the answer is not grounded."""
    reply = DocAnswer(answer_found=True, answer="It is 10% [7].", citations=[7])
    agent, _, _, _ = make_agent([make_hit(1, "x")], reply=reply)
    result = await agent.ask("question", session_id=SESSION)
    assert result.answer == "It is 10% ." and result.citations == [] and not result.grounded


def test_clean_citations_merges_list_and_text_and_drops_invalid_numbers() -> None:
    """Numbers come from the text and the list, in first-seen order, unique; out-of-range marks are removed."""
    text, numbers = clean_citations("A [2]. B [9]. C [1] [2].", [3, 1], source_count=3)
    assert numbers == [2, 1, 3]
    assert text == "A [2]. B . C [1] [2]."
