"""The Doc QA agent: a fixed LangGraph workflow over the chat's uploaded documents (plan 02, section 6).

    START -> rewrite_query -+-> dense_search ---+-> fuse -> rerank -+-> answer ----> END
                            +-> keyword_search -+                   +-> not_found -> END

  - rewrite_query: a follow-up ("and for Azure?") becomes a standalone search query (only with history).
  - dense_search and keyword_search are two edges out of one node, so LangGraph runs them IN
    PARALLEL; each writes its own state key, and `fuse` waits for both (a join edge).
  - rerank orders the fused candidates with the cross-encoder and picks the numbered sources.
  - answer: the strong LLM writes a cited answer (structured output DocAnswer).
  - not_found: no candidates (or all below RAG_MIN_RERANK_SCORE) -> a fixed reply, no LLM call.

Same design as DBAgent: a fixed graph (the LLM can't skip retrieval or grounding), nodes as
methods, every dependency injectable so tests use fakes.
"""

import re
from typing import Literal
from uuid import UUID

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from app.agents.citations import CITATION_MARK, clean_citations  # noqa: F401  (re-exported for tests)
from app.agents.doc.models import Citation, DocAgentResult, DocAgentState, DocAnswer, Hit, RetrievalTrace
from app.agents.doc.prompts import ANSWER_PROMPT, NOT_FOUND_TEXT, REWRITE_PROMPT
from app.agents.doc.retrieval import Retriever, select_sources
from app.agents.doc.store import DocumentStore
from app.chat.memory import ConversationTurn, format_history
from app.llm import LLM_ERRORS, get_llm

REWRITE_MAX_TOKENS = 100
SNIPPET_CHARS = 200
TABLE_SEPARATOR = re.compile(r"\|(?:-+\|)+")  # the "|---|---|" row of a markdown table
CHUNK_KINDS = {"text": "passage", "table": "table", "picture": "figure"}  # chunk_type -> words a reader expects


class DocAgent:
    """Answers questions about a chat's documents with numbered, checkable citations.

    Usage:
        agent = DocAgent(store, retriever)
        result = await agent.ask("What credit applies if AWS breaches its SLA?", session_id=chat_id)
    """

    name = "doc"  # saved in chat_messages.payload.agent (the UI renders doc answers differently)

    def __init__(
        self,
        store: DocumentStore,
        retriever: Retriever,
        top_k: int = 6,
        context_tokens: int = 3000,
        min_rerank_score: float | None = None,
        rewriter: Runnable[list[BaseMessage], BaseMessage] | None = None,
        answerer: Runnable[list[BaseMessage], DocAnswer] | None = None,
    ) -> None:
        """Keep the dependencies and compile the graph once.

        rewriter: fast LLM (default), only called when there is history.
        answerer: strong LLM with structured output DocAnswer (default; json_schema = Groq constrains the reply).
        """
        self._store, self._retriever = store, retriever
        self._top_k, self._context_tokens, self._min_rerank_score = top_k, context_tokens, min_rerank_score
        self._rewriter = rewriter or get_llm("fast", max_tokens=REWRITE_MAX_TOKENS)
        self._answerer = answerer or get_llm("strong").with_structured_output(DocAnswer, method="json_schema")
        self.graph = self._build_graph()

    @property
    def retriever(self) -> Retriever:
        """The retriever (dense, keyword, fuse, rerank): the R9 eval compares the search modes with it."""
        return self._retriever

    def _build_graph(self):
        """Nodes, the parallel fan-out to both searches, the join into fuse, and the final router."""
        builder = StateGraph(DocAgentState)
        builder.add_node("rewrite_query", self.rewrite_query)
        builder.add_node("dense_search", self.dense_search)
        builder.add_node("keyword_search", self.keyword_search)
        builder.add_node("fuse", self.fuse)
        builder.add_node("rerank", self.rerank)
        builder.add_node("answer", self.answer)
        builder.add_node("not_found", self.not_found)

        builder.add_edge(START, "rewrite_query")
        builder.add_edge("rewrite_query", "dense_search")  # two edges out of one node:
        builder.add_edge("rewrite_query", "keyword_search")  # both run in the same step, in parallel
        builder.add_edge(["dense_search", "keyword_search"], "fuse")  # join: fuse waits for BOTH
        builder.add_edge("fuse", "rerank")
        builder.add_conditional_edges("rerank", self.route_after_rerank)
        builder.add_edge("answer", END)
        builder.add_edge("not_found", END)
        return builder.compile(name="doc agent")  # the run name in traces (Langfuse)

    async def ask(
        self, question: str, history: list[ConversationTurn] | None = None, session_id: UUID | None = None
    ) -> DocAgentResult:
        """Answer one question about the documents of chat `session_id` (the same signature as DBAgent.ask).

        Before searching: if the chat's documents exist but none is ready yet, say so instead
        of searching an empty index. The graph runs with astream to record which nodes ran.
        """
        early = await self._documents_not_ready(question, session_id)
        if early is not None:
            return early

        steps: list[str] = []
        state: DocAgentState = {}
        history_text = format_history(history or [])
        initial: DocAgentState = {"question": question, "history": history_text, "session_id": session_id}
        async for mode, chunk in self.graph.astream(initial, stream_mode=["updates", "values"]):
            if mode == "updates":
                steps.extend(chunk)
            else:
                state = chunk
        return self._result(question, state, steps)

    async def _documents_not_ready(self, question: str, session_id: UUID | None) -> DocAgentResult | None:
        """A finished result when there is nothing searchable yet; None when at least one document is ready."""
        statuses = await self._store.session_document_status(session_id) if session_id else []
        if any(status == "ready" for status, _ in statuses):
            return None
        busy = [stage or status for status, stage in statuses if status in ("queued", "processing")]
        if busy:
            stage = busy[0].replace("_", " ")
            text = f"Your document is still being processed (stage: {stage}). Ask again in a moment."
        elif statuses:
            text = "None of the documents in this chat could be processed, so there is nothing to search yet."
        else:
            text = "There are no documents in this chat yet. Upload one to ask questions about it."
        return DocAgentResult(
            question=question,
            answer=text,
            success=False,
            grounded=False,
            standalone_query=question,
            reasoning=["Checked the chat's documents first: none is ready to search yet, so nothing was searched."],
        )

    # ---------------------------------------------------------------- nodes

    async def rewrite_query(self, state: DocAgentState) -> DocAgentState:
        """Node: make a follow-up searchable on its own. No history -> the question as is, no LLM call.

        If the rewrite fails, searching with the original question is still better than failing.
        """
        if not state.get("history"):
            return {"standalone_query": state["question"]}
        messages = REWRITE_PROMPT.format_messages(history=state["history"], question=state["question"])
        try:
            reply = await self._rewriter.ainvoke(messages)
            rewritten = str(reply.content).strip().splitlines()[0].strip(' "') if str(reply.content).strip() else ""
        except LLM_ERRORS:
            rewritten = ""
        return {"standalone_query": rewritten or state["question"]}

    async def dense_search(self, state: DocAgentState) -> DocAgentState:
        """Node: the chat's chunks nearest in meaning to the standalone query."""
        return {"dense_ids": await self._retriever.dense(state["session_id"], state["standalone_query"])}

    async def keyword_search(self, state: DocAgentState) -> DocAgentState:
        """Node: the chat's chunks containing the query's words."""
        return {"keyword_ids": await self._retriever.keyword(state["session_id"], state["standalone_query"])}

    async def fuse(self, state: DocAgentState) -> DocAgentState:
        """Node: Reciprocal Rank Fusion of both lists, chunks loaded with their ranks."""
        return {"candidates": await self._retriever.fuse(state["dense_ids"], state["keyword_ids"])}

    async def rerank(self, state: DocAgentState) -> DocAgentState:
        """Node: cross-encoder order, then the top sources that fit the answer's token budget.

        Reranking uses the STANDALONE query: for "and for Azure?" the original words alone say
        nothing about SLA credits.
        """
        candidates = await self._retriever.rerank(state["standalone_query"], state["candidates"])
        return {"candidates": candidates, "sources": select_sources(candidates, self._top_k, self._context_tokens)}

    async def answer(self, state: DocAgentState) -> DocAgentState:
        """Node: the strong LLM writes the cited answer from the numbered sources."""
        messages = ANSWER_PROMPT.format_messages(
            history=state.get("history", ""), question=state["question"], sources=format_sources(state["sources"])
        )
        try:
            return {"answer": await self._answerer.ainvoke(messages)}
        except LLM_ERRORS as error:
            return {"error": f"The model returned no valid answer: {error}"}

    def not_found(self, state: DocAgentState) -> DocAgentState:
        """Node: nothing relevant was retrieved; answer that without calling an LLM."""
        return {"answer": DocAnswer(answer_found=False, answer=NOT_FOUND_TEXT, citations=[])}

    def route_after_rerank(self, state: DocAgentState) -> Literal["answer", "not_found"]:
        """Router: not_found when there are no sources, or the best one scores below RAG_MIN_RERANK_SCORE."""
        sources = state.get("sources") or []
        if not sources:
            return "not_found"
        best = sources[0].rerank_score
        if self._min_rerank_score is not None and best is not None and best < self._min_rerank_score:
            return "not_found"
        return "answer"

    # ------------------------------------------------------------- result

    def _result(self, question: str, state: DocAgentState, steps: list[str]) -> DocAgentResult:
        """Turn the final state into a DocAgentResult: clean citations, decide `grounded`, attach the trace."""
        sources: list[Hit] = state.get("sources") or []
        used_ids = {hit.chunk_id for hit in sources}
        trace = [
            RetrievalTrace(
                chunk_id=h.chunk_id,
                chunk_type=h.chunk_type,
                page_start=h.page_start,
                dense_rank=h.dense_rank,
                keyword_rank=h.keyword_rank,
                rrf_score=h.rrf_score,
                rerank_score=h.rerank_score,
                used=h.chunk_id in used_ids,
            )
            for h in state.get("candidates") or []
        ]
        reply: DocAnswer | None = state.get("answer")
        if reply is None:  # the answer LLM failed
            return DocAgentResult(
                question=question, answer="Sorry, I couldn't write an answer right now. Please try again.",
                success=False, grounded=False, standalone_query=state.get("standalone_query", question),
                retrieval=trace, steps=steps, error=state.get("error"),
                reasoning=[*explain_search(question, state), "The answer model failed, so no answer was written."],
            )  # fmt: skip

        text, numbers = clean_citations(reply.answer, reply.citations, len(sources))
        citations = [to_citation(n, sources[n - 1]) for n in numbers]
        grounded = reply.answer_found and bool(citations)
        reasoning = [*explain_search(question, state), explain_outcome(steps, grounded, len(citations), len(sources))]
        return DocAgentResult(
            question=question, answer=text, success=grounded, grounded=grounded,
            standalone_query=state.get("standalone_query", question), citations=citations, retrieval=trace, steps=steps,
            reasoning=reasoning,
        )  # fmt: skip


def explain_search(question: str, state: DocAgentState) -> list[str]:
    """Plain sentences on how the search went (UI flow panel), from facts in the final state, no LLM call."""
    lines = []
    standalone = state.get("standalone_query", question)
    if not same_words(standalone, question):
        lines.append(f'Rewrote the follow-up into a standalone search query: "{standalone}".')
    candidates: list[Hit] = state.get("candidates") or []
    both = sum(hit.dense_rank is not None and hit.keyword_rank is not None for hit in candidates)
    lines.append(
        f"Searched this chat's documents by meaning and by keywords: {len(candidates)} candidate chunks"
        + (f", {both} of them found by both searches." if candidates else ".")
    )
    sources: list[Hit] = state.get("sources") or []
    if sources:
        best = sources[0]
        score = f", rerank score {best.rerank_score:+.2f}" if best.rerank_score is not None else ""
        lines.append(
            f"Kept the {len(sources)} most relevant as sources; the best match is a "
            f"{CHUNK_KINDS.get(best.chunk_type, best.chunk_type)} on page "
            f"{best.page_start} of {best.filename}{score}."
        )
    return lines


def same_words(a: str, b: str) -> bool:
    """True if two texts have the same words, ignoring case and punctuation.

    The rewriter often hands a question back unchanged except for a dropped "?"; that is not a
    rewrite worth telling the reader about.
    """
    return re.findall(r"\w+", a.casefold()) == re.findall(r"\w+", b.casefold())


def explain_outcome(steps: list[str], grounded: bool, cited: int, sources: int) -> str:
    """One sentence on how the answer was produced (or why there is none)."""
    if "not_found" in steps:
        return "Nothing relevant was found, so it says so without asking the answer model."
    if grounded:
        return f"The answer model wrote the answer from the sources and cites {cited} of the {sources}."
    return "The sources didn't contain the answer, so it says so instead of guessing."


def format_sources(sources: list[Hit]) -> str:
    """Number the sources [1]..[n] with where each comes from, so the LLM can cite them and say "page 4"."""
    blocks = []
    for n, hit in enumerate(sources, start=1):
        pages = f"page {hit.page_start}" if hit.page_start == hit.page_end else f"pages {hit.page_start}-{hit.page_end}"
        section = f", section: {hit.heading_path}" if hit.heading_path else ""
        blocks.append(f"[{n}] ({hit.filename}, {pages}{section}, {hit.chunk_type})\n{hit.content}")
    return "\n\n".join(blocks)


def to_citation(n: int, hit: Hit) -> Citation:
    """A Citation for source [n], with a short snippet and, for pictures, the crop's URL."""
    picture_url = f"/api/documents/{hit.document_id}/pictures/{hit.chunk_id}" if hit.chunk_type == "picture" else None
    readable = [line for line in hit.content.splitlines() if not TABLE_SEPARATOR.fullmatch(line.strip())]
    snippet = " ".join(" ".join(readable).split())[:SNIPPET_CHARS]  # one line, without "|---|---|" rows
    return Citation(
        n=n, chunk_id=hit.chunk_id, document_id=hit.document_id, filename=hit.filename, page_start=hit.page_start,
        page_end=hit.page_end, heading_path=hit.heading_path, chunk_type=hit.chunk_type, snippet=snippet,
        picture_url=picture_url,
    )  # fmt: skip
