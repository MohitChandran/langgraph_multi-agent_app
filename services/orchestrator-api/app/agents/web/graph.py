"""The web search agent: a FIXED LangGraph workflow over the DuckDuckGo MCP `search` tool (plannings/04, 8.2).

    START -> check_query --rejected--------------------------------------------> finish -> END
                 '--ok--> search --failed (no results / error / unavailable)----> finish
                            '--ok--> parse --no usable results------------------> finish
                                       '--results--> answer --------------------> finish

Why a fixed workflow and not a tool-calling agent (plan 04, D8): the orchestrator is already the loop that
decides WHAT to search. This agent does one search per call, in a fixed order, so the query check can never
be skipped, and it makes at most one LLM call (the cited answer), none when there is nothing to answer from.

The MCP server is the agent's tool: `search` runs in the server process and returns text, which parse()
turns into numbered sources. The answer cites them as [n]; the WebSearchResult keeps only the cited ones,
with their original numbers (the orchestrator relabels them [W1], [W2]...).

Same interface as the dummy and the other agents: `name`, `async ask(question, history, session_id)`.

Try it (needs network + Groq):
    uv run python -m app.agents.web.graph "history of the Battle of Gaugamela"
"""

import argparse
import asyncio
import logging
import time
from typing import Any, Literal, Protocol
from uuid import UUID

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from app.agents.citations import CITATION_MARK, clean_citations
from app.agents.web.models import WebAgentState, WebAnswer, WebSearchResult, WebSource
from app.agents.web.parsing import UNAVAILABLE_TEXT, failure_text, parse_results
from app.agents.web.prompts import ANSWER_PROMPT
from app.agents.web.query_guard import rejected_reason, rejection_text
from app.chat.memory import ConversationTurn
from app.config import get_settings
from app.llm import LLM_ERRORS, get_llm

logger = logging.getLogger(__name__)

# gpt-oss's hidden reasoning counts toward max_tokens: 700 was too few for "why" questions (Groq answered
# json_validate_failed: "max completion tokens reached", found by the W6 smoke eval). 2,000 still bounds the cost.
ANSWER_MAX_TOKENS = 2000
NOT_FOUND_TEXT = "The web results I found don't answer this."
ANSWER_FAILED_TEXT = "I found web results but couldn't write an answer right now. Please try again."


class SearchBackend(Protocol):
    """What the agent needs from the MCP connection (WebSearchMCP, or a fake in tests)."""

    available: bool

    async def search(self, query: str, max_results: int) -> str:
        """The server's text output for one search."""
        ...


class WebSearchAgent:
    """Answers questions from live web search results, with [n] citations that are real URLs.

    Usage:
        agent = WebSearchAgent(web_mcp)
        result = await agent.ask("history of the Battle of Gaugamela")

    The search backend and the answer LLM are injected, so tests use fakes (no network, no Groq).
    """

    name = "web"  # saved in chat_messages.payload.agent, like "db" and "doc"

    def __init__(
        self,
        backend: SearchBackend,
        answerer: Runnable[list[BaseMessage], WebAnswer] | None = None,
        max_results: int = 6,
        snippet_chars: int = 400,
    ) -> None:
        """Store the dependencies and compile the workflow once.

        Args:
            backend: the MCP connection (WebSearchMCP).
            answerer: returns a WebAnswer. Default: the strong Groq model with structured output (plan 04, D9).
            max_results: results per search (WEB_MAX_RESULTS); snippet_chars: cap per snippet (WEB_SNIPPET_CHARS).
        """
        self.backend = backend
        self._answerer = answerer or get_llm("strong", max_tokens=ANSWER_MAX_TOKENS).with_structured_output(
            WebAnswer, method="json_schema"
        )
        self.max_results = max_results
        self.snippet_chars = snippet_chars
        self.graph = self._build_graph()

    @property
    def available(self) -> bool:
        """False when the MCP server isn't running: the orchestrator then doesn't offer search_web."""
        return self.backend.available

    def _build_graph(self):  # noqa: ANN202  (LangGraph's compiled type adds nothing here)
        """Wire the nodes and routers together."""
        builder = StateGraph(WebAgentState)
        builder.add_node("check_query", self.check_query)
        builder.add_node("search", self.search)
        builder.add_node("parse", self.parse)
        builder.add_node("answer", self.answer)
        builder.add_edge(START, "check_query")
        builder.add_conditional_edges("check_query", self.route_after_check)
        builder.add_conditional_edges("search", self.route_after_search)
        builder.add_conditional_edges("parse", self.route_after_parse)
        builder.add_edge("answer", END)
        return builder.compile()

    async def ask(
        self, question: str, history: list[ConversationTurn] | None = None, session_id: UUID | None = None
    ) -> WebSearchResult:
        """Search the web for one question and answer it with citations.

        `question` is the query the orchestrator wrote for search_web (already self-contained), so it's searched
        as is. history and session_id are accepted for the common agent interface and not used.
        """
        steps: list[str] = []
        state: WebAgentState = {}
        async for mode, chunk in self.graph.astream({"question": question}, stream_mode=["updates", "values"]):
            if mode == "updates":
                steps.extend(chunk)
            else:
                state = chunk
        return build_result(question, state, steps)

    # ------------------------------------------------------------------ nodes

    def check_query(self, state: WebAgentState) -> WebAgentState:
        """Node: refuse queries carrying company data (emails, ids, phone or account numbers): nothing is sent."""
        return {"rejected": rejected_reason(state["question"])}

    async def search(self, state: WebAgentState) -> WebAgentState:
        """Node: one MCP search; the server's failure texts and any exception become a `failure`."""
        try:
            raw = await self.backend.search(state["question"], self.max_results)
        except Exception as error:  # noqa: BLE001  (timeout, server gone, tool error: the orchestrator continues)
            logger.warning("Web search failed: %s: %s", type(error).__name__, error)
            return {"failure": UNAVAILABLE_TEXT, "error": f"{type(error).__name__}: {error}"[:300]}
        failed = failure_text(raw)
        if failed:
            return {"raw": raw, "failure": failed[0], "error": failed[1]}
        return {"raw": raw}

    def parse(self, state: WebAgentState) -> WebAgentState:
        """Node: the text into numbered sources; none usable is reported like "no results"."""
        sources = parse_results(state["raw"], self.snippet_chars)
        if not sources:
            return {"sources": [], "failure": failure_text("No results were found for your search query.")[0]}
        return {"sources": sources}

    async def answer(self, state: WebAgentState) -> WebAgentState:
        """Node: the strong LLM writes the cited answer from the numbered results (the only LLM call)."""
        messages = ANSWER_PROMPT.format_messages(question=state["question"], sources=format_sources(state["sources"]))
        try:
            return {"reply": await self._answerer.ainvoke(messages)}
        except LLM_ERRORS as error:
            return {"error": f"The model returned no valid answer: {error}"[:300]}

    # ------------------------------------------------------------------ routers

    def route_after_check(self, state: WebAgentState) -> Literal["search", "__end__"]:
        """A rejected query ends here: it is never searched."""
        return END if state.get("rejected") else "search"

    def route_after_search(self, state: WebAgentState) -> Literal["parse", "__end__"]:
        """A failed search ends here, without an LLM call."""
        return END if state.get("failure") else "parse"

    def route_after_parse(self, state: WebAgentState) -> Literal["answer", "__end__"]:
        """No usable results: end without an LLM call."""
        return "answer" if state.get("sources") else END


# ---------------------------------------------------------------------- pure helpers


def format_sources(sources: list[WebSource]) -> str:
    """The numbered results as the answer prompt shows them: [n] title, URL, snippet."""
    return "\n\n".join(f"[{s.n}] {s.title}\nURL: {s.url}\n{s.snippet}" for s in sources)


def build_result(question: str, state: dict[str, Any], steps: list[str]) -> WebSearchResult:
    """Turn the final state into a WebSearchResult: the answer (or why there is none), the cited sources, the trace.

    success = the model found the answer AND cited at least one real result. Only cited sources are returned,
    keeping their original numbers, so every [n] in the answer matches a source. If the model listed its sources
    but wrote no [n] in the text (seen in the W6 smoke eval), the marks are appended once at the end, so the
    answer always points to the sources shown under it.
    """
    common = {"question": question, "query": question, "steps": steps, "error": state.get("error")}
    if state.get("rejected"):
        reason = state["rejected"]
        return WebSearchResult(
            **common, answer=rejection_text(reason), success=False,
            reasoning=[f"Rejected the query before searching: it contains {reason}. Nothing was sent to the web."],
        )  # fmt: skip
    if state.get("failure"):
        return WebSearchResult(
            **common, answer=state["failure"], success=False,
            reasoning=[f'Searched DuckDuckGo for "{question}": {state.get("error") or "no usable results"}.'],
        )  # fmt: skip
    sources: list[WebSource] = state.get("sources") or []
    searched = f'Searched DuckDuckGo for "{question}" → {len(sources)} results.'
    reply: WebAnswer | None = state.get("reply")
    if reply is None:
        return WebSearchResult(
            **common, answer=ANSWER_FAILED_TEXT, success=False,
            reasoning=[searched, "The answer model failed, so no answer was written."],
        )  # fmt: skip
    text, numbers = clean_citations(reply.answer, reply.citations, len(sources))
    cited = [sources[n - 1] for n in sorted(numbers)]  # listed 1, 2, 3... however they were cited
    success = reply.answer_found and bool(cited)
    if success and not CITATION_MARK.search(text):
        text = f"{text} " + "".join(f"[{source.n}]" for source in cited)
    outcome = (
        f"Answered from {len(cited)} of them, each claim cited."
        if success
        else "The results didn't contain the answer, so none is claimed."
    )
    return WebSearchResult(
        **common, answer=text if success else (text or NOT_FOUND_TEXT), success=success, sources=cited,
        reasoning=[searched, outcome],
    )  # fmt: skip


async def _main() -> None:
    """CLI: start the MCP server, ask one question, print the cited answer and the sources."""
    from app.agents.web.mcp import WebSearchMCP

    parser = argparse.ArgumentParser(description="Ask the web search agent a question.")
    parser.add_argument("question", nargs="*", help="the question / search query")
    args = parser.parse_args()
    question = " ".join(args.question) or "history of the Battle of Gaugamela"
    settings = get_settings()
    async with WebSearchMCP(settings) as mcp:
        agent = WebSearchAgent(mcp, max_results=settings.web_max_results, snippet_chars=settings.web_snippet_chars)
        started = time.perf_counter()
        result = await agent.ask(question)
    print(f"Question: {question}\nSteps   : {' -> '.join(result.steps)}   success: {result.success}")
    print(f"Time    : {time.perf_counter() - started:.1f}s")
    for line in result.reasoning:
        print(f"  - {line}")
    print(f"\nAnswer:\n{result.answer}\n")
    for source in result.sources:
        print(f"[{source.n}] {source.title}\n    {source.url}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s - %(message)s")
    asyncio.run(_main())
