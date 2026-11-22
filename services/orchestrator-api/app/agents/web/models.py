"""Data shapes of the web search agent (plannings/04, 8.1).

WebSource and WebSearchResult were first written for the dummy agent (plan 03) and moved here unchanged, so
the real agent, the dummy (kept for tests) and the orchestrator share one definition.
"""

from typing import TypedDict

from pydantic import BaseModel, Field


class WebSource(BaseModel):
    """One web page the answer used; the UI shows it as a numbered link under the answer."""

    n: int  # the [n] used in the answer text
    title: str
    url: str
    snippet: str  # a short excerpt of the page


class WebSearchResult(BaseModel):
    """What the web search agent returns. The core fields match DBAgentResult and DocAgentResult.

    Every agent result has question, answer, success, steps, reasoning and error, so the chat
    service and the orchestrator can treat all agents alike; `query` and `sources` are web-specific.
    """

    question: str
    answer: str
    success: bool
    query: str = Field(description="The search query that was sent to the search engine.")
    sources: list[WebSource] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list, description="Workflow steps in the order they ran.")
    reasoning: list[str] = Field(default_factory=list, description="Short sentences on what the agent did and why.")
    error: str | None = None


class WebAnswer(BaseModel):
    """What the answer LLM must return (structured output), like the Doc agent's DocAnswer."""

    answer_found: bool = Field(description="false if the numbered search results do not contain the answer.")
    answer: str = Field(description="The answer in markdown, with [n] after every sentence that uses result n.")
    citations: list[int] = Field(description="Numbers of the results the answer used, e.g. [1, 3].")


class WebAgentState(TypedDict, total=False):
    """What the web agent's graph knows at each step (one search per question)."""

    question: str  # the search_web query the orchestrator wrote (also what is searched)
    rejected: str | None  # why the query check refused it (company data), or None
    raw: str  # the MCP server's text output
    sources: list[WebSource]  # parsed results, numbered 1..n
    reply: WebAnswer  # the answer LLM's structured output
    failure: str | None  # a user-facing reason the search produced no answer (no results, error, unavailable)
    error: str | None  # technical detail for logs and the UI
