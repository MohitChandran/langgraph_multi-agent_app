"""Dummy web search agent: returns a hardcoded result. No network, no LLM, no MCP.

Why it exists: the orchestrator plans and routes between four agents (db, doc, web, email), but the
real web search agent (Phase 4, DuckDuckGo through MCP) isn't built yet. This stand-in has the same
shape as the real agents (a `name` and an async `ask`), so the orchestrator's planning, parallel
steps and answer combining can be built and tested now. When the real agent is built, it replaces
this class and the code calling it doesn't change.

The result is clearly fake on purpose: the answer starts with "[dummy]", the provider is a made-up
"Example Cloud", and every URL uses example.com (a domain reserved for examples), so nobody mistakes
it for real web data.

Try it:  uv run python -m app.agents.web.dummy "Did our cloud vendor change its SLA recently?"
"""

import asyncio
import sys
from uuid import UUID

from app.agents.web.models import WebSearchResult, WebSource
from app.chat.memory import ConversationTurn

DUMMY_ANSWER = (
    "[dummy] Example Cloud updated its compute SLA on 2026-10-01: the monthly uptime commitment is now 99.99%, "
    "with a 10% service credit below 99.99% and 30% below 99.0% [1]. "
    "A news article says the new terms apply to all contracts renewed from November 2026 [2]."
)


DUMMY_SOURCES = [
    WebSource(
        n=1,
        title="Example Cloud Compute Service Level Agreement",
        url="https://example.com/example-cloud/sla",
        snippet="Effective 2026-10-01. Monthly Uptime Percentage commitment: 99.99%. Service credits: 10% below "
        "99.99%, 30% below 99.0%.",
    ),
    WebSource(
        n=2,
        title="Example Cloud tightens its uptime SLA (Example Tech News)",
        url="https://example.com/news/example-cloud-sla-update",
        snippet="The revised SLA applies to all contracts renewed from November 2026 onward.",
    ),
]


class DummyWebSearchAgent:
    """Stand-in for the web search agent: same interface as DBAgent/DocAgent, fixed result.

    `delay_seconds` simulates a slow web search. It makes the orchestrator's parallel steps
    visible: two agents that each take 1 s should finish together in ~1 s, not 2 s.
    """

    name = "web"  # saved in chat_messages.payload.agent, like "db" and "doc"

    def __init__(self, delay_seconds: float = 0.0) -> None:
        """Store how long each ask() should pretend to take (0 = answer instantly)."""
        self.delay_seconds = delay_seconds

    async def ask(
        self, question: str, history: list[ConversationTurn] | None = None, session_id: UUID | None = None
    ) -> WebSearchResult:
        """Return the hardcoded web result for any question.

        `history` and `session_id` are accepted (and ignored) so this fits the same interface as the
        real agents. The question is echoed back as the search query.
        """
        await asyncio.sleep(self.delay_seconds)
        return WebSearchResult(
            question=question,
            answer=DUMMY_ANSWER,
            success=True,
            query=question.strip(),
            sources=[source.model_copy() for source in DUMMY_SOURCES],
            steps=["dummy_search"],
            reasoning=["Dummy agent: returned a hardcoded result. No web search was made."],
        )


async def _main(question: str) -> None:
    """CLI helper: print the dummy result for a question."""
    result = await DummyWebSearchAgent().ask(question)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(_main(" ".join(sys.argv[1:]) or "Did our cloud vendor change its SLA recently?"))
