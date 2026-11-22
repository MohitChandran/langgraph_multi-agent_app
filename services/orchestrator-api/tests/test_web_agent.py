"""Tests for the web search agent (app/agents/web, plannings/04 stages W3-W4).

No network and no Groq: a FakeBackend stands in for the MCP server (it returns the REAL server output saved in
tests/fixtures/ddg_search_results.txt, or a failure text, or raises) and a FakeAnswerer for the LLM. One live
test starts the real server through uvx; it's marked `network` and skipped unless you run `pytest -m network`.
"""

from pathlib import Path

import pytest

from app.agents.web.graph import ANSWER_FAILED_TEXT, WebSearchAgent
from app.agents.web.mcp import WebSearchMCP, WebSearchUnavailableError, content_text, server_config
from app.agents.web.models import WebAnswer
from app.agents.web.parsing import NO_RESULTS_TEXT, UNAVAILABLE_TEXT, failure_text, parse_results
from app.agents.web.query_guard import looks_like_phone, rejected_reason
from app.config import get_settings
from app.llm import LLM_ERRORS

FIXTURE = (Path(__file__).parent / "fixtures" / "ddg_search_results.txt").read_text()
NO_RESULTS = (
    "No results were found for your search query. This could be due to DuckDuckGo's bot detection or the query "
    "returned no matches. Please try rephrasing your search or try again in a few minutes."
)


class FakeBackend:
    """Stands in for WebSearchMCP: returns `reply` (text) or raises it; records every search."""

    def __init__(self, reply: object = FIXTURE, available: bool = True) -> None:
        """Set what each search returns."""
        self.reply, self.available = reply, available
        self.searches: list[tuple[str, int]] = []

    async def search(self, query: str, max_results: int) -> str:
        """Record the call, then return or raise the scripted reply."""
        self.searches.append((query, max_results))
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply


class FakeAnswerer:
    """Stands in for the structured-output LLM: returns a fixed WebAnswer (or raises); records the prompts."""

    def __init__(self, reply: WebAnswer | BaseException) -> None:
        """Set the scripted reply."""
        self.reply = reply
        self.prompts: list = []

    async def ainvoke(self, messages: list) -> WebAnswer:
        """Record the messages, then return or raise."""
        self.prompts.append(messages)
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply


def agent(backend: FakeBackend | None = None, reply: WebAnswer | BaseException | None = None) -> WebSearchAgent:
    """A WebSearchAgent on fakes; by default the answer cites results 1 and 3."""
    answer = reply or WebAnswer(answer_found=True, answer="Fought in 331 BC [1]. Alexander won [3].", citations=[1, 3])
    return WebSearchAgent(backend or FakeBackend(), FakeAnswerer(answer))


# --------------------------------------------------------------------------- query check (section 7)


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("contact anna@company.example about the SLA", "an email address"),
        ("payment status of INV-2026-0273", "an invoice or contract number"),
        ("renewal terms ctr-2024-013", "an invoice or contract number"),
        ("IBAN GB82WEST12345698765432", "a long account-like number"),
        ("call +91 98765 43210", "a phone number"),
        ("office (022) 2345-6789", "a phone number"),
        ("x" * 301, "it is longer than 300 characters"),
    ],
)
def test_queries_carrying_company_data_are_rejected(query: str, reason: str) -> None:
    """Each kind of company data the supervisor could leak into a search is caught, with the reason."""
    assert rejected_reason(query) == reason


@pytest.mark.parametrize(
    "query", ["AWS EC2 uptime SLA 2026", "SLA changes 2025 2026", "history of the Battle of Gaugamela", "GPT-4 release"]
)
def test_normal_queries_pass(query: str) -> None:
    """Public questions pass, including years side by side (not a phone number)."""
    assert rejected_reason(query) is None and not looks_like_phone(query)


# --------------------------------------------------------------------------- parsing (8.4, 8.5)


def test_the_real_server_output_parses_into_numbered_sources() -> None:
    """The saved real response: 6 sources with title, http(s) URL and a one-line snippet, numbered 1..6."""
    sources = parse_results(FIXTURE)
    assert [s.n for s in sources] == [1, 2, 3, 4, 5, 6]
    assert sources[0].title == "Battle of Gaugamela - Wikipedia"
    assert sources[0].url == "https://en.wikipedia.org/wiki/Battle_of_Gaugamela"
    assert all(s.url.startswith("https://") and s.snippet and "\n" not in s.snippet for s in sources)


def test_parsing_drops_bad_and_duplicate_urls_and_caps_snippets() -> None:
    """A non-http URL and a repeated URL are dropped (renumbering the rest); long snippets are cut."""
    text = (
        "Found 3 search results:\n\n"
        "1. Good\n   URL: https://a.example/x\n   Summary: " + "word " * 200 + "\n\n"
        "2. Bad scheme\n   URL: javascript:alert(1)\n   Summary: no\n\n"
        "3. Same again\n   URL: https://a.example/x\n   Summary: dup\n\n"
        "4. Other\n   URL: http://b.example/y\n   Summary: fine\n"
    )
    sources = parse_results(text, snippet_chars=50)
    assert [(s.n, s.url) for s in sources] == [(1, "https://a.example/x"), (2, "http://b.example/y")]
    assert len(sources[0].snippet) == 50 and sources[0].snippet.endswith("…")


def test_the_servers_failure_texts_are_recognized() -> None:
    """ "No results" (which is also what a bot block looks like) and "An error occurred" map to honest answers."""
    assert failure_text(NO_RESULTS)[0] == NO_RESULTS_TEXT
    answer, detail = failure_text("An error occurred while searching: HTTP 503")
    assert answer == "Web search failed: HTTP 503." and detail == "HTTP 503"
    assert failure_text(FIXTURE) is None


# --------------------------------------------------------------------------- the agent (8.2)


async def test_a_cited_answer_keeps_only_the_cited_sources_with_their_numbers() -> None:
    """Happy path: one search, one LLM call, success; the result lists sources 1 and 3 (not renumbered)."""
    web = agent()
    result = await web.ask("history of the Battle of Gaugamela")
    assert result.success and result.answer == "Fought in 331 BC [1]. Alexander won [3]."
    assert [s.n for s in result.sources] == [1, 3] and result.sources[1].url.startswith("https://")
    assert result.steps == ["check_query", "search", "parse", "answer"] and result.query == result.question
    assert web.backend.searches == [("history of the Battle of Gaugamela", 6)]  # one search, max 6 results
    prompt = web._answerer.prompts[0][-1].content
    assert "[1] Battle of Gaugamela - Wikipedia\nURL: https://en.wikipedia.org/wiki/Battle_of_Gaugamela" in prompt


async def test_citation_numbers_that_dont_exist_are_dropped() -> None:
    """[9] with 6 results is removed from the text and the list; the real [2] stays."""
    reply = WebAnswer(answer_found=True, answer="True [2]. Invented [9].", citations=[2, 9])
    result = await agent(reply=reply).ask("q")
    assert result.answer == "True [2]. Invented ." and [s.n for s in result.sources] == [2]


async def test_answer_not_found_is_not_a_success() -> None:
    """The model says the results don't answer it: success False, no sources claimed."""
    reply = WebAnswer(answer_found=False, answer="The results don't say.", citations=[])
    result = await agent(reply=reply).ask("q")
    assert not result.success and result.sources == [] and result.answer == "The results don't say."


async def test_no_results_and_server_errors_end_without_an_llm_call() -> None:
    """The server's "no results" / error texts and an exception give honest answers; the LLM is never called."""
    for reply, expected in [
        (NO_RESULTS, NO_RESULTS_TEXT),
        ("An error occurred while searching: rate limited", "Web search failed: rate limited."),
        (TimeoutError(), UNAVAILABLE_TEXT),
        ("Found 0 search results:\n", NO_RESULTS_TEXT),  # nothing parsable
    ]:
        web = agent(FakeBackend(reply))
        result = await web.ask("q")
        assert not result.success and result.answer == expected and web._answerer.prompts == []


async def test_a_rejected_query_is_never_searched() -> None:
    """An email address in the query: rejected at check_query, the backend is never called."""
    web = agent()
    result = await web.ask("status for anna@company.example")
    assert not result.success and result.steps == ["check_query"] and web.backend.searches == []
    assert result.answer.startswith("I can't send internal company data (an email address)")


async def test_an_llm_failure_gives_a_try_again_answer() -> None:
    """The answer model failing (a Groq error) is a failed result, not a crash."""
    error = next(e for e in LLM_ERRORS if e.__name__ == "OutputParserException")("bad json")
    result = await agent(reply=error).ask("q")
    assert not result.success and result.answer == ANSWER_FAILED_TEXT and result.error


def test_availability_follows_the_mcp_connection() -> None:
    """The orchestrator offers search_web only when the MCP session is open."""
    assert agent(FakeBackend(available=True)).available
    assert not agent(FakeBackend(available=False)).available


# --------------------------------------------------------------------------- the MCP connection (section 6)


def test_server_config_pins_the_package_and_keeps_real_urls() -> None:
    """uvx --from <pinned package>, real URLs (no ref:// tokens), SafeSearch and region via the environment."""
    config = server_config(get_settings())
    assert config["command"] == "uvx" and config["transport"] == "stdio"
    assert config["args"][:3] == ["--from", get_settings().web_mcp_package, "duckduckgo-mcp-server"]
    assert "--ref-url-threshold" in config["args"] and config["env"]["DDG_SAFE_SEARCH"] == "STRICT"


def test_content_text_joins_text_blocks() -> None:
    """MCP tool results come as content blocks; the text parts are joined."""
    assert content_text([{"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}]) == "ab"
    assert content_text("plain") == "plain"


async def test_searching_before_the_session_is_open_raises() -> None:
    """A WebSearchMCP that never started is unavailable and says so."""
    mcp = WebSearchMCP(get_settings())
    assert not mcp.available
    with pytest.raises(WebSearchUnavailableError):
        await mcp.search("q", 3)


@pytest.mark.network
async def test_the_real_server_starts_keeps_only_search_and_answers() -> None:
    """Live: uvx starts the pinned server; it offers 3 tools, we keep `search`; one real search parses."""
    async with WebSearchMCP(get_settings()) as mcp:
        assert mcp.available and set(mcp.server_tools) >= {"search", "fetch_content", "expand_link"}
        sources = parse_results(await mcp.search("Battle of Gaugamela", 3))
    assert sources and all(s.url.startswith("http") for s in sources)
