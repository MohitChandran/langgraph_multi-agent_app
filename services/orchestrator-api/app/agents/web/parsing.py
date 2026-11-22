"""Turning the DuckDuckGo MCP server's text output into numbered sources (plannings/04, 8.4 and 8.5).

The server returns TEXT, not JSON (duckduckgo-mcp-server 0.7.0, pinned):

    Found 3 search results:

    1. Alexander the Great - Wikipedia
       URL: https://en.wikipedia.org/wiki/Alexander_the_Great
       Summary: Alexander III of Macedon ... was king of ...

A real response is saved as tests/fixtures/ddg_search_results.txt and the parser is tested against it: if a
new server version changes the format, that test fails instead of the agent quietly finding nothing.

Failures come back as text too. "No results" and "DuckDuckGo blocked us as a bot" are the SAME message, so
the answer has to say either may be the cause (plan gotcha 5).
"""

import re

from app.agents.web.models import WebSource

ITEM = re.compile(
    r"^\s*(\d+)\.\s+([^\n]+?)\s*\n\s*URL:\s*(\S+)\s*\n\s*Summary:\s*(.*?)(?=\n\s*\d+\.\s|\Z)",
    re.DOTALL | re.MULTILINE,
)
NO_RESULTS_PREFIX = "No results were found for your search query."
ERROR_PREFIX = "An error occurred while searching:"
CUT = "…"

NO_RESULTS_TEXT = (
    "I found no web results for that. DuckDuckGo may also be limiting searches; try again in a few minutes or rephrase."
)
UNAVAILABLE_TEXT = "Web search is temporarily unavailable."


def parse_results(text: str, snippet_chars: int = 400) -> list[WebSource]:
    """The search results as WebSources numbered 1..n, in the server's order.

    Dropped: URLs that aren't http(s) (nothing else should become a clickable link) and repeated URLs.
    Each snippet is put on one line and cut to `snippet_chars`, so a result can't flood the answer prompt.
    """
    sources: list[WebSource] = []
    seen: set[str] = set()
    for match in ITEM.finditer(text):
        title, url, summary = match.group(2).strip(), match.group(3).strip(), " ".join(match.group(4).split())
        if not url.startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        snippet = summary if len(summary) <= snippet_chars else summary[: snippet_chars - 1].rstrip() + CUT
        sources.append(WebSource(n=len(sources) + 1, title=title, url=url, snippet=snippet))
    return sources


def failure_text(text: str) -> tuple[str, str] | None:
    """(user-facing answer, technical detail) when the server's text is a failure message, else None (plan 8.5)."""
    stripped = text.strip()
    if stripped.startswith(NO_RESULTS_PREFIX):
        return NO_RESULTS_TEXT, "no results (or DuckDuckGo bot detection)"
    if stripped.startswith(ERROR_PREFIX):
        reason = " ".join(stripped.removeprefix(ERROR_PREFIX).split())[:200]
        return f"Web search failed: {reason}.", reason
    return None
