"""The connection to the DuckDuckGo MCP server: one long-lived session, only its `search` tool (plannings/04, 6).

MCP (Model Context Protocol) lets an app use tools that live in ANOTHER program: the app (MCP client) starts
or connects to an MCP server, asks it which tools it has (`list_tools`), and calls them (`call_tool`). Here
the server is the open-source `duckduckgo-mcp-server`; it scrapes DuckDuckGo's HTML results (no API key).

    mcp = WebSearchMCP(settings)
    await mcp.start()                       # starts the server process, opens the session (~2-3 s, once)
    text = await mcp.search("Battle of Gaugamela", max_results=6)   # ~1-3 s per search
    await mcp.close()

Design (plan 04, D6/D7, gotchas 1-3):
  - The server runs as its OWN process via `uvx --from duckduckgo-mcp-server==<pinned>`: it needs `mcp` 2.x,
    langchain-mcp-adapters needs `mcp` 1.x, so they can't share our environment. Talking over stdio, the
    versions don't matter (the protocol is the same). Never add the server to pyproject.toml.
  - ONE session for the app's lifetime: MultiServerMCPClient.get_tools() would start a new server process for
    every call. client.session("ddg") + load_mcp_tools(session) keeps one.
  - Client-side allow-list: the server also offers fetch_content (whole pages) and expand_link; we keep only
    `search` (a server can't switch its own tools off; we simply never load the others).
  - If the server can't start (no network, wrong version), `available` stays False and the orchestrator
    doesn't offer search_web; the rest of the app works. Restart the API to retry.
  - The server logs to stderr (and prints MCP deprecation warnings): normal, not an error.

Try it (needs network):
    uv run python -m app.agents.web.mcp "Battle of Gaugamela"
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import AsyncExitStack
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

SERVER_NAME = "ddg"
ALLOWED_TOOL = "search"
START_TIMEOUT_SECONDS = 90  # the first start downloads the server package with uvx


class WebSearchUnavailableError(RuntimeError):
    """The MCP session isn't open (the server failed to start, or the app is shutting down)."""


def server_config(settings: Settings) -> dict[str, Any]:
    """How to start the DuckDuckGo MCP server: a stdio process launched with uvx, pinned, with our settings.

    --ref-url-threshold 100000: the server would replace long URLs with ref:// tokens; we want the real URLs,
    so citations are real links. SafeSearch and region come from the environment (DDG_SAFE_SEARCH, DDG_REGION).
    PATH is passed so the child process finds uvx (the MCP client doesn't inherit the whole environment).
    """
    return {
        "transport": "stdio",
        "command": "uvx",
        "args": [
            "--from", settings.web_mcp_package, "duckduckgo-mcp-server",
            "--ref-url-threshold", "100000",
            "--search-rpm", str(settings.web_search_rpm),
        ],
        "env": {
            "DDG_SAFE_SEARCH": settings.web_safe_search,
            "DDG_REGION": settings.web_region,
            "PATH": os.environ.get("PATH", ""),
        },
    }  # fmt: skip


def content_text(content: Any) -> str:
    """The text of a tool result: MCP returns a list of content blocks ([{"type": "text", "text": ...}]) or a string."""
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


class WebSearchMCP:
    """One long-lived MCP session to the DuckDuckGo server (a separate process started with uvx)."""

    def __init__(self, settings: Settings) -> None:
        """Prepare the server config; nothing is started until start()."""
        self.config = server_config(settings)
        self.timeout_seconds = settings.web_search_timeout_seconds
        self.available = False
        self.server_tools: list[str] = []  # every tool the server offers (we keep only ALLOWED_TOOL)
        self._search_tool: Any = None
        self._stack: AsyncExitStack | None = None

    async def start(self) -> None:
        """Start the server and open the session. Never raises: on failure, `available` stays False (logged)."""
        try:
            await asyncio.wait_for(self._open(), START_TIMEOUT_SECONDS)
        except Exception:  # noqa: BLE001  (the app must start without web search rather than not at all)
            logger.exception("The DuckDuckGo MCP server could not start; search_web will not be offered")
            await self.close()
            return
        logger.info("DuckDuckGo MCP server ready (tools offered: %s; using: %s)", self.server_tools, ALLOWED_TOOL)

    async def _open(self) -> None:
        """Open the stdio session, list the server's tools and keep only `search`."""
        self._stack = AsyncExitStack()
        client = MultiServerMCPClient({SERVER_NAME: self.config})
        session = await self._stack.enter_async_context(client.session(SERVER_NAME))
        # handle_tool_errors=False: a tool error raises (ToolException) instead of becoming text we'd have to sniff.
        tools = await load_mcp_tools(session, handle_tool_errors=False)
        self.server_tools = [tool.name for tool in tools]
        self._search_tool = next((tool for tool in tools if tool.name == ALLOWED_TOOL), None)
        if self._search_tool is None:
            raise RuntimeError(f"the server has no '{ALLOWED_TOOL}' tool (it offers {self.server_tools})")
        self.available = True

    async def search(self, query: str, max_results: int) -> str:
        """Run one search and return the server's text (see parsing.py for its format).

        Raises WebSearchUnavailableError if the session isn't open, TimeoutError after WEB_SEARCH_TIMEOUT_SECONDS,
        or the tool's own error. The web agent turns all of them into a "temporarily unavailable" answer.
        """
        if not self.available:
            raise WebSearchUnavailableError("web search is not available")
        content = await asyncio.wait_for(
            self._search_tool.ainvoke({"query": query, "max_results": max_results}), self.timeout_seconds
        )
        return content_text(content)

    async def close(self) -> None:
        """Close the session and stop the server process (safe to call twice, or after a failed start)."""
        self.available = False
        stack, self._stack = self._stack, None
        if stack is not None:
            try:
                await stack.aclose()
            except Exception:  # noqa: BLE001  (shutting down: nothing left to protect)
                logger.warning("Closing the MCP session failed", exc_info=True)

    async def __aenter__(self) -> "WebSearchMCP":
        """`async with WebSearchMCP(settings) as mcp:` starts it (check mcp.available)."""
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the session when leaving the block."""
        await self.close()


async def _main(query: str) -> None:
    """CLI: start the server, show its tools, run one search and print the raw text the agent will parse."""
    started = time.perf_counter()
    async with WebSearchMCP(get_settings()) as mcp:
        print(f"Server tools: {mcp.server_tools}; using only: {ALLOWED_TOOL!r}; available: {mcp.available}")
        print(f"Started in {time.perf_counter() - started:.1f}s")
        if not mcp.available:
            return
        started = time.perf_counter()
        text = await mcp.search(query, get_settings().web_max_results)
        print(f"Search took {time.perf_counter() - started:.1f}s\n\n{text}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s - %(message)s")
    asyncio.run(_main(" ".join(sys.argv[1:]) or "Battle of Gaugamela"))
