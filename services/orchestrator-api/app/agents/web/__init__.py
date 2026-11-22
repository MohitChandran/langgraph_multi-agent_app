"""Web search agent (project plan Phase 4; plannings/04 Part B): live web data through the DuckDuckGo MCP server.

Modules:
    mcp          WebSearchMCP: one long-lived MCP session to the DuckDuckGo server (its own process, via uvx);
                 only its `search` tool is used
    query_guard  rejects queries that carry company data (emails, ids, phone/account numbers) before any search
    parsing      the server's text output -> numbered WebSources; its failure texts -> honest answers
    prompts      the cited-answer prompt (results are untrusted data)
    graph        WebSearchAgent: the fixed LangGraph workflow check_query -> search -> parse -> answer, and a CLI
    models       WebSource, WebSearchResult (shared with the dummy), WebAnswer, WebAgentState
    dummy        DummyWebSearchAgent: hardcoded results, no network. Kept for tests (plan 03's orchestrator tests).
"""
