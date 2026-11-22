"""Tests for the orchestrator's tools (app/orchestrator/tools.py): schemas, summaries, relabelling, AgentTools.

What the supervisor LLM sees must be short, contain what it needs (the SQL, the sources) and never let
two agents' citations collide. AgentTools must turn every agent failure into a text the loop can
continue with. Fake agents and the dummy email agent: no LLM, no database.
"""

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

from app.agents.db.models import DBAgentResult
from app.agents.doc.models import Citation, DocAgentResult
from app.agents.email.dummy import DummyEmailAgent
from app.agents.web.dummy import DummyWebSearchAgent
from app.orchestrator.models import AgentCall, EmailDecision, Limits
from app.orchestrator.tools import (
    AgentTools,
    AskDatabaseArgs,
    AskDocumentsArgs,
    DraftEmailArgs,
    SearchWebArgs,
    ToolRequest,
    TurnContext,
    next_labels,
    relabel_citations,
    summarize_for_llm,
    tool_definitions,
)

SESSION = uuid4()


def db_result(rows: int = 3, success: bool = True) -> DBAgentResult:
    """A DB agent result with `rows` invoice rows."""
    return DBAgentResult(
        question="Which invoices are overdue?", answer="There are overdue invoices.", success=success,
        sql="SELECT v.name, i.amount FROM invoices i JOIN vendors v ON v.id = i.vendor_id LIMIT 100",
        columns=["name", "amount"], rows=[{"name": f"Vendor {n}", "amount": 100 + n} for n in range(rows)],
    )  # fmt: skip


def citation(n: int, page: int, heading: str = "4.2 SLA Terms") -> Citation:
    """A citation of q3-report.pdf."""
    return Citation(
        n=n, chunk_id=n, document_id=uuid4(), filename="q3-report.pdf", page_start=page, page_end=page,
        heading_path=heading, chunk_type="text", snippet="...",
    )  # fmt: skip


def doc_result(answer: str = "The credit is 10% [2], rising to 25% [5].", numbers: tuple = (2, 5)) -> DocAgentResult:
    """A grounded Doc agent result citing sources 2 and 5 (the Doc agent keeps its source numbers)."""
    return DocAgentResult(
        question="q", answer=answer, success=True, grounded=True, standalone_query="q",
        citations=[citation(n, page=n + 2) for n in numbers],
    )  # fmt: skip


class FakeAgent:
    """An agent that returns a fixed result, sleeps, or raises; records what it was asked."""

    def __init__(self, result=None, delay: float = 0.0, error: Exception | None = None) -> None:  # noqa: ANN001
        """Set the behaviour: return `result` after `delay` seconds, or raise `error`."""
        self.result, self.delay, self.error = result, delay, error
        self.asked: list[tuple] = []

    async def ask(self, question, history=None, session_id=None):  # noqa: ANN001, ANN201
        """Record the call, then sleep / raise / return as configured."""
        self.asked.append((question, history, session_id))
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


class FakeDocuments:
    """A DocumentStore stand-in: every chat has the given filenames."""

    def __init__(self, filenames: list[str]) -> None:
        """Store the filenames list_documents will return."""
        self.filenames = filenames

    async def list_documents(self, session_id):  # noqa: ANN001, ANN201
        """Documents with just a .filename."""
        return [SimpleNamespace(filename=name) for name in self.filenames]


def make_tools(db=None, doc=None, web=None, filenames=(), limits=None):  # noqa: ANN001, ANN201
    """AgentTools wired to fakes (defaults: a DB agent returning db_result(), the dummy web and email agents)."""
    return AgentTools(
        db or FakeAgent(db_result()), doc or FakeAgent(doc_result()), web or DummyWebSearchAgent(), DummyEmailAgent(),
        FakeDocuments(list(filenames)), limits or Limits(),
    )  # fmt: skip


# --------------------------------------------------------------------------- schemas and definitions


def test_tool_definitions_include_only_available_tools_and_list_the_files() -> None:
    """The LLM sees the available tools only, with our names; ask_documents names the chat's files."""
    definitions = tool_definitions(["ask_database", "ask_documents"], ["q3-report.pdf", "policy.pdf"])
    functions = [d["function"] for d in definitions]
    assert [f["name"] for f in functions] == ["ask_database", "ask_documents"]
    assert "THIS chat: q3-report.pdf, policy.pdf." in functions[1]["description"]
    assert functions[0]["parameters"]["required"] == ["question"]


# --------------------------------------------------------------------------- summaries


def test_db_summary_is_capped_and_contains_the_sql() -> None:
    """100 rows: the supervisor gets the SQL, the row count and only the first 10 rows, within 2,000 chars."""
    text, labels = summarize_for_llm("db", db_result(rows=100))
    assert len(text) <= 2000 and labels == {}
    assert "SQL: SELECT v.name, i.amount" in text and "Rows: 100, first 10 as JSON" in text
    assert "Vendor 9" in text and "Vendor 10" not in text


def test_a_long_answer_never_cuts_the_rows_in_half() -> None:
    """A long answer gets at most half the space; the rows that still fit are WHOLE rows, valid JSON."""
    result = db_result(rows=10)
    result.answer = "x" * 5000
    text, _ = summarize_for_llm("db", result, max_chars=600)

    assert len(text) <= 600 and "x… [truncated]\nSQL:" in text
    rows = json.loads(text.split(" as JSON: ", 1)[1])
    assert 0 < len(rows) < 10 and rows == result.rows[: len(rows)]
    assert f"Rows: 10, first {len(rows)} as JSON" in text


def test_anything_too_long_is_still_cut_with_a_marker() -> None:
    """The final cap is a safety net: even a huge SQL can't push the text over the limit."""
    result = db_result()
    result.sql = "SELECT " + "x, " * 2000
    text, _ = summarize_for_llm("db", result, max_chars=300)
    assert len(text) == 300 and text.endswith("… [truncated]")


def test_failed_db_answer_says_so() -> None:
    """success=False is reported as "could not answer", without SQL or rows."""
    text, _ = summarize_for_llm("db", db_result(success=False))
    assert text == "The database agent could not answer: There are overdue invoices."


def test_doc_summary_relabels_citations_and_lists_sources() -> None:
    """[2] and [5] become [D1] and [D2]; the labels map back to the original numbers for the UI."""
    text, labels = summarize_for_llm("doc", doc_result())
    assert labels == {"D1": 2, "D2": 5}
    assert text.startswith("Answer (with citation labels): The credit is 10% [D1], rising to 25% [D2].")
    assert "Sources: [D1] q3-report.pdf p.4 (4.2 SLA Terms); [D2] q3-report.pdf p.7 (4.2 SLA Terms)" in text


async def test_doc_numbering_continues_across_calls_and_web_uses_w() -> None:
    """A second doc call starts at D3; web results get W labels; the counters come from earlier calls."""
    _, labels = summarize_for_llm("doc", doc_result(), label_start=3)
    assert labels == {"D3": 2, "D4": 5}
    calls = [AgentCall(call_id="1", tool="ask_documents", agent="doc", input={}, status="ok", citation_labels=labels)]
    assert next_labels(calls) == {"D": 5, "W": 1}

    web = await DummyWebSearchAgent().ask("Did Example Cloud change its SLA?")
    text, web_labels = summarize_for_llm("web", web)
    assert web_labels == {"W1": 1, "W2": 2} and "[W1]" in text and "https://example.com/example-cloud/sla" in text


def test_relabel_drops_marks_that_are_not_citations() -> None:
    """A [9] that isn't among the citations is removed rather than left dangling."""
    assert relabel_citations("A [1] B [9]", [1], "D", 1) == ("A [D1] B ", {"D1": 1})


def test_ungrounded_doc_answer_says_nothing_was_found() -> None:
    """No citations: the supervisor is told the documents didn't have it."""
    result = doc_result(answer="I couldn't find that in the documents.", numbers=())
    text, labels = summarize_for_llm("doc", result)
    assert labels == {} and text.startswith("The document agent found no answer")


# --------------------------------------------------------------------------- AgentTools


async def test_available_tools_depend_on_the_chats_documents() -> None:
    """ask_documents is offered only when the chat has documents, and the filenames come back too."""
    assert await make_tools().available(SESSION) == (["ask_database", "search_web", "draft_email"], [])
    tools, names = await make_tools(filenames=["q3-report.pdf"]).available(SESSION)
    assert "ask_documents" in tools and names == ["q3-report.pdf"]


async def test_search_web_is_offered_only_while_the_web_agent_can_search() -> None:
    """The real web agent reports `available` (its MCP server is running); when it's down, search_web disappears."""

    class OfflineWeb(FakeAgent):
        """A web agent whose MCP server isn't running."""

        available = False

    tools, _ = await make_tools(web=OfflineWeb()).available(SESSION)
    assert tools == ["ask_database", "draft_email"]


async def test_run_records_the_call_and_passes_history_and_session() -> None:
    """A successful call: status ok, the full result kept for the UI, the agent got history + session id."""
    db = FakeAgent(db_result())
    outcome = (await make_tools(db=db).run([ToolRequest("c1", "ask_database", AskDatabaseArgs(question="q?"))],
                                           TurnContext(SESSION, history=["turn"])))[0]  # fmt: skip
    assert outcome.call.status == "ok" and outcome.call.agent == "db" and outcome.call.result["rows"]
    assert outcome.text.startswith("Answer: There are overdue invoices.")
    assert db.asked == [("q?", ["turn"], SESSION)]


async def test_two_doc_calls_in_one_message_get_different_labels() -> None:
    """Concurrent calls are summarized in order, so their citation labels never collide."""
    requests = [ToolRequest(f"c{n}", "ask_documents", AskDocumentsArgs(question=f"q{n}")) for n in (1, 2)]
    outcomes = await make_tools().run(requests, TurnContext(SESSION))
    assert [o.call.citation_labels for o in outcomes] == [{"D1": 2, "D2": 5}, {"D3": 2, "D4": 5}]


async def test_timeouts_and_crashes_become_error_texts() -> None:
    """S6: a slow agent is stopped and a crashing one is caught; both give a text and the loop goes on."""
    tools = make_tools(
        db=FakeAgent(db_result(), delay=1.0), web=FakeAgent(error=RuntimeError("search engine down")),
        limits=Limits(tool_timeout_seconds=0.05),
    )  # fmt: skip
    requests = [
        ToolRequest("c1", "ask_database", AskDatabaseArgs(question="q")),
        ToolRequest("c2", "search_web", SearchWebArgs(query="q")),
    ]
    slow, broken = await tools.run(requests, TurnContext(SESSION))
    assert (
        slow.call.status == "error"
        and slow.text == "The database agent failed: it took longer than 0.05 s and was stopped"
    )
    assert broken.text == "The web search agent failed: RuntimeError: search engine down"


async def test_email_draft_waits_then_approve_sends_and_reject_does_not() -> None:
    """draft_email returns a pending draft (no ToolMessage yet); the decision then gives sent / rejected."""
    tools = make_tools()
    args = DraftEmailArgs(to=["accounts@company.example"], subject="Invoices", content="3 invoices are due.")

    drafted = await tools.draft_email("e1", args, TurnContext(SESSION))
    assert drafted.text is None and drafted.call.status == "awaiting_approval"
    assert drafted.pending.draft["to"] == ["accounts@company.example"]
    assert "3 invoices are due." in drafted.pending.draft["body"]

    sent = await tools.resolve_email(drafted.pending, EmailDecision(approved=True, edited_body="Short."))
    assert sent.call.status == "sent" and sent.text == "Approved by the user and sent to accounts@company.example."
    assert sent.call.result["draft"]["body"] == "Short."

    second = await tools.draft_email("e2", args, TurnContext(SESSION))
    rejected = await tools.resolve_email(second.pending, EmailDecision(approved=False))
    assert rejected.call.status == "rejected" and "Do not draft another email" in rejected.text


async def test_deciding_a_draft_that_no_longer_exists_is_an_error() -> None:
    """After a restart the dummy has forgotten the draft: status error, and the text says nothing was sent."""
    tools = make_tools()
    drafted = await tools.draft_email(
        "e1", DraftEmailArgs(to=["a@b.example"], subject="s", content="c"), TurnContext(SESSION)
    )
    tools.email_agent = DummyEmailAgent()  # a fresh agent = the API restarted

    outcome = await tools.resolve_email(drafted.pending, EmailDecision(approved=True))
    assert outcome.call.status == "error" and outcome.text.startswith("NOT SENT: the email could not be sent (Unknown")
