"""The agents as tools: what the supervisor LLM sees, and the code that runs the agents behind them (plan 03, 5).

Three parts:

1. Tool SCHEMAS (AskDatabaseArgs, ...): the arguments of each tool, as Pydantic models. They are sent to
   the LLM as JSON schemas (tool_definitions), and the same models validate whatever the LLM sends back
   (guards.check_call, rule S8). One definition, two uses: the LLM can't be told one thing and checked
   against another.
2. SUMMARIES (summarize_for_llm): an agent's full result (all rows, the retrieval trace) is for the UI.
   The supervisor gets a short text (max ~2,000 characters): enough to answer, cheap in tokens, and a
   small surface for prompt injection from documents or web pages.
3. AgentTools: runs the agent behind each tool with a timeout, turns failures into an error text the
   loop can continue with, and records every call as an AgentCall for the UI.

Try a real summary (needs Postgres + Groq):
    uv run python -m app.orchestrator.tools "Which vendors have overdue invoices?"
"""

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field, field_validator

from app.chat.memory import ConversationTurn
from app.orchestrator.models import TOOL_AGENTS, AgentCall, EmailDecision, Limits, PendingEmail

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
CITATION_MARK = re.compile(r"\[(\d+)\]")
TRUNCATED = "… [truncated]"
ROWS_FOR_LLM = 10  # the supervisor sees the first 10 rows; the UI shows them all
MAX_FILES_IN_DESCRIPTION = 10

AGENT_TITLES = {"db": "database agent", "doc": "document agent", "web": "web search agent", "email": "email agent"}
LABEL_PREFIXES = {"doc": "D", "web": "W"}  # citation labels in multi-agent answers: [D1], [W1] (plan D10)


# --------------------------------------------------------------------------- 1. tool schemas


class AskDatabaseArgs(BaseModel):
    """Arguments of ask_database."""

    question: str = Field(min_length=1, max_length=500, description="One self-contained question in plain English.")


class AskDocumentsArgs(BaseModel):
    """Arguments of ask_documents."""

    question: str = Field(min_length=1, max_length=500, description="One self-contained question in plain English.")


class SearchWebArgs(BaseModel):
    """Arguments of search_web."""

    query: str = Field(min_length=1, max_length=300, description="What to search the web for.")


class DraftEmailArgs(BaseModel):
    """Arguments of draft_email. The recipient COUNT limit is a setting, so it's checked in guards (S12)."""

    # No max length in the schema on purpose: Groq validates tool arguments against the schema itself and answers a
    # violation with a 400 the supervisor can't learn from. The count is checked by the guard (S12, a setting) instead,
    # which refuses with a message the LLM can act on ("at most 1 per email").
    to: list[str] = Field(min_length=1, description="The ONE recipient's email address, exactly as the user wrote it.")
    subject: str = Field(min_length=1, max_length=150)
    # Found with gpt-oss-20b on Groq: when it put a markdown TABLE in `content`, it sometimes broke the
    # tool call's JSON (closed it before `subject`/`to`), and Groq rejected the call with a 400. Plain
    # lines are safer, and the email agent writes the greeting and sign-off around them anyway.
    content: str = Field(
        min_length=1,
        max_length=5000,
        description="The facts the email must contain, as plain text: one fact per line or a simple '-' list. "
        "No tables, no greeting or signature (the email agent writes those).",
    )

    @field_validator("to")
    @classmethod
    def _valid_addresses(cls, addresses: list[str]) -> list[str]:
        """Every recipient must look like an email address (S12); surrounding spaces are removed."""
        cleaned = [address.strip() for address in addresses]
        bad = [address for address in cleaned if not EMAIL_PATTERN.fullmatch(address)]
        if bad:
            raise ValueError(f"not a valid email address: {', '.join(bad)}")
        return cleaned


TOOL_ARGS: dict[str, type[BaseModel]] = {
    "ask_database": AskDatabaseArgs,
    "ask_documents": AskDocumentsArgs,
    "search_web": SearchWebArgs,
    "draft_email": DraftEmailArgs,
}

# The descriptions ARE prompts: they are how the LLM decides which tool fits (plan 5.1, kept word for word).
TOOL_DESCRIPTIONS = {
    "ask_database": "Answer a question about the company's business data in PostgreSQL: vendors, invoices, contracts, "
    "tickets, users (employees), audit logs. Read-only. Ask ONE self-contained question.",
    "ask_documents": "Answer a question from the documents uploaded to THIS chat: {filenames}. "
    "Returns an answer with page citations.",
    "search_web": "Search the public internet (DuckDuckGo) for public, current or general-knowledge information. "
    "Never put company data (emails, invoice numbers, amounts, names of employees) in the query.",
    "draft_email": "Draft an email for the user to review. It is NOT sent until the user approves it. Only use when "
    "the user explicitly asks to email/send something, to exactly one recipient the user wrote, and put the actual "
    "information (from earlier tool results) in `content`. Call it alone, after you have gathered the data.",
}


def tool_description(name: str, document_names: list[str]) -> str:
    """The description of one tool; ask_documents lists this chat's files so the LLM knows what's in them."""
    if name != "ask_documents":
        return TOOL_DESCRIPTIONS[name]
    shown = document_names[:MAX_FILES_IN_DESCRIPTION]
    more = len(document_names) - len(shown)
    filenames = ", ".join(shown) + (f" and {more} more" if more > 0 else "")
    return TOOL_DESCRIPTIONS[name].format(filenames=filenames)


def tool_definitions(available: list[str], document_names: list[str]) -> list[dict[str, Any]]:
    """The tools in OpenAI's function format, for `llm.bind_tools(...)`. Only AVAILABLE tools are included (S7).

    convert_to_openai_tool turns the Pydantic schema into JSON schema; we then set the tool's name and
    description ourselves (otherwise it would use the class name and docstring).
    """
    definitions = []
    for name in available:
        definition = convert_to_openai_tool(TOOL_ARGS[name])
        definition["function"]["name"] = name
        definition["function"]["description"] = tool_description(name, document_names)
        definitions.append(definition)
    return definitions


# --------------------------------------------------------------------------- 2. summaries for the LLM


def cap(text: str, max_chars: int) -> str:
    """Cut text to max_chars, marking the cut so the LLM knows something is missing."""
    return text if len(text) <= max_chars else text[: max_chars - len(TRUNCATED)].rstrip() + TRUNCATED


def relabel_citations(text: str, numbers: list[int], prefix: str, start: int) -> tuple[str, dict[str, int]]:
    """Rename an agent's [n] citations to [D1], [D2]... (or W) so two agents' [1]s can't collide (plan D10).

    Labels are given in citation order, starting at `start` (a second doc call in the same question
    continues: D4, D5...). Returns the new text and {label: original n}; the UI uses that map to find
    the source behind a label. A [n] that isn't a real citation is removed.

    Example: ("Credit is 10% [2], see [5].", [2, 5], "D", 1) -> ("Credit is 10% [D1], see [D2].", {"D1": 2, "D2": 5})
    """
    labels = {f"{prefix}{start + index}": n for index, n in enumerate(numbers)}
    label_of = {n: label for label, n in labels.items()}
    renamed = CITATION_MARK.sub(lambda m: f"[{label_of[int(m.group(1))]}]" if int(m.group(1)) in label_of else "", text)
    return renamed, labels


def summarize_for_llm(agent: str, result: Any, label_start: int = 1, max_chars: int = 2000) -> tuple[str, dict]:
    """The short text the supervisor sees for an agent's result, plus the citation labels it used (plan 5.3).

    db:  answer + SQL + the first 10 rows as JSON (the SQL lets the supervisor check what was actually asked)
    doc: answer with [D1] labels + one line per source (file, page, section)
    web: answer with [W1] labels + one line per source (title, URL)
    """
    labels: dict[str, int] = {}
    if agent == "db":
        text = _db_summary(result, max_chars)
    elif agent == "doc":
        text, labels = _doc_summary(result, label_start)
    elif agent == "web":
        text, labels = _web_summary(result, label_start)
    else:
        text = result.answer
    return cap(text, max_chars), labels


def _db_summary(result: Any, max_chars: int) -> str:
    """Answer, SQL and the first rows of a DBAgentResult, fitted into max_chars.

    The rows are the exact data the supervisor may copy (e.g. into an email), so they must never be cut
    in the middle: the answer gets at most half the space, then WHOLE rows are added while they fit
    (up to 10). The final cap() in summarize_for_llm stays as a safety net (e.g. a huge SQL).
    """
    if not result.success:
        return f"The database agent could not answer: {result.answer}"
    head = f"Answer: {cap(result.answer, max_chars // 2)}\nSQL: {result.sql}\n"
    limit = " (row limit reached)" if result.hit_row_limit else ""
    rows: list[dict[str, Any]] = []
    for row in result.rows[:ROWS_FOR_LLM]:
        label = f"Rows: {len(result.rows)}{limit}, first {len(rows) + 1} as JSON: "
        if len(head) + len(label) + len(_json([*rows, row])) > max_chars:
            break
        rows.append(row)
    shown = "as JSON" if len(rows) == len(result.rows) else f"first {len(rows)} as JSON"
    return f"{head}Rows: {len(result.rows)}{limit}, {shown}: {_json(rows)}"


def _json(rows: list[dict[str, Any]]) -> str:
    """Rows as compact JSON; dates and decimals become strings."""
    return json.dumps(rows, default=str, ensure_ascii=False)


def _doc_summary(result: Any, label_start: int) -> tuple[str, dict[str, int]]:
    """Answer with [D1] labels and the matching sources of a DocAgentResult."""
    if not result.citations:
        return f"The document agent found no answer in this chat's documents: {result.answer}", {}
    text, labels = relabel_citations(result.answer, [c.n for c in result.citations], "D", label_start)
    by_n = {c.n: c for c in result.citations}
    sources = "; ".join(f"[{label}] {source_line(by_n[n])}" for label, n in labels.items())
    return f"Answer (with citation labels): {text}\nSources: {sources}", labels


def source_line(citation: Any) -> str:
    """'q3-report.pdf p.4 (4.2 SLA Terms)': where a document citation comes from."""
    section = f" ({citation.heading_path})" if citation.heading_path else ""
    return f"{citation.filename} p.{citation.page_start}{section}"


def _web_summary(result: Any, label_start: int) -> tuple[str, dict[str, int]]:
    """Answer with [W1] labels and the matching pages of a web search result."""
    if not result.success:
        return f"The web search agent could not answer: {result.answer}", {}
    text, labels = relabel_citations(result.answer, [s.n for s in result.sources], "W", label_start)
    by_n = {s.n: s for s in result.sources}
    sources = "; ".join(f"[{label}] {by_n[n].title} {by_n[n].url}" for label, n in labels.items())
    return f"Answer: {text}\nSources: {sources}", labels


def next_labels(calls: list[AgentCall]) -> dict[str, int]:
    """The next free label number per prefix, e.g. {"D": 4, "W": 1} after one doc call that issued D1-D3.

    Uses the HIGHEST number issued so far (not a count), so a gap can never cause a label to be reused.
    """
    used = [label for call in calls for label in call.citation_labels]
    return {
        prefix: 1 + max((int(label[1:]) for label in used if label.startswith(prefix)), default=0)
        for prefix in LABEL_PREFIXES.values()
    }


# --------------------------------------------------------------------------- 3. running the agents


class AskAgent(Protocol):
    """Anything with the agents' common interface: DBAgent, DocAgent, DummyWebSearchAgent."""

    name: str

    async def ask(self, question: str, history: list[ConversationTurn] | None = None, session_id: UUID | None = None):
        """Answer one question; returns a result with .answer and .success."""


class EmailAgent(Protocol):
    """The email agent's two-step interface: draft (ask) now, apply the human's decision (resume) later."""

    async def ask(self, question: str, history=None, session_id=None, *, context: str | None = None):  # noqa: ANN001
        """Draft an email and return it for approval; never sends."""

    async def resume(self, draft_id: UUID, approved: bool, edited_body: str | None = None):
        """Send (approved) or discard (rejected) a draft."""


class DocumentLister(Protocol):
    """The one DocumentStore method AgentTools needs (so tests can pass a tiny fake)."""

    async def list_documents(self, session_id: UUID) -> list:
        """A chat's documents, each with a .filename."""


@dataclass(frozen=True)
class TurnContext:
    """What every tool call of one question shares: the chat, its recent history, the next free citation labels."""

    session_id: UUID | None
    history: list[ConversationTurn] = field(default_factory=list)
    next_labels: dict[str, int] = field(default_factory=lambda: {"D": 1, "W": 1})


@dataclass(frozen=True)
class ToolRequest:
    """One tool call that passed the guards: its id, the tool, and its VALIDATED arguments."""

    call_id: str
    name: str
    args: BaseModel


@dataclass
class ToolOutcome:
    """What running a tool produced: the record for the UI and the text for the supervisor.

    `text` is the ToolMessage content. It is None only while an email draft waits for approval: that
    tool call gets its ToolMessage after the human decides (plan gotcha 2).
    """

    call: AgentCall
    text: str | None
    pending: PendingEmail | None = None


class AgentTools:
    """Runs the agents behind the tools. Built once at startup; the agents are injected (real or dummy).

    When the real web (MCP) and email (A2A) agents arrive, they are passed in here instead of the
    dummies, and only the small wrappers below might change; the graph doesn't.
    """

    def __init__(
        self,
        db_agent: AskAgent,
        doc_agent: AskAgent,
        web_agent: AskAgent,
        email_agent: EmailAgent,
        documents: DocumentLister,
        limits: Limits,
    ) -> None:
        """Store the agents, the document store (to know which chats have files) and the limits."""
        self.agents: dict[str, Any] = {"db": db_agent, "doc": doc_agent, "web": web_agent}
        self.email_agent = email_agent
        self.documents = documents
        self.limits = limits

    async def available(self, session_id: UUID | None) -> tuple[list[str], list[str]]:
        """The tools the supervisor may use in this chat, and the chat's document filenames.

        ask_documents only exists when the chat has at least one document (any status: the Doc agent
        itself says "still processing"). The filenames go into ask_documents' description.
        search_web only exists while the web agent can search (its MCP server is running, plan 04 section 6);
        an agent without an `available` attribute (the dummy) always can.
        """
        documents = await self.documents.list_documents(session_id) if session_id else []
        names = [document.filename for document in documents]
        web = getattr(self.agents["web"], "available", True)
        tools = ["ask_database", *(["ask_documents"] if names else []), *(["search_web"] if web else []), "draft_email"]
        return tools, names

    async def run(self, requests: list[ToolRequest], ctx: TurnContext) -> list[ToolOutcome]:
        """Run the agents behind ask_database / ask_documents / search_web calls and summarize their results.

        Several calls run CONCURRENTLY (asyncio.gather), but their summaries are built in call order, so
        citation labels are handed out deterministically (D1-D3 to the first doc call, D4... to the next).
        """
        executed = await asyncio.gather(*(self._execute(request, ctx) for request in requests))
        labels = dict(ctx.next_labels)
        outcomes = []
        for request, (result, error, duration_ms) in zip(requests, executed, strict=True):
            agent = TOOL_AGENTS[request.name]
            call = AgentCall(
                call_id=request.call_id, tool=request.name, agent=agent, input=request.args.model_dump(),
                status="ok", duration_ms=duration_ms,
            )  # fmt: skip
            if error is not None:
                call.status, call.error = "error", error
                outcomes.append(ToolOutcome(call, f"The {AGENT_TITLES[agent]} failed: {error}"))
                continue
            prefix = LABEL_PREFIXES.get(agent, "")
            text, call.citation_labels = summarize_for_llm(
                agent, result, labels.get(prefix, 1), self.limits.tool_result_chars
            )
            labels[prefix] = labels.get(prefix, 1) + len(call.citation_labels)
            call.result = result.model_dump(mode="json")
            outcomes.append(ToolOutcome(call, text))
        return outcomes

    async def _execute(self, request: ToolRequest, ctx: TurnContext) -> tuple[Any, str | None, int]:
        """Call one agent with the tool timeout (S6). Returns (result, error text or None, duration in ms).

        Any exception is turned into a short error text instead of crashing the question: the loop
        continues, and the supervisor can try another way or explain the failure.
        """
        agent = self.agents[TOOL_AGENTS[request.name]]
        text = getattr(request.args, "question", None) or getattr(request.args, "query", "")
        started = time.perf_counter()
        try:
            coroutine = agent.ask(text, ctx.history, ctx.session_id)
            result = await asyncio.wait_for(coroutine, self.limits.tool_timeout_seconds)
            return result, None, _ms_since(started)
        except TimeoutError:
            return None, self._timeout_text(), _ms_since(started)
        except Exception as error:  # noqa: BLE001  (an agent bug must not end the whole question)
            return None, short_error(error), _ms_since(started)

    def _timeout_text(self) -> str:
        """The error text for an agent that hit the tool timeout (S6)."""
        return f"it took longer than {self.limits.tool_timeout_seconds:g} s and was stopped"

    async def draft_email(self, call_id: str, args: DraftEmailArgs, ctx: TurnContext) -> ToolOutcome:
        """Ask the email agent for a DRAFT. Nothing is sent; the result waits for the human (PendingEmail).

        The dummy agent reads the recipients from the request text, so they always match `args.to`.
        A failure returns an error outcome with a ToolMessage text, like the other tools.
        """
        call = AgentCall(call_id=call_id, tool="draft_email", agent="email", input=args.model_dump(), status="error")
        request = f"Email to {', '.join(args.to)}: {args.subject}"
        started = time.perf_counter()
        try:
            coroutine = self.email_agent.ask(request, ctx.history, ctx.session_id, context=args.content)
            result = await asyncio.wait_for(coroutine, self.limits.tool_timeout_seconds)
        except TimeoutError:
            result, call.error = None, self._timeout_text()
        except Exception as error:  # noqa: BLE001
            result, call.error = None, short_error(error)
        call.duration_ms = _ms_since(started)
        if result is not None and (result.status != "input_required" or result.draft is None):
            call.error = result.error or "the email agent returned no draft"
        if call.error:
            return ToolOutcome(call, f"The email agent failed: {call.error}")

        call.status, call.result = "awaiting_approval", result.model_dump(mode="json")
        pending = PendingEmail(tool_call_id=call_id, call=call, draft=result.draft.model_dump(mode="json"))
        return ToolOutcome(call, None, pending)

    async def resolve_email(self, pending: PendingEmail, decision: EmailDecision) -> ToolOutcome:
        """Apply the human's decision to the draft, and say what happened in the draft_email ToolMessage.

        approved + sent -> status "sent"; rejected -> "rejected" (and the supervisor is told not to try
        again); anything else (draft gone after a restart, a failure) -> "error", nothing was sent.
        """
        call = pending.call.model_copy(deep=True)
        started = time.perf_counter()
        try:
            draft_id = UUID(pending.draft["draft_id"])
            coroutine = self.email_agent.resume(draft_id, decision.approved, decision.edited_body)
            result = await asyncio.wait_for(coroutine, self.limits.tool_timeout_seconds)
        except TimeoutError:
            result, error = None, self._timeout_text()
        except Exception as exc:  # noqa: BLE001
            result, error = None, short_error(exc)
        call.duration_ms += _ms_since(started)

        if result is not None:
            call.result = result.model_dump(mode="json")
            if result.status == "completed":
                call.status = "sent"
                return ToolOutcome(call, f"Approved by the user and sent to {', '.join(result.draft.to)}.")
            if result.status == "rejected" and result.success:
                call.status = "rejected"
                rejected = "The user REJECTED the draft. Nothing was sent. Do not draft another email. Tell the user."
                return ToolOutcome(call, rejected)
            error = result.error or result.answer
        call.status, call.error = "error", error
        # Found in a real run: "The email could not be sent: ..." alone was ignored once, and the supervisor
        # offered to send the draft. Said as plainly as the rejection, with what to tell the user.
        return ToolOutcome(
            call, f"NOT SENT: the email could not be sent ({error}). Tell the user it was not sent; they can ask again."
        )


def short_error(error: BaseException) -> str:
    """A one-line error for the supervisor and the UI: the exception type and the start of its message."""
    message = " ".join(str(error).split())[:200]
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _ms_since(started: float) -> int:
    """Milliseconds since a time.perf_counter() reading."""
    return int((time.perf_counter() - started) * 1000)


async def _main(question: str) -> None:
    """CLI: ask the real DB agent, then print exactly what the supervisor would see for that result."""
    from app.agents.db.database import ReadOnlyDatabase
    from app.agents.db.graph import DBAgent
    from app.agents.db.schema import load_schema
    from app.config import get_settings

    async with ReadOnlyDatabase(get_settings().readonly_database_url) as db:
        result = await DBAgent(db, await load_schema(db)).ask(question)
    text, _ = summarize_for_llm("db", result, max_chars=Limits.from_settings(get_settings()).tool_result_chars)
    print(f"--- the DB agent returned {len(result.rows)} rows; the supervisor sees {len(text)} characters:\n")
    print(text)


if __name__ == "__main__":
    asyncio.run(_main(" ".join(sys.argv[1:]) or "Which vendors have overdue invoices?"))
