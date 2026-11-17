"""Data shapes for the orchestrator (plan 03, section 7).

Two kinds of objects live here:

- Pydantic models (Budget, AgentCall, PendingEmail, EmailDecision, OrchestratorResult): validated,
  easy to test, and what the API returns.
- OrchestratorState: the LangGraph state. It holds ONLY plain JSON (dicts, lists, strings) and
  LangChain messages. Why: when a question pauses for email approval, the state is saved to Postgres
  (a "checkpoint") and loaded back on resume. The checkpoint loader only trusts an allow-list of
  classes; our own Pydantic classes would be warned about or blocked, and resuming would break.
  So graph nodes convert on the way in (`Budget.model_validate(state["budget"])`) and back on the
  way out (`budget.model_dump()`).
"""

import operator
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.config import Settings

# tool name -> the agent behind it (AgentCall.agent, the UI's chip label)
TOOL_AGENTS = {"ask_database": "db", "ask_documents": "doc", "search_web": "web", "draft_email": "email"}

CallStatus = Literal["ok", "error", "refused", "awaiting_approval", "sent", "rejected"]
StopReason = Literal["answered", "tool_budget", "supervisor_limit", "llm_error_fallback", "llm_error_forced"]


@dataclass(frozen=True)
class Limits:
    """The per-question limits from plan section 6, read once from the settings.

    A separate small object (instead of passing Settings everywhere) keeps the guards pure and
    lets tests use tiny limits, e.g. `Limits(max_tool_calls=1)`.
    """

    max_tool_calls: int = 4  # S1: tool calls executed per question
    max_calls_per_tool: int = 2  # S2: per ask_*/search_web tool
    max_email_drafts: int = 1  # S2: draft_email (fixed by the user's decision: one email per question)
    max_supervisor_calls: int = 6  # S3
    max_recipients: int = 5  # S12
    tool_result_chars: int = 2000  # cap on what a tool result sends back to the supervisor
    tool_timeout_seconds: float = 60.0  # S6
    turn_timeout_seconds: float = 120.0  # S5

    @classmethod
    def from_settings(cls, settings: Settings) -> "Limits":
        """Build the limits from the ORCH_* settings (.env)."""
        return cls(
            max_tool_calls=settings.orch_max_tool_calls,
            max_calls_per_tool=settings.orch_max_calls_per_tool,
            max_supervisor_calls=settings.orch_max_supervisor_calls,
            max_recipients=settings.orch_max_email_recipients,
            tool_result_chars=settings.orch_tool_result_chars,
            tool_timeout_seconds=settings.orch_tool_timeout_seconds,
            turn_timeout_seconds=settings.orch_turn_timeout_seconds,
        )


class Budget(BaseModel):
    """What one question has used so far. The guards compare it with the Limits."""

    supervisor_calls: int = 0  # LLM calls made by the supervisor (S3)
    tool_calls: int = 0  # tool calls EXECUTED, refused ones don't count (S1)
    per_tool: dict[str, int] = Field(default_factory=dict)  # executed calls per tool name (S2)
    llm_errors_in_a_row: int = 0  # consecutive supervisor LLM failures (S19)
    seen_calls: list[str] = Field(default_factory=list)  # normalized "tool|args" keys, for duplicates (S9)


class AgentCall(BaseModel):
    """One tool call, as the UI shows it in the "calls" panel: what was asked, what came back."""

    call_id: str  # the id the LLM gave the tool call; its ToolMessage answers this id
    tool: str  # ask_database | ask_documents | search_web | draft_email (or whatever the LLM invented)
    agent: str  # db | doc | web | email ("" for an unknown tool)
    input: dict[str, Any]  # the arguments (validated when the call ran; raw when it was refused)
    status: CallStatus
    result: dict[str, Any] | None = None  # the agent's full result, e.g. DBAgentResult with all rows
    citation_labels: dict[str, int] = Field(default_factory=dict)  # "D1" -> the agent's citation n (plan D10)
    error: str | None = None  # why it failed or was refused
    duration_ms: int = 0


class PendingEmail(BaseModel):
    """An email draft waiting for the human. Its tool call gets its ToolMessage only after the decision."""

    tool_call_id: str
    call: AgentCall  # status "awaiting_approval"; result = the email agent's draft result
    draft: dict[str, Any]  # draft_id, to, subject, body: exactly what the approval card shows


class EmailDecision(BaseModel):
    """The human's decision on a draft: the value the paused graph is resumed with (S14)."""

    approved: bool
    edited_body: str | None = Field(default=None, max_length=5000)


class OrchestratorState(TypedDict, total=False):
    """What the orchestrator graph knows at each step (plain JSON + LangChain messages only, see above)."""

    messages: Annotated[list[AnyMessage], add_messages]  # THIS question only: question, AI replies, tool results
    question: str
    session_id: str
    history: list[dict[str, Any]]  # earlier exchanges (ConversationTurn as dicts), passed on to the agents
    history_text: str  # the same, formatted for the supervisor's system prompt
    user_emails: list[str]  # addresses the user typed: the only allowed recipients (S11)
    document_names: list[str]  # this chat's files, listed in the ask_documents tool description
    available_tools: list[str]  # tools the supervisor may call in this chat (S7)
    budget: dict[str, Any]  # Budget.model_dump()
    calls: list[dict[str, Any]]  # [AgentCall.model_dump(mode="json")], replaced whole by each node
    pending_email: dict[str, Any] | None  # PendingEmail.model_dump(mode="json")
    email_decision: dict[str, Any] | None  # EmailDecision.model_dump()
    final_answer: str
    passthrough: bool  # final_answer is the single agent's own answer, unchanged (D9)
    stop_reason: StopReason
    # node names in the order they ran (UI flow, tests). operator.add = each node returns only ITS name
    # (["tools"]) and LangGraph appends it, instead of every node copying the whole list.
    steps: Annotated[list[str], operator.add]


class OrchestratorResult(BaseModel):
    """The orchestrator's output; saved as chat_messages.payload.result."""

    question: str
    answer: str
    status: Literal["completed", "awaiting_approval", "failed"]
    success: bool
    calls: list[AgentCall] = Field(default_factory=list)
    email: dict[str, Any] | None = None  # the draft (while awaiting) or the final email outcome
    stop_reason: str | None = None
    steps: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)  # built by code from the calls, e.g. "Asked the database: ..."
    budget: dict[str, int] = Field(default_factory=dict)  # {"tool_calls": 2, "supervisor_calls": 3}
    # True when `answer` is the single agent's own answer, unchanged (plan D9): its citations are then
    # that agent's own [1], [2]... instead of [D1]/[W1] labels.
    passthrough: bool = False
    error: str | None = None
