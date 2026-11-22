"""The orchestrator: a ReAct supervisor loop over the four agents, as our own LangGraph graph (plan 03, 3 and 9).

ReAct = Reason + Act. An LLM (the "supervisor") sees the agents as tools. It decides which one to call,
OUR code checks and runs the call, the result goes back to the LLM, and it decides again, until it
answers with text:

    START -> prepare -> supervisor --tool calls--> tools --email drafted--> approval (PAUSE) -> send_email
                          ^  |  ^                   |                                            |
                          |  |  '---budget left-----'                                            |
                          |  |  '------------------------------------------------------------------'
                          |  |--no tool calls (the answer) ---------------------------------> finalize -> END
                          |  |--limits reached, or the LLM fails after some calls -> force_answer -> finalize
                          '--'--the LLM fails before any call ------------------------> fallback -> finalize
                    (retry once)

Who decides what:
  - the LLM decides WHICH tool to call with WHAT input, and writes the final answer;
  - the code decides WHETHER a call may run (guards.py), WHEN the loop stops (the routers below), and
    that an email is never sent without a human (the only path to send_email goes through approval).

Pausing for the human: the approval node calls interrupt(). LangGraph saves the state (a "checkpoint")
and ask() returns status "awaiting_approval". Later, resume_email() continues the SAME run from that
checkpoint with the human's decision. LangGraph re-runs the paused node from its first line on resume,
so approval does nothing but interrupt(): drafting happens before it (tools), sending after it (send_email).

Try it (real Groq and the real agents: DB, Doc with --session, web search over MCP, the email service when running):
    uv run python -m app.orchestrator.graph "hi, what can you do?"
    uv run python -m app.orchestrator.graph "Which vendors have overdue invoices?"
    uv run python -m app.orchestrator.graph "Which invoices are pending? Email them to accounts@company.example"
    uv run python -m app.orchestrator.graph --postgres "..."   # pause in Postgres (app.checkpoints) instead of memory
    uv run python -m app.orchestrator.graph --session <chat id> "..."   # a chat's documents and memory
    uv run python -m app.orchestrator.graph --diagram
"""

import argparse
import asyncio
import logging
import re
import time
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import asdict
from typing import Any, Literal
from uuid import UUID, uuid4

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.chat.memory import ConversationTurn, format_history
from app.config import get_settings
from app.llm import LLM_ERRORS, get_llm
from app.observability import get_tracing
from app.orchestrator.guards import call_key, check_call, count_call, extract_user_emails
from app.orchestrator.models import (
    TOOL_AGENTS,
    AgentCall,
    Budget,
    EmailDecision,
    OrchestratorResult,
    OrchestratorState,
    PendingEmail,
)
from app.orchestrator.prompts import FORCE_ANSWER_INSTRUCTION, supervisor_prompt
from app.orchestrator.tools import (
    AGENT_TITLES,
    CITATION_MARK,
    TOOL_ARGS,
    AgentTools,
    ToolRequest,
    TurnContext,
    next_labels,
    short_error,
    tool_definitions,
)

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 30  # S4: LangGraph stops a run after this many steps, whatever the routers say
MAX_LLM_ERRORS = 2  # S19: two supervisor failures in a row -> stop asking the LLM
ASK_TOOLS = ("ask_database", "ask_documents", "search_web")  # the tools that answer questions (D9)
# [D1], [W2]: the citation labels of multi-agent answers (D10). The optional leading space is matched so a
# dropped label doesn't leave "changed [W1] ." behind.
LABEL_MARK = re.compile(r"\s?\[([DW]\d+)\]")
# gpt-oss sometimes writes its own citation style, 【W1】 or 【W1†L3】 (found in a real run): turned into [W1].
FULLWIDTH_LABEL = re.compile(r"【\s*([DW]\d+)[^】]*】")

FAILED_TEXT = "Sorry, something went wrong while answering. Please try again."
TIMEOUT_TEXT = "This took too long and was stopped."
TOO_MANY_STEPS_TEXT = "The request took too many steps."
NOT_WAITING_TEXT = "This email is no longer waiting for approval."
INVALID_JSON = "Invalid arguments for {name}: they are not valid JSON. Fix them and call again."

STOP_TEXTS = {  # one reasoning line when the loop didn't end with the supervisor's own answer
    "tool_budget": "Stopped: the tool budget for this question was used up, so the answer uses the results so far.",
    "supervisor_limit": "Stopped: the step limit for this question was reached, so the answer uses the results so far.",
    "llm_error_fallback": "The orchestrator's LLM was unavailable, so the fixed rule picked the agent "
    "(a chat with documents -> the Document agent, otherwise the Database agent).",
    "llm_error_forced": "The orchestrator's LLM failed mid-way, so the answer was written from the results so far.",
}

UpdateCallback = Callable[[dict[str, Any]], None]  # receives {node name: what the node changed}


class Orchestrator:
    """Answers any question by calling the DB, Doc, Web and Email agents as tools, within safe limits.

    Usage:
        orchestrator = Orchestrator(tools, checkpointer)
        result = await orchestrator.ask("Which vendors have overdue invoices?", history, session_id, message_id)
        if result.status == "awaiting_approval":            # an email draft waits for the human
            result = await orchestrator.resume_email(session_id, message_id, EmailDecision(approved=True))

    Dependencies are injected (like DBAgent / DocAgent), so tests pass a scripted fake LLM, fake
    agents and an in-memory checkpointer: no Groq, no database.
    """

    name = "orchestrator"  # saved in chat_messages.payload.agent

    def __init__(
        self,
        tools: AgentTools,
        checkpointer: BaseCheckpointSaver | None = None,
        chat_model: Any = None,
    ) -> None:
        """Store the dependencies and compile the graph once.

        Args:
            tools: runs the agents behind the tools; also carries the limits (tools.limits).
            checkpointer: where a paused run is saved. Default: in memory (CLI, tests). The API passes
                the Postgres saver (checkpoint.py), so a pause survives a restart.
            chat_model: the supervisor LLM; it must support bind_tools(). Default: the "orchestrator"
                Groq model, with its output capped at ORCH_SUPERVISOR_MAX_TOKENS (S20).
        """
        self.tools = tools
        self.limits = tools.limits
        self.checkpointer = checkpointer or InMemorySaver()
        self.chat_model = chat_model or get_llm("orchestrator", max_tokens=get_settings().orch_supervisor_max_tokens)
        self.graph = self._build_graph()

    def _build_graph(self):  # noqa: ANN202  (LangGraph's compiled type is long and adds nothing here)
        """Wire the nodes and routers together and compile them with the checkpointer."""
        builder = StateGraph(OrchestratorState)
        builder.add_node("prepare", self.prepare)
        builder.add_node("supervisor", self.supervisor)
        builder.add_node("tools", self.run_tools)
        builder.add_node("approval", self.approval)
        builder.add_node("send_email", self.send_email)
        builder.add_node("force_answer", self.force_answer)
        builder.add_node("fallback", self.fallback)
        builder.add_node("finalize", self.finalize)

        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "supervisor")
        builder.add_conditional_edges("supervisor", self.route_after_supervisor)
        builder.add_conditional_edges("tools", self.route_after_tools)
        builder.add_edge("approval", "send_email")  # S13: the only way to send_email is through the human
        builder.add_conditional_edges("send_email", self.route_after_send_email)
        builder.add_edge("force_answer", "finalize")
        builder.add_edge("fallback", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=self.checkpointer, name="orchestrator")

    # ------------------------------------------------------------------ public API

    async def ask(
        self,
        question: str,
        history: list[ConversationTurn] | None = None,
        session_id: UUID | None = None,
        message_id: int | str | None = None,
        on_update: UpdateCallback | None = None,
    ) -> OrchestratorResult:
        """Run the loop for one question. Returns "completed", "failed", or "awaiting_approval" (paused).

        Args:
            history: earlier exchanges of the chat (memory). They go into the supervisor's system
                prompt and are passed on to the agents; they're NOT added to the message list (D8).
            session_id: the chat; decides which documents ask_documents can search.
            message_id: the pending answer's id. With session_id it names the checkpoint thread
                "<session_id>:<message_id>", which resume_email() must use again (one thread per turn, D7).
            on_update: called with each node's changes as it finishes (the CLI prints them).
        """
        initial: OrchestratorState = {
            "question": question,
            "session_id": str(session_id) if session_id else "",
            "history": [asdict(turn) for turn in history or []],
        }
        thread_id = thread_id_for(session_id, message_id if message_id is not None else uuid4().hex)
        return await self._run(initial, thread_id, question, on_update)

    async def resume_email(
        self,
        session_id: UUID | None,
        message_id: int | str,
        decision: EmailDecision,
        on_update: UpdateCallback | None = None,
    ) -> OrchestratorResult:
        """Continue a paused turn with the human's decision on its email draft (approve, edit, reject).

        The thread must be paused at the approval node; otherwise (already decided, deleted, never
        paused) the result is "failed" with NOT_WAITING_TEXT, and nothing is sent.
        """
        thread_id = thread_id_for(session_id, message_id)
        snapshot = await self.graph.aget_state(run_config(thread_id))
        if "approval" not in snapshot.next:
            return OrchestratorResult(
                question=snapshot.values.get("question", ""), answer=NOT_WAITING_TEXT, status="failed",
                success=False, error=NOT_WAITING_TEXT,
            )  # fmt: skip
        command = Command(resume=decision.model_dump())
        return await self._run(command, thread_id, snapshot.values["question"], on_update)

    async def _run(
        self, graph_input: Any, thread_id: str, question: str, on_update: UpdateCallback | None
    ) -> OrchestratorResult:
        """Run (or resume) the graph within the turn timeout and turn its final state into a result.

        astream with two modes: "updates" = {node: changes} after each node (for on_update and to
        notice the pause), "values" = the whole state after each step (the last one is the final state).
        durability="exit": the checkpoint is written only when the run stops or pauses, not after every
        node; only a paused turn needs to be saved (D7).
        A paused run keeps its thread; a finished or failed one deletes it.
        """
        # Langfuse (no-op when off): the handler records every node and LLM call of this run, nested under the
        # current trace; run_name is what the graph's root run is called there.
        config = {**run_config(thread_id), "callbacks": get_tracing().callbacks(), "run_name": "orchestrator"}
        state: dict[str, Any] = {}
        paused = False

        async def run() -> None:
            """Consume the stream, remembering the latest state and whether the graph paused."""
            nonlocal state, paused
            stream = self.graph.astream(graph_input, config, stream_mode=["updates", "values"], durability="exit")
            async for mode, chunk in stream:
                if mode == "values":
                    state = chunk
                elif "__interrupt__" in chunk:
                    paused = True
                elif on_update:
                    on_update(chunk)

        error: str | None = None
        try:
            await asyncio.wait_for(run(), self.limits.turn_timeout_seconds)  # S5
        except TimeoutError:
            error = TIMEOUT_TEXT
        except GraphRecursionError:  # S4
            error = TOO_MANY_STEPS_TEXT
        except Exception:  # noqa: BLE001  (any bug becomes a failed answer, not a 500; the details go to the log)
            logger.exception("Orchestrator failed (thread %s)", thread_id)
            error = FAILED_TEXT

        if paused and error is None:
            return build_result(state, question, "awaiting_approval")
        await self._delete_thread(thread_id)
        if error is not None:
            return build_result(state, question, "failed", error=error)
        return build_result(state, question, "completed")

    async def _delete_thread(self, thread_id: str) -> None:
        """Delete a finished turn's checkpoint. A failure is only logged: the startup cleanup removes orphans."""
        try:
            await self.checkpointer.adelete_thread(thread_id)
        except Exception:  # noqa: BLE001
            logger.warning("Could not delete checkpoint thread %s", thread_id, exc_info=True)

    # ------------------------------------------------------------------ nodes

    async def prepare(self, state: OrchestratorState) -> OrchestratorState:
        """Set up the turn: which tools this chat offers, allowed email recipients, memory text, empty budget.

        user_emails (S11) come ONLY from the user's own words: this question and the earlier
        questions of the chat. Never from answers, tool results or documents.
        """
        history = state.get("history", [])
        available, document_names = await self.tools.available(as_uuid(state.get("session_id")))
        return {
            "available_tools": available,
            "document_names": document_names,
            "user_emails": extract_user_emails([state["question"], *(turn["question"] for turn in history)]),
            "history_text": format_history([ConversationTurn(**turn) for turn in history]),
            "budget": Budget().model_dump(),
            "calls": [],
            "pending_email": None,
            "email_decision": None,
            "messages": [HumanMessage(state["question"])],
            "steps": ["prepare"],
        }

    async def supervisor(self, state: OrchestratorState) -> OrchestratorState:
        """One LLM call: the supervisor reads everything so far and either calls tools or answers.

        Only the AVAILABLE tools are bound (S7). With no tool calls left (e.g. the email was the last
        one), no tools are bound at all, so the LLM can only answer. Never tool_choice="none": Groq
        rejects that with a 400 (plan gotcha 1).
        A failed or empty reply is not added to the messages; it only counts as an error (S19), and
        the router decides whether to retry.
        """
        budget = Budget.model_validate(state["budget"])
        budget.supervisor_calls += 1
        tool_calls_left = self.limits.max_tool_calls - budget.tool_calls
        system = self._system_message(state, tool_calls_left)
        llm = self.chat_model
        if tool_calls_left > 0:
            llm = llm.bind_tools(tool_definitions(state["available_tools"], state["document_names"]))

        reply: AIMessage | None
        try:
            reply = await llm.ainvoke([system, *state["messages"]])
        except LLM_ERRORS as error:
            logger.warning("Supervisor LLM call failed: %s", short_error(error))
            reply = None
        if reply is not None and not (reply.tool_calls or reply.invalid_tool_calls or message_text(reply).strip()):
            logger.warning("Supervisor LLM returned an empty reply")
            reply = None

        update: OrchestratorState = {"steps": ["supervisor"]}
        if reply is None:
            budget.llm_errors_in_a_row += 1
        else:
            budget.llm_errors_in_a_row = 0
            update["messages"] = [reply]
        update["budget"] = budget.model_dump()
        return update

    async def run_tools(self, state: OrchestratorState) -> OrchestratorState:
        """Check every tool call of the supervisor's last message, run the allowed ones, answer each one.

        For each call, in order: the guards (S1, S2, S7-S12). A refused call gets its reason as the
        ToolMessage (the LLM can correct itself). Allowed ask_*/search_web calls run concurrently
        (AgentTools.run). An allowed draft_email only creates a DRAFT and sets pending_email; its
        ToolMessage comes later, from send_email, after the human decided.
        Every call is recorded in `calls` (the UI's calls panel), refused ones too.
        """
        message: AIMessage = state["messages"][-1]
        budget = Budget.model_validate(state["budget"])
        calls = [AgentCall.model_validate(call) for call in state["calls"]]
        ctx = self._context(state, calls)
        records: dict[str, AgentCall] = {}
        replies: dict[str, ToolMessage] = {}
        requests: list[ToolRequest] = []
        email: ToolRequest | None = None
        update: OrchestratorState = {"steps": ["tools"]}

        # Arguments that weren't even valid JSON: Groq still expects an answer for each such call.
        for index, bad in enumerate(message.invalid_tool_calls):
            call_id = bad.get("id") or f"invalid-{index}"
            refusal = INVALID_JSON.format(name=bad.get("name") or "the tool")
            records[call_id] = refused_call(call_id, bad.get("name") or "", {}, refusal)
            replies[call_id] = ToolMessage(refusal, tool_call_id=call_id, status="error")

        calls_in_message = len(message.tool_calls) + len(message.invalid_tool_calls)
        for tool_call in message.tool_calls:
            name, call_id = tool_call["name"], tool_call["id"]
            verdict = check_call(
                name, tool_call["args"], budget=budget, limits=self.limits, available=state["available_tools"],
                user_emails=state["user_emails"], calls_in_message=calls_in_message,
            )  # fmt: skip
            if not verdict.allowed:
                records[call_id] = refused_call(call_id, name, tool_call["args"], verdict.error)
                replies[call_id] = ToolMessage(verdict.error, tool_call_id=call_id, name=name, status="error")
                continue
            count_call(budget, name, verdict.key)  # before the next call is checked (no slipping under a limit)
            request = ToolRequest(call_id, name, verdict.args)
            if name == "draft_email":
                email = request
            else:
                requests.append(request)

        for outcome in await self.tools.run(requests, ctx):
            call = outcome.call
            records[call.call_id] = call
            status = "success" if call.status == "ok" else "error"
            replies[call.call_id] = ToolMessage(outcome.text, tool_call_id=call.call_id, name=call.tool, status=status)

        if email is not None:
            outcome = await self.tools.draft_email(email.call_id, email.args, ctx)
            records[email.call_id] = outcome.call
            if outcome.pending is not None:
                update["pending_email"] = outcome.pending.model_dump(mode="json")
            else:  # the email agent failed to draft: answer the call now, like any failed tool
                replies[email.call_id] = ToolMessage(
                    outcome.text, tool_call_id=email.call_id, name="draft_email", status="error"
                )

        order = [call.get("id") or f"invalid-{i}" for i, call in enumerate(message.invalid_tool_calls)]
        order += [tool_call["id"] for tool_call in message.tool_calls]
        calls += [records[call_id] for call_id in order if call_id in records]
        update["messages"] = [replies[call_id] for call_id in order if call_id in replies]
        update["calls"] = [call.model_dump(mode="json") for call in calls]
        update["budget"] = budget.model_dump()
        return update

    async def approval(self, state: OrchestratorState) -> OrchestratorState:
        """PAUSE until a human approves, edits or rejects the draft. Nothing else happens here.

        interrupt() saves the run and stops it; ask() returns "awaiting_approval" with the draft.
        On resume_email(), LangGraph runs this node AGAIN from its first line, and this time
        interrupt() returns the decision, validated against EmailDecision (S14). That re-run is why
        drafting and sending live in other nodes: they must happen exactly once.
        """
        pending = state["pending_email"]
        decision = interrupt(
            {"type": "email_approval", "draft": pending["draft"], "call_id": pending["tool_call_id"]},
            response_schema=EmailDecision,
        )
        return {"email_decision": EmailDecision.model_validate(decision).model_dump(), "steps": ["approval"]}

    async def send_email(self, state: OrchestratorState) -> OrchestratorState:
        """Apply the human's decision (send or discard) and give draft_email its ToolMessage at last.

        The AgentCall changes from "awaiting_approval" to "sent", "rejected" or "error".
        """
        pending = PendingEmail.model_validate(state["pending_email"])
        decision = EmailDecision.model_validate(state["email_decision"])
        outcome = await self.tools.resolve_email(pending, decision)
        calls = [
            outcome.call.model_dump(mode="json") if call["call_id"] == pending.tool_call_id else call
            for call in state["calls"]
        ]
        status = "error" if outcome.call.status == "error" else "success"
        reply = ToolMessage(outcome.text, tool_call_id=pending.tool_call_id, name="draft_email", status=status)
        return {
            "messages": [reply],
            "calls": calls,
            "pending_email": None,
            "steps": ["send_email"],
        }

    async def force_answer(self, state: OrchestratorState) -> OrchestratorState:
        """End the loop with an answer from the results so far, when a limit stopped it or the LLM failed.

        S18: every tool call must have a ToolMessage before the next LLM call, so any unanswered one
        gets "Not run: ..." first. S17: the LLM is called with NO tools bound, plus an instruction to
        answer now (not tool_choice="none", which Groq rejects). If even that fails, the answer is
        built by code from the agents' own answers (S19): the user always gets something.
        """
        budget = Budget.model_validate(state["budget"])
        calls = [AgentCall.model_validate(call) for call in state["calls"]]
        dangling = unanswered_tool_calls(state["messages"])
        system = self._system_message(state, tool_calls_left=0)
        text = ""
        try:
            reply = await self.chat_model.ainvoke(
                [system, *state["messages"], *dangling, HumanMessage(FORCE_ANSWER_INSTRUCTION)]
            )
            text = message_text(reply).strip()
        except LLM_ERRORS as error:
            logger.warning("Forced answer failed: %s", short_error(error))
        if not text:
            text = answer_from_calls(calls)

        if budget.llm_errors_in_a_row:
            stop_reason = "llm_error_forced"
        elif budget.tool_calls >= self.limits.max_tool_calls:
            stop_reason = "tool_budget"
        else:
            stop_reason = "supervisor_limit"
        return {"messages": [*dangling, AIMessage(text)], "stop_reason": stop_reason, "steps": ["force_answer"]}

    async def fallback(self, state: OrchestratorState) -> OrchestratorState:
        """The supervisor LLM failed twice before any tool ran: use the old fixed rule instead (D11).

        A chat with documents -> the Doc agent, otherwise -> the DB agent, with the user's question as
        is. It runs through AgentTools like any call, so it's recorded, timed and summarized the same way.
        """
        budget = Budget.model_validate(state["budget"])
        calls = [AgentCall.model_validate(call) for call in state["calls"]]
        tool = "ask_documents" if "ask_documents" in state["available_tools"] else "ask_database"
        args = TOOL_ARGS[tool](question=state["question"][:500])
        count_call(budget, tool, call_key(tool, args.model_dump()))
        request = ToolRequest(f"fallback-{uuid4().hex[:8]}", tool, args)
        [outcome] = await self.tools.run([request], self._context(state, calls))
        return {
            "calls": [*state["calls"], outcome.call.model_dump(mode="json")],
            "budget": budget.model_dump(),
            "stop_reason": "llm_error_fallback",
            "steps": ["fallback"],
        }

    async def finalize(self, state: OrchestratorState) -> OrchestratorState:
        """Choose the final answer text (see final_answer) and record why the loop stopped."""
        calls = [AgentCall.model_validate(call) for call in state["calls"]]
        answer, passthrough = final_answer(calls, state["messages"])
        return {
            "final_answer": answer,
            "passthrough": passthrough,
            "stop_reason": state.get("stop_reason") or "answered",
            "steps": ["finalize"],
        }

    # ------------------------------------------------------------------ routers

    def route_after_supervisor(
        self, state: OrchestratorState
    ) -> Literal["tools", "supervisor", "force_answer", "fallback", "finalize"]:
        """After an LLM call: run its tool calls, retry a failure once, give up safely, or finish (S19, S3).

        Errors: 1 in a row -> retry (if the supervisor-call limit allows); otherwise no tool has run
        yet -> fallback (the fixed rule), else -> force_answer (answer from the results so far).
        """
        budget = Budget.model_validate(state["budget"])
        if budget.llm_errors_in_a_row:
            retry_allowed = budget.supervisor_calls < self.limits.max_supervisor_calls
            if budget.llm_errors_in_a_row < MAX_LLM_ERRORS and retry_allowed:
                return "supervisor"
            return "fallback" if budget.tool_calls == 0 else "force_answer"
        last: AIMessage = state["messages"][-1]
        return "tools" if last.tool_calls or last.invalid_tool_calls else "finalize"

    def route_after_tools(self, state: OrchestratorState) -> Literal["approval", "force_answer", "supervisor"]:
        """After the tools: pause for a draft, stop at a limit (S1, S3), or let the supervisor continue."""
        if state.get("pending_email"):
            return "approval"
        budget = Budget.model_validate(state["budget"])
        out_of_tools = budget.tool_calls >= self.limits.max_tool_calls
        out_of_steps = budget.supervisor_calls >= self.limits.max_supervisor_calls
        return "force_answer" if out_of_tools or out_of_steps else "supervisor"

    def route_after_send_email(self, state: OrchestratorState) -> Literal["force_answer", "supervisor"]:
        """After the email decision: the supervisor writes the final answer, unless its call limit is reached."""
        budget = Budget.model_validate(state["budget"])
        return "force_answer" if budget.supervisor_calls >= self.limits.max_supervisor_calls else "supervisor"

    # ------------------------------------------------------------------ helpers

    def _system_message(self, state: OrchestratorState, tool_calls_left: int) -> SystemMessage:
        """The supervisor's system prompt for this call (rebuilt every time: the budget line changes)."""
        prompt = supervisor_prompt(
            state["available_tools"], state["document_names"], state["history_text"], tool_calls_left
        )
        return SystemMessage(prompt)

    def _context(self, state: OrchestratorState, calls: list[AgentCall]) -> TurnContext:
        """What the tool calls of this step share: the chat, its history, the citation labels, the question."""
        return TurnContext(
            session_id=as_uuid(state.get("session_id")),
            history=[ConversationTurn(**turn) for turn in state.get("history", [])],
            next_labels=next_labels(calls),
            question=state.get("question", ""),
        )


# ---------------------------------------------------------------------- pure helpers (easy to test)


def thread_id_for(session_id: UUID | None, message_id: int | str) -> str:
    """The checkpoint thread of one turn: "<session_id>:<message_id>" (D7). The cleanup reads the message id back."""
    return f"{session_id}:{message_id}"


def run_config(thread_id: str) -> dict[str, Any]:
    """LangGraph's run config: which checkpoint thread, and the last-resort step limit (S4)."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


def as_uuid(value: str | None) -> UUID | None:
    """The session id is kept as a string in the state (plain JSON); the agents want a UUID."""
    return UUID(value) if value else None


def message_text(message: BaseMessage) -> str:
    """The text of a message, whether its content is a string or a list of content blocks."""
    if isinstance(message.content, str):
        return message.content
    return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in message.content)


def refused_call(call_id: str, name: str, args: Any, error: str) -> AgentCall:
    """The record of a call the guards refused (shown in the UI with its reason)."""
    return AgentCall(
        call_id=call_id, tool=name, agent=TOOL_AGENTS.get(name, ""),
        input=args if isinstance(args, dict) else {}, status="refused", error=error,
    )  # fmt: skip


def unanswered_tool_calls(messages: list[AnyMessage]) -> list[ToolMessage]:
    """S18: a "Not run" ToolMessage for every tool call that has none yet (the LLM API requires one each)."""
    answered = {message.tool_call_id for message in messages if isinstance(message, ToolMessage)}
    return [
        ToolMessage(
            "Not run: the tool budget for this question is used up.",
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
            status="error",
        )  # fmt: skip
        for message in messages
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
        if tool_call["id"] not in answered
    ]


def final_answer(calls: list[AgentCall], messages: list[AnyMessage]) -> tuple[str, bool]:
    """The answer text, and whether it's a passthrough of one agent's own answer (D9).

    1. Passthrough: exactly one ask_*/search_web call ran, it succeeded, and no email was involved
       -> that agent's answer, unchanged (exact numbers, its own [n] citations). The most common
       turn ("Which vendors...?") then can't be paraphrased wrongly by the supervisor.
    2. Otherwise the supervisor's (or the forced) final text, with citation labels that were never
       issued removed ([D7] when only D1-D3 exist): a label must always lead to a real source.
       Labels written 【W1】 (gpt-oss's own style) become [W1] first, so the UI turns them into badges.
    3. No text at all (the fallback path): the agent's answer, or why it failed.
    """
    executed = [call for call in calls if call.tool in ASK_TOOLS and call.status != "refused"]
    emailed = any(call.tool == "draft_email" for call in calls)
    only = executed[0] if len(executed) == 1 else None
    if only and only.status == "ok" and only.result and only.result.get("success") and not emailed:
        return only.result["answer"], True

    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and not last.tool_calls and message_text(last).strip():
        issued = {label for call in calls for label in call.citation_labels}
        text = FULLWIDTH_LABEL.sub(r"[\1]", message_text(last))
        text = LABEL_MARK.sub(lambda m: m.group(0) if m.group(1) in issued else "", text)
        return text.strip(), False

    if only and only.status == "ok" and only.result:
        return only.result["answer"], True
    if only and only.error:
        return f"Sorry, I couldn't answer that: the {AGENT_TITLES[only.agent]} failed ({only.error}).", False
    return FAILED_TEXT, False


def labelled_answer(call: AgentCall) -> str:
    """An agent's own answer with its [n] citations renamed to the labels this turn issued ([D1], [W1])."""
    answer = (call.result or {}).get("answer", "")
    label_of = {n: label for label, n in call.citation_labels.items()}
    if not label_of:
        return answer
    return CITATION_MARK.sub(lambda m: f"[{label_of[int(m.group(1))]}]" if int(m.group(1)) in label_of else "", answer)


def answer_from_calls(calls: list[AgentCall]) -> str:
    """S19: when even the forced answer fails, join the agents' own answers into one text (no LLM)."""
    parts = [
        f"**{AGENT_TITLES[call.agent].capitalize()}:** {labelled_answer(call)}"
        for call in calls
        if call.tool in ASK_TOOLS and call.status == "ok" and call.result
    ]
    if not parts:
        return "Sorry, I couldn't finish the answer. Please try again."
    return "I couldn't write a combined answer, so here is what each agent found:\n\n" + "\n\n".join(parts)


def describe_call(call: AgentCall) -> str:
    """One reasoning line for a call, written by code (not the LLM), e.g. 'Asked the database agent: "..." → 8 rows.'"""
    asked = call.input.get("question") or call.input.get("query") or ""
    result = call.result or {}
    if call.status == "refused":
        return f"Refused {call.tool or 'an unknown tool'}: {call.error}"
    if call.status == "error":
        return f"The {AGENT_TITLES.get(call.agent, 'agent')} failed: {call.error}"
    if call.tool == "draft_email":
        to = ", ".join(call.input.get("to", []))
        return {
            "awaiting_approval": f"Drafted an email to {to}; it waits for your approval.",
            "sent": f"Sent the email to {to} after your approval.",
            "rejected": f"You rejected the email to {to}; nothing was sent.",
        }.get(call.status, f"Email to {to}: {call.status}.")
    if call.tool == "ask_database":
        rows = len(result.get("rows", []))
        found = f"{rows} row{'s' if rows != 1 else ''}" if result.get("success") else "it could not answer"
        return f'Asked the database agent: "{asked}" → {found}.'
    if call.tool == "ask_documents":
        sources = len(result.get("citations", []))
        found = f"{sources} cited source{'s' if sources != 1 else ''}" if sources else "nothing found"
        return f'Asked the document agent: "{asked}" → {found}.'
    pages = len(result.get("sources", []))
    return f'Searched the web: "{asked}" → {pages} page{"s" if pages != 1 else ""}.'


def explain(calls: list[AgentCall], stop_reason: str | None) -> list[str]:
    """The reasoning lines shown in the UI: one per call, plus why the loop stopped if it wasn't a normal answer."""
    lines = [describe_call(call) for call in calls] or ["Answered directly, without asking an agent."]
    if stop_reason in STOP_TEXTS:
        lines.append(STOP_TEXTS[stop_reason])
    return lines


def email_summary(calls: list[AgentCall]) -> dict[str, Any] | None:
    """The email of this turn for the UI: the draft (while awaiting) or what happened to it. None = no email."""
    call = next((c for c in calls if c.tool == "draft_email" and c.status != "refused"), None)
    if call is None:
        return None
    result = call.result or {}
    draft = result.get("draft") or {}
    return {
        "status": call.status,
        "draft_id": draft.get("draft_id"),
        "to": draft.get("to") or call.input.get("to", []),
        "subject": draft.get("subject") or call.input.get("subject", ""),
        "body": draft.get("body") or call.input.get("content", ""),
        "message_id": result.get("message_id"),
        "issues": draft.get("issues") or [],  # the email crew's reviewer notes, shown on the approval card
        "error": call.error,
    }


def build_result(state: dict[str, Any], question: str, status: str, error: str | None = None) -> OrchestratorResult:
    """Package the graph's state as an OrchestratorResult (what the API saves and returns)."""
    calls = [AgentCall.model_validate(call) for call in state.get("calls", [])]
    budget = Budget.model_validate(state.get("budget") or {})
    stop_reason = state.get("stop_reason")
    email = email_summary(calls)
    if status == "awaiting_approval":
        answer = f"I drafted an email to {', '.join(email['to'])}. Please review it and approve it before it is sent."
    elif status == "failed":
        answer = error or FAILED_TEXT
    else:
        answer = state.get("final_answer") or FAILED_TEXT
    return OrchestratorResult(
        question=question,
        answer=answer,
        status=status,
        success=status != "failed",
        calls=calls,
        email=email,
        stop_reason=stop_reason,
        steps=state.get("steps", []),
        reasoning=explain(calls, stop_reason),
        budget={"tool_calls": budget.tool_calls, "supervisor_calls": budget.supervisor_calls},
        error=error,
        passthrough=bool(state.get("passthrough")) and status == "completed",
    )


# ---------------------------------------------------------------------- CLI


def print_update(update: dict[str, Any]) -> None:
    """CLI: print what each node just did, so you can watch the ReAct loop step by step."""
    for node, changes in update.items():
        changes = changes or {}
        if node == "prepare":
            print(f"  prepare     tools: {', '.join(changes['available_tools'])}")
        elif node == "supervisor":
            [reply] = changes.get("messages") or [None]
            if reply is None:
                print("  supervisor  LLM call failed (it may be retried)")
            elif reply.tool_calls:
                for tool_call in reply.tool_calls:
                    print(f"  supervisor  -> {tool_call['name']}({_short(tool_call['args'])})")
            else:
                print(f"  supervisor  answers: {_short(message_text(reply))}")
        elif node == "tools":
            for message in changes.get("messages", []):
                mark = "ok " if message.status == "success" else "ERR"
                print(f"  tools       {mark} {message.name}: {len(message.content)} chars: {_short(message.content)}")
            if changes.get("pending_email"):
                print("  tools       draft_email: draft created, waiting for approval")
        elif node == "send_email":
            print(f"  send_email  {_short(changes['messages'][0].content)}")
        else:
            print(f"  {node}")


def _short(value: Any, limit: int = 110) -> str:
    """One line, cut to `limit` characters (CLI output only)."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def print_result(result: OrchestratorResult, seconds: float) -> None:
    """CLI: the outcome of one run (or resume)."""
    print(f"\nStatus: {result.status}   stop: {result.stop_reason}   passthrough: {result.passthrough}")
    print(f"Budget: {result.budget}   steps: {' -> '.join(result.steps)}   time: {seconds:.1f}s")
    for line in result.reasoning:
        print(f"  - {line}")
    print(f"\nAnswer:\n{result.answer}\n")


def ask_decision() -> EmailDecision:
    """CLI: approve, edit or reject the draft in the terminal."""
    while True:
        choice = input("approve / edit / reject? ").strip().lower()
        if choice in ("approve", "a"):
            return EmailDecision(approved=True)
        if choice in ("reject", "r"):
            return EmailDecision(approved=False)
        if choice in ("edit", "e"):
            return EmailDecision(approved=True, edited_body=input("New body (one line): "))


async def _main() -> None:
    """CLI: run the orchestrator with the same agents as the API: DB, Doc (with --session), web search (DuckDuckGo
    MCP server) and the email service (A2A, when it's running; otherwise draft_email isn't offered)."""
    from app.agents.db.database import ReadOnlyDatabase
    from app.agents.db.graph import DBAgent
    from app.agents.db.schema import load_schema
    from app.agents.doc.store import DocumentStore
    from app.agents.email.a2a_agent import A2AEmailAgent
    from app.agents.web.graph import WebSearchAgent
    from app.agents.web.mcp import WebSearchMCP
    from app.chat.memory import load_history
    from app.chat.store import ChatStore
    from app.orchestrator.checkpoint import CheckpointStore
    from app.orchestrator.models import Limits

    parser = argparse.ArgumentParser(description="Ask the orchestrator a question.")
    parser.add_argument("question", nargs="*", help="question in plain English")
    parser.add_argument("--session", type=UUID, help="a chat id: use its documents and its memory")
    parser.add_argument("--postgres", action="store_true", help="checkpoint in Postgres (app.checkpoints)")
    parser.add_argument("--diagram", action="store_true", help="print the graph as a Mermaid diagram")
    args = parser.parse_args()
    question = " ".join(args.question) or "Which vendors have overdue invoices?"
    settings = get_settings()
    limits = Limits.from_settings(settings)

    async with AsyncExitStack() as stack:
        db = await stack.enter_async_context(ReadOnlyDatabase(settings.readonly_database_url))
        documents = await stack.enter_async_context(DocumentStore(settings.app_database_url))
        chat_store = await stack.enter_async_context(ChatStore(settings.app_database_url))
        doc_agent = await _cli_doc_agent(documents) if args.session else None  # loading models takes a while
        web_mcp = WebSearchMCP(settings)  # starts the DuckDuckGo MCP server (uvx), as the API's startup does
        await web_mcp.start()
        stack.push_async_callback(web_mcp.close)
        web_agent = WebSearchAgent(
            web_mcp, max_results=settings.web_max_results, snippet_chars=settings.web_snippet_chars
        )
        email_agent = A2AEmailAgent.from_settings(settings)
        await email_agent.start()
        stack.push_async_callback(email_agent.close)
        tools = AgentTools(
            DBAgent(db, await load_schema(db)), doc_agent, web_agent, email_agent, documents, limits,
        )  # fmt: skip
        checkpointer = None
        if args.postgres:
            checkpoints = await stack.enter_async_context(CheckpointStore(settings.app_database_url))
            await checkpoints.setup()
            checkpointer = checkpoints.saver
        orchestrator = Orchestrator(tools, checkpointer)
        if args.diagram:
            print(orchestrator.graph.get_graph().draw_mermaid())
            return

        history = await load_history(chat_store, args.session, settings.chat_memory_turns) if args.session else []
        message_id = f"cli-{uuid4().hex[:8]}"
        print(f"Question: {question}\n")
        started = time.perf_counter()
        result = await orchestrator.ask(question, history, args.session, message_id, on_update=print_update)
        print_result(result, time.perf_counter() - started)
        while result.status == "awaiting_approval":
            email = result.email or {}
            print(f"--- EMAIL DRAFT (thread {thread_id_for(args.session, message_id)})")
            print(f"To: {', '.join(email.get('to', []))}\nSubject: {email.get('subject')}\n\n{email.get('body')}\n---")
            decision = ask_decision()
            started = time.perf_counter()
            result = await orchestrator.resume_email(args.session, message_id, decision, on_update=print_update)
            print_result(result, time.perf_counter() - started)


async def _cli_doc_agent(documents: Any) -> Any:
    """Build the real Doc agent for the CLI (loads the embedding and rerank models)."""
    from app.agents.doc.embeddings import Embedder, Reranker
    from app.agents.doc.graph import DocAgent
    from app.agents.doc.retrieval import Retriever

    settings = get_settings()
    models = settings.data_dir / "models"
    embedder = await asyncio.to_thread(Embedder, settings.rag_embedding_model, models)
    reranker = await asyncio.to_thread(Reranker, settings.rag_rerank_model, models) if settings.rag_rerank else None
    retriever = Retriever(documents, embedder, reranker, settings.rag_candidates)
    return DocAgent(
        documents, retriever, settings.rag_top_k, settings.rag_context_tokens, settings.rag_min_rerank_score
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s - %(message)s")
    asyncio.run(_main())
