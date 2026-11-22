"""Tests for the orchestrator's ReAct loop (app/orchestrator/graph.py), plan 03 section 17.

The REAL graph runs with a scripted fake LLM (tests/orchestrator_fakes.py), fake DB/Doc agents, the
dummy web/email agents and LangGraph's in-memory checkpointer: every limit, refusal, fallback and the
email pause is tested exactly, with no Groq call and no database.
"""

import itertools
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

import app.orchestrator.graph as graph_module
from app.agents.email.dummy import DummyEmailAgent
from app.chat.memory import ConversationTurn
from app.orchestrator.graph import (
    INVALID_JSON,
    NOT_WAITING_TEXT,
    TIMEOUT_TEXT,
    TOO_MANY_STEPS_TEXT,
    Orchestrator,
    final_answer,
    run_config,
    thread_id_for,
    unanswered_tool_calls,
)
from app.orchestrator.guards import BUDGET_USED_UP, DUPLICATE_CALL
from app.orchestrator.models import AgentCall, EmailDecision, Limits
from app.orchestrator.prompts import FORCE_ANSWER_INSTRUCTION
from tests.orchestrator_fakes import ScriptedLLM, answer, call, calls, groq_down
from tests.test_orchestrator_tools import FakeAgent, db_result, make_tools

SESSION = uuid4()
QUESTION = "Which vendors have overdue invoices?"
DB_ANSWER = "There are overdue invoices."  # what db_result() answers


def orchestrator(llm: ScriptedLLM, **tool_options) -> Orchestrator:  # noqa: ANN003
    """The real Orchestrator wired to the fake LLM and fake agents (see make_tools for the options)."""
    return Orchestrator(make_tools(**tool_options), chat_model=llm)


class CountingEmailAgent(DummyEmailAgent):
    """The dummy email agent, counting how many drafts it was asked for."""

    def __init__(self) -> None:
        """No drafts yet."""
        super().__init__()
        self.drafts = 0

    async def ask(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        """Count, then draft as usual."""
        self.drafts += 1
        return await super().ask(*args, **kwargs)


def is_forced(messages: list[BaseMessage]) -> bool:
    """True if this LLM call is the forced answer (it ends with the force-answer instruction)."""
    return isinstance(messages[-1], HumanMessage) and messages[-1].content == FORCE_ANSWER_INSTRUCTION


# --------------------------------------------------------------------------- answers


async def test_a_greeting_is_answered_without_tools() -> None:
    """No tool call: the supervisor's text is the answer; nothing was asked of any agent."""
    llm = ScriptedLLM(answer("Hi! I can answer questions about invoices, documents and more."))
    result = await orchestrator(llm).ask("hi, what can you do?", session_id=SESSION)

    assert result.status == "completed" and result.answer.startswith("Hi!")
    assert result.calls == [] and result.stop_reason == "answered" and not result.passthrough
    assert result.steps == ["prepare", "supervisor", "finalize"]
    assert result.reasoning == ["Answered directly, without asking an agent."]
    assert result.budget == {"tool_calls": 0, "supervisor_calls": 1}


async def test_one_database_call_passes_the_agents_answer_through() -> None:
    """D9: one successful call -> the agent's own answer, unchanged (not the supervisor's paraphrase)."""
    db = FakeAgent(db_result())
    llm = ScriptedLLM(call("ask_database", question=QUESTION), answer("Paraphrase: some vendors are late."))
    result = await orchestrator(llm, db=db).ask(QUESTION, session_id=SESSION)

    assert result.passthrough and result.answer == DB_ANSWER
    assert [(c.tool, c.status) for c in result.calls] == [("ask_database", "ok")]
    assert result.steps == ["prepare", "supervisor", "tools", "supervisor", "finalize"]
    assert db.asked[0][0] == QUESTION
    seen = llm.calls[1].messages[-1]  # what the supervisor got back for its call
    assert isinstance(seen, ToolMessage) and seen.content.startswith(f"Answer: {DB_ANSWER}\nSQL: SELECT")
    assert result.reasoning == [f'Asked the database agent: "{QUESTION}" → 3 rows.']


async def test_two_agents_use_the_supervisors_answer_and_drop_labels_never_issued() -> None:
    """Two calls -> the supervisor's combined text; [W9] wasn't issued (only W1, W2) so it's removed."""
    llm = ScriptedLLM(
        call("ask_database", question="Which vendors are overdue?"),
        call("search_web", query="Example Cloud SLA change"),
        answer("Vendor 0 is overdue (database). The SLA changed [W1] [W9]."),
    )
    result = await orchestrator(llm).ask("Overdue vendors, and did Example Cloud change its SLA?", session_id=SESSION)

    assert not result.passthrough and result.answer == "Vendor 0 is overdue (database). The SLA changed [W1]."
    assert [c.agent for c in result.calls] == ["db", "web"] and result.calls[1].citation_labels == {"W1": 1, "W2": 2}
    assert result.stop_reason == "answered"


def test_fullwidth_labels_become_ascii_and_unknown_ones_are_still_dropped() -> None:
    """gpt-oss's own citation style 【W1】 / 【W2†L3】 is normalized to [W1]; 【W9】 (never issued) is removed."""
    web = AgentCall(call_id="w", tool="search_web", agent="web", input={}, status="ok",
                    result={"answer": "x", "success": True}, citation_labels={"W1": 1, "W2": 2})  # fmt: skip
    db = AgentCall(call_id="d", tool="ask_database", agent="db", input={}, status="ok",
                   result={"answer": "y", "success": True})  # fmt: skip
    text, passthrough = final_answer([db, web], [answer("SLA is 99.99% 【W1】【W2†L3】 and more 【W9】.")])
    assert text == "SLA is 99.99% [W1][W2] and more." and not passthrough


# --------------------------------------------------------------------------- limits (S1, S3, S17)


async def test_an_llm_that_always_calls_tools_stops_at_the_budget_with_a_forced_answer() -> None:
    """S1 + S17: exactly 4 calls run, then the answer is forced with NO tools bound."""
    numbers = itertools.count(1)

    def keep_calling(messages: list[BaseMessage]) -> AIMessage:
        """A different call every time (no duplicates), alternating tools so no per-tool cap is hit first."""
        if is_forced(messages):
            return answer("Forced answer from 4 results.")
        n = next(numbers)
        return call("ask_database", question=f"question {n}") if n % 2 else call("search_web", query=f"query {n}")

    llm = ScriptedLLM(keep_calling)
    result = await orchestrator(llm).ask("Find out everything", session_id=SESSION)

    assert [c.status for c in result.calls] == ["ok"] * 4 and result.budget["tool_calls"] == 4
    assert result.stop_reason == "tool_budget" and result.steps[-2:] == ["force_answer", "finalize"]
    assert result.answer == "Forced answer from 4 results." and len(llm.supervisor_calls) == 4
    forced = llm.calls[-1]
    assert forced.tools is None and forced.messages[-1].content == FORCE_ANSWER_INSTRUCTION
    assert "Stopped: the tool budget" in result.reasoning[-1]


async def test_several_calls_in_one_message_run_and_the_budget_holds_inside_a_message() -> None:
    """3 calls in one message all run; in the next message the 4th runs and the 5th is refused (S1)."""
    db = FakeAgent(db_result())
    llm = ScriptedLLM(
        calls(
            ("ask_database", {"question": "q1"}), ("search_web", {"query": "w1"}), ("ask_database", {"question": "q2"})
        ),
        calls(("search_web", {"query": "w2"}), ("ask_database", {"question": "q3"})),
        answer("Answer from four results."),
    )
    result = await orchestrator(llm, db=db).ask("Everything at once", session_id=SESSION)

    assert [c.status for c in result.calls] == ["ok", "ok", "ok", "ok", "refused"]
    assert result.calls[-1].error == BUDGET_USED_UP and [q for q, *_ in db.asked] == ["q1", "q2"]
    tool_messages = [m for m in llm.calls[1].messages if isinstance(m, ToolMessage)]
    assert [m.name for m in tool_messages] == ["ask_database", "search_web", "ask_database"]  # in call order
    assert result.stop_reason == "tool_budget" and result.answer == "Answer from four results."


async def test_repeating_the_same_call_ends_at_the_supervisor_limit() -> None:
    """S9 + S3: duplicates are refused (not run, not counted); after 6 supervisor calls the answer is forced.

    The forced call here returns a tool call instead of text, so the answer is built by code from the
    agents' own answers (S19): the user still gets the data.
    """
    llm = ScriptedLLM(call("ask_database", question=QUESTION))
    result = await orchestrator(llm).ask(QUESTION, session_id=SESSION)

    assert [c.status for c in result.calls] == ["ok"] + ["refused"] * 5
    assert all(c.error == DUPLICATE_CALL for c in result.calls[1:]) and result.budget["tool_calls"] == 1
    assert result.budget["supervisor_calls"] == 6 and result.stop_reason == "supervisor_limit"
    assert result.passthrough and result.answer == DB_ANSWER  # D9 still applies: one successful call


async def test_a_turn_with_a_failed_call_and_no_llm_gets_a_text_built_from_the_results() -> None:
    """S19: when the forced LLM call fails too, the agents' own answers are joined by code."""
    llm = ScriptedLLM(
        calls(("ask_database", {"question": "q1"}), ("search_web", {"query": "w1"})),
        groq_down(),
    )
    result = await orchestrator(llm).ask("Two things", session_id=SESSION)

    assert result.stop_reason == "llm_error_forced" and not result.passthrough
    assert result.answer.startswith("I couldn't write a combined answer")
    assert f"**Database agent:** {DB_ANSWER}" in result.answer and "[W1]" in result.answer


# --------------------------------------------------------------------------- LLM failures (S19, D11)


async def test_one_llm_error_is_retried() -> None:
    """The first supervisor call fails, the retry works: a normal answer, 2 supervisor calls."""
    result = await orchestrator(ScriptedLLM(groq_down(), answer("Hello."))).ask("hi", session_id=SESSION)
    assert result.answer == "Hello." and result.stop_reason == "answered"
    assert result.budget["supervisor_calls"] == 2


async def test_two_llm_errors_before_any_call_use_the_fixed_rule() -> None:
    """D11: no documents -> the DB agent answers; with documents -> the Doc agent. Recorded as a call."""
    db = FakeAgent(db_result())
    result = await orchestrator(ScriptedLLM(groq_down()), db=db).ask(QUESTION, session_id=SESSION)
    assert result.stop_reason == "llm_error_fallback" and result.passthrough and result.answer == DB_ANSWER
    assert [c.tool for c in result.calls] == ["ask_database"] and db.asked[0][0] == QUESTION
    assert "fixed rule" in result.reasoning[-1]

    with_docs = orchestrator(ScriptedLLM(groq_down()), filenames=["q3-report.pdf"])
    result = await with_docs.ask("What does the SLA say?", session_id=SESSION)
    assert [c.tool for c in result.calls] == ["ask_documents"] and result.answer.startswith("The credit is 10% [2]")


async def test_llm_errors_after_a_call_force_an_answer() -> None:
    """Errors after a tool ran -> force_answer (not the fallback rule); one good call still passes through."""
    llm = ScriptedLLM(call("ask_database", question=QUESTION), groq_down())
    result = await orchestrator(llm).ask(QUESTION, session_id=SESSION)
    assert result.stop_reason == "llm_error_forced" and "force_answer" in result.steps
    assert result.answer == DB_ANSWER


# --------------------------------------------------------------------------- tools and refusals


async def test_a_slow_agent_times_out_and_the_loop_continues() -> None:
    """S6: the call becomes an error ToolMessage; the supervisor still answers."""
    llm = ScriptedLLM(call("ask_database", question=QUESTION), answer("The database didn't answer in time."))
    slow = orchestrator(llm, db=FakeAgent(db_result(), delay=1.0), limits=Limits(tool_timeout_seconds=0.05))
    result = await slow.ask(QUESTION, session_id=SESSION)

    assert result.status == "completed" and result.calls[0].status == "error" and not result.passthrough
    assert result.answer == "The database didn't answer in time."
    seen = llm.calls[1].messages[-1]
    assert seen.status == "error" and "took longer than 0.05 s" in seen.content


async def test_refused_calls_get_their_reason_as_the_tool_message() -> None:
    """S7 (no documents in this chat) and unparsable arguments: refused, recorded, answered."""
    unparsable = AIMessage(
        content="",
        invalid_tool_calls=[{"name": "ask_database", "args": "{oops", "id": "bad-1", "error": "bad JSON",
                             "type": "invalid_tool_call"}],
    )  # fmt: skip
    llm = ScriptedLLM(call("ask_documents", question="What does the PDF say?"), unparsable, answer("Sorry."))
    result = await orchestrator(llm).ask("What does the PDF say?", session_id=SESSION)

    assert [c.status for c in result.calls] == ["refused", "refused"]
    assert result.calls[0].error.startswith("Tool ask_documents is not available here.")
    assert result.calls[1].error == INVALID_JSON.format(name="ask_database")
    answered = [m.tool_call_id for m in llm.calls[-1].messages if isinstance(m, ToolMessage)]
    assert answered[-1] == "bad-1" and result.answer == "Sorry."


async def test_only_available_tools_are_bound_and_the_prompt_shows_files_and_budget() -> None:
    """S7: ask_documents is bound only in a chat with documents; the system prompt names the file."""
    llm = ScriptedLLM(answer("Hi."))
    await orchestrator(llm).ask("hi", session_id=SESSION)
    assert llm.calls[0].tool_names == ["ask_database", "search_web", "draft_email"]

    llm = ScriptedLLM(answer("Hi."))
    await orchestrator(llm, filenames=["q3-report.pdf"]).ask("hi", session_id=SESSION)
    assert llm.calls[0].tool_names == ["ask_database", "ask_documents", "search_web", "draft_email"]
    system = llm.calls[0].messages[0]
    assert isinstance(system, SystemMessage) and "THIS chat: q3-report.pdf" in system.content
    assert "at most 4 more tool calls" in system.content


async def test_history_goes_into_the_system_prompt_and_to_the_agents_not_into_messages() -> None:
    """D8: the earlier exchange is in the system prompt; the messages are only this question; agents get it too."""
    turn = ConversationTurn(question=QUESTION, answer="Vendor 0.", sql="SELECT 1")
    db = FakeAgent(db_result())
    llm = ScriptedLLM(call("ask_database", question="Which overdue vendors are in India?"), answer("x"))
    await orchestrator(llm, db=db).ask("and only India?", history=[turn], session_id=SESSION)

    system, *messages = llm.calls[0].messages
    assert f"Q1: {QUESTION}" in system.content and "SQL1: SELECT 1" in system.content
    assert [m.content for m in messages] == ["and only India?"]
    assert db.asked[0][1] == [turn] and db.asked[0][2] == SESSION


def test_dangling_tool_calls_get_a_not_run_message() -> None:
    """S18: every tool call without a ToolMessage gets one before the forced LLM call."""
    ai = AIMessage(content="", tool_calls=[
        {"name": "ask_database", "args": {}, "id": "a", "type": "tool_call"},
        {"name": "search_web", "args": {}, "id": "b", "type": "tool_call"},
    ])  # fmt: skip
    [missing] = unanswered_tool_calls([HumanMessage("q"), ai, ToolMessage("done", tool_call_id="a")])
    assert missing.tool_call_id == "b" and missing.content.startswith("Not run:") and missing.status == "error"


# --------------------------------------------------------------------------- the email pause (S11, S13, S14)

EMAIL_QUESTION = f"{QUESTION} Email them to accounts@company.example"


def email_script(final_text: str) -> ScriptedLLM:
    """Data first, then a draft to the address the user typed, then the final answer."""
    return ScriptedLLM(
        call("ask_database", question=QUESTION),
        call("draft_email", to=["accounts@company.example"], subject="Overdue invoices", content="3 vendors are late."),
        answer(final_text),
    )


async def test_an_email_pauses_and_approving_sends_it_once() -> None:
    """The run pauses with the draft; approving (with an edited body) sends it; drafting ran only once."""
    llm = email_script("Done: I emailed the list to accounts@company.example.")
    orch = orchestrator(llm)
    orch.tools.email_agent = email_agent = CountingEmailAgent()

    paused = await orch.ask(EMAIL_QUESTION, session_id=SESSION, message_id=1)
    assert paused.status == "awaiting_approval" and paused.success
    assert paused.answer == (
        "I drafted an email to accounts@company.example. Please review it and approve it before it is sent."
    )
    assert [c.status for c in paused.calls] == ["ok", "awaiting_approval"]
    assert paused.email["status"] == "awaiting_approval" and "3 vendors are late." in paused.email["body"]
    config = run_config(thread_id_for(SESSION, 1))
    assert (await orch.graph.aget_state(config)).next == ("approval",)

    done = await orch.resume_email(SESSION, 1, EmailDecision(approved=True, edited_body="Edited body."))
    assert done.status == "completed" and done.answer == "Done: I emailed the list to accounts@company.example."
    assert done.email["status"] == "sent" and done.email["body"] == "Edited body." and not done.passthrough
    assert done.email["message_id"].startswith("dummy-") and email_agent.drafts == 1
    assert done.steps == [
        "prepare", "supervisor", "tools", "supervisor", "tools", "approval", "send_email", "supervisor", "finalize",
    ]  # fmt: skip
    assert llm.calls[-1].messages[-1].content == "Approved by the user and sent to accounts@company.example."
    assert await orch.checkpointer.aget_tuple(config) is None  # the finished turn's thread is deleted


async def test_rejecting_sends_nothing_and_a_second_draft_is_refused() -> None:
    """Reject -> "rejected"; the LLM tries another draft -> refused (one email per question, S2)."""
    llm = ScriptedLLM(
        call("draft_email", to=["accounts@company.example"], subject="Hello", content="Hi."),
        call("draft_email", to=["accounts@company.example"], subject="Hello again", content="Hi!"),
        answer("Okay, nothing was sent."),
    )
    orch = orchestrator(llm)
    await orch.ask("Say hello to accounts@company.example by email", message_id=2)
    done = await orch.resume_email(None, 2, EmailDecision(approved=False))

    assert [c.status for c in done.calls] == ["rejected", "refused"]
    assert done.calls[1].error == "Not run: draft_email can be used at most once per question."
    assert done.email["status"] == "rejected" and done.answer == "Okay, nothing was sent."
    assert "REJECTED" in [m for m in llm.calls[1].messages if isinstance(m, ToolMessage)][0].content


async def test_s11_an_address_found_only_in_data_is_refused_but_one_typed_earlier_is_allowed() -> None:
    """A recipient must come from the user's own words: this question or an earlier question of the chat."""
    llm = ScriptedLLM(
        call("draft_email", to=["ceo@vendor.example"], subject="s", content="c"),
        answer("I can only email addresses you give me."),
    )
    refused = await orchestrator(llm).ask("Email the vendor's CEO the summary", session_id=SESSION)
    assert refused.status == "completed" and refused.calls[0].status == "refused" and refused.email is None
    assert refused.calls[0].error.startswith("ceo@vendor.example was not given by the user.")

    earlier = ConversationTurn(question="My address is me@company.example", answer="Noted.", sql=None)
    llm = ScriptedLLM(call("draft_email", to=["me@company.example"], subject="s", content="c"))
    paused = await orchestrator(llm).ask("Email it to me", history=[earlier], session_id=SESSION)
    assert paused.status == "awaiting_approval"


async def test_an_email_to_two_recipients_is_refused_and_one_works() -> None:
    """One recipient per email (S12, plan 04): a two-address draft is refused with a reason the LLM can act on."""
    two = ["accounts@company.example", "legal@company.example"]
    llm = ScriptedLLM(
        call("draft_email", to=two, subject="s", content="c"),
        call("draft_email", to=two[:1], subject="s", content="c"),
    )
    paused = await orchestrator(llm).ask(f"Email it to {two[0]} and {two[1]}", session_id=SESSION)
    assert [c.status for c in paused.calls] == ["refused", "awaiting_approval"]
    assert paused.calls[0].error == "Too many recipients (2): at most 1 per email."
    assert paused.email["to"] == ["accounts@company.example"]


def test_the_supervisor_prompt_has_the_open_scope_rule() -> None:
    """General factual questions go to search_web (with sources); only small talk is answered without tools."""
    from app.orchestrator.prompts import supervisor_prompt

    prompt = supervisor_prompt(["ask_database", "search_web", "draft_email"], [], "", 4)
    assert "anything else factual or current (history, science, news, public information) -> search_web" in prompt
    assert "Exactly one recipient per email" in prompt


async def test_resuming_a_turn_that_is_not_paused_fails_safely() -> None:
    """An unknown thread, or a turn that already finished, can't be resumed: failed, nothing sent."""
    orch = orchestrator(ScriptedLLM(answer("Hi.")))
    unknown = await orch.resume_email(SESSION, 404, EmailDecision(approved=True))
    assert unknown.status == "failed" and unknown.answer == NOT_WAITING_TEXT

    await orch.ask("hi", session_id=SESSION, message_id=5)
    assert (await orch.resume_email(SESSION, 5, EmailDecision(approved=True))).answer == NOT_WAITING_TEXT


# --------------------------------------------------------------------------- whole-turn limits (S4, S5)


async def test_turn_timeout_and_step_limit_give_failed_results(monkeypatch) -> None:  # noqa: ANN001
    """S5: a turn over its time is stopped; S4: a run over the step limit is stopped. Both delete the thread."""
    slow = orchestrator(
        ScriptedLLM(call("ask_database", question="q"), answer("x")),
        db=FakeAgent(db_result(), delay=1.0), limits=Limits(turn_timeout_seconds=0.05),
    )  # fmt: skip
    result = await slow.ask("q", session_id=SESSION, message_id=7)
    assert result.status == "failed" and result.answer == TIMEOUT_TEXT and not result.success
    assert await slow.checkpointer.aget_tuple(run_config(thread_id_for(SESSION, 7))) is None

    monkeypatch.setattr(graph_module, "RECURSION_LIMIT", 3)
    looping = orchestrator(ScriptedLLM(call("ask_database", question="q"), answer("x")))
    result = await looping.ask("q", session_id=SESSION, message_id=8)
    assert result.status == "failed" and result.answer == TOO_MANY_STEPS_TEXT
