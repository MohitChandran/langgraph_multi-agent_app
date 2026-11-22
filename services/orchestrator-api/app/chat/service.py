"""One chat turn, start to finish (plan 01 section 6; plan 03 section 11.1 for the orchestrator).

    1  no session_id?  create the session (a chat is created by its first message)
    2  claim the session: one reply at a time per chat (404 if unknown, 409 if busy)
    3  an email in this chat still waits for approval?  409 (S15), nothing is saved
    4  load memory: the last few exchanges of THIS chat (before saving the new question!)
    5  save the question + a 'pending' answer slot, in one transaction
    6  the input guard checks the message (plan 04): BLOCKED -> the refusal is saved as the answer and
       nothing else runs (no orchestrator, no agent, no tool); PASSED -> go on
    7  run the orchestrator with the question and the memory
    8  save its result into that slot: complete, error, or awaiting_approval (an email draft waits)
    9  ALWAYS release the claim (finally), so a crash can't leave the chat stuck "busy"

An answer that waits for approval is finished later by resolve_email_approval(): the user's
decision resumes the paused orchestrator run, and the SAME message gets the final answer.

This module knows nothing about HTTP: routes.py turns its exceptions into status codes.
It takes the store, the guard and the orchestrator as parameters, so tests pass fakes.
"""

import logging
import time
from typing import Any, Protocol
from uuid import UUID

import psycopg

from app.chat.memory import ConversationTurn, load_history
from app.chat.models import ApprovalResponse, ChatResponse, Message
from app.chat.store import ChatStore
from app.guardrails.guard import GuardResult
from app.orchestrator.models import EmailDecision, OrchestratorResult

logger = logging.getLogger(__name__)

TITLE_MAX_CHARS = 60
AGENT_FAILED_TEXT = "Sorry, something went wrong while answering. Please try again."
APPROVAL_PENDING_TEXT = "Approve or reject the pending email first."


class SessionNotFoundError(LookupError):
    """The session doesn't exist (wrong id, or deleted while a reply was being generated)."""


class SessionBusyError(RuntimeError):
    """The session is already generating a reply; only one at a time is allowed."""


class ApprovalPendingError(RuntimeError):
    """An answer in this chat waits for an email decision; no new message until then (S15)."""


class MessageNotFoundError(LookupError):
    """No message with this id (the approval endpoint)."""


class NotAwaitingApprovalError(RuntimeError):
    """The message isn't waiting for an email decision (already decided, or never had an email)."""


class ChatOrchestrator(Protocol):
    """What the chat service needs from the orchestrator (app/orchestrator/graph.py). Tests pass fakes.

    A Protocol is "duck typing with a type check": any object with matching methods fits,
    without inheriting from anything.
    """

    name: str

    async def ask(
        self,
        question: str,
        history: list[ConversationTurn] | None = None,
        session_id: UUID | None = None,
        message_id: int | str | None = None,
    ) -> OrchestratorResult:
        """Answer one question; may pause for an email approval (status "awaiting_approval")."""
        ...

    async def resume_email(self, session_id: UUID | None, message_id: int | str, decision: EmailDecision) -> Any:
        """Continue a paused turn with the human's decision."""
        ...


class ChatGuard(Protocol):
    """What the chat service needs from the input guard (app/guardrails/guard.py). Tests pass fakes."""

    async def check(self, message: str) -> GuardResult:
        """passed / blocked (with the refusal text) for one message; never raises."""
        ...


def make_title(content: str) -> str:
    """Title for a new chat: its first question, cut to TITLE_MAX_CHARS at a word boundary."""
    text = " ".join(content.split())  # collapse newlines and repeated spaces
    if len(text) <= TITLE_MAX_CHARS:
        return text
    cut = text[: TITLE_MAX_CHARS - 1].rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;:") + "…"


async def send_message(
    store: ChatStore,
    guard: ChatGuard,
    orchestrator: ChatOrchestrator,
    session_id: UUID | None,
    content: str,
    memory_turns: int = 3,
) -> ChatResponse:
    """Run one turn and return the session plus the saved question and answer.

    `memory_turns` = how many earlier exchanges of this chat the orchestrator sees (CHAT_MEMORY_TURNS).

    The guard runs AFTER the question is saved, so a blocked turn is part of the transcript like any
    answer (a reload shows it), and BEFORE the orchestrator, so a blocked message reaches no LLM but
    the guard's own models, no agent and no tool.

    If the orchestrator fails (e.g. Groq unreachable after its retries and fallbacks), the answer
    is saved with status 'error' and a readable text, and the turn still returns normally: the
    failure is part of the transcript, exactly what a reload would show.

    Raises:
        SessionNotFoundError: unknown session_id, or the chat was deleted mid-turn.
        SessionBusyError: another reply is still being generated in this chat.
        ApprovalPendingError: an email in this chat waits for approval (S15).
    """
    if session_id is None:
        session_id = (await store.create_session(make_title(content))).id

    await _claim(store, session_id)
    try:
        # Checked AFTER claiming: the claim and the awaiting status are both set by the turn that
        # holds the chat, so no second turn can slip in between the check and the start.
        if await store.has_awaiting_approval(session_id):
            raise ApprovalPendingError(APPROVAL_PENDING_TEXT)
        # Memory first: once the question is saved, it would show up in its own history.
        history = await load_history(store, session_id, memory_turns)
        try:
            user_message, pending = await store.add_turn(session_id, content)
        except psycopg.errors.ForeignKeyViolation as error:  # deleted between claim and insert
            raise SessionNotFoundError(f"Chat {session_id} was deleted.") from error

        started = time.perf_counter()
        verdict = await guard.check(content)
        if verdict.status == "blocked":
            payload = guard_payload(verdict)
            assistant_message = await store.complete_message(pending.id, verdict.answer, payload, _elapsed_ms(started))
        else:
            try:
                result = await orchestrator.ask(content, history, session_id, pending.id)
            except Exception:  # the orchestrator catches its own failures; this is the last safety net
                logger.exception("Orchestrator crashed for message %s", pending.id)
                result = None
            passed = {"guard": guard_summary(verdict)}
            assistant_message = await save_result(store, pending.id, result, _elapsed_ms(started), passed)
    finally:
        session = await store.release_session(session_id)

    if session is None or assistant_message is None:
        raise SessionNotFoundError(f"Chat {session_id} was deleted while answering.")
    return ChatResponse(session=session, user_message=user_message, assistant_message=assistant_message)


async def resolve_email_approval(
    store: ChatStore, orchestrator: ChatOrchestrator, message_id: int, decision: EmailDecision
) -> ApprovalResponse:
    """Apply the user's decision on an email draft and finish the paused answer (plan 03, 11.1).

    The paused orchestrator run is resumed (send or discard the draft, then the final answer), and
    the SAME assistant message is completed: its content and payload are replaced by the final result.

    Raises:
        MessageNotFoundError: no such message (404).
        NotAwaitingApprovalError: the message isn't waiting for a decision (409).
        SessionBusyError: the chat is busy, e.g. the same decision was sent twice at once (409).
        SessionNotFoundError: the chat was deleted meanwhile (404).
    """
    found = await store.get_message(message_id)
    if found is None:
        raise MessageNotFoundError(f"Message {message_id} does not exist.")
    session_id, message = found
    if message.status != "awaiting_approval":
        raise NotAwaitingApprovalError("This message is not waiting for an email approval.")

    await _claim(store, session_id)
    try:
        # Re-read under the claim: a decision that raced this one may have finished in between.
        found = await store.get_message(message_id)
        if found is None or found[1].status != "awaiting_approval":
            raise NotAwaitingApprovalError("This message is not waiting for an email approval.")
        started = time.perf_counter()
        try:
            result = await orchestrator.resume_email(session_id, message_id, decision)
        except Exception:
            logger.exception("Orchestrator crashed resuming message %s", message_id)
            result = None
        latency = (message.latency_ms or 0) + _elapsed_ms(started)  # time before the pause + after it
        kept = {"guard": message.payload["guard"]} if "guard" in message.payload else None  # from the first half
        assistant_message = await save_result(store, message_id, result, latency, kept)
    finally:
        session = await store.release_session(session_id)

    if session is None or assistant_message is None:
        raise SessionNotFoundError(f"Chat {session_id} was deleted while answering.")
    return ApprovalResponse(session=session, assistant_message=assistant_message)


async def save_result(
    store: ChatStore,
    message_id: int,
    result: OrchestratorResult | None,
    latency_ms: int,
    extra: dict[str, Any] | None = None,
) -> Message | None:
    """Fill the answer slot according to the result's status. None if the message vanished (chat deleted).

    completed          -> 'complete', the answer as content, the result as payload
    awaiting_approval  -> 'awaiting_approval', "I drafted an email ..." as content (the UI shows the draft)
    failed / crashed   -> 'error', a readable text
    `extra` is merged into the payload (the guard's "passed" verdict, shown in the UI flow).
    """
    if result is None or result.status == "failed":
        text = result.answer if result is not None else AGENT_FAILED_TEXT
        return await store.fail_message(message_id, text, latency_ms)
    payload = orchestrator_payload(result) | (extra or {})
    if result.status == "awaiting_approval":
        return await store.mark_awaiting_approval(message_id, result.answer, payload, latency_ms)
    return await store.complete_message(message_id, result.answer, payload, latency_ms)


def orchestrator_payload(result: OrchestratorResult) -> dict[str, Any]:
    """What is saved with an answer: who answered, the route summary, and the full result (minus the answer).

    "route" keeps the shape the UI's flow panel already reads ({router, agent, reason}): "router": "llm"
    shows as "Orchestrator", "agent" lists the agents used in order (e.g. "db → email"), and "reason"
    is the first reasoning line. The answer itself is the message content, so it isn't stored twice.
    """
    agents = [call.agent for call in result.calls if call.status != "refused" and call.agent]
    route = {
        "router": "llm",
        "agent": " → ".join(agents) or "none",
        "reason": result.reasoning[0] if result.reasoning else "",
    }
    return {
        "agent": "orchestrator",
        "route": route,
        "result": result.model_dump(mode="json", exclude={"question", "answer"}),
    }


def guard_payload(verdict: GuardResult) -> dict[str, Any]:
    """What is saved with a BLOCKED turn: agent "guard", a route for the UI's flow panel, and the verdict.

    The refusal itself is the message content (verdict.answer), so it isn't stored twice.
    """
    reason = verdict.category or verdict.rail or "blocked"
    return {
        "agent": "guard",
        "route": {"router": "guard", "agent": "guard", "reason": reason},
        "result": verdict.model_dump(mode="json", exclude={"answer"}),
    }


def guard_summary(verdict: GuardResult) -> dict[str, Any]:
    """The guard's verdict on a PASSED turn, kept in the orchestrator's payload ("Guardrail ✓" in the UI)."""
    return verdict.model_dump(mode="json", include={"status", "duration_ms", "jailbreak_score", "jailbreak_flagged"})


async def _claim(store: ChatStore, session_id: UUID) -> None:
    """Mark the chat busy, or raise: SessionNotFoundError if it doesn't exist, SessionBusyError if it's busy."""
    if await store.claim_session(session_id) is None:
        if await store.get_session(session_id) is None:
            raise SessionNotFoundError(f"Chat {session_id} does not exist.")
        raise SessionBusyError("A reply is still being generated in this chat.")


def _elapsed_ms(started: float) -> int:
    """Milliseconds since `started` (a time.perf_counter() value)."""
    return round((time.perf_counter() - started) * 1000)
