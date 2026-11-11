"""One chat turn, start to finish (plan 01, section 6).

    1  no session_id?  create the session (a chat is created by its first message)
    2  claim the session: one reply at a time per chat (404 if unknown, 409 if busy)
    3  save the question + a 'pending' answer slot, in one transaction
    4  run the agent
    5  save the answer (or the error) into that slot
    6  ALWAYS release the claim (finally), so a crash can't leave the chat stuck "busy"

This module knows nothing about HTTP: routes.py turns its exceptions into status codes.
It takes the store and the agent as parameters, so tests pass a fake agent.
"""

import logging
import time
from typing import Protocol
from uuid import UUID

import psycopg

from app.agents.db.models import DBAgentResult
from app.chat.models import ChatResponse, Message
from app.chat.store import ChatStore

logger = logging.getLogger(__name__)

TITLE_MAX_CHARS = 60
AGENT_FAILED_TEXT = "Sorry, something went wrong while answering. Please try again."


class SessionNotFoundError(LookupError):
    """The session doesn't exist (wrong id, or deleted while a reply was being generated)."""


class SessionBusyError(RuntimeError):
    """The session is already generating a reply; only one at a time is allowed."""


class ChatAgent(Protocol):
    """What the chat service needs from an agent: just an async `ask`.

    A Protocol is "duck typing with a type check": any object with a matching `ask`
    method fits, without inheriting from anything. DBAgent fits; so does a test fake.
    """

    async def ask(self, question: str) -> DBAgentResult:
        """Answer one question."""
        ...


def make_title(content: str) -> str:
    """Title for a new chat: its first question, cut to TITLE_MAX_CHARS at a word boundary."""
    text = " ".join(content.split())  # collapse newlines and repeated spaces
    if len(text) <= TITLE_MAX_CHARS:
        return text
    cut = text[: TITLE_MAX_CHARS - 1].rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;:") + "…"


async def send_message(store: ChatStore, agent: ChatAgent, session_id: UUID | None, content: str) -> ChatResponse:
    """Run one turn and return the session plus the saved question and answer.

    If the agent itself crashes (e.g. Groq unreachable after its retries), the answer is
    saved with status 'error' and a readable text, and the turn still returns normally:
    the failure is part of the transcript, exactly what a reload would show.

    Raises:
        SessionNotFoundError: unknown session_id, or the chat was deleted mid-turn.
        SessionBusyError: another reply is still being generated in this chat.
    """
    if session_id is None:
        session_id = (await store.create_session(make_title(content))).id

    if await store.claim_session(session_id) is None:
        if await store.get_session(session_id) is None:
            raise SessionNotFoundError(f"Chat {session_id} does not exist.")
        raise SessionBusyError("A reply is still being generated in this chat.")

    try:
        try:
            user_message, pending = await store.add_turn(session_id, content)
        except psycopg.errors.ForeignKeyViolation as error:  # deleted between claim and insert
            raise SessionNotFoundError(f"Chat {session_id} was deleted.") from error

        started = time.perf_counter()
        assistant_message = await _run_agent(store, agent, pending, content, started)
    finally:
        session = await store.release_session(session_id)

    if session is None or assistant_message is None:
        raise SessionNotFoundError(f"Chat {session_id} was deleted while answering.")
    return ChatResponse(session=session, user_message=user_message, assistant_message=assistant_message)


async def _run_agent(
    store: ChatStore, agent: ChatAgent, pending: Message, question: str, started: float
) -> Message | None:
    """Ask the agent and fill the pending answer with its result, or with an error if it crashed.

    The payload keeps everything the UI's "Show SQL & data" panel needs (SQL, rows,
    steps...) under "result"; the answer text itself goes into the message content.
    "agent" records which agent answered: later chats can mix agents (Phase 2 routing).
    Returns None if the message vanished (its chat was deleted while the agent ran).
    """
    try:
        result = await agent.ask(question)
    except Exception:
        logger.exception("Agent failed for message %s", pending.id)
        return await store.fail_message(pending.id, AGENT_FAILED_TEXT, _elapsed_ms(started))

    payload = {"agent": "db", "result": result.model_dump(mode="json", exclude={"question", "answer"})}
    return await store.complete_message(pending.id, result.answer, payload, _elapsed_ms(started))


def _elapsed_ms(started: float) -> int:
    """Milliseconds since `started` (a time.perf_counter() value)."""
    return round((time.perf_counter() - started) * 1000)
