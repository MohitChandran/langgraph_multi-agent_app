"""Agent memory: the last few exchanges of ONE chat, re-read from Postgres on every turn.

An LLM remembers nothing between requests. "Memory" is us putting the recent conversation
back into the prompt each time (plan 01, section 5):

  - only THIS chat's messages are loaded (WHERE session_id = ...), so chats never mix;
  - only the last N exchanges (CHAT_MEMORY_TURNS, default 3), so turn 50 costs the same
    tokens as turn 4. Sending the whole chat would make every turn more expensive than
    the last and hit Groq's tokens-per-minute limits within a few turns;
  - only question + SQL + a SHORT answer, never the result rows (they'd bloat the prompt);
  - only 'complete' messages, so failed or pending replies don't pollute memory.

The SQL is included on purpose: for a follow-up like "only the ones in India", the
easiest correct move is to take the previous query and add a filter. For Doc agent answers
the cited files and pages are included instead ("Sources1: q3-report.pdf p.2").
"""

from dataclasses import dataclass
from uuid import UUID

from app.chat.store import ChatStore

MAX_ANSWER_CHARS = 300


@dataclass(frozen=True)
class ConversationTurn:
    """One earlier exchange, as the agent will see it."""

    question: str
    answer: str
    sql: str | None  # the SQL that produced the answer, so a follow-up can modify it
    sources: str | None = None  # Doc agent answers: where they came from, e.g. "q3-report.pdf p.2, p.4"


async def load_history(store: ChatStore, session_id: UUID, max_turns: int) -> list[ConversationTurn]:
    """Return the last `max_turns` complete question -> answer exchanges of a chat, oldest first.

    Must be called BEFORE the new question is saved; otherwise the question would
    appear in its own history (plan gotcha #4).

    Reads 2 * max_turns messages (newest first, via the index), reverses them, and pairs
    each user message with the assistant message right after it. A question whose answer
    failed has no complete answer after it, so it's skipped.
    """
    if max_turns <= 0:
        return []
    rows = await store.recent_complete_messages(session_id, limit=max_turns * 2)
    rows.reverse()  # oldest first
    turns = [
        ConversationTurn(
            question=question["content"],
            answer=answer["content"],
            sql=answer["sql"],
            sources=_sources_text(answer["citations"]) if answer["agent"] == "doc" else None,
        )
        for question, answer in zip(rows, rows[1:], strict=False)
        if question["role"] == "user" and answer["role"] == "assistant"
    ]
    return turns[-max_turns:]


def _sources_text(citations: list[dict] | None) -> str | None:
    """'q3-report.pdf p.2, p.4' from a doc answer's citations (unique file+page, in citation order)."""
    if not citations:
        return None
    seen: dict[str, list[int]] = {}
    for citation in citations:
        pages = seen.setdefault(citation["filename"], [])
        if citation["page_start"] not in pages:
            pages.append(citation["page_start"])
    return "; ".join(f"{name} " + ", ".join(f"p.{page}" for page in pages) for name, pages in seen.items())


def format_history(turns: list[ConversationTurn]) -> str:
    """Render turns as the text block placed before "Question:" in the prompts. Empty list -> "".

    Example:
        Conversation so far (most recent last):
        Q1: Which vendors have overdue invoices?
        SQL1: SELECT v.name, ... WHERE i.status = 'overdue' ...
        A1: Amazon Web Services, Lenovo and DHL have overdue invoices ...

    Returning "" for no history means a first question gets exactly the same prompt
    as before memory existed.
    """
    if not turns:
        return ""
    lines = ["Conversation so far (most recent last):"]
    for number, turn in enumerate(turns, start=1):
        lines.append(f"Q{number}: {turn.question}")
        if turn.sql:
            lines.append(f"SQL{number}: {turn.sql}")
        if turn.sources:
            lines.append(f"Sources{number}: {turn.sources}")
        lines.append(f"A{number}: {_shorten(turn.answer)}")
    return "\n".join(lines) + "\n\n"


def _shorten(text: str, limit: int = MAX_ANSWER_CHARS) -> str:
    """Put the text on one line and cut it to `limit` characters (with an ellipsis if cut)."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"
