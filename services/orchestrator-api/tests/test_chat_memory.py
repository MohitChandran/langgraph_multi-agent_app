"""Unit tests for agent memory (plan 01, stage S3): formatting, and what actually reaches the prompts.

The window / isolation / error-exclusion rules are tested end to end in test_chat_api.py.
"""

from app.chat.memory import MAX_ANSWER_CHARS, ConversationTurn, format_history
from tests.test_db_agent import FakeDatabase, FakeSQLWriter, draft, make_agent

TURN = ConversationTurn(
    question="Which vendors have overdue invoices?",
    answer="Amazon Web Services and Lenovo have overdue invoices.",
    sql="SELECT v.name FROM invoices AS i JOIN vendors AS v ON v.id = i.vendor_id WHERE i.status = 'overdue'",
)


def test_no_history_formats_to_empty_string() -> None:
    """A first question gets no history block at all, so its prompt is exactly as before memory."""
    assert format_history([]) == ""


def test_history_block_numbers_questions_sql_and_answers() -> None:
    """Each turn shows Q, SQL and A with matching numbers, most recent last."""
    second = ConversationTurn(question="only the ones in India", answer="Infosys.", sql=None)
    text = format_history([TURN, second])
    assert text.startswith("Conversation so far (most recent last):\n")
    assert f"Q1: {TURN.question}\nSQL1: {TURN.sql}\nA1: {TURN.answer}\n" in text
    assert "Q2: only the ones in India\nA2: Infosys." in text  # no SQL line when there was no SQL
    assert text.endswith("\n\n")  # blank line before "Question:"


def test_long_answers_are_cut_to_keep_the_prompt_small() -> None:
    """A long (multi-line, markdown) answer becomes one line of at most MAX_ANSWER_CHARS characters."""
    long_answer = "* " + "\n* ".join(f"Vendor number {n} owes money" for n in range(60))
    text = format_history([ConversationTurn(question="q", answer=long_answer, sql=None)])
    answer_line = text.splitlines()[2]
    assert answer_line.startswith("A1: ") and answer_line.endswith("…")
    assert MAX_ANSWER_CHARS - 5 <= len(answer_line) - len("A1: ") <= MAX_ANSWER_CHARS
    assert "\n*" not in answer_line


async def test_history_reaches_both_the_sql_prompt_and_the_answer_prompt() -> None:
    """The DB agent puts the history (with the previous SQL) before the question in both LLM calls."""
    writer = FakeSQLWriter(draft())
    agent, answer_prompts = make_agent(writer, FakeDatabase())

    await agent.ask("only the ones in India", history=[TURN])

    sql_prompt = writer.prompts[0][-1].content
    assert sql_prompt.startswith("Conversation so far")
    assert f"SQL1: {TURN.sql}" in sql_prompt
    assert sql_prompt.endswith("Question: only the ones in India")
    assert "Q1: Which vendors have overdue invoices?" in answer_prompts[0][-1].content
