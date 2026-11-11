"""Workflow tests for the DB agent (app/agents/db/graph.py), with FAKE LLMs and a FAKE database.

We aren't testing whether the LLM writes good SQL here (that's the eval in step 5).
We're testing OUR logic: does the graph retry on errors, feed the error back,
give up after MAX_ATTEMPTS, skip the database for out-of-scope questions, and
survive a broken LLM reply? Fakes make that fast, free and deterministic.
"""

from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda

from app.agents.db.database import QueryExecutionError, QueryResult
from app.agents.db.graph import MAX_ATTEMPTS, DBAgent
from app.agents.db.models import SQLDraft
from app.agents.db.safety import ValidatedQuery
from app.agents.db.schema import Column, DatabaseSchema, Table

SCHEMA = DatabaseSchema(tables=[Table("vendors", None, [Column("name", "text", False)])])
GOOD_SQL = "SELECT name FROM vendors"


class FakeSQLWriter:
    """Stands in for the LLM that writes SQL: returns scripted replies in order and records every prompt."""

    def __init__(self, *replies: SQLDraft | Exception) -> None:
        """Store the replies to give, one per call. An Exception reply is raised instead of returned."""
        self.replies = list(replies)
        self.prompts: list[list[BaseMessage]] = []

    def __call__(self, messages: list[BaseMessage]) -> SQLDraft:
        """Record the prompt and return (or raise) the next scripted reply."""
        self.prompts.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeDatabase:
    """Stands in for ReadOnlyDatabase: fails the first `failures` queries, then returns one row."""

    def __init__(self, failures: int = 0) -> None:
        """Set how many queries should fail with a Postgres-style error before succeeding."""
        self.failures = failures
        self.queries: list[str] = []

    async def run_query(self, query: ValidatedQuery) -> QueryResult:
        """Record the SQL and either raise a fake Postgres error or return a fixed result."""
        self.queries.append(query.sql)
        if len(self.queries) <= self.failures:
            raise QueryExecutionError('column "nme" does not exist')
        return QueryResult(columns=["name"], rows=[{"name": "Dell Technologies"}], hit_row_limit=False)


def draft(sql: str = GOOD_SQL, can_answer: bool = True) -> SQLDraft:
    """Build an SQLDraft the way the LLM would return it."""
    return SQLDraft(can_answer=can_answer, sql=sql, explanation="test")


def make_agent(writer: FakeSQLWriter, db: Any) -> tuple[DBAgent, list[list[BaseMessage]]]:
    """Build a DBAgent wired to fakes. Returns the agent and the list of prompts the answer LLM received."""
    answer_prompts: list[list[BaseMessage]] = []

    def fake_answer(messages: list[BaseMessage]) -> AIMessage:
        """Fake answer LLM: records the prompt, returns a fixed answer."""
        answer_prompts.append(messages)
        return AIMessage(content="Dell Technologies.")

    agent = DBAgent(db, SCHEMA, sql_writer=RunnableLambda(writer), answer_writer=RunnableLambda(fake_answer))
    return agent, answer_prompts


async def test_happy_path_answers_in_one_attempt() -> None:
    """Good SQL on the first try: one attempt, the row limit is added, and the answer comes from the rows."""
    db = FakeDatabase()
    agent, answer_prompts = make_agent(FakeSQLWriter(draft()), db)

    result = await agent.ask("Which vendors do we have?")

    assert result.success and result.attempts == 1
    assert result.answer == "Dell Technologies."
    assert db.queries == ["SELECT name FROM vendors LIMIT 100"]
    assert "Dell Technologies" in answer_prompts[0][-1].content  # the answer LLM saw the rows


async def test_safety_rejection_is_fed_back_and_repaired() -> None:
    """SELECT * is rejected; the second prompt contains that error and the bad SQL, and the retry succeeds."""
    writer = FakeSQLWriter(draft("SELECT * FROM vendors"), draft())
    db = FakeDatabase()
    agent, _ = make_agent(writer, db)

    result = await agent.ask("Which vendors do we have?")

    assert result.success and result.attempts == 2
    assert result.steps == ["write_sql", "validate_sql", "write_sql", "validate_sql", "execute_sql", "answer"]
    repair_prompt = writer.prompts[1][-1].content
    assert "SELECT * FROM vendors" in repair_prompt and "SELECT * is not allowed" in repair_prompt
    assert len(db.queries) == 1  # the rejected query never reached the database


async def test_database_error_is_fed_back_and_repaired() -> None:
    """A Postgres error (bad column) goes back to the LLM, and the second query works."""
    writer = FakeSQLWriter(draft("SELECT nme FROM vendors"), draft())
    agent, _ = make_agent(writer, FakeDatabase(failures=1))

    result = await agent.ask("Which vendors do we have?")

    assert result.success and result.attempts == 2
    assert 'column "nme" does not exist' in writer.prompts[1][-1].content


async def test_gives_up_after_max_attempts() -> None:
    """If every attempt fails, the agent stops at MAX_ATTEMPTS and answers honestly, without the answer LLM."""
    writer = FakeSQLWriter(*[draft("DROP TABLE vendors")] * MAX_ATTEMPTS)
    db = FakeDatabase()
    agent, answer_prompts = make_agent(writer, db)

    result = await agent.ask("Delete all vendors")

    assert not result.success and result.attempts == MAX_ATTEMPTS
    assert result.answer.startswith("Sorry, I couldn't build a working query")
    assert db.queries == [] and answer_prompts == []


async def test_out_of_scope_question_skips_the_database() -> None:
    """can_answer=false goes straight to a fixed answer: no SQL is run and no answer LLM call is made."""
    db = FakeDatabase()
    agent, answer_prompts = make_agent(FakeSQLWriter(draft("", can_answer=False)), db)

    result = await agent.ask("What's the weather in Chennai?")

    assert not result.success and result.attempts == 1
    assert result.answer.startswith("I can't answer that from the operations database")
    assert db.queries == [] and answer_prompts == []


async def test_broken_llm_reply_is_retried_with_a_fresh_prompt() -> None:
    """If the LLM returns unparsable output, that counts as an attempt and the next try starts fresh."""
    writer = FakeSQLWriter(OutputParserException("empty reply"), draft())
    agent, _ = make_agent(writer, FakeDatabase())

    result = await agent.ask("Which vendors do we have?")

    assert result.success and result.attempts == 2
    assert writer.prompts[1][-1].content == "Question: Which vendors do we have?"  # not the repair prompt
