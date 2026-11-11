"""The DB agent's LangGraph workflow.

A FIXED workflow (not a free-roaming tool-calling agent): the LLM only writes SQL and
the final answer; our code decides the order of steps, so the safety checks can
never be skipped.

    START -> write_sql -> validate_sql -> execute_sql -> answer -> END
                ^              |               |
                '--- retry ----'---------------'   (on error, at most MAX_ATTEMPTS writes)
    write_sql -> answer directly when the question can't be answered from the database.

LangGraph concepts used here:
  - StateGraph(DBAgentState): a graph whose nodes share one state dict.
  - Node: a function that receives the state and returns the keys it changed.
  - Conditional edge: a "router" function that looks at the state and returns
    the name of the next node. That's how the retry loop is built.

Try it:
    uv run python -m app.agents.db.graph "Which pending invoices are due this week?"
    uv run python -m app.agents.db.graph --diagram      # print the graph as a Mermaid diagram
"""

import argparse
import asyncio
import json
import time
from datetime import date
from typing import Literal

import groq
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.agents.db.database import QueryExecutionError, ReadOnlyDatabase
from app.agents.db.models import DBAgentResult, DBAgentState, SQLDraft
from app.agents.db.prompts import ANSWER_PROMPT, REPAIR_SQL_PROMPT, WRITE_SQL_PROMPT
from app.agents.db.safety import SQLValidationError, validate_sql
from app.agents.db.schema import DatabaseSchema, load_schema
from app.chat.memory import ConversationTurn, format_history
from app.config import get_settings
from app.llm import get_llm

MAX_ATTEMPTS = 3  # first try + 2 repairs; more rarely helps and costs tokens and time

# Errors from the LLM call itself that are worth another attempt: API errors that
# remain after the Groq client's own retries, and replies that don't parse into SQLDraft.
LLM_ERRORS = (groq.APIError, OutputParserException, ValidationError)
MAX_ROWS_CHARS = 12_000  # cap on the rows JSON sent to the answer LLM (~3k tokens)
ANSWER_MAX_TOKENS = 500  # answers are a few sentences; also keeps us under Groq's output-tokens-per-minute limit


class DBAgent:
    """Answers questions about the business database by writing, checking and running SQL.

    Usage:
        agent = DBAgent(db, schema)
        result = await agent.ask("Which vendors have overdue invoices?")

    The LLMs can be injected so tests can replace them with fakes (no API calls).
    """

    def __init__(
        self,
        db: ReadOnlyDatabase,
        schema: DatabaseSchema,
        sql_writer: Runnable[list[BaseMessage], SQLDraft] | None = None,
        answer_writer: Runnable[list[BaseMessage], BaseMessage] | None = None,
    ) -> None:
        """Store dependencies and compile the workflow once (compiling is reused across questions).

        Args:
            db: open read-only connection pool.
            schema: reflected schema; turned into prompt text once here, not per question.
            sql_writer: returns an SQLDraft. Default: the strong Groq model with structured
                output (json_schema = Groq constrains the reply to SQLDraft's JSON schema).
            answer_writer: returns a chat message. Default: the fast Groq model, capped at ANSWER_MAX_TOKENS.
        """
        self._db = db
        self._schema_text = schema.to_prompt()
        self._sql_writer = sql_writer or get_llm("strong").with_structured_output(SQLDraft, method="json_schema")
        self._answer_writer = answer_writer or get_llm("fast", max_tokens=ANSWER_MAX_TOKENS)
        self.graph = self._build_graph()

    def _build_graph(self):
        """Wire the nodes and edges together and compile them into a runnable graph."""
        builder = StateGraph(DBAgentState)
        builder.add_node("write_sql", self.write_sql)
        builder.add_node("validate_sql", self.validate_sql)
        builder.add_node("execute_sql", self.execute_sql)
        builder.add_node("answer", self.answer)

        builder.add_edge(START, "write_sql")
        builder.add_conditional_edges("write_sql", self.route_after_write)
        builder.add_conditional_edges("validate_sql", self.route_after_validate)
        builder.add_conditional_edges("execute_sql", self.route_after_execute)
        builder.add_edge("answer", END)
        return builder.compile()

    async def ask(self, question: str, history: list[ConversationTurn] | None = None) -> DBAgentResult:
        """Run the whole workflow for one question and package the final state as a DBAgentResult.

        Args:
            question: the new question.
            history: earlier exchanges of the same chat (memory), oldest first. None or []
                means no memory: the CLI, /api/db/ask and the eval call it like that.
                It's formatted into prompt text ONCE here, not again in every node.

        Uses astream (not ainvoke) so we can also record WHICH nodes ran, in order:
        "updates" events name the node that just finished ({node_name: changes}),
        "values" events carry the full state after each step; the last one is the final state.
        """
        steps: list[str] = []
        state: DBAgentState = {}
        initial: DBAgentState = {"question": question, "history": format_history(history or []), "attempts": 0}
        async for mode, chunk in self.graph.astream(initial, stream_mode=["updates", "values"]):
            if mode == "updates":
                steps.extend(chunk)
            else:
                state = chunk
        draft = state.get("draft")
        result = state.get("result")
        return DBAgentResult(
            question=question,
            answer=state["answer"],
            success=result is not None and not state.get("error"),
            sql=state["query"].sql if "query" in state else (draft.sql if draft else None) or None,
            explanation=draft.explanation if draft else None,
            columns=result.columns if result else [],
            rows=result.rows if result else [],
            hit_row_limit=result.hit_row_limit if result else False,
            attempts=state.get("attempts", 0),
            steps=steps,
            error=state.get("error"),
        )

    # ---------------------------------------------------------------- nodes

    async def write_sql(self, state: DBAgentState) -> DBAgentState:
        """Node: ask the strong LLM for SQL. On a retry, show it its previous SQL and the error.

        This one node is both "generate" and "repair": the only difference is the prompt.
        Feeding back the exact error (e.g. Postgres's 'Perhaps you meant invoices.amount')
        lets the model fix the real problem instead of guessing again.
        """
        attempts = state.get("attempts", 0) + 1
        values = {
            "schema": self._schema_text,
            "today": date.today().isoformat(),
            "history": state.get("history", ""),
            "question": state["question"],
        }
        if state.get("error") and state.get("draft"):
            messages = REPAIR_SQL_PROMPT.format_messages(
                **values, previous_sql=state["draft"].sql, error=state["error"]
            )
        else:
            messages = WRITE_SQL_PROMPT.format_messages(**values)
        try:
            draft = await self._sql_writer.ainvoke(messages)
        except LLM_ERRORS as error:
            # LLM APIs occasionally return empty/invalid output (Groq: 'json_validate_failed').
            # Count it as a failed attempt and let the retry loop try again. draft=None so
            # the next try uses the fresh prompt: this failure wasn't the SQL's fault.
            return {"draft": None, "attempts": attempts, "error": f"The model returned no valid response: {error}"}
        return {"draft": draft, "attempts": attempts, "error": None}

    def validate_sql(self, state: DBAgentState) -> DBAgentState:
        """Node: run the safety gate. Success stores the ValidatedQuery; failure stores the reason."""
        try:
            return {"query": validate_sql(state["draft"].sql), "error": None}
        except SQLValidationError as error:
            return {"error": f"Rejected by safety checks: {error}"}

    async def execute_sql(self, state: DBAgentState) -> DBAgentState:
        """Node: run the validated query on the read-only database. Failure stores Postgres's message."""
        try:
            return {"result": await self._db.run_query(state["query"]), "error": None}
        except QueryExecutionError as error:
            return {"error": f"Database error: {error}"}

    async def answer(self, state: DBAgentState) -> DBAgentState:
        """Node: write the final answer.

        Three cases:
          - the question is out of scope  -> fixed message, no LLM call needed
          - every attempt failed          -> honest failure message with the last error
          - we have rows                  -> the fast LLM summarizes ONLY those rows
        """
        draft = state.get("draft")
        if draft and not draft.can_answer:
            return {"answer": f"I can't answer that from the operations database. {draft.explanation}"}
        if state.get("error") or "result" not in state:
            return {
                "answer": f"Sorry, I couldn't build a working query after {state['attempts']} attempts. "
                f"Last error: {state.get('error')}"
            }

        result = state["result"]
        rows_json = json.dumps(result.rows, ensure_ascii=False)
        if len(rows_json) > MAX_ROWS_CHARS:
            rows_json = rows_json[:MAX_ROWS_CHARS] + " ... [truncated]"
        messages = ANSWER_PROMPT.format_messages(
            history=state.get("history", ""),
            question=state["question"],
            explanation=draft.explanation,
            hit_row_limit=result.hit_row_limit,
            row_count=result.row_count,
            rows=rows_json,
        )
        reply = await self._answer_writer.ainvoke(messages)
        return {"answer": str(reply.content).strip()}

    # -------------------------------------------------------------- routers

    def route_after_write(self, state: DBAgentState) -> Literal["validate_sql", "write_sql", "answer"]:
        """Router: retry if the LLM call failed; skip to the answer if the question is out of scope."""
        if state.get("error"):
            return self._retry_or_give_up(state)
        return "validate_sql" if state["draft"].can_answer else "answer"

    def route_after_validate(self, state: DBAgentState) -> Literal["execute_sql", "write_sql", "answer"]:
        """Router: run the query if it passed the safety gate, otherwise retry or give up."""
        return "execute_sql" if not state.get("error") else self._retry_or_give_up(state)

    def route_after_execute(self, state: DBAgentState) -> Literal["answer", "write_sql"]:
        """Router: answer if the query ran, otherwise retry or give up."""
        return "answer" if not state.get("error") else self._retry_or_give_up(state)

    @staticmethod
    def _retry_or_give_up(state: DBAgentState) -> Literal["write_sql", "answer"]:
        """Go back to write_sql while attempts remain; after MAX_ATTEMPTS, go to answer (which reports the failure)."""
        return "write_sql" if state["attempts"] < MAX_ATTEMPTS else "answer"


async def _main() -> None:
    """CLI: ask one question and print the SQL, the first rows and the answer (or the graph diagram)."""
    parser = argparse.ArgumentParser(description="Ask the DB agent a question.")
    parser.add_argument("question", nargs="*", help="question in plain English")
    parser.add_argument("--diagram", action="store_true", help="print the workflow as a Mermaid diagram")
    args = parser.parse_args()
    question = " ".join(args.question) or "Which pending invoices are due this week?"

    async with ReadOnlyDatabase(get_settings().readonly_database_url) as db:
        agent = DBAgent(db, await load_schema(db))
        if args.diagram:
            print(agent.graph.get_graph().draw_mermaid())
            return
        started = time.perf_counter()
        result = await agent.ask(question)
        elapsed = time.perf_counter() - started

    print(f"Question : {result.question}")
    print(f"SQL      : {result.sql}")
    print(f"Explains : {result.explanation}")
    print(f"Rows     : {len(result.rows)}{' (row limit reached)' if result.hit_row_limit else ''}")
    for row in result.rows[:10]:
        print(f"           {row}")
    if len(result.rows) > 10:
        print(f"           ... {len(result.rows) - 10} more")
    print(f"Attempts : {result.attempts}   Success: {result.success}   Time: {elapsed:.1f}s")
    print(f"\nAnswer:\n{result.answer}")


if __name__ == "__main__":
    asyncio.run(_main())
