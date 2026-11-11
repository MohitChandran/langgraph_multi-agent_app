"""Data shapes used by the DB agent.

Three kinds of models live here:
  - SQLDraft       what the LLM must return when writing SQL (structured output)
  - DBAgentState   the "memory" LangGraph passes from step to step in the workflow
  - DBAgentResult  what the agent finally returns to its caller (CLI now, orchestrator in Phase 2)
"""

from typing import Any, TypedDict

from pydantic import BaseModel, Field

from app.agents.db.database import QueryResult
from app.agents.db.safety import ValidatedQuery


class SQLDraft(BaseModel):
    """The LLM's answer when asked to write SQL.

    With structured output, Groq forces the model to reply with exactly these
    fields as JSON, and LangChain parses them into this class. No regex or
    "find the ```sql block" parsing of free text, which breaks easily.
    The Field descriptions are sent to the model as part of the JSON schema,
    so they act as extra instructions.
    """

    can_answer: bool = Field(
        description="false if the question cannot be answered from the given tables (e.g. weather, HR salaries)."
    )
    sql: str = Field(description="One PostgreSQL SELECT statement. Empty string if can_answer is false.")
    explanation: str = Field(description="One sentence: what the query does, or why the question cannot be answered.")


class DBAgentState(TypedDict, total=False):
    """Everything the workflow knows at a given moment.

    LangGraph passes this dict to every node; a node returns only the keys it
    changed and LangGraph merges them in. total=False means keys may be missing
    until a node sets them (e.g. `result` only exists after execute_sql).
    """

    question: str  # the user's question (input)
    history: str  # recent conversation of this chat, already formatted for the prompt ("" = none)
    draft: SQLDraft | None  # latest SQL written by the LLM (None if the LLM call failed)
    query: ValidatedQuery  # the draft after it passed the safety gate
    result: QueryResult  # rows returned by Postgres
    error: str | None  # last validation/execution error; sent back to the LLM on retry
    attempts: int  # how many times write_sql has run
    answer: str  # final plain-English answer (output)


class DBAgentResult(BaseModel):
    """The agent's final output: the answer plus everything needed to check it.

    Returning the SQL and rows (not just the answer) lets a user or the UI verify
    the agent's work, which builds trust and makes debugging easy.
    """

    question: str
    answer: str
    success: bool = Field(description="True if a query ran and the answer is based on its rows.")
    sql: str | None = None
    explanation: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    hit_row_limit: bool = False
    attempts: int = 0
    steps: list[str] = Field(default_factory=list, description="Workflow nodes in the order they ran.")
    error: str | None = None
