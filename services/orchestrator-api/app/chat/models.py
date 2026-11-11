"""Pydantic models for the chat API (plan 01, section 7).

Two kinds:
  - response models (Session, Message, ChatResponse, MessagePage): what the API returns.
    They're built straight from database rows, so their fields match the table columns.
  - request models (ChatRequest, RenameRequest): what the API accepts. FastAPI validates
    them and answers 422 with a clear message if a field is missing or too long.
"""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class Session(BaseModel):
    """One conversation: one item in the sidebar."""

    id: UUID
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime = Field(description="Last activity; the sidebar is sorted by it, newest first.")
    is_running: bool = Field(description="True while a reply is being generated in this chat.")


class Message(BaseModel):
    """One message in a conversation: a question or an answer."""

    id: int
    role: Literal["user", "assistant"]
    content: str = Field(description="The question, or the final answer as markdown.")
    status: Literal["pending", "complete", "error"]
    payload: dict[str, Any] = Field(description='Assistant extras, e.g. {"agent": "db", "result": {...}}.')
    latency_ms: int | None
    created_at: datetime


class ChatRequest(BaseModel):
    """Body of POST /api/chat: one new message, in a new chat (session_id null) or an existing one."""

    session_id: UUID | None = Field(default=None, description="null starts a new chat.")
    content: str = Field(min_length=1, max_length=1_000)


class ChatResponse(BaseModel):
    """Result of one turn: the (possibly new) session and the two messages that were saved."""

    session: Session
    user_message: Message
    assistant_message: Message


class MessagePage(BaseModel):
    """A page of messages, oldest to newest. has_more=True means older messages exist (use `before`)."""

    items: list[Message]
    has_more: bool


class RenameRequest(BaseModel):
    """Body of PATCH /api/sessions/{id}."""

    title: str = Field(min_length=1, max_length=100)
