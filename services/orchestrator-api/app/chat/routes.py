"""HTTP endpoints for chat sessions (plan 01, section 7; plan 03, section 11.2).

    POST   /api/chat                              send a message (new chat if session_id is null)
    POST   /api/messages/{id}/email-approval      approve / edit / reject the email draft of an answer
    GET    /api/sessions                          sidebar: chats, most recently active first
    GET    /api/sessions/{id}/messages            transcript, oldest to newest (?limit, ?before)
    PATCH  /api/sessions/{id}                     rename
    DELETE /api/sessions/{id}                     delete (messages go with it)

This layer only does HTTP: validation (Pydantic, automatic 422), status codes, and
turning the service's exceptions into 404/409. The logic lives in service.py and store.py.
Path ids are typed (UUID, int), so FastAPI answers 422 for a malformed id before our code runs.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response, status

from app.chat.models import ApprovalResponse, ChatRequest, ChatResponse, MessagePage, RenameRequest, Session
from app.chat.service import (
    ApprovalPendingError,
    MessageNotFoundError,
    NotAwaitingApprovalError,
    SessionBusyError,
    SessionNotFoundError,
    resolve_email_approval,
    send_message,
)
from app.config import get_settings
from app.dependencies import ChatStoreDep, OrchestratorDep
from app.orchestrator.models import EmailDecision

router = APIRouter(prefix="/api", tags=["chat"])

# Service exceptions -> HTTP status codes (shared by both endpoints that run the orchestrator).
ERROR_STATUS = {
    SessionNotFoundError: status.HTTP_404_NOT_FOUND,
    MessageNotFoundError: status.HTTP_404_NOT_FOUND,
    SessionBusyError: status.HTTP_409_CONFLICT,
    ApprovalPendingError: status.HTTP_409_CONFLICT,
    NotAwaitingApprovalError: status.HTTP_409_CONFLICT,
}


@router.post("/chat")
async def chat(body: ChatRequest, store: ChatStoreDep, orchestrator: OrchestratorDep) -> ChatResponse:
    """Send one message and wait for the answer. Both are saved before this returns.

    The orchestrator (plan 03) decides which agents to ask: the database, this chat's documents, the
    web, and/or an email draft. The agents see the last CHAT_MEMORY_TURNS exchanges of this chat.
    If it drafts an email, the answer comes back with status "awaiting_approval": decide with
    POST /api/messages/{id}/email-approval.
    404: the session doesn't exist. 409: this chat is still answering, or an email waits for approval.
    """
    memory_turns = get_settings().chat_memory_turns
    try:
        return await send_message(store, orchestrator, body.session_id, body.content.strip(), memory_turns)
    except tuple(ERROR_STATUS) as error:
        raise HTTPException(ERROR_STATUS[type(error)], str(error)) from error


@router.post("/messages/{message_id}/email-approval")
async def email_approval(
    message_id: int, body: EmailDecision, store: ChatStoreDep, orchestrator: OrchestratorDep
) -> ApprovalResponse:
    """Approve (optionally with an edited body) or reject the email draft of an answer.

    The paused orchestrator run continues: it sends or discards the draft and writes the final
    answer, which replaces the "awaiting approval" message. Nothing is ever sent without this call.
    404: no such message. 409: the message isn't waiting for a decision, or the chat is busy.
    422: invalid body (e.g. `approved` missing, or edited_body over 5,000 characters).
    """
    try:
        return await resolve_email_approval(store, orchestrator, message_id, body)
    except tuple(ERROR_STATUS) as error:
        raise HTTPException(ERROR_STATUS[type(error)], str(error)) from error


@router.get("/sessions")
async def list_sessions(store: ChatStoreDep) -> list[Session]:
    """All chats for the sidebar, most recently active first (max 100)."""
    return await store.list_sessions()


@router.get("/sessions/{session_id}/messages")
async def list_messages(
    session_id: UUID,
    store: ChatStoreDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    before: Annotated[int | None, Query(description="Only messages older than this message id.")] = None,
) -> MessagePage:
    """A chat's messages, oldest to newest. has_more=True means older ones exist: pass before=<first id>."""
    if await store.get_session(session_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Chat {session_id} does not exist.")
    return await store.list_messages(session_id, limit=limit, before=before)


@router.patch("/sessions/{session_id}")
async def rename_session(session_id: UUID, body: RenameRequest, store: ChatStoreDep) -> Session:
    """Rename a chat."""
    session = await store.rename_session(session_id, body.title.strip())
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Chat {session_id} does not exist.")
    return session


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: UUID, store: ChatStoreDep) -> Response:
    """Delete a chat and all its messages. 204 = done, no body."""
    if not await store.delete_session(session_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Chat {session_id} does not exist.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
