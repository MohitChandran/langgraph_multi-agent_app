"""HTTP endpoints for chat sessions (plan 01, section 7).

    POST   /api/chat                          send a message (new chat if session_id is null)
    GET    /api/sessions                      sidebar: chats, most recently active first
    GET    /api/sessions/{id}/messages        transcript, oldest to newest (?limit, ?before)
    PATCH  /api/sessions/{id}                 rename
    DELETE /api/sessions/{id}                 delete (messages go with it)

This layer only does HTTP: validation (Pydantic, automatic 422), status codes, and
turning the service's exceptions into 404/409. The logic lives in service.py and store.py.
Path ids are typed `UUID`, so FastAPI answers 422 for a malformed id before our code runs.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response, status

from app.chat.models import ChatRequest, ChatResponse, MessagePage, RenameRequest, Session
from app.chat.service import SessionBusyError, SessionNotFoundError, send_message
from app.dependencies import ChatStoreDep, DBAgentDep

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat")
async def chat(body: ChatRequest, store: ChatStoreDep, agent: DBAgentDep) -> ChatResponse:
    """Send one message and wait for the answer. Both are saved before this returns.

    404: the session doesn't exist. 409: this chat is still answering a previous message.
    """
    try:
        return await send_message(store, agent, body.session_id, body.content.strip())
    except SessionNotFoundError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
    except SessionBusyError as error:
        raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error


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
