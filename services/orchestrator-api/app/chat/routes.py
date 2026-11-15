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
from app.config import get_settings
from app.dependencies import ChatStoreDep, DBAgentDep, DocAgentDep, DocumentStoreDep

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat")
async def chat(
    body: ChatRequest, store: ChatStoreDep, documents: DocumentStoreDep, db_agent: DBAgentDep, doc_agent: DocAgentDep
) -> ChatResponse:
    """Send one message and wait for the answer. Both are saved before this returns.

    Which agent answers (plan 02, decision D12, until the Phase 2 orchestrator replaces it):
    a chat with at least one uploaded document -> the Doc agent; otherwise -> the DB agent.
    New chats (no session_id) go to the DB agent; uploading a file creates its own chat.
    The agent sees the last CHAT_MEMORY_TURNS exchanges of this chat as memory.
    404: the session doesn't exist. 409: this chat is still answering a previous message.
    """
    chat_documents = await documents.list_documents(body.session_id) if body.session_id else []
    agent = doc_agent if chat_documents else db_agent
    route = {"router": "rule", "agent": agent.name, "reason": route_reason([d.filename for d in chat_documents])}
    memory_turns = get_settings().chat_memory_turns
    try:
        return await send_message(store, agent, body.session_id, body.content.strip(), memory_turns, route)
    except SessionNotFoundError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
    except SessionBusyError as error:
        raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error


def route_reason(filenames: list[str]) -> str:
    """Why the message went to this agent, in one sentence (saved with the answer, shown in the UI's flow panel).

    Today the router is a fixed rule; the Phase 2 orchestrator will replace it with an LLM decision
    and its own reasoning ("router": "llm").
    """
    if not filenames:
        return "This chat has no uploaded documents, so the question goes to the Database agent (business data)."
    shown = ", ".join(filenames[:2]) + (f" and {len(filenames) - 2} more" if len(filenames) > 2 else "")
    count = f"{len(filenames)} uploaded document" + ("" if len(filenames) == 1 else "s")
    return f"This chat has {count} ({shown}), so questions here go to the Document agent."


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
