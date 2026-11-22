"""FastAPI dependencies: how endpoints get the shared objects built at startup.

Endpoints declare what they need, e.g. `agent: DBAgentDep`, and FastAPI calls these
functions to supply it (dependency injection). Why not read request.app.state directly
in each endpoint? Because tests can then replace a dependency in one line:

    app.dependency_overrides[get_db_agent] = lambda: FakeAgent()

so API tests run without Groq, and without the real lifespan.
"""

from typing import Annotated

from fastapi import Depends, Request

from app.agents.db.graph import DBAgent
from app.agents.doc.store import DocumentStore
from app.chat.store import ChatStore
from app.guardrails.guard import InputGuard
from app.orchestrator.graph import Orchestrator


def get_db_agent(request: Request) -> DBAgent:
    """The DB agent compiled once in the lifespan (main.py)."""
    return request.app.state.db_agent


def get_chat_store(request: Request) -> ChatStore:
    """The chat store opened once in the lifespan (main.py)."""
    return request.app.state.chat_store


def get_orchestrator(request: Request) -> Orchestrator:
    """The orchestrator (ReAct supervisor over all agents) built once in the lifespan (main.py)."""
    return request.app.state.orchestrator


def get_input_guard(request: Request) -> InputGuard:
    """The input guard (NeMo Guardrails) built once in the lifespan (main.py); checks messages before any agent."""
    return request.app.state.input_guard


def get_document_store(request: Request) -> DocumentStore:
    """The document store opened once in the lifespan (main.py)."""
    return request.app.state.document_store


# Annotated aliases keep endpoint signatures short: `store: ChatStoreDep`.
DBAgentDep = Annotated[DBAgent, Depends(get_db_agent)]
ChatStoreDep = Annotated[ChatStore, Depends(get_chat_store)]
DocumentStoreDep = Annotated[DocumentStore, Depends(get_document_store)]
OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator)]
InputGuardDep = Annotated[InputGuard, Depends(get_input_guard)]
