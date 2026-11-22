"""Builds the email service's FastAPI app: the A2A routes, the API-key check, and the startup/shutdown steps.

create_app() takes its parts as arguments (the crew, the mailer, the task store), so tests run the REAL A2A routes
with fakes and an in-memory task store, and the service (app/main.py) runs them with the real ones.

    GET  /.well-known/agent-card.json   the agent card (public: clients read it before calling)
    POST /                              A2A JSON-RPC: SendMessage, GetTask, CancelTask... (needs X-API-Key)
    GET  /health                        liveness (public)
    GET  /health/ready                  readiness: task store, SMTP server, tracing (public; app/health.py)
"""

import asyncio
import hmac
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import DatabaseTaskStore, TaskStore
from fastapi import FastAPI, Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.agent import EmailCrew
from app.card import API_KEY_HEADER, build_agent_card
from app.config import Settings, get_settings
from app.health import Check, Off, readiness
from app.hitl import Drafter, EmailExecutor, Mailer
from app.mailer import SmtpMailer
from app.store import build_task_store, create_engine
from app.tracing import get_tracing, setup_tracing

logger = logging.getLogger(__name__)

# A paused task (waiting for a human) is dropped from memory after 10 idle minutes; it stays in the task store and
# is loaded again when the decision arrives (plan 05, D5).
INTERRUPTED_TASK_TIMEOUT_SECONDS = 600
# Readable without the key: the card (clients need it to learn how to call) and the health check. /docs and
# /openapi.json only DESCRIBE the API; calling it still needs the key.
PUBLIC_PATHS = frozenset({"/health", "/health/ready", "/.well-known/agent-card.json", "/docs", "/openapi.json"})


class ApiKeyMiddleware:
    """Refuses every request without the right X-API-Key header (401), except GETs of the public paths.

    Plain ASGI middleware: it runs before routing, so a request without the key never reaches the A2A handler
    and no task is created. hmac.compare_digest takes the same time however much of the key matches, so the key
    can't be guessed character by character from response times.
    """

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        """Wrap the app; remember the key."""
        self.app = app
        self.api_key = api_key.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Let public GETs and non-HTTP traffic (lifespan) through; check the key on everything else."""
        if scope["type"] != "http" or (scope["method"] in ("GET", "HEAD") and scope["path"] in PUBLIC_PATHS):
            await self.app(scope, receive, send)
            return
        given = Headers(scope=scope).get(API_KEY_HEADER, "").encode()
        if not hmac.compare_digest(given, self.api_key):
            response = JSONResponse({"error": f"missing or wrong {API_KEY_HEADER} header"}, status_code=401)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def create_app(
    *,
    settings: Settings | None = None,
    drafter: Drafter | None = None,
    mailer: Mailer | None = None,
    task_store: TaskStore | None = None,
) -> FastAPI:
    """The email service's app. Omitted parts are the real ones: the CrewAI crew, the SMTP mailer, the task store
    in Postgres."""

    settings = settings or get_settings()
    engine = None
    if task_store is None:
        engine = create_engine(settings)  # no connection yet
        task_store = build_task_store(engine)
    card = build_agent_card(settings)
    executor = EmailExecutor(drafter or EmailCrew(settings), mailer or SmtpMailer(settings), settings)
    handler = DefaultRequestHandlerV2(
        agent_executor=executor,
        task_store=task_store,
        agent_card=card,
        interrupted_task_timeout=INTERRUPTED_TASK_TIMEOUT_SECONDS,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Startup: tracing, then email.tasks now (a wrong password or missing schema fails here).
        Shutdown: close the pool, then send the last spans."""
        tracing = setup_tracing(settings)
        try:
            if isinstance(task_store, DatabaseTaskStore):
                await task_store.initialize()
                logger.info("Task store ready (email.tasks)")
            logger.info("Email service ready at %s (SMTP %s:%d)", settings.email_service_public_url,
                        settings.smtp_host, settings.smtp_port)  # fmt: skip
            yield
        finally:
            if engine is not None:
                await engine.dispose()
            await asyncio.to_thread(tracing.shutdown)

    app = FastAPI(title="Email A2A Service", version="0.1.0", lifespan=lifespan)
    app.add_middleware(ApiKeyMiddleware, api_key=settings.email_service_api_key.get_secret_value())
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url="/"),
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness check (public: no API key needed)."""
        return {"status": "ok"}

    checks = _readiness_checks(settings, engine)

    @app.get("/health/ready")
    async def ready(response: Response) -> dict:
        """Readiness: 200 when ready or degraded, 503 when the task store is down."""
        status, results = await readiness(checks)
        if status == "not_ready":
            response.status_code = 503
        return {"status": status, "checks": results}

    return app


def _readiness_checks(settings: Settings, engine: AsyncEngine | None) -> list[Check]:
    """What GET /health/ready probes: the task store (required), the SMTP server and tracing (optional)."""

    async def task_store() -> str:
        """Postgres answers as email_rw (in-memory store in tests: nothing to probe)."""
        if engine is None:
            return "in memory"
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return "email.tasks (Postgres)"

    async def smtp_server() -> str:
        """The SMTP server accepts a TCP connection (nothing is sent, no login)."""
        _, writer = await asyncio.open_connection(settings.smtp_host, settings.smtp_port)
        writer.close()
        await writer.wait_closed()
        return f"{settings.smtp_host}:{settings.smtp_port}"

    async def tracing() -> str:
        """Langfuse accepts the keys (a blocking HTTP call: in a thread)."""
        client = get_tracing().client
        if client is None:
            raise Off("no Langfuse keys")
        if not await asyncio.to_thread(client.auth_check):
            raise RuntimeError("Langfuse rejected the keys")
        return settings.langfuse_base_url

    return [
        Check("task store", True, task_store),
        Check("smtp server", False, smtp_server),
        Check("tracing (Langfuse)", False, tracing),
    ]
