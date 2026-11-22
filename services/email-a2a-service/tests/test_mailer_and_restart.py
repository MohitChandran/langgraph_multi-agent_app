"""Integration tests for stage E3 (plan 05): the real SMTP mailer (to Mailpit) and approvals that survive a restart.

Both need docker compose running (Postgres + Mailpit) and this service's .env; otherwise they are skipped.
"""

from collections.abc import AsyncIterator
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import text

from app.client_demo import artifact, make_client, send_data, state
from app.config import Settings
from app.fakes import FakeDrafter, FakeMailer
from app.mailer import TASK_ID_HEADER, SmtpMailer
from app.server import create_app
from app.store import create_engine
from tests.conftest import settings_or_skip

MAILPIT_API = "http://localhost:8025/api/v1"


@pytest.fixture
async def mailpit() -> AsyncIterator[httpx.AsyncClient]:
    """An HTTP client for Mailpit's API; skips when Mailpit isn't running."""
    async with httpx.AsyncClient(base_url=MAILPIT_API, timeout=10) as client:
        try:
            await client.get("/info")
        except httpx.HTTPError:
            pytest.skip("Mailpit not available")
        yield client


@pytest.mark.integration
async def test_the_mailer_delivers_one_email_with_the_task_id(mailpit: httpx.AsyncClient) -> None:
    """SmtpMailer -> Mailpit: the email arrives with our subject, recipient, Message-ID and X-A2A-Task-Id header."""
    settings = settings_or_skip().model_copy(update={"smtp_host": "localhost", "smtp_port": 1025,
                                                     "smtp_starttls": False, "smtp_use_tls": False})  # fmt: skip
    token = uuid4().hex[:8]
    receipt = await SmtpMailer(settings).send(
        to="accounts@company.example", subject=f"[pytest {token}] Pending invoices", body="Hello", task_id=token
    )
    found = (await mailpit.get("/search", params={"query": f"subject:{token}"})).json()["messages"]
    try:
        [message] = found
        assert message["To"][0]["Address"] == "accounts@company.example"
        assert f"<{message['MessageID']}>" == receipt.message_id
        headers = (await mailpit.get(f"/message/{message['ID']}/headers")).json()
        # Header names are case-insensitive (RFC 5322); Mailpit (Go) returns them as "X-A2a-Task-Id".
        by_name = {name.lower(): values for name, values in headers.items()}
        assert by_name[TASK_ID_HEADER.lower()] == [token]
    finally:
        await mailpit.request("DELETE", "/messages", json={"IDs": [m["ID"] for m in found]})


async def start_service(settings: Settings, mailer: FakeMailer) -> tuple:
    """A fresh app (its own engine and DatabaseTaskStore, like a new process) and an A2A client for it."""
    app = create_app(settings=settings, drafter=FakeDrafter(), mailer=mailer)
    return app, httpx.ASGITransport(app=app)


@pytest.mark.integration
async def test_an_approval_survives_a_restart_of_the_service(engine) -> None:  # noqa: ANN001  (skips without Postgres)
    """#1 on one app instance -> shut it down -> a NEW instance (new pool, new store) -> #2 approve completes."""
    # tracing off: tests never send spans to Langfuse
    settings = settings_or_skip().model_copy(
        update={"email_service_public_url": "http://test", "langfuse_public_key": ""}
    )
    key = {"X-API-Key": settings.email_service_api_key.get_secret_value()}
    mailer = FakeMailer()
    task_ids: list[str] = []
    try:
        app, transport = await start_service(settings, mailer)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test", headers=key) as http:
                client = await make_client("http://test", http)
                task = await send_data(client, {"to": "accounts@company.example", "subject": "Restart test",
                                                "content": "8 invoices, 73,400 USD."})  # fmt: skip
                task_ids.append(task.id)
                assert state(task) == "INPUT_REQUIRED"
        # The first instance is gone (its pool closed). Only Postgres remembers the paused task.
        app, transport = await start_service(settings, mailer)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test", headers=key) as http:
                client = await make_client("http://test", http)
                done = await send_data(client, {"decision": "approve"}, task.id, task.context_id)
        assert state(done) == "COMPLETED" and artifact(done, "receipt") is not None
        assert len(mailer.sent) == 1 and mailer.sent[0]["subject"] == "Restart test"
    finally:
        cleanup = create_engine(settings)
        async with cleanup.begin() as conn:
            await conn.execute(text("DELETE FROM tasks WHERE id = ANY(:ids)"), {"ids": task_ids})
        await cleanup.dispose()
