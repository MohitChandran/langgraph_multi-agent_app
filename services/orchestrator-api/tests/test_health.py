"""Tests for readiness (app/health.py + GET /api/health/ready). No database, no Groq: the checks are fakes."""

import asyncio

import httpx
import pytest

from app.health import Check, Off, overall, readiness, run_check
from app.main import app


def check(name: str, required: bool, result: str | BaseException, delay: float = 0.0) -> Check:
    """A check whose probe returns `result` (a detail) or raises it (an exception), after `delay` seconds."""

    async def probe() -> str:
        """The scripted probe."""
        await asyncio.sleep(delay)
        if isinstance(result, BaseException):
            raise result
        return result

    return Check(name, required, probe)


async def test_a_check_reports_ok_off_error_and_timeout() -> None:
    """ok with its detail; Off -> "off"; an exception -> "error" with type and message; too slow -> "error"."""
    assert (await run_check(check("a", True, "fine")))["status"] == "ok"
    off = await run_check(check("b", False, Off("no keys")))
    assert off["status"] == "off" and off["detail"] == "no keys"
    error = await run_check(check("c", True, ConnectionError("refused")))
    assert error["status"] == "error" and error["detail"] == "ConnectionError: refused"
    slow = await run_check(check("d", False, "late", delay=1.0), timeout=0.1)
    assert slow["status"] == "error" and "no answer within 0.1 s" in slow["detail"]


def test_the_overall_status() -> None:
    """A failed required check -> not_ready; only optional failures -> degraded; "off" never degrades."""
    ok = {"status": "ok", "required": True}
    off = {"status": "off", "required": False}
    optional_down = {"status": "error", "required": False}
    required_down = {"status": "error", "required": True}
    assert overall([ok, off]) == "ready"
    assert overall([ok, optional_down]) == "degraded"
    assert overall([required_down, optional_down]) == "not_ready"


async def test_checks_run_concurrently() -> None:
    """Three 0.3 s checks finish together (~0.3 s), not one after another."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    status, results = await readiness([check(name, False, "ok", delay=0.3) for name in "abc"])
    assert status == "ready" and len(results) == 3 and loop.time() - started < 0.6


@pytest.fixture
async def client() -> httpx.AsyncClient:
    """An HTTP client for the app (no lifespan: each test sets the checks)."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        yield http
    app.state.readiness_checks = []


async def test_the_endpoint_answers_200_when_degraded_and_503_when_not_ready(client: httpx.AsyncClient) -> None:
    """Degraded (web search down) is still 200 with the reason; a database down is 503."""
    app.state.readiness_checks = [check("database", True, "ok"), check("web search", False, RuntimeError("down"))]
    degraded = await client.get("/api/health/ready")
    assert degraded.status_code == 200 and degraded.json()["status"] == "degraded"
    assert degraded.json()["checks"]["web search"]["detail"] == "RuntimeError: down"
    app.state.readiness_checks = [check("database", True, ConnectionError("refused"))]
    down = await client.get("/api/health/ready")
    assert down.status_code == 503 and down.json()["status"] == "not_ready"


async def test_the_worker_reports_whether_its_loop_is_alive() -> None:
    """running: False before start, True while running, False after stop."""
    from app.agents.doc.worker import IngestionWorker

    class EmptyStore:
        """A queue that never has work."""

        async def claim_next_document(self):  # noqa: ANN201
            """No document waiting."""
            return None

    worker = IngestionWorker(EmptyStore(), lambda document: None, data_dir=None, idle_seconds=0.05)
    assert not worker.running
    worker.start()
    await asyncio.sleep(0.01)
    assert worker.running
    await worker.stop()
    assert not worker.running
