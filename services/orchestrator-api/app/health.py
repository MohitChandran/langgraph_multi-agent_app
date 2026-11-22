"""Readiness: is every part the API depends on working RIGHT NOW? (GET /api/health/ready)

/api/health only says "the process is up" (liveness). Readiness asks each dependency, concurrently, with a short
timeout, and sums it up:

    ready       every check is fine
    degraded    an OPTIONAL part is down (web search, the email service, tracing, the ingestion worker):
                the API still answers, that feature is just missing for now        -> HTTP 200
    not_ready   a REQUIRED part is down (a database): questions would fail          -> HTTP 503

A check returns a short detail when fine, or raises with the reason. "off" means deliberately disabled (no Langfuse
keys, GUARD_ENABLED=false): it is reported, but it doesn't degrade anything. Docker's healthcheck calls this
endpoint, so one side effect is useful: the email check reconnects to the email service when it's back.
Nothing here calls Groq (that would cost tokens on every probe).
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

CHECK_TIMEOUT_SECONDS = 3.0

Status = Literal["ready", "degraded", "not_ready"]


class Off(Exception):
    """Raised by a check whose feature is deliberately switched off (reported as "off", not as a failure)."""


@dataclass(frozen=True)
class Check:
    """One dependency: its name, whether the API can work without it, and the probe."""

    name: str
    required: bool
    probe: Callable[[], Awaitable[str]]  # returns a short detail; raises when not OK (Off when switched off)


async def run_check(check: Check, timeout: float = CHECK_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Run one probe with a timeout: {"status": ok | off | error, "required", "detail", "ms"}."""
    started = time.perf_counter()
    try:
        detail, status = await asyncio.wait_for(check.probe(), timeout), "ok"
    except Off as off:
        detail, status = str(off), "off"
    except TimeoutError:
        detail, status = f"no answer within {timeout:g} s", "error"
    except Exception as error:  # noqa: BLE001  (any failure is a "not OK", with its reason)
        detail, status = f"{type(error).__name__}: {' '.join(str(error).split())[:200]}", "error"
    ms = round((time.perf_counter() - started) * 1000)
    return {"status": status, "required": check.required, "detail": detail, "ms": ms}


async def readiness(checks: list[Check]) -> tuple[Status, dict[str, dict[str, Any]]]:
    """Run every check concurrently; the overall status and each check's result."""
    results = await asyncio.gather(*(run_check(check) for check in checks))
    by_name = {check.name: result for check, result in zip(checks, results, strict=True)}
    return overall(list(by_name.values())), by_name


def overall(results: list[dict[str, Any]]) -> Status:
    """not_ready if a required check failed, degraded if an optional one failed, else ready ("off" is fine)."""
    failed = [result for result in results if result["status"] == "error"]
    if any(result["required"] for result in failed):
        return "not_ready"
    return "degraded" if failed else "ready"
