"""Tests for the per-client request limits (app/ratelimit.py). A fake clock; no database, no Groq."""

import httpx
import pytest

from app.main import app
from app.ratelimit import RateLimiter, Rule, refusal


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        """Start at 1000 s."""
        self.now = 1000.0

    def __call__(self) -> float:
        """The current time."""
        return self.now


def limiter(clock: Clock, limit: int = 3, window: float = 60) -> RateLimiter:
    """A limiter with one rule for group "questions"."""
    return RateLimiter({"questions": [Rule(limit, window, "questions", "minute")]}, clock)


def test_the_window_slides() -> None:
    """3 per minute: the 4th is refused with the wait until the oldest leaves the window; then allowed again."""
    clock = Clock()
    limits = limiter(clock)
    for second in (0, 10, 20):
        clock.now = 1000 + second
        assert limits.hit("1.2.3.4", "questions") is None
    clock.now = 1030
    rule, wait = limits.hit("1.2.3.4", "questions")
    assert rule.limit == 3 and wait == pytest.approx(30)
    clock.now = 1060.5  # the request at 1000 has left the window
    assert limits.hit("1.2.3.4", "questions") is None


def test_clients_and_groups_are_counted_separately() -> None:
    """Another address has its own budget; a group without rules is never limited."""
    clock = Clock()
    limits = limiter(clock, limit=1)
    assert limits.hit("1.1.1.1", "questions") is None
    assert limits.hit("1.1.1.1", "questions") is not None
    assert limits.hit("2.2.2.2", "questions") is None
    assert limits.hit("1.1.1.1", "uploads") is None


def test_refused_requests_do_not_count() -> None:
    """Retrying while refused doesn't extend the wait: the window only holds allowed requests."""
    clock = Clock()
    limits = limiter(clock, limit=1)
    limits.hit("c", "questions")
    for _ in range(5):
        clock.now += 10
        limits.hit("c", "questions")
    clock.now = 1060.1
    assert limits.hit("c", "questions") is None


def test_every_rule_must_allow_and_a_refusal_is_readable() -> None:
    """Per-minute AND per-day: the day rule refuses even when the minute rule allows; the 429 says why and when."""
    clock = Clock()
    limits = RateLimiter(
        {"questions": [Rule(5, 60, "questions", "minute"), Rule(2, 86_400, "questions", "day")]}, clock
    )
    limits.hit("c", "questions")
    clock.now += 120
    limits.hit("c", "questions")
    clock.now += 120
    rule, wait = limits.hit("c", "questions")
    assert rule.window_text == "day"
    error = refusal(rule, wait)
    assert error.status_code == 429 and int(error.headers["Retry-After"]) == pytest.approx(86_400 - 240, abs=1)
    assert error.detail.startswith("Too many questions: at most 2 per day. Please try again in")
    assert refusal(Rule(6, 60, "questions", "minute"), 4.2).detail.endswith("try again in 5 s.")


def test_quiet_clients_are_forgotten() -> None:
    """The sweep drops clients with nothing inside any window (memory stays bounded)."""
    clock = Clock()
    limits = limiter(clock)
    limits.hit("old", "questions")
    clock.now += 61
    limits._sweep(clock.now)
    assert not limits._hits


def test_zero_switches_a_rule_off() -> None:
    """from_settings drops rules with limit 0."""
    from app.config import get_settings

    settings = get_settings().model_copy(update={"rate_limit_questions_per_day": 0, "rate_limit_uploads_per_hour": 0})
    limits = RateLimiter.from_settings(settings)
    assert [rule.window_text for rule in limits.groups["questions"]] == ["minute"] and limits.groups["uploads"] == []


async def test_the_api_answers_429_with_retry_after() -> None:
    """Through the real route (/api/db/ask): over the limit -> 429 + Retry-After, before the guard or agent runs."""
    clock = Clock()
    app.state.rate_limiter = RateLimiter({"questions": [Rule(1, 60, "questions", "minute")]}, clock)
    app.state.rate_limiter.hit("127.0.0.1", "questions")  # this client's one question is used up
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/db/ask", json={"question": "How many vendors?"})
        assert response.status_code == 429 and response.headers["Retry-After"] == "60"
        assert response.json()["detail"].startswith("Too many questions: at most 1 per minute")
    finally:
        app.state.rate_limiter = None
