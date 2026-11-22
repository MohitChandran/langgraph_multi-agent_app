"""Request rate limiting: how often ONE client may call the expensive endpoints (Phase 8 hardening).

Every question costs Groq tokens, and the free tier's daily budget is shared by everyone using this API: one client
(a script, a stuck browser tab) could otherwise spend it all. So each client gets its own limits per group of
endpoints:

    questions   POST /api/chat, POST /api/messages/{id}/email-approval, POST /api/db/ask
                (each runs the guard and agents)              RATE_LIMIT_QUESTIONS_PER_MINUTE / _PER_DAY
    uploads     POST /api/documents (parsing + vision model)  RATE_LIMIT_UPLOADS_PER_HOUR

How: a SLIDING WINDOW per (client, rule): the times of the client's recent requests; a request is allowed when
fewer than `limit` of them fall inside the window. A refused request gets 429 with Retry-After = when the oldest
one leaves the window, and a sentence the UI shows as is. Every request counts, including ones that then fail
(e.g. 409): the limit is on hammering the API, not on successful answers.

The client is request.client.host: the browser's address (behind the Docker frontend, uvicorn's --proxy-headers
puts the real address there). Counters live in this process's memory: right for one uvicorn worker; several
workers or replicas would each count separately (a shared store such as Redis would be the next step).
"""

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import HTTPException, Request

from app.config import Settings

SWEEP_EVERY = 1_000  # requests between clean-ups of clients that went quiet


@dataclass(frozen=True)
class Rule:
    """At most `limit` requests per `window_seconds`; `what` names them in the refusal ("questions")."""

    limit: int
    window_seconds: float
    what: str
    window_text: str  # "minute", "day"...


class RateLimiter:
    """Sliding-window counters per client and rule, in memory."""

    def __init__(self, groups: dict[str, list[Rule]], clock: Callable[[], float] = time.monotonic) -> None:
        """groups: endpoint group -> its rules. clock: injectable for tests."""
        self.groups = groups
        self.clock = clock
        self._hits: dict[tuple[str, str, int], deque[float]] = {}
        self._requests = 0

    @classmethod
    def from_settings(cls, settings: Settings) -> "RateLimiter":
        """The limits from RATE_LIMIT_* (a limit of 0 switches that rule off)."""
        questions = [
            Rule(settings.rate_limit_questions_per_minute, 60, "questions", "minute"),
            Rule(settings.rate_limit_questions_per_day, 86_400, "questions", "day"),
        ]
        uploads = [Rule(settings.rate_limit_uploads_per_hour, 3_600, "uploads", "hour")]
        return cls({
            "questions": [rule for rule in questions if rule.limit > 0],
            "uploads": [rule for rule in uploads if rule.limit > 0],
        })  # fmt: skip

    def hit(self, client: str, group: str) -> tuple[Rule, float] | None:
        """Record one request of `client` in `group`, or refuse it: (the rule it broke, seconds until allowed).

        A refused request is NOT recorded, so a client that keeps retrying doesn't push its own window further.
        """
        now = self.clock()
        self._requests += 1
        if self._requests % SWEEP_EVERY == 0:
            self._sweep(now)
        rules = self.groups.get(group, [])
        windows = []
        for index, rule in enumerate(rules):
            times = self._hits.setdefault((client, group, index), deque())
            while times and times[0] <= now - rule.window_seconds:
                times.popleft()
            if len(times) >= rule.limit:
                return rule, times[0] + rule.window_seconds - now
            windows.append(times)
        for times in windows:
            times.append(now)
        return None

    def _sweep(self, now: float) -> None:
        """Forget clients with no request inside any window (keeps memory bounded)."""
        longest = max((rule.window_seconds for rules in self.groups.values() for rule in rules), default=0)
        for key in [key for key, times in self._hits.items() if not times or times[-1] <= now - longest]:
            del self._hits[key]


def refusal(rule: Rule, retry_after: float) -> HTTPException:
    """The 429 for a broken rule: a sentence for the UI and Retry-After (whole seconds, at least 1)."""
    seconds = max(1, math.ceil(retry_after))
    wait = f"{seconds} s" if seconds < 120 else f"{math.ceil(seconds / 60)} min"
    detail = f"Too many {rule.what}: at most {rule.limit} per {rule.window_text}. Please try again in {wait}."
    return HTTPException(status_code=429, detail=detail, headers={"Retry-After": str(seconds)})


def rate_limit(group: str) -> Callable[[Request], None]:
    """A FastAPI dependency for a route of `group`: raises 429 when this client is over a limit.

    Uses app.state.rate_limiter (built at startup); without one (tests without the lifespan, RATE_LIMIT_ENABLED=false)
    nothing is limited.
    """

    def check(request: Request) -> None:
        """Count this request for its client; refuse it when over a limit."""
        limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
        if limiter is None:
            return
        client = request.client.host if request.client else "unknown"
        refused = limiter.hit(client, group)
        if refused is not None:
            raise refusal(*refused)

    return check
