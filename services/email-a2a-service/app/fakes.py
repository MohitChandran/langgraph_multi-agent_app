"""Stand-ins for the crew and the mailer: instant, free and predictable (tests, and the service until E3/E4).

FakeDrafter writes a fixed, polite draft from the request (no LLM). FakeMailer "sends" by recording the email in
memory and returning a receipt (no SMTP). Both can be told to fail, to test the FAILED paths.
"""

import asyncio
from datetime import UTC, datetime
from email.utils import make_msgid

from app.models import CrewDraft, EmailRequest, Receipt


class FakeDrafter:
    """A crew stand-in: the draft repeats the content between a greeting and a sign-off."""

    def __init__(self, *, fail: bool = False, delay_seconds: float = 0.0, issues: list[str] | None = None) -> None:
        """fail=True raises; delay_seconds > the crew timeout simulates a hang; issues = reviewer notes to return."""
        self.fail = fail
        self.delay_seconds = delay_seconds
        self.issues = issues or []
        self.requests: list[EmailRequest] = []

    async def draft(self, request: EmailRequest) -> CrewDraft:
        """Record the request and return the fixed draft (or fail / stall as configured)."""
        self.requests.append(request)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.fail:
            raise RuntimeError("Groq is unreachable")
        body = f"Hello,\n\n{request.content}\n\nBest regards,\nOps Assistant"
        return CrewDraft(subject=request.subject, body=body, issues=self.issues)


class FakeMailer:
    """An SMTP stand-in: `sent` lists every email "sent", so tests can check it was exactly one (or none)."""

    def __init__(self, *, fail: bool = False) -> None:
        """fail=True raises like a refused SMTP connection."""
        self.fail = fail
        self.sent: list[dict[str, str]] = []

    async def send(self, *, to: str, subject: str, body: str, task_id: str) -> Receipt:
        """Record the email and return a receipt with a real-looking Message-ID."""
        if self.fail:
            raise ConnectionRefusedError("SMTP server refused the connection")
        self.sent.append({"to": to, "subject": subject, "body": body, "task_id": task_id})
        return Receipt(message_id=make_msgid(domain="ops-assistant"), sent_at=datetime.now(UTC), to=to, subject=subject)
