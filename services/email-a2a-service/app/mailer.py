"""Sending the approved email over SMTP (plan 05, section 6.5).

Only EmailExecutor._decide calls this, and only after a human approved the draft (app/hitl.py).

Development: Mailpit (docker compose) on localhost:1025 catches every email; see them at http://localhost:8025.
A real mailbox later: SMTP_HOST/PORT/USERNAME/PASSWORD + SMTP_STARTTLS (port 587) or SMTP_USE_TLS (port 465) in
this service's .env; nothing in the code changes.

Try it (from services/email-a2a-service; sends ONE test email with the current settings):
    uv run python -m app.mailer accounts@company.example
"""

import argparse
import asyncio
from datetime import UTC, datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

import aiosmtplib

from app.checks import recipient_allowed
from app.config import Settings, get_settings
from app.models import Receipt

TASK_ID_HEADER = "X-A2A-Task-Id"


class SmtpMailer:
    """Sends one plain-text email per call with aiosmtplib (async: the event loop isn't blocked while sending)."""

    def __init__(self, settings: Settings) -> None:
        """Keep the SMTP settings."""
        self.settings = settings

    def build(self, *, to: str, subject: str, body: str, task_id: str) -> EmailMessage:
        """The email: From/To/Subject/Date, our own Message-ID, the A2A task id (traceability), a plain-text body.

        Python's EmailMessage refuses header values with line breaks, so a subject can't inject extra headers.
        """
        message = EmailMessage()
        message["From"] = self.settings.email_from
        message["To"] = to
        message["Subject"] = subject
        message["Date"] = formatdate(localtime=True)
        message["Message-ID"] = make_msgid(domain="ops-assistant")
        message[TASK_ID_HEADER] = task_id  # which A2A task (and so which chat approval) sent this email
        message.set_content(body)
        return message

    async def send(self, *, to: str, subject: str, body: str, task_id: str) -> Receipt:
        """Send it; raise on any SMTP error (the executor turns that into a FAILED task)."""
        s = self.settings
        message = self.build(to=to, subject=subject, body=body, task_id=task_id)
        await aiosmtplib.send(
            message,
            hostname=s.smtp_host,
            port=s.smtp_port,
            username=s.smtp_username or None,
            password=s.smtp_password.get_secret_value() or None,
            start_tls=s.smtp_starttls,  # explicit True/False: never "upgrade if the server offers it" by surprise
            use_tls=s.smtp_use_tls,
            timeout=s.smtp_timeout_seconds,
        )
        return Receipt(message_id=message["Message-ID"], sent_at=datetime.now(UTC), to=to, subject=subject)


async def _main(to: str) -> None:
    """CLI: send one test email with the configured SMTP settings and print the receipt (allow-list applies)."""
    settings = get_settings()
    if not settings.email_allow_any_domain and not recipient_allowed(to, settings.allowed_recipients):
        raise SystemExit(f"{to} is not on EMAIL_ALLOWED_DOMAINS; add it there first")
    body = "This is a test email from the email service."
    receipt = await SmtpMailer(settings).send(to=to, subject="Ops assistant SMTP test", body=body, task_id="cli-test")
    print(receipt.model_dump_json(indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send one test email with the current SMTP settings.")
    parser.add_argument("to", help="the recipient (must be allowed by EMAIL_ALLOWED_DOMAINS)")
    asyncio.run(_main(parser.parse_args().to))
