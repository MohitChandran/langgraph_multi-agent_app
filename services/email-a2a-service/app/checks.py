"""The email service's own guard: deterministic checks, no LLM (plan 05, section 6.4).

The orchestrator already checks recipients (plan 03, S10-S12) and a human approves every email, but this service
SENDS email, so it checks again itself (plan 04's principle: each agent guards its own risks). Pure functions: each
one returns normally or raises CheckFailed with a reason a person can read.

    check_request   message #1: one valid recipient, on the allow-list
    check_body      a draft or an edited body: size, and no card numbers, IBANs or passwords
    check_draft     the crew's output: check_body, and the recipient is ALWAYS the request's (code, not the LLM)
"""

import re

from app.config import Settings
from app.models import MAX_BODY, CrewDraft, Draft, EmailRequest

# One plain address: no spaces, commas or angle brackets (so "a@x.com, b@y.com" or "Name <a@x.com>" fail).
ADDRESS = re.compile(r"[^@\s,;<>\"']+@[^@\s,;<>\"']+\.[A-Za-z]{2,}")
# 13-19 digits, optionally grouped by spaces or dashes: the shape of a payment card number (checked with Luhn).
CARD_CANDIDATE = re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])")
# Two letters, two check digits, then 11-30 letters/digits, optionally in groups of 4: an IBAN (checked mod 97).
IBAN_CANDIDATE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b")
# "password: hunter2", "pwd = ...", "api key: ...", "secret: ..."
SECRET = re.compile(r"\b(?:password|passwd|pwd|passcode|secret|api[ _-]?key|access[ _-]?token)\s*[:=]\s*\S+", re.I)


class CheckFailed(ValueError):
    """A check refused something; str(error) is the reason (shown to the orchestrator and the user)."""


def check_request(request: EmailRequest, settings: Settings) -> None:
    """Message #1: exactly one well-formed recipient, allowed in this environment."""
    to = request.to.strip()
    if not ADDRESS.fullmatch(to):
        raise CheckFailed(f"'{request.to}' is not one valid email address (exactly one recipient is allowed)")
    if not settings.email_allow_any_domain and not recipient_allowed(to, settings.allowed_recipients):
        raise CheckFailed(f"sending to '{to}' is not allowed in this environment (EMAIL_ALLOWED_DOMAINS)")


def recipient_allowed(address: str, allowed: list[str]) -> bool:
    """True if the address's domain, or the address itself, is on the allow-list (case-insensitive)."""
    address = address.lower()
    domain = address.rsplit("@", 1)[-1]
    return any(entry == address if "@" in entry else entry == domain for entry in allowed)


def check_body(body: str) -> None:
    """A body that may be sent: not empty, not too long, and no card numbers, IBANs or passwords in it."""
    if not body.strip():
        raise CheckFailed("the email body is empty")
    if len(body) > MAX_BODY:
        raise CheckFailed(f"the email body is too long ({len(body)} characters, the limit is {MAX_BODY})")
    if any(luhn_valid(re.sub(r"\D", "", match)) for match in CARD_CANDIDATE.findall(body)):
        raise CheckFailed("the email contains what looks like a payment card number")
    if any(iban_valid(match.replace(" ", "")) for match in IBAN_CANDIDATE.findall(body)):
        raise CheckFailed("the email contains what looks like a bank account number (IBAN)")
    if SECRET.search(body):
        raise CheckFailed("the email contains what looks like a password or secret")


def check_draft(draft: CrewDraft, request: EmailRequest) -> Draft:
    """The crew's draft, checked like any body, with the recipient set by code from the request (the LLM can't
    change who gets the email)."""
    check_body(draft.body)
    subject = " ".join(draft.subject.split())  # one line: a header can't contain line breaks
    return Draft(to=request.to.strip(), subject=subject, body=draft.body.strip(), issues=draft.issues)


def luhn_valid(digits: str) -> bool:
    """The Luhn checksum every payment card number satisfies (random 16-digit numbers mostly don't)."""
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def iban_valid(candidate: str) -> bool:
    """The IBAN mod-97 check: move the first 4 characters to the end, letters -> numbers (A=10...), remainder 1."""
    if not 15 <= len(candidate) <= 34:
        return False
    rearranged = candidate[4:] + candidate[:4]
    number = "".join(str(int(char, 36)) for char in rearranged)
    return int(number) % 97 == 1
