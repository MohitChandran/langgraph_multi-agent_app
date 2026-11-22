"""Tests for the email service's deterministic checks (app/checks.py, plan 05 section 6.4). Pure functions."""

import pytest

from app.checks import CheckFailed, check_body, check_draft, check_request, iban_valid, luhn_valid
from app.config import Settings
from app.models import CrewDraft, EmailRequest


def settings(**overrides: object) -> Settings:
    """Settings for the checks alone (no .env); the allow-list is the development default unless overridden."""
    base = {"email_service_api_key": "k", "email_database_url": "postgresql+asyncpg://unused/db", "groq_api_key": "g"}
    return Settings(_env_file=None, **{**base, **overrides})


def request(to: str = "accounts@company.example") -> EmailRequest:
    """A valid draft request to `to`."""
    return EmailRequest(to=to, subject="Pending invoices", content="8 invoices, 73,400 USD.", request="email it")


@pytest.mark.parametrize("to", ["accounts@company.example", "ACCOUNTS@Company.Example", "a.b+c@example.com"])
def test_one_address_on_an_allowed_domain_passes(to: str) -> None:
    """One plain address whose domain is on the allow-list (case doesn't matter)."""
    check_request(request(to), settings())


@pytest.mark.parametrize(
    "to",
    [
        "accounts@company.example, legal@company.example",  # two recipients
        "accounts@company.example;legal@company.example",
        "Accounts <accounts@company.example>",  # display-name form
        "accounts.company.example",  # no @
        "accounts@company",  # no top-level domain
    ],
)
def test_anything_but_one_plain_address_is_refused(to: str) -> None:
    """Exactly one recipient, plain syntax."""
    with pytest.raises(CheckFailed, match="not one valid email address"):
        check_request(request(to), settings())


def test_a_domain_off_the_allow_list_is_refused_unless_any_domain_is_allowed() -> None:
    """ceo@gmail.com is refused in development; EMAIL_ALLOW_ANY_DOMAIN=true lets it through."""
    with pytest.raises(CheckFailed, match="not allowed in this environment"):
        check_request(request("ceo@gmail.com"), settings())
    check_request(request("ceo@gmail.com"), settings(email_allow_any_domain=True))


def test_an_exact_address_on_the_allow_list_allows_only_that_address() -> None:
    """ "me@gmail.com" in the list allows me@gmail.com, not every Gmail user."""
    allowed = settings(email_allowed_domains="company.example, me@gmail.com")
    check_request(request("me@gmail.com"), allowed)
    with pytest.raises(CheckFailed, match="not allowed"):
        check_request(request("someone@gmail.com"), allowed)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("Card: 4111 1111 1111 1111, exp 12/29", "payment card"),
        ("Pay to DE89 3704 0044 0532 0130 00 by Friday", "IBAN"),
        ("The VPN password: Hunter2!", "password"),
        ("api_key = sk-123456", "password or secret"),
        ("", "empty"),
        ("x" * 5001, "too long"),
    ],
)
def test_bodies_with_secrets_or_bad_sizes_are_refused(body: str, reason: str) -> None:
    """Card numbers (Luhn), IBANs (mod 97), password-like secrets, empty and too-long bodies."""
    with pytest.raises(CheckFailed, match=reason):
        check_body(body)


def test_normal_business_numbers_pass() -> None:
    """Invoice ids, amounts, dates, phone numbers and a random 16-digit number that fails Luhn are fine."""
    check_body(
        "INV-2026-0265: 21,837.86 USD due 2026-11-26. Call +1 415 555 0100. Reference 1234567812345678. "
        "Our password policy changed last week."
    )


def test_the_checksums() -> None:
    """Luhn and IBAN mod-97 on known valid and invalid values."""
    assert luhn_valid("4111111111111111") and not luhn_valid("4111111111111112")
    assert iban_valid("DE89370400440532013000") and not iban_valid("DE89370400440532013001")


def test_the_draft_recipient_is_always_the_requests() -> None:
    """Whatever the crew wrote, `to` comes from the request (code), and the body is checked."""
    draft = check_draft(CrewDraft(subject=" Pending ", body=" Hello ", issues=["n/a"]), request())
    assert draft.to == "accounts@company.example" and draft.subject == "Pending" and draft.body == "Hello"
    with pytest.raises(CheckFailed):
        check_draft(CrewDraft(subject="s", body="card 4111111111111111"), request())


def test_control_characters_are_removed_from_every_text_field() -> None:
    """A NUL in a crew-written subject (seen in the plan 05 eval) is dropped; line breaks and tabs stay."""
    draft = CrewDraft(subject="Due (Nov 16\x0021)", body="Hello\n\tteam\x07")
    assert draft.subject == "Due (Nov 1621)" and draft.body == "Hello\n\tteam"
    assert request(to="accounts@company.example\x00").to == "accounts@company.example"
