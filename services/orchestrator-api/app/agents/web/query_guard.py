"""The web query check: company data must never leave through a web search (plannings/04, section 7).

The input guard checks what the USER typed, but the search query is written by the orchestrator's LLM. It
could put data from an earlier tool result into a query ("payment status of INV-2026-0273 for
anna@company.example"), and that query goes to DuckDuckGo, a third party. So every query is checked here,
in code, before any search. A rejected query is never sent; the supervisor gets the reason and can retry
with a clean query.

Rejected: an email address, a phone number, an invoice/contract id (INV-..., CTR-...), a run of 8+ digits
(account or IBAN-like numbers), or more than 300 characters. Pure functions, no I/O.
"""

import re

MAX_QUERY_CHARS = 300
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
COMPANY_ID = re.compile(r"\b(?:INV|CTR)-\d", re.IGNORECASE)
LONG_NUMBER = re.compile(r"\d{8,}")
# A phone-number candidate: digits with separators (spaces, dots, dashes, parentheses, a leading +).
PHONE_CANDIDATE = re.compile(r"\+?\(?\d[\d\s().-]{5,}\d")
SEPARATORS = re.compile(r"[\s().+-]+")


def looks_like_phone(text: str) -> bool:
    """True for a sequence like "+91 98765 43210" or "(022) 2345-6789": 7+ digits split by separators.

    Lists of 4-digit numbers ("2025 2026", "1990 2000 2010") are years, not phone numbers, so they pass.
    """
    for match in PHONE_CANDIDATE.finditer(text):
        groups = [group for group in SEPARATORS.split(match.group()) if group]
        digits = sum(len(group) for group in groups)
        if digits >= 7 and len(groups) >= 2 and not all(len(group) == 4 for group in groups):
            return True
    return False


def rejected_reason(query: str) -> str | None:
    """Why this query must not go to a web search, or None if it's fine. The first problem found is reported."""
    if len(query) > MAX_QUERY_CHARS:
        return f"it is longer than {MAX_QUERY_CHARS} characters"
    if EMAIL.search(query):
        return "an email address"
    if COMPANY_ID.search(query):
        return "an invoice or contract number"
    if LONG_NUMBER.search(query):
        return "a long account-like number"
    if looks_like_phone(query):
        return "a phone number"
    return None


def rejection_text(reason: str) -> str:
    """The answer the supervisor (and the user) gets for a rejected query; nothing was searched."""
    return f"I can't send internal company data ({reason}) to a web search. Ask again without it."
