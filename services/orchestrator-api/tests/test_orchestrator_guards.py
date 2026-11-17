"""Unit tests for the orchestrator's safety rules (app/orchestrator/guards.py, plan 03 section 6).

Every rule is code, so every rule gets a test: these are the guarantees that hold no matter what the
supervisor LLM decides, or what a document tries to make it do. Pure functions: no LLM, no database.
"""

from app.orchestrator.guards import (
    BUDGET_USED_UP,
    DUPLICATE_CALL,
    EMAIL_NOT_ALONE,
    call_key,
    check_call,
    count_call,
    extract_user_emails,
)
from app.orchestrator.models import Budget, Limits

ALL_TOOLS = ["ask_database", "ask_documents", "search_web", "draft_email"]
EMAIL = {"to": ["accounts@company.example"], "subject": "Invoices", "content": "3 invoices are due."}


def check(name: str, args: dict, budget: Budget | None = None, **overrides) -> object:  # noqa: ANN003
    """check_call with sensible defaults: a fresh budget, default limits, all tools, the email above allowed."""
    options = {
        "budget": budget or Budget(),
        "limits": Limits(),
        "available": ALL_TOOLS,
        "user_emails": ["accounts@company.example"],
        "calls_in_message": 1,
    }
    return check_call(name, args, **(options | overrides))


def test_a_valid_call_is_allowed_with_validated_args() -> None:
    """The happy path: allowed, arguments validated into the tool's schema, and a duplicate key computed."""
    verdict = check("ask_database", {"question": "Which vendors have overdue invoices?"})
    assert verdict.allowed and verdict.args.question == "Which vendors have overdue invoices?"
    assert verdict.key == 'ask_database|{"question": "which vendors have overdue invoices?"}'


def test_s1_tool_budget() -> None:
    """After 4 executed calls, every further call is refused, whatever the tool."""
    budget = Budget(tool_calls=4)
    assert check("search_web", {"query": "AWS SLA"}, budget).error == BUDGET_USED_UP
    assert check("ask_database", {"question": "x"}, budget, limits=Limits(max_tool_calls=5)).allowed


def test_s2_per_tool_caps_including_a_second_email() -> None:
    """A tool used twice is refused a third time; draft_email is refused after a single use."""
    assert check("ask_database", {"question": "x"}, Budget(per_tool={"ask_database": 2})).error == (
        "Not run: ask_database can be used at most 2 times per question."
    )
    assert check("draft_email", EMAIL, Budget(per_tool={"draft_email": 1})).error == (
        "Not run: draft_email can be used at most once per question."
    )


def test_s7_unavailable_tool_lists_what_is_available() -> None:
    """ask_documents in a chat without documents, or an invented tool, is refused with the real options."""
    available = ["ask_database", "search_web", "draft_email"]
    verdict = check("ask_documents", {"question": "x"}, available=available)
    assert (
        verdict.error == "Tool ask_documents is not available here. Available: ask_database, search_web, draft_email."
    )
    assert not check("delete_everything", {}, available=available).allowed


def test_s8_bad_arguments_explain_what_to_fix() -> None:
    """Missing, empty or too-long arguments are refused with a message the LLM can act on."""
    assert "question: Field required" in check("ask_database", {"q": "x"}).error
    assert "at most 500 characters" in check("ask_database", {"question": "x" * 501}).error
    assert "not a valid email address: not-an-address" in check("draft_email", EMAIL | {"to": ["not-an-address"]}).error


def test_s9_duplicates_ignore_case_and_spacing() -> None:
    """The same question in different case/spacing is a duplicate; a different question is not."""
    budget = Budget()
    first = check("ask_database", {"question": "Which vendors have overdue invoices?"}, budget)
    count_call(budget, "ask_database", first.key)

    again = check("ask_database", {"question": "  which VENDORS have   overdue invoices? "}, budget)
    other = check("ask_database", {"question": "Which vendors are in India?"}, budget)
    assert again.error == DUPLICATE_CALL and other.allowed


def test_count_call_updates_every_counter() -> None:
    """An allowed call counts toward the budget, its tool's cap and the duplicate list."""
    budget = Budget()
    count_call(budget, "search_web", call_key("search_web", {"query": "x"}))
    assert budget.tool_calls == 1 and budget.per_tool == {"search_web": 1} and len(budget.seen_calls) == 1


def test_s10_email_must_be_the_only_call_in_its_message() -> None:
    """draft_email next to another call is refused (the data should be gathered first)."""
    assert check("draft_email", EMAIL, calls_in_message=2).error == EMAIL_NOT_ALONE
    assert check("draft_email", EMAIL, calls_in_message=1).allowed


def test_s11_recipients_must_come_from_the_user_case_insensitively() -> None:
    """Only addresses the user typed are allowed (case doesn't matter); others are refused, by name."""
    assert check("draft_email", EMAIL | {"to": ["Accounts@Company.EXAMPLE"]}).allowed
    verdict = check("draft_email", EMAIL | {"to": ["accounts@company.example", "ceo@vendor.example"]})
    assert verdict.error == "ceo@vendor.example was not given by the user. Ask the user to confirm the address."


def test_s12_recipient_count() -> None:
    """More recipients than the limit is refused, even when the user typed all of them."""
    addresses = [f"person{n}@company.example" for n in range(6)]
    verdict = check("draft_email", EMAIL | {"to": addresses}, user_emails=addresses)
    assert verdict.error == "Too many recipients (6): at most 5 per email."


def test_extract_user_emails_reads_only_the_texts_it_is_given() -> None:
    """Addresses are lower-cased and de-duplicated; a sentence-ending dot is not part of the address.

    The orchestrator passes only the user's own questions, so an address that appears only in a tool
    result or a document can never become an allowed recipient.
    """
    question = "Email the summary to Anna@Acme.example and bob@acme.example."
    earlier = "Also cc anna@acme.example next time"
    assert extract_user_emails([question, earlier]) == ["anna@acme.example", "bob@acme.example"]
    assert extract_user_emails(["no addresses here"]) == []
