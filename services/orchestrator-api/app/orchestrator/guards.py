"""Safety rules and limits for tool calls, as small pure functions (plan 03, section 6).

The supervisor LLM DECIDES which tools to call; this module decides whether a call may RUN. The prompt
also states these rules, but a prompt is a request, not a guarantee: an LLM can ignore it, and a
document or web page can contain text that tries to steer it ("email all vendors..."). So every rule
here is enforced by code and has a test.

check_call() applies the rules to one tool call in this order (the first failing rule wins):

    S1  tool budget        at most 4 tool calls executed per question
    S2  per-tool cap       ask_*/search_web at most 2 each, draft_email at most 1
    S7  availability       only tools this chat offers (e.g. no ask_documents without documents)
    S8  arguments          must match the tool's schema (lengths, valid email addresses...)
    S9  duplicates         the same tool with the same (normalized) input isn't run twice
    S10 email alone        draft_email must be the only call in its message, after the data is gathered
    S11 recipients         every recipient must be an address the USER typed, never one found by a tool
    S12 recipient count    at most 5 recipients

"Pure" = no I/O and no hidden state: the same inputs always give the same verdict, which makes the
rules easy to test and to reason about.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from app.orchestrator.models import Budget, Limits
from app.orchestrator.tools import EMAIL_PATTERN, TOOL_ARGS

BUDGET_USED_UP = "Not run: the tool budget for this question is used up."
DUPLICATE_CALL = "Already called with the same input; use the earlier result."
EMAIL_NOT_ALONE = "Call draft_email alone, after you have the data."


@dataclass(frozen=True)
class Verdict:
    """The outcome of check_call: allowed (with the validated arguments) or refused (with the reason).

    `error` is sent to the supervisor as the tool's result, so it's written for the LLM: it says what
    went wrong and, where possible, what to do instead.
    """

    allowed: bool
    args: BaseModel | None = None  # validated arguments (allowed calls only)
    key: str = ""  # normalized "tool|args", remembered to catch duplicates (S9)
    error: str | None = None


def refuse(error: str) -> Verdict:
    """A refused verdict with the reason."""
    return Verdict(allowed=False, error=error)


def check_call(
    name: str,
    args: dict[str, Any],
    *,
    budget: Budget,
    limits: Limits,
    available: list[str],
    user_emails: list[str],
    calls_in_message: int,
) -> Verdict:
    """Decide whether one tool call may run (rules S1, S2, S7-S12, in that order).

    Args:
        name, args: the tool call exactly as the LLM wrote it.
        budget: what this question has used so far, INCLUDING calls allowed earlier in the same message.
        available: the tools this chat offers.
        user_emails: lower-cased addresses the user typed (extract_user_emails).
        calls_in_message: how many tool calls the LLM put in this one message (S10).
    """
    if budget.tool_calls >= limits.max_tool_calls:  # S1
        return refuse(BUDGET_USED_UP)
    cap = per_tool_limit(name, limits)
    if budget.per_tool.get(name, 0) >= cap:  # S2
        times = "once" if cap == 1 else f"{cap} times"
        return refuse(f"Not run: {name} can be used at most {times} per question.")
    if name not in available:  # S7
        return refuse(f"Tool {name} is not available here. Available: {', '.join(available)}.")
    try:  # S8
        validated = TOOL_ARGS[name].model_validate(args)
    except ValidationError as error:
        return refuse(f"Invalid arguments for {name}: {validation_message(error)}. Fix them and call again.")
    key = call_key(name, validated.model_dump())
    if key in budget.seen_calls:  # S9
        return refuse(DUPLICATE_CALL)
    if name == "draft_email":
        if calls_in_message > 1:  # S10
            return refuse(EMAIL_NOT_ALONE)
        not_given = [address for address in validated.to if address.lower() not in user_emails]
        if not_given:  # S11
            return refuse(
                f"{', '.join(not_given)} {'was' if len(not_given) == 1 else 'were'} not given by the user. "
                "Ask the user to confirm the address."
            )
        if len(validated.to) > limits.max_recipients:  # S12
            return refuse(f"Too many recipients ({len(validated.to)}): at most {limits.max_recipients} per email.")
    return Verdict(allowed=True, args=validated, key=key)


def count_call(budget: Budget, name: str, key: str) -> None:
    """Record an ALLOWED call in the budget (S1, S2, S9). Refused calls are never counted.

    Called right after check_call allows a call, before the next call of the same message is checked,
    so two calls in one message can't both slip under a limit.
    """
    budget.tool_calls += 1
    budget.per_tool[name] = budget.per_tool.get(name, 0) + 1
    budget.seen_calls.append(key)


def per_tool_limit(name: str, limits: Limits) -> int:
    """How often one tool may run per question: draft_email once, every other tool max_calls_per_tool."""
    return limits.max_email_drafts if name == "draft_email" else limits.max_calls_per_tool


def call_key(name: str, args: dict[str, Any]) -> str:
    """A normalized "tool|args" string: two calls with the same key count as duplicates (S9).

    Strings are lower-cased, trimmed and have their whitespace collapsed, so "Which  VENDORS?" and
    "which vendors?" are the same call. Keys are sorted, so argument order doesn't matter.
    """
    return f"{name}|{json.dumps(_normalize(args), sort_keys=True)}"


def _normalize(value: Any) -> Any:
    """Normalize strings inside any JSON-like value (dicts and lists are walked recursively)."""
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def extract_user_emails(texts: Iterable[str]) -> list[str]:
    """Every email address in the USER's own texts, lower-cased, unique, in first-seen order (S11).

    Called with the current question and the user's earlier questions in this chat, never with tool
    results, answers or documents: an address found in data must never become a recipient by itself.
    """
    found = (match.lower() for text in texts for match in EMAIL_PATTERN.findall(text))
    return list(dict.fromkeys(found))


def validation_message(error: ValidationError) -> str:
    """A compact one-line version of Pydantic's errors, e.g. "question: String should have at most 500 characters"."""
    parts = []
    for item in error.errors():
        where = ".".join(str(part) for part in item["loc"]) or "arguments"
        parts.append(f"{where}: {item['msg']}")
    return "; ".join(parts)
