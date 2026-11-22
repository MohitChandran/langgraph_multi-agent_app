"""The data the two services exchange over A2A, and what the crew returns (plan 05, section 5).

Everything travels as A2A DATA parts (JSON objects), never as free text, so nothing has to be parsed out of prose:

    message #1 (orchestrator -> here)   EmailRequest   {to, subject, content, request}
    task artifact "draft"   (here -> orchestrator)    Draft      {to, subject, body, issues}
    message #2 (orchestrator -> here)   Decision       {decision: approve | reject, edited_body}
    task artifact "receipt" (here -> orchestrator)    Receipt    {message_id, sent_at, to, subject}

Pydantic validates every incoming part: a malformed request is refused before any LLM or SMTP work starts.
The length limits mirror the orchestrator's (plan 03), so a request it accepts is never refused here for its size.
Every text field drops control characters (keeping line breaks and tabs): the plan 05 eval found a "\u0000" in a
crew-written subject, which no email (and no Postgres JSONB column) should carry.
"""

import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def without_control_characters(text: str) -> str:
    """The text without NUL and other control characters; line breaks (\n, \r) and tabs stay."""
    return CONTROL_CHARACTERS.sub("", text)


# A str that is cleaned on validation: used for every text field below.
Text = Annotated[str, AfterValidator(without_control_characters)]

MAX_SUBJECT = 150
MAX_CONTENT = 5_000
MAX_REQUEST = 1_000
MAX_BODY = 5_000


class EmailRequest(BaseModel):
    """Message #1: what to write and to whom. `to` is ONE address (plan 04: exactly one recipient)."""

    model_config = ConfigDict(extra="forbid")  # an unknown field is a contract mismatch: refuse it

    to: Text = Field(min_length=3, max_length=254)
    subject: Text = Field(min_length=1, max_length=MAX_SUBJECT)
    content: Text = Field(min_length=1, max_length=MAX_CONTENT)  # the facts to put in the email (data, not orders)
    request: Text = Field(default="", max_length=MAX_REQUEST)  # the user's question, for tone and purpose


class CrewDraft(BaseModel):
    """What the CrewAI crew returns: the crew writes subject and body only; it never chooses the recipient."""

    subject: Text = Field(min_length=1, max_length=MAX_SUBJECT)
    body: Text = Field(min_length=1, max_length=MAX_BODY)
    issues: list[Text] = Field(default_factory=list)  # the reviewer's notes, e.g. "a number in the content is unclear"


class Draft(CrewDraft):
    """The "draft" artifact the human reviews: the crew's draft plus the recipient, copied by CODE from the request."""

    to: Text


class Decision(BaseModel):
    """Message #2: the human's decision on the draft. edited_body = the user's own text, sent as written (D9)."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject"]
    edited_body: Text | None = Field(default=None, max_length=MAX_BODY)


class Receipt(BaseModel):
    """The "receipt" artifact after a successful send: proof of what left, and when."""

    message_id: str  # the email's Message-ID header, e.g. "<...@ops-assistant>"
    sent_at: datetime
    to: str
    subject: str
