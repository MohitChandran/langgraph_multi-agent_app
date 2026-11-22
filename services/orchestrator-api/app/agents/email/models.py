"""What the email agent returns to the orchestrator, real (A2A) or dummy (plan 05, section 8.1).

Shared by A2AEmailAgent (a2a_agent.py, the client of the email service) and DummyEmailAgent (dummy.py, tests and
the CLI without the service), so the orchestrator handles both the same way.
"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class EmailDraft(BaseModel):
    """An email waiting for approval: exactly what the UI's approval card shows.

    draft_id is the A2A task id in the email service; context_id is that task's context. The decision (message #2)
    must carry both, so both are saved in the paused orchestrator run (plan 05, gotcha 3).
    """

    draft_id: UUID
    context_id: str | None = None
    to: list[str]
    subject: str
    body: str
    issues: list[str] = Field(default_factory=list)  # the email crew's reviewer notes (empty when all is fine)


class EmailAgentResult(BaseModel):
    """What the email agent returns. The core fields match the other agents' results.

    status follows the A2A task states:
      input_required - a draft exists and needs a human decision (needs_approval = True)
      completed      - approved and sent (message_id is set)
      rejected       - the human said no; nothing was sent
      failed         - the service refused or failed (error says why); nothing was sent
    """

    question: str
    answer: str
    success: bool
    status: Literal["input_required", "completed", "rejected", "failed"]
    needs_approval: bool
    draft: EmailDraft | None = None
    message_id: str | None = None
    steps: list[str] = Field(default_factory=list, description="Workflow steps in the order they ran.")
    reasoning: list[str] = Field(default_factory=list, description="Short sentences on what the agent did and why.")
    error: str | None = None
