"""Dummy email agent: drafts a hardcoded email and waits for approval. Never sends anything.

Why it exists: the real email agent (Phases 5-6) is a separate CrewAI service reached over A2A, and it
must never send an email without a human's approval. That gives it a two-step lifecycle, which this
stand-in imitates so the orchestrator's pause/resume can be built and tested now:

    ask(...)     -> a DRAFT with status "input_required"   (the orchestrator pauses, the UI shows Approve/Reject)
    resume(...)  -> status "completed" (approved: "sent")  or "rejected" (nothing sent)

These status names are the ones the A2A protocol uses for tasks, so swapping in the real agent later
keeps the same states. "Sending" here only creates a fake message id: no SMTP, no Mailpit, no network.

Try it:  uv run python -m app.agents.email.dummy "Email a summary of pending invoices to accounts@company.example"
"""

import asyncio
import re
import sys
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.chat.memory import ConversationTurn

DEFAULT_RECIPIENT = "accounts@company.example"  # reserved .example domain: can never reach a real inbox
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


class EmailDraft(BaseModel):
    """An email waiting for approval: exactly what the UI's approval card shows."""

    draft_id: UUID
    to: list[str]
    subject: str
    body: str


class EmailAgentResult(BaseModel):
    """What the email agent returns. The core fields match the other agents' results.

    status follows the A2A task states:
      input_required - a draft exists and needs a human decision (needs_approval = True)
      completed      - approved and "sent" (message_id is set)
      rejected       - the human said no; nothing was sent
    """

    question: str
    answer: str
    success: bool
    status: Literal["input_required", "completed", "rejected"]
    needs_approval: bool
    draft: EmailDraft | None = None
    message_id: str | None = None
    steps: list[str] = Field(default_factory=list, description="Workflow steps in the order they ran.")
    reasoning: list[str] = Field(default_factory=list, description="Short sentences on what the agent did and why.")
    error: str | None = None


class DummyEmailAgent:
    """Stand-in for the email agent: same `name` + async `ask` as the others, plus `resume` for the decision.

    Drafts are kept in memory (a dict) between ask() and resume(); the real agent keeps them in its own
    service. `delay_seconds` simulates a slow agent, like DummyWebSearchAgent.
    """

    name = "email"  # saved in chat_messages.payload.agent

    def __init__(self, delay_seconds: float = 0.0) -> None:
        """Start with no drafts; store how long each call should pretend to take."""
        self.delay_seconds = delay_seconds
        self._drafts: dict[UUID, tuple[str, EmailDraft]] = {}  # draft_id -> (original request, draft)

    async def ask(
        self,
        question: str,
        history: list[ConversationTurn] | None = None,
        session_id: UUID | None = None,
        *,
        context: str | None = None,
    ) -> EmailAgentResult:
        """Draft an email for the request and return it for approval. Nothing is sent.

        Recipients are the email addresses found in the request (default: DEFAULT_RECIPIENT).
        `context` is the content to put in the email, e.g. the DB agent's answer from an earlier
        orchestrator step; without it the body says what the real agent would have filled in.
        `history` and `session_id` are accepted (and ignored) to match the other agents' interface.
        """
        await asyncio.sleep(self.delay_seconds)
        recipients = list(dict.fromkeys(EMAIL_PATTERN.findall(question))) or [DEFAULT_RECIPIENT]
        content = context or "[dummy: the real email agent would write this from the previous step's results]"
        draft = EmailDraft(
            draft_id=uuid4(),
            to=recipients,
            subject="[dummy] Summary you requested",
            body=f"Hello,\n\nHere is the summary you asked for:\n\n{content}\n\nBest regards,\nOps Assistant",
        )
        self._drafts[draft.draft_id] = (question, draft)
        return EmailAgentResult(
            question=question,
            answer=f"I drafted an email to {', '.join(recipients)}. Please review it and approve before it is sent.",
            success=True,
            status="input_required",
            needs_approval=True,
            draft=draft,
            steps=["dummy_draft"],
            reasoning=["Dummy agent: built a hardcoded draft. It waits for a human decision; nothing was sent."],
        )

    async def resume(self, draft_id: UUID, approved: bool, edited_body: str | None = None) -> EmailAgentResult:
        """Apply the human's decision to a draft: approve ("send" it) or reject it.

        `edited_body` replaces the body before sending (the UI's Edit button). Each draft can be decided
        once; an unknown or already-decided draft returns success=False with an error, it never raises.
        """
        await asyncio.sleep(self.delay_seconds)
        entry = self._drafts.pop(draft_id, None)
        if entry is None:
            return EmailAgentResult(
                question="",
                answer="That email draft no longer exists, so nothing was sent.",
                success=False,
                status="rejected",
                needs_approval=False,
                steps=["dummy_resume"],
                error=f"Unknown or already decided draft: {draft_id}",
            )

        question, draft = entry
        if edited_body is not None:
            draft = draft.model_copy(update={"body": edited_body})
        if not approved:
            return EmailAgentResult(
                question=question,
                answer="Okay, I won't send that email.",
                success=True,
                status="rejected",
                needs_approval=False,
                draft=draft,
                steps=["dummy_draft", "dummy_rejected"],
                reasoning=["The human rejected the draft, so nothing was sent."],
            )
        return EmailAgentResult(
            question=question,
            answer=f"[dummy] Email sent to {', '.join(draft.to)}.",
            success=True,
            status="completed",
            needs_approval=False,
            draft=draft,
            message_id=f"dummy-{draft.draft_id}",
            steps=["dummy_draft", "dummy_send"],
            reasoning=["The human approved the draft. Dummy agent: no email actually left the machine."],
        )


async def _main(question: str) -> None:
    """CLI helper: draft an email for the request, then approve it, printing both results."""
    agent = DummyEmailAgent()
    drafted = await agent.ask(question)
    print("--- after ask():", drafted.model_dump_json(indent=2), sep="\n")
    sent = await agent.resume(drafted.draft.draft_id, approved=True)
    print("--- after resume(approved=True):", sent.model_dump_json(indent=2), sep="\n")


if __name__ == "__main__":
    asyncio.run(_main(" ".join(sys.argv[1:]) or "Email a summary of pending invoices to accounts@company.example"))
