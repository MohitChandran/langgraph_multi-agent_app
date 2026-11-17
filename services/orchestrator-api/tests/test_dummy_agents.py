"""Tests for the dummy web and email agents (stand-ins until the real Phase 4-6 agents exist).

They pin down the behaviour the orchestrator will rely on: the same interface as DBAgent/DocAgent,
clearly-fake results, the email draft -> approve/reject lifecycle, and that slow agents can run
in parallel. No database, network or LLM needed.
"""

import asyncio
import time
from uuid import uuid4

from app.agents.email.dummy import DEFAULT_RECIPIENT, DummyEmailAgent
from app.agents.web.dummy import DummyWebSearchAgent


async def test_web_agent_returns_clearly_fake_result_with_sources() -> None:
    """Same call shape as the real agents (question, history, session_id); the answer is labelled fake."""
    result = await DummyWebSearchAgent().ask("Did AWS change its SLA?", [], uuid4())

    assert DummyWebSearchAgent.name == "web"
    assert result.success and result.answer.startswith("[dummy]")
    assert result.question == result.query == "Did AWS change its SLA?"
    assert [source.n for source in result.sources] == [1, 2]
    assert all(source.url.startswith("https://example.com/") for source in result.sources)


async def test_email_ask_returns_a_draft_waiting_for_approval() -> None:
    """ask() never sends: it returns a draft with status input_required, addressed to the emails in the request."""
    result = await DummyEmailAgent().ask("Send the summary to anna@acme.example and bob@acme.example")

    assert DummyEmailAgent.name == "email"
    assert result.status == "input_required" and result.needs_approval
    assert result.draft is not None and result.draft.to == ["anna@acme.example", "bob@acme.example"]
    assert result.message_id is None


async def test_email_uses_default_recipient_and_context() -> None:
    """No address in the request -> the default .example recipient; `context` becomes the email content."""
    result = await DummyEmailAgent().ask("Email the summary", context="3 invoices are due this week.")

    assert result.draft.to == [DEFAULT_RECIPIENT]
    assert "3 invoices are due this week." in result.draft.body


async def test_email_approve_sends_with_edited_body() -> None:
    """Approving returns status completed with a message id, and an edited body replaces the draft's."""
    agent = DummyEmailAgent()
    drafted = await agent.ask("Email the summary to anna@acme.example")

    sent = await agent.resume(drafted.draft.draft_id, approved=True, edited_body="Short version.")

    assert sent.status == "completed" and not sent.needs_approval
    assert sent.message_id == f"dummy-{drafted.draft.draft_id}"
    assert sent.draft.body == "Short version."


async def test_email_reject_sends_nothing_and_draft_cannot_be_reused() -> None:
    """Rejecting sends nothing; deciding the same draft a second time fails cleanly instead of raising."""
    agent = DummyEmailAgent()
    drafted = await agent.ask("Email the summary")

    rejected = await agent.resume(drafted.draft.draft_id, approved=False)
    again = await agent.resume(drafted.draft.draft_id, approved=True)

    assert rejected.status == "rejected" and rejected.message_id is None
    assert not again.success and again.error and again.message_id is None


async def test_slow_dummy_agents_can_run_in_parallel() -> None:
    """Two agents taking 0.3 s each finish together in ~0.3 s when awaited concurrently (what the orchestrator does)."""
    started = time.perf_counter()
    await asyncio.gather(DummyWebSearchAgent(delay_seconds=0.3).ask("q"), DummyEmailAgent(delay_seconds=0.3).ask("q"))

    assert time.perf_counter() - started < 0.5
