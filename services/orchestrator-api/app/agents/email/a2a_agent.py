"""The email agent as the orchestrator sees it: an A2A client of the email service (plan 05, section 8).

The email agent is not code in this process: it is a separate service (services/email-a2a-service, CrewAI) that this
client talks to over the A2A protocol, in two messages on ONE A2A task:

    draft(to, subject, content, request)   message #1 -> the task comes back INPUT_REQUIRED with a "draft" artifact
                                            -> EmailAgentResult(status="input_required", draft=...)
       ... the orchestrator pauses (LangGraph interrupt); the human approves, edits or rejects in the UI ...
    resume(draft_id, approved, edited_body) message #2 on the same task id + context id
                                            -> COMPLETED + "receipt" (sent) / REJECTED / INPUT_REQUIRED or FAILED
                                               (not sent, with the service's reason)

Same interface as DummyEmailAgent, so AgentTools doesn't care which one it has. Nothing here raises into the
orchestrator: every failure (service down, timeout, refused request) becomes success=False with a readable reason.

The client is built from the service's agent card (GET /.well-known/agent-card.json) at startup. If the service is
not reachable then, `available` is False and the orchestrator doesn't offer draft_email (like search_web without its
MCP server); a later call tries to connect again, so an approval still works once the service is back.

Try it (needs the email service on EMAIL_AGENT_URL; sends to Mailpit):
    uv run python -m app.agents.email.a2a_agent accounts@company.example
"""

import argparse
import asyncio
import logging
import time
from typing import Any
from uuid import UUID, uuid4

import httpx
from a2a.client import Client, ClientConfig, ClientFactory
from a2a.helpers import get_data_parts, new_data_part
from a2a.types import CancelTaskRequest, Message, Role, SendMessageRequest, Task, TaskState

from app.agents.email.models import EmailAgentResult, EmailDraft
from app.config import Settings, get_settings
from app.observability import get_tracing

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"
REQUEST_MAX_CHARS = 1_000  # the service's limit for the user's question (it's only used for tone)
RECHECK_SECONDS = 30.0  # while unavailable: how often refresh() looks for the service again
DRAFT_ARTIFACT, RECEIPT_ARTIFACT = "draft", "receipt"


class A2AEmailAgent:
    """Client of the email service over A2A. Built once at startup (start()), closed at shutdown (close())."""

    name = "email"  # saved in chat_messages.payload.agent, like the other agents

    def __init__(
        self,
        url: str,
        api_key: str,
        draft_timeout_seconds: float = 120.0,
        decision_timeout_seconds: float = 45.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Remember where the service is and how long to wait; nothing connects until start().

        transport: tests pass httpx.ASGITransport(app) to talk to an in-memory A2A server; None = real HTTP.
        """
        self.url = url
        self._api_key = api_key
        self.draft_timeout_seconds = draft_timeout_seconds
        self.decision_timeout_seconds = decision_timeout_seconds
        self._transport = transport
        self._http: httpx.AsyncClient | None = None
        self._client: Client | None = None
        self._healthy = True  # the last readiness check reached the service (see check())
        self._last_attempt = float("-inf")  # when refresh() last looked for the service

    @classmethod
    def from_settings(cls, settings: Settings) -> "A2AEmailAgent":
        """The agent configured by EMAIL_AGENT_URL / EMAIL_AGENT_API_KEY / EMAIL_*_TIMEOUT_SECONDS."""
        return cls(
            settings.email_agent_url,
            settings.email_agent_api_key.get_secret_value(),
            settings.email_draft_timeout_seconds,
            settings.email_decision_timeout_seconds,
        )

    @property
    def available(self) -> bool:
        """True once the agent card was read and the last health check passed: AgentTools offers draft_email only
        then."""
        return self._client is not None and self._healthy

    async def check(self) -> str:
        """Readiness probe (app/health.py): connect if not yet connected, then GET the service's /health.

        Availability follows the result: a service that went down hides draft_email, one that came back (or started
        after the API) offers it again, without restarting the API. Raises when the service can't be reached.
        """
        if not self._api_key:
            raise ConnectionError("EMAIL_AGENT_API_KEY is not set")
        if not await self._connect():
            self._healthy = False
            raise ConnectionError(f"not reachable at {self.url}")
        try:
            response = await self._http.get(self.url.rstrip("/") + "/health", timeout=3.0)
            response.raise_for_status()
        except httpx.HTTPError:
            self._healthy = False
            raise
        self._healthy = True
        return f"A2A service at {self.url}"

    async def start(self) -> None:
        """Connect at startup; on failure log why and stay unavailable (the API still starts)."""
        if not self._api_key:
            logger.warning("EMAIL_AGENT_API_KEY is not set: draft_email is not offered")
            return
        if await self._connect():
            logger.info("Email agent ready: A2A service at %s", self.url)

    async def refresh(self) -> None:
        """While unavailable, look for the service again, at most every RECHECK_SECONDS (called before each question
        by AgentTools.available): draft_email comes back once the service is up, without restarting the API."""
        if self.available or not self._api_key:
            return
        now = time.monotonic()
        if now - self._last_attempt < RECHECK_SECONDS:
            return
        self._last_attempt = now
        try:
            await asyncio.wait_for(self.check(), 3.0)
            logger.info("Email service is reachable again: draft_email is offered")
        except Exception:  # noqa: BLE001  (still down: try again later)
            pass

    async def close(self) -> None:
        """Close the A2A client and its HTTP connections (shutdown)."""
        if self._client is not None:
            await self._client.close()
            self._client = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _connect(self) -> bool:
        """Build the A2A client from the agent card (once); False if the service can't be reached."""
        if self._client is not None:
            return True
        if not self._api_key:
            return False
        if self._http is None:
            # The key goes on every request; reads may take as long as a crew draft.
            timeout = httpx.Timeout(self.draft_timeout_seconds, connect=5.0)
            self._http = httpx.AsyncClient(
                headers={API_KEY_HEADER: self._api_key}, timeout=timeout, transport=self._transport
            )
        try:
            factory = ClientFactory(ClientConfig(httpx_client=self._http, streaming=False))
            self._client = await factory.create_from_url(self.url)
        except Exception as error:  # noqa: BLE001  (connection refused, bad card...)
            logger.warning("Email service not reachable at %s (%s): draft_email is not offered", self.url, error)
            return False
        return True

    async def draft(self, to: str, subject: str, content: str, *, request: str = "") -> EmailAgentResult:
        """Message #1: ask the service for a draft. Returns input_required + the draft, or failed + the reason."""
        question = request or f"Email to {to}: {subject}"
        data = {"to": to, "subject": subject, "content": content, "request": request[:REQUEST_MAX_CHARS]}
        with get_tracing().span("email service: draft", as_type="tool", input=data) as span:
            task, error = await self._send(data, timeout=self.draft_timeout_seconds)
            span.update(output={"state": _state(task), "error": error})
        if task is None:
            return _failed(question, f"I couldn't get a draft: {error}", error)
        draft = _artifact(task, DRAFT_ARTIFACT)
        if task.status.state != TaskState.TASK_STATE_INPUT_REQUIRED or draft is None:
            reason = _status_error(task) or f"the email service ended the task as {_state(task)}"
            return _failed(question, f"The email service did not draft the email: {reason}", reason, task)
        email = EmailDraft(
            draft_id=UUID(task.id), context_id=task.context_id, to=[draft["to"]], subject=draft["subject"],
            body=draft["body"], issues=[str(issue) for issue in draft.get("issues") or []],
        )  # fmt: skip
        reasoning = [
            f"Sent the request to the email service over A2A (task {task.id[:8]}).",
            "Its CrewAI crew drafted the email (a writer, then a compliance reviewer checking it against the facts).",
            "The task is paused (INPUT_REQUIRED): nothing is sent until you approve.",
        ]
        if email.issues:
            reasoning.append(f"Reviewer notes: {'; '.join(email.issues)}")
        return EmailAgentResult(
            question=question,
            answer=f"I drafted an email to {to}. Please review it and approve before it is sent.",
            success=True, status="input_required", needs_approval=True, draft=email,
            steps=["a2a_send_message", "crew_draft", "input_required"], reasoning=reasoning,
        )  # fmt: skip

    async def resume(
        self, draft_id: UUID, approved: bool, edited_body: str | None = None, *, context_id: str | None = None
    ) -> EmailAgentResult:
        """Message #2 on the same task: the human's decision. Sent / rejected / not sent with the reason."""
        data: dict[str, Any] = {"decision": "approve" if approved else "reject"}
        if approved and edited_body is not None:
            data["edited_body"] = edited_body
        with get_tracing().span("email service: decision", as_type="tool", input=data) as span:
            task, error = await self._send(data, str(draft_id), context_id or "", self.decision_timeout_seconds)
            span.update(output={"state": _state(task), "error": error})
        steps = ["a2a_send_message", "decision"]
        if task is None:
            if "terminal state" in (error or ""):
                error = "this draft was already decided"
            return _failed("", f"The email was not sent: {error}", error)
        draft = _artifact(task, DRAFT_ARTIFACT) or {}
        email = EmailDraft(
            draft_id=draft_id, context_id=task.context_id, to=[draft.get("to", "")], subject=draft.get("subject", ""),
            body=edited_body if edited_body is not None else draft.get("body", ""),
        )  # fmt: skip
        state = task.status.state
        if state == TaskState.TASK_STATE_COMPLETED and approved:
            receipt = _artifact(task, RECEIPT_ARTIFACT) or {}
            return EmailAgentResult(
                question="", answer=f"Email sent to {email.to[0]}.", success=True, status="completed",
                needs_approval=False, draft=email, message_id=receipt.get("message_id"), steps=[*steps, "sent"],
                reasoning=["You approved the draft; the email service sent it over SMTP and returned a receipt."],
            )  # fmt: skip
        if state == TaskState.TASK_STATE_REJECTED and not approved:
            return EmailAgentResult(
                question="", answer="Okay, I won't send that email.", success=True, status="rejected",
                needs_approval=False, draft=email, steps=[*steps, "rejected"],
                reasoning=["You rejected the draft, so nothing was sent."],
            )  # fmt: skip
        reason = _status_error(task) or f"the email service ended the task as {_state(task)}"
        if state == TaskState.TASK_STATE_INPUT_REQUIRED:
            # The service refused the edit and is still waiting. No second approval round in this version (plan 05,
            # 8.3): cancel the task, so nothing is left waiting that could be approved later.
            await self._cancel(task.id)
        return _failed("", f"The email was not sent: {reason}", reason, task)

    async def _send(
        self, data: dict[str, Any], task_id: str = "", context_id: str = "", timeout: float = 60.0
    ) -> tuple[Task | None, str | None]:
        """Send one data part; (the task as the service left it, None) or (None, why it failed)."""
        if not await self._connect():
            return None, "the email service is not reachable"
        # The current span (email service: draft/decision) as a W3C traceparent: the service opens its own spans under
        # it, so Langfuse shows ONE trace across both services. Metadata, not data: it isn't part of the email contract.
        traceparent = get_tracing().traceparent()
        message = Message(
            message_id=str(uuid4()), role=Role.ROLE_USER, parts=[new_data_part(data)],
            task_id=task_id, context_id=context_id, metadata={"traceparent": traceparent} if traceparent else None,
        )  # fmt: skip

        async def last_task() -> Task | None:
            """The task from the (single, non-streaming) response."""
            task = None
            async for response in self._client.send_message(SendMessageRequest(message=message)):
                if response.HasField("task"):
                    task = response.task
            return task

        try:
            task = await asyncio.wait_for(last_task(), timeout)
        except TimeoutError:
            return None, f"the email service did not answer within {timeout:g} s"
        except Exception as error:  # noqa: BLE001  (refused by the service, network error...)
            return None, " ".join(str(error).split())[:200] or type(error).__name__
        return (task, None) if task is not None else (None, "the email service answered without a task")

    async def _cancel(self, task_id: str) -> None:
        """Best effort: cancel a task nobody will decide any more."""
        try:
            await self._client.cancel_task(CancelTaskRequest(id=task_id))
        except Exception as error:  # noqa: BLE001
            logger.warning("Could not cancel email task %s: %s", task_id, error)


def _artifact(task: Task, name: str) -> dict[str, Any] | None:
    """The data of the task's artifact with this name ("draft", "receipt"), or None."""
    for artifact in task.artifacts:
        if artifact.name == name:
            data = get_data_parts(artifact.parts)
            return data[0] if data and isinstance(data[0], dict) else None
    return None


def _status_error(task: Task) -> str | None:
    """The "error" the service put in the task's status message (why it refused, failed or paused), if any."""
    data = get_data_parts(task.status.message.parts) if task.status.HasField("message") else []
    return str(data[0]["error"]) if data and isinstance(data[0], dict) and data[0].get("error") else None


def _state(task: Task | None) -> str | None:
    """The task state's short name ("INPUT_REQUIRED"), or None without a task."""
    return TaskState.Name(task.status.state).removeprefix("TASK_STATE_") if task is not None else None


def _failed(question: str, answer: str, error: str | None, task: Task | None = None) -> EmailAgentResult:
    """A failure result: nothing was sent, `error` says why."""
    steps = ["a2a_send_message", *([_state(task).lower()] if task is not None else [])]
    return EmailAgentResult(
        question=question, answer=answer, success=False, status="failed", needs_approval=False,
        steps=steps, reasoning=[f"Nothing was sent: {error}"], error=error,
    )  # fmt: skip


async def _main(to: str) -> None:
    """CLI: draft through the real service, show the draft, approve it (the email lands in Mailpit)."""
    agent = A2AEmailAgent.from_settings(get_settings())
    await agent.start()
    try:
        drafted = await agent.draft(to, "Pending invoices due this week", "8 invoices are pending, 73,400 USD in "
                                    "total.", request=f"Email the pending invoices to {to}")  # fmt: skip
        print("--- draft():", drafted.model_dump_json(indent=2), sep="\n")
        if drafted.draft is not None:
            sent = await agent.resume(drafted.draft.draft_id, True, context_id=drafted.draft.context_id)
            print("--- resume(approved):", sent.model_dump_json(indent=2), sep="\n")
    finally:
        await agent.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draft and approve one email through the email service (A2A).")
    parser.add_argument("to", nargs="?", default="accounts@company.example")
    asyncio.run(_main(parser.parse_args().to))
