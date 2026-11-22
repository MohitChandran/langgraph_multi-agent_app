"""Human in the loop: the A2A task lifecycle of one email (plan 05, sections 3 and 6.2).

An A2A server hands every incoming message to an AgentExecutor. This one, EmailExecutor, runs TWICE per email,
because a human sits in between:

    message #1  {to, subject, content, request}           task: SUBMITTED -> WORKING -> crew drafts -> checks
                                                           -> artifact "draft" -> INPUT_REQUIRED, and execute() RETURNS
    (the task waits in Postgres for minutes, hours, restarts; nothing runs meanwhile)
    message #2  same task id: {decision, edited_body}     approve: checks -> SMTP -> artifact "receipt" -> COMPLETED
                                                           reject:  REJECTED, nothing sent

"Pausing" is therefore not a sleeping coroutine: it is a STATE (INPUT_REQUIRED) saved with the task, and the next
message on the same task picks it up. The SDK refuses messages on a finished task (COMPLETED, REJECTED, FAILED), so an
email can't be approved twice. Sending happens in exactly one place: _decide(), on an explicit approval.

The crew and the mailer are injected (fakes in tests, the real CrewAI crew and SMTP in the service).

Tracing (app/tracing.py): each message is one span, "email service: draft" / "email service: decision", opened with
the orchestrator's traceparent (from the message metadata) as its parent, so it lands inside the orchestrator's
trace; the checks, the crew and the SMTP send are child spans.
"""

import asyncio
import logging
from typing import Any, Protocol

from a2a.helpers import get_data_parts, new_data_part, new_task
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Task, TaskState
from google.protobuf.json_format import MessageToDict
from pydantic import ValidationError

from app.checks import CheckFailed, check_body, check_draft, check_request
from app.config import Settings
from app.models import CrewDraft, Decision, Draft, EmailRequest, Receipt
from app.tracing import get_tracing, parse_traceparent

logger = logging.getLogger(__name__)

DRAFT_ARTIFACT = "draft"
RECEIPT_ARTIFACT = "receipt"


class Drafter(Protocol):
    """Writes the draft: the CrewAI crew (app/agent.py) or a fake."""

    async def draft(self, request: EmailRequest) -> CrewDraft:
        """Subject + body for the request (the recipient is not the crew's business)."""
        ...


class Mailer(Protocol):
    """Sends one email: SMTP (app/mailer.py) or a fake."""

    async def send(self, *, to: str, subject: str, body: str, task_id: str) -> Receipt:
        """Send it and return the receipt; raise on any SMTP error."""
        ...


class EmailExecutor(AgentExecutor):
    """The A2A executor of the email service: draft and pause on message #1, act on the decision on message #2."""

    def __init__(self, drafter: Drafter, mailer: Mailer, settings: Settings) -> None:
        """Keep the crew, the mailer and the settings (allow-list, crew timeout)."""
        self.drafter = drafter
        self.mailer = mailer
        self.settings = settings

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Route the message: no task yet -> a new draft; a task waiting for input -> the human's decision.

        All of it runs inside one span that continues the caller's trace (its traceparent in the message metadata).
        """
        updater = TrackedUpdater(event_queue, context.task_id, context.context_id)
        task = context.current_task
        name = "email service: draft" if task is None else "email service: decision"
        parent = parse_traceparent(_metadata(context.message).get("traceparent"))
        with get_tracing().span(name, as_type="agent", input=_data_or_none(context.message),
                                trace_context=parent) as span:  # fmt: skip
            if task is None:
                # A2A 1.x: the executor must publish the Task itself before any status update (plan 05, gotcha 2).
                await event_queue.enqueue_event(
                    new_task(context.task_id, context.context_id, TaskState.TASK_STATE_SUBMITTED)
                )
                await self._draft(context.message, updater)
            elif task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED:
                await self._decide(context.message, task, updater)
            else:  # the SDK already refuses finished tasks; anything else (e.g. WORKING) must not be touched
                await updater.failed(
                    _reply(updater, {"error": f"unexpected task state {TaskState.Name(task.status.state)}"})
                )
            span.update(output={"task_id": context.task_id, "state": updater.outcome, **updater.detail})

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A client cancelled the task (e.g. while waiting for approval): nothing is sent, the task ends CANCELED."""
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()

    async def _draft(self, message: Message | None, updater: TaskUpdater) -> None:
        """Message #1: validate and check the request, let the crew write, check the draft, then pause for a human."""
        await updater.start_work()
        with get_tracing().span("request checks", as_type="guardrail") as checks:
            try:
                request = EmailRequest.model_validate(_first_data(message))
                check_request(request, self.settings)
            except (ValidationError, CheckFailed, ValueError) as error:
                logger.info("Draft request refused: %s", error)
                checks.update(output={"passed": False, "reason": _reason(error)}, level="WARNING")
                await updater.reject(_reply(updater, {"error": _reason(error)}))
                return
            checks.update(output={"passed": True, "to": request.to})
        try:
            crew_draft = await asyncio.wait_for(self.drafter.draft(request), self.settings.email_crew_timeout_seconds)
            with get_tracing().span("draft checks", as_type="guardrail") as checks:
                try:
                    draft = check_draft(crew_draft, request)
                except CheckFailed as error:
                    checks.update(output={"passed": False, "reason": str(error)}, level="WARNING")
                    raise
                checks.update(output={"passed": True})
        except TimeoutError:
            await updater.failed(_reply(updater, {"error": "the email crew took too long; nothing was drafted"}))
            return
        except CheckFailed as error:
            await updater.failed(_reply(updater, {"error": f"the draft failed a safety check: {error}"}))
            return
        except Exception as error:  # noqa: BLE001  (the crew's own errors: Groq down, bad output...)
            logger.exception("The email crew failed")
            await updater.failed(_reply(updater, {"error": f"the email crew failed: {type(error).__name__}"}))
            return
        await updater.add_artifact([new_data_part(draft.model_dump())], name=DRAFT_ARTIFACT)
        # The pause: INPUT_REQUIRED with a note saying what input is expected, then RETURN.
        await updater.requires_input(_reply(updater, {"action": "approve_or_reject", "draft_artifact": DRAFT_ARTIFACT}))

    async def _decide(self, message: Message | None, task: Task, updater: TaskUpdater) -> None:
        """Message #2: reject -> REJECTED; approve -> re-check the (edited) body, send ONCE, COMPLETED + receipt."""
        try:
            decision = Decision.model_validate(_first_data(message))
        except (ValidationError, ValueError) as error:  # stays paused: the client can send a valid decision
            await updater.requires_input(_reply(updater, {"error": f"invalid decision: {_reason(error)}"}))
            return
        if decision.decision == "reject":
            await updater.reject(_reply(updater, {"result": "rejected"}))
            return
        if _artifact_data(task, RECEIPT_ARTIFACT) is not None:  # belt and braces: never a second send for one task
            await updater.failed(_reply(updater, {"error": "this email was already sent"}))
            return
        draft = Draft.model_validate(_artifact_data(task, DRAFT_ARTIFACT))
        body = decision.edited_body if decision.edited_body is not None else draft.body
        with get_tracing().span(
            "body checks", as_type="guardrail", input={"edited": decision.edited_body is not None}
        ) as checks:
            try:
                check_body(body)
            except CheckFailed as error:  # an edit the service won't send: stay paused, say why (plan 05, 8.3)
                checks.update(output={"passed": False, "reason": str(error)}, level="WARNING")
                await updater.requires_input(_reply(updater, {"error": f"the edited email was not sent: {error}"}))
                return
            checks.update(output={"passed": True})
        await updater.start_work()  # WORKING while sending: a second decision now finds no INPUT_REQUIRED task
        with get_tracing().span("smtp send", as_type="tool", input={"to": draft.to, "subject": draft.subject}) as smtp:
            try:
                receipt = await self.mailer.send(to=draft.to, subject=draft.subject, body=body, task_id=task.id)
            except Exception as error:  # noqa: BLE001  (SMTP refused, timeout, DNS...)
                logger.exception("Sending task %s failed", task.id)
                smtp.update(level="ERROR", status_message=f"{type(error).__name__}: {error}"[:200])
                await updater.failed(_reply(updater, {"error": f"SMTP error: {type(error).__name__}: {error}"[:300]}))
                return
            smtp.update(output=receipt.model_dump(mode="json"))
        await updater.add_artifact([new_data_part(receipt.model_dump(mode="json"))], name=RECEIPT_ARTIFACT)
        await updater.complete(_reply(updater, {"result": "sent", "message_id": receipt.message_id}))


class TrackedUpdater(TaskUpdater):
    """A TaskUpdater that remembers where it left the task (the span's output): the state and its data part."""

    outcome = "WORKING"
    detail: dict[str, Any] = {}

    def _note(self, state: str, message: Message | None) -> None:
        """Remember the state and the status message's data (e.g. {"error": "..."})."""
        self.outcome = state
        data = get_data_parts(message.parts) if message is not None else []
        self.detail = data[0] if data and isinstance(data[0], dict) else {}

    async def complete(self, message: Message | None = None) -> None:
        """COMPLETED (sent)."""
        self._note("COMPLETED", message)
        await super().complete(message)

    async def failed(self, message: Message | None = None) -> None:
        """FAILED (crew, checks or SMTP)."""
        self._note("FAILED", message)
        await super().failed(message)

    async def reject(self, message: Message | None = None) -> None:
        """REJECTED (refused request, or the human said no)."""
        self._note("REJECTED", message)
        await super().reject(message)

    async def requires_input(self, message: Message | None = None) -> None:
        """INPUT_REQUIRED (paused for the human)."""
        self._note("INPUT_REQUIRED", message)
        await super().requires_input(message)


def _metadata(message: Message | None) -> dict[str, Any]:
    """The message's metadata as a dict ({} without one): the orchestrator puts its traceparent there."""
    return MessageToDict(message.metadata) if message is not None and message.HasField("metadata") else {}


def _data_or_none(message: Message | None) -> dict[str, Any] | None:
    """The message's data part for the span's input, or None when there is none."""
    try:
        return _first_data(message)
    except ValueError:
        return None


def _first_data(message: Message | None) -> dict[str, Any]:
    """The first data part of a message, as a dict; ValueError when there is none (e.g. a text-only message)."""
    data = get_data_parts(message.parts) if message is not None else []
    if not data or not isinstance(data[0], dict):
        raise ValueError("the message must carry one JSON object as a data part")
    return data[0]


def _artifact_data(task: Task, name: str) -> dict[str, Any] | None:
    """The data of the task's artifact with this name (the draft, the receipt), or None."""
    for artifact in task.artifacts:
        if artifact.name == name:
            data = get_data_parts(artifact.parts)
            return data[0] if data else None
    return None


def _reply(updater: TaskUpdater, data: dict[str, Any]) -> Message:
    """A status message from the agent carrying one data part (why it paused, failed, refused...)."""
    return updater.new_agent_message([new_data_part(data)])


def _reason(error: Exception) -> str:
    """A short, readable reason: pydantic's first error ("subject: String should have at most 150 characters")."""
    if isinstance(error, ValidationError):
        first = error.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or "message"
        return f"{field}: {first['msg']}"
    return str(error)
