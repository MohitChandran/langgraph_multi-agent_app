"""A small A2A client for this service: the demo CLI, and the helpers the tests use (plan 05, stage E2).

It plays the orchestrator's part, so the whole approval flow can be tried without it:

    uv run python -m app.client_demo --to accounts@company.example                 # draft, then approve
    uv run python -m app.client_demo --to accounts@company.example --reject        # draft, then reject
    uv run python -m app.client_demo --to accounts@company.example --edit "Hi team, ..."   # approve an edited body

Message #1 creates a task and comes back paused (INPUT_REQUIRED) with the draft; message #2, on the SAME task id
and context id, carries the decision. The service must be running (uv run uvicorn app.main:app --port 8001).
"""

import argparse
import asyncio
import json
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import Client, ClientConfig, ClientFactory
from a2a.helpers import get_data_parts, new_data_part
from a2a.types import Message, Role, SendMessageRequest, Task, TaskState

from app.card import API_KEY_HEADER
from app.config import get_settings


async def make_client(url: str, http: httpx.AsyncClient) -> Client:
    """An A2A client built from the agent card at `url` (the card says where and how to send messages).

    `http` carries the X-API-Key header on every request; streaming=False = one response per message.
    """
    return await ClientFactory(ClientConfig(httpx_client=http, streaming=False)).create_from_url(url)


async def send_data(
    client: Client, data: dict[str, Any], task_id: str = "", context_id: str = "", metadata: dict | None = None
) -> Task:
    """Send one data part (a new task, or a follow-up on `task_id`) and return the task as the service left it.

    metadata: e.g. {"traceparent": ...}, as the orchestrator sends it (tracing across services).
    """
    message = Message(
        message_id=str(uuid4()),
        role=Role.ROLE_USER,
        parts=[new_data_part(data)],
        task_id=task_id,
        context_id=context_id,
        metadata=metadata,
    )
    task = None
    async for response in client.send_message(SendMessageRequest(message=message)):
        if response.HasField("task"):
            task = response.task
    if task is None:
        raise RuntimeError("the agent answered without a task")
    return task


def artifact(task: Task, name: str) -> dict[str, Any] | None:
    """The data of the task's artifact with this name ("draft", "receipt"), or None."""
    for item in task.artifacts:
        if item.name == name:
            data = get_data_parts(item.parts)
            return data[0] if data else None
    return None


def status_data(task: Task) -> dict[str, Any]:
    """The data part of the task's status message (why it paused, failed or was refused), or {}."""
    data = get_data_parts(task.status.message.parts) if task.HasField("status") else []
    return data[0] if data else {}


def state(task: Task) -> str:
    """The task state's short name, e.g. "INPUT_REQUIRED"."""
    return TaskState.Name(task.status.state).removeprefix("TASK_STATE_")


async def demo(to: str, reject: bool, edit: str | None) -> None:
    """Draft -> show the draft -> decide -> show the result."""
    settings = get_settings()
    headers = {API_KEY_HEADER: settings.email_service_api_key.get_secret_value()}
    async with httpx.AsyncClient(headers=headers, timeout=120) as http:
        client = await make_client(settings.email_service_public_url, http)
        request = {
            "to": to,
            "subject": "Pending invoices due this week",
            "content": "8 invoices are pending and due this week, 73,400 USD in total. The largest: AWS "
            "INV-2026-0265, 21,837.86 USD, due 2026-11-26.",
            "request": f"Find the pending invoices due this week and email a summary to {to}",
        }
        task = await send_data(client, request)
        print(f"#1 -> task {task.id}: {state(task)}  {json.dumps(status_data(task))}")
        draft = artifact(task, "draft")
        if draft is None:
            return
        print(f"\nDraft to {draft['to']}\nSubject: {draft['subject']}\n\n{draft['body']}\n")
        decision = {"decision": "reject"} if reject else {"decision": "approve", "edited_body": edit}
        task = await send_data(client, decision, task.id, task.context_id)
        print(f"#2 {decision['decision']} -> {state(task)}  {json.dumps(status_data(task))}")
        if receipt := artifact(task, "receipt"):
            print(f"Receipt: {json.dumps(receipt)}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Try the email service's approval flow over A2A.")
    parser.add_argument("--to", default="accounts@company.example", help="the ONE recipient")
    parser.add_argument("--reject", action="store_true", help="reject the draft instead of approving it")
    parser.add_argument("--edit", help="approve this body instead of the draft's")
    args = parser.parse_args()
    asyncio.run(demo(args.to, args.reject, args.edit))


if __name__ == "__main__":
    main()
