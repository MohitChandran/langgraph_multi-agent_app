"""Test doubles for the orchestrator: a scripted supervisor LLM and helpers to write its replies.

Not a test module (no test_ prefix): test_orchestrator_graph.py and test_chat_api.py import from it.

Why a fake LLM: the loop's logic (limits, refusals, the email pause, fallbacks) must be tested exactly
and repeatably, without Groq calls. The fake has the two methods the orchestrator uses:
bind_tools(tools) and ainvoke(messages). It returns the replies it was given, in order, and records
every call, including whether tools were bound (the forced answer must have none, S17).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import groq
import httpx
from langchain_core.messages import AIMessage, BaseMessage

Reply = AIMessage | BaseException | Callable[[list[BaseMessage]], AIMessage | BaseException]


@dataclass
class LLMCall:
    """One call the fake received: the messages, and the tool definitions bound (None = no tools)."""

    messages: list[BaseMessage]
    tools: list[dict[str, Any]] | None

    @property
    def tool_names(self) -> list[str]:
        """Names of the bound tools ([] when none were bound)."""
        return [tool["function"]["name"] for tool in self.tools or []]


class ScriptedLLM:
    """A fake supervisor: returns `replies` in order; the LAST reply repeats forever (e.g. "always call a tool").

    A reply can be an AIMessage, an exception to raise (an LLM failure), or a function of the
    messages returning either (to react to what the loop sent, e.g. a different question each time).
    """

    def __init__(self, *replies: Reply) -> None:
        """Store the script; nothing has been called yet."""
        self.replies = list(replies)
        self.calls: list[LLMCall] = []

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any) -> "BoundLLM":
        """Like ChatGroq.bind_tools: returns a model that 'knows' these tools."""
        return BoundLLM(self, tools)

    async def ainvoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        """A call WITHOUT tools bound (the forced answer)."""
        return self.reply(messages, None)

    def reply(self, messages: list[BaseMessage], tools: list[dict[str, Any]] | None) -> AIMessage:
        """Record the call and return (or raise) the next scripted reply."""
        self.calls.append(LLMCall(list(messages), tools))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if callable(reply) and not isinstance(reply, BaseMessage):
            reply = reply(messages)
        if isinstance(reply, BaseException):
            raise reply
        return fresh(reply)

    @property
    def supervisor_calls(self) -> list[LLMCall]:
        """The calls made with tools bound."""
        return [call for call in self.calls if call.tools is not None]


class BoundLLM:
    """What ScriptedLLM.bind_tools returns: the same script, with the tools recorded on each call."""

    def __init__(self, llm: ScriptedLLM, tools: list[dict[str, Any]]) -> None:
        """Remember the fake and the bound tools."""
        self.llm, self.tools = llm, tools

    async def ainvoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        """A supervisor call with tools bound."""
        return self.llm.reply(messages, self.tools)


def fresh(reply: AIMessage) -> AIMessage:
    """A copy of a scripted reply with NEW tool call ids, so a repeated reply is a new message, like a real LLM's."""
    copy = reply.model_copy(deep=True)
    copy.tool_calls = [{**tool_call, "id": f"call-{uuid4().hex[:8]}"} for tool_call in copy.tool_calls]
    return copy


def calls(*tool_calls: tuple[str, dict[str, Any]]) -> AIMessage:
    """An AI message calling one or more tools: calls(("ask_database", {"question": "q"}), ...)."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": "scripted", "type": "tool_call"} for name, args in tool_calls],
    )


def call(name: str, **args: Any) -> AIMessage:
    """An AI message calling ONE tool: call("ask_database", question="q")."""
    return calls((name, args))


def answer(text: str) -> AIMessage:
    """An AI message with a final answer (no tool calls)."""
    return AIMessage(content=text)


def groq_down() -> groq.APIConnectionError:
    """The error ChatGroq raises when Groq can't be reached (one of app.llm.LLM_ERRORS)."""
    return groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
