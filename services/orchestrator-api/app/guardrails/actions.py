"""The guard's two checks, as the async "actions" NeMo Guardrails calls from config/rails.co (plannings/04, 5.4).

    prompt_guard_check(text)   rail 1, ADVISORY: Prompt Guard 86M returns the probability that the message
                               attacks the assistant's instructions; >= threshold -> "flag". It never blocks:
                               it scored normal messages ("Ignore the cancelled invoices...") as high as real
                               attacks (plan 04, section 16). Its score is recorded and picks the refusal text.
    safety_model_check(text)   rail 2, DECIDES: a small LLM classifies the message against SAFETY_POLICY
                               (harmful, illegal, or a jailbreak) and answers JSON {"verdict", "category", "reason"}

safety_model_check returns "allow", "block", "jailbreak" (blocked and Prompt Guard flagged it or the model
called it a jailbreak: a different refusal text) or "error". NeMo's flows branch on that word; any failure
(Groq error, timeout, unparsable output) is "error", which the flow turns into a BLOCK (fail closed: an
unchecked message is never processed).

Why the guard calls Groq directly (AsyncGroq) instead of get_llm(): the guard models have their own
Groq quotas (Prompt Guard 14,400 requests/day, allam-2-7b 7,000/day), separate from the orchestrator's
models (1,000/day). They're small and fast, and the guard must not eat the orchestrator's quota.

The trace: NeMo only reports WHICH rail blocked. For the UI and the logs we also want the score, the
category and the reason, so each check appends a small dict to a per-check list held in a ContextVar.
InputGuard.check() sets a fresh list before calling NeMo and reads it afterwards. A ContextVar is
"a global variable per async task": two messages checked at the same time each get their own list.
"""

import asyncio
import json
import logging
import math
from contextvars import ContextVar
from typing import Any, Literal

from groq import AsyncGroq

from app.config import Settings
from app.guardrails.policy import SAFETY_POLICY
from app.observability import get_tracing, usage_of

logger = logging.getLogger(__name__)

Verdict = Literal["allow", "block", "jailbreak", "error"]
Signal = Literal["ok", "flag", "error"]  # Prompt Guard's advisory result
JAILBREAK_RAIL = "jailbreak check"
SAFETY_RAIL = "safety check"
SAFETY_MAX_TOKENS = 60  # the JSON verdict is short; also keeps the call cheap

# The per-check trace (see the module docstring). None = no check is collecting (e.g. a direct call in a test).
_trace: ContextVar[list[dict[str, Any]] | None] = ContextVar("guard_trace", default=None)


def start_trace() -> tuple[list[dict[str, Any]], object]:
    """Start collecting details for one check: returns the list the actions will fill, and a token for reset_trace."""
    entries: list[dict[str, Any]] = []
    return entries, _trace.set(entries)


def reset_trace(token: object) -> None:
    """Stop collecting (restores whatever was there before start_trace)."""
    _trace.reset(token)


def prompt_guard_flagged() -> bool:
    """True if rail 1 (Prompt Guard) flagged the message being checked right now."""
    return any(e["rail"] == JAILBREAK_RAIL and e["verdict"] == "flag" for e in _trace.get() or ())


def record(entry: dict[str, Any]) -> None:
    """Add one rail's details to the current check's trace, if a check is collecting."""
    entries = _trace.get()
    if entries is not None:  # `if entries:` would be wrong: an empty list is falsy
        entries.append(entry)


class GuardActions:
    """The two checks, bound to one Groq client and the guard settings.

    InputGuard registers the bound methods with NeMo (`rails.register_action(actions.prompt_guard_check, ...)`);
    NeMo reads their signatures and passes `text` from the flow (`execute prompt_guard_check(text=$user_message)`).
    """

    def __init__(
        self,
        client: Any,
        jailbreak_model: str,
        jailbreak_threshold: float,
        safety_model: str,
        timeout_seconds: float,
    ) -> None:
        """Store the client (AsyncGroq, or a fake in tests) and the models, threshold and timeout."""
        self.client = client
        self.jailbreak_model = jailbreak_model
        self.jailbreak_threshold = jailbreak_threshold
        self.safety_model = safety_model
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_settings(cls, settings: Settings, client: Any = None) -> "GuardActions":
        """Build the actions from GUARD_* settings, with a real AsyncGroq client unless one is given.

        The client retries 429/5xx like the rest of the project (LLM_MAX_RETRIES); the per-call timeout
        (GUARD_TIMEOUT_SECONDS) still caps every check, retries included.
        """
        client = client or AsyncGroq(
            api_key=settings.groq_api_key.get_secret_value(), max_retries=settings.llm_max_retries
        )
        return cls(
            client,
            settings.guard_jailbreak_model,
            settings.guard_jailbreak_threshold,
            settings.guard_safety_model,
            settings.guard_timeout_seconds,
        )

    async def prompt_guard_check(self, text: str) -> Signal:
        """Rail 1 (advisory): how likely is this message an attack on the assistant's instructions?

        Returns "flag" if the score >= threshold, else "ok"; "error" if the check failed. The flow ignores the
        return value: rail 1 never blocks (a failure doesn't either, since it decides nothing). The score goes
        into the trace, where safety_model_check and the UI read it.

        Prompt Guard answers with a probability as text, e.g. "0.0004" for a normal question and "0.9995"
        for "ignore previous instructions...". It isn't a chat model: there's no system prompt, the
        message itself is the input. (It reads ~512 tokens; chat messages are capped at 1,000 characters.)
        """
        with get_tracing().span("prompt guard", as_type="generation", input=text, model=self.jailbreak_model) as gen:
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.jailbreak_model, messages=[{"role": "user", "content": text}]
                    ),
                    self.timeout_seconds,
                )
                score = float(response.choices[0].message.content.strip())
                if not math.isfinite(score):
                    raise ValueError(f"not a probability: {score}")
            except Exception as error:  # noqa: BLE001  (never raise into NeMo)
                gen.update(level="ERROR", status_message=str(error)[:200])
                return self._failed(JAILBREAK_RAIL, error)
            gen.update(output={"score": score}, usage_details=usage_of(response))
        signal: Signal = "flag" if score >= self.jailbreak_threshold else "ok"
        record({"rail": JAILBREAK_RAIL, "verdict": signal, "score": round(score, 4)})
        return signal

    async def safety_model_check(self, text: str) -> Verdict:
        """Rail 2 (decides): is the message harmful, illegal or a jailbreak? The small LLM answers JSON.

        JSON mode (response_format) and temperature 0 are required: without JSON mode the model once
        returned broken JSON while planning. Anything but a clean "allow"/"block" verdict is "error".
        A block becomes "jailbreak" when Prompt Guard flagged this message (rail 1, read from the trace) or
        the model's category says jailbreak: the user then gets the "change how the assistant works" text.
        """
        with get_tracing().span("safety model", as_type="generation", input=text, model=self.safety_model) as gen:
            return await self._safety_check(text, gen)

    async def _safety_check(self, text: str, gen: Any) -> Verdict:
        """The body of safety_model_check, inside its Langfuse generation (`gen`, a no-op when tracing is off)."""
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.safety_model,
                    temperature=0,
                    max_tokens=SAFETY_MAX_TOKENS,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SAFETY_POLICY},
                        {"role": "user", "content": f"Message to classify: {text}"},
                    ],
                ),
                self.timeout_seconds,
            )
            data = json.loads(response.choices[0].message.content)
            verdict = data.get("verdict") if isinstance(data, dict) else None
            if verdict not in ("allow", "block"):
                raise ValueError(f"unexpected verdict: {verdict!r}")
        except Exception as error:  # noqa: BLE001
            gen.update(level="ERROR", status_message=str(error)[:200])
            return self._failed(SAFETY_RAIL, error)
        gen.update(output=data, usage_details=usage_of(response))
        category = str(data.get("category") or "") or None
        if verdict == "block" and (prompt_guard_flagged() or "jailbreak" in (category or "").lower()):
            verdict = "jailbreak"
        record({"rail": SAFETY_RAIL, "verdict": verdict, "category": category, "reason": data.get("reason")})
        return verdict

    def _failed(self, rail: str, error: BaseException) -> Verdict:
        """Log a failed check and record it; the flow will block the message as "unavailable"."""
        detail = " ".join(str(error).split())[:200]
        message = f"{type(error).__name__}: {detail}" if detail else type(error).__name__  # TimeoutError has no text
        logger.warning("Guard %s failed: %s", rail, message)
        record({"rail": rail, "verdict": "error", "error": message})
        return "error"
