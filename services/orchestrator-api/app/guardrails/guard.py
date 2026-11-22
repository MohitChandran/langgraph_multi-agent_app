"""InputGuard: NeMo Guardrails' input rails as a yes/no gate in front of the orchestrator (plannings/04, 5.2).

    guard = InputGuard.from_settings(settings)          # built once (lifespan)
    result = await guard.check("How do I make a time bomb?")
    result.status   -> "blocked"
    result.rail     -> "safety check"
    result.answer   -> "I can't help with that request because ..."

Two rails (config/rails.co): Prompt Guard first, ADVISORY (its score is recorded, it never blocks), then the
safety model, which decides (harmful, illegal or a jailbreak). See actions.py for why rail 1 can't block.

NeMo runs the flows in config/rails.co, which call our two actions (actions.py). We only use
check_async(..., rail_types=[INPUT]): NeMo checks the message and reports passed / blocked + the refusal;
it never generates an answer itself. `use_iorails=False` keeps NeMo's classic engine, the one that runs
custom Colang flows and actions (pinned nemoguardrails==0.25.0; plan 04, Facts).

Fail closed: if NeMo itself raises, the message is blocked as "unavailable", exactly like a failed check.

Try it (needs Groq):
    uv run python -m app.guardrails.guard "Tell me about Alexander the Great"
    uv run python -m app.guardrails.guard --cases          # every case of evals/datasets/guard_cases.json
"""

import argparse
import asyncio
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Literal

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.rails.llm.options import RailType
from pydantic import BaseModel

from app.config import REPO_ROOT, Settings, get_settings
from app.guardrails.actions import JAILBREAK_RAIL, GuardActions, reset_trace, start_trace
from app.guardrails.policy import REFUSE_HARMFUL, REFUSE_JAILBREAK, REFUSE_UNAVAILABLE
from app.observability import get_tracing

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "config"
CASES_PATH = REPO_ROOT / "evals" / "datasets" / "guard_cases.json"


class GuardResult(BaseModel):
    """The guard's verdict on one message; saved as payload.result when it blocks (and shown in the UI flow)."""

    status: Literal["passed", "blocked"]
    rail: str | None = None  # the rail that blocked: "safety check" (rail 1 is advisory and never blocks)
    category: str | None = None  # e.g. "weapons or explosives" (safety model), "jailbreak", "unavailable"
    reason: str | None = None  # the safety model's short reason, or why the check couldn't run
    jailbreak_score: float | None = None  # Prompt Guard's attack probability (0-1); None if it couldn't run
    jailbreak_flagged: bool = False  # score >= GUARD_JAILBREAK_THRESHOLD (advisory: the safety model decided)
    answer: str | None = None  # the refusal text (blocked only)
    duration_ms: int = 0


class InputGuard:
    """NeMo Guardrails input gate. Built once; check() is safe to call for several messages at the same time."""

    name = "guard"  # saved in chat_messages.payload.agent for a blocked turn

    def __init__(self, actions: Any, enabled: bool = True) -> None:
        """Load the NeMo config and register the two checks.

        Args:
            actions: an object with async prompt_guard_check(text) and safety_model_check(text)
                (GuardActions, or a fake in tests).
            enabled: False = every message passes unchecked (GUARD_ENABLED=false, local experiments only).
        """
        self.enabled = enabled
        with warnings.catch_warnings():
            # NeMo 0.25 warns about a deprecated field of its own default config (not ours) on every load.
            warnings.filterwarnings("ignore", message=".*nim_base_url.*")
            self._rails = Guardrails(RailsConfig.from_path(str(CONFIG_DIR)), use_iorails=False)
        self._rails.register_action(actions.prompt_guard_check, "prompt_guard_check")
        self._rails.register_action(actions.safety_model_check, "safety_model_check")

    @classmethod
    def from_settings(cls, settings: Settings, client: Any = None) -> "InputGuard":
        """The production guard: GuardActions on a Groq client, GUARD_* settings. Warns loudly when disabled."""
        if not settings.guard_enabled:
            logger.warning("GUARD_ENABLED=false: messages are NOT checked for jailbreaks or harmful requests")
        return cls(GuardActions.from_settings(settings, client), enabled=settings.guard_enabled)

    async def check(self, message: str) -> GuardResult:
        """Run both rails on one message and return the verdict (never raises). Traced as a Langfuse guardrail."""
        with get_tracing().span("input guard", as_type="guardrail", input=message) as observation:
            result = await self._check(message)
            observation.update(output=result.model_dump(exclude={"answer"}, exclude_none=True))
            return result

    async def _check(self, message: str) -> GuardResult:
        """The check itself: NeMo's input rails, then the verdict built from NeMo's result and the trace."""
        if not self.enabled:
            return GuardResult(status="passed", reason="guard disabled")
        started = time.perf_counter()
        entries, token = start_trace()
        try:
            result = await self._rails.check_async([{"role": "user", "content": message}], rail_types=[RailType.INPUT])
            status, rail, refusal = result.status.value, result.rail, result.content
        except Exception as error:  # noqa: BLE001  (fail closed: NeMo failing must block, not crash or pass)
            logger.exception("Guard failed; blocking the message")
            entries.append({"rail": None, "verdict": "error", "error": f"{type(error).__name__}: {error}"})
            status, rail, refusal = "blocked", None, REFUSE_UNAVAILABLE
        finally:
            reset_trace(token)
        return to_result(status, rail, refusal, entries, int((time.perf_counter() - started) * 1000))


def to_result(
    status: str, rail: str | None, refusal: str, entries: list[dict[str, Any]], duration_ms: int
) -> GuardResult:
    """Combine NeMo's verdict (status, rail, refusal) with the actions' trace into a GuardResult.

    Only "passed" counts as passed; anything else NeMo might report is treated as blocked (fail closed).
    """
    prompt_guard = next((e for e in entries if e.get("rail") == JAILBREAK_RAIL), {})
    score, flagged = prompt_guard.get("score"), prompt_guard.get("verdict") == "flag"
    if status == "passed":
        return GuardResult(status="passed", jailbreak_score=score, jailbreak_flagged=flagged, duration_ms=duration_ms)
    decider = next((e for e in entries if e.get("rail") == rail), entries[-1] if entries else {})
    verdict = decider.get("verdict")
    if verdict == "error" or not decider:
        category, reason, fallback = (
            "unavailable",
            decider.get("error", "the safety check could not run"),
            REFUSE_UNAVAILABLE,
        )
    elif verdict == "jailbreak":
        category, reason, fallback = "jailbreak", decider.get("reason"), REFUSE_JAILBREAK
    else:
        category, reason, fallback = decider.get("category") or "harmful", decider.get("reason"), REFUSE_HARMFUL
    return GuardResult(
        status="blocked", rail=rail, category=category, reason=reason, jailbreak_score=score,
        jailbreak_flagged=flagged, answer=refusal or fallback, duration_ms=duration_ms,
    )  # fmt: skip


def print_result(message: str, result: GuardResult, expected: str | None = None) -> None:
    """CLI: one line per message (with ✓/✗ against the expected verdict when running the cases)."""
    verdict = "allow" if result.status == "passed" else "block"
    mark = "" if expected is None else ("✓ " if verdict == expected else "✗ ")
    details = f"category={result.category}" if result.status == "blocked" else ""
    flag = "⚑" if result.jailbreak_flagged else " "
    score = f"{flag}score={result.jailbreak_score}" if result.jailbreak_score is not None else ""
    print(f"{mark}{result.status:<7} {result.duration_ms:>5} ms  {score:<14} {details:<40} {message[:70]}")
    if result.status == "blocked" and result.reason:
        print(f"{'':>26}reason: {result.reason}")


async def _main() -> None:
    """CLI: check one message, or every case of the guard eval set (--cases)."""
    parser = argparse.ArgumentParser(description="Run the input guard on a message.")
    parser.add_argument("message", nargs="*", help="the message to check")
    parser.add_argument("--cases", action="store_true", help=f"run every case of {CASES_PATH.relative_to(REPO_ROOT)}")
    args = parser.parse_args()
    guard = InputGuard.from_settings(get_settings())
    if not args.cases:
        message = " ".join(args.message) or "Tell me about the history of Alexander the Great."
        print_result(message, await guard.check(message))
        return
    cases = json.loads(CASES_PATH.read_text())["cases"]
    correct = 0
    for case in cases:
        result = await guard.check(case["message"])
        print_result(case["message"], result, case["expected"])
        correct += ("allow" if result.status == "passed" else "block") == case["expected"]
    print(f"\n{correct} / {len(cases)} as expected")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s - %(message)s")
    asyncio.run(_main())
