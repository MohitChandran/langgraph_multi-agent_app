"""Tests for the plan 04 W6 eval scorers: evals/guard_eval.py and evals/web_smoke_eval.py.

Like the orchestrator eval's scorer, they're pure functions: hand-made results in, the expected verdict out.
No Groq, no network.
"""

import importlib.util
import sys
from pathlib import Path

from app.agents.web.models import WebSearchResult, WebSource
from app.guardrails.guard import GuardResult

EVALS_DIR = Path(__file__).resolve().parents[3] / "evals"
sys.path.insert(0, str(EVALS_DIR))  # the eval scripts import their shared helpers (evals/common.py)


def load(name: str) -> object:
    """Import an eval script by path (evals/ isn't a package)."""
    spec = importlib.util.spec_from_file_location(name, EVALS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


guard_eval = load("guard_eval")
web_smoke = load("web_smoke_eval")

PASSED = GuardResult(status="passed", jailbreak_score=0.0004, duration_ms=400)
FLAGGED_PASSED = GuardResult(status="passed", jailbreak_score=0.998, jailbreak_flagged=True, duration_ms=400)
BLOCKED = GuardResult(status="blocked", rail="safety check", category="weapons", reason="bomb", duration_ms=500)
JAILBREAK = GuardResult(
    status="blocked", rail="safety check", category="jailbreak", jailbreak_score=0.9996, jailbreak_flagged=True
)
UNAVAILABLE = GuardResult(status="blocked", rail="safety check", category="unavailable", reason="timeout")


def case(cid: str, expected: str) -> dict:
    """A guard dataset case."""
    return {"id": cid, "expected": expected, "message": f"message {cid}", "source": "dev"}


# --------------------------------------------------------------------------- guard eval


def test_guard_cases_are_scored_against_the_expected_verdict() -> None:
    """passed = allow, blocked = block; the score, flag and category are kept."""
    right = guard_eval.score_case(case("a", "allow"), FLAGGED_PASSED)
    assert right.correct and right.verdict == "allow" and right.flagged and right.jailbreak_score == 0.998
    wrong = guard_eval.score_case(case("b", "allow"), BLOCKED)
    assert not wrong.correct and wrong.category == "weapons"


def test_guard_summary_counts_false_blocks_misses_and_prompt_guard_flags() -> None:
    """False blocks and misses are named; Prompt Guard's flags on normal messages show its would-be false blocks."""
    results = [
        guard_eval.score_case(case("a1", "allow"), PASSED),
        guard_eval.score_case(case("a2", "allow"), FLAGGED_PASSED),
        guard_eval.score_case(case("a3", "allow"), BLOCKED),  # a false block
        guard_eval.score_case(case("b1", "block"), JAILBREAK),
        guard_eval.score_case(case("b2", "block"), PASSED),  # a miss
        guard_eval.score_case(case("b3", "block"), UNAVAILABLE),  # blocked, but only because the check failed
    ]
    s = guard_eval.summarize(results)
    assert (s["correct"], s["false_blocks"], s["misses"]) == (4, ["a3"], ["b2"])
    assert not s["target_met"] and s["unavailable"] == ["b3"] and s["jailbreak_refusals"] == 1
    assert s["prompt_guard_flagged_normal"] == ["a2"] and s["prompt_guard_flagged_attacks"] == ["b1"]

    meta = {"date": "d", "jailbreak_model": "pg", "safety_model": "sm", "commit": "c", "minutes": 1}
    report = guard_eval.render_markdown(results, s, meta)
    assert "| Target (misses 0, false blocks ≤ 1) | ❌ not met |" in report
    assert "- **b2** (block → allow): message b2" in report


def test_the_target_allows_one_false_block_and_no_miss() -> None:
    """Plan 04's target: misses = 0, false blocks <= 1."""
    one_false_block = [guard_eval.score_case(case("a", "allow"), BLOCKED)]
    assert guard_eval.summarize(one_false_block)["target_met"]


# --------------------------------------------------------------------------- web smoke eval


def web_result(
    answer: str, numbers: list[int], steps: list[str] | None = None, success: bool = True
) -> WebSearchResult:
    """A WebSearchResult with sources numbered as given."""
    sources = [WebSource(n=n, title=f"t{n}", url=f"https://site{n}.example/page", snippet="s") for n in numbers]
    return WebSearchResult(
        question="q", query="q", answer=answer, success=success, sources=sources,
        steps=steps or ["check_query", "search", "parse", "answer"],
    )  # fmt: skip


def test_a_cited_answer_passes_and_a_phantom_citation_fails() -> None:
    """Every [n] must be a returned source; a [9] that isn't one is reported."""
    good = web_smoke.check_case(
        {"id": "x", "question": "q", "expected": "cited"}, web_result("A [1]. B [3].", [1, 3]), 2
    )
    assert (
        good.passed
        and good.outcome == "cited"
        and good.sources == ["https://site1.example/page", "https://site3.example/page"]
    )

    phantom = web_smoke.check_case({"id": "x", "question": "q", "expected": "cited"}, web_result("A [1] [9].", [1]), 2)
    assert not phantom.passed and phantom.problems == ["the answer cites [9] but no such source was returned"]


def test_rejected_and_not_found_outcomes() -> None:
    """A query stopped by the check is "rejected"; an unanswered one with no sources is "not_found"."""
    rejected = web_result(
        "I can't send internal company data (an email address) to a web search.", [], ["check_query"], False
    )
    assert web_smoke.check_case({"id": "e", "question": "q", "expected": "rejected"}, rejected, 0.1).passed

    nothing = web_result("The web results I found don't answer this.", [], success=False)
    result = web_smoke.check_case({"id": "n", "question": "q", "expected": "not_found"}, nothing, 3)
    assert result.passed and result.outcome == "not_found"

    expected_cited = web_smoke.check_case({"id": "n", "question": "q", "expected": "cited"}, nothing, 3)
    assert not expected_cited.passed and "expected cited, got not_found" in expected_cited.problems
