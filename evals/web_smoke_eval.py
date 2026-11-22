"""Web search smoke eval (plannings/04, section 13): a few live questions straight to the web agent.

The web agent runs for real: the DuckDuckGo MCP server (started with uvx) and gpt-oss for the cited answer;
no orchestrator. The live web changes from day to day, so this is a SMOKE test: it checks the shape of each
outcome (evals/datasets/web_smoke.json, `expected`), not the exact words:

    cited      success, at least one source, every source an http(s) URL, every [n] in the answer a returned source
    rejected   stopped by the query check before any search (only the check_query step ran)
    not_found  no answer claimed (no results, or the results don't answer it): success False, no sources

Run from the repo root (needs network + Groq; ~1 minute):
    uv run --project services/orchestrator-api python evals/web_smoke_eval.py
Writes evals/reports/web-smoke-<date>.md and .json. check_case is a pure function, tested in
services/orchestrator-api/tests/test_guard_eval.py.
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any
from urllib.parse import urlparse

from common import DATASETS, REPO, REPORTS, git_commit, median_max, pct  # also puts the API on sys.path

from app.agents.citations import CITATION_MARK  # noqa: E402

DATASET = DATASETS / "web_smoke.json"
REJECTED_PREFIX = "I can't send internal company data"


@dataclass
class SmokeResult:
    """One live question: what was expected, what happened, and why it doesn't match (if it doesn't)."""

    id: str
    question: str
    expected: str
    outcome: str  # cited | rejected | not_found | failed
    passed: bool
    problems: list[str]
    sources: list[str]  # the cited URLs
    steps: list[str]
    seconds: float
    answer: str


def outcome_of(result: Any) -> str:
    """Classify a WebSearchResult: cited / rejected / not_found / failed (an error answer)."""
    if result.steps == ["check_query"]:
        return "rejected"
    if result.success:
        return "cited"
    return "failed" if result.error and "no results" not in (result.error or "") else "not_found"


def check_case(case: dict[str, Any], result: Any, seconds: float) -> SmokeResult:
    """Check one result against its expected outcome (pure)."""
    outcome, problems = outcome_of(result), []
    if outcome != case["expected"]:
        problems.append(f"expected {case['expected']}, got {outcome}")
    if case["expected"] == "cited":
        numbers = {source.n for source in result.sources}
        cited = {int(n) for n in CITATION_MARK.findall(result.answer)}
        if not result.sources:
            problems.append("no sources")
        if any(urlparse(source.url).scheme not in ("http", "https") for source in result.sources):
            problems.append("a source is not an http(s) link")
        if cited - numbers:
            problems.append(f"the answer cites {sorted(cited - numbers)} but no such source was returned")
        if not cited:
            problems.append("the answer has no [n] citation")
    if case["expected"] == "rejected" and not result.answer.startswith(REJECTED_PREFIX):
        problems.append("rejected without the company-data explanation")
    if case["expected"] == "not_found" and result.sources:
        problems.append("claims sources for a question with no real answer")
    return SmokeResult(
        id=case["id"], question=case["question"], expected=case["expected"], outcome=outcome,
        passed=not problems, problems=problems, sources=[source.url for source in result.sources],
        steps=result.steps, seconds=round(seconds, 1), answer=" ".join(result.answer.split()),
    )  # fmt: skip


def render_markdown(results: list[SmokeResult], meta: dict[str, Any]) -> str:
    """The report: one row per question, then each answer with its links."""
    passed = sum(r.passed for r in results)
    median, maximum = median_max([r.seconds for r in results])
    lines = [
        f"# Web search smoke eval: {meta['date']}",
        "",
        f"Live DuckDuckGo (`{meta['package']}` via MCP) + answer model `{meta['model']}` · commit `{meta['commit']}` "
        f"· run time {meta['minutes']} min. The live web changes: this checks the shape of each outcome, not exact "
        "answers. Dataset: `evals/datasets/web_smoke.json`.",
        "",
        f"**{pct(passed, len(results))} as expected** · time per question: median {median} s, max {maximum} s.",
        "",
        "| Case | Expected | Got | ✓ | Sources | Steps | Time |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        hosts = ", ".join(dict.fromkeys(urlparse(url).hostname or url for url in r.sources))
        lines.append(
            f"| {r.id} | {r.expected} | {r.outcome} | {'✅' if r.passed else '❌'} | {len(r.sources)}: {hosts} | "
            f"{' → '.join(r.steps)} | {r.seconds} s |"
        )
    lines += ["", "## Answers", ""]
    for r in results:
        lines += [f"### {r.id}: {r.question}", ""]
        lines += [f"- ❌ {problem}" for problem in r.problems]
        lines += [f"{r.answer[:600]}{'…' if len(r.answer) > 600 else ''}", ""]
        lines += [f"- {url}" for url in r.sources]
        lines += [""]
    return "\n".join(lines) + "\n"


async def run(cases: list[dict[str, Any]], pause: float) -> tuple[list[SmokeResult], dict[str, str]]:
    """Start the MCP server once, ask every question, close it."""
    from app.agents.web.graph import WebSearchAgent
    from app.agents.web.mcp import WebSearchMCP
    from app.config import get_settings

    settings = get_settings()
    results = []
    async with WebSearchMCP(settings) as mcp:
        if not mcp.available:
            raise SystemExit("The DuckDuckGo MCP server could not start (see the log above).")
        agent = WebSearchAgent(mcp, max_results=settings.web_max_results, snippet_chars=settings.web_snippet_chars)
        for number, case in enumerate(cases, start=1):
            started = time.perf_counter()
            result = check_case(case, await agent.ask(case["question"]), time.perf_counter() - started)
            results.append(result)
            print(f"[{number}/{len(cases)}] {'✓' if result.passed else '✗'} {case['id']:<18} {result.outcome}"
                  f" ({len(result.sources)} sources, {result.seconds}s)", flush=True)  # fmt: skip
            if pause and number < len(cases):
                await asyncio.sleep(pause)
    return results, {"package": settings.web_mcp_package, "model": settings.groq_strong_model}


def main() -> None:
    """CLI: run the smoke eval and write the report."""
    parser = argparse.ArgumentParser(description="Run the web search smoke eval (live).")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between questions")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s - %(message)s")

    cases = json.loads(DATASET.read_text())["cases"]
    started = time.perf_counter()
    results, info = asyncio.run(run(cases, args.pause))
    meta = {"date": date.today().isoformat(), "commit": git_commit(), **info,
            "minutes": round((time.perf_counter() - started) / 60, 1)}  # fmt: skip
    stem = REPORTS / f"web-smoke-{meta['date']}"
    stem.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "cases": [asdict(r) for r in results]}, indent=2) + "\n"
    )
    stem.with_suffix(".md").write_text(render_markdown(results, meta))
    report = stem.with_suffix(".md").relative_to(REPO)
    print(f"\n{pct(sum(r.passed for r in results), len(results))} as expected. Report: {report}")


if __name__ == "__main__":
    main()
