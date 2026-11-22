"""RAG eval for the Doc QA agent (plannings/02, section 12, stage R9): retrieval quality and answer quality.

The sample report (data/samples/q3-compliance-report.pdf: text, a multi-page table, charts, a flow diagram, a
scanned page) is uploaded into a fresh chat and processed by the real ingestion pipeline (Docling, vision,
chunking, embeddings), exactly as an upload in the UI. Then, for every question of
evals/datasets/rag_questions.json:

1. RETRIEVAL, four ways, no LLM needed:
       dense only · keyword only · hybrid (RRF fusion) · hybrid + cross-encoder rerank (what the agent uses)
   A hit is a retrieved chunk on an expected page. Metrics: hit@1, hit@3, hit@6 (6 = RAG_TOP_K) and MRR
   (the average of 1 / rank of the first hit).
2. ANSWERS, through the real Doc agent:
   - correctness: every expected fact is in the answer (a direct text check first; if that fails, a small LLM
     judge decides whether the answer says the same in other words). Chart readings must be "approximately"
     right, within the case's tolerance;
   - citation correctness: every cited chunk is on an expected page;
   - not-found accuracy: questions the document can't answer must get "not found", the others an answer.
3. THE REFUSAL THRESHOLD (RAG_MIN_RERANK_SCORE): the best rerank score of answerable vs not-in-document
   questions. If the two groups don't overlap, the midpoint between them is recommended.

Run from the repo root (needs Postgres + Groq; ingestion takes 1-3 minutes):
    uv run --project services/orchestrator-api python evals/rag_eval.py
    ... --session <chat id>   # reuse a chat where the sample PDF is already processed (with --keep earlier)
    ... --retrieval-only      # steps 1 and 3 only: no answers, so no Groq tokens (the search runs locally)
Writes evals/reports/rag-<date>.md and .json (rag-retrieval-<date>.* with --retrieval-only). The scorer is pure
functions, tested in services/orchestrator-api/tests/test_rag_eval.py.
"""

import argparse
import asyncio
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import httpx  # noqa: E402
from common import DATASETS, REPO, REPORTS, git_commit, median_max, pct  # also puts the API on sys.path

DATASET = DATASETS / "rag_questions.json"
MODES = ("dense", "keyword", "hybrid", "hybrid+rerank")
TOP_K = (1, 3, 6)
APPROXIMATE_WORDS = re.compile(r"approx|about|around|roughly|nearly|~|≈|estimat", re.IGNORECASE)
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
DOCUMENT_TIMEOUT_SECONDS = 600


# --------------------------------------------------------------------------- scoring (pure)


def first_hit_rank(pages: list[tuple[int, int]], expected_pages: list[int]) -> int | None:
    """1-based rank of the first retrieved chunk that is on an expected page (chunks can span pages), or None."""
    for rank, (start, end) in enumerate(pages, start=1):
        if any(start <= page <= end for page in expected_pages):
            return rank
    return None


def retrieval_metrics(ranks: list[int | None]) -> dict[str, float]:
    """hit@1/3/6 (share of questions with a hit in the top k) and MRR, over the given first-hit ranks."""
    if not ranks:
        return {**{f"hit@{k}": 0.0 for k in TOP_K}, "mrr": 0.0}
    metrics = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / len(ranks) for k in TOP_K}
    metrics["mrr"] = sum(1 / r for r in ranks if r) / len(ranks)
    return {name: round(value, 3) for name, value in metrics.items()}


def normalize_text(text: str) -> str:
    """Lower-case, unify dashes and spaces, drop thousands separators, so "25,000" == "25000"."""
    text = text.casefold().replace("–", "-").replace("—", "-").replace(" ", " ").replace("\xa0", " ")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    return " ".join(text.split())


def facts_in_answer(answer: str, facts: list[str], approximate: bool = False, tolerance: float = 0.0) -> list[str]:
    """The expected facts the answer does NOT contain (empty = all present). Direct text check, no LLM.

    approximate=True (a value read off a chart axis): the first fact is a number; the answer must contain a
    number within `tolerance` of it AND say it's approximate ("approximately", "about", "~"...).
    """
    text = normalize_text(answer)
    missing = []
    for index, fact in enumerate(facts):
        if approximate and index == 0:
            target = float(fact)
            numbers = [float(n.replace(",", "")) for n in NUMBER.findall(text)]
            if not any(abs(n - target) <= tolerance * target for n in numbers) or not APPROXIMATE_WORDS.search(answer):
                missing.append(f"≈{fact} (±{round(tolerance * 100)}%, said to be approximate)")
        elif normalize_text(fact) not in text:
            missing.append(fact)
    return missing


def citations_on_expected_pages(citations: list[dict], expected_pages: list[int]) -> bool:
    """Every cited chunk touches an expected page (a chunk can span pages). No citations -> False."""
    if not citations:
        return False
    return all(any(c["page_start"] <= p <= c["page_end"] for p in expected_pages) for c in citations)


def recommend_threshold(answerable: list[float], not_in_doc: list[float]) -> tuple[float | None, str]:
    """RAG_MIN_RERANK_SCORE from the best rerank scores: the midpoint if the groups separate, else None + why."""
    if not answerable or not not_in_doc:
        return None, "not enough questions in one of the groups"
    lowest_answerable, highest_unanswerable = min(answerable), max(not_in_doc)
    if highest_unanswerable >= lowest_answerable:
        return None, (
            f"the groups overlap (lowest answerable {lowest_answerable:.2f} ≤ highest not-in-document "
            f"{highest_unanswerable:.2f}): a threshold would refuse answerable questions, so none is set; the answer "
            "model's own answer_found decides"
        )
    threshold = round((lowest_answerable + highest_unanswerable) / 2, 2)
    return threshold, (
        f"midpoint between the highest not-in-document score ({highest_unanswerable:.2f}) and the lowest "
        f"answerable score ({lowest_answerable:.2f})"
    )


@dataclass
class RagCaseResult:
    """One question: where retrieval found it in each mode, and how the agent answered."""

    id: str
    kind: str
    question: str
    expected_pages: list[int]
    ranks: dict[str, int | None]  # mode -> rank of the first hit (None = not in the candidates)
    best_rerank_score: float | None
    grounded: bool
    answer: str
    cited_pages: list[list[int]]
    correct: bool | None  # None for not-in-document questions
    judged_by: str  # "text" | "llm judge" | "-"
    missing: list[str]
    citations_ok: bool | None
    not_found_ok: bool  # not-in-document -> not grounded; answerable -> grounded
    seconds: float
    notes: list[str] = field(default_factory=list)


def summarize(results: list[RagCaseResult]) -> dict[str, Any]:
    """Retrieval per mode, answer quality, the threshold recommendation, latency."""
    answerable = [r for r in results if r.kind != "not_in_doc"]
    not_in_doc = [r for r in results if r.kind == "not_in_doc"]
    threshold, why = recommend_threshold(
        [r.best_rerank_score for r in answerable if r.best_rerank_score is not None],
        [r.best_rerank_score for r in not_in_doc if r.best_rerank_score is not None],
    )
    median, maximum = median_max([r.seconds for r in results])
    return {
        "questions": len(results),
        "answerable": len(answerable),
        "not_in_doc": len(not_in_doc),
        "retrieval": {mode: retrieval_metrics([r.ranks.get(mode) for r in answerable]) for mode in MODES},
        "correct": sum(bool(r.correct) for r in answerable),
        "judged_by_llm": sum(r.judged_by == "llm judge" for r in answerable),
        "citations_ok": sum(bool(r.citations_ok) for r in answerable),
        "answered_when_possible": sum(r.grounded for r in answerable),
        "refused_when_not_in_doc": sum(not r.grounded for r in not_in_doc),
        "by_kind": {
            kind: {"cases": len(group), "correct": sum(bool(r.correct) for r in group)}
            for kind in ("text", "table", "picture", "ocr")
            if (group := [r for r in answerable if r.kind == kind])
        },
        "threshold": threshold,
        "threshold_reason": why,
        "scores_answerable": sorted(r.best_rerank_score for r in answerable if r.best_rerank_score is not None),
        "scores_not_in_doc": sorted(r.best_rerank_score for r in not_in_doc if r.best_rerank_score is not None),
        "latency_median_s": median,
        "latency_max_s": maximum,
    }


def render_markdown(results: list[RagCaseResult], summary: dict[str, Any], meta: dict[str, Any]) -> str:
    """The report: retrieval comparison, answer quality, the threshold, then every question."""
    s, a = summary, summary["answerable"]
    answers = not meta.get("retrieval_only")
    lines = [
        f"# RAG eval (Doc QA agent): {meta['date']}",
        "",
        f"Document `{meta['document']}` ({meta['chunks']} chunks) · embeddings `{meta['embedding_model']}` · rerank "
        f"`{meta['rerank_model']}` · answer model `{meta['answer_model']}` · commit `{meta['commit']}` · "
        f"{s['questions']} questions ({a} answerable, {s['not_in_doc']} not in the document) · run time "
        f"{meta['minutes']} min. Dataset: `evals/datasets/rag_questions.json`.",
        "",
        "## Retrieval: which search finds the right page",
        "",
        "A hit = a retrieved chunk on an expected page. hit@k = share of the answerable questions with a hit in the "
        "top k (the agent uses the top 6). MRR = average of 1 / rank of the first hit.",
        "",
        "| Mode | hit@1 | hit@3 | hit@6 | MRR |",
        "|---|---|---|---|---|",
        *(
            f"| {mode} | {m['hit@1']:.0%} | {m['hit@3']:.0%} | {m['hit@6']:.0%} | {m['mrr']:.3f} |"
            for mode, m in s["retrieval"].items()
        ),
        "",
        *(_answer_section(s, a) if answers else ["_Retrieval only (`--retrieval-only`): the answers aren't run._", ""]),
        "## Refusal threshold (RAG_MIN_RERANK_SCORE)",
        "",
        f"Best rerank score per question: answerable {', '.join(f'{x:.2f}' for x in s['scores_answerable'])}; "
        f"not in the document {', '.join(f'{x:.2f}' for x in s['scores_not_in_doc'])}.",
        "",
        f"**Recommendation: {'no threshold' if s['threshold'] is None else s['threshold']}** "
        f"({s['threshold_reason']}).",
        "",
        "## Questions",
        "",
        "| Q | Kind | Pages | Rank: dense / keyword / hybrid / +rerank | Best score | Answer | Citations | Time |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ranks = " / ".join("—" if r.ranks.get(m) is None else str(r.ranks[m]) for m in MODES)
        verdict = "✅" if r.correct else ("—" if r.correct is None else "❌")
        if r.kind == "not_in_doc":
            verdict = "✅ not found" if r.not_found_ok else "❌ answered"
        score = "" if r.best_rerank_score is None else f"{r.best_rerank_score:.2f}"
        cites = "" if r.citations_ok is None else ("✅" if r.citations_ok else "❌")
        lines.append(
            f"| {r.id} | {r.kind} | {','.join(map(str, r.expected_pages)) or '—'} | {ranks} | {score} | {verdict} | "
            f"{cites} | {r.seconds} s |"
        )
    misses = [r for r in results if r.correct is False or not r.not_found_ok or r.citations_ok is False]
    if misses and answers:
        lines += ["", "## Misses", ""]
        for r in misses:
            lines += [f"### {r.id} ({r.kind}): {r.question}", ""]
            lines += [f"- missing: {m}" for m in r.missing]
            if r.citations_ok is False:
                lines.append(f"- cited pages {r.cited_pages}, expected {r.expected_pages}")
            if not r.not_found_ok:
                lines.append("- answered a question the document can't answer" if r.kind == "not_in_doc" else
                             "- said not found, but the document has the answer")  # fmt: skip
            lines += ["", f"Answer: {r.answer[:500]}", ""]
    return "\n".join(lines) + "\n"


def _answer_section(s: dict[str, Any], a: int) -> list[str]:
    """The report's "Answers" section (left out of a retrieval-only report)."""
    return [
        "## Answers",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Correct answers** (all expected facts) | **{pct(s['correct'], a)}** ({s['judged_by_llm']} decided by "
        "the LLM judge, the rest by a direct text check) |",
        f"| Citations on the expected pages | {pct(s['citations_ok'], a)} |",
        f"| Answered when the document has the answer | {pct(s['answered_when_possible'], a)} |",
        f'| Said "not found" when it doesn\'t | {pct(s["refused_when_not_in_doc"], s["not_in_doc"])} |',
        f"| Latency per answer | median {s['latency_median_s']} s, max {s['latency_max_s']} s |",
        "",
        "| Kind | Correct |",
        "|---|---|",
        *(f"| {kind} | {pct(g['correct'], g['cases'])} |" for kind, g in s["by_kind"].items()),
        "",
    ]


# --------------------------------------------------------------------------- the LLM judge


JUDGE_PROMPT = """You check whether an ANSWER states every FACT in the list. Same meaning counts (different \
wording, units written out, "over" vs "above"); a different number does not. Reply with JSON only: \
{{"all_present": true | false, "missing": ["<facts that are not stated>"]}}

FACTS: {facts}

ANSWER: {answer}"""


async def judge(llm: Any, answer: str, facts: list[str]) -> list[str]:
    """Ask the fast model whether the answer states the facts in other words; returns the facts it says are missing."""
    reply = await llm.ainvoke(JUDGE_PROMPT.format(facts=json.dumps(facts), answer=answer))
    data = json.loads(reply.content if isinstance(reply.content, str) else str(reply.content))
    return [] if data.get("all_present") else [str(m) for m in data.get("missing") or facts]


# --------------------------------------------------------------------------- running


async def ingest(http: httpx.AsyncClient, path: Path) -> tuple[str, int]:
    """Upload the sample PDF into a new chat and wait until it's processed; returns (chat id, chunk count)."""
    files = {"file": (path.name, path.read_bytes(), "application/pdf")}
    response = await http.post("/api/documents", files=files)
    response.raise_for_status()
    session_id = response.json()["session"]["id"]
    print(f"Uploaded {path.name} into chat {session_id}; processing (Docling, vision, embeddings)...", flush=True)
    deadline = time.monotonic() + DOCUMENT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        [document] = (await http.get(f"/api/sessions/{session_id}/documents")).json()
        if document["status"] == "ready":
            return session_id, document["chunk_count"]
        if document["status"] == "failed":
            raise RuntimeError(f"processing failed: {document['error']}")
        await asyncio.sleep(3)
    raise TimeoutError("the document was not processed in time")


async def evaluate_case(
    case: dict[str, Any], session_id: Any, doc_agent: Any, judge_llm: Any, answer: bool = True
) -> RagCaseResult:
    """Retrieval in the four modes, then (unless answer=False) the real agent's answer, scored."""
    retriever, question, expected = doc_agent.retriever, case["question"], case["expected_pages"]
    dense_ids, keyword_ids = await asyncio.gather(
        retriever.dense(session_id, question), retriever.keyword(session_id, question)
    )
    lists = {
        "dense": await retriever.fuse(dense_ids, []),
        "keyword": await retriever.fuse([], keyword_ids),
        "hybrid": await retriever.fuse(dense_ids, keyword_ids),
    }
    lists["hybrid+rerank"] = await retriever.rerank(question, list(lists["hybrid"]))
    ranks = {mode: first_hit_rank([(h.page_start, h.page_end) for h in hits], expected) for mode, hits in lists.items()}
    best = lists["hybrid+rerank"][0].rerank_score if lists["hybrid+rerank"] else None
    if not answer:
        return RagCaseResult(
            id=case["id"], kind=case["kind"], question=question, expected_pages=expected, ranks=ranks,
            best_rerank_score=None if best is None else round(best, 3), grounded=False, answer="", cited_pages=[],
            correct=None, judged_by="-", missing=[], citations_ok=None, not_found_ok=True, seconds=0.0,
            notes=["retrieval only"],
        )  # fmt: skip

    started = time.perf_counter()
    result = await doc_agent.ask(question, session_id=session_id)
    seconds = round(time.perf_counter() - started, 1)
    citations = [c.model_dump() for c in result.citations]
    if case["kind"] == "not_in_doc":
        correct, judged_by, missing, citations_ok = None, "-", [], None
    else:
        missing = facts_in_answer(result.answer, case["expected_facts"], case.get("approximate", False),
                                  case.get("tolerance", 0.0))  # fmt: skip
        judged_by = "text"
        if missing and not case.get("approximate") and result.grounded:
            missing, judged_by = await judge(judge_llm, result.answer, case["expected_facts"]), "llm judge"
        correct, citations_ok = result.grounded and not missing, citations_on_expected_pages(citations, expected)
    return RagCaseResult(
        id=case["id"], kind=case["kind"], question=question, expected_pages=expected, ranks=ranks,
        best_rerank_score=None if best is None else round(best, 3), grounded=result.grounded,
        answer=" ".join(result.answer.split()), cited_pages=[[c["page_start"], c["page_end"]] for c in citations],
        correct=correct, judged_by=judged_by, missing=missing, citations_ok=citations_ok,
        not_found_ok=result.grounded != (case["kind"] == "not_in_doc"), seconds=seconds,
    )  # fmt: skip


async def run(
    cases: list[dict], document: Path, session: str | None, keep: bool, pause: float, answers: bool = True
) -> tuple[list, dict]:
    """Start the app in-process (its ingestion worker and Doc agent), process the PDF, evaluate, clean up."""
    from uuid import UUID

    from app.config import get_settings
    from app.llm import get_llm
    from app.main import app

    logging.getLogger().setLevel(logging.WARNING)
    settings = get_settings()
    judge_llm = get_llm("fast", max_tokens=200).bind(response_format={"type": "json_object"})
    results, chunks = [], 0
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://eval", timeout=600) as http:
            if session is None:
                session, chunks = await ingest(http, document)
            else:
                [doc] = (await http.get(f"/api/sessions/{session}/documents")).json()
                chunks = doc["chunk_count"]
            try:
                for number, case in enumerate(cases, start=1):
                    result = await evaluate_case(case, UUID(session), app.state.doc_agent, judge_llm, answers)
                    results.append(result)
                    verdict = {True: "✓", False: "✗", None: "✓" if result.not_found_ok else "✗"}[result.correct]
                    ranks = " ".join(f"{m}:{result.ranks[m] or '-'}" for m in MODES)
                    shown = verdict if answers else f"score {result.best_rerank_score}"
                    print(f"[{number:>2}/{len(cases)}] {shown} {case['id']} {case['kind']:<10} {ranks}", flush=True)
                    if pause and answers and number < len(cases):
                        await asyncio.sleep(pause)
            finally:
                if not keep:
                    for doc in (await http.get(f"/api/sessions/{session}/documents")).json():
                        await http.delete(f"/api/documents/{doc['id']}")
                    await http.delete(f"/api/sessions/{session}")
                else:
                    print(f"Kept chat {session} (reuse it with --session {session})")
    meta = {
        "embedding_model": settings.rag_embedding_model, "rerank_model": settings.rag_rerank_model,
        "answer_model": settings.groq_strong_model, "chunks": chunks,
    }  # fmt: skip
    return results, meta


def main() -> None:
    """CLI: run the RAG eval and write the report."""
    parser = argparse.ArgumentParser(description="Run the RAG eval for the Doc QA agent.")
    parser.add_argument("--session", help="a chat where the sample PDF is already processed (skips ingestion)")
    parser.add_argument("--keep", action="store_true", help="keep the eval chat (to reuse it with --session)")
    parser.add_argument("--pause", type=float, default=4.0, help="seconds between questions (Groq tokens/min limit)")
    parser.add_argument("--only", nargs="*", help="question ids to run (default: all)")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the answers (no Groq tokens needed)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s - %(message)s")

    data = json.loads(DATASET.read_text())
    cases = [q for q in data["questions"] if not args.only or q["id"] in set(args.only)]
    document = REPO / data["document"]
    started = time.perf_counter()
    answers = not args.retrieval_only
    results, info = asyncio.run(run(cases, document, args.session, args.keep, args.pause, answers))
    summary = summarize(results)
    meta = {"date": date.today().isoformat(), "commit": git_commit(), "document": data["document"], **info,
            "minutes": round((time.perf_counter() - started) / 60, 1), "retrieval_only": not answers}  # fmt: skip
    stem = REPORTS / f"rag-{'' if answers else 'retrieval-'}{meta['date']}"
    payload = {"meta": meta, "summary": summary, "cases": [asdict(r) for r in results]}
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    stem.with_suffix(".md").write_text(render_markdown(results, summary, meta))
    rerank = summary["retrieval"]["hybrid+rerank"]
    correct = f"correct {pct(summary['correct'], summary['answerable'])}; " if answers else ""
    print(f"\nhit@6 (hybrid+rerank) {rerank['hit@6']:.0%}, MRR {rerank['mrr']:.3f}; {correct}"
          f"threshold {summary['threshold']}.")  # fmt: skip
    print(f"Report: {stem.with_suffix('.md').relative_to(REPO)}")


if __name__ == "__main__":
    main()
