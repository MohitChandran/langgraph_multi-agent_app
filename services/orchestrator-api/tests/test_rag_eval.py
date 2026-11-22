"""Tests for the RAG eval's scorer (evals/rag_eval.py, plan 02 stage R9): pure functions, no models, no database."""

import importlib.util
import json
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parents[3] / "evals"
sys.path.insert(0, str(EVALS_DIR))  # the eval scripts import their shared helpers (evals/common.py)
spec = importlib.util.spec_from_file_location("rag_eval", EVALS_DIR / "rag_eval.py")
rag_eval = importlib.util.module_from_spec(spec)
sys.modules["rag_eval"] = rag_eval  # dataclasses look their module up by name
spec.loader.exec_module(rag_eval)


def test_first_hit_rank_counts_chunks_that_span_an_expected_page() -> None:
    """A chunk on pages 1-2 is a hit for page 2; the rank is 1-based; no hit -> None."""
    assert rag_eval.first_hit_rank([(4, 4), (1, 2), (2, 2)], [2]) == 2
    assert rag_eval.first_hit_rank([(4, 4)], [2]) is None


def test_retrieval_metrics() -> None:
    """hit@k = share of questions with the first hit within k; MRR averages 1/rank (a miss counts 0)."""
    metrics = rag_eval.retrieval_metrics([1, 2, 5, None])
    assert metrics == {"hit@1": 0.25, "hit@3": 0.5, "hit@6": 0.75, "mrr": round((1 + 0.5 + 0.2) / 4, 3)}


def test_facts_are_found_through_formatting_differences() -> None:
    """Case, thousands separators and dash styles don't matter; a missing fact is reported."""
    answer = "Invoices ABOVE USD 25000 need approval; cloud contracts need a 90–day termination for convenience."
    assert rag_eval.facts_in_answer(answer, ["above USD 25,000", "90-day termination for convenience"]) == []
    assert rag_eval.facts_in_answer(answer, ["Net 15"]) == ["Net 15"]


def test_chart_readings_must_be_close_and_said_to_be_approximate() -> None:
    """Within ±10% AND hedged ("approximately"); an exact-sounding or far-off number fails."""
    facts = ["420", "USD thousands"]
    assert rag_eval.facts_in_answer("Approximately 430 USD thousands.", facts, True, 0.1) == []
    assert rag_eval.facts_in_answer("Exactly 430 USD thousands.", facts, True, 0.1) == [
        "≈420 (±10%, said to be approximate)"
    ]
    assert rag_eval.facts_in_answer("About 520 USD thousands.", facts, True, 0.1) == [
        "≈420 (±10%, said to be approximate)"
    ]


def test_citations_must_all_be_on_expected_pages() -> None:
    """Every cited chunk must touch an expected page; no citations at all is not correct either."""
    on_page = [{"page_start": 2, "page_end": 3}, {"page_start": 3, "page_end": 3}]
    assert rag_eval.citations_on_expected_pages(on_page, [3])
    assert not rag_eval.citations_on_expected_pages([*on_page, {"page_start": 5, "page_end": 5}], [3])
    assert not rag_eval.citations_on_expected_pages([], [3])


def test_threshold_is_the_midpoint_only_when_the_groups_separate() -> None:
    """Separate groups -> the midpoint; overlapping groups -> no threshold, with the reason."""
    threshold, why = rag_eval.recommend_threshold([2.5, 4.0, 6.1], [-3.0, 0.5])
    assert threshold == 1.5 and why.startswith("midpoint")
    threshold, why = rag_eval.recommend_threshold([0.2, 4.0], [1.0])
    assert threshold is None and "overlap" in why


def test_the_dataset_is_valid() -> None:
    """Every answerable question has pages and facts; approximate ones a tolerance; not-in-doc ones neither."""
    data = json.loads((EVALS_DIR / "datasets" / "rag_questions.json").read_text())
    for q in data["questions"]:
        if q["kind"] == "not_in_doc":
            assert not q["expected_pages"] and not q["expected_facts"], q["id"]
        else:
            assert q["expected_pages"] and q["expected_facts"], q["id"]
        if q.get("approximate"):
            assert 0 < q["tolerance"] < 1 and float(q["expected_facts"][0]), q["id"]


def test_a_retrieval_only_report_has_no_answer_section() -> None:
    """--retrieval-only: retrieval and the threshold are reported; the answer metrics and misses are left out."""
    result = rag_eval.RagCaseResult(
        id="q1", kind="text", question="?", expected_pages=[2], ranks=dict.fromkeys(rag_eval.MODES, 1),
        best_rerank_score=5.0, grounded=False, answer="", cited_pages=[], correct=None, judged_by="-", missing=[],
        citations_ok=None, not_found_ok=True, seconds=0.0,
    )  # fmt: skip
    meta = {"date": "2026-11-22", "document": "d.pdf", "chunks": 9, "embedding_model": "e", "rerank_model": "r",
            "answer_model": "a", "commit": "c", "minutes": 1.0, "retrieval_only": True}  # fmt: skip
    report = rag_eval.render_markdown([result], rag_eval.summarize([result]), meta)
    assert "## Retrieval" in report and "## Refusal threshold" in report
    assert "## Answers" not in report and "## Misses" not in report and "Retrieval only" in report
