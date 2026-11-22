# RAG eval (Doc QA agent): 2026-11-22

Document `data/samples/q3-compliance-report.pdf` (20 chunks) · embeddings `ibm-granite/granite-embedding-small-english-r2` · rerank `Xenova/ms-marco-MiniLM-L-6-v2` · answer model `openai/gpt-oss-20b` · commit `25b02fc+changes` · 23 questions (20 answerable, 3 not in the document) · run time 2.4 min. Dataset: `evals/datasets/rag_questions.json`.

## Retrieval: which search finds the right page

A hit = a retrieved chunk on an expected page. hit@k = share of the answerable questions with a hit in the top k (the agent uses the top 6). MRR = average of 1 / rank of the first hit.

| Mode | hit@1 | hit@3 | hit@6 | MRR |
|---|---|---|---|---|
| dense | 85% | 100% | 100% | 0.917 |
| keyword | 55% | 100% | 100% | 0.758 |
| hybrid | 95% | 100% | 100% | 0.975 |
| hybrid+rerank | 90% | 100% | 100% | 0.950 |

_Retrieval only (`--retrieval-only`): the answers aren't run._

## Refusal threshold (RAG_MIN_RERANK_SCORE)

Best rerank score per question: answerable -0.30, 2.06, 3.03, 3.52, 3.98, 4.13, 5.35, 5.75, 5.77, 5.85, 6.00, 6.45, 6.59, 7.26, 7.96, 8.46, 8.57, 8.59, 8.85, 9.99; not in the document -11.24, -10.06, -0.02.

**Recommendation: no threshold** (the groups overlap (lowest answerable -0.30 ≤ highest not-in-document -0.02): a threshold would refuse answerable questions, so none is set; the answer model's own answer_found decides).

## Questions

| Q | Kind | Pages | Rank: dense / keyword / hybrid / +rerank | Best score | Answer | Citations | Time |
|---|---|---|---|---|---|---|---|
| q01 | text | 1 | 1 / 3 / 1 / 1 | 8.85 | — |  | 0.0 s |
| q02 | text | 1 | 1 / 1 / 1 / 1 | 8.59 | — |  | 0.0 s |
| q03 | text | 1 | 1 / 1 / 1 / 1 | 8.46 | — |  | 0.0 s |
| q04 | text | 1 | 1 / 1 / 1 / 1 | 7.96 | — |  | 0.0 s |
| q05 | text | 1 | 1 / 2 / 1 / 1 | 7.26 | — |  | 0.0 s |
| q06 | text | 2 | 1 / 2 / 1 / 1 | 9.99 | — |  | 0.0 s |
| q07 | table | 2 | 1 / 2 / 1 / 1 | 3.98 | — |  | 0.0 s |
| q08 | table | 2 | 2 / 2 / 1 / 2 | 4.13 | — |  | 0.0 s |
| q09 | table | 2 | 2 / 2 / 1 / 1 | 3.52 | — |  | 0.0 s |
| q10 | table | 5 | 1 / 1 / 1 / 1 | 5.75 | — |  | 0.0 s |
| q11 | table | 5 | 1 / 1 / 1 / 2 | -0.30 | — |  | 0.0 s |
| q12 | picture | 2 | 1 / 1 / 1 / 1 | 3.03 | — |  | 0.0 s |
| q13 | picture | 2 | 1 / 1 / 1 / 1 | 6.00 | — |  | 0.0 s |
| q14 | picture | 3 | 1 / 2 / 1 / 1 | 6.59 | — |  | 0.0 s |
| q15 | picture | 3 | 1 / 2 / 1 / 1 | 5.35 | — |  | 0.0 s |
| q16 | picture | 3 | 1 / 1 / 1 / 1 | 5.77 | — |  | 0.0 s |
| q17 | picture | 3 | 1 / 1 / 1 / 1 | 2.06 | — |  | 0.0 s |
| q18 | ocr | 4 | 1 / 1 / 1 / 1 | 5.85 | — |  | 0.0 s |
| q19 | ocr | 4 | 1 / 3 / 1 / 1 | 8.57 | — |  | 0.0 s |
| q20 | ocr | 4 | 3 / 1 / 2 / 1 | 6.45 | — |  | 0.0 s |
| q21 | not_in_doc | — | — / — / — / — | -11.24 | ✅ not found |  | 0.0 s |
| q22 | not_in_doc | — | — / — / — / — | -0.02 | ✅ not found |  | 0.0 s |
| q23 | not_in_doc | — | — / — / — / — | -10.06 | ✅ not found |  | 0.0 s |
