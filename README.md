# Enterprise Multi-Agent Ops Platform

A LangGraph supervisor that routes requests to a Database agent (safe Text-to-SQL), a Doc QA agent
(hybrid RAG), a Web Search agent (MCP), and a separate Email service (A2A) with human approval.
See `project_plan.txt` for the full design. This README is filled in properly in Phase 8.

## Quickstart (Phase 0)

Prerequisites: Docker with Compose, [uv](https://docs.astral.sh/uv/), Python 3.12.

```bash
cp .env.example .env              # then set GROQ_API_KEY (and change passwords if you like)
./scripts/reset_db.sh             # starts Postgres + Mailpit, creates the schema, loads mock data

cd services/orchestrator-api
uv sync
uv run python -m app.llm          # Groq hello-world: lists models, calls both tiers
```

## Chat UI (for testing the agents)

Two terminals, from the repo root:

```bash
# 1. API (FastAPI + DB agent) on http://localhost:8000  (API docs: /docs)
cd services/orchestrator-api && uv run uvicorn app.main:app --reload --port 8000

# 2. Frontend (React + Vite) on http://localhost:5173
cd services/frontend && npm install && npm run dev
```

Open http://localhost:5173 and ask a question. Under each answer, "Show SQL & data" shows the
workflow steps that ran, the exact SQL and the returned rows. Vite forwards `/api/*` to port 8000.

Chats are saved in Postgres (schema `app`, see `plannings/01-chat-sessions-and-memory.md`): the
sidebar lists them, each has its own URL (`/c/<id>`), and the agent remembers the last 3
exchanges of the chat you're in (`CHAT_MEMORY_TURNS`), so follow-ups like "only the ones in India" work.

### Documents (RAG agent)

Attach a PDF, Word, PowerPoint, Markdown, text or image file with the paperclip (or drag it onto the chat).
It is processed in the background (Docling parsing with OCR, chart descriptions by a vision model, chunking,
embeddings in pgvector); a chip shows its progress. From then on that chat is answered by the Doc agent:
hybrid search (vectors + keywords, fused with RRF, then reranked) and answers with numbered citations that
open the PDF at the right page. Chats without documents keep using the DB agent.

```bash
uv run scripts/make_sample_docs.py      # test PDF with tables, charts, a diagram and a scanned page -> data/samples/
cd services/orchestrator-api
uv run python -m app.agents.doc.parsing ../../data/samples/q3-compliance-report.pdf --chunks   # parse + chunk
uv run python -m app.agents.doc.retrieval "credit if AWS breaches the SLA"                     # every search stage
uv run pytest -m slow                   # real-Docling tests (skipped by default)
```

The first API start downloads the embedding and rerank models (~0.3 GB) into `data/models/`.

Two 7-page test documents (a scanned page, tables and charts) are in `evals/test_documents/`, with 14 verified
questions and expected answers in `evals/datasets/test_documents_questions.json`. Regenerate them with
`uv run evals/make_test_documents.py`.

`reset_db.sh` can be re-run any time to get a clean database. The mock data is deterministic
(same rows on every machine) and its dates are relative to today, so time-based questions
("invoices due this week") always have answers.

- Mailpit inbox: http://localhost:8025
- Postgres: `localhost:5432` (admin role for seeding; `readonly_agent` role for the DB agent)

## Tracing (Langfuse)

Every chat message becomes one trace in a self-hosted Langfuse: what was asked, which guard checks and agents
ran, every LLM call with its model, tokens and latency, and the answer. Traces of one chat share a session.

```bash
docker compose -f docker-compose.langfuse.yml up -d   # first start pulls ~3 GB; ready after 1-3 minutes
```

Open http://localhost:3000 and log in with `LANGFUSE_INIT_USER_EMAIL` / `LANGFUSE_INIT_USER_PASSWORD` from
`.env`. The first start creates the project and its API keys from `.env` (see the Langfuse section of
`.env.example`), so the API sends traces as soon as it restarts. Without the two Langfuse keys, tracing is off
and nothing is sent.

```
chat turn                       one trace per message; session = the chat
├─ input guard                  guardrail: the verdict
│  ├─ prompt guard              generation: model, tokens
│  └─ safety model              generation: model, tokens
└─ orchestrator                 the LangGraph run: prepare → supervisor → tools → finalize
   └─ tools → db agent / doc agent / web agent   each agent's own graph, nested under its tool call
```

An email approval is its own trace ("email decision") in the same session; `POST /api/db/ask` is traced as
"db ask". The code is in `services/orchestrator-api/app/observability.py`.

## Browsing the data

Connect any SQL client (e.g. DBeaver) with the values from your `.env`:
host `localhost`, port `5432`, database `ops_platform`, user `ops_admin`, password `admin@123`
(local development only; set by `POSTGRES_PASSWORD` in `.env`).

| Table | Rows | Contents |
|-------|------|----------|
| `vendors` | 18 | Real vendor names in 5 categories (cloud, hardware, consulting, logistics, legal) |
| `users` | 25 | Employees with department and role |
| `contracts` | ~29 | Service agreements: value, dates, SLA uptime (cloud only), auto-renew, status |
| `invoices` | 300 | Monthly contract billing + one-off purchases, with status, due and paid dates |
| `tickets` | 100 | Ops/support tickets, linked to vendors by topic |
| `audit_logs` | 400 | Who approved, renewed, closed what, and when |

Vendor names are real companies, but every amount, date, contract and invoice is fictional.

## Repository layout

| Path | What lives there |
|------|------------------|
| `services/orchestrator-api/` | FastAPI gateway + LangGraph supervisor + internal agents |
| `services/email-a2a-service/` | Standalone email agent behind an A2A server |
| `services/frontend/` | React + Vite UI |
| `scripts/db/` | SQL run once when the Postgres container is created (schema, read-only role) |
| `scripts/seed_db.py` | Fake business data |
| `evals/` | Eval datasets, runner, and generated reports |
