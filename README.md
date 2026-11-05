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

`reset_db.sh` can be re-run any time to get a clean database. The mock data is deterministic
(same rows on every machine) and its dates are relative to today, so time-based questions
("invoices due this week") always have answers.

- Mailpit inbox: http://localhost:8025
- Postgres: `localhost:5432` (admin role for seeding; `readonly_agent` role for the DB agent)

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
