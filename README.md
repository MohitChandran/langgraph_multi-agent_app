# Enterprise Multi-Agent Ops Platform

A LangGraph supervisor that routes requests to a Database agent (safe Text-to-SQL), a Doc QA agent
(hybrid RAG), a Web Search agent (MCP), and a separate Email service (A2A) with human approval.
See `project_plan.txt` for the full design. This README is filled in properly in Phase 8.

## Quickstart (Phase 0)

Prerequisites: Docker with Compose, [uv](https://docs.astral.sh/uv/), Python 3.12.

```bash
cp .env.example .env              # then set GROQ_API_KEY (and change passwords if you like)
docker compose up -d              # Postgres + pgvector, Mailpit
uv run scripts/seed_db.py         # fill tables with realistic fake data

cd services/orchestrator-api
uv sync
uv run python -m app.llm          # Groq hello-world: lists models, calls both tiers
```

- Mailpit inbox: http://localhost:8025
- Postgres: `localhost:5432` (admin role for seeding; `readonly_agent` role for the DB agent)

## Repository layout

| Path | What lives there |
|------|------------------|
| `services/orchestrator-api/` | FastAPI gateway + LangGraph supervisor + internal agents |
| `services/email-a2a-service/` | Standalone email agent behind an A2A server |
| `services/frontend/` | React + Vite UI |
| `scripts/db/` | SQL run once when the Postgres container is created (schema, read-only role) |
| `scripts/seed_db.py` | Fake business data |
| `evals/` | Eval datasets, runner, and generated reports |
