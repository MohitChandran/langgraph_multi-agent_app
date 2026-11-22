"""The email service: a CrewAI email agent behind an A2A server (plan 05).

The orchestrator (services/orchestrator-api) talks to this service over the A2A protocol: it asks for a draft,
the service writes it with a CrewAI crew and PAUSES the A2A task until a human approves, edits or rejects it;
only an approval sends the email (SMTP). The app itself is built in app/server.py; this module is what uvicorn runs.

Run it (from services/email-a2a-service):
    uv run uvicorn app.main:app --port 8001
Agent card: http://localhost:8001/.well-known/agent-card.json · API docs: http://localhost:8001/docs
"""

import logging

from app.server import create_app  # importing `app` first switches CrewAI's telemetry off (app/__init__.py)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s")

app = create_app()
