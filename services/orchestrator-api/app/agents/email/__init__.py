"""Email & dispatch agent (project plan Phases 5-6, plan 05).

The email agent itself is a separate service (services/email-a2a-service: a CrewAI crew behind an A2A server, with
human approval before anything is sent). This package is the orchestrator's side of it:
    a2a_agent - A2AEmailAgent: the A2A client of that service (draft -> paused for approval -> resume)
    models    - EmailDraft, EmailAgentResult: what both agents return
    dummy     - DummyEmailAgent: hardcoded drafts, never sends anything (tests, and runs without the service)
"""
