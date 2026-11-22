"""The input guardrail: every message is checked BEFORE the orchestrator sees it (plannings/04, Part A).

Built with NeMo Guardrails (input rails only). Two checks run in order, cheapest first:
  1. "jailbreak check": Prompt Guard 86M scores how likely the message tries to override the
     assistant's instructions ("ignore previous instructions...");
  2. "safety check": a small LLM (allam-2-7b) reads a fixed policy and says allow / block for harmful
     or illegal requests. Any other topic is allowed (open scope).
If rail 1 blocks, rail 2 never runs. If a check fails or times out, the message is BLOCKED (fail closed).
The guard models know nothing about the app's data or tools: they only read the message.

Modules:
    policy.py   the safety policy and the three refusal texts
    actions.py  the two checks as async functions NeMo calls (Groq), plus the per-check trace
    guard.py    InputGuard (the NeMo rails), GuardResult, and a CLI
    config/     NeMo's config: which flows run (config.yml) and the flows themselves (rails.co, Colang 1.0)
"""
