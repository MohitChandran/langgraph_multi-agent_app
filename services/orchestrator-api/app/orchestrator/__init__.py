"""The orchestrator: a ReAct supervisor that answers a chat question by calling the agents as tools.

Plan: plannings/03-orchestrator.md. The supervisor LLM sees four tools (ask_database, ask_documents,
search_web, draft_email), calls one, reads a short summary of the result, decides the next, and
answers when it has enough. Every limit and safety rule is enforced by our code, not by the prompt.

Modules:
    models      - the data shapes: per-question budget, one agent call, the pending email, graph state, result
    tools       - what the LLM sees (tool schemas) and AgentTools, which runs the real agents behind them
    guards      - every safety and limit check as small pure functions (plan section 6)
    prompts     - the supervisor's system prompt and the forced-answer instruction (O2)
    graph       - the Orchestrator class: the LangGraph loop, ask() and resume_email() (O2, O3)
    checkpoint  - the Postgres checkpointer that lets a question pause for email approval (O3)
"""
