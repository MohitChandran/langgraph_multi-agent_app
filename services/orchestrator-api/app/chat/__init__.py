"""Chat sessions (conversations) and their memory. Plan: plannings/01-chat-sessions-and-memory.md.

Modules:
    store    - ChatStore: Postgres access for chat_sessions / chat_messages in schema `app`   (S1, S2)
    models   - Pydantic API models: Session, Message, ChatRequest, ...                        (S2)
    service  - one chat turn: claim session, save question, run the agent, save the answer   (S2)
    routes   - FastAPI endpoints under /api/chat and /api/sessions                           (S2)
    memory   - load the last few exchanges of a chat and format them for the prompt          (S3)
"""
