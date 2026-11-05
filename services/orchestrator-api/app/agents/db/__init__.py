"""Database agent: answers plain-English questions by writing, checking and running read-only SQL.

Modules:
    safety    - checks LLM-generated SQL before it can reach the database (no LLM involved)
    schema    - reads the database schema so the LLM knows the tables and columns   (step 3)
    database  - async connection pool on the read-only Postgres user               (step 3)
    prompts   - prompt templates for SQL generation, repair and answering          (step 4)
    models    - Pydantic models for the agent's inputs and outputs                 (step 4)
    graph     - the LangGraph workflow that wires everything together              (step 4)
"""
