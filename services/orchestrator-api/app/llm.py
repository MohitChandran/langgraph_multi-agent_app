"""Groq LLM client setup.

This is the ONLY place where chat models are created. Agents ask for a model
tier ("strong" or "fast") instead of hardcoding model names, so swapping models
or providers later means changing one file (or just the .env), not every agent.

Run it directly for a "hello world" connectivity check:
    uv run python -m app.llm
"""

from typing import Literal

from groq import Groq
from langchain_groq import ChatGroq

from app.config import get_settings

ModelTier = Literal["strong", "fast"]


def get_llm(tier: ModelTier = "strong", temperature: float = 0.0) -> ChatGroq:
    """Return a LangChain chat model backed by Groq.

    Args:
        tier: "strong" for reasoning-heavy steps (routing, SQL generation),
              "fast" for cheap high-volume steps (summaries, query rewrites).
        temperature: 0.0 = deterministic output, which we want for routing and SQL.

    The Groq SDK retries 429 (rate limit) and 5xx errors with exponential backoff
    up to `max_retries` times. That matters on Groq's free tier, which has
    per-minute rate limits.
    """
    settings = get_settings()
    model = settings.groq_strong_model if tier == "strong" else settings.groq_fast_model
    return ChatGroq(
        model=model,
        api_key=settings.groq_api_key,
        temperature=temperature,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout_seconds,
    )


def list_groq_models() -> list[str]:
    """Return the IDs of all models currently available on your Groq account, sorted.

    Groq adds and deprecates models often, so use this to check that the models
    configured in .env still exist before picking them.
    """
    client = Groq(api_key=get_settings().groq_api_key.get_secret_value())
    return sorted(model.id for model in client.models.list().data)


def main() -> None:
    """Hello-world check: list available models, then send one prompt to each tier."""
    print("Models available on your Groq account:")
    for model_id in list_groq_models():
        print(f"  - {model_id}")

    for tier in ("strong", "fast"):
        llm = get_llm(tier)
        reply = llm.invoke("In one sentence: what is a multi-agent system?")
        usage = reply.usage_metadata or {}
        print(f"\n[{tier}] {llm.model_name}")
        print(f"  reply : {reply.content}")
        print(f"  tokens: in={usage.get('input_tokens')} out={usage.get('output_tokens')}")


if __name__ == "__main__":
    main()
