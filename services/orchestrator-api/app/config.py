"""Central configuration for the orchestrator service.

Every setting (API keys, model names, database URLs) is read from environment
variables, or from the repo-root `.env` file during local development.
Pydantic validates the values at startup, so a missing or malformed setting
crashes the app immediately with a clear error instead of failing later
in the middle of a user request.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# app/config.py -> app -> orchestrator-api -> services -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """All configuration values for the orchestrator, loaded from env vars / .env.

    Field names map to env vars case-insensitively, e.g. `groq_api_key` <- GROQ_API_KEY.
    """

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env also holds settings for other services; ignore those
    )

    # --- LLM (Groq) ---
    # SecretStr hides the value in logs and reprs: print(settings) shows '**********'.
    groq_api_key: SecretStr
    groq_strong_model: str = "llama-3.3-70b-versatile"  # routing, SQL generation
    groq_fast_model: str = "llama-3.1-8b-instant"  # cheap steps: summaries, rewrites
    llm_max_retries: int = 3  # retries with exponential backoff on 429 / 5xx errors
    llm_timeout_seconds: float = 60.0

    # --- Database ---
    # The DB agent only ever gets the read-only URL (defense in depth, see project plan 3-B).
    readonly_database_url: str = ""


@lru_cache
def get_settings() -> Settings:
    """Build the Settings object once and return the same instance on every call.

    lru_cache makes this a cheap singleton: the .env file is parsed only once,
    and every module that calls get_settings() shares the same validated config.
    """
    return Settings()
