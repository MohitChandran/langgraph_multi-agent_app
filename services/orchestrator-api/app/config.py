"""Central configuration for the orchestrator service.

Every setting (API keys, model names, database URLs) is read from environment
variables, or from the repo-root `.env` file during local development.
Pydantic validates the values at startup, so a missing or malformed setting
crashes the app immediately with a clear error instead of failing later
in the middle of a user request.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
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
    # Required: no default, so a missing value fails at startup, not on the first question.
    readonly_database_url: str
    # Chat sessions and messages live in schema `app`, reached only through the app_rw user.
    app_database_url: str

    # --- Chat memory ---
    # Previous question/answer exchanges of the same chat that the agent sees (constant prompt cost per turn).
    chat_memory_turns: int = 3

    # --- Document Q&A (RAG), plan 02 section 9 ---
    data_dir: Path = REPO_ROOT / "data"  # uploaded files + picture crops: data/documents/<document_id>/
    groq_vision_model: str = "qwen/qwen3.8-27b"  # describes charts/diagrams found in documents
    # Changing the embedding model means re-embedding every document (vectors from two models don't mix).
    rag_embedding_model: str = "ibm-granite/granite-embedding-small-english-r2"
    rag_rerank: bool = True
    rag_rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rag_chunk_tokens: int = 350  # max tokens of a chunk's content (one idea per chunk)
    rag_candidates: int = 20  # chunks fetched by EACH search (vector and keyword) before fusion
    rag_top_k: int = 6  # sources given to the answer LLM
    rag_context_tokens: int = 3000  # token budget for those sources
    rag_min_rerank_score: float | None = None  # None = no threshold; set from the R9 eval
    rag_max_file_mb: int = 25
    rag_max_pages: int = 150
    rag_max_pictures_per_doc: int = 20  # vision calls are slow on Groq's free tier (~4 per minute)
    rag_min_picture_area: float = 0.02  # skip pictures smaller than 2% of their page

    @field_validator("data_dir")
    @classmethod
    def _resolve_data_dir(cls, value: Path) -> Path:
        """A relative DATA_DIR (e.g. "./data") is relative to the repo root, not to wherever the API was started."""
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()

    @field_validator("rag_min_rerank_score", mode="before")
    @classmethod
    def _empty_means_none(cls, value: object) -> object:
        """`RAG_MIN_RERANK_SCORE=` (empty in .env) means "no threshold" instead of a parse error."""
        return None if value == "" else value


@lru_cache
def get_settings() -> Settings:
    """Build the Settings object once and return the same instance on every call.

    lru_cache makes this a cheap singleton: the .env file is parsed only once,
    and every module that calls get_settings() shares the same validated config.
    """
    return Settings()
