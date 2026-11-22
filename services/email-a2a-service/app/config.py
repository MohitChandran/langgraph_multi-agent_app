"""Configuration for the email service (plan 05, section 7).

Every setting is read from environment variables, or from THIS service's own `.env` file
(services/email-a2a-service/.env), not the repo-root one: the email service is a separate program with its own
secrets (the SMTP password, the API key the orchestrator must present). Pydantic validates everything at startup,
so a missing key stops the service right away with a clear error.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# app/config.py -> app -> email-a2a-service
SERVICE_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """All configuration values of the email service. Field names map to env vars, e.g. `smtp_host` <- SMTP_HOST."""

    model_config = SettingsConfigDict(env_file=SERVICE_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- A2A server ---
    # The orchestrator sends this in the X-API-Key header; without it, nobody can ask this service to send email.
    email_service_api_key: SecretStr
    # Where clients reach this service; written into the agent card (the A2A "business card").
    email_service_public_url: str = "http://localhost:8001"

    # --- Storage: the A2A tasks (drafts waiting for approval) live in Postgres schema `email` (role email_rw) ---
    email_database_url: SecretStr  # postgresql+asyncpg://email_rw:<password>@localhost:5432/ops_platform

    # --- The CrewAI crew (writer -> compliance reviewer) on Groq ---
    groq_api_key: SecretStr
    email_crew_model: str = "qwen/qwen3.8-27b"  # its own Groq quota, not the orchestrator's gpt-oss-20b
    email_crew_context_window: int = 8192  # CrewAI doesn't know this model's window; tell it
    email_crew_timeout_seconds: float = 90.0

    # --- Who may receive email ---
    email_from: str = "ops-assistant@company.example"
    # Comma-separated. A domain ("company.example") allows every address there; a full address
    # ("me@gmail.com") allows only that address. Ignored when EMAIL_ALLOW_ANY_DOMAIN=true.
    email_allowed_domains: str = "company.example,example.com"
    email_allow_any_domain: bool = False

    # --- SMTP (development: Mailpit on localhost:1025, no login, no TLS) ---
    smtp_host: str = "localhost"
    smtp_port: int = 1025
    smtp_username: str = ""
    smtp_password: SecretStr = SecretStr("")
    smtp_starttls: bool = False  # port 587 providers (Gmail, most others): true
    smtp_use_tls: bool = False  # port 465 providers ("SMTPS"): true
    smtp_timeout_seconds: float = 20.0

    # --- Tracing (Langfuse): the SAME project keys as the orchestrator, so both services write into one trace ---
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_base_url: str = "http://localhost:3000"

    @property
    def allowed_recipients(self) -> list[str]:
        """EMAIL_ALLOWED_DOMAINS as a clean, lower-case list (empty entries dropped)."""
        return [item.strip().lower() for item in self.email_allowed_domains.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    """Build the Settings once (the .env file is parsed only once) and share the same instance everywhere."""
    return Settings()
