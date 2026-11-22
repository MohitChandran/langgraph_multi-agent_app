# The email service: the CrewAI email crew behind an A2A server (plan 05). Built by docker compose (profile "app").
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /service

# Its OWN locked environment (CrewAI's 140-odd packages), never the orchestrator's.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

COPY app ./app

# A plain user: nothing here needs root. CrewAI keeps small files under its home directory.
RUN useradd --create-home --uid 10001 email
USER email
ENV PYTHONUNBUFFERED=1

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
