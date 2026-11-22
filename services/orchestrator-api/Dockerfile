# The API: FastAPI + the LangGraph orchestrator + the DB, Doc and Web agents (project plan Phase 8).
# Built by docker compose (profile "app"), see the README:   docker compose --profile app up -d --build
FROM python:3.12-slim-bookworm

# Fonts for rendering PDF pages: Docling's layout model reads page IMAGES, and PDFs that don't embed their fonts
# (like the generated sample report) are drawn with the system's fonts. The slim image has none: symbols became empty
# boxes (detected as "icons") and headings were missed (7 instead of 17 on the sample). DejaVu covers the symbols,
# Liberation has Helvetica/Arial/Times-compatible shapes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# uv installs the LOCKED dependencies (uv.lock); pinned like everything else. Only needed while building.
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# The repo's layout: app/config.py finds the repo root three folders up (data/ lives there, mounted by compose).
WORKDIR /repo/services/orchestrator-api

# Dependencies first: this (big) layer is reused until uv.lock changes. CPU-only torch comes from the index that
# pyproject.toml names, so no CUDA libraries end up in the image.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

# The DuckDuckGo MCP server (web search), installed now: locally the API downloads it with uvx at start; in the image
# it is just there (WEB_MCP_COMMAND below). Its own environment, so its mcp 2.x never meets ours (plan 04).
ARG DDG_MCP_PACKAGE=duckduckgo-mcp-server==0.7.0
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_TOOL_DIR=/opt/uv-tools UV_TOOL_BIN_DIR=/usr/local/bin uv tool install "$DDG_MCP_PACKAGE"

COPY app ./app

# Compose runs this as YOUR user (HOST_UID), so files written into the mounted data/ stay yours. Model caches go
# there too (they survive rebuilds): HuggingFace (Docling's layout/table/OCR models) and the rest.
ENV WEB_MCP_COMMAND=duckduckgo-mcp-server \
    HOME=/tmp \
    HF_HOME=/repo/data/cache/huggingface \
    XDG_CACHE_HOME=/repo/data/cache \
    PYTHONUNBUFFERED=1

EXPOSE 8000
# --proxy-headers: behind the frontend's nginx, the client address (used by the rate limits) comes from
# X-Forwarded-For. Only the frontend reaches this port (compose doesn't publish it).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
