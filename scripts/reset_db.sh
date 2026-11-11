#!/usr/bin/env bash
# Rebuild the database from scratch: schema + read-only role + mock data.
#
# Use it after cloning the repo, after editing scripts/db/*.sql, or whenever
# you want a clean copy of the data. Run from anywhere:  ./scripts/reset_db.sh
#
# WARNING: deletes ALL data in the local Postgres container (it's mock data,
# so nothing is lost that the seed script can't recreate).
set -euo pipefail

cd "$(dirname "$0")/.."  # repo root, where docker-compose.yml and .env live

if [[ ! -f .env ]]; then
    echo "No .env file found. Create it first:  cp .env.example .env" >&2
    exit 1
fi

echo "==> Removing containers and the Postgres data volume"
docker compose down -v

echo "==> Starting containers (schema, read-only role and app schema are created on first start)"
docker compose up -d --wait  # --wait blocks until the healthchecks pass

echo "==> Loading mock data"
uv run scripts/seed_db.py

echo "==> Done. Database is ready."
