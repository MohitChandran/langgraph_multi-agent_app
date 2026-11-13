#!/bin/bash
# Creates the `app` schema and its own login role `app_rw` for the application's
# chat data (sessions and messages). See plannings/01-chat-sessions-and-memory.md, section 4.1.
#
# Why a separate schema + role instead of more tables in `public`:
#   - readonly_agent (the DB agent) automatically gets SELECT on every new table in
#     `public` (02_readonly_role.sh). Chat tables there would be readable by
#     LLM-written SQL. It gets NOTHING on schema `app`, so Postgres refuses.
#   - app_rw gets nothing on the business tables in `public`, so chat code can't touch them.
#   - ops_admin stays reserved for seeding and schema changes.
#
# The tables themselves are created by the API at startup (ChatStore.setup()).
# Like the other init scripts, this runs only when the data volume is first created:
# on an existing machine run ./scripts/reset_db.sh once.
set -euo pipefail

psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -v app_password="$APP_DB_PASSWORD" -v db_name="$POSTGRES_DB" <<'EOSQL'
CREATE ROLE app_rw LOGIN PASSWORD :'app_password';
GRANT CONNECT ON DATABASE :"db_name" TO app_rw;

-- app_rw OWNS schema app: it can create, read and write its own tables there, nothing else.
CREATE SCHEMA app AUTHORIZATION app_rw;
-- app first: unqualified table names mean app.<table>. public second: that's where the pgvector
-- extension lives (type `vector`, operator `<=>`), needed by the document tables (plan 02, 4.1).
-- Safe: app_rw has no grants on public's business tables.
ALTER ROLE app_rw SET search_path = app, public;
ALTER ROLE app_rw SET statement_timeout = '10s';   -- chat queries are tiny; anything slower is a bug

-- Deliberately NOT granted: readonly_agent gets nothing on schema app; app_rw gets nothing on public's tables.
EOSQL
