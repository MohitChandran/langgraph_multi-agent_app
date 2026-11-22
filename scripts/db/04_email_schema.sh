#!/bin/bash
# Creates the `email` schema and its own login role `email_rw` for the email service's A2A tasks
# (drafts waiting for human approval). See plannings/05-email-agent-a2a.md, section 6.6.
#
# Same pattern as 03_app_schema.sh, for the same reasons:
#   - email_rw owns schema `email` and nothing else: it can't read the business tables or the chats;
#   - readonly_agent (LLM-written SQL) and app_rw (the orchestrator's chats) get NOTHING on `email`,
#     so a draft or its recipient can't leak through the DB agent.
#
# The table itself (email.tasks) is created by the email service at startup (a2a-sdk's DatabaseTaskStore).
# Runs only when the Postgres data volume is first created. On an existing machine, run it once by hand:
#   set -a; . ./.env; set +a
#   docker exec -i -e EMAIL_DB_PASSWORD="$EMAIL_DB_PASSWORD" ops-postgres bash -s < scripts/db/04_email_schema.sh
set -euo pipefail

psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -v email_password="$EMAIL_DB_PASSWORD" -v db_name="$POSTGRES_DB" <<'EOSQL'
CREATE ROLE email_rw LOGIN PASSWORD :'email_password';
GRANT CONNECT ON DATABASE :"db_name" TO email_rw;

-- email_rw OWNS schema email: it creates, reads and writes its own tables there, nothing else.
CREATE SCHEMA email AUTHORIZATION email_rw;
-- Unqualified names mean email.<table>: the SDK creates its table as plain "tasks", which lands here.
ALTER ROLE email_rw SET search_path = email;
ALTER ROLE email_rw SET statement_timeout = '10s';   -- task reads/writes are tiny; anything slower is a bug

-- Deliberately NOT granted: readonly_agent and app_rw get nothing on schema email.
EOSQL
