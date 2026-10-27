#!/bin/bash
# Creates the read-only database role used by the DB agent (safety layer 1 of 4).
#
# Even if the LLM generates "DROP TABLE invoices", Postgres itself refuses,
# because this role was never granted anything except SELECT.
# This is a shell script (not .sql) so it can read the password from an env var.
set -euo pipefail

psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -v ro_password="$READONLY_DB_PASSWORD" -v db_name="$POSTGRES_DB" <<'EOSQL'
CREATE ROLE readonly_agent LOGIN PASSWORD :'ro_password';

GRANT CONNECT ON DATABASE :"db_name" TO readonly_agent;
GRANT USAGE ON SCHEMA public TO readonly_agent;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_agent;
-- Tables created later by the admin are readable too:
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readonly_agent;

-- Extra guardrails applied to every session of this role:
ALTER ROLE readonly_agent SET default_transaction_read_only = on;  -- writes fail even if granted by mistake
ALTER ROLE readonly_agent SET statement_timeout = '5s';            -- kills runaway queries (huge joins)
EOSQL
