-- Business schema for the Database agent.
-- Runs once, automatically, when the Postgres container is first created.
--
-- The COMMENTs are deliberate: in Phase 1 the DB agent reads them during schema
-- reflection and shows them to the LLM, which makes Text-to-SQL much more accurate
-- (the LLM learns what "status" values exist, what currency amounts are in, etc.).

CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector, used by the Doc QA agent in Phase 3

CREATE TABLE vendors (
    id                  SERIAL PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,
    contact_email       TEXT NOT NULL,
    country             TEXT NOT NULL,
    payment_terms_days  INTEGER NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE vendors IS 'Companies we buy services or goods from.';
COMMENT ON COLUMN vendors.payment_terms_days IS 'Days after issue_date an invoice must be paid, e.g. 30 = Net 30.';

CREATE TABLE users (
    id          SERIAL PRIMARY KEY,
    full_name   TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    department  TEXT NOT NULL,
    role        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE users IS 'Internal employees.';
COMMENT ON COLUMN users.department IS 'One of: Finance, Engineering, Operations, Procurement, Support, Legal.';

CREATE TABLE invoices (
    id              SERIAL PRIMARY KEY,
    invoice_number  TEXT NOT NULL UNIQUE,
    vendor_id       INTEGER NOT NULL REFERENCES vendors(id),
    amount          NUMERIC(12, 2) NOT NULL CHECK (amount > 0),
    currency        CHAR(3) NOT NULL DEFAULT 'USD',
    status          TEXT NOT NULL CHECK (status IN ('pending', 'paid', 'overdue', 'cancelled')),
    issue_date      DATE NOT NULL,
    due_date        DATE NOT NULL,
    approved_by     INTEGER REFERENCES users(id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE invoices IS 'Bills received from vendors.';
COMMENT ON COLUMN invoices.status IS 'pending = not yet due and unpaid; overdue = past due_date and unpaid; paid; cancelled.';
COMMENT ON COLUMN invoices.approved_by IS 'User who approved payment. NULL if not approved yet.';

CREATE TABLE tickets (
    id           SERIAL PRIMARY KEY,
    title        TEXT NOT NULL,
    description  TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('open', 'in_progress', 'resolved', 'closed')),
    priority     TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    created_by   INTEGER NOT NULL REFERENCES users(id),
    assigned_to  INTEGER REFERENCES users(id),
    vendor_id    INTEGER REFERENCES vendors(id),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at  TIMESTAMPTZ
);
COMMENT ON TABLE tickets IS 'Internal support / operations tickets, optionally about a vendor.';
COMMENT ON COLUMN tickets.resolved_at IS 'NULL while the ticket is open or in progress.';

CREATE TABLE audit_logs (
    id           BIGSERIAL PRIMARY KEY,
    user_id      INTEGER NOT NULL REFERENCES users(id),
    action       TEXT NOT NULL,
    entity_type  TEXT NOT NULL,
    entity_id    INTEGER NOT NULL,
    details      JSONB NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE audit_logs IS 'Who did what to which record, e.g. action=approve_invoice, entity_type=invoice.';

-- Indexes for the most common filters the DB agent will generate.
CREATE INDEX idx_invoices_status_due ON invoices (status, due_date);
CREATE INDEX idx_tickets_status ON tickets (status);
CREATE INDEX idx_audit_logs_entity ON audit_logs (entity_type, entity_id);
