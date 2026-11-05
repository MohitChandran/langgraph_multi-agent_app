-- Business schema for the Database agent.
-- Runs once, automatically, when the Postgres container is first created.
-- To apply changes to this file, recreate the database: ./scripts/reset_db.sh
--
-- The COMMENTs are deliberate: in Phase 1 the DB agent reads them during schema
-- reflection and shows them to the LLM, which makes Text-to-SQL much more accurate
-- (the LLM learns what "status" values exist, what currency amounts are in, etc.).

CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector, used by the Doc QA agent in Phase 3

CREATE TABLE vendors (
    id                  SERIAL PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,
    category            TEXT NOT NULL CHECK (category IN ('cloud', 'hardware', 'consulting', 'logistics', 'legal')),
    contact_email       TEXT NOT NULL,
    country             TEXT NOT NULL,
    payment_terms_days  INTEGER NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE vendors IS 'Companies we buy services or goods from.';
COMMENT ON COLUMN vendors.category IS 'One of: cloud, hardware, consulting, logistics, legal.';
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

CREATE TABLE contracts (
    id                   SERIAL PRIMARY KEY,
    contract_number      TEXT NOT NULL UNIQUE,
    vendor_id            INTEGER NOT NULL REFERENCES vendors(id),
    service_description  TEXT NOT NULL,
    annual_value         NUMERIC(12, 2) NOT NULL CHECK (annual_value > 0),
    currency             CHAR(3) NOT NULL DEFAULT 'USD',
    start_date           DATE NOT NULL,
    end_date             DATE NOT NULL CHECK (end_date > start_date),
    sla_uptime_pct       NUMERIC(5, 2) CHECK (sla_uptime_pct BETWEEN 90 AND 100),
    auto_renew           BOOLEAN NOT NULL DEFAULT false,
    status               TEXT NOT NULL CHECK (status IN ('active', 'expired', 'terminated')),
    owner_id             INTEGER NOT NULL REFERENCES users(id),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE contracts IS 'Service agreements with vendors. A vendor can have several contracts over time.';
COMMENT ON COLUMN contracts.sla_uptime_pct IS 'Guaranteed uptime in percent, e.g. 99.95. Only set for cloud vendors, NULL otherwise.';
COMMENT ON COLUMN contracts.status IS 'active = currently running; expired = end_date has passed; terminated = ended early.';
COMMENT ON COLUMN contracts.owner_id IS 'Employee responsible for this contract.';

CREATE TABLE invoices (
    id              SERIAL PRIMARY KEY,
    invoice_number  TEXT NOT NULL UNIQUE,
    vendor_id       INTEGER NOT NULL REFERENCES vendors(id),
    contract_id     INTEGER REFERENCES contracts(id),
    amount          NUMERIC(12, 2) NOT NULL CHECK (amount > 0),
    currency        CHAR(3) NOT NULL DEFAULT 'USD',
    status          TEXT NOT NULL CHECK (status IN ('pending', 'paid', 'overdue', 'cancelled')),
    issue_date      DATE NOT NULL,
    due_date        DATE NOT NULL,
    paid_date       DATE,
    approved_by     INTEGER REFERENCES users(id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE invoices IS 'Bills received from vendors.';
COMMENT ON COLUMN invoices.contract_id IS 'Contract this invoice is billed under. NULL for one-off purchases.';
COMMENT ON COLUMN invoices.status IS 'pending = not yet due and unpaid; overdue = past due_date and unpaid; paid; cancelled.';
COMMENT ON COLUMN invoices.paid_date IS 'Date the invoice was paid. Only set when status = paid.';
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
COMMENT ON COLUMN audit_logs.entity_type IS 'One of: invoice, vendor, contract, ticket. entity_id is the id in that table.';

-- Indexes for the most common filters the DB agent will generate.
CREATE INDEX idx_invoices_status_due ON invoices (status, due_date);
CREATE INDEX idx_contracts_end_date ON contracts (end_date);
CREATE INDEX idx_tickets_status ON tickets (status);
CREATE INDEX idx_audit_logs_entity ON audit_logs (entity_type, entity_id);
