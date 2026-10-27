# /// script
# requires-python = ">=3.12"
# dependencies = ["faker>=30", "psycopg[binary]>=3.2", "python-dotenv>=1.0"]
# ///
"""Fill the business tables with realistic fake data.

Run from the repo root (after `docker compose up -d`):
    uv run scripts/seed_db.py

The block at the top of this file is "inline script metadata" (PEP 723):
`uv run` reads it and installs faker/psycopg into a throwaway environment,
so this one-off script doesn't need its own pyproject.toml.

The data is generated relative to TODAY and with a fixed random seed, so:
  - "pending invoices due this week" (Scenario 1 in the plan) always has results;
  - every run produces the same rows, which keeps the SQL eval answers stable.
Running it again wipes the tables and re-seeds them.
"""

import os
import random
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from faker import Faker
from psycopg.types.json import Jsonb

SEED = 42
N_VENDORS = 15
N_USERS = 25
N_INVOICES = 120
N_TICKETS = 60
N_AUDIT_LOGS = 200

DEPARTMENTS = ["Finance", "Engineering", "Operations", "Procurement", "Support", "Legal"]
ROLES = ["Analyst", "Manager", "Engineer", "Director", "Specialist"]
TICKET_TOPICS = [
    ("Invoice amount mismatch", "Billed amount does not match the purchase order."),
    ("Duplicate invoice received", "Vendor sent the same invoice twice."),
    ("SLA breach on cloud hosting", "Service was unavailable longer than the SLA allows."),
    ("Update vendor bank details", "Vendor requested a change of payout account."),
    ("Contract renewal review", "Contract expires soon and needs legal review."),
    ("Late delivery of hardware", "Shipment arrived two weeks after the agreed date."),
    ("Access request for billing portal", "New team member needs vendor portal access."),
    ("Tax information missing", "Invoice lacks the vendor's tax registration number."),
]


def get_database_url() -> str:
    """Read the ADMIN database URL from the repo-root .env file.

    We use the admin role here because seeding writes data; the read-only
    role used by the DB agent would (correctly) be refused.
    """
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set. Copy .env.example to .env first.")
    return url


def make_vendors(fake: Faker) -> list[tuple]:
    """Generate vendor rows: (name, contact_email, country, payment_terms_days)."""
    names = set()
    while len(names) < N_VENDORS:  # vendor names must be unique (UNIQUE constraint)
        names.add(fake.company())
    return [
        (
            name,
            f"billing@{fake.domain_name()}",
            fake.country(),
            random.choice([15, 30, 45, 60]),
        )
        for name in sorted(names)
    ]


def make_users(fake: Faker) -> list[tuple]:
    """Generate employee rows: (full_name, email, department, role)."""
    rows = []
    for i in range(N_USERS):
        name = fake.name()
        slug = name.lower().replace(" ", ".").replace("'", "")
        email = f"{slug}.{i}@company.com"  # index keeps emails unique (UNIQUE constraint)
        rows.append((name, email, random.choice(DEPARTMENTS), random.choice(ROLES)))
    return rows


def pick_invoice_status(due_date: date, today: date) -> str:
    """Choose a status that is consistent with the due date.

    An invoice past its due date can't be 'pending' (it's 'overdue' or paid);
    one not yet due can't be 'overdue'. Consistent data means the LLM's SQL
    is tested on realistic logic, not on random noise.
    """
    if due_date < today:
        return random.choices(["paid", "overdue", "cancelled"], weights=[60, 30, 10])[0]
    return random.choices(["pending", "paid", "cancelled"], weights=[80, 15, 5])[0]


def make_invoices(today: date) -> list[tuple]:
    """Generate invoice rows: (invoice_number, vendor_id, amount, currency, status,
    issue_date, due_date, approved_by).

    The first 4 invoices are forced to be 'pending' and due before Sunday,
    so Scenario 1 ("pending invoices due this week") always returns rows.
    """
    days_left_in_week = 6 - today.weekday()  # Monday=0 ... Sunday=6
    rows = []
    for i in range(N_INVOICES):
        if i < 4:
            due_date = today + timedelta(days=random.randint(0, days_left_in_week))
            status = "pending"
        else:
            due_date = today + timedelta(days=random.randint(-90, 45))
            status = pick_invoice_status(due_date, today)
        issue_date = due_date - timedelta(days=random.choice([15, 30, 45, 60]))
        approved_by = random.randint(1, N_USERS) if status == "paid" or random.random() < 0.3 else None
        rows.append(
            (
                f"INV-{today.year}-{i + 1:04d}",
                random.randint(1, N_VENDORS),
                round(random.uniform(250, 75_000), 2),
                random.choices(["USD", "EUR", "INR"], weights=[70, 20, 10])[0],
                status,
                issue_date,
                due_date,
                approved_by,
            )
        )
    return rows


def make_tickets(fake: Faker, now: datetime) -> list[tuple]:
    """Generate ticket rows: (title, description, status, priority, created_by,
    assigned_to, vendor_id, created_at, resolved_at).

    resolved_at is only set for resolved/closed tickets, and always after created_at.
    """
    rows = []
    for _ in range(N_TICKETS):
        title, description = random.choice(TICKET_TOPICS)
        status = random.choices(["open", "in_progress", "resolved", "closed"], weights=[30, 25, 25, 20])[0]
        created_at = now - timedelta(days=random.randint(0, 120), hours=random.randint(0, 23))
        resolved_at = created_at + timedelta(days=random.randint(1, 14)) if status in ("resolved", "closed") else None
        rows.append(
            (
                title,
                f"{description} {fake.sentence()}",
                status,
                random.choices(["low", "medium", "high", "critical"], weights=[30, 40, 20, 10])[0],
                random.randint(1, N_USERS),
                random.randint(1, N_USERS) if status != "open" else None,
                random.randint(1, N_VENDORS) if random.random() < 0.7 else None,
                created_at,
                resolved_at,
            )
        )
    return rows


def make_audit_logs(now: datetime) -> list[tuple]:
    """Generate audit rows: (user_id, action, entity_type, entity_id, details, created_at)."""
    actions = [
        ("approve_invoice", "invoice", N_INVOICES),
        ("reject_invoice", "invoice", N_INVOICES),
        ("update_vendor", "vendor", N_VENDORS),
        ("close_ticket", "ticket", N_TICKETS),
        ("assign_ticket", "ticket", N_TICKETS),
    ]
    rows = []
    for _ in range(N_AUDIT_LOGS):
        action, entity_type, max_id = random.choice(actions)
        rows.append(
            (
                random.randint(1, N_USERS),
                action,
                entity_type,
                random.randint(1, max_id),
                Jsonb({"source": random.choice(["web", "api", "batch_job"])}),
                now - timedelta(days=random.randint(0, 90), minutes=random.randint(0, 1439)),
            )
        )
    return rows


def seed(conn: psycopg.Connection) -> None:
    """Wipe all business tables and insert freshly generated rows in one transaction.

    TRUNCATE ... RESTART IDENTITY resets the SERIAL counters, so ids start at 1
    again. That's why the generators can safely use random.randint(1, N_USERS)
    as foreign keys. If anything fails, the transaction rolls back and the
    database is left untouched.
    """
    random.seed(SEED)
    fake = Faker()
    Faker.seed(SEED)
    today = date.today()
    now = datetime.combine(today, time(9, 0), tzinfo=UTC)

    with conn.transaction(), conn.cursor() as cur:
        cur.execute("TRUNCATE audit_logs, tickets, invoices, users, vendors RESTART IDENTITY CASCADE")
        cur.executemany(
            "INSERT INTO vendors (name, contact_email, country, payment_terms_days) VALUES (%s, %s, %s, %s)",
            make_vendors(fake),
        )
        cur.executemany(
            "INSERT INTO users (full_name, email, department, role) VALUES (%s, %s, %s, %s)",
            make_users(fake),
        )
        cur.executemany(
            "INSERT INTO invoices (invoice_number, vendor_id, amount, currency, status, issue_date, due_date,"
            " approved_by) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            make_invoices(today),
        )
        cur.executemany(
            "INSERT INTO tickets (title, description, status, priority, created_by, assigned_to, vendor_id,"
            " created_at, resolved_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            make_tickets(fake, now),
        )
        cur.executemany(
            "INSERT INTO audit_logs (user_id, action, entity_type, entity_id, details, created_at)"
            " VALUES (%s, %s, %s, %s, %s, %s)",
            make_audit_logs(now),
        )


def print_summary(conn: psycopg.Connection) -> None:
    """Print row counts per table plus the Scenario 1 check, to confirm seeding worked."""
    for table in ("vendors", "users", "invoices", "tickets", "audit_logs"):
        count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"  {table:<11} {count} rows")
    due_this_week = conn.execute(
        "SELECT count(*) FROM invoices WHERE status = 'pending'"
        " AND due_date BETWEEN current_date AND date_trunc('week', current_date)::date + 6"
    ).fetchone()[0]
    print(f"  pending invoices due this week: {due_this_week}")


def main() -> None:
    """Connect with the admin URL, seed all tables, and print a summary."""
    with psycopg.connect(get_database_url()) as conn:
        seed(conn)
        print("Seeded database:")
        print_summary(conn)


if __name__ == "__main__":
    main()
