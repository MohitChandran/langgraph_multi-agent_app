# /// script
# requires-python = ">=3.12"
# dependencies = ["faker==40.44.0", "psycopg[binary]==3.3.6", "python-dotenv==1.2.4"]
# ///
"""Fill the business tables with realistic, reproducible mock data.

Run from the repo root (after `docker compose up -d`):
    uv run scripts/seed_db.py
or rebuild everything from scratch with:
    ./scripts/reset_db.sh

The block at the top of this file is "inline script metadata" (PEP 723):
`uv run` reads it and installs the pinned libraries into a throwaway environment.
Exact versions are pinned on purpose: Faker can generate different names in a
different version, and we want every clone of the repo to get identical data.

Reproducible data:
  - A fixed random seed means every run creates the same rows.
  - Dates are generated relative to TODAY, so questions like "pending invoices due
    this week" or "contracts expiring in the next 60 days" always have answers.
  - Data is internally consistent (an overdue invoice is past its due date, a paid
    invoice has a paid_date, cloud contracts have an SLA, ...), so the DB agent is
    tested on realistic logic instead of random noise.

Vendor names are real companies (so the Web Search agent can look up their public
SLAs later), but every contract, invoice, amount and date here is fictional.
Running the script again wipes the tables and re-seeds them.
"""

import os
import random
import re
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from faker import Faker
from psycopg import sql
from psycopg.types.json import Jsonb

SEED = 42
N_USERS = 25
N_INVOICES = 300
N_TICKETS = 100
N_AUDIT_LOGS = 400
HISTORY_DAYS = 210  # invoices and tickets go back ~7 months

# (name, category, country, payment_terms_days)
VENDORS = [
    ("Amazon Web Services", "cloud", "United States", 30),
    ("Microsoft Azure", "cloud", "United States", 30),
    ("Google Cloud", "cloud", "United States", 30),
    ("DigitalOcean", "cloud", "United States", 15),
    ("Oracle Cloud Infrastructure", "cloud", "United States", 45),
    ("Dell Technologies", "hardware", "United States", 45),
    ("Lenovo", "hardware", "China", 60),
    ("Cisco Systems", "hardware", "United States", 45),
    ("HP Inc.", "hardware", "United States", 30),
    ("Deloitte", "consulting", "United Kingdom", 30),
    ("Accenture", "consulting", "Ireland", 45),
    ("Infosys", "consulting", "India", 30),
    ("Thoughtworks", "consulting", "United States", 30),
    ("DHL Supply Chain", "logistics", "Germany", 15),
    ("FedEx", "logistics", "United States", 15),
    ("Blue Dart Express", "logistics", "India", 15),
    ("Baker McKenzie", "legal", "United States", 30),
    ("Khaitan & Co", "legal", "India", 30),
]

CURRENCY_BY_COUNTRY = {
    "United States": "USD",
    "China": "USD",
    "United Kingdom": "GBP",
    "Ireland": "EUR",
    "Germany": "EUR",
    "India": "INR",
}
# Rough multiplier from USD, so an INR invoice looks like a realistic INR amount.
CURRENCY_SCALE = {"USD": 1, "EUR": 0.92, "GBP": 0.79, "INR": 84}

# Annual contract value range in USD, per vendor category.
ANNUAL_VALUE_USD = {
    "cloud": (40_000, 600_000),
    "hardware": (20_000, 250_000),
    "consulting": (60_000, 900_000),
    "logistics": (10_000, 120_000),
    "legal": (15_000, 200_000),
}

SERVICES = {
    "cloud": ["Compute and storage hosting", "Managed Kubernetes platform", "Data warehouse platform"],
    "hardware": ["Laptop fleet supply", "Network equipment and support", "Server maintenance"],
    "consulting": ["ERP implementation", "Security audit", "Data platform advisory"],
    "logistics": ["Office equipment delivery", "International freight"],
    "legal": ["Contract review retainer", "Regulatory compliance advisory"],
}
SLA_OPTIONS = [99.5, 99.9, 99.95, 99.99]

DEPARTMENTS = ["Finance", "Engineering", "Operations", "Procurement", "Support", "Legal"]
ROLES = ["Analyst", "Manager", "Engineer", "Director", "Specialist"]

# (title, description, vendor category the ticket is about: a category, "any", or None = internal)
TICKET_TOPICS = [
    ("SLA breach on cloud hosting", "Service was unavailable longer than the SLA allows.", "cloud"),
    ("Unexpected spike in cloud bill", "Monthly charges are well above the forecast.", "cloud"),
    ("Late delivery of hardware", "Shipment arrived two weeks after the agreed date.", "hardware"),
    ("Defective laptops in last batch", "Several units failed on first boot.", "hardware"),
    ("Consulting milestone delayed", "Deliverable is late versus the statement of work.", "consulting"),
    ("Shipment damaged in transit", "Equipment arrived with visible damage.", "logistics"),
    ("Contract renewal review", "Contract expires soon and needs legal review.", "legal"),
    ("Invoice amount mismatch", "Billed amount does not match the purchase order.", "any"),
    ("Duplicate invoice received", "Vendor sent the same invoice twice.", "any"),
    ("Update vendor bank details", "Vendor requested a change of payout account.", "any"),
    ("Access request for billing portal", "New team member needs vendor portal access.", None),
]

AUDIT_ACTIONS = [  # (action, entity_type)
    ("approve_invoice", "invoice"),
    ("reject_invoice", "invoice"),
    ("update_vendor", "vendor"),
    ("renew_contract", "contract"),
    ("close_ticket", "ticket"),
    ("assign_ticket", "ticket"),
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


def slugify(text: str) -> str:
    """Turn a name into a URL/email-safe slug: 'HP Inc.' -> 'hp-inc'."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def random_date(start: date, end: date) -> date:
    """Return a random date between start and end (both inclusive)."""
    return start + timedelta(days=random.randint(0, (end - start).days))


def business_hours(day: date) -> datetime:
    """Return a UTC timestamp on `day` at a random time between 09:00 and 17:59.

    Used for created_at columns. Setting them explicitly (instead of letting the
    database default to now()) keeps the data identical on every reseed.
    """
    return datetime.combine(day, time(random.randint(9, 17), random.randint(0, 59)), tzinfo=UTC)


def make_vendors(today: date) -> list[dict]:
    """Build vendor rows from the fixed VENDORS list.

    Emails use the reserved `.example.com` domain, so no real address can ever
    receive mail, even when the Email agent sends something later. Vendors are
    created 3-4 years ago, i.e. before their first contract starts.
    """
    return [
        {
            "name": name,
            "category": category,
            "contact_email": f"billing@{slugify(name)}.example.com",
            "country": country,
            "payment_terms_days": terms,
            "created_at": business_hours(random_date(today - timedelta(days=4 * 365), today - timedelta(days=3 * 366))),
        }
        for name, category, country, terms in VENDORS
    ]


def make_users(fake: Faker, today: date) -> list[dict]:
    """Generate employee rows with fake names, unique company emails and a joining date."""
    rows = []
    for i in range(N_USERS):
        name = fake.name()
        rows.append(
            {
                "full_name": name,
                "email": f"{slugify(name).replace('-', '.')}.{i}@company.example",  # index keeps emails unique
                "department": random.choice(DEPARTMENTS),
                "role": random.choice(ROLES),
                "created_at": business_hours(random_date(today - timedelta(days=5 * 365), today - timedelta(days=30))),
            }
        )
    return rows


def make_contracts(vendors: list[dict], user_ids: list[int], today: date) -> list[dict]:
    """Generate 1-3 contracts per vendor, with status consistent with the dates.

    - end_date in the past -> 'expired'
    - a few are 'terminated' early (end_date moved to the termination date)
    - 3 active contracts are forced to end in the next 10-60 days, so
      "which contracts expire soon?" always has answers.
    Only cloud contracts get an SLA uptime percentage.
    """
    rows = []
    for vendor in vendors:
        currency = CURRENCY_BY_COUNTRY[vendor["country"]]
        low, high = ANNUAL_VALUE_USD[vendor["category"]]
        n_contracts = random.randint(1, 3) if vendor["category"] == "cloud" else random.randint(1, 2)
        for _ in range(n_contracts):
            start = random_date(today - timedelta(days=3 * 365), today - timedelta(days=90))
            end = start + timedelta(days=365 * random.choice([1, 2, 3]))
            status = "expired" if end < today else "active"
            if status == "active" and random.random() < 0.1:
                status = "terminated"
                end = random_date(start + timedelta(days=30), today)
            rows.append(
                {
                    "vendor_id": vendor["id"],
                    "service_description": random.choice(SERVICES[vendor["category"]]),
                    "annual_value": round(random.uniform(low, high) * CURRENCY_SCALE[currency], 2),
                    "currency": currency,
                    "start_date": start,
                    "end_date": end,
                    "sla_uptime_pct": random.choice(SLA_OPTIONS) if vendor["category"] == "cloud" else None,
                    "auto_renew": random.random() < 0.4,
                    "status": status,
                    "owner_id": random.choice(user_ids),
                    "created_at": business_hours(start - timedelta(days=random.randint(7, 30))),  # signed before start
                }
            )

    for contract in random.sample(rows, 3):
        contract["end_date"] = today + timedelta(days=random.randint(10, 60))
        contract["status"] = "active"

    rows.sort(key=lambda c: c["start_date"])  # number contracts in chronological order
    for i, contract in enumerate(rows, start=1):
        contract["contract_number"] = f"CTR-{contract['start_date'].year}-{i:03d}"
    return rows


def pick_invoice_status(due_date: date, today: date) -> str:
    """Choose a status that is consistent with the due date.

    An invoice past its due date can't be 'pending' (it's 'overdue' or settled);
    one not yet due can't be 'overdue'.
    """
    if due_date < today:
        return random.choices(["paid", "overdue", "cancelled"], weights=[70, 22, 8])[0]
    return random.choices(["pending", "paid", "cancelled"], weights=[80, 15, 5])[0]


def make_invoices(
    vendors_by_id: dict[int, dict], contracts: list[dict], user_ids: list[int], today: date
) -> list[dict]:
    """Generate invoices over the last HISTORY_DAYS days.

    - ~75% are billed under a contract (monthly share of its annual value, issued
      while the contract was running); the rest are one-off purchases.
    - due_date = issue_date + the vendor's payment terms.
    - The first 4 are forced to be 'pending' and due before Sunday, so
      Scenario 1 ("pending invoices due this week") always returns rows.
    - paid invoices get a paid_date; invoice numbers follow issue-date order.
    """
    window_start = today - timedelta(days=HISTORY_DAYS)
    billable = [c for c in contracts if c["end_date"] >= window_start]
    active = [c for c in contracts if c["status"] == "active"]
    days_left_in_week = 6 - today.weekday()  # Monday=0 ... Sunday=6

    rows = []
    for i in range(N_INVOICES):
        contract = None
        if i < 4:  # forced "pending, due this week"
            contract = random.choice(active)
            vendor = vendors_by_id[contract["vendor_id"]]
            due_date = today + timedelta(days=random.randint(0, days_left_in_week))
            issue_date = due_date - timedelta(days=vendor["payment_terms_days"])
            status = "pending"
        else:
            if random.random() < 0.75:
                contract = random.choice(billable)
                vendor = vendors_by_id[contract["vendor_id"]]
                issue_date = random_date(max(contract["start_date"], window_start), min(contract["end_date"], today))
            else:
                vendor = random.choice(list(vendors_by_id.values()))
                issue_date = random_date(window_start, today)
            due_date = issue_date + timedelta(days=vendor["payment_terms_days"])
            status = pick_invoice_status(due_date, today)

        currency = CURRENCY_BY_COUNTRY[vendor["country"]]
        if contract:
            amount = float(contract["annual_value"]) / 12 * random.uniform(0.85, 1.15)
        else:
            amount = random.uniform(500, 25_000) * CURRENCY_SCALE[currency]

        paid_date = None
        if status == "paid":
            paid_date = min(issue_date + timedelta(days=random.randint(0, vendor["payment_terms_days"] + 15)), today)

        rows.append(
            {
                "vendor_id": vendor["id"],
                "contract_id": contract["id"] if contract else None,
                "amount": round(amount, 2),
                "currency": currency,
                "status": status,
                "issue_date": issue_date,
                "due_date": due_date,
                "paid_date": paid_date,
                "approved_by": random.choice(user_ids) if status == "paid" or random.random() < 0.3 else None,
                "created_at": business_hours(issue_date),  # entered into the system on the day it was issued
            }
        )

    rows.sort(key=lambda r: r["issue_date"])
    for i, invoice in enumerate(rows, start=1):
        invoice["invoice_number"] = f"INV-{invoice['issue_date'].year}-{i:04d}"
    return rows


def make_tickets(fake: Faker, vendors: list[dict], user_ids: list[int], now: datetime) -> list[dict]:
    """Generate tickets whose vendor matches the topic (SLA breaches are about cloud vendors, etc.).

    resolved_at is only set for resolved/closed tickets, is always after
    created_at, and never in the future. Open tickets have no assignee yet.
    """
    rows = []
    for _ in range(N_TICKETS):
        title, description, topic_category = random.choice(TICKET_TOPICS)
        if topic_category is None:
            vendor_id = None
        elif topic_category == "any":
            vendor_id = random.choice(vendors)["id"]
        else:
            vendor_id = random.choice([v for v in vendors if v["category"] == topic_category])["id"]

        status = random.choices(["open", "in_progress", "resolved", "closed"], weights=[30, 25, 25, 20])[0]
        created_at = now - timedelta(days=random.randint(0, HISTORY_DAYS), hours=random.randint(0, 23))
        resolved_at = None
        if status in ("resolved", "closed"):
            resolved_at = min(created_at + timedelta(days=random.randint(1, 14)), now)

        rows.append(
            {
                "title": title,
                "description": f"{description} {fake.sentence()}",
                "status": status,
                "priority": random.choices(["low", "medium", "high", "critical"], weights=[30, 40, 20, 10])[0],
                "created_by": random.choice(user_ids),
                "assigned_to": random.choice(user_ids) if status != "open" else None,
                "vendor_id": vendor_id,
                "created_at": created_at,
                "resolved_at": resolved_at,
            }
        )
    return rows


def make_audit_logs(user_ids: list[int], ids_by_entity: dict[str, list[int]], now: datetime) -> list[dict]:
    """Generate audit entries that point at records that really exist (e.g. approve_invoice -> a real invoice id)."""
    rows = []
    for _ in range(N_AUDIT_LOGS):
        action, entity_type = random.choice(AUDIT_ACTIONS)
        rows.append(
            {
                "user_id": random.choice(user_ids),
                "action": action,
                "entity_type": entity_type,
                "entity_id": random.choice(ids_by_entity[entity_type]),
                "details": Jsonb({"source": random.choice(["web", "api", "batch_job"])}),
                "created_at": now - timedelta(days=random.randint(0, HISTORY_DAYS), minutes=random.randint(0, 1439)),
            }
        )
    return rows


def insert_rows(cur: psycopg.Cursor, table: str, rows: list[dict]) -> None:
    """INSERT a list of dicts into `table` and write each new database id back into its dict.

    The column list comes from the dict keys. psycopg.sql quotes the table and
    column names safely (never build SQL with f-strings). RETURNING id gives us
    the real ids, which later tables need as foreign keys.
    """
    columns = list(rows[0])
    query = sql.SQL("INSERT INTO {table} ({columns}) VALUES ({values}) RETURNING id").format(
        table=sql.Identifier(table),
        columns=sql.SQL(", ").join(map(sql.Identifier, columns)),
        values=sql.SQL(", ").join([sql.Placeholder()] * len(columns)),
    )
    cur.executemany(query, [tuple(row[c] for c in columns) for row in rows], returning=True)
    for row, _ in zip(rows, cur.results(), strict=True):
        row["id"] = cur.fetchone()[0]


def seed(conn: psycopg.Connection) -> None:
    """Wipe all business tables and insert freshly generated rows in one transaction.

    Tables are filled parent-first (vendors, users -> contracts -> invoices ...)
    because children need their parents' ids. If anything fails, the
    transaction rolls back and the database is left untouched.
    """
    random.seed(SEED)
    fake = Faker()
    Faker.seed(SEED)
    today = date.today()
    now = datetime.combine(today, time(9, 0), tzinfo=UTC)

    with conn.transaction(), conn.cursor() as cur:
        cur.execute("TRUNCATE audit_logs, tickets, invoices, contracts, users, vendors RESTART IDENTITY CASCADE")

        vendors = make_vendors(today)
        insert_rows(cur, "vendors", vendors)
        users = make_users(fake, today)
        insert_rows(cur, "users", users)
        user_ids = [u["id"] for u in users]

        contracts = make_contracts(vendors, user_ids, today)
        insert_rows(cur, "contracts", contracts)
        invoices = make_invoices({v["id"]: v for v in vendors}, contracts, user_ids, today)
        insert_rows(cur, "invoices", invoices)
        tickets = make_tickets(fake, vendors, user_ids, now)
        insert_rows(cur, "tickets", tickets)

        ids_by_entity = {
            "invoice": [r["id"] for r in invoices],
            "vendor": [r["id"] for r in vendors],
            "contract": [r["id"] for r in contracts],
            "ticket": [r["id"] for r in tickets],
        }
        insert_rows(cur, "audit_logs", make_audit_logs(user_ids, ids_by_entity, now))


def print_summary(conn: psycopg.Connection) -> None:
    """Print row counts per table plus the demo-scenario checks, to confirm seeding worked."""
    for table in ("vendors", "users", "contracts", "invoices", "tickets", "audit_logs"):
        count = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))).fetchone()[0]
        print(f"  {table:<11} {count} rows")
    checks = {
        "pending invoices due this week": "SELECT count(*) FROM invoices WHERE status = 'pending'"
        " AND due_date BETWEEN current_date AND date_trunc('week', current_date)::date + 6",
        "active contracts ending in 60 days": "SELECT count(*) FROM contracts WHERE status = 'active'"
        " AND end_date BETWEEN current_date AND current_date + 60",
    }
    for label, query in checks.items():
        print(f"  {label}: {conn.execute(query).fetchone()[0]}")


def main() -> None:
    """Connect with the admin URL, seed all tables, and print a summary."""
    with psycopg.connect(get_database_url()) as conn:
        seed(conn)
        print("Seeded database:")
        print_summary(conn)


if __name__ == "__main__":
    main()
