"""
M1 data layer: SQLite schema and writes.

Design notes:
- Long format: one row = one company x one metric x one fiscal year,
  so get_metric can use fully parameterised SQL (no column names built from strings -> no SQL injection).
- Primary key (ticker, metric, fiscal_year): each value is stored once, so a single record
  must be chosen before inserting.
- `value` stores the raw figure (USD); scaling to B / M is left to the display layer.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    ticker          TEXT PRIMARY KEY,
    cik             INTEGER NOT NULL,
    name            TEXT NOT NULL,
    fiscal_year_end TEXT
);

CREATE TABLE IF NOT EXISTS financials (
    ticker       TEXT NOT NULL,
    metric       TEXT NOT NULL,
    fiscal_year  INTEGER NOT NULL,
    value        REAL NOT NULL,
    unit         TEXT NOT NULL,
    period_start TEXT,
    period_end   TEXT NOT NULL,
    tag          TEXT NOT NULL,
    accn         TEXT NOT NULL,
    form         TEXT NOT NULL,
    filed        TEXT NOT NULL,
    PRIMARY KEY (ticker, metric, fiscal_year)
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # access columns by name: row["value"]
    conn.executescript(SCHEMA)
    return conn


def upsert_company(conn, ticker, cik, name, fiscal_year_end=None):
    conn.execute(
        "INSERT OR REPLACE INTO companies VALUES (?, ?, ?, ?)",
        (ticker, cik, name, fiscal_year_end),
    )


def upsert_financial(conn, row: dict):
    """Insert one metric value; on primary-key conflict, replace it (new downloads overwrite old values)."""
    conn.execute(
        """INSERT OR REPLACE INTO financials
           (ticker, metric, fiscal_year, value, unit, period_start, period_end,
            tag, accn, form, filed)
           VALUES (:ticker, :metric, :fiscal_year, :value, :unit, :period_start,
                   :period_end, :tag, :accn, :form, :filed)""",
        row,
    )
