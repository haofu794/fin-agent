"""
M1 数据层：SQLite 表结构与读写。

设计要点（详见学习笔记）：
- 长表（long format）：一行 = 一家公司 × 一个指标 × 一个财年
  → get_metric 能用参数化 SQL 查询，避免拼接列名带来的 SQL injection。
- 主键 (ticker, metric, fiscal_year)：同一个数只能存一份，插入前必须先选定一条记录。
- value 存原始值（美元），换算成 B / M 留给显示层。
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
    conn.row_factory = sqlite3.Row  # 让查询结果可以用列名访问：row["value"]
    conn.executescript(SCHEMA)
    return conn


def upsert_company(conn, ticker, cik, name, fiscal_year_end=None):
    conn.execute(
        "INSERT OR REPLACE INTO companies VALUES (?, ?, ?, ?)",
        (ticker, cik, name, fiscal_year_end),
    )


def upsert_financial(conn, row: dict):
    """插入一条指标；主键冲突时覆盖（重新下载数据时用新值替换旧值）。"""
    conn.execute(
        """INSERT OR REPLACE INTO financials
           (ticker, metric, fiscal_year, value, unit, period_start, period_end,
            tag, accn, form, filed)
           VALUES (:ticker, :metric, :fiscal_year, :value, :unit, :period_start,
                   :period_end, :tag, :accn, :form, :filed)""",
        row,
    )
