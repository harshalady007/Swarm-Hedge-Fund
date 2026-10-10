"""SQLite schema, the per-day response cache and the point-in-time snapshot store."""
import json
import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    trading_day TEXT PRIMARY KEY,
    run_at TEXT NOT NULL,
    data_mode TEXT NOT NULL,          -- live | offline
    agent_mode TEXT NOT NULL,         -- llm | rule-based
    n_reviewed INTEGER,
    n_buy INTEGER,
    n_sell INTEGER,
    note TEXT
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trading_day TEXT NOT NULL,
    symbol TEXT NOT NULL,
    sector TEXT,
    held INTEGER NOT NULL,
    screen_score REAL,
    close REAL,
    action_raw TEXT,                  -- what the manager said
    action TEXT NOT NULL,             -- after code enforcement
    confidence INTEGER,
    status TEXT NOT NULL,             -- APPROVED | VETOED | HOLD | EXIT
    veto_reason TEXT,
    risk_note TEXT,
    entry_low REAL, entry_high REAL, stop REAL, target REAL, qty INTEGER,
    detail TEXT,                      -- JSON: every agent output
    agent_mode TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (trading_day, symbol)
);
CREATE TABLE IF NOT EXISTS agent_io (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL REFERENCES signals(id),
    agent TEXT NOT NULL,
    model TEXT,
    input_text TEXT NOT NULL,         -- exactly what the agent was shown
    output_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER REFERENCES signals(id),
    symbol TEXT NOT NULL,
    sector TEXT,
    side TEXT NOT NULL,               -- BUY | SELL
    created_day TEXT NOT NULL,
    status TEXT NOT NULL,             -- PENDING | FILLED | EXPIRED
    entry_low REAL, entry_high REAL, stop REAL, target REAL,
    qty INTEGER,
    reserved REAL DEFAULT 0,          -- cash held back while pending
    resolved_day TEXT,
    fill_price REAL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER REFERENCES orders(id),
    signal_day TEXT NOT NULL,
    symbol TEXT NOT NULL,
    sector TEXT,
    qty INTEGER NOT NULL,
    entry_day TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_cash REAL NOT NULL,         -- qty * price + cost
    stop REAL NOT NULL,
    target REAL NOT NULL,
    entry_high REAL,
    checked_through TEXT,             -- last bar date checked for stop/target
    status TEXT NOT NULL,             -- OPEN | CLOSED
    exit_day TEXT,
    exit_price REAL,
    exit_reason TEXT,                 -- STOP | STOP_GAP | TARGET | TARGET_GAP | SELL_SIGNAL
    exit_cash REAL,                   -- qty * price - cost
    pnl REAL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day TEXT NOT NULL,
    kind TEXT NOT NULL,               -- FILL | EXPIRED | EXIT
    symbol TEXT,
    message TEXT NOT NULL,
    notified INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS equity_curve (
    day TEXT PRIMARY KEY,
    cash REAL NOT NULL,
    reserved REAL NOT NULL,
    positions_value REAL NOT NULL,
    equity REAL NOT NULL,
    nifty_close REAL
);
CREATE TABLE IF NOT EXISTS http_cache (
    key TEXT PRIMARY KEY,             -- dataset|args|day
    fetched_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    trading_day TEXT NOT NULL,
    symbol TEXT NOT NULL,             -- _MARKET for market-wide datasets
    dataset TEXT NOT NULL,
    payload TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (trading_day, symbol, dataset)
);
CREATE TABLE IF NOT EXISTS fii_dii (
    date TEXT NOT NULL,
    category TEXT NOT NULL,           -- FII | DII
    buy_cr REAL, sell_cr REAL, net_cr REAL,
    source TEXT,
    PRIMARY KEY (date, category)
);
CREATE TABLE IF NOT EXISTS delivery (
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    deliv_pct REAL,
    deliv_qty REAL,
    traded_qty REAL,
    source TEXT,
    PRIMARY KEY (date, symbol)
);
CREATE TABLE IF NOT EXISTS doc_summaries (
    url TEXT PRIMARY KEY,
    symbol TEXT,
    filed_at TEXT,
    subject TEXT,
    n_chars INTEGER,
    summary TEXT NOT NULL,            -- JSON
    model TEXT,
    created_at TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    con.commit()
    return con


def get_meta(con, key, default=None):
    row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(con, key, value):
    con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value)))
    con.commit()


# ------------------------------------------------------------- response cache
def cache_get(con, key):
    row = con.execute("SELECT payload FROM http_cache WHERE key = ?", (key,)).fetchone()
    return json.loads(row["payload"]) if row else None


def cache_put(con, key, payload):
    con.execute("INSERT OR REPLACE INTO http_cache (key, fetched_at, payload) VALUES (?, ?, ?)",
                (key, now_iso(), json.dumps(payload, default=str)))
    con.commit()


# ------------------------------------------------------------- snapshot store
def snapshot(con, trading_day, symbol, dataset, payload):
    """Save what was fetched for (day, symbol, dataset). The first write wins,
    so reruns never overwrite the point-in-time record."""
    con.execute("INSERT OR IGNORE INTO snapshots (trading_day, symbol, dataset, payload, fetched_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (trading_day, symbol, dataset, json.dumps(payload, default=str), now_iso()))
    con.commit()
