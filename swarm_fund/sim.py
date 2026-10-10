"""Offline simulation: run the daily pipeline over consecutive business days with
--offline --asof against a scratch database, checking the paper-trading invariants
after every day.

    python sim.py --days 90 --start 2026-03-02 --db /tmp/sim.db
"""
import argparse
import hashlib
import io
import os
from contextlib import redirect_stdout

import pandas as pd

import config
import db
import journal
import pipeline

TABLES = ("runs", "signals", "agent_io", "orders", "positions", "events", "equity_curve", "snapshots")


def fingerprint(con):
    h = hashlib.sha256()
    for t in TABLES:
        for row in con.execute(f"SELECT * FROM {t} ORDER BY 1"):
            vals = [v for k, v in zip(row.keys(), tuple(row)) if k not in ("created_at", "fetched_at", "run_at")]
            h.update(repr(vals).encode())
    return h.hexdigest()


def check_invariants(con):
    """Raise AssertionError on any broken rule. Returns a few counters."""
    problems = []
    for p in con.execute("SELECT * FROM positions"):
        if p["entry_day"] <= p["signal_day"]:
            problems.append(f"{p['symbol']} entered {p['entry_day']} on/before signal {p['signal_day']}")
        if p["entry_price"] > p["entry_high"] + 1e-9:
            problems.append(f"{p['symbol']} filled {p['entry_price']:.2f} above entry top {p['entry_high']:.2f}")
        if p["exit_reason"] in ("STOP", "STOP_GAP") and p["exit_price"] > p["stop"] + 1e-9:
            problems.append(f"{p['symbol']} stop exit {p['exit_price']:.2f} above stop {p['stop']:.2f}")
        if p["exit_day"] is not None and p["exit_day"] < p["entry_day"]:
            problems.append(f"{p['symbol']} exited before entry")
    cash = journal.cash(con)
    if cash < -0.01:
        problems.append(f"cash negative: {cash:.2f}")
    if cash - journal.reserved(con) < -0.01:
        problems.append(f"cash minus reservations negative: {cash - journal.reserved(con):.2f}")
    opens = journal.open_positions(con)
    if len(opens) > config.MAX_POSITIONS:
        problems.append(f"{len(opens)} open positions")
    per_sector = {}
    for p in opens:
        per_sector[p["sector"]] = per_sector.get(p["sector"], 0) + 1
    for s, n in per_sector.items():
        if n > config.MAX_PER_SECTOR:
            problems.append(f"{n} open positions in {s}")
    for r in con.execute("SELECT trading_day, COUNT(*) n FROM signals WHERE status='APPROVED' GROUP BY trading_day"):
        if r["n"] > config.MAX_BUYS_PER_DAY:
            problems.append(f"{r['n']} BUYs approved on {r['trading_day']}")
    for r in con.execute("SELECT COUNT(*) n FROM (SELECT symbol FROM positions WHERE status='OPEN' "
                         "GROUP BY symbol HAVING COUNT(*) > 1)"):
        if r["n"]:
            problems.append("a symbol has two open positions")
    if problems:
        raise AssertionError("; ".join(problems))
    return {"open": len(opens), "cash": cash}


def simulate(db_path, start="2026-03-02", days=90, verbose=False):
    if os.path.exists(db_path):
        os.remove(db_path)
    dates = [d.date().isoformat() for d in pd.bdate_range(start, periods=days)]
    for i, d in enumerate(dates):
        buf = io.StringIO()
        with redirect_stdout(buf):
            pipeline.run(db_path, offline=True, asof=d, quiet=True)
        con = db.connect(db_path)
        check_invariants(con)
        if i % 10 == 0 or i == len(dates) - 1:
            # rerunning the same day must change nothing at all
            before = fingerprint(con)
            with redirect_stdout(io.StringIO()):
                out = pipeline.run(db_path, offline=True, asof=d, quiet=True)
            con2 = db.connect(db_path)
            assert out["skipped"], f"second run on {d} was not skipped"
            assert fingerprint(con2) == before, f"second run on {d} changed the database"
        if verbose:
            print(f"{d} ok")
    con = db.connect(db_path)
    stats = journal.trade_stats([dict(r) for r in con.execute("SELECT * FROM positions WHERE status='CLOSED'")])
    counts = {r["status"]: r["n"] for r in con.execute("SELECT status, COUNT(*) n FROM signals GROUP BY status")}
    exits = {r["exit_reason"]: r["n"] for r in con.execute(
        "SELECT exit_reason, COUNT(*) n FROM positions WHERE status='CLOSED' GROUP BY exit_reason")}
    expired = con.execute("SELECT COUNT(*) FROM orders WHERE status='EXPIRED'").fetchone()[0]
    return {"days": len(dates), "first": dates[0], "last": dates[-1], "signals": counts, "exits": exits,
            "expired_orders": expired, "positions": con.execute("SELECT COUNT(*) FROM positions").fetchone()[0],
            "closed": stats, "report": journal.report(con)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--start", default="2026-03-02")
    ap.add_argument("--db", default="sim.db")
    a = ap.parse_args()
    res = simulate(a.db, a.start, a.days, verbose=True)
    print(res["report"])
    print({k: v for k, v in res.items() if k != "report"})
    print("ALL INVARIANTS HELD")
