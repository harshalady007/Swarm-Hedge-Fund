"""Signal journal and paper portfolio.

Fill rules (deliberately conservative):
  * A BUY decided after the close of day S fills at the open of the next session,
    only if that open is at or below the entry zone top and above the stop.
    Otherwise it expires unfilled. Its cash is reserved while it is pending.
  * Stops and targets are checked against each later day's low and high,
    including the fill day. When both are touched on one day, the stop wins.
  * An open at or below the stop exits at that open (a gap through the stop),
    not at the stop price. An open at or above the target exits at that open.
  * A SELL call exits at the next session's open.
  * Every fill pays COST_PER_SIDE.
"""
import json

import pandas as pd

import config
import db

C = config.COST_PER_SIDE


# ------------------------------------------------------------------ state
def ensure_start(con, day, nifty_close):
    if db.get_meta(con, "start_date") is None:
        db.set_meta(con, "start_date", day)
        db.set_meta(con, "start_capital", config.STARTING_CAPITAL)
        db.set_meta(con, "start_nifty", nifty_close)


def start_capital(con):
    return float(db.get_meta(con, "start_capital", config.STARTING_CAPITAL))


def cash(con):
    """Cash is derived from the trade ledger, never stored, so it can be audited."""
    spent = con.execute("SELECT COALESCE(SUM(entry_cash), 0) FROM positions").fetchone()[0]
    back = con.execute("SELECT COALESCE(SUM(exit_cash), 0) FROM positions WHERE status='CLOSED'").fetchone()[0]
    return start_capital(con) - spent + back


def reserved(con):
    return con.execute("SELECT COALESCE(SUM(reserved), 0) FROM orders "
                       "WHERE status='PENDING' AND side='BUY'").fetchone()[0]


def open_positions(con):
    return [dict(r) for r in con.execute("SELECT * FROM positions WHERE status='OPEN' ORDER BY id")]


def pending_orders(con):
    return [dict(r) for r in con.execute("SELECT * FROM orders WHERE status='PENDING' ORDER BY id")]


def held_symbols(con):
    return {p["symbol"] for p in open_positions(con)}


def last_close(prices, symbol, day=None):
    df = prices.get(symbol)
    if df is None or not len(df):
        return None
    if day is not None:
        df = df[df.index <= pd.Timestamp(day)]
    return float(df["Close"].iloc[-1]) if len(df) else None


def mark(con, prices, day=None):
    """Cash, reservations, market value of open positions and equity at the latest close."""
    pos_value = 0.0
    for p in open_positions(con):
        px = last_close(prices, p["symbol"], day)
        pos_value += p["qty"] * (px if px is not None else p["entry_price"])
    c = cash(con)
    return {"cash": c, "reserved": reserved(con), "positions_value": pos_value, "equity": c + pos_value}


def record_equity(con, day, prices, nifty_close):
    m = mark(con, prices, day)
    con.execute("INSERT OR REPLACE INTO equity_curve (day, cash, reserved, positions_value, equity, nifty_close)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (day, m["cash"], m["reserved"], m["positions_value"], m["equity"], nifty_close))
    con.commit()
    return m


def _event(con, day, kind, symbol, message):
    con.execute("INSERT INTO events (day, kind, symbol, message) VALUES (?, ?, ?, ?)",
                (day, kind, symbol, message))


def unnotified_events(con):
    rows = [dict(r) for r in con.execute("SELECT * FROM events WHERE notified=0 ORDER BY day, id")]
    con.execute("UPDATE events SET notified=1 WHERE notified=0")
    con.commit()
    return rows


# --------------------------------------------------------------- advancing
def advance(con, prices, calendar, upto_day):
    """Apply every bar after the last processed day up to `upto_day`.

    calendar: sorted list of trading days (date strings) from the benchmark; the
    "next session" after a signal is the next date in this calendar.
    Idempotent: bars already applied are never applied again.
    """
    last = db.get_meta(con, "last_advanced")
    new_days = [d for d in calendar if (last is None or d > last) and d <= upto_day]
    if last is None:
        # First run: nothing can be pending yet; just start the clock.
        db.set_meta(con, "last_advanced", upto_day)
        return []
    for day in new_days:
        _apply_day(con, prices, day, calendar)
        db.set_meta(con, "last_advanced", day)
    con.commit()
    return new_days


def _bar(prices, symbol, day):
    df = prices.get(symbol)
    if df is None:
        return None
    ts = pd.Timestamp(day)
    if ts not in df.index:
        return None
    row = df.loc[ts]
    return {k: float(row[k]) for k in ("Open", "High", "Low", "Close")}


def _next_session(calendar, day):
    later = [d for d in calendar if d > day]
    return later[0] if later else None


def _apply_day(con, prices, day, calendar):
    # 1) orders whose next session is today
    for o in pending_orders(con):
        if _next_session(calendar, o["created_day"]) != day:
            if o["created_day"] < day:   # its session has passed without data
                _resolve(con, o, "EXPIRED", day, note="no price bar for the next session")
                _event(con, day, "EXPIRED", o["symbol"], f"{o['side']} {o['symbol']} expired: no price bar")
            continue
        bar = _bar(prices, o["symbol"], day)
        if bar is None:
            _resolve(con, o, "EXPIRED", day, note="no price bar for the next session")
            _event(con, day, "EXPIRED", o["symbol"], f"{o['side']} {o['symbol']} expired: no price bar for {day}")
            continue
        if o["side"] == "SELL":
            for p in open_positions(con):
                if p["symbol"] == o["symbol"]:
                    _close(con, p, day, bar["Open"], "SELL_SIGNAL")
            _resolve(con, o, "FILLED", day, fill_price=bar["Open"])
            continue
        if bar["Open"] <= o["entry_high"] and bar["Open"] > o["stop"]:
            _resolve(con, o, "FILLED", day, fill_price=bar["Open"])
            cur = con.execute(
                "INSERT INTO positions (order_id, signal_day, symbol, sector, qty, entry_day, entry_price,"
                " entry_cash, stop, target, entry_high, status) VALUES (?,?,?,?,?,?,?,?,?,?,?, 'OPEN')",
                (o["id"], o["created_day"], o["symbol"], o["sector"], o["qty"], day, bar["Open"],
                 o["qty"] * bar["Open"] * (1 + C), o["stop"], o["target"], o["entry_high"]))
            _event(con, day, "FILL", o["symbol"],
                   f"BUY {o['symbol']} filled: {o['qty']} @ {bar['Open']:.2f} "
                   f"(stop {o['stop']:.2f}, target {o['target']:.2f})")
            pos = dict(con.execute("SELECT * FROM positions WHERE id=?", (cur.lastrowid,)).fetchone())
            _check_exits(con, pos, bar, day, filled_today=True)
        else:
            why = (f"open {bar['Open']:.2f} above entry top {o['entry_high']:.2f}"
                   if bar["Open"] > o["entry_high"] else f"open {bar['Open']:.2f} at or below stop {o['stop']:.2f}")
            _resolve(con, o, "EXPIRED", day, note=why)
            _event(con, day, "EXPIRED", o["symbol"], f"BUY {o['symbol']} expired unfilled: {why}")

    # 2) stops and targets on positions that were already open before today
    for p in open_positions(con):
        if p["entry_day"] >= day:
            continue
        bar = _bar(prices, p["symbol"], day)
        if bar is not None:
            _check_exits(con, p, bar, day, filled_today=False)


def _check_exits(con, p, bar, day, filled_today):
    stop, target = p["stop"], p["target"]
    if not filled_today and bar["Open"] <= stop:
        _close(con, p, day, bar["Open"], "STOP_GAP")
    elif bar["Low"] <= stop:                       # stop wins when both are touched
        _close(con, p, day, stop, "STOP")
    elif not filled_today and bar["Open"] >= target:
        _close(con, p, day, bar["Open"], "TARGET_GAP")
    elif bar["High"] >= target:
        _close(con, p, day, target, "TARGET")


def _close(con, p, day, price, reason):
    exit_cash = p["qty"] * price * (1 - C)
    pnl = exit_cash - p["entry_cash"]
    con.execute("UPDATE positions SET status='CLOSED', exit_day=?, exit_price=?, exit_reason=?,"
                " exit_cash=?, pnl=? WHERE id=?", (day, price, reason, exit_cash, pnl, p["id"]))
    _event(con, day, "EXIT", p["symbol"],
           f"{p['symbol']} exited ({reason}): {p['qty']} @ {price:.2f}, P&L {pnl:+,.0f} "
           f"({pnl / p['entry_cash']:+.1%})")


def _resolve(con, o, status, day, fill_price=None, note=None):
    con.execute("UPDATE orders SET status=?, resolved_day=?, fill_price=?, note=?, reserved=0 WHERE id=?",
                (status, day, fill_price, note, o["id"]))


# --------------------------------------------------------------- logging
def already_processed(con, day):
    return con.execute("SELECT 1 FROM runs WHERE trading_day=?", (day,)).fetchone() is not None


def log_signal(con, day, row, detail, agent_io):
    cur = con.execute(
        "INSERT INTO signals (trading_day, symbol, sector, held, screen_score, close, action_raw, action,"
        " confidence, status, veto_reason, risk_note, entry_low, entry_high, stop, target, qty, detail,"
        " agent_mode, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (day, row["symbol"], row["sector"], int(row["held"]), row.get("screen_score"), row.get("close"),
         row.get("action_raw"), row["action"], row.get("confidence"), row["status"], row.get("veto_reason"),
         row.get("risk_note"), row.get("entry_low"), row.get("entry_high"), row.get("stop"),
         row.get("target"), row.get("qty"), json.dumps(detail, default=str), row.get("agent_mode"),
         db.now_iso()))
    sid = cur.lastrowid
    for a in agent_io:
        con.execute("INSERT INTO agent_io (signal_id, agent, model, input_text, output_json, error, created_at)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (sid, a["agent"], a.get("model"), a["input_text"],
                     json.dumps(a.get("output"), default=str), a.get("error"), db.now_iso()))
    return sid


def place_order(con, signal_id, day, row):
    if row["status"] == "APPROVED":
        reserve = row["qty"] * row["entry_high"] * (1 + C)
        con.execute("INSERT INTO orders (signal_id, symbol, sector, side, created_day, status, entry_low,"
                    " entry_high, stop, target, qty, reserved) VALUES (?,?,?,?,?, 'PENDING', ?,?,?,?,?,?)",
                    (signal_id, row["symbol"], row["sector"], "BUY", day, row["entry_low"], row["entry_high"],
                     row["stop"], row["target"], row["qty"], reserve))
    elif row["status"] == "EXIT":
        qty = sum(p["qty"] for p in open_positions(con) if p["symbol"] == row["symbol"])
        con.execute("INSERT INTO orders (signal_id, symbol, sector, side, created_day, status, qty)"
                    " VALUES (?,?,?,?,?, 'PENDING', ?)",
                    (signal_id, row["symbol"], row["sector"], "SELL", day, qty))


def finish_run(con, day, data_mode, agent_mode, n_reviewed, n_buy, n_sell, note=""):
    con.execute("INSERT INTO runs (trading_day, run_at, data_mode, agent_mode, n_reviewed, n_buy, n_sell, note)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (day, db.now_iso(), data_mode, agent_mode, n_reviewed, n_buy, n_sell, note))
    con.commit()


# --------------------------------------------------------------- reporting
def trade_stats(closed):
    """Win rate and average win/loss in percent of the cash put in."""
    rets = [p["pnl"] / p["entry_cash"] * 100 for p in closed]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    return {
        "n": len(rets),
        "win_rate_pct": (len(wins) / len(rets) * 100) if rets else None,
        "avg_win_pct": (sum(wins) / len(wins)) if wins else None,
        "avg_loss_pct": (sum(losses) / len(losses)) if losses else None,
        "realised_pnl": sum(p["pnl"] for p in closed),
    }


def sample_note(n):
    if n == 0:
        return "no closed trades yet - nothing to learn from the record"
    if n < 20:
        return f"only {n} closed trade(s) - far too few to judge skill; treat as anecdote"
    if n < 50:
        return f"{n} closed trades - still a small sample"
    return f"{n} closed trades"


def report(con):
    start = db.get_meta(con, "start_date")
    if start is None:
        return "No runs recorded yet."
    last = con.execute("SELECT * FROM equity_curve ORDER BY day DESC LIMIT 1").fetchone()
    cap, nifty0 = start_capital(con), float(db.get_meta(con, "start_nifty"))
    closed = [dict(r) for r in con.execute("SELECT * FROM positions WHERE status='CLOSED' ORDER BY exit_day")]
    st = trade_stats(closed)
    counts = {r["status"]: r["n"] for r in con.execute("SELECT status, COUNT(*) n FROM signals GROUP BY status")}
    runs = con.execute("SELECT agent_mode, COUNT(*) n FROM runs GROUP BY agent_mode").fetchall()
    lines = ["PAPER PORTFOLIO REPORT (paper trading - no real orders)",
             f"Start {start}: capital {cap:,.0f}, NIFTY {nifty0:,.2f}"]
    if last:
        ret = (last["equity"] / cap - 1) * 100
        nret = (last["nifty_close"] / nifty0 - 1) * 100 if last["nifty_close"] else float("nan")
        lines += [f"As of {last['day']}: equity {last['equity']:,.0f} (cash {last['cash']:,.0f}, "
                  f"reserved {last['reserved']:,.0f}, positions {last['positions_value']:,.0f})",
                  f"Return {ret:+.2f}% vs NIFTY buy-and-hold {nret:+.2f}% (difference {ret - nret:+.2f} pp)"]
    fmt = lambda v: "n/a" if v is None else f"{v:+.2f}%"  # noqa: E731
    win_rate = "n/a" if st["win_rate_pct"] is None else f"{st['win_rate_pct']:.0f}%"
    lines += [f"Closed trades: {st['n']}  win rate: {win_rate}  "
              f"avg win {fmt(st['avg_win_pct'])}  avg loss {fmt(st['avg_loss_pct'])}  "
              f"realised P&L {st['realised_pnl']:+,.0f}",
              f"  ({sample_note(st['n'])})"]
    opens = open_positions(con)
    lines.append(f"Open positions: {len(opens)}")
    for p in opens:
        lines.append(f"  {p['symbol']:<11} {p['qty']:>5} @ {p['entry_price']:.2f} since {p['entry_day']}"
                     f"  stop {p['stop']:.2f} target {p['target']:.2f}")
    for p in closed[-10:]:
        lines.append(f"  closed {p['symbol']:<11} {p['entry_day']} -> {p['exit_day']} {p['exit_reason']:<11}"
                     f" P&L {p['pnl']:+,.0f}")
    lines.append("Signals: " + ", ".join(f"{k.lower()} {counts.get(k, 0)}"
                                         for k in ("APPROVED", "VETOED", "HOLD", "EXIT")))
    lines.append("Runs by agent mode: " + (", ".join(f"{r['agent_mode']} {r['n']}" for r in runs) or "none"))
    return "\n".join(lines)
