"""Code-side calculations that turn raw datasets into finished numbers for the agents.

All maths lives here (or in indicators.py) so it can be tested and audited.
Lists handed to agents are chronological, oldest first.
"""
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config


def _r(x, places=1):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return int(round(float(x))) if places == 0 else round(float(x), places)


def pct_change(new, old):
    """Percent change, or None when either side is missing or the base is not positive
    (growth from a loss or from zero is not a meaningful percentage)."""
    if new is None or old is None or old <= 0:
        return None
    return (new / old - 1.0) * 100.0


# ============================================================ technical (phase 1)
def _px(x):
    """Prices: one decimal above 1,000 rupees, two below."""
    return round(float(x), 1 if abs(x) >= 1000 else 2)


def compact_bars(df: pd.DataFrame, n=None):
    """Last n daily bars as compact CSV rows (volume in thousands of shares)."""
    n = n or config.TECH_DAILY_BARS
    tail = df.tail(n)
    rows = [f"{d.strftime('%Y-%m-%d')},{_px(r.Open)},{_px(r.High)},{_px(r.Low)},{_px(r.Close)},"
            f"{int(round(r.Volume / 1000))}" for d, r in tail.iterrows()]
    return {"columns": "date,open,high,low,close,volume_thousands", "rows": rows}


def weekly_closes(df: pd.DataFrame, weeks=None):
    """Friday-ending weekly closes for the past year; the last week may be partial."""
    weeks = weeks or config.TECH_WEEKLY_CLOSES
    s = df["Close"].resample("W-FRI").last().dropna()
    s = s[s.index <= df.index[-1] + pd.Timedelta(days=6)].tail(weeks)
    rows = [f"{d.strftime('%Y-%m-%d')}:{_px(v)}" for d, v in s.items()]
    partial = df.index[-1].weekday() != 4
    return {"rows": rows, "last_week_partial": partial}


# ========================================================= fundamental (phase 1)
def _qkey(iso):
    d = date.fromisoformat(iso)
    return d.year, (d.month - 1) // 3 + 1


def _prev_q(key, back=1):
    y, q = key
    idx = y * 4 + (q - 1) - back
    return idx // 4, idx % 4 + 1


def quarterly_table(records, day: date, is_asof=False):
    """Quarter-on-quarter and year-on-year growth and margins, matched by quarter
    (never by column position, because Yahoo skips quarters). Returns (table, gaps)."""
    gaps = []
    recs = sorted(records or [], key=lambda r: r["period_end"])
    if is_asof:
        lag = timedelta(days=config.RESULTS_PUBLICATION_LAG_DAYS)
        hidden = [r for r in recs if date.fromisoformat(r["period_end"]) + lag > day]
        recs = [r for r in recs if date.fromisoformat(r["period_end"]) + lag <= day]
        if hidden:
            gaps.append(f"{len(hidden)} quarter(s) hidden because they would not have been published by {day}")
    if not recs:
        return None, gaps + ["quarterly results: no quarters available"]
    by_q = {_qkey(r["period_end"]): r for r in recs}
    rows = []
    for r in recs:
        k = _qkey(r["period_end"])
        prev, year_ago = by_q.get(_prev_q(k, 1)), by_q.get(_prev_q(k, 4))
        rev, op, net, eps = (r.get(x) for x in ("revenue", "operating_profit", "net_profit", "eps"))
        row = {"period_end": r["period_end"],
               "revenue_cr": _r(rev / 1e7 if rev is not None else None, 0),
               "operating_profit_cr": _r(op / 1e7 if op is not None else None, 0),
               "net_profit_cr": _r(net / 1e7 if net is not None else None, 0),
               "eps": _r(eps, 2),
               "operating_margin_pct": _r(op / rev * 100 if (op is not None and rev) else None),
               "net_margin_pct": _r(net / rev * 100 if (net is not None and rev) else None)}
        if r.get("operating_profit_basis") == "EBIT":
            row["operating_profit_basis"] = "EBIT (Operating Income missing)"
        for name, field in (("revenue", "revenue"), ("net_profit", "net_profit"), ("eps", "eps")):
            row[f"{name}_qoq_pct"] = _r(pct_change(r.get(field), prev.get(field)) if prev else None)
            row[f"{name}_yoy_pct"] = _r(pct_change(r.get(field), year_ago.get(field)) if year_ago else None)
        rows.append({k2: v for k2, v in row.items() if v is not None})
    # missing quarters inside the range
    first, last = _qkey(recs[0]["period_end"]), _qkey(recs[-1]["period_end"])
    k, missing = last, []
    while k != first:
        k = _prev_q(k)
        if k not in by_q and k != first:
            missing.append(f"Q{k[1]} {k[0]}")
    if missing:
        gaps.append("quarters missing from Yahoo's data: " + ", ".join(sorted(missing)))
    for field, label in (("operating_profit", "operating profit"), ("eps", "EPS"), ("revenue", "revenue")):
        if all(r.get(field) is None for r in recs):
            gaps.append(f"{label} not reported by Yahoo for any quarter")
    age = (day - date.fromisoformat(recs[-1]["period_end"])).days
    if age > 150:
        gaps.append(f"latest quarter ended {recs[-1]['period_end']}, {age} days ago: Yahoo's data looks stale")
    table = {"quarters": rows[-config.QUARTERS_TO_SHOW:], "latest_quarter_age_days": age,
             "margin_trend": margin_trend(rows)}
    return table, gaps


def margin_trend(rows):
    """Latest operating (else net) margin against the average of up to three prior
    quarters and against the year-ago quarter, in percentage points."""
    key = "operating_margin_pct" if any("operating_margin_pct" in r for r in rows) else "net_margin_pct"
    series = [(r["period_end"], r.get(key)) for r in rows if r.get(key) is not None]
    if len(series) < 2:
        return None
    latest_day, latest = series[-1]
    prior = [v for _, v in series[-4:-1]]
    out = {"measure": key, "latest_pct": latest, "prior_avg_pct": _r(sum(prior) / len(prior)),
           "change_vs_prior_avg_pp": _r(latest - sum(prior) / len(prior)), "prior_quarters_used": len(prior)}
    ya = [v for d, v in series if _qkey(d) == _prev_q(_qkey(latest_day), 4)]
    if ya:
        out["change_vs_year_ago_pp"] = _r(latest - ya[0])
    return out


# ============================================================== macro (phase 1)
def window_return(close: pd.Series, n):
    c = close.dropna()
    if len(c) <= n or c.iloc[-1 - n] == 0:
        return None
    return (c.iloc[-1] / c.iloc[-1 - n] - 1.0) * 100.0


def usable_index(df, day: date):
    """A sector index is usable only with enough history and a recent last bar.
    (Yahoo answers some NSE index tickers with a single quote row.)"""
    if df is None or len(df) < config.SECTOR_INDEX_MIN_BARS:
        return False, f"{0 if df is None else len(df)} bars (< {config.SECTOR_INDEX_MIN_BARS})"
    stale = (day - df.index[-1].date()).days
    if stale > config.SECTOR_INDEX_MAX_STALE_DAYS:
        return False, f"last bar {stale} days old"
    return True, f"{len(df)} bars"


def pick_sector_benchmarks(prices, day: date):
    """{sector: (ticker | None, reason)} after checking every Yahoo candidate."""
    out = {}
    for sector, candidates in config.SECTOR_INDEX_CANDIDATES.items():
        chosen, notes = None, []
        for t in candidates:
            ok, why = usable_index(prices.get(t), day)
            notes.append(f"{t}: {why}")
            if ok:
                chosen = t
                break
        out[sector] = (chosen, "; ".join(notes) if notes else "no Yahoo index configured")
    return out


def sector_context(sym, sector, prices, benchmarks, nifty: pd.DataFrame):
    """20- and 60-day sector return and the stock's return relative to sector and NIFTY.
    Relative strength is the simple difference in percentage points."""
    stock = prices[sym]["Close"]
    chosen, why = benchmarks.get(sector, (None, ""))
    out, gaps = {"sector": sector}, []
    if chosen:
        sec_close = prices[chosen]["Close"]
        out["sector_benchmark"] = f"Yahoo index {chosen}"
        sec = {n: window_return(sec_close, n) for n in (20, 60)}
    else:
        peers = [s for s, sec_name in config.UNIVERSE.items() if sec_name == sector and s != sym
                 and s in prices and len(prices[s]) > 61]
        if peers:
            out["sector_benchmark"] = f"equal-weight basket of {len(peers)} peer(s): {', '.join(peers)}"
            sec = {}
            for n in (20, 60):
                rets = [window_return(prices[p]["Close"], n) for p in peers]
                rets = [x for x in rets if x is not None]
                sec[n] = sum(rets) / len(rets) if rets else None
        else:
            sec = {20: None, 60: None}
            gaps.append(f"sector performance: no usable Yahoo index ({why}) and no {sector} peers in the universe")
    nif = {n: window_return(nifty["Close"], n) for n in (20, 60)}
    for n in (20, 60):
        st = window_return(stock, n)
        out[f"stock_ret{n}_pct"] = _r(st, 2)
        out[f"sector_ret{n}_pct"] = _r(sec[n], 2)
        out[f"nifty_ret{n}_pct"] = _r(nif[n], 2)
        out[f"stock_vs_sector_{n}d_pp"] = _r(st - sec[n], 2) if st is not None and sec[n] is not None else None
        out[f"stock_vs_nifty_{n}d_pp"] = _r(st - nif[n], 2) if st is not None and nif[n] is not None else None
        out[f"sector_vs_nifty_{n}d_pp"] = (_r(sec[n] - nif[n], 2)
                                           if sec[n] is not None and nif[n] is not None else None)
    return {k: v for k, v in out.items() if v is not None}, gaps


# ======================================================= manager context (phase 1)
def track_record(closed):
    """Summary of recently closed paper trades, with the sample size stated plainly."""
    rets = [p["pnl"] / p["entry_cash"] * 100 for p in closed]
    wins, losses = [r for r in rets if r > 0], [r for r in rets if r <= 0]
    from journal import sample_note
    return {"closed_trades": len(rets),
            "win_rate_pct": _r(len(wins) / len(rets) * 100, 0) if rets else None,
            "avg_win_pct": _r(np.mean(wins), 2) if wins else None,
            "avg_loss_pct": _r(np.mean(losses), 2) if losses else None,
            "sample_size_warning": sample_note(len(rets))}


# ============================================================ phase 2: exchange
def trading_days_between(a: date, b: date) -> int:
    """Weekday sessions from a (inclusive) to b (exclusive), less NSE_HOLIDAYS.
    0 means b is a itself; 1 means b is the next session."""
    return int(np.busday_count(a, b, holidays=config.NSE_HOLIDAYS))


def earnings_info(res, day: date):
    """Next results date from the NSE calendar result, for the risk officer and manager.
    status: known | none_announced | unknown (calendar unavailable)."""
    if res is None or res.get("status") == "unavailable":
        return {"status": "unknown", "reason": (res or {}).get("error") or "calendar not fetched"}
    rows = res.get("data") or []
    upcoming = sorted(r["date"] for r in rows if r.get("date") and r["date"] >= day.isoformat())
    past = sorted(r["date"] for r in rows if r.get("date") and r["date"] < day.isoformat())
    if not upcoming:
        return {"status": "none_announced", "last_results_meeting": past[-1] if past else None}
    nxt = date.fromisoformat(upcoming[0])
    return {"status": "known", "date": nxt.isoformat(), "trading_days_away": trading_days_between(day, nxt),
            "last_results_meeting": past[-1] if past else None}


def insider_summary(rows, day: date, window=None):
    """Net value bought or sold by promoters and by other insiders, transaction
    counts and the largest single trade, over trades disclosed in the window."""
    window = window or config.INSIDER_DAYS
    start = (day - timedelta(days=window)).isoformat()
    groups = {"promoters": {"buy": 0.0, "sell": 0.0, "n": 0}, "other_insiders": {"buy": 0.0, "sell": 0.0, "n": 0}}
    largest, non_trade, undated = None, 0, 0
    for r in rows or []:
        if not r.get("disclosed"):
            undated += 1
            continue
        if not (start <= r["disclosed"] <= day.isoformat()):
            continue
        kind = (r.get("type") or "").strip().lower()
        if kind not in ("buy", "sell") or r.get("value") is None:
            non_trade += 1                      # pledges, revocations, invocations
            continue
        g = groups["promoters" if "promoter" in (r.get("category") or "").lower() else "other_insiders"]
        g[kind] += r["value"]
        g["n"] += 1
        if largest is None or r["value"] > largest["value"]:
            largest = r
    out = {"window_days": window}
    for name, g in groups.items():
        out[name] = {"bought_cr": _r(g["buy"] / 1e7, 2), "sold_cr": _r(g["sell"] / 1e7, 2),
                     "net_value_cr": _r((g["buy"] - g["sell"]) / 1e7, 2), "transactions": g["n"]}
    out["largest_trade"] = None if largest is None else {
        "disclosed": largest["disclosed"], "name": largest.get("name"), "category": largest.get("category"),
        "type": largest.get("type"), "mode": largest.get("mode"), "value_cr": _r(largest["value"] / 1e7, 2)}
    if non_trade:
        out["non_trade_disclosures"] = non_trade
    if undated:
        out["undated_rows_ignored"] = undated
    return out


def flow_summary(rows, day: date, windows=None):
    """Latest FII/DII day plus rolling net totals built from our own stored history.
    Says plainly when fewer sessions are stored than the window asks for."""
    windows = windows or config.FLOW_WINDOWS
    by_date = {}
    for r in rows or []:
        if r["date"] <= day.isoformat() and r.get("net_cr") is not None:
            by_date.setdefault(r["date"], {})[r["category"]] = r["net_cr"]
    dates = sorted(d for d, v in by_date.items() if "FII" in v and "DII" in v)
    if not dates:
        return None, ["FII/DII flows: no stored sessions on or before this date"]
    latest = dates[-1]
    out = {"latest_session": latest, "fii_net_cr_latest": _r(by_date[latest]["FII"], 0),
           "dii_net_cr_latest": _r(by_date[latest]["DII"], 0)}
    notes = []
    for w in windows:
        span = dates[-w:]
        out[f"fii_net_cr_{w}d"] = _r(sum(by_date[d]["FII"] for d in span), 0)
        out[f"dii_net_cr_{w}d"] = _r(sum(by_date[d]["DII"] for d in span), 0)
        out[f"sessions_in_{w}d_total"] = len(span)
        if len(span) < w:
            notes.append(f"only {len(span)} of {w} sessions stored so far, so the {w}-day totals are partial")
    age = (day - date.fromisoformat(latest)).days
    if age > 4:
        notes.append(f"latest stored session is {latest}, {age} days before {day}")
    if notes:
        out["history_note"] = "; ".join(notes)
    return out, []
