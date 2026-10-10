"""NSE exchange data behind one small interface: one function per dataset, each
returning the standard fetch result with plain dicts and lists inside.

Transport: `nsepython` (nsefetch for JSON APIs, get_bhavcopy for the archive).
Library choice, tested from a cloud host on 2026-10-10:
  * nsefin 0.1.5 was tried first and rejected: importing it makes a network call
    (it builds a client at import time, which would break --offline); its FII/DII
    path is missing a slash and always fails; its bhavcopy helper prints errors
    and returns an empty table (indistinguishable from "no data") and labels the
    number of trades as volume; and none of its methods filter by symbol.
  * nsepython 2.101 works for every dataset below and raises on failure.
    Note: it talks to NSE through curl_cffi impersonating Chrome's TLS
    fingerprint, because NSE's bot manager blocks plain clients by fingerprint.
To change library, rewrite _nse_json() and _nse_bhavcopy() only.
nsefetch re-warms its session and retries once by itself on 401/403/404/429/503;
the standard single retry in fetch.get_dataset covers timeouts and bad bodies.

Insider trading: NSE's /api/corporates-pit returned zero rows for every
date-filtered query on 2026-10-10, even market-wide over three months, which
cannot be true. So each run first asks a market-wide canary window; if that is
empty the dataset is reported UNAVAILABLE rather than "no insider trades".
"""
import time
from datetime import date, datetime, timedelta
from urllib.parse import quote

import numpy as np

import config
from data import seed_for
from fetch import get_dataset

_last_call = [0.0]


def _pause():
    wait = config.EXCHANGE_PAUSE_SECONDS - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def _nse_json(url):
    from nsepython import nsefetch     # imported lazily: never loaded in --offline runs
    _pause()
    return nsefetch(url)


def _nse_bhavcopy(day: date):
    from nsepython import get_bhavcopy
    _pause()
    return get_bhavcopy(day.strftime("%d-%m-%Y"))


def _ddmmyyyy(d: date) -> str:
    return d.strftime("%d-%m-%Y")


def _parse_nse_dt(s):
    s = (s or "").strip()
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


# ============================================================ announcements
def announcements(ctx, symbol):
    """Corporate announcements filed in the last ANNOUNCEMENT_DAYS days, oldest first."""
    start = ctx.day - timedelta(days=config.ANNOUNCEMENT_DAYS)

    def live():
        url = (f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={quote(symbol)}"
               f"&from_date={_ddmmyyyy(start)}&to_date={_ddmmyyyy(ctx.day)}")
        rows = _nse_json(url)
        if not isinstance(rows, list):
            raise ValueError(f"unexpected reply type {type(rows).__name__}")
        return clean_announcements(rows, symbol, ctx.day)
    return get_dataset(ctx, "announcements", symbol, live, lambda: fake_announcements(symbol, ctx.day))


def clean_announcements(rows, symbol, day):
    cutoff = datetime.combine(day, datetime.max.time())
    out = []
    for r in rows:
        if r.get("symbol") and r["symbol"] != symbol:
            continue
        when = _parse_nse_dt(r.get("an_dt") or r.get("sort_date"))
        if when is None or when > cutoff:
            continue
        out.append({"date": when.strftime("%Y-%m-%d %H:%M"), "subject": (r.get("desc") or "").strip(),
                    "description": (r.get("attchmntText") or "").strip()[:300],
                    "link": r.get("attchmntFile") or ""})
    return sorted(out, key=lambda x: x["date"])


_ANN = [("Copy of Newspaper Publication", "has informed the Exchange about Copy of Newspaper Publication"),
        ("Trading Window-XBRL", "has informed the Exchange about closure of trading window"),
        ("Analysts/Institutional Investor Meet/Con. Call Updates", "has informed the Exchange about schedule of analyst meet"),
        ("Press Release", "has informed the Exchange about a large order win"),
        ("Updates", "has informed the Exchange about receipt of a tax demand order"),
        ("Investor Presentation", "has informed the Exchange about Investor Presentation"),
        ("Analysts/Institutional Investor Meet/Con. Call Updates", "has informed the Exchange about Transcript of earnings call")]


def fake_announcements(symbol, day):
    name = config.COMPANY_NAMES.get(symbol, symbol)
    out = []
    for back in range(config.ANNOUNCEMENT_DAYS + 1):
        d = day - timedelta(days=back)
        rng = np.random.default_rng(seed_for(symbol, d.isoformat(), "ann"))
        if rng.random() < 0.12:
            subj, text = _ANN[int(rng.integers(0, len(_ANN)))]
            out.append({"date": f"{d.isoformat()} 18:{int(rng.integers(10, 59))}", "subject": subj,
                        "description": f"{name} {text}",
                        "link": f"offline://nse/{symbol}/{d.isoformat()}/{subj.split()[0].lower()}.pdf"})
    return sorted(out, key=lambda x: x["date"])


# ========================================================= results calendar
def results_calendar(ctx, symbol):
    """Board meetings for financial results from NSE's event calendar.
    The calendar is published for today; a past date cannot be reconstructed
    without lookahead, so --asof runs leave it out."""
    def live():
        rows = _nse_json(f"https://www.nseindia.com/api/event-calendar?index=equities&symbol={quote(symbol)}")
        if not isinstance(rows, list):
            raise ValueError(f"unexpected reply type {type(rows).__name__}")
        out = []
        for r in rows:
            if r.get("symbol") not in (None, symbol) or "result" not in (r.get("purpose") or "").lower():
                continue
            d = _parse_nse_dt(r.get("date"))
            if d is not None:
                out.append({"date": d.date().isoformat(), "purpose": r.get("purpose"),
                            "description": (r.get("bm_desc") or "")[:200]})
        return sorted(out, key=lambda x: x["date"])
    return get_dataset(ctx, "results_calendar", symbol, live, lambda: fake_results_calendar(symbol, ctx.day),
                       asof_supported=False, asof_reason="NSE serves only today's calendar")


def fake_results_calendar(symbol, day):
    """Quarterly results 12-40 days after quarter end, announced 7 days ahead."""
    rng = np.random.default_rng(seed_for(symbol, "results"))
    offset = int(rng.integers(12, 41))
    out = []
    for y in range(day.year - 1, day.year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            meet = date(y, m, d) + timedelta(days=offset)
            while meet.weekday() > 4:
                meet += timedelta(days=1)
            if meet - timedelta(days=7) <= day:
                out.append({"date": meet.isoformat(), "purpose": "Financial Results",
                            "description": f"To consider financial results for the quarter ended {y}-{m:02d}-{d}"})
    return sorted(out, key=lambda x: x["date"])[-4:]


# ============================================================ insider trades
def insider_canary(ctx):
    """Market-wide insider filings over a short window. NSE companies file insider
    trades every day, so an empty answer means the endpoint is broken."""
    start = ctx.day - timedelta(days=config.INSIDER_CANARY_DAYS)

    def live():
        d = _nse_json(f"https://www.nseindia.com/api/corporates-pit?index=equities"
                      f"&from_date={_ddmmyyyy(start)}&to_date={_ddmmyyyy(ctx.day)}")
        rows = d.get("data", []) if isinstance(d, dict) else d
        if not rows:
            raise RuntimeError(f"market-wide insider query for {config.INSIDER_CANARY_DAYS} days returned "
                               "0 rows; the endpoint is not returning data")
        return {"rows": len(rows)}
    return get_dataset(ctx, "insider_canary", "_MARKET", live, lambda: {"rows": 50})


def insider_trades(ctx, symbol, canary=None):
    """Insider and promoter trades disclosed in the last INSIDER_DAYS days."""
    canary = canary or insider_canary(ctx)
    start = ctx.day - timedelta(days=config.INSIDER_DAYS)

    def live():
        if canary.get("status") != "ok":
            raise RuntimeError(f"insider endpoint failed its health check ({canary.get('error')})")
        d = _nse_json(f"https://www.nseindia.com/api/corporates-pit?index=equities&symbol={quote(symbol)}"
                      f"&from_date={_ddmmyyyy(start)}&to_date={_ddmmyyyy(ctx.day)}")
        rows = d.get("data", []) if isinstance(d, dict) else d
        return clean_insider(rows, symbol)
    return get_dataset(ctx, "insider", symbol, live, lambda: fake_insider(symbol, ctx.day),
                       cache_key_extra=canary.get("status", ""))


def clean_insider(rows, symbol):
    out = []
    for r in rows or []:
        if r.get("symbol") not in (None, symbol):
            continue
        disclosed = _parse_nse_dt(r.get("date")) or _parse_nse_dt(r.get("intimDt"))
        out.append({"disclosed": disclosed.date().isoformat() if disclosed else None,
                    "name": r.get("acqName"), "category": r.get("personCategory") or "",
                    "type": r.get("tdpTransactionType") or "", "mode": r.get("acqMode") or "",
                    "value": _num(r.get("secVal")), "quantity": _num(r.get("secAcq"))})
    return out


def fake_insider(symbol, day):
    rng = np.random.default_rng(seed_for(symbol, day.strftime("%Y-%m"), "insider"))
    out = []
    for _ in range(int(rng.integers(0, 6))):
        back = int(rng.integers(1, config.INSIDER_DAYS))
        cat = ["Promoters", "Promoter Group", "Director", "Employees/Designated Employees"][int(rng.integers(0, 4))]
        typ = "Buy" if rng.random() < 0.45 else "Sell"
        val = float(rng.lognormal(16, 1.5))
        out.append({"disclosed": (day - timedelta(days=back)).isoformat(), "name": f"Person {int(rng.integers(1, 9))}",
                    "category": cat, "type": typ, "mode": "Market Sale" if typ == "Sell" else "Market Purchase",
                    "value": round(val), "quantity": round(val / 1000)})
    return out


# ================================================================ FII / DII
def fii_dii(ctx):
    """Fetch the latest FII/DII cash-market figures and store them. The endpoint
    serves only the latest session, so history accumulates in SQLite."""
    def live():
        rows = _nse_json("https://www.nseindia.com/api/fiidiiTradeReact")
        if not isinstance(rows, list) or not rows:
            raise ValueError("no FII/DII rows")
        return clean_fii_dii(rows)
    res = get_dataset(ctx, "fii_dii", "_MARKET", live, lambda: fake_fii_dii(ctx.day),
                      asof_supported=False, asof_reason="NSE serves only the latest session")
    if res["status"] == "ok":
        store_fii_dii(ctx.con, res["data"], "offline-fake" if ctx.offline else "nse")
    return res


def clean_fii_dii(rows):
    out = []
    for r in rows:
        d = _parse_nse_dt(r.get("date"))
        cat = (r.get("category") or "").upper()
        cat = "FII" if cat.startswith("FII") or cat.startswith("FPI") else ("DII" if cat.startswith("DII") else None)
        if d is None or cat is None:
            continue
        out.append({"date": d.date().isoformat(), "category": cat, "buy_cr": _num(r.get("buyValue")),
                    "sell_cr": _num(r.get("sellValue")), "net_cr": _num(r.get("netValue"))})
    if not out:
        raise ValueError("FII/DII rows had no usable date/category")
    return out


def store_fii_dii(con, rows, source):
    for r in rows:
        con.execute("INSERT OR IGNORE INTO fii_dii (date, category, buy_cr, sell_cr, net_cr, source)"
                    " VALUES (?,?,?,?,?,?)", (r["date"], r["category"], r["buy_cr"], r["sell_cr"], r["net_cr"], source))
    con.commit()


def stored_flows(con, upto_day: str):
    return [dict(r) for r in con.execute("SELECT * FROM fii_dii WHERE date <= ? ORDER BY date", (upto_day,))]


def fake_fii_dii(day):
    rng = np.random.default_rng(seed_for(day.isoformat(), "flows"))
    fii, dii = float(rng.normal(-500, 2500)), float(rng.normal(800, 2000))
    return [{"date": day.isoformat(), "category": "FII", "buy_cr": 12000.0, "sell_cr": round(12000 - fii, 2),
             "net_cr": round(fii, 2)},
            {"date": day.isoformat(), "category": "DII", "buy_cr": 11000.0, "sell_cr": round(11000 - dii, 2),
             "net_cr": round(dii, 2)}]

