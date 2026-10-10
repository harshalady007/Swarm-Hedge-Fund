"""`python main.py doctor`: call every live data source once for one stock and print
source, OK/FAIL, rows returned and time taken. Uses a throwaway in-memory database,
so nothing is cached and the journal is untouched."""
import time
from datetime import datetime

import config
import data
import db
from fetch import Ctx


def checks(ctx, sym):
    """(name, callable) pairs; each callable returns a fetch result dict or a row count."""
    yield "yahoo prices", lambda: _prices(ctx, [sym, config.BENCHMARK, config.VIX])
    yield "yahoo fundamentals", lambda: data.fundamentals(ctx, sym)
    yield "google news rss", lambda: data.headlines(ctx, sym)
    yield "yahoo quarterly results", lambda: data.quarterly(ctx, sym)
    yield "yahoo sector indices", lambda: _sector_indices(ctx)
    yield "telegram (config only)", _telegram
    yield "anthropic key (config)", _anthropic


def _prices(ctx, tickers):
    p = data.load_prices(ctx, tickers)
    missing = [t for t in tickers if t not in p]
    rows = sum(len(df) for df in p.values())
    if missing:
        return {"status": "unavailable", "data": None, "error": f"missing {missing}", "rows": rows}
    return {"status": "ok", "rows": rows, "detail": ", ".join(f"{t}:{len(df)}" for t, df in p.items())}


def _sector_indices(ctx):
    import features
    p = data.load_prices(ctx, config.sector_index_tickers())
    picks = features.pick_sector_benchmarks(p, ctx.day)
    usable = [t for t, _ in picks.values() if t]
    detail = "; ".join(f"{s}: {t or 'basket fallback'}" for s, (t, _) in picks.items())
    return {"status": "ok" if usable else "unavailable", "rows": len(usable),
            "detail": f"{len(usable)} usable index(es). {detail}",
            "error": None if usable else "no Yahoo sector index has usable history; all sectors use baskets"}


def _telegram():
    ok = bool(config.TELEGRAM_TOKEN and config.TELEGRAM_CHAT_ID)
    return {"status": "set" if ok else "not set", "rows": 0,
            "error": None if ok else "TELEGRAM_TOKEN / TELEGRAM_CHAT_ID not set (console only)",
            "detail": "configured (no message sent)" if ok else ""}


def _anthropic():
    ok = bool(config.ANTHROPIC_API_KEY)
    return {"status": "set" if ok else "not set", "rows": 0,
            "error": None if ok else "ANTHROPIC_API_KEY not set: runs will use rule-based agents",
            "detail": f"analysts {config.ANALYST_MODEL}, manager {config.MANAGER_MODEL}" if ok else ""}


def _rows(res):
    if "rows" in res:
        return res["rows"]
    d = res.get("data")
    if isinstance(d, (list, dict)):
        return len(d)
    return 0


def run(sym="RELIANCE"):
    ctx = Ctx(con=db.connect(":memory:"), day=datetime.now(data.IST).date(), offline=False, is_asof=False)
    print(f"doctor: checking live sources for {sym} on {ctx.day_str}\n")
    print(f"{'source':<26}{'result':<8}{'rows':>7}{'secs':>8}  detail")
    print("-" * 100)
    failures = 0
    for name, fn in checks(ctx, sym):
        t = time.time()
        try:
            res = fn()
        except Exception as e:  # noqa: BLE001
            res = {"status": "unavailable", "error": f"{type(e).__name__}: {e}"}
        secs = time.time() - t
        status = res.get("status")
        label = {"ok": "OK", "empty": "EMPTY", "set": "SET", "not set": "NOT SET"}.get(status, "FAIL")
        failures += label == "FAIL"
        detail = res.get("error") or res.get("detail") or ""
        print(f"{name:<26}{label:<8}{_rows(res):>7}{secs:>8.1f}  {str(detail)[:120]}")
    print("\nEMPTY means the source answered with nothing to report; FAIL means it could not be reached "
          "or returned something unusable.")
    return 1 if failures else 0
