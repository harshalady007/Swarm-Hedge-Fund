"""`python main.py doctor`: call every live data source once for one stock and print
source, OK/FAIL, rows returned and time taken. Uses a throwaway in-memory database,
so nothing is cached and the journal is untouched."""
import json
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
    import exchange
    yield "nse announcements", lambda: exchange.announcements(ctx, sym)
    yield "nse results calendar", lambda: _calendar(ctx, sym)
    canary = {}
    yield "nse insider (market canary)", lambda: canary.setdefault("r", exchange.insider_canary(ctx))
    yield "nse insider (this stock)", lambda: exchange.insider_trades(ctx, sym, canary.get("r"))
    yield "nse fii/dii", lambda: exchange.fii_dii(ctx)
    yield "nse bhavcopy (delivery %)", lambda: _bhavcopy(ctx, sym)
    yield "nse transcript pdf + pypdf", lambda: _pdf(ctx, sym)
    yield "google news link decode", lambda: _decode(ctx, sym)
    yield "jina reader article", lambda: _jina(ctx, sym)
    yield "telegram (config only)", _telegram
    yield f"llm: {config.LLM_PROVIDER} (1 tiny call)", _llm


def _prices(ctx, tickers):
    p = data.load_prices(ctx, tickers)
    missing = [t for t in tickers if t not in p]
    rows = sum(len(df) for df in p.values())
    if missing:
        return {"status": "unavailable", "data": None, "error": f"missing {missing}", "rows": rows}
    return {"status": "ok", "rows": rows, "detail": ", ".join(f"{t}:{len(df)}" for t, df in p.items())}


def _bhavcopy(ctx, sym):
    """Latest published bhavcopy: try today, then step back up to five weekdays."""
    import exchange
    from datetime import timedelta
    d, tried = ctx.day, []
    for _ in range(6):
        if d.weekday() < 5:
            try:
                rows = exchange.parse_bhavcopy(exchange._nse_bhavcopy(d), d, list(config.UNIVERSE))
                mine = next((r for r in rows if r["symbol"] == sym), {})
                return {"status": "ok", "rows": len(rows),
                        "detail": f"{d}: {sym} delivery {mine.get('deliv_pct')}%" + (f" (tried {tried})" if tried else "")}
            except Exception as e:  # noqa: BLE001
                tried.append(f"{d}: {str(e)[:60]}")
        d -= timedelta(days=1)
    return {"status": "unavailable", "error": "; ".join(tried)[:200]}


def _pdf(ctx, sym):
    import documents
    import exchange
    ann = exchange.announcements(ctx, sym)
    if ann["status"] == "unavailable":
        return ann
    doc = documents.pick_document(ann["data"], ctx.day)
    if doc is None:
        return {"status": "empty", "rows": 0,
                "detail": f"no transcript/presentation filed in {config.CONCALL_LOOKBACK_DAYS} days"}
    text = documents.extract_pdf_text(documents._download(doc["link"]))
    return {"status": "ok" if len(text) > 500 else "unavailable", "rows": len(text),
            "detail": f"{doc['date']} {doc['subject']}: {len(text):,} characters extracted",
            "error": None if len(text) > 500 else "no text layer"}


def _first_headline(ctx, sym):
    res = data.headlines(ctx, sym)
    if res["status"] != "ok":
        raise RuntimeError(f"no headline to test with ({res['status']})")
    return res["data"][0]


def _decode(ctx, sym):
    url = data.resolve_google_news(_first_headline(ctx, sym)["link"])
    return {"status": "ok", "rows": 1, "detail": url[:110]}


def _jina(ctx, sym):
    url = data.resolve_google_news(_first_headline(ctx, sym)["link"])
    text = data.jina_text(url)
    return {"status": "ok", "rows": len(text.split()), "detail": f"{len(text.split())} words from {url[:70]}"}


def _calendar(ctx, sym):
    import exchange
    import features
    res = exchange.results_calendar(ctx, sym)
    if res["status"] != "unavailable":
        res["detail"] = str(features.earnings_info(res, ctx.day))
    return res


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


def _llm():
    """One small real request: proves the key, the model name and JSON replies work."""
    import agents
    llm = agents.make_llm(offline=False)
    if not llm.enabled:
        return {"status": "not set", "rows": 0,
                "error": f"{agents.key_name(llm.provider)} not set: runs will use rule-based agents"}
    schema = {"type": "object", "additionalProperties": False, "required": ["ok"],
              "properties": {"ok": {"type": "boolean"}}}
    models = dict.fromkeys([config.ANALYST_MODEL, config.MANAGER_MODEL])
    answers = []
    for model in models:
        obj, err = llm.json_call(model, 'Health check. Reply with the JSON object {"ok": true}.', "ping", schema,
                                 2000, "low")
        if obj is None:
            return {"status": "unavailable", "rows": len(answers), "error": f"{model}: {err}"}
        answers.append(f"{model} -> {json.dumps(obj)}")
    return {"status": "ok", "rows": len(answers), "detail": "; ".join(answers)}


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
    print(f"{"source":<30}{'result':<8}{'rows':>7}{'secs':>8}  detail")
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
        print(f"{name:<30}{label:<8}{_rows(res):>7}{secs:>8.1f}  {str(detail)[:120]}")
    print("\nEMPTY means the source answered with nothing to report; FAIL means it could not be reached "
          "or returned something unusable.")
    return 1 if failures else 0
