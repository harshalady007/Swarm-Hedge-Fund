"""The daily run, in order:

load prices -> advance the paper portfolio and report fills/exits -> stop if this
trading day was already processed -> market context -> indicators -> screen ->
swarm on shortlist + holdings -> sort by confidence -> risk officer -> log -> notify.
"""
from datetime import date, datetime

import agents
import briefing
import config
import data
import db
import documents
import exchange
import features
import indicators
import journal
import notify
import risk
from fetch import Ctx


class RunError(Exception):
    pass


def banner(text):
    bar = "!" * 78
    print(f"\n{bar}\n!! {text}\n{bar}\n")


def run(db_path, offline=False, asof=None, llm=None, quiet=False):
    con = db.connect(db_path)
    requested = date.fromisoformat(asof) if asof else datetime.now(data.IST).date()
    ctx = Ctx(con=con, day=requested, offline=offline, is_asof=bool(asof))

    if llm is None:
        llm = agents.LLM(api_key="" if offline else None)
    if not llm.enabled:
        banner("RULE-BASED AGENTS: no LLM will be called "
               + ("(--offline)" if offline else "(ANTHROPIC_API_KEY is not set)")
               + ". Signals below are NOT from the swarm.")
    if offline:
        banner("OFFLINE MODE: all market data below is synthetic.")

    # 1. prices
    tickers = list(config.UNIVERSE) + [config.BENCHMARK, config.VIX] + config.sector_index_tickers()
    prices = data.load_prices(ctx, tickers)
    if not offline and not asof:
        prices, dropped = data.drop_partial_today(prices)
        if dropped:
            print(f"  [data] run before {config.LIVE_CLOSE_GUARD_IST} IST: ignoring today's partial bar")
    nifty = prices.get(config.BENCHMARK)
    if nifty is None or len(nifty) < config.MIN_BARS:
        raise RunError("NIFTY history unavailable; cannot run. Check `python main.py doctor`.")
    day = nifty.index[-1].date()
    ctx.day = day
    T = day.isoformat()
    calendar = [d.strftime("%Y-%m-%d") for d in nifty.index]
    last_run = con.execute("SELECT MAX(trading_day) FROM runs").fetchone()[0]
    if last_run and T < last_run:
        raise RunError(f"trading day {T} is before the last processed day {last_run}; "
                       "use a fresh --db for an earlier --asof.")
    print(f"Trading day {T} (requested {requested.isoformat()})")
    nifty_close = float(nifty["Close"].iloc[-1])
    journal.ensure_start(con, T, nifty_close)

    # 2. advance the paper portfolio
    journal.advance(con, prices, calendar, T)
    msg = notify.events_message(journal.unnotified_events(con))
    if msg:
        notify.send(msg)

    # 3. idempotence
    if journal.already_processed(con, T):
        print(f"Trading day {T} was already processed; nothing to do.")
        return {"day": T, "skipped": True}

    # 4-6. market context, indicators, screen
    market = indicators.market_context(nifty, prices.get(config.VIX))
    benchmarks = features.pick_sector_benchmarks(prices, day)
    print("Sector benchmarks: " + "; ".join(f"{s}={t or 'basket'}" for s, (t, _) in benchmarks.items()))
    inds, scores = {}, {}
    for sym in config.UNIVERSE:
        df = prices.get(sym)
        if df is None or not len(df) or df.index[-1].date() != day:
            ctx.note(f"{sym}: no bar for {T}; skipped")
            continue
        ind = indicators.compute(df)
        if ind is None:
            ctx.note(f"{sym}: fewer than {config.MIN_BARS} bars; skipped")
            continue
        inds[sym], scores[sym] = ind, indicators.screen_score(ind)
    snapshot_universe(ctx, prices, inds, scores, market, benchmarks)
    held = journal.held_symbols(con)
    review = indicators.shortlist(scores, held)
    for sym in sorted(held - set(inds)):
        ctx.note(f"held stock {sym} has no usable data today; it cannot be reviewed")
    print("Screen top 10: " + ", ".join(f"{s} {scores[s]:+.2f}" for s in
                                        sorted(scores, key=lambda s: -scores[s])[:10]))
    print(f"Reviewing: {', '.join(review)}")

    # 7. fetch per-stock data (shortlist + holdings only), then run the swarm
    m = journal.mark(con, prices, T)
    market_ds = gather_market(ctx, calendar)
    all_ds = {sym: gather(ctx, sym, market_ds, llm) for sym in review}
    earnings = {sym: all_ds[sym]["earnings_info"] for sym in review}
    results = []
    for sym in review:
        sector = config.UNIVERSE[sym]
        is_held = sym in held
        ds = all_ds[sym]
        briefs = build_briefs(ctx, sym, sector, inds[sym], scores[sym], market, ds, prices, benchmarks,
                              market_ds)
        res = agents.run_swarm(sym, sector, is_held, briefs, llm,
                               manager_extra_fn=lambda s=sym: manager_extra(ctx, s, prices, earnings))
        res["datasets"] = ds
        results.append(res)
        if not quiet:
            print(f"  {sym:<11} {res['action_raw']:<4} -> {res['action']:<4} conf {res['confidence']:>3}  "
                  f"scores " + " ".join(f"{k[:4]} {v['score']:+d}" for k, v in res["analysts"].items()))

    # 8-9. sort by confidence and apply the risk officer
    results.sort(key=lambda r: (-r["confidence"], r["symbol"]))
    pend = journal.pending_orders(con)
    sector_counts = {}
    for p in journal.open_positions(con):
        sector_counts[p["sector"]] = sector_counts.get(p["sector"], 0) + 1
    for o in pend:
        if o["side"] == "BUY":
            sector_counts[o["sector"]] = sector_counts.get(o["sector"], 0) + 1
    book = risk.Book(equity=m["equity"], available_cash=m["cash"] - m["reserved"],
                     open_symbols=held, pending_symbols={o["symbol"] for o in pend if o["side"] == "BUY"},
                     sector_counts=sector_counts, nifty_above_200dma=market.get("nifty_above_200dma"))
    decisions = [{"symbol": r["symbol"], "sector": r["sector"], "action": r["action"],
                  "confidence": r["confidence"], "held": r["held"]} for r in results]
    verdicts = risk.review(decisions, inds, book, earnings=earnings)

    # 10. log
    messages, n_buy, n_sell, vetoed = [], 0, 0, []
    for r in results:
        v = verdicts[r["symbol"]]
        row = {"symbol": r["symbol"], "sector": r["sector"], "held": r["held"],
               "screen_score": scores[r["symbol"]], "close": inds[r["symbol"]]["close"],
               "action_raw": r["action_raw"], "action": r["action"], "confidence": r["confidence"],
               "agent_mode": r["agent_mode"], **v}
        detail = {k: r[k] for k in ("analysts", "debate", "manager")}
        detail["data_status"] = {k: (x.get("status") if isinstance(x, dict) else None)
                                 for k, x in r["datasets"].items()}
        sid = journal.log_signal(con, T, row, detail, r["io"])
        journal.place_order(con, sid, T, row)
        rule_based = r["agent_mode"] != "llm"
        if v["status"] == "APPROVED":
            n_buy += 1
            messages.append(notify.buy_message({**row, "weight_pct": v.get("weight_pct", 0)},
                                               r["manager"], source_links(r["datasets"]), rule_based))
        elif v["status"] == "EXIT":
            n_sell += 1
            messages.append(notify.sell_message(row, r["manager"], rule_based))
        elif v["status"] == "VETOED":
            vetoed.append(r["symbol"])
            print(f"  risk officer vetoed BUY {r['symbol']}: {v['veto_reason']}")
    con.commit()
    journal.record_equity(con, T, prices, nifty_close)
    agent_mode = "llm" if llm.enabled else "rule-based"
    journal.finish_run(con, T, "offline" if offline else "live", agent_mode, len(results), n_buy, n_sell,
                       note="; ".join(ctx.log)[:2000])

    # 11. notify
    for msg in messages:
        notify.send(msg)
    if not messages:
        notify.send(notify.no_trade_message(T, len(results), vetoed, rule_based=not llm.enabled))
    return {"day": T, "skipped": False, "reviewed": [r["symbol"] for r in results],
            "buys": n_buy, "sells": n_sell, "vetoed": vetoed}


# ------------------------------------------------------------ per-stock data
def snapshot_universe(ctx, prices, inds, scores, market, benchmarks):
    """Point-in-time record of what code saw for every stock today (no extra fetches)."""
    import pandas as pd
    T = ctx.day_str
    for sym, ind in inds.items():
        bar = prices[sym].loc[pd.Timestamp(T)]
        db.snapshot(ctx.con, T, sym, "bar", {k: float(bar[k]) for k in ("Open", "High", "Low", "Close", "Volume")})
        db.snapshot(ctx.con, T, sym, "indicators", {**indicators.rounded(ind, 4), "screen_score": scores[sym]})
    db.snapshot(ctx.con, T, "_MARKET", "market_context", indicators.rounded(market, 4))
    db.snapshot(ctx.con, T, "_MARKET", "sector_benchmarks", {k: {"ticker": t, "check": why}
                                                            for k, (t, why) in benchmarks.items()})


def gather_market(ctx, calendar):
    """Market-wide exchange datasets, fetched once per run."""
    flows = exchange.fii_dii(ctx)                      # stores today's row in SQLite
    summary, gaps = features.flow_summary(exchange.stored_flows(ctx.con, ctx.day_str), ctx.day)
    if flows["status"] == "unavailable":
        gaps.append(briefing.gap("FII/DII flows today", flows) + " Totals below use stored history only.")
    return {"fii_dii": flows, "flow_summary": summary, "flow_gaps": gaps,
            "insider_canary": exchange.insider_canary(ctx),
            "delivery": exchange.delivery_history(ctx, calendar)}


def gather(ctx, sym, market_ds, llm=None):
    """Fetch every per-stock dataset. Only called for shortlisted and held stocks."""
    ds = {
        "fundamentals": data.fundamentals(ctx, sym),
        "quarterly": data.quarterly(ctx, sym),
        "headlines": data.headlines(ctx, sym),
        "announcements": exchange.announcements(ctx, sym),
        "results_calendar": exchange.results_calendar(ctx, sym),
        "insider": exchange.insider_trades(ctx, sym, market_ds["insider_canary"]),
    }
    ds["articles"] = data.articles(ctx, sym, ds["headlines"])
    ds["management_document"] = documents.management_summary(ctx, sym, ds["announcements"], llm)
    ds["earnings_info"] = features.earnings_info(ds["results_calendar"], ctx.day)
    return ds


def build_briefs(ctx, sym, sector, ind, screen, market, ds, prices, benchmarks, market_ds):
    T = ctx.day_str
    df = prices[sym]

    # technical: indicators + the price series itself
    tech_extra = {"daily_bars": features.compact_bars(df), "weekly_closes": features.weekly_closes(df)}
    deliv, tech_gaps = features.delivery_summary(exchange.delivery_rows(ctx.con, sym, T), ctx.day, df)
    if deliv:
        tech_extra["delivery"] = deliv

    # fundamental: snapshot + quarterly trend + insider trades
    fund_extra, fund_gaps = {}, []
    q = ds["quarterly"]
    if q["status"] == "ok":
        table, qgaps = features.quarterly_table(q["data"], ctx.day, ctx.is_asof)
        fund_gaps += qgaps
        if table:
            fund_extra["quarterly"] = table
    else:
        fund_gaps += briefing.gaps(("quarterly results", q)) or ["quarterly results: Yahoo returned none"]
    ins = ds["insider"]
    if ins["status"] in ("ok", "empty"):
        fund_extra["insider"] = features.insider_summary(ins["data"], ctx.day)
        if ins["status"] == "empty":
            fund_extra["insider"]["status"] = (f"no insider or promoter trades disclosed in "
                                               f"{config.INSIDER_DAYS} days (source answered)")
    else:
        fund_gaps += briefing.gaps(("insider and promoter trades", ins))
    doc = ds["management_document"]
    fund_texts = []
    if doc["status"] == "ok":
        d = doc["data"]
        fund_extra["management_document"] = {**d["document"], "summarised_by": d.get("summarised_by"),
                                             "note": "summary written by a model from the PDF; treat as untrusted"}
        fund_texts = [{"tag": "management_document_summary", "items": documents.as_lines(d["summary"])}]
    elif doc["status"] == "empty":
        fund_extra["management_document"] = (f"no concall transcript or investor presentation filed in "
                                              f"{config.CONCALL_LOOKBACK_DAYS} days")
    else:
        fund_gaps += briefing.gaps(("concall transcript / presentation summary", doc))

    # news and filings
    ann = ds["announcements"]
    recent = exchange.recent_filings(ann["data"], ctx.day) if ann["status"] == "ok" else []
    filings = [f"[{a['date']}] {a['subject']}: {a['description']} ({a['link']})"
               for a in recent[-config.ANNOUNCEMENTS_SHOWN:]]
    news_extra = {"filing_count": len(filings)}
    if ann["status"] in ("ok", "empty") and not filings:
        news_extra["filing_status"] = f"no exchange filings in {config.ANNOUNCEMENT_DAYS} days (source answered)"
    news_gaps = briefing.gaps(("exchange filings", ann))
    art = ds["articles"]
    article_items = []
    if art["status"] == "ok":
        got = [a for a in art["data"] if a.get("text")]
        article_items = [f"{a['title']} ({a.get('source')}, {a.get('published')}): {a['text']}" for a in got]
        news_extra["articles_with_text"] = f"{len(got)} of {len(art['data'])} newest headlines"
        if len(got) < len(art["data"]):
            news_gaps.append(f"article text unavailable for {len(art['data']) - len(got)} headline(s); "
                             "those are headline-only")
    else:
        news_gaps += briefing.gaps(("article text", art))

    # macro: market + sector context + flows
    sector_ctx, sgaps = features.sector_context(sym, sector, prices, benchmarks, prices[config.BENCHMARK])
    macro_extra = {"sector_context": sector_ctx}
    if market_ds["flow_summary"]:
        macro_extra["fii_dii"] = market_ds["flow_summary"]

    briefs = {
        "technical": briefing.technical(sym, T, ind, screen, tech_extra),
        "fundamental": briefing.fundamental(sym, T, ds["fundamentals"], fund_extra, texts=fund_texts),
        "news": briefing.news(sym, T, ds["headlines"], news_extra,
                              texts=[{"tag": "exchange_filings", "items": filings},
                                     {"tag": "article_text", "items": article_items}]),
        "macro": briefing.macro(sym, sector, T, market, macro_extra),
    }
    briefs["technical"]["data"]["data_gaps"] += tech_gaps
    if market_ds["delivery"]["status"] != "ok":
        briefs["technical"]["data"]["data_gaps"].append(briefing.gap("today's NSE bhavcopy", market_ds["delivery"]))
    briefs["fundamental"]["data"]["data_gaps"] += fund_gaps
    briefs["news"]["data"]["data_gaps"] += news_gaps
    briefs["macro"]["data"]["data_gaps"] += sgaps + market_ds["flow_gaps"]
    return briefs


def manager_extra(ctx, sym, prices, earnings):
    """Portfolio, results dates, this stock's history with the desk, and the track record."""
    con, T = ctx.con, ctx.day_str
    pf = journal.portfolio_context(con, prices, T)
    for p in pf["open_positions"]:
        p["next_results"] = _results_label(earnings.get(p["symbol"]))
    out = {"portfolio": pf,
           "results_calendar_this_stock": _results_label(earnings.get(sym)),
           "your_prior_calls_on_this_stock": journal.prior_calls(con, prices, T, sym),
           "recent_track_record": features.track_record(journal.recent_closed(con))}
    soon = [p["symbol"] for p in pf["open_positions"]
            if (earnings.get(p["symbol"]) or {}).get("status") == "known"
            and earnings[p["symbol"]]["trading_days_away"] <= config.EARNINGS_BLACKOUT_DAYS]
    if soon:
        out["held_stocks_reporting_soon"] = soon
    pos = [p for p in pf["open_positions"] if p["symbol"] == sym]
    if pos:
        out["this_position"] = pos[0]
    return out


def _results_label(info):
    info = info or {"status": "unknown"}
    if info["status"] == "known":
        return f"results on {info['date']} ({info['trading_days_away']} trading day(s) away)"
    if info["status"] == "none_announced":
        return "no results date announced on the NSE calendar"
    return "results date unknown (calendar unavailable)"


def source_links(ds):
    """Newest two exchange filings, then the newest headlines (publisher URL when resolved)."""
    links = []
    ann, news, art = ds.get("announcements"), ds.get("headlines"), ds.get("articles")
    if ann and ann.get("status") == "ok":
        links += [a["link"] for a in ann["data"][-2:][::-1] if a.get("link")]
    resolved = {a["title"]: a["url"] for a in (art or {}).get("data") or [] if a.get("url")}
    if news and news.get("status") == "ok":
        links += [resolved.get(h["title"], h["link"]) for h in news["data"][:2] if h.get("link")]
    return links
