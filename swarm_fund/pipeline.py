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
    held = journal.held_symbols(con)
    review = indicators.shortlist(scores, held)
    for sym in sorted(held - set(inds)):
        ctx.note(f"held stock {sym} has no usable data today; it cannot be reviewed")
    print("Screen top 10: " + ", ".join(f"{s} {scores[s]:+.2f}" for s in
                                        sorted(scores, key=lambda s: -scores[s])[:10]))
    print(f"Reviewing: {', '.join(review)}")

    # 7. swarm
    m = journal.mark(con, prices, T)
    results = []
    for sym in review:
        sector = config.UNIVERSE[sym]
        is_held = sym in held
        ds = gather(ctx, sym)
        briefs = build_briefs(ctx, sym, sector, inds[sym], scores[sym], market, ds, prices, benchmarks)
        res = agents.run_swarm(sym, sector, is_held, briefs, llm,
                               manager_extra_fn=lambda s=sym, d=ds: manager_extra(ctx, s, prices, d))
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
    verdicts = risk.review(decisions, inds, book, earnings={r["symbol"]: r["datasets"].get("earnings_info")
                                                            for r in results})

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
def gather(ctx, sym):
    """Fetch every per-stock dataset. Only called for shortlisted and held stocks."""
    return {
        "fundamentals": data.fundamentals(ctx, sym),
        "quarterly": data.quarterly(ctx, sym),
        "headlines": data.headlines(ctx, sym),
    }


def build_briefs(ctx, sym, sector, ind, screen, market, ds, prices, benchmarks):
    T = ctx.day_str
    df = prices[sym]

    # technical: indicators + the price series itself
    tech_extra = {"daily_bars": features.compact_bars(df), "weekly_closes": features.weekly_closes(df)}

    # fundamental: snapshot + quarterly trend
    fund_extra, fund_gaps = {}, []
    q = ds["quarterly"]
    if q["status"] == "ok":
        table, qgaps = features.quarterly_table(q["data"], ctx.day, ctx.is_asof)
        fund_gaps += qgaps
        if table:
            fund_extra["quarterly"] = table
    else:
        fund_gaps += briefing.gaps(("quarterly results", q)) or ["quarterly results: Yahoo returned none"]

    # macro: market + sector context
    sector_ctx, sgaps = features.sector_context(sym, sector, prices, benchmarks, prices[config.BENCHMARK])
    macro_extra = {"sector_context": sector_ctx}

    briefs = {
        "technical": briefing.technical(sym, T, ind, screen, tech_extra),
        "fundamental": briefing.fundamental(sym, T, ds["fundamentals"], fund_extra),
        "news": briefing.news(sym, T, ds["headlines"]),
        "macro": briefing.macro(sym, sector, T, market, macro_extra),
    }
    briefs["fundamental"]["data"]["data_gaps"] += fund_gaps
    briefs["macro"]["data"]["data_gaps"] += sgaps
    return briefs


def manager_extra(ctx, sym, prices, ds):
    """Portfolio, this stock's history with the desk, and the desk's track record."""
    con, T = ctx.con, ctx.day_str
    out = {"portfolio": journal.portfolio_context(con, prices, T),
           "your_prior_calls_on_this_stock": journal.prior_calls(con, prices, T, sym),
           "recent_track_record": features.track_record(journal.recent_closed(con))}
    pos = [p for p in out["portfolio"]["open_positions"] if p["symbol"] == sym]
    if pos:
        out["this_position"] = pos[0]
    return out


def source_links(ds):
    links = []
    for key in ("announcements", "headlines"):
        res = ds.get(key)
        if res and res.get("status") == "ok":
            for item in res["data"][:2]:
                if item.get("link"):
                    links.append(item["link"])
    return links
