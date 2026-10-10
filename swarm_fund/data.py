"""Yahoo Finance prices and fundamentals, Google News headlines, and their offline fakes.

Nothing here is shown to an agent directly; briefing.py decides what each agent sees.
"""
import email.utils
import json
import zlib
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

import config
import db
from fetch import BROWSER_UA, get_dataset, http_get, with_retry

IST = timezone(timedelta(hours=5, minutes=30))


def seed_for(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode())


# ====================================================================== prices
def load_prices(ctx, tickers):
    """Return {ticker: DataFrame[Open, High, Low, Close, Volume]} ending on or before ctx.day.

    Live data is one bulk Yahoo download, cached in SQLite for the day.
    """
    if ctx.offline:
        out = {t: fake_prices(t) for t in tickers}
    else:
        out = _yahoo_prices(ctx, tickers)
    end = pd.Timestamp(ctx.day)
    out = {t: df[df.index <= end] for t, df in out.items() if df is not None and len(df)}
    if ctx.offline:
        for t in config.sector_index_tickers():
            if t in out and t not in FAKE_INDEX_WITH_HISTORY:
                out[t] = out[t].iloc[-1:]
    return out


def _yahoo_prices(ctx, tickers):
    key = f"prices|{ctx.day_str}|{int(ctx.is_asof)}|{','.join(sorted(tickers))}"
    cached = db.cache_get(ctx.con, key)
    if cached is not None:
        return {t: _frame_from_json(v) for t, v in cached.items()}
    import yfinance as yf
    start = ctx.day - timedelta(days=int(365 * config.HISTORY_YEARS) + 30)
    end = ctx.day + timedelta(days=1)

    def download():
        raw = yf.download([config.yahoo_ticker(t) for t in tickers], start=start.isoformat(),
                          end=end.isoformat(), interval="1d", auto_adjust=True, group_by="ticker",
                          progress=False, threads=True, timeout=config.HTTP_TIMEOUT * 2)
        if raw is None or raw.empty:
            raise RuntimeError("Yahoo returned no price data")
        return raw

    raw = with_retry(download)
    out = {}
    for t in tickers:
        yt = config.yahoo_ticker(t)
        try:
            df = raw[yt] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            ctx.note(f"prices {t}: not in Yahoo response")
            continue
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        if len(df) == 0:
            ctx.note(f"prices {t}: Yahoo returned no rows")
            continue
        out[t] = df
    db.cache_put(ctx.con, key, {t: _frame_to_json(df) for t, df in out.items()})
    return out



def _frame_to_json(df):
    return {"index": [d.strftime("%Y-%m-%d") for d in df.index],
            "cols": list(df.columns), "values": df.astype(float).round(4).values.tolist()}


def _frame_from_json(d):
    return pd.DataFrame(d["values"], index=pd.to_datetime(d["index"]), columns=d["cols"])


def drop_partial_today(prices, now_ist=None):
    """A live run before the close guard time must not treat today's bar as final."""
    now_ist = now_ist or datetime.now(IST)
    hh, mm = (int(x) for x in config.LIVE_CLOSE_GUARD_IST.split(":"))
    if (now_ist.hour, now_ist.minute) >= (hh, mm):
        return prices, False
    today = pd.Timestamp(now_ist.date())
    dropped = False
    for t, df in prices.items():
        if len(df) and df.index[-1] == today:
            prices[t] = df.iloc[:-1]
            dropped = True
    return prices, dropped


# Synthetic market used by --offline. Generated once per ticker over a fixed
# calendar, so any --asof date sees the same history up to that date.
_FAKE_START, _FAKE_END = "2022-01-03", "2028-12-29"


def _fake_market_returns():
    rng = np.random.default_rng(7)
    days = pd.bdate_range(_FAKE_START, _FAKE_END)
    t = np.arange(len(days))
    drift = 0.0004 + 0.0012 * np.sin(t / 90.0)        # alternating bull and bear spells
    return days, drift + rng.normal(0, 0.009, len(days))


# Offline sector indices mirror what Yahoo actually served on 2026-10-10: three
# with history, the rest a single quote row. Both code paths get exercised.
FAKE_INDEX_WITH_HISTORY = {"^CNXIT", "^NSEBANK", "^CNXPHARMA"}


def fake_prices(ticker):
    days, mkt = _fake_market_returns()
    if ticker == config.BENCHMARK:
        rets, price0, vol, turnover = mkt, 18000.0, 0.006, None
    elif ticker == config.VIX:
        rng = np.random.default_rng(seed_for(ticker))
        level = 14 + 4 * np.sin(np.arange(len(days)) / 70.0) + rng.normal(0, 0.8, len(days))
        level = np.clip(level, 9, 40)
        return pd.DataFrame({"Open": level, "High": level * 1.03, "Low": level * 0.97,
                             "Close": level, "Volume": 0.0}, index=days)
    else:
        rng = np.random.default_rng(seed_for(ticker))
        beta = rng.uniform(0.6, 1.3)
        idio_vol = rng.uniform(0.008, 0.016)
        phase, period = rng.uniform(0, 6.3), rng.uniform(40, 120)
        idio_drift = 0.0012 * np.sin(np.arange(len(days)) / period + phase)
        rets = beta * mkt + idio_drift + rng.normal(0, idio_vol, len(days))
        price0 = float(rng.uniform(150, 4000))
        vol = idio_vol
        turnover = float(rng.choice([18.0, 60.0, 150.0, 400.0, 900.0]))  # crore per day
    rng = np.random.default_rng(seed_for(ticker, "ohlc"))
    close = price0 * np.exp(np.cumsum(rets))
    prev = np.concatenate([[price0], close[:-1]])
    opens = prev * (1 + rng.normal(0, vol * 0.5, len(days)))
    high = np.maximum(opens, close) * (1 + np.abs(rng.normal(0, vol * 0.6, len(days))))
    low = np.minimum(opens, close) * (1 - np.abs(rng.normal(0, vol * 0.6, len(days))))
    if turnover is None:
        volume = np.full(len(days), 2.5e8)
    else:
        volume = turnover * 1e7 / close * rng.lognormal(0, 0.35, len(days))
    return pd.DataFrame({"Open": opens, "High": high, "Low": low, "Close": close,
                         "Volume": np.round(volume)}, index=days)


# ================================================================ fundamentals
_FUND_KEYS = {
    "trailingPE": "pe", "forwardPE": "forward_pe", "priceToBook": "price_to_book",
    "returnOnEquity": "roe", "debtToEquity": "debt_to_equity", "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth", "profitMargins": "profit_margin", "marketCap": "market_cap",
    "longName": "company",
}


def fundamentals(ctx, symbol):
    """Yahoo's current snapshot. It describes today, so --asof runs leave it out."""
    def live():
        import yfinance as yf
        info = yf.Ticker(config.yahoo_ticker(symbol)).info or {}
        out = {v: info.get(k) for k, v in _FUND_KEYS.items() if info.get(k) is not None}
        return out if len(out) > 1 else {}
    return get_dataset(ctx, "fundamentals", symbol, live, lambda: fake_fundamentals(symbol),
                       asof_supported=False,
                       asof_reason="Yahoo only serves today's ratios")


def fake_fundamentals(symbol):
    rng = np.random.default_rng(seed_for(symbol, "fund"))
    return {"company": config.COMPANY_NAMES.get(symbol, symbol),
            "pe": round(float(rng.uniform(8, 70)), 1), "forward_pe": round(float(rng.uniform(8, 60)), 1),
            "price_to_book": round(float(rng.uniform(1, 15)), 2), "roe": round(float(rng.uniform(0.05, 0.35)), 3),
            "debt_to_equity": round(float(rng.uniform(0, 150)), 1),
            "revenue_growth": round(float(rng.normal(0.10, 0.08)), 3),
            "earnings_growth": round(float(rng.normal(0.10, 0.15)), 3),
            "profit_margin": round(float(rng.uniform(0.04, 0.30)), 3),
            "market_cap": float(rng.uniform(1e12, 2e13))}


# ============================================================ quarterly results
_Q_ROWS = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "operating_profit": ["Operating Income"],
    "operating_profit_ebit": ["EBIT"],
    "net_profit": ["Net Income", "Net Income Common Stockholders"],
    "eps": ["Diluted EPS", "Basic EPS"],
}


def quarterly(ctx, symbol):
    """Quarterly income statement lines from Yahoo (rupees, EPS in rupees per share)."""
    def live():
        import yfinance as yf
        return quarterly_records(yf.Ticker(config.yahoo_ticker(symbol)).quarterly_income_stmt)
    return get_dataset(ctx, "quarterly", symbol, live, lambda: fake_quarterly(symbol, ctx.day))


def quarterly_records(stmt):
    """DataFrame (rows = line items, columns = period ends) -> list of dicts, oldest first."""
    if stmt is None or stmt.empty:
        return []

    def pick(names, col):
        for n in names:
            if n in stmt.index:
                v = stmt.loc[n, col]
                if v is not None and not (isinstance(v, float) and np.isnan(v)) and pd.notna(v):
                    return float(v)
        return None

    out = []
    for col in stmt.columns:
        rec = {"period_end": pd.Timestamp(col).date().isoformat()}
        rec["revenue"] = pick(_Q_ROWS["revenue"], col)
        op = pick(_Q_ROWS["operating_profit"], col)
        if op is None:
            op = pick(_Q_ROWS["operating_profit_ebit"], col)
            rec["operating_profit_basis"] = "EBIT" if op is not None else None
        rec["operating_profit"] = op
        rec["net_profit"] = pick(_Q_ROWS["net_profit"], col)
        rec["eps"] = pick(_Q_ROWS["eps"], col)
        if any(rec.get(k) is not None for k in ("revenue", "net_profit", "eps")):
            out.append(rec)
    return sorted(out, key=lambda r: r["period_end"])


def fake_quarterly(symbol, day):
    rng = np.random.default_rng(seed_for(symbol, "quarters"))
    ends = [d.date() for d in pd.date_range("2023-03-31", periods=24, freq="QE")]
    ends = [e for e in ends if e + timedelta(days=45) <= day][-6:]
    rev = float(rng.uniform(5_000, 60_000)) * 1e7
    margin = float(rng.uniform(0.10, 0.30))
    shares = float(rng.uniform(1e9, 6e9))
    drop = int(rng.integers(0, 4)) if rng.random() < 0.3 else None     # Yahoo often skips a quarter
    out = []
    for i, e in enumerate(ends):
        r2 = np.random.default_rng(seed_for(symbol, e.isoformat()))
        rev *= 1 + r2.normal(0.03, 0.04)
        m = margin + r2.normal(0, 0.015)
        net = rev * m * 0.75
        rec = {"period_end": e.isoformat(), "revenue": round(rev), "operating_profit": round(rev * m),
               "net_profit": round(net), "eps": round(net / shares, 2)}
        if config.UNIVERSE.get(symbol) == "Financials":
            rec["operating_profit"] = None                          # as Yahoo does for banks
        if drop is not None and i == drop:
            continue
        out.append(rec)
    return out


# ==================================================================== headlines
def headlines(ctx, symbol):
    """Up to NEWS_MAX_HEADLINES titles from Google News RSS for the last NEWS_DAYS days."""
    name = config.COMPANY_NAMES.get(symbol, symbol)

    def live():
        if ctx.is_asof:   # Google's date operators let us ask for the past window
            start = ctx.day - timedelta(days=config.NEWS_DAYS)
            q = f'"{name}" after:{start.isoformat()} before:{(ctx.day + timedelta(days=1)).isoformat()}'
        else:
            q = f'"{name}" when:{config.NEWS_DAYS}d'
        url = (f"https://news.google.com/rss/search?q={quote_plus(q)}"
               "&hl=en-IN&gl=IN&ceid=IN:en")
        r = http_get(url, headers={"User-Agent": BROWSER_UA})
        return parse_news_rss(r.text, ctx.day)
    return get_dataset(ctx, "headlines", symbol, live, lambda: fake_headlines(symbol, ctx.day))


def parse_news_rss(xml_text, day, limit=None):
    """Parse Google News RSS, keep items published on or before `day` (IST), newest first."""
    import feedparser
    feed = feedparser.parse(xml_text)
    if feed.bozo and not feed.entries:
        raise ValueError(f"unparseable RSS: {feed.bozo_exception}")
    cutoff = datetime.combine(day + timedelta(days=1), datetime.min.time(), IST)
    earliest = cutoff - timedelta(days=config.NEWS_DAYS + 1)
    items = []
    for e in feed.entries:
        try:
            published = email.utils.parsedate_to_datetime(e.get("published", ""))
        except (TypeError, ValueError):
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if not (earliest <= published < cutoff):
            continue
        title = (e.get("title") or "").strip()
        source = (e.get("source") or {}).get("title", "") if isinstance(e.get("source"), dict) else ""
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        items.append({"title": title, "source": source, "link": e.get("link", ""),
                      "published": published.astimezone(IST).strftime("%Y-%m-%d %H:%M")})
    items.sort(key=lambda x: x["published"], reverse=True)
    return items[: limit or config.NEWS_MAX_HEADLINES]


_GOOD = ["{n} shares rise after brokerage upgrade", "{n} wins large order, stock gains",
         "{n} reports strong quarterly volume growth", "Analysts raise target price on {n}"]
_BAD = ["{n} shares slip as margins come under pressure", "{n} faces regulatory scrutiny over disclosures",
        "Brokerage downgrades {n} on valuation concerns", "{n} stock falls after weak demand commentary"]
_NEUTRAL = ["{n} to hold board meeting next month", "{n} announces management appointment",
            "Sensex, Nifty end flat; {n} among most traded"]


def fake_headlines(symbol, day):
    rng = np.random.default_rng(seed_for(symbol, day.isoformat(), "news"))
    name = config.COMPANY_NAMES.get(symbol, symbol)
    pool = _GOOD + _BAD + _NEUTRAL
    n = int(rng.integers(2, config.NEWS_MAX_HEADLINES + 1))
    picks = rng.choice(len(pool), size=n, replace=False)
    out = []
    for i, k in enumerate(sorted(picks)):
        when = datetime.combine(day, datetime.min.time()) + timedelta(hours=17) - timedelta(hours=11 * i)
        out.append({"title": pool[k].format(n=name), "source": "Offline Wire",
                    "link": f"https://example.invalid/{symbol}/{day.isoformat()}/{k}",
                    "published": when.strftime("%Y-%m-%d %H:%M")})
    return out


def dumps(obj) -> str:
    return json.dumps(obj, default=str, separators=(",", ":"))


# ================================================================= article text
# Google News RSS links are not HTTP redirects any more. The article page carries
# a signature and timestamp that Google's own batchexecute endpoint exchanges for
# the publisher URL; this is what the (currently broken) googlenewsdecoder
# package does, written out here so it has no extra dependencies.
JINA_UA = "swarm-signal-desk/1.0 (personal research; low volume)"   # a browser UA trips Jina's bot check
_jina_last = [0.0]
_BAD_TITLES = ("just a moment", "page not found", "access denied", "403 forbidden", "404", "attention required",
               "are you a robot", "subscribe to continue")


def resolve_google_news(link):
    import re
    from urllib.parse import urlparse
    if "news.google.com" not in link:
        return link
    gid = urlparse(link).path.rstrip("/").split("/")[-1]
    page = http_get(f"https://news.google.com/rss/articles/{gid}").text
    sg, ts = re.search(r'data-n-a-sg="([^"]+)"', page), re.search(r'data-n-a-ts="([^"]+)"', page)
    if not (sg and ts):
        raise ValueError("Google News page has no decoding signature")
    inner = ["garturlreq", [["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1, None, None, None, None,
                             None, 0, 1], "X", "X", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0],
             gid, int(ts.group(1)), sg.group(1)]
    import requests
    r = requests.post("https://news.google.com/_/DotsSplashUi/data/batchexecute",
                      data={"f.req": json.dumps([[["Fbv4je", json.dumps(inner), None, "generic"]]])},
                      headers={"User-Agent": BROWSER_UA,
                               "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
                      timeout=config.HTTP_TIMEOUT)
    r.raise_for_status()
    url = json.loads(json.loads(r.text.split("\n\n", 1)[1])[0][2])[1]
    if not str(url).startswith("http"):
        raise ValueError("could not decode Google News link")
    return url


def jina_text(url):
    """Readable text of an article through Jina Reader (free, about 20 requests a minute)."""
    import time
    wait = config.JINA_MIN_INTERVAL_SECONDS - (time.time() - _jina_last[0])
    if wait > 0:
        time.sleep(wait)
    _jina_last[0] = time.time()
    r = http_get(f"https://r.jina.ai/{url}", headers={"User-Agent": JINA_UA, "Accept": "text/plain"}, timeout=45)
    return clean_article(r.text)


def clean_article(text, max_words=None):
    """Strip Jina's header lines and markdown clutter, reject error pages, truncate."""
    import re
    max_words = max_words or config.ARTICLE_MAX_WORDS
    title = ""
    m = re.search(r"^Title:\s*(.*)$", text, flags=re.M)
    if m:
        title = m.group(1).strip()
    if any(b in title.lower() for b in _BAD_TITLES):
        raise ValueError(f"publisher returned an error page ({title[:60]})")
    body = text.split("Markdown Content:", 1)[1] if "Markdown Content:" in text else text
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", body)                  # images
    body = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", body)                 # links -> their text
    body = re.sub(r"(?m)^\s*[#>*\-|=_]+\s*", " ", body)                   # markdown line prefixes
    words = body.split()
    if len(words) < 60:
        raise ValueError(f"too little article text ({len(words)} words)")
    clipped = words[:max_words]
    return " ".join(clipped) + (" …" if len(words) > max_words else "")


def articles(ctx, symbol, headlines_res):
    """Readable text for the newest ARTICLES_PER_STOCK headlines. Failed fetches fall
    back to headline-only and are counted. Each article URL is cached for good."""
    def live():
        if headlines_res is None or headlines_res.get("status") != "ok":
            return []
        out = []
        for h in headlines_res["data"][: config.ARTICLES_PER_STOCK]:
            key = f"article|{h['link']}"
            hit = db.cache_get(ctx.con, key)
            if hit is None:
                try:
                    url = with_retry(lambda: resolve_google_news(h["link"]))
                    hit = {"url": url, "text": jina_text(url)}
                    db.cache_put(ctx.con, key, hit)
                except Exception as e:  # noqa: BLE001 - fall back to the headline alone
                    hit = {"url": None, "text": None, "error": f"{type(e).__name__}: {str(e)[:120]}"}
            out.append({"title": h["title"], "source": h.get("source"), "published": h.get("published"), **hit})
        if out and all(a.get("text") is None for a in out):
            raise RuntimeError("every article fetch failed: " + "; ".join(a.get("error", "") for a in out)[:300])
        return out

    def fake():
        if headlines_res is None or headlines_res.get("status") != "ok":
            return []
        return [{"title": h["title"], "source": h.get("source"), "published": h.get("published"),
                 "url": h["link"], "text": f"(synthetic offline article) {h['title']}. " + "Details follow. " * 30}
                for h in headlines_res["data"][: config.ARTICLES_PER_STOCK]]

    return get_dataset(ctx, "articles", symbol, live, fake, asof_supported=False,
                       asof_reason="articles may have been edited since that date", retries=0)
