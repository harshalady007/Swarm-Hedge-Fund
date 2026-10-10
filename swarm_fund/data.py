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
    return {t: df[df.index <= end] for t, df in out.items() if df is not None and len(df)}


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
