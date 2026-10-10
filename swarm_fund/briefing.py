"""Builds exactly what each agent is shown. Code fetches and computes; agents read.

A brief is {"agent", "data": JSON-able dict, "texts": [{"tag", "items": [str]}]}.
render() turns it into the prompt text, and that text is what the journal stores
as the agent's input, so every call can be audited later.
"""
import json
import math

import config
from fetch import is_ok
from indicators import rounded

CHARS_PER_TOKEN = 3.5


# ------------------------------------------------------------------ gaps
def gap(name, res, what=None):
    """Describe a dataset that did not arrive, or None when it did.

    "empty" is a fact (the source answered and had nothing), so it is not a gap.
    """
    if res is None:
        return f"{name}: UNAVAILABLE (not fetched). Missing data is not a signal."
    if res.get("status") == "unavailable":
        return (f"{name}: UNAVAILABLE today ({res.get('error') or 'fetch failed'}). "
                "Missing data is not a signal either way.")
    return None


def gaps(*pairs):
    return [g for g in (gap(n, r) for n, r in pairs) if g]


# --------------------------------------------------------------- render
def _clean(text: str) -> str:
    """Untrusted text cannot open or close our tags: angle brackets are replaced."""
    return str(text).replace("<", "‹").replace(">", "›").strip()


def render(brief) -> str:
    out = [f"DATA FOR THE {brief['agent'].upper()} ROLE (computed by code; JSON):",
           json.dumps(brief["data"], default=_json_default, indent=None, separators=(",", ":"))]
    for block in brief.get("texts", []):
        tag = f"untrusted_{block['tag']}"
        body = "\n".join(_clean(x) for x in block["items"]) if block["items"] else "(none)"
        out.append(f"<{tag}>\n{body}\n</{tag}>")
    return "\n\n".join(out)


def _json_default(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    return str(o)


def est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def fit(brief, ceiling=None):
    """Trim the longest untrusted block, then the longest list in the data, until
    the rendered brief fits the token ceiling. Each trim is noted in the data."""
    ceiling = ceiling or config.PAYLOAD_TOKEN_CEILING
    trimmed = []
    for _ in range(400):
        if est_tokens(render(brief)) <= ceiling:
            break
        blocks = [b for b in brief.get("texts", []) if b["items"]]
        if blocks:
            b = max(blocks, key=lambda b: sum(len(x) for x in b["items"]))
            longest = max(range(len(b["items"])), key=lambda i: len(b["items"][i]))
            item = b["items"][longest]
            if len(item) > 400:
                b["items"][longest] = item[: int(len(item) * 0.8)] + " …[truncated]"
            else:
                b["items"].pop()
            trimmed.append(b["tag"])
            continue
        key = _longest_list(brief["data"])
        if key is None:
            break
        container, k = key
        v = container[k]
        container[k] = v[1:] if isinstance(v, list) else v[: int(len(v) * 0.8)] + " …[truncated]"
        trimmed.append(k)
    if trimmed:
        brief["data"]["trimmed_to_fit"] = sorted(set(trimmed))
    return brief


def _longest_list(d):
    """Find the biggest list (or long string) anywhere in nested dicts.

    Convention: every list in a brief is chronological, oldest first, so dropping
    the first element drops the oldest item. Returns (container, key) or None."""
    best, stack = None, [d]
    while stack:
        cur = stack.pop()
        for k, v in cur.items():
            if isinstance(v, list) and len(v) > 1:
                size = len(v)
            elif isinstance(v, str) and len(v) > 300:
                size = len(v) // 40
            else:
                size = 0
            if size and (best is None or size > best[2]):
                best = (cur, k, size)
            if isinstance(v, dict):
                stack.append(v)
    return None if best is None else (best[0], best[1])


# --------------------------------------------------------------- briefs
def _pct(x, places=1):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x * 100, places)


def technical(sym, day, ind, screen, extra=None):
    data = {"symbol": sym, "as_of": day, "indicators": rounded(ind),
            "screen_score": screen,
            "notes": "sma = simple moving average; RSI and ATR use Wilder smoothing; returns and "
                     "distances are percent; volume_ratio = today's volume / prior 20-day average; "
                     "turnover in crore rupees per day (20-day average).",
            "data_gaps": []}
    data.update(extra or {})
    return {"agent": "technical", "data": data, "texts": []}


def fundamental(sym, day, fund_res, extra=None, texts=None):
    snap = None
    if is_ok(fund_res):
        f = fund_res["data"]
        snap = {"company": f.get("company"), "pe": f.get("pe"), "forward_pe": f.get("forward_pe"),
                "price_to_book": f.get("price_to_book"), "roe_pct": _pct(f.get("roe")),
                "debt_to_equity": f.get("debt_to_equity"), "revenue_growth_pct": _pct(f.get("revenue_growth")),
                "earnings_growth_pct": _pct(f.get("earnings_growth")), "profit_margin_pct": _pct(f.get("profit_margin")),
                "market_cap_cr": round(f["market_cap"] / 1e7) if f.get("market_cap") else None}
        snap = {k: v for k, v in snap.items() if v is not None}
    data = {"symbol": sym, "as_of": day, "snapshot": snap,
            "notes": "snapshot is Yahoo's current trailing data; growth is year on year.",
            "data_gaps": gaps(("fundamentals snapshot", fund_res))}
    data.update(extra or {})
    return {"agent": "fundamental", "data": data, "texts": texts or []}


def news(sym, day, news_res, extra=None, texts=None):
    items = []
    if is_ok(news_res):
        items = [f"[{h['published']}] {h['title']} ({h['source']})" for h in news_res["data"]]
    data = {"symbol": sym, "company": config.COMPANY_NAMES.get(sym, sym), "as_of": day,
            "headline_count": len(items),
            "source_ranking": "exchange filings are primary sources from the company and outrank headlines",
            "headline_status": ("none found in the last %d days (source answered)" % config.NEWS_DAYS
                                if news_res and news_res.get("status") == "empty" else None),
            "data_gaps": gaps(("headlines", news_res))}
    data = {k: v for k, v in data.items() if v is not None}
    data.update(extra or {})
    blocks = (texts or []) + [{"tag": "headlines", "items": items}]
    return {"agent": "news", "data": data, "texts": blocks}


def macro(sym, sector, day, market, extra=None):
    data = {"symbol": sym, "sector": sector, "as_of": day, "market": rounded(market),
            "notes": "returns in percent; vix_avg20 excludes today.", "data_gaps": []}
    data.update(extra or {})
    return {"agent": "macro", "data": data, "texts": []}


def debate(sym, day, ind, reports):
    return {"agent": "debate", "data": {"symbol": sym, "as_of": day, "indicators": rounded(ind),
                                        "analyst_reports": reports}, "texts": []}


def manager(sym, sector, day, ind, held, reports, bull, bear, extra=None):
    data = {"symbol": sym, "sector": sector, "as_of": day, "currently_held": held,
            "allowed_actions": ["SELL", "HOLD"] if held else ["BUY", "HOLD"],
            "indicators": rounded(ind), "analyst_reports": reports,
            "bull_case": bull, "bear_case": bear}
    data.update(extra or {})
    return {"agent": "manager", "data": data, "texts": []}
