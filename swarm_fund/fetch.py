"""One way to fetch anything: timeout, one retry, per-day cache, visible failure.

Every dataset comes back as a plain dict:

    {"status": "ok" | "empty" | "unavailable", "data": ..., "error": str | None,
     "source": str}

"empty" means the source answered and there was nothing to report (for example
no announcements this month). "unavailable" means we could not find out. The
briefings turn every non-ok status into a data_gaps entry that says which one
it was, so an agent never mistakes a dead endpoint for a quiet company.
"""
import json
import time
from dataclasses import dataclass, field
from datetime import date

import requests

import config
import db

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


@dataclass
class Ctx:
    """Everything a fetcher needs to know about the run."""
    con: object
    day: date                 # the trading day being processed
    offline: bool
    is_asof: bool             # True when --asof pretends today is an earlier date
    log: list = field(default_factory=list)

    @property
    def day_str(self) -> str:
        return self.day.isoformat()

    def note(self, msg: str):
        self.log.append(msg)
        print(f"  [data] {msg}")


def result(status, data=None, error=None, source=""):
    return {"status": status, "data": data, "error": error, "source": source}


def is_ok(res) -> bool:
    return bool(res) and res.get("status") == "ok"


def with_retry(fn, retries=None, pause=1.5):
    """Call fn(); on any exception wait and try again, `retries` more times."""
    retries = config.HTTP_RETRIES if retries is None else retries
    last = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - every source failure is handled the same way
            last = e
            if attempt < retries:
                time.sleep(pause)
    raise last


def http_get(url, params=None, headers=None, timeout=None, stream=False):
    h = {"User-Agent": BROWSER_UA, "Accept": "*/*"}
    h.update(headers or {})
    r = requests.get(url, params=params, headers=h, timeout=timeout or config.HTTP_TIMEOUT,
                     stream=stream)
    r.raise_for_status()
    return r


def _is_empty(data) -> bool:
    return data is None or (hasattr(data, "__len__") and len(data) == 0)


def get_dataset(ctx: Ctx, dataset: str, symbol: str, live_fn, fake_fn,
                asof_supported=True, asof_reason="cannot be reconstructed for a past date",
                cache_key_extra="", snapshot=True):
    """Fetch one dataset for one symbol (use "_MARKET" for market-wide data).

    live_fn() returns the cleaned data or raises. fake_fn() returns deterministic
    offline data. Successful answers are cached for the day, so a rerun makes no
    network calls. Every answer, good or bad, goes to the snapshot store.
    """
    if ctx.offline:
        data = fake_fn()
        res = result("empty" if _is_empty(data) else "ok", data, source="offline-fake")
    elif ctx.is_asof and not asof_supported:
        res = result("unavailable", None, f"left out of --asof runs: {asof_reason}", "asof-policy")
    else:
        key = f"{dataset}|{symbol}|{ctx.day_str}|{int(ctx.is_asof)}|{cache_key_extra}"
        cached = db.cache_get(ctx.con, key)
        if cached is not None:
            return cached
        try:
            data = with_retry(live_fn)
            res = result("empty" if _is_empty(data) else "ok", data, source="live")
            db.cache_put(ctx.con, key, res)
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {str(e)[:200]}"
            res = result("unavailable", None, msg, "live")
            ctx.note(f"{dataset} {symbol} unavailable ({msg})")
    if snapshot:
        _snapshot(ctx, symbol, dataset, res)
    return res


def _snapshot(ctx, symbol, dataset, res):
    """First good answer for a day wins; a later good answer replaces an earlier failure."""
    row = ctx.con.execute("SELECT payload FROM snapshots WHERE trading_day=? AND symbol=? AND dataset=?",
                          (ctx.day_str, symbol, dataset)).fetchone()
    if row is not None:
        if json.loads(row["payload"]).get("status") in ("ok", "empty"):
            return
        ctx.con.execute("DELETE FROM snapshots WHERE trading_day=? AND symbol=? AND dataset=?",
                        (ctx.day_str, symbol, dataset))
    db.snapshot(ctx.con, ctx.day_str, symbol, dataset, res)
