"""Per-stock indicators and the code screen. Pure functions over a price DataFrame."""
import math

import numpy as np
import pandas as pd

import config


def sma(values, n):
    values = np.asarray(values, dtype=float)
    return float(values[-n:].mean()) if len(values) >= n else float("nan")


def wilder_rsi(close, n=14):
    """RSI with Wilder smoothing: seed with the simple mean of the first n moves."""
    close = np.asarray(close, dtype=float)
    if len(close) < n + 1:
        return float("nan")
    delta = np.diff(close)
    gains, losses = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    avg_gain, avg_loss = gains[:n].mean(), losses[:n].mean()
    for g, l in zip(gains[n:], losses[n:]):
        avg_gain = (avg_gain * (n - 1) + g) / n
        avg_loss = (avg_loss * (n - 1) + l) / n
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def wilder_atr(high, low, close, n=14):
    """ATR with Wilder smoothing. True range needs a previous close, so it starts at bar 2."""
    high, low, close = (np.asarray(x, dtype=float) for x in (high, low, close))
    if len(close) < n + 1:
        return float("nan")
    prev = close[:-1]
    tr = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - prev), np.abs(low[1:] - prev)])
    atr = tr[:n].mean()
    for x in tr[n:]:
        atr = (atr * (n - 1) + x) / n
    return float(atr)


def pct_return(close, n):
    close = np.asarray(close, dtype=float)
    if len(close) <= n or close[-1 - n] == 0:
        return float("nan")
    return (close[-1] / close[-1 - n] - 1.0) * 100.0


def compute(df: pd.DataFrame):
    """The 13 indicators. Returns None when history is too short to trust them."""
    if df is None or len(df) < config.MIN_BARS:
        return None
    c, h, l, v = (df[k].to_numpy(dtype=float) for k in ("Close", "High", "Low", "Volume"))
    last252 = slice(-252, None)
    hi52, lo52 = float(np.max(h[last252])), float(np.min(l[last252]))
    prior_vol = v[-21:-1]                           # the 20 sessions before today
    avg_vol = float(prior_vol.mean()) if len(prior_vol) == 20 else float("nan")
    turnover = c[-20:] * v[-20:]
    ind = {
        "close": float(c[-1]),
        "sma20": sma(c, 20), "sma50": sma(c, 50), "sma200": sma(c, 200),
        "rsi14": wilder_rsi(c, 14),
        "atr14": wilder_atr(h, l, c, 14),
        "ret5_pct": pct_return(c, 5), "ret20_pct": pct_return(c, 20), "ret60_pct": pct_return(c, 60),
        "pct_below_52w_high": (hi52 - c[-1]) / hi52 * 100.0 if hi52 else float("nan"),
        "pct_above_52w_low": (c[-1] - lo52) / lo52 * 100.0 if lo52 else float("nan"),
        "volume_ratio": float(v[-1] / avg_vol) if avg_vol and avg_vol > 0 else float("nan"),
        "turnover_cr_20d": float(turnover.mean() / 1e7),
    }
    if any(math.isnan(ind[k]) for k in ("close", "sma200", "rsi14", "atr14")):
        return None
    return ind


def screen_score(ind) -> float:
    """Cheap code score that decides which stocks the agents see."""
    s = 0.0
    if ind["close"] > ind["sma50"] > ind["sma200"]:
        s += 2.0
    elif ind["close"] > ind["sma200"]:
        s += 1.0
    r20 = ind["ret20_pct"]
    if not math.isnan(r20):
        s += max(-2.0, min(2.0, r20 / 5.0))
    rsi = ind["rsi14"]
    if 45 <= rsi <= 70:
        s += 1.0
    if rsi > 80:
        s -= 1.5
    if not math.isnan(ind["volume_ratio"]) and ind["volume_ratio"] > 1.2:
        s += 0.5
    if ind["pct_below_52w_high"] <= 5.0:
        s += 0.5
    return round(s, 3)


def shortlist(scores: dict, held: set, n=None):
    """Top-n by screen score among stocks not held, plus every held stock."""
    n = config.SHORTLIST_SIZE if n is None else n
    candidates = sorted((s for s in scores if s not in held), key=lambda s: (-scores[s], s))
    return candidates[:n] + sorted(s for s in held if s in scores)


def market_context(nifty: pd.DataFrame, vix: pd.DataFrame | None):
    """NIFTY trend and India VIX, all computed here."""
    c = nifty["Close"].to_numpy(dtype=float)
    ctx = {
        "nifty_close": float(c[-1]),
        "nifty_sma50": sma(c, 50), "nifty_sma200": sma(c, 200),
        "nifty_ret5_pct": pct_return(c, 5), "nifty_ret20_pct": pct_return(c, 20),
        "nifty_ret60_pct": pct_return(c, 60), "nifty_rsi14": wilder_rsi(c, 14),
    }
    ctx["nifty_above_200dma"] = bool(c[-1] > ctx["nifty_sma200"]) if not math.isnan(ctx["nifty_sma200"]) else None
    if vix is not None and len(vix) >= 21:
        vc = vix["Close"].to_numpy(dtype=float)
        ctx.update({"vix": float(vc[-1]), "vix_avg20": sma(vc[:-1], 20),
                    "vix_change5_pct": pct_return(vc, 5)})
    return ctx


def rounded(d, places=2):
    """Round floats for prompts and messages; NaN becomes None so JSON stays valid."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = None if math.isnan(v) else round(v, places)
        else:
            out[k] = v
    return out
