"""Indicators against values worked out by hand."""
import numpy as np
import pandas as pd
import pytest

import indicators as ind


def test_wilder_rsi_small_by_hand():
    # deltas 1, -0.5, 1 seed avg gain 2/3, avg loss 1/6; then +0.5 and -1 leave gain = loss
    close = [10, 11, 10.5, 11.5, 12, 11]
    assert ind.wilder_rsi(close, 3) == pytest.approx(50.0)


def test_wilder_rsi_edges():
    assert ind.wilder_rsi(list(range(20)), 14) == 100.0
    assert np.isnan(ind.wilder_rsi([1, 2, 3], 14))


def test_wilder_atr_small_by_hand():
    high, low, close = [10, 12, 11, 13], [9, 10, 9.5, 11], [9.5, 11, 10, 12]
    # true ranges 2.5, 1.5, 3.0 -> seed (2.5 + 1.5) / 2 = 2.0 -> (2.0 + 3.0) / 2 = 2.5
    assert ind.wilder_atr(high, low, close, 2) == pytest.approx(2.5)


def linear_frame(n=260):
    i = np.arange(n, dtype=float)
    close = 100 + i
    vol = np.full(n, 1000.0)
    vol[-1] = 2000.0
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": vol},
                        index=pd.bdate_range("2025-01-01", periods=n))


def test_compute_on_linear_series():
    r = ind.compute(linear_frame())
    assert r["close"] == 359
    assert r["sma20"] == pytest.approx(349.5)
    assert r["sma50"] == pytest.approx(334.5)
    assert r["sma200"] == pytest.approx(259.5)
    assert r["rsi14"] == 100.0
    assert r["atr14"] == pytest.approx(2.0)
    assert r["ret5_pct"] == pytest.approx((359 / 354 - 1) * 100)
    assert r["ret20_pct"] == pytest.approx((359 / 339 - 1) * 100)
    assert r["ret60_pct"] == pytest.approx((359 / 299 - 1) * 100)
    assert r["pct_below_52w_high"] == pytest.approx(1 / 360 * 100)       # high 360
    assert r["pct_above_52w_low"] == pytest.approx((359 - 107) / 107 * 100)  # low of bar 8 is 107
    assert r["volume_ratio"] == pytest.approx(2.0)
    # closes 340..358 at 1000 shares plus 359 at 2000, averaged over 20 days, in crore
    assert r["turnover_cr_20d"] == pytest.approx((6_631_000 + 718_000) / 20 / 1e7)


def test_compute_needs_enough_history():
    assert ind.compute(linear_frame(150)) is None


def test_screen_score_by_hand():
    r = ind.compute(linear_frame())
    # trend +2, ret20 5.90/5 = +1.18, RSI 100 -> -1.5, volume x2 +0.5, at the high +0.5
    expected = 2 + (359 / 339 - 1) * 100 / 5 - 1.5 + 0.5 + 0.5
    assert ind.screen_score(r) == pytest.approx(expected, abs=1e-3)


def test_screen_score_caps_and_branches():
    base = {"close": 100, "sma50": 110, "sma200": 90, "ret20_pct": -40.0, "rsi14": 50,
            "volume_ratio": 1.0, "pct_below_52w_high": 20}
    # only above the 200-day: +1; return capped at -2; RSI band +1
    assert ind.screen_score(base) == pytest.approx(0.0)
    base.update(close=80, ret20_pct=30.0, rsi14=81)
    assert ind.screen_score(base) == pytest.approx(0 + 2 - 1.5)


def test_shortlist_adds_every_held_stock():
    scores = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0, "G": -1}
    assert ind.shortlist(scores, held={"G", "A"}, n=2) == ["B", "C", "A", "G"]
