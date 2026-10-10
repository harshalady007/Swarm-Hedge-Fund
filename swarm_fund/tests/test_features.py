"""Phase 1 calculations: price-series compaction, quarterly growth, sector relative
strength and the manager's portfolio and track-record context."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

import config
import db
import features
import journal

CR = 1e7


# ------------------------------------------------------------------ technical
def frame(closes, start="2026-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": np.full(len(c), 123456.0)},
                        index=idx)


def test_compact_bars_rows_and_rounding():
    df = frame([999.994, 1234.56, 1500.04])
    out = features.compact_bars(df, n=2)
    assert out["columns"].startswith("date,open")
    assert out["rows"] == ["2026-01-02,1234.6,1235.6,1233.6,1234.6,123", "2026-01-05,1500.0,1501.0,1499.0,1500.0,123"]


def test_weekly_closes_take_friday_close():
    df = frame(range(100, 115))              # 2026-01-01 (Thu) .. 2026-01-21 (Wed)
    out = features.weekly_closes(df)
    assert out["rows"][:2] == ["2026-01-02:101.0", "2026-01-09:106.0"]
    assert out["last_week_partial"] is True


# ---------------------------------------------------------------- quarterly
def rec(end, rev, op, net, eps):
    return {"period_end": end, "revenue": rev * CR, "operating_profit": op * CR if op is not None else None,
            "net_profit": net * CR, "eps": eps}


# Shaped like Yahoo's real TCS reply on 2026-10-10: the Sep-2025 quarter is missing.
RECORDS = [rec("2025-03-31", 1000, 250, 200, 10.0), rec("2025-06-30", 1100, 260, 210, 10.5),
           rec("2025-12-31", 1200, 300, 240, 12.0), rec("2026-03-31", 1250, 300, 250, 12.5),
           rec("2026-06-30", 1320, 330, 264, 13.2)]


def test_growth_is_matched_by_quarter_not_position():
    table, gaps = features.quarterly_table(RECORDS, date(2026, 10, 9))
    q = {r["period_end"]: r for r in table["quarters"]}
    assert "revenue_qoq_pct" not in q["2025-12-31"]                    # previous quarter is missing
    assert q["2026-03-31"]["revenue_qoq_pct"] == pytest.approx(4.2)     # 1250 / 1200
    assert q["2026-03-31"]["revenue_yoy_pct"] == pytest.approx(25.0)    # 1250 / 1000
    assert q["2026-06-30"]["net_profit_yoy_pct"] == pytest.approx(25.7)  # 264 / 210
    assert q["2026-06-30"]["eps_qoq_pct"] == pytest.approx(5.6)         # 13.2 / 12.5
    assert q["2026-06-30"]["operating_margin_pct"] == pytest.approx(25.0)
    assert q["2026-06-30"]["revenue_cr"] == 1320
    assert any("Q3 2025" in g for g in gaps)


def test_margin_trend_by_hand():
    table, _ = features.quarterly_table(RECORDS, date(2026, 10, 9))
    mt = table["margin_trend"]
    # latest 25.0%; prior three Jun-25 23.6, Dec-25 25.0, Mar-26 24.0 -> average 24.2; year ago 23.6
    assert mt["latest_pct"] == 25.0
    assert mt["change_vs_prior_avg_pp"] == pytest.approx(0.8, abs=0.05)
    assert mt["change_vs_year_ago_pp"] == pytest.approx(25.0 - 23.6, abs=0.05)


def test_negative_base_gives_no_growth():
    assert features.pct_change(100, -50) is None
    assert features.pct_change(100, 0) is None
    assert features.pct_change(None, 10) is None
    assert features.pct_change(110, 100) == pytest.approx(10.0)


def test_asof_hides_unpublished_quarters():
    table, gaps = features.quarterly_table(RECORDS, date(2026, 7, 15), is_asof=True)
    assert table["quarters"][-1]["period_end"] == "2026-03-31"
    assert any("hidden" in g for g in gaps)


def test_missing_fields_and_staleness_are_reported():
    recs = [{"period_end": "2025-03-31", "revenue": 900 * CR, "operating_profit": None, "net_profit": 150 * CR,
             "eps": None},
            {"period_end": "2025-06-30", "revenue": 950 * CR, "operating_profit": None, "net_profit": 160 * CR,
             "eps": 10.6}]
    table, gaps = features.quarterly_table(recs, date(2026, 10, 9))
    assert any("operating profit not reported" in g for g in gaps)
    assert any("stale" in g for g in gaps)
    assert table["margin_trend"]["measure"] == "net_margin_pct"


def test_no_quarters_is_a_gap():
    table, gaps = features.quarterly_table([], date(2026, 10, 9))
    assert table is None and gaps


# ------------------------------------------------------------------- sector
def test_usable_index_rejects_single_row_and_stale():
    assert features.usable_index(frame([100.0]), date(2026, 1, 1))[0] is False
    df = frame(np.linspace(100, 120, 100))
    assert features.usable_index(df, df.index[-1].date())[0] is True
    assert features.usable_index(df, df.index[-1].date() + pd.Timedelta(days=10))[0] is False


def _prices_for_sector():
    n = 80
    stock = frame(100 * np.ones(n))
    stock.iloc[-20:, stock.columns.get_loc("Close")] = 110.0     # +10% over 20 days
    peer1 = frame(np.linspace(100, 100, n))
    peer1.iloc[-1, peer1.columns.get_loc("Close")] = 104.0        # +4% over 20 and 60 days
    peer2 = frame(100 * np.ones(n))                                # 0%
    nifty = frame(100 * np.ones(n))
    nifty.iloc[-1, nifty.columns.get_loc("Close")] = 102.0         # +2%
    index = frame(100 * np.ones(n))
    index.iloc[-1, index.columns.get_loc("Close")] = 101.0         # +1%
    return {"AAA": stock, "BBB": peer1, "CCC": peer2, "^NSEI": nifty, "^IDX": index}


def test_sector_context_basket(monkeypatch):
    monkeypatch.setattr(config, "UNIVERSE", {"AAA": "Widgets", "BBB": "Widgets", "CCC": "Widgets"})
    prices = _prices_for_sector()
    out, gaps = features.sector_context("AAA", "Widgets", prices, {"Widgets": (None, "none")}, prices["^NSEI"])
    assert "basket of 2 peer(s)" in out["sector_benchmark"]
    assert out["sector_ret20_pct"] == pytest.approx(2.0)            # mean of +4% and 0%
    assert out["stock_ret20_pct"] == pytest.approx(10.0)
    assert out["stock_vs_sector_20d_pp"] == pytest.approx(8.0)
    assert out["stock_vs_nifty_20d_pp"] == pytest.approx(8.0)
    assert out["sector_vs_nifty_20d_pp"] == pytest.approx(0.0)
    assert gaps == []


def test_sector_context_index(monkeypatch):
    monkeypatch.setattr(config, "UNIVERSE", {"AAA": "Widgets"})
    prices = _prices_for_sector()
    out, _ = features.sector_context("AAA", "Widgets", prices, {"Widgets": ("^IDX", "ok")}, prices["^NSEI"])
    assert out["sector_benchmark"] == "Yahoo index ^IDX"
    assert out["sector_ret20_pct"] == pytest.approx(1.0)
    assert out["stock_vs_sector_20d_pp"] == pytest.approx(9.0)


def test_sector_context_without_index_or_peers_is_a_gap(monkeypatch):
    monkeypatch.setattr(config, "UNIVERSE", {"AAA": "Telecom"})
    prices = _prices_for_sector()
    out, gaps = features.sector_context("AAA", "Telecom", prices, {"Telecom": (None, "none")}, prices["^NSEI"])
    assert "sector_ret20_pct" not in out and "stock_vs_nifty_20d_pp" in out
    assert gaps and "no Telecom peers" in gaps[0]


def test_pick_sector_benchmarks_falls_through_candidates(monkeypatch):
    monkeypatch.setattr(config, "SECTOR_INDEX_CANDIDATES", {"Fin": ["^ONE_ROW", "^GOOD"], "Tel": []})
    good = frame(np.linspace(100, 110, 100))
    day = good.index[-1].date()
    picks = features.pick_sector_benchmarks({"^ONE_ROW": frame([1.0]), "^GOOD": good}, day)
    assert picks["Fin"][0] == "^GOOD" and "^ONE_ROW: 1 bars" in picks["Fin"][1]
    assert picks["Tel"][0] is None


# ---------------------------------------------------------- manager context
def test_track_record_states_sample_size():
    tr = features.track_record([{"pnl": 50, "entry_cash": 1000}, {"pnl": -20, "entry_cash": 1000}])
    assert tr["closed_trades"] == 2 and tr["win_rate_pct"] == 50
    assert "far too few" in tr["sample_size_warning"]
    assert "no closed trades" in features.track_record([])["sample_size_warning"]


def test_portfolio_context_and_prior_calls():
    con = db.connect(":memory:")
    journal.ensure_start(con, "2026-01-05", 100.0)
    prices = {"AAA": frame([100.0, 100.0, 110.0], start="2026-01-05")}
    sid = journal.log_signal(con, "2026-01-05", {"symbol": "AAA", "sector": "IT", "held": False, "close": 100.0,
                                                 "action": "BUY", "confidence": 70, "status": "APPROVED",
                                                 "entry_high": 101.0}, {}, [])
    journal.place_order(con, sid, "2026-01-05", {"status": "APPROVED", "symbol": "AAA", "sector": "IT", "qty": 100,
                                                 "entry_low": 99.5, "entry_high": 101.0, "stop": 95.0,
                                                 "target": 120.0})
    db.set_meta(con, "last_advanced", "2026-01-05")
    journal.advance(con, prices, ["2026-01-05", "2026-01-06", "2026-01-07"], "2026-01-07")
    ctx = journal.portfolio_context(con, prices, "2026-01-07")
    p = ctx["open_positions"][0]
    assert p["unrealised_pct"] == pytest.approx(10.0)
    assert ctx["sector_exposure_pct"]["IT"] == pytest.approx(11000 / ctx["equity_rupees"] * 100, abs=0.1)
    calls = journal.prior_calls(con, prices, "2026-01-07", "AAA")
    assert calls[0]["action"] == "BUY" and calls[0]["move_since_pct"] == pytest.approx(10.0)
    assert "still open" in calls[0]["trade"]
