"""Each paper-fill rule, on hand-built bars."""
import pandas as pd
import pytest

import config
import db
import journal

C = config.COST_PER_SIDE
SIGNAL_DAY = "2026-01-05"
DAYS = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]


def setup(bars, symbol="AAA"):
    """bars: list of (open, high, low, close) for DAYS[0..]."""
    con = db.connect(":memory:")
    journal.ensure_start(con, DAYS[0], 100.0)
    db.set_meta(con, "last_advanced", SIGNAL_DAY)
    idx = pd.to_datetime(DAYS[: len(bars)])
    df = pd.DataFrame(bars, columns=["Open", "High", "Low", "Close"], index=idx)
    df["Volume"] = 1000.0
    return con, {symbol: df}, DAYS[: len(bars)]


def add_buy(con, entry_high=101.0, stop=95.0, target=110.0, qty=10, day=SIGNAL_DAY, symbol="AAA"):
    journal.place_order(con, None, day, {"status": "APPROVED", "symbol": symbol, "sector": "IT", "qty": qty,
                                          "entry_low": 99.5, "entry_high": entry_high, "stop": stop,
                                          "target": target})


def positions(con):
    return [dict(r) for r in con.execute("SELECT * FROM positions ORDER BY id")]


def orders(con):
    return [dict(r) for r in con.execute("SELECT * FROM orders ORDER BY id")]


def test_reservation_while_pending():
    con, prices, cal = setup([(100, 100, 100, 100)])
    add_buy(con, qty=10)
    assert journal.reserved(con) == pytest.approx(10 * 101 * (1 + C))
    assert journal.cash(con) == pytest.approx(config.STARTING_CAPITAL)


def test_fills_at_next_open_inside_zone():
    con, prices, cal = setup([(100, 100, 100, 100), (100.5, 102, 99, 101)])
    add_buy(con)
    journal.advance(con, prices, cal, cal[-1])
    p = positions(con)[0]
    assert p["entry_day"] == "2026-01-06" > p["signal_day"]
    assert p["entry_price"] == 100.5
    assert p["entry_cash"] == pytest.approx(10 * 100.5 * (1 + C))
    assert journal.reserved(con) == 0
    assert journal.cash(con) == pytest.approx(config.STARTING_CAPITAL - 10 * 100.5 * (1 + C))


def test_open_above_zone_expires():
    con, prices, cal = setup([(100, 100, 100, 100), (101.5, 103, 101, 102)])
    add_buy(con)
    journal.advance(con, prices, cal, cal[-1])
    assert positions(con) == []
    o = orders(con)[0]
    assert o["status"] == "EXPIRED" and "above entry top" in o["note"] and o["reserved"] == 0


def test_open_at_or_below_stop_expires():
    con, prices, cal = setup([(100, 100, 100, 100), (95, 96, 94, 95)])
    add_buy(con)
    journal.advance(con, prices, cal, cal[-1])
    assert positions(con) == [] and orders(con)[0]["status"] == "EXPIRED"


def test_order_only_fills_on_the_next_session():
    # no bar for the next session (2026-01-06 missing for this stock) -> expires, never fills later
    con, prices, cal = setup([(100, 100, 100, 100), (100, 101, 99, 100), (100, 101, 99, 100)])
    prices["AAA"] = prices["AAA"].drop(pd.Timestamp("2026-01-06"))
    add_buy(con)
    journal.advance(con, prices, cal, cal[-1])
    assert positions(con) == [] and orders(con)[0]["status"] == "EXPIRED"


def test_stop_touched_on_fill_day_exits_at_stop():
    con, prices, cal = setup([(100, 100, 100, 100), (100, 100.5, 94, 96)])
    add_buy(con)
    journal.advance(con, prices, cal, cal[-1])
    p = positions(con)[0]
    assert p["exit_reason"] == "STOP" and p["exit_price"] == 95.0 and p["exit_day"] == "2026-01-06"


def test_stop_wins_when_both_touched():
    con, prices, cal = setup([(100, 100, 100, 100), (100, 101, 99, 100), (100, 112, 94, 100)])
    add_buy(con)
    journal.advance(con, prices, cal, cal[-1])
    p = positions(con)[0]
    assert p["exit_reason"] == "STOP" and p["exit_price"] == 95.0


def test_gap_through_stop_exits_at_open():
    con, prices, cal = setup([(100, 100, 100, 100), (100, 101, 99, 100), (90, 92, 88, 91)])
    add_buy(con)
    journal.advance(con, prices, cal, cal[-1])
    p = positions(con)[0]
    assert p["exit_reason"] == "STOP_GAP" and p["exit_price"] == 90.0
    assert p["pnl"] == pytest.approx(10 * 90 * (1 - C) - 10 * 100 * (1 + C))


def test_target_hit_exits_at_target():
    con, prices, cal = setup([(100, 100, 100, 100), (100, 101, 99, 100), (104, 111, 103, 109)])
    add_buy(con)
    journal.advance(con, prices, cal, cal[-1])
    p = positions(con)[0]
    assert p["exit_reason"] == "TARGET" and p["exit_price"] == 110.0


def test_gap_above_target_exits_at_open():
    con, prices, cal = setup([(100, 100, 100, 100), (100, 101, 99, 100), (113, 115, 112, 114)])
    add_buy(con)
    journal.advance(con, prices, cal, cal[-1])
    p = positions(con)[0]
    assert p["exit_reason"] == "TARGET_GAP" and p["exit_price"] == 113.0


def test_sell_signal_exits_at_next_open():
    con, prices, cal = setup([(100, 100, 100, 100), (100, 101, 99, 100), (100, 101, 99, 100),
                              (102, 103, 96, 97)])
    add_buy(con)
    journal.advance(con, prices, cal, "2026-01-07")
    journal.place_order(con, None, "2026-01-07", {"status": "EXIT", "symbol": "AAA", "sector": "IT"})
    journal.advance(con, prices, cal, "2026-01-08")
    p = positions(con)[0]
    assert p["exit_reason"] == "SELL_SIGNAL" and p["exit_price"] == 102.0 and p["exit_day"] == "2026-01-08"


def test_advance_is_idempotent():
    con, prices, cal = setup([(100, 100, 100, 100), (100, 101, 99, 100), (90, 92, 88, 91)])
    add_buy(con)
    journal.advance(con, prices, cal, cal[-1])
    before = (positions(con), orders(con), journal.cash(con))
    assert journal.advance(con, prices, cal, cal[-1]) == []
    assert (positions(con), orders(con), journal.cash(con)) == before


def test_trade_stats_and_sample_note():
    closed = [{"pnl": 100, "entry_cash": 1000}, {"pnl": -50, "entry_cash": 1000}, {"pnl": 300, "entry_cash": 1000}]
    st = journal.trade_stats(closed)
    assert st["n"] == 3 and st["win_rate_pct"] == pytest.approx(200 / 3)
    assert st["avg_win_pct"] == pytest.approx(20.0) and st["avg_loss_pct"] == pytest.approx(-5.0)
    assert "far too few" in journal.sample_note(3)
