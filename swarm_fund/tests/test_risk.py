"""Risk officer vetoes and sizing."""
import math

import pytest

import config
import risk

IND = {"close": 1000.0, "atr14": 20.0, "turnover_cr_20d": 100.0}


def book(**kw):
    b = dict(equity=500_000.0, available_cash=500_000.0, open_symbols=set(), pending_symbols=set(),
             sector_counts={}, nifty_above_200dma=True)
    b.update(kw)
    return risk.Book(**b)


def buy(sym="AAA", conf=70, sector="IT"):
    return {"symbol": sym, "sector": sector, "action": "BUY", "confidence": conf, "held": False}


def test_levels():
    lv = risk.levels(1000, 20)
    assert lv == pytest.approx({"entry_low": 995, "entry_high": 1010, "stop": 960, "target": 1080})


def test_size_by_risk_weight_and_cash():
    c = config.COST_PER_SIDE
    loss = 1010 * (1 + c) - 960 * (1 - c)
    qty, binding = risk.size(500_000, 500_000, 1010, 960, weak_market=False)
    assert binding == "weight"          # risk allows ~96 shares, 10% weight allows 49
    assert qty == math.floor(50_000 / (1010 * (1 + c)))
    qty, binding = risk.size(500_000, 500_000, 1010, 800, weak_market=False)
    assert binding == "risk" and qty == math.floor(5_000 / (1010 * (1 + c) - 800 * (1 - c)))
    qty, binding = risk.size(500_000, 10_000, 1010, 960, weak_market=False)
    assert binding == "cash" and qty == math.floor(10_000 / (1010 * (1 + c)))
    assert loss > 0


def test_weak_market_halves_risk_budget():
    strong, _ = risk.size(500_000, 500_000, 1010, 800, weak_market=False)
    weak, _ = risk.size(500_000, 500_000, 1010, 800, weak_market=True)
    assert weak == pytest.approx(strong / 2, abs=1)


def test_approves_clean_buy():
    v = risk.review([buy()], {"AAA": IND}, book())["AAA"]
    assert v["status"] == "APPROVED" and v["qty"] > 0 and v["veto_reason"] is None


@pytest.mark.parametrize("kw, ind, reason", [
    ({}, {"turnover_cr_20d": 10.0}, "turnover"),
    ({}, {"atr14": 50.0}, "stop 10.0% below close"),
    ({"open_symbols": {"AAA"}}, {}, "already held"),
    ({"pending_symbols": {"AAA"}}, {}, "already held or pending"),
    ({"open_symbols": {f"S{i}" for i in range(8)}}, {}, "8 positions"),
    ({"sector_counts": {"IT": 2}}, {}, "2 positions already in IT"),
    ({"available_cash": 500.0}, {}, "not enough cash for one share"),
])
def test_vetoes(kw, ind, reason):
    v = risk.review([buy()], {"AAA": {**IND, **ind}}, book(**kw))["AAA"]
    assert v["status"] == "VETOED"
    assert reason in v["veto_reason"]


def test_low_confidence_veto():
    v = risk.review([buy(conf=59)], {"AAA": IND}, book())["AAA"]
    assert v["status"] == "VETOED" and "confidence 59" in v["veto_reason"]


def test_daily_cap_takes_highest_confidence_first():
    syms = ["A", "B", "C", "D"]
    decisions = [buy(s, conf=c, sector=f"S{s}") for s, c in zip(syms, (61, 90, 75, 80))]
    v = risk.review(decisions, {s: IND for s in syms}, book())
    assert [s for s in syms if v[s]["status"] == "APPROVED"] == ["B", "C", "D"]
    assert "daily BUY limit" in v["A"]["veto_reason"]


def test_sector_cap_counts_todays_approvals():
    decisions = [buy("A", 90, "IT"), buy("B", 80, "IT")]
    v = risk.review(decisions, {"A": IND, "B": IND}, book(sector_counts={"IT": 1}))
    assert v["A"]["status"] == "APPROVED" and v["B"]["status"] == "VETOED"


def test_cash_is_consumed_by_earlier_approvals():
    # 50,000 buys A's 49 shares (about 49,564 with costs) and leaves less than one more share
    decisions = [buy("A", 90, "X"), buy("B", 80, "Y")]
    v = risk.review(decisions, {"A": IND, "B": IND}, book(available_cash=50_000.0))
    assert v["A"]["status"] == "APPROVED"
    assert v["B"]["status"] == "VETOED" and "cash" in v["B"]["veto_reason"]


def test_sell_and_hold_pass_through():
    d = [{"symbol": "H", "sector": "IT", "action": "SELL", "confidence": 70, "held": True},
         {"symbol": "Q", "sector": "IT", "action": "HOLD", "confidence": 50, "held": False}]
    v = risk.review(d, {"H": IND, "Q": IND}, book())
    assert v["H"]["status"] == "EXIT" and "next open" in v["H"]["risk_note"]
    assert v["Q"]["status"] == "HOLD"


# ---- earnings blackout (phase 2 rule)
@pytest.mark.parametrize("days, vetoed", [(0, True), (1, True), (3, True), (4, False), (-1, False)])
def test_earnings_blackout(days, vetoed):
    e = {"AAA": {"status": "known", "date": "2026-10-14", "trading_days_away": days}}
    v = risk.review([buy()], {"AAA": IND}, book(), earnings=e)["AAA"]
    assert (v["status"] == "VETOED") is vetoed
    if vetoed:
        assert "results due 2026-10-14" in v["veto_reason"]


def test_unknown_results_date_does_not_block_but_is_noted():
    v = risk.review([buy()], {"AAA": IND}, book(), earnings={"AAA": {"status": "unknown"}})["AAA"]
    assert v["status"] == "APPROVED" and "results date unknown" in v["risk_note"]
    v = risk.review([buy()], {"AAA": IND}, book(), earnings={})["AAA"]
    assert "results date unknown" in v["risk_note"]
