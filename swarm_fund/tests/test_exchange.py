"""Phase 2: exchange parsing, insider aggregation, rolling flows, the earnings
calendar, and the rule that a broken source reads as UNAVAILABLE, never as empty."""
import os
import subprocess
import sys
import textwrap
from datetime import date

import pytest

import config
import db
import exchange
import features
from fetch import Ctx

DAY = date(2026, 10, 9)   # a Friday


def ctx(offline=False, is_asof=False, day=DAY):
    return Ctx(con=db.connect(":memory:"), day=day, offline=offline, is_asof=is_asof)


# --------------------------------------------------------------- earnings
@pytest.mark.parametrize("when, days", [("2026-10-09", 0), ("2026-10-12", 1), ("2026-10-14", 3),
                                        ("2026-10-15", 4)])
def test_trading_days_skip_weekends(when, days):
    assert features.trading_days_between(DAY, date.fromisoformat(when)) == days


def test_trading_days_respect_holidays(monkeypatch):
    monkeypatch.setattr(config, "NSE_HOLIDAYS", ["2026-10-12"])
    assert features.trading_days_between(DAY, date(2026, 10, 14)) == 2


def test_earnings_info_states():
    res = {"status": "ok", "data": [{"date": "2026-07-20"}, {"date": "2026-10-14"}, {"date": "2027-01-20"}]}
    info = features.earnings_info(res, DAY)
    assert info == {"status": "known", "date": "2026-10-14", "trading_days_away": 3,
                    "last_results_meeting": "2026-07-20"}
    assert features.earnings_info({"status": "ok", "data": [{"date": "2026-07-20"}]}, DAY)["status"] == "none_announced"
    assert features.earnings_info({"status": "empty", "data": []}, DAY)["status"] == "none_announced"
    unk = features.earnings_info({"status": "unavailable", "error": "HTTP 403"}, DAY)
    assert unk["status"] == "unknown" and "403" in unk["reason"]


def test_results_calendar_is_left_out_of_asof_runs():
    res = exchange.results_calendar(ctx(is_asof=True), "TCS")
    assert res["status"] == "unavailable" and "--asof" in res["error"]
    assert features.earnings_info(res, DAY)["status"] == "unknown"


# ----------------------------------------------------------------- insider
def ins(disclosed, category, kind, value, name="X"):
    return {"disclosed": disclosed, "category": category, "type": kind, "value": value, "name": name, "mode": "m"}


def test_insider_aggregation_by_hand():
    rows = [ins("2026-09-01", "Promoter Group", "Sell", 50e7, "P1"),       # 50 cr
            ins("2026-09-02", "Promoters", "Buy", 10e7, "P2"),              # 10 cr
            ins("2026-09-03", "Director", "Buy", 2e7),                      # 2 cr
            ins("2026-09-04", "Employees/Designated Employees", "Sell", 1e7),
            ins("2026-09-05", "Promoters", "Pledge", 99e7),                 # not a trade
            ins("2026-05-01", "Promoters", "Sell", 500e7),                  # outside 90 days
            ins("2026-10-10", "Promoters", "Sell", 500e7)]                  # after the run date
    s = features.insider_summary(rows, DAY, window=90)
    assert s["promoters"] == {"bought_cr": 10.0, "sold_cr": 50.0, "net_value_cr": -40.0, "transactions": 2}
    assert s["other_insiders"] == {"bought_cr": 2.0, "sold_cr": 1.0, "net_value_cr": 1.0, "transactions": 2}
    assert s["largest_trade"]["name"] == "P1" and s["largest_trade"]["value_cr"] == 50.0
    assert s["non_trade_disclosures"] == 1


def test_clean_insider_parses_nse_rows():
    rows = [{"symbol": "TCS", "acqName": "A", "personCategory": "Promoters", "tdpTransactionType": "Sell",
             "acqMode": "Market Sale", "secVal": "1,23,45,678", "secAcq": "100", "date": "18-Feb-2026 19:06"},
            {"symbol": "INFY", "acqName": "B"}]
    out = exchange.clean_insider(rows, "TCS")
    assert out == [{"disclosed": "2026-02-18", "name": "A", "category": "Promoters", "type": "Sell",
                    "mode": "Market Sale", "value": 12345678.0, "quantity": 100.0}]


def test_dead_insider_endpoint_is_unavailable_not_empty(monkeypatch):
    monkeypatch.setattr(exchange, "_nse_json", lambda url: {"acqNameList": [], "data": []})
    monkeypatch.setattr(config, "HTTP_RETRIES", 0)
    c = ctx()
    canary = exchange.insider_canary(c)
    assert canary["status"] == "unavailable" and "0 rows" in canary["error"]
    res = exchange.insider_trades(c, "TCS", canary)
    assert res["status"] == "unavailable" and "health check" in res["error"]


def test_healthy_endpoint_with_no_trades_is_empty(monkeypatch):
    def fake(url):
        return {"data": [{"symbol": "OTHER"}] * 40} if "symbol=" not in url else {"data": []}
    monkeypatch.setattr(exchange, "_nse_json", fake)
    c = ctx()
    res = exchange.insider_trades(c, "TCS", exchange.insider_canary(c))
    assert res["status"] == "empty"


# ------------------------------------------------------------- announcements
def test_clean_announcements_filters_and_sorts():
    rows = [{"symbol": "TCS", "an_dt": "09-Oct-2026 18:43:52", "desc": "Press Release",
             "attchmntText": "t" * 500, "attchmntFile": "https://x/a.pdf"},
            {"symbol": "TCS", "an_dt": "01-Oct-2026 09:00:00", "desc": "Updates", "attchmntText": "u",
             "attchmntFile": ""},
            {"symbol": "TCS", "an_dt": "10-Oct-2026 08:00:00", "desc": "Future", "attchmntText": "", "attchmntFile": ""},
            {"symbol": "INFY", "an_dt": "05-Oct-2026 08:00:00", "desc": "Other company"}]
    out = exchange.clean_announcements(rows, "TCS", DAY)
    assert [a["subject"] for a in out] == ["Updates", "Press Release"]
    assert len(out[1]["description"]) == 300 and out[1]["link"] == "https://x/a.pdf"


def test_announcement_failure_is_visible(monkeypatch):
    def boom(url):
        raise RuntimeError("HTTP 403")
    monkeypatch.setattr(exchange, "_nse_json", boom)
    monkeypatch.setattr(config, "HTTP_RETRIES", 0)
    res = exchange.announcements(ctx(), "TCS")
    assert res["status"] == "unavailable" and "403" in res["error"]


# ------------------------------------------------------------------- flows
def test_clean_fii_dii():
    rows = [{"buyValue": "15626.53", "category": "DII", "date": "09-Oct-2026", "netValue": "4743.26",
             "sellValue": "10883.27"},
            {"buyValue": "10373.99", "category": "FII/FPI", "date": "09-Oct-2026", "netValue": "-3568.9",
             "sellValue": "13942.89"}]
    out = exchange.clean_fii_dii(rows)
    assert {r["category"]: r["net_cr"] for r in out} == {"DII": 4743.26, "FII": -3568.9}


def flows(days):
    out = []
    for i, d in enumerate(days):
        out += [{"date": d, "category": "FII", "net_cr": -100.0 * (i + 1)},
                {"date": d, "category": "DII", "net_cr": 50.0}]
    return out


def test_flow_summary_partial_history_is_stated():
    rows = flows(["2026-10-07", "2026-10-08", "2026-10-09"])
    s, gaps = features.flow_summary(rows, DAY)
    assert s["latest_session"] == "2026-10-09" and s["fii_net_cr_latest"] == -300
    assert s["fii_net_cr_5d"] == -600 and s["dii_net_cr_5d"] == 150 and s["sessions_in_5d_total"] == 3
    assert "only 3 of 5 sessions" in s["history_note"] and "only 3 of 20" in s["history_note"]
    assert gaps == []


def test_flow_summary_rolling_windows_and_no_lookahead():
    days = [f"2026-09-{d:02d}" for d in range(1, 31)] + ["2026-10-01", "2026-10-12"]
    s, _ = features.flow_summary(flows(days), DAY)                    # 2026-10-12 is after DAY
    assert s["latest_session"] == "2026-10-01"
    # last five stored sessions are items 27..31 -> -(27+28+29+30+31)*100
    assert s["fii_net_cr_5d"] == -14500 and s["sessions_in_20d_total"] == 20
    assert "latest stored session is 2026-10-01" in s["history_note"]


def test_flow_summary_without_history_is_a_gap():
    s, gaps = features.flow_summary([], DAY)
    assert s is None and gaps


def test_fii_dii_accumulates_in_sqlite():
    c = ctx(offline=True)
    for d in (date(2026, 10, 7), date(2026, 10, 8), date(2026, 10, 9)):
        c.day = d
        exchange.fii_dii(c)
    stored = exchange.stored_flows(c.con, "2026-10-09")
    assert len(stored) == 6
    s, _ = features.flow_summary(stored, DAY)
    assert s["sessions_in_5d_total"] == 3


# ---------------------------------------------------------- offline isolation
def test_offline_run_makes_no_network_calls(tmp_path):
    """Run the whole pipeline offline in a fresh interpreter with sockets disabled."""
    code = textwrap.dedent(f"""
        import socket, sys
        def blocked(*a, **k):
            raise RuntimeError("network call attempted in offline mode")
        socket.socket.connect = blocked
        socket.create_connection = blocked
        sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})
        import pipeline
        pipeline.run({str(tmp_path / 'o.db')!r}, offline=True, asof="2026-10-09")
        for mod in ("nsepython", "yfinance", "anthropic"):
            assert mod not in sys.modules, mod + " was imported in offline mode"
        print("OFFLINE-OK")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                         env={**os.environ, "ANTHROPIC_API_KEY": "should-be-ignored-offline"})
    assert "OFFLINE-OK" in out.stdout, out.stderr[-2000:]
