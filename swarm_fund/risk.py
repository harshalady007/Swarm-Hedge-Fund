"""The risk officer. Plain code with the last word: the LLM supplies only an action
and a confidence number; levels, size and vetoes are decided here."""
import math
from dataclasses import dataclass, field

import config


@dataclass
class Book:
    """What the risk officer needs to know about the paper portfolio right now."""
    equity: float
    available_cash: float                 # cash minus reservations for pending orders
    open_symbols: set = field(default_factory=set)
    pending_symbols: set = field(default_factory=set)
    sector_counts: dict = field(default_factory=dict)   # open + pending, by sector
    nifty_above_200dma: bool | None = True

    @property
    def slots_used(self) -> int:
        return len(self.open_symbols | self.pending_symbols)


def levels(close: float, atr: float) -> dict:
    return {
        "entry_low": close * (1 - config.ENTRY_ZONE_BELOW),
        "entry_high": close * (1 + config.ENTRY_ZONE_ABOVE),
        "stop": close - config.STOP_ATR * atr,
        "target": close + config.TARGET_ATR * atr,
    }


def size(equity, available_cash, entry_high, stop, weak_market):
    """Shares such that a stop-out costs RISK_PER_TRADE of equity (halved in a weak
    market), then capped at MAX_POSITION_WEIGHT of equity and by available cash.
    Returns (qty, binding_constraint)."""
    c = config.COST_PER_SIDE
    buy_cost = entry_high * (1 + c)                  # worst-case fill inside the zone
    loss_per_share = buy_cost - stop * (1 - c)
    budget = equity * config.RISK_PER_TRADE * (config.WEAK_MARKET_RISK_FACTOR if weak_market else 1.0)
    caps = {
        "risk": math.floor(budget / loss_per_share) if loss_per_share > 0 else 0,
        "weight": math.floor(equity * config.MAX_POSITION_WEIGHT / buy_cost),
        "cash": math.floor(max(available_cash, 0.0) / buy_cost),
    }
    binding = min(caps, key=caps.get)
    return max(caps[binding], 0), binding


def earnings_check(earnings):
    """earnings: {"status": "known"|"none_announced"|"unknown", "date", "trading_days_away"}.
    Returns (veto_reason | None, note | None)."""
    status = (earnings or {}).get("status", "unknown")
    if status == "known":
        days = earnings.get("trading_days_away")
        if days is not None and 0 <= days <= config.EARNINGS_BLACKOUT_DAYS:
            return (f"results due {earnings.get('date')} ({days} trading day(s) away, "
                    f"blackout is {config.EARNINGS_BLACKOUT_DAYS})"), None
        return None, f"next results {earnings.get('date')}"
    if status == "none_announced":
        return None, "no results date announced on the NSE calendar"
    return None, "results date unknown - check by hand"


def review(decisions, indicators, book: Book, earnings=None):
    """Apply the rules to the manager's decisions.

    decisions: list of {"symbol", "sector", "action", "confidence", "held"}.
    Returns a list of verdict dicts in the same symbols, BUYs judged highest
    confidence first.
    """
    earnings = earnings or {}
    weak = book.nifty_above_200dma is False
    approved_today = 0
    slots = book.slots_used
    sectors = dict(book.sector_counts)
    cash = book.available_cash
    verdicts = {}

    buys = sorted((d for d in decisions if d["action"] == "BUY"),
                  key=lambda d: (-d.get("confidence", 0), d["symbol"]))
    for d in buys:
        sym, ind = d["symbol"], indicators[d["symbol"]]
        close = ind["close"]
        lv = levels(close, ind["atr14"])
        reasons = []
        if d.get("confidence", 0) < config.MIN_CONFIDENCE:
            reasons.append(f"confidence {d.get('confidence', 0)} below {config.MIN_CONFIDENCE}")
        if sym in book.open_symbols or sym in book.pending_symbols:
            reasons.append("already held or pending")
        if slots >= config.MAX_POSITIONS:
            reasons.append(f"{config.MAX_POSITIONS} positions already open or pending")
        if sectors.get(d["sector"], 0) >= config.MAX_PER_SECTOR:
            reasons.append(f"{config.MAX_PER_SECTOR} positions already in {d['sector']}")
        stop_dist = (close - lv["stop"]) / close
        if stop_dist > config.MAX_STOP_DISTANCE:
            reasons.append(f"stop {stop_dist:.1%} below close exceeds {config.MAX_STOP_DISTANCE:.0%}")
        if not (ind["turnover_cr_20d"] >= config.MIN_TURNOVER_CR):
            reasons.append(f"turnover {ind['turnover_cr_20d']:.1f} cr below {config.MIN_TURNOVER_CR:.0f} cr")
        e_veto, e_note = earnings_check(earnings.get(sym))
        if e_veto:
            reasons.append(e_veto)
        qty, binding = size(book.equity, cash, lv["entry_high"], lv["stop"], weak)
        if qty < 1:
            reasons.append("not enough cash for one share" if binding == "cash"
                           else f"{binding} limit allows less than one share")
        if not reasons and approved_today >= config.MAX_BUYS_PER_DAY:
            reasons.append(f"daily BUY limit ({config.MAX_BUYS_PER_DAY}) reached")

        notes = []
        if weak:
            notes.append("NIFTY below 200-day average: risk budget halved")
        if e_note:
            notes.append(e_note)
        if reasons:
            verdicts[sym] = {"status": "VETOED", "veto_reason": "; ".join(reasons),
                             "risk_note": "; ".join(notes), **lv, "qty": 0}
            continue
        buy_cash = qty * lv["entry_high"] * (1 + config.COST_PER_SIDE)
        notes.insert(0, f"size bound by {binding}; {buy_cash / book.equity:.1%} of equity; "
                        f"loss at stop about {qty * (lv['entry_high'] - lv['stop']):,.0f} rupees")
        verdicts[sym] = {"status": "APPROVED", "veto_reason": None, "risk_note": "; ".join(notes),
                         **lv, "qty": qty, "weight_pct": buy_cash / book.equity * 100}
        approved_today += 1
        slots += 1
        sectors[d["sector"]] = sectors.get(d["sector"], 0) + 1
        cash -= buy_cash

    for d in decisions:
        if d["symbol"] in verdicts:
            continue
        if d["action"] == "SELL":
            verdicts[d["symbol"]] = {"status": "EXIT", "veto_reason": None,
                                     "risk_note": "exit at next open", "qty": None}
        else:
            verdicts[d["symbol"]] = {"status": "HOLD", "veto_reason": None, "risk_note": "", "qty": None}
    return verdicts
