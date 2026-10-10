"""Console always; Telegram when TELEGRAM_TOKEN and TELEGRAM_CHAT_ID are set.
Every message is marked as paper. Plain text only, so nothing needs escaping."""
import requests

import config

TAG = "[PAPER]"
RULE_BASED = "RULE-BASED AGENTS - NO LLM WAS CALLED"


def fmt_money(x):
    return f"{x:,.2f}"


def buy_message(sig, decision, links, rule_based=False):
    lines = []
    if rule_based:
        lines.append(f"!! {RULE_BASED} !!")
    lines += [
        f"{TAG} BUY {sig['symbol']} ({sig['sector']})",
        f"Entry zone {fmt_money(sig['entry_low'])} - {fmt_money(sig['entry_high'])}",
        f"Stop {fmt_money(sig['stop'])} | Target {fmt_money(sig['target'])}",
        f"Qty {sig['qty']} ({sig.get('weight_pct', 0):.1f}% of portfolio) | Confidence {sig['confidence']}",
        "Thesis:",
    ]
    lines += [f"  - {t}" for t in decision.get("thesis", [])]
    lines.append(f"Kill condition: {decision.get('kill_condition') or '(none given)'}")
    lines.append(f"Risk officer: {sig.get('risk_note') or 'ok'}")
    if links:
        lines.append("Sources:")
        lines += [f"  {u}" for u in links[:4]]
    lines.append("Place by hand only if you agree. This tool never places orders.")
    return "\n".join(lines)


def sell_message(sig, decision, rule_based=False):
    lines = [f"!! {RULE_BASED} !!"] if rule_based else []
    lines += [f"{TAG} SELL {sig['symbol']} - exit at next open (confidence {sig['confidence']})", "Reasoning:"]
    lines += [f"  - {t}" for t in decision.get("thesis", [])] or ["  - (no reasoning given)"]
    return "\n".join(lines)


def events_message(events):
    if not events:
        return None
    return "\n".join([f"{TAG} Paper portfolio updates:"] + [f"  {e['day']} {e['message']}" for e in events])


def no_trade_message(day, n_reviewed, vetoed, rule_based=False):
    prefix = f"!! {RULE_BASED} !!\n" if rule_based else ""
    extra = f" ({len(vetoed)} BUY call(s) blocked by the risk officer)" if vetoed else ""
    return f"{prefix}{TAG} No trade today ({day}). Reviewed {n_reviewed} stock(s){extra}."


def send(text):
    print("\n" + text + "\n")
    if not (config.TELEGRAM_TOKEN and config.TELEGRAM_CHAT_ID):
        return False
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    ok = True
    for chunk in [text[i:i + 3900] for i in range(0, len(text), 3900)]:
        try:
            r = requests.post(url, json={"chat_id": config.TELEGRAM_CHAT_ID, "text": chunk,
                                         "disable_web_page_preview": True}, timeout=config.HTTP_TIMEOUT)
            if r.status_code != 200:
                print(f"  [telegram] send failed: HTTP {r.status_code} {r.text[:200]}")
                ok = False
        except requests.RequestException as e:
            print(f"  [telegram] send failed: {e}")
            ok = False
    return ok
