"""Every tunable lives here. Secrets come from environment variables only."""
import os

# ---------------------------------------------------------------- universe
# NSE symbol -> sector. Yahoo needs a ".NS" suffix (see yahoo_ticker()).
UNIVERSE = {
    "RELIANCE": "Energy", "ONGC": "Energy", "COALINDIA": "Energy",
    "NTPC": "Utilities", "POWERGRID": "Utilities",
    "HDFCBANK": "Financials", "ICICIBANK": "Financials", "SBIN": "Financials",
    "KOTAKBANK": "Financials", "AXISBANK": "Financials", "BAJFINANCE": "Financials",
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "ITC": "Consumer", "HINDUNILVR": "Consumer", "NESTLEIND": "Consumer",
    "TITAN": "Consumer", "ASIANPAINT": "Consumer",
    "MARUTI": "Auto", "M&M": "Auto",
    "SUNPHARMA": "Pharma",
    "LT": "Industrials", "ADANIPORTS": "Industrials", "ULTRACEMCO": "Industrials",
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals",
    "BHARTIARTL": "Telecom",
}
# Used for news searches and messages.
COMPANY_NAMES = {
    "RELIANCE": "Reliance Industries", "ONGC": "ONGC", "COALINDIA": "Coal India",
    "NTPC": "NTPC", "POWERGRID": "Power Grid Corporation",
    "HDFCBANK": "HDFC Bank", "ICICIBANK": "ICICI Bank", "SBIN": "State Bank of India",
    "KOTAKBANK": "Kotak Mahindra Bank", "AXISBANK": "Axis Bank", "BAJFINANCE": "Bajaj Finance",
    "TCS": "Tata Consultancy Services", "INFY": "Infosys", "WIPRO": "Wipro",
    "HCLTECH": "HCLTech", "TECHM": "Tech Mahindra",
    "ITC": "ITC Limited", "HINDUNILVR": "Hindustan Unilever", "NESTLEIND": "Nestle India",
    "TITAN": "Titan Company", "ASIANPAINT": "Asian Paints",
    "MARUTI": "Maruti Suzuki", "M&M": "Mahindra & Mahindra",
    "SUNPHARMA": "Sun Pharmaceutical", "LT": "Larsen & Toubro", "ADANIPORTS": "Adani Ports",
    "ULTRACEMCO": "UltraTech Cement", "TATASTEEL": "Tata Steel", "JSWSTEEL": "JSW Steel",
    "BHARTIARTL": "Bharti Airtel",
}
BENCHMARK = "^NSEI"          # NIFTY 50 on Yahoo
VIX = "^INDIAVIX"            # India VIX on Yahoo


def yahoo_ticker(symbol: str) -> str:
    """NSE stocks need ".NS"; indices (^...) and full Yahoo tickers (X.NS) pass through."""
    return symbol if symbol.startswith("^") or "." in symbol else f"{symbol}.NS"


def sector_index_tickers():
    return sorted({t for ts in SECTOR_INDEX_CANDIDATES.values() for t in ts})


# Yahoo sector index candidates, tried in order. Each is checked at runtime for
# enough history; when none qualifies the code falls back to an equal-weighted
# basket of the sector's other members in UNIVERSE. Checked 2026-10-10 from a
# cloud host: only ^CNXIT, ^NSEBANK and ^CNXPHARMA returned a year of history;
# the others returned a single quote row, which the runtime check rejects.
SECTOR_INDEX_CANDIDATES = {
    "Energy": ["^CNXENERGY"],
    "Utilities": [],
    "Financials": ["NIFTY_FIN_SERVICE.NS", "^NSEBANK"],  # NIFTY BANK is a proxy
    "IT": ["^CNXIT"],
    "Consumer": ["^CNXFMCG"],                           # FMCG is a proxy
    "Auto": ["^CNXAUTO"],
    "Pharma": ["^CNXPHARMA"],
    "Industrials": ["^CNXINFRA"],
    "Metals": ["^CNXMETAL"],
    "Telecom": [],
}
SECTOR_INDEX_MIN_BARS = 70        # need 60-day returns plus slack
SECTOR_INDEX_MAX_STALE_DAYS = 5   # last bar must be this close to the run day

# ---------------------------------------------------------------- data
HISTORY_YEARS = 2
MIN_BARS = 210                    # skip stocks with less history than this
NEWS_DAYS = 7
NEWS_MAX_HEADLINES = 8
HTTP_TIMEOUT = 15                 # seconds, every fetch
HTTP_RETRIES = 1                  # one retry after the first failure
EXCHANGE_PAUSE_SECONDS = 1.0      # pause between calls to exchange endpoints
LIVE_CLOSE_GUARD_IST = "15:45"    # a live run before this drops today's (partial) bar

# Phase 1
TECH_DAILY_BARS = 60
TECH_WEEKLY_CLOSES = 52
QUARTERS_TO_SHOW = 6
# A quarter is treated as public this many days after it ends (SEBI allows 45
# days for Q1-Q3 and 60 for Q4); used to stop --asof runs seeing later results.
RESULTS_PUBLICATION_LAG_DAYS = 60
TRACK_RECORD_CLOSED_TRADES = 20
PRIOR_CALLS_SHOWN = 5

# Phase 2
ANNOUNCEMENT_DAYS = 30
ANNOUNCEMENTS_SHOWN = 12
INSIDER_DAYS = 90
INSIDER_CANARY_DAYS = 14          # a market-wide window that cannot honestly be empty
EARNINGS_BLACKOUT_DAYS = 3        # trading days
# Trading days are counted as weekdays minus these dates. Add NSE's published
# holidays for the year (YYYY-MM-DD); leaving it empty can make a results date
# look one session further away than it is around a holiday.
NSE_HOLIDAYS: list[str] = []
FLOW_WINDOWS = (5, 20)

# Phase 3
DELIVERY_AVG_DAYS = 20
DELIVERY_BACKFILL_DAYS = 20       # bhavcopy days fetched once to seed history
CONCALL_LOOKBACK_DAYS = 100
CONCALL_MAX_PDF_BYTES = 15_000_000
CONCALL_MAX_CHARS_TO_MODEL = 60_000
ARTICLES_PER_STOCK = 3
ARTICLE_MAX_WORDS = 400
JINA_MIN_INTERVAL_SECONDS = 3.2   # ~19 requests a minute, under the free 20/min

# ---------------------------------------------------------------- screen
SHORTLIST_SIZE = 5                # top-N not-held stocks the swarm reviews

# ---------------------------------------------------------------- risk officer
MIN_CONFIDENCE = 60
MAX_POSITIONS = 8
MAX_PER_SECTOR = 2
MAX_BUYS_PER_DAY = 3
ENTRY_ZONE_BELOW = 0.005          # close - 0.5%
ENTRY_ZONE_ABOVE = 0.01           # close + 1%
STOP_ATR = 2.0
TARGET_ATR = 4.0
MAX_STOP_DISTANCE = 0.08          # veto when stop is >8% below close
MIN_TURNOVER_CR = 25.0            # 20-day average daily turnover, crore rupees
RISK_PER_TRADE = 0.01             # loss at stop as a share of equity
MAX_POSITION_WEIGHT = 0.10
WEAK_MARKET_RISK_FACTOR = 0.5     # applied when NIFTY < its 200-day average

# ---------------------------------------------------------------- paper portfolio
STARTING_CAPITAL = 500_000.0
COST_PER_SIDE = 0.0015

# ---------------------------------------------------------------- secrets
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------- agents / LLM
# Which LLM service runs the swarm: "anthropic" or "sarvam". When unset, the
# provider whose key is present is used (Anthropic first if both are set).
LLM_PROVIDERS = ("anthropic", "sarvam")
LLM_PROVIDER = os.environ.get("SWARM_LLM_PROVIDER", "").strip().lower() or (
    "anthropic" if ANTHROPIC_API_KEY else ("sarvam" if SARVAM_API_KEY else "anthropic"))
_DEFAULT_MODELS = {"anthropic": ("claude-haiku-5-5", "claude-sonnet-5-5"),
                   "sarvam": ("sarvam-105b", "sarvam-105b")}
_defaults = _DEFAULT_MODELS.get(LLM_PROVIDER, _DEFAULT_MODELS["anthropic"])
ANALYST_MODEL = os.environ.get("SWARM_ANALYST_MODEL", _defaults[0])
MANAGER_MODEL = os.environ.get("SWARM_MANAGER_MODEL", _defaults[1])
# Effort: low | medium | high | max. Anthropic: output_config.effort. Sarvam:
# reasoning_effort (medium is Sarvam's own default, so it is simply not sent).
ANALYST_EFFORT = os.environ.get("SWARM_ANALYST_EFFORT", "low")
MANAGER_EFFORT = os.environ.get("SWARM_MANAGER_EFFORT", "medium")
ANALYST_MAX_TOKENS = 4000         # thinking/reasoning counts toward this, so leave room
MANAGER_MAX_TOKENS = 8000
LLM_TIMEOUT = 90
# Anthropic only: on a safety-classifier refusal the API reruns the manager call
# on a fallback model ("fallbacks": "default"). SWARM_MANAGER_FALLBACK=0 turns it off.
MANAGER_REFUSAL_FALLBACK = os.environ.get("SWARM_MANAGER_FALLBACK", "1") != "0"
PAYLOAD_TOKEN_CEILING = 3000      # rough per-agent data budget (chars / 4)

# Sarvam: OpenAI-style chat completions. sarvam-105b is documented on the v1
# route (v2 needs beta access). Temperature 0.2 is v1's own default; kept low
# so reruns of the same brief give similar answers.
SARVAM_URL = os.environ.get("SARVAM_URL", "https://api.sarvam.ai/v1/chat/completions")
SARVAM_TEMPERATURE = float(os.environ.get("SARVAM_TEMPERATURE", "0.2"))
# Sarvam does not publish sarvam-105b's context window, so concall text sent for
# summary is capped lower than for Claude (about 10k tokens instead of 15k).
SARVAM_MAX_DOC_CHARS = 40_000

DEFAULT_DB = os.environ.get("SWARM_DB", "journal.db")
