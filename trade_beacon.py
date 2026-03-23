"""
Trade Beacon v2.2.2-SAFE - Forex Signal Generator (INSTITUTIONAL GRADE)
MULTI-MODE EDITION: Generates Aggressive + Conservative signals simultaneously
with tier-based selective enhancement (sentiment/backtest only on top-tier)

CHANGES v2.2.2-SAFE:
- NEW PAIRS: GBPAUD and GBPCAD added (GBP crosses — model calibrated to GBP behavior)
  Research: BIS 2025 top-25 traded pairs, 80-120 pip daily range, BoE-driven trends
  GBPAUD: BoE vs RBA divergence — sustained directional momentum, not USD-sensitive
  GBPCAD: BoE vs oil/BoC — commodity-linked trends, strongest in US session
- EURUSD permanently blocked — 7 trades, 0% WR, -177 pips, FinBERT gate failed to help
- 1-HOUR TREND FILTER: Multi-timeframe confirmation (Elder Triple Screen principle)
  Score 55-61 passes ONLY if 1h EMA12>EMA26>EMA50 agrees with 15m direction
  Score 62+: 1h disagreement BLOCKS even high-scoring signals (raises all signal quality)
  Score 62+ with inconclusive 1h: passes on 15m strength alone (original behavior)
  Research: MTF confirmation raises WR ~15-20% vs single-timeframe (Murphy 2002)
- THRESHOLD LOWERED: 62->55 (aggressive), 62->57 (conservative)
  1-hour filter replaces raw score threshold as the quality gate
  Unlocks 2-4 more signals/day on trending markets without letting noise through
- B_min_score: 62->55 to match new threshold (Tier C still blocked)
- max_open_positions: 3->5 (more pairs need higher capacity)
- GBPCAD/GBPAUD: pure technical, no FinBERT, no USD restriction
  Calendar blackout still protects against RBA/BoC high-impact events
"""

import logging
import sys
import json
import os
import hashlib
import copy
import time
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf
import tempfile as _tempfile
yf.set_tz_cache_location(_tempfile.mkdtemp(prefix="yf_tz_"))
import requests
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

from performance_tracker import PerformanceTracker

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("trade-beacon")

# ── Mode guard ────────────────────────────────────────────────────────────────
SIGNAL_ONLY_MODE = True
if not SIGNAL_ONLY_MODE:
    raise RuntimeError("Execution logic disabled in signal-only mode")
log.info("Signal-only mode validated")

# ── Constants ─────────────────────────────────────────────────────────────────
PAIRS = [
    "USDJPY=X", "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X",
    "USDCAD=X", "USDCHF=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X",
    "GBPAUD=X", "GBPCAD=X",   # v2.2.0: GBP crosses — BIS top-25, BoE-driven trends
]
SPREADS = {
    "USDJPY": 0.002,  "EURUSD": 0.00015, "GBPUSD": 0.0002,
    "AUDUSD": 0.00018,"NZDUSD": 0.0002,  "USDCAD": 0.0002,
    "USDCHF": 0.0002, "EURJPY": 0.002,   "GBPJPY": 0.003,
    "EURGBP": 0.00015,"GBPAUD": 0.0004,  "GBPCAD": 0.0004,
}
CORRELATED_PAIRS = [
    {"EURUSD=X", "GBPUSD=X"}, {"EURUSD=X", "EURGBP=X"},
    {"GBPUSD=X", "EURGBP=X"}, {"USDJPY=X", "EURJPY=X"},
    {"USDJPY=X", "GBPJPY=X"}, {"EURJPY=X", "GBPJPY=X"},
    # v2.2.0: GBP crosses correlated with GBPUSD (shared GBP leg)
    {"GBPUSD=X", "GBPAUD=X"}, {"GBPUSD=X", "GBPCAD=X"},
    {"GBPAUD=X", "GBPCAD=X"}, {"GBPJPY=X", "GBPAUD=X"},
]
INTERVAL = "15m"
LOOKBACK = "14d"
MIN_ROWS = 220

# Permanently blocked — data confirmed no edge regardless of conditions
_PERMANENTLY_BLOCKED_PAIRS = frozenset(["EURUSD", "EURGBP"])

_USD_RESTRICTED_PAIRS = frozenset(["AUDUSD","USDCAD","USDCHF","USDJPY","NZDUSD","EURJPY"])
# Note: EURUSD removed — it is in _PERMANENTLY_BLOCKED_PAIRS (0% WR, 7 trades, -177 pips)

# ── HuggingFace FinBERT ────────────────────────────────────────────────────────
HF_API_KEY        = os.getenv("HF_API_KEY", "")
HF_PRIMARY_MODEL  = "ProsusAI/finbert"
HF_FALLBACK_MODEL = "yiyanghkust/finbert-tone"
HF_API_BASE       = "https://router.huggingface.co/hf-inference/models"

HF_LABEL_MAP: Dict[str, float] = {
    "positive": 1.0, "Positive": 1.0,
    "negative":-1.0, "Negative":-1.0,
    "neutral":  0.0, "Neutral":  0.0,
    "LABEL_0":  0.0,
    "LABEL_1":  1.0,
    "LABEL_2": -1.0,
}

# ── Browserless ───────────────────────────────────────────────────────────────
BROWSERLESS_TOKEN = os.getenv("BROWSERLESS_TOKEN", "")

# ── Global state ──────────────────────────────────────────────────────────────
CONFIG              = None
MODE                = None
USE_SENTIMENT       = False
SETTINGS            = None
PERFORMANCE_TRACKER = None

# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def price_to_pips(pair: str, price_diff: float) -> float:
    return abs(price_diff) / (0.01 if "JPY" in pair else 0.0001)

def ensure_series(data):
    return data.iloc[:, 0].squeeze() if isinstance(data, pd.DataFrame) else data.squeeze()

def retry_with_backoff(max_retries=3, backoff_factor=10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "rate limit" in str(e).lower() or "429" in str(e):
                        if attempt < max_retries - 1:
                            wait = (2 ** attempt) * backoff_factor
                            log.warning(f"Rate limited, waiting {wait}s...")
                            time.sleep(wait)
                        else:
                            raise
                    else:
                        raise
        return wrapper
    return decorator

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

def _default_config() -> Dict:
    return {
        "mode": "conservative",
        "use_sentiment": False,
        "settings": {
            "aggressive": {
                "threshold": 55, "min_adx": 15,          # v2.2.0: 62->55
                "rsi_oversold": 30, "rsi_overbought": 70,
                "min_risk_reward": 2.5, "atr_stop_multiplier": 2.5,
                "atr_target_multiplier": 7.0, "max_correlated_signals": 2,
            },
            "conservative": {
                "threshold": 57, "min_adx": 17,          # v2.2.0: 62->57
                "rsi_oversold": 30, "rsi_overbought": 70,
                "min_risk_reward": 2.5, "atr_stop_multiplier": 2.5,
                "atr_target_multiplier": 7.0, "max_correlated_signals": 1,
            },
        },
        "tiers": {"A_plus_min_score": 80, "A_min_score": 72, "B_min_score": 55},  # v2.2.0: 62->55
        "advanced": {
            "enable_session_filtering": True,
            "enable_correlation_filter": True,
            "cache_ttl_minutes": 5,
            "parallel_workers": 4,
            "atr_stop_multiplier": 2.5,
            "atr_target_multiplier": 7.0,
            "session_bonuses": {
                "ASIAN":    {"JPY_pairs": 3, "AUD_NZD_pairs": 3, "other": 0},
                "EUROPEAN": {"EUR_GBP_pairs": 3, "EUR_GBP_crosses": 3, "GBP_crosses": 3, "other": 0},
                "OVERLAP":  {"all_major_pairs": 0},
                "US":       {"USD_majors": 3, "GBP_crosses": 3, "other": 0},
                "LATE_US":  {"all_major_pairs": 0},
            },
            "session_thresholds": {
                "ASIAN": 999, "EUROPEAN": 48, "US": 48, "LATE_US": 60, "OVERLAP": 999,
            },
            "pair_limits": {
                "GBPUSD": 5, "GBPJPY": 1, "GBPAUD": 3, "GBPCAD": 3,
                "NZDUSD": 1, "EURGBP": 0, "EURUSD": 0, "default": 3,
            },
            "validation": {
                "max_signal_age_seconds": 900,
                "session_expiry_minutes": {
                    "EUROPEAN": 240, "US": 240, "ASIAN": 15, "LATE_US": 15, "OVERLAP": 15,
                },
                "min_sl_pips": {"JPY_pairs": 20, "other": 12},
                "max_spread_ratio": 0.25, "max_sl_distance_pct": 0.02,
                "max_tp_distance_pct": 0.05, "require_direction": True,
                "reject_missing_pips": True,
            },
            "usd_pair_rules": {
                "enabled": True,
                "restricted_pairs": ["AUDUSD","USDCAD","USDCHF","USDJPY","NZDUSD","EURJPY"],
                "allowed_sessions": ["US"],
                "min_score": 65,
                "blackout_before_minutes": 45,
                "blackout_after_minutes": 30,
                "sentiment_gate": True,
                "sentiment_gate_threshold": 0.1,  # v2.2.1: raised from 0.0 — neutral not enough, need weak positive
            },
        },
        "risk_management": {
            "max_daily_risk_pips": 150, "max_open_positions": 5,   # v2.2.0: 3->5
            "stop_trading_on_drawdown_pips": 100,
            "equity_protection": {"enable": False, "max_consecutive_losses": 3, "pause_minutes_after_hit": 120},
        },
        "performance_tracking": {
            "enable": True, "history_file": "signal_state/signal_history.json",
            "idempotency": {"enabled": True}, "analytics": {"track_by_pair": True},
        },
        "performance_tuning": {
            "auto_adjust_thresholds": False, "min_trades_for_optimization": 50,
            "target_win_rate": 0.5, "optimization_inputs": {"min_expectancy": 5},
        },
    }

def load_config() -> Dict:
    config_path = Path("config.json")
    if not config_path.exists():
        log.warning("config.json not found, using defaults")
        return _default_config()
    with open(config_path) as f:
        config = json.load(f)
    if config.get("mode") not in ["aggressive", "conservative"]:
        config["mode"] = "conservative"
    log.info(f"Config loaded: mode={config['mode']}, sentiment={config.get('use_sentiment', False)}")
    return config

def validate_api_keys() -> bool:
    if not os.getenv("NEWSAPI_KEY") or not os.getenv("MARKETAUX_API_KEY"):
        log.warning("Sentiment disabled: NEWSAPI_KEY or MARKETAUX_API_KEY not set")
        return False
    try:
        r = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"country": "us", "pageSize": 1, "apiKey": os.getenv("NEWSAPI_KEY")},
            timeout=8,
        )
        if r.status_code == 401:
            log.warning("NewsAPI key invalid (401) — sentiment disabled")
            return False
        if r.status_code == 429:
            log.warning("NewsAPI rate limited (429) — proceeding anyway, will retry per-signal")
        log.info("NewsAPI validated")
    except Exception as e:
        log.warning(f"NewsAPI validation request failed ({e}) — proceeding with sentiment enabled")
    return True

CONFIG        = load_config()
MODE          = CONFIG["mode"]
USE_SENTIMENT = CONFIG.get("use_sentiment", False) and validate_api_keys()
SETTINGS      = CONFIG["settings"][MODE]
CACHE_TTL     = CONFIG.get("advanced", {}).get("cache_ttl_minutes", 5) * 60

if CONFIG.get("performance_tracking", {}).get("enable", True):
    try:
        PERFORMANCE_TRACKER = PerformanceTracker(
            history_file=CONFIG["performance_tracking"].get(
                "history_file", "signal_state/signal_history.json"
            )
        )
        log.info("Performance tracker initialized")
    except Exception as e:
        log.error(f"Tracker init failed: {e}")
        PERFORMANCE_TRACKER = None

# ═════════════════════════════════════════════════════════════════════════════
# MARKET DATA CACHE
# ═════════════════════════════════════════════════════════════════════════════

class MarketDataCache:
    def __init__(self, ttl: int = 300):
        self.ttl         = ttl
        self._cache      : Dict = {}
        self._timestamps : Dict = {}
        self._lock       = threading.Lock()

    def get(self, key: str) -> Optional[pd.DataFrame]:
        with self._lock:
            if key not in self._cache:
                return None
            if time.time() - self._timestamps.get(key, 0) > self.ttl:
                return None
            return self._cache[key]

    def set(self, key: str, value: pd.DataFrame):
        with self._lock:
            self._cache[key]      = value
            self._timestamps[key] = time.time()

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

_cache    = MarketDataCache(ttl=CACHE_TTL)
_cache_1h = MarketDataCache(ttl=3600)  # 1h trend cache — 60min TTL (1h candles don't change fast)
_yf_lock  = threading.Lock()           # Serialises yfinance calls — prevents SQLite cache bleed
                                       # between parallel threads (GBPAUD/GBPCAD/EURJPY identical
                                       # scores bug). Each pair's data is isolated.

# ═════════════════════════════════════════════════════════════════════════════
# ECONOMIC CALENDAR
# ═════════════════════════════════════════════════════════════════════════════

_FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_CALENDAR_CACHE  = Path("signal_state/economic_calendar_cache.json")
_BLACKOUT_BEFORE = 30
_BLACKOUT_AFTER  = 15

def _get_et_utc_offset(dt: datetime = None) -> int:
    d = (dt or datetime.now(timezone.utc))
    year = d.year
    mar1 = datetime(year, 3, 1)
    edt_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)
    nov1 = datetime(year, 11, 1)
    edt_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    if edt_start.date() <= d.date() < edt_end.date():
        return -4
    return -5

_CURRENCY_PAIRS: Dict[str, List[str]] = {
    "USD": ["EURUSD","GBPUSD","USDJPY","USDCAD","USDCHF","AUDUSD","NZDUSD"],
    "EUR": ["EURUSD","EURGBP","EURJPY"],
    "GBP": ["GBPUSD","EURGBP","GBPJPY","GBPAUD","GBPCAD"],   # v2.2.0: added GBP crosses
    "JPY": ["USDJPY","EURJPY","GBPJPY"],
    "AUD": ["AUDUSD","GBPAUD"],   # v2.2.0: RBA events now blackout GBPAUD
    "NZD": ["NZDUSD"],
    "CAD": ["USDCAD","GBPCAD"],   # v2.2.0: BoC events now blackout GBPCAD
    "CHF": ["USDCHF"],
    "US":  ["EURUSD","GBPUSD","USDJPY","USDCAD","USDCHF","AUDUSD","NZDUSD"],
    "EU":  ["EURUSD","EURGBP","EURJPY"],
    "GB":  ["GBPUSD","EURGBP","GBPJPY","GBPAUD","GBPCAD"],
    "UK":  ["GBPUSD","EURGBP","GBPJPY","GBPAUD","GBPCAD"],
    "JP":  ["USDJPY","EURJPY","GBPJPY"],
    "AU":  ["AUDUSD","GBPAUD"],   # v2.2.0: AU country code covers GBPAUD
    "NZ":  ["NZDUSD"],
    "CA":  ["USDCAD","GBPCAD"],   # v2.2.0: CA country code covers GBPCAD
    "CH":  ["USDCHF"],
    "CNY": [], "CN": [],
}

def _parse_ff_time(date_str: str, time_str: str) -> Optional[datetime]:
    import re as _re
    if date_str and 'T' in date_str:
        try:
            clean = date_str.replace('Z', '+00:00')
            tz_match = _re.search(r'([+-])(\d{2}):(\d{2})$', clean)
            if tz_match:
                sign, hh, mm = tz_match.group(1), tz_match.group(2), tz_match.group(3)
                offset_min = (int(hh) * 60 + int(mm)) * (1 if sign == '+' else -1)
                naive_str = clean[:19]
                dt_naive = datetime.strptime(naive_str, "%Y-%m-%dT%H:%M:%S")
                dt_utc = dt_naive.replace(tzinfo=timezone.utc) - timedelta(minutes=offset_min)
            else:
                dt_utc = datetime.fromisoformat(clean).astimezone(timezone.utc)
            return dt_utc
        except Exception:
            return None

    if not time_str or time_str.lower().strip() in ("tentative", "all day", ""):
        return None
    try:
        t = time_str.lower().replace(" ", "")
        m = _re.match(r"(\d+):(\d+)(am|pm)", t)
        if not m:
            return None
        hour, minute, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
        if ampm == "pm" and hour != 12: hour += 12
        elif ampm == "am" and hour == 12: hour = 0
        parsed_date = None
        for fmt in ("%Y-%m-%d", "%m-%d-%Y", "%m/%d/%Y", "%Y/%m/%d"):
            try:
                parsed_date = datetime.strptime(date_str, fmt); break
            except ValueError: continue
        if parsed_date is None: return None
        et_offset = _get_et_utc_offset(parsed_date)
        return parsed_date.replace(hour=hour, minute=minute, tzinfo=timezone.utc) - timedelta(hours=et_offset)
    except Exception:
        return None

def _load_economic_calendar() -> List[Dict]:
    _CALENDAR_CACHE.parent.mkdir(exist_ok=True)
    if _CALENDAR_CACHE.exists():
        try:
            cache     = json.loads(_CALENDAR_CACHE.read_text())
            cached_at = datetime.fromisoformat(cache.get("cached_at", "2000-01-01T00:00:00+00:00"))
            age_h     = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
            if age_h < 24:
                events = cache.get("events", [])
                log.info(f"Calendar: {len(events)} high-impact events from cache (age {age_h:.1f}h)")
                if len(events) == 0 and age_h > 1:
                    log.info("Calendar: cache has 0 events and is >1h old — re-fetching to verify")
                else:
                    return events
        except Exception as e:
            log.warning(f"Calendar cache read failed: {e} — fetching fresh")
    try:
        resp = requests.get(_FF_CALENDAR_URL, timeout=10, headers={"User-Agent": "TradeBeacon/2.1.7"})
        log.info(f"Calendar: FF HTTP {resp.status_code} — {len(resp.content)} bytes received")
        if resp.status_code != 200:
            log.warning(f"Calendar: FF returned {resp.status_code} — no news filtering")
            return []
        raw = resp.json()
        log.info(f"Calendar: {len(raw)} total events in FF feed (all impacts)")
        high_impact_all = [e for e in raw if e.get("impact","").lower() == "high"]
        log.info(f"Calendar: {len(high_impact_all)} high-impact events before currency filter")
        ff_countries = sorted(set(e.get("country","").upper() for e in high_impact_all))
        log.info(f"Calendar: FF high-impact country codes = {ff_countries}")
        if high_impact_all:
            sample = high_impact_all[0]
            log.info(f"Calendar: Sample event fields = {list(sample.keys())}")
            log.info(f"Calendar: Sample date='{sample.get('date','')}' time='{sample.get('time','')}' country='{sample.get('country','')}'")

        events = []
        _dropped_currency = 0
        _dropped_date     = 0
        for item in raw:
            if item.get("impact", "").lower() != "high": continue
            currency = item.get("country", "").upper()
            if currency not in _CURRENCY_PAIRS:
                _dropped_currency += 1; continue
            if not _CURRENCY_PAIRS[currency]: continue
            date_str = item.get("date", "")
            utc_time = _parse_ff_time(date_str, item.get("time", ""))
            if utc_time is None:
                parsed_date = None
                for fmt in ("%Y-%m-%d", "%m-%d-%Y", "%m/%d/%Y", "%Y/%m/%d"):
                    try:
                        parsed_date = datetime.strptime(date_str, fmt); break
                    except ValueError: continue
                if parsed_date is None:
                    _dropped_date += 1; continue
                utc_time = parsed_date.replace(hour=13, minute=30, tzinfo=timezone.utc)
            events.append({"title": item.get("title","Unknown"), "currency": currency, "utc_time": utc_time.isoformat()})

        if _dropped_currency > 0:
            log.info(f"Calendar: {_dropped_currency} events dropped — currency not in tracked pairs")
        if _dropped_date > 0:
            log.warning(f"Calendar: {_dropped_date} events dropped — could not parse date field")

        _CALENDAR_CACHE.write_text(json.dumps({"cached_at": datetime.now(timezone.utc).isoformat(), "source": _FF_CALENDAR_URL, "events": events}, indent=2))
        log.info(f"Calendar: fetched {len(events)} high-impact events from FF — cached")
        if len(events) == 0 and len(high_impact_all) > 0:
            log.warning(f"Calendar: 0 events loaded from {len(high_impact_all)} high-impact ({_dropped_currency} currency-miss, {_dropped_date} date-parse-fail)")
        elif len(events) == 0:
            log.info("Calendar: genuinely no high-impact events this week in FF feed")
        return events
    except Exception as e:
        log.warning(f"Calendar: fetch failed ({e}) — no news filtering this run")
        return []

def _is_calendar_blackout(pair: str, events: List[Dict]) -> Tuple[bool, str]:
    if not events: return False, ""
    now   = datetime.now(timezone.utc)
    clean = pair.replace("=X", "").upper()
    usd_rules = CONFIG.get("advanced", {}).get("usd_pair_rules", {})
    if usd_rules.get("enabled", False) and clean in _USD_RESTRICTED_PAIRS:
        bb = usd_rules.get("blackout_before_minutes", _BLACKOUT_BEFORE)
        ba = usd_rules.get("blackout_after_minutes",  _BLACKOUT_AFTER)
    else:
        bb = _BLACKOUT_BEFORE
        ba = _BLACKOUT_AFTER
    for ev in events:
        if clean not in _CURRENCY_PAIRS.get(ev.get("currency", ""), []): continue
        try:
            et = datetime.fromisoformat(ev["utc_time"])
        except Exception: continue
        if et - timedelta(minutes=bb) <= now <= et + timedelta(minutes=ba):
            mins  = int((et - now).total_seconds() / 60)
            label = (f"{ev['title']} ({ev['currency']}) in {mins}min" if mins >= 0
                     else f"{ev['title']} ({ev['currency']}) {abs(mins)}min ago")
            return True, label
    return False, ""

def _get_upcoming_events(hours: int = 4) -> List[Dict]:
    if not ECONOMIC_CALENDAR: return []
    now    = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours)
    result = []
    for ev in ECONOMIC_CALENDAR:
        try:
            t = datetime.fromisoformat(ev["utc_time"])
            if now - timedelta(minutes=15) <= t <= cutoff:
                mins = int((t - now).total_seconds() / 60)
                result.append({"title": ev["title"], "currency": ev["currency"],
                                "utc_time": t.strftime("%H:%M UTC"), "mins_to": mins, "passed": mins < 0})
        except Exception: pass
    return sorted(result, key=lambda x: x["mins_to"])

ECONOMIC_CALENDAR: List[Dict] = []
try:
    ECONOMIC_CALENDAR = _load_economic_calendar()
except Exception as e:
    log.warning(f"Economic calendar init failed: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# LIVE PRICE VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

_BROWSERLESS_URL  = "https://production-sfo.browserless.io/content"
_XRATES_URL       = "https://www.x-rates.com/calculator/?from={f}&to={t}&amount=1"
_PRICE_DIVERGENCE = 0.005
_PRICE_CACHE: Dict[str, Tuple[float, float]] = {}
_PRICE_CACHE_TTL  = 300

import re as _re_live
_XRATES_RE = _re_live.compile(r'ccOutputRslt[^>]*>([\d,.]+)')

def _fetch_xrates_price(pair: str) -> Optional[float]:
    if not BROWSERLESS_TOKEN: return None
    clean = pair.replace("=X", "").upper()
    if clean in _PRICE_CACHE:
        price, ts = _PRICE_CACHE[clean]
        if time.time() - ts < _PRICE_CACHE_TTL:
            log.debug(f"LivePrice: {clean} = {price:.5f} (cache)")
            return price
    if len(clean) != 6: return None
    f, t = clean[:3], clean[3:]
    try:
        resp = requests.post(f"{_BROWSERLESS_URL}?token={BROWSERLESS_TOKEN}",
                             json={"url": _XRATES_URL.format(f=f, t=t)}, timeout=15)
        if resp.status_code != 200: return None
        m = _XRATES_RE.search(resp.text)
        if not m: return None
        price = float(m.group(1).replace(",", ""))
        if price <= 0: return None
        _PRICE_CACHE[clean] = (price, time.time())
        log.info(f"LivePrice: {clean} = {price:.5f} (X-Rates via Browserless)")
        return price
    except Exception as e:
        log.warning(f"LivePrice: Browserless error for {clean}: {e}")
        return None

def _validate_signal_prices(signals: List[Dict]) -> List[Dict]:
    if not BROWSERLESS_TOKEN or not signals: return signals
    validated = []
    for sig in signals:
        pair     = sig.get("pair", "")
        yf_price = sig.get("entry_price", 0.0)
        sl, tp   = sig.get("sl", 0.0), sig.get("tp", 0.0)
        live     = _fetch_xrates_price(pair)
        if live is None or yf_price <= 0:
            validated.append(sig); continue
        div = abs(live - yf_price) / yf_price
        if div > _PRICE_DIVERGENCE:
            delta = live - yf_price
            log.warning(f"LivePrice: {pair} diverged {div:.2%} — yfinance={yf_price:.5f} X-Rates={live:.5f} — updating entry")
            sig = {**sig, "entry_price": round(live, 5), "sl": round(sl + delta, 5),
                   "tp": round(tp + delta, 5), "price_source": "xrates", "price_divergence": round(div, 5)}
        else:
            log.info(f"LivePrice: {pair} confirmed ({div:.3%} divergence)")
            sig = {**sig, "price_source": "yfinance_confirmed", "price_divergence": round(div, 5)}
        validated.append(sig)
    return validated

# ═════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═════════════════════════════════════════════════════════════════════════════

def last(series: pd.Series):
    if series is None or series.empty:
        return None
    try:
        val = float(series.iloc[-1])
        return None if pd.isna(val) else val
    except Exception:
        return None

@retry_with_backoff(max_retries=3, backoff_factor=10)
def download(pair: str) -> Tuple[pd.DataFrame, bool]:
    import time as _time, random as _random
    cache_key = pair.replace("=X", "")
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached, True

    max_retries = 3
    df = None

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait = 0.5 * attempt + _random.uniform(0, 0.3)
                log.info(f"{pair} 15m retry {attempt}/{max_retries-1} (SQLite lock backoff {wait:.1f}s)")
                _time.sleep(wait)
            with _yf_lock:
                df = yf.download(pair, interval="15m", period=LOOKBACK, progress=False, auto_adjust=True, threads=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna()
                if len(df) >= MIN_ROWS:
                    break
                else:
                    df = None
            else:
                log.warning(f"{pair} 15m empty on attempt {attempt+1} — possible SQLite lock")
                df = None
        except Exception as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                log.warning(f"{pair} SQLite lock exception on attempt {attempt+1}, retrying...")
                df = None; continue
            log.error(f"{pair} download failed: {e}")
            return pd.DataFrame(), False

    if df is None or df.empty:
        log.warning(f"{pair} 15m failed after {max_retries} attempts — trying 1h fallback")
        try:
            with _yf_lock:
                df = yf.download(pair, interval="1h", period="60d", progress=False, auto_adjust=True, threads=False)
            if df is None or df.empty: return pd.DataFrame(), False
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df) < MIN_ROWS: return pd.DataFrame(), False
        except Exception as e:
            log.error(f"{pair} 1h fallback failed: {e}")
            return pd.DataFrame(), False

    # Staleness guard
    try:
        last_ts = pd.Timestamp(df.index[-1])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        else:
            last_ts = last_ts.tz_convert("UTC")
        last_ts_dt = last_ts.to_pydatetime()
        age_min = (datetime.now(timezone.utc) - last_ts_dt).total_seconds() / 60
        if age_min > 60:
            log.warning(f"{pair} data is stale — last candle {age_min:.0f}min ago. Skipping.")
            return pd.DataFrame(), False
        log.debug(f"{pair} data freshness OK — last candle {age_min:.1f}min ago")
    except Exception as e:
        log.warning(f"{pair} staleness check failed: {e} — proceeding cautiously")

    _cache.set(cache_key, df)
    return df, True

def ema(s, p):         return EMAIndicator(s, window=p).ema_indicator()
def rsi(s, p=14):      return RSIIndicator(s, window=p).rsi()
def adx_calc(h, l, c): return ADXIndicator(h, l, c, window=14).adx()
def atr_calc(h, l, c): return AverageTrueRange(h, l, c, window=14).average_true_range()
def get_spread(pair):  return SPREADS.get(pair.replace("=X", ""), 0.0002)

def get_1h_trend(pair: str) -> Optional[str]:
    """
    1-Hour Trend Filter — v2.2.2-SAFE
    Returns 'BULL', 'BEAR', or None (inconclusive/data unavailable).
    Cached per-pair for 60 minutes via _cache_1h.

    Logic: Checks if EMA12 > EMA26 > EMA50 on 1-hour chart.
    - BULL: all three EMAs in bullish alignment (12 > 26 > 50)
    - BEAR: all three EMAs in bearish alignment (12 < 26 < 50)
    - None: mixed structure — 1h has no clear trend bias

    Research basis: Elder Triple Screen / Murphy Multi-Timeframe Analysis.
    MTF confirmation raises WR ~15-20% by filtering counter-trend entries.
    Using EMA50 (not EMA200) on 1h to keep it responsive to intraday shifts.
    """
    cache_key = f"1h_{pair.replace('=X','')}"
    cached = _cache_1h.get(cache_key)
    if cached is not None:
        try:
            val = cached.iloc[0]["trend"] if not cached.empty else None
            # "NONE" sentinel means inconclusive — convert back to Python None
            return None if val == "NONE" else val
        except Exception:
            pass

    try:
        with _yf_lock:
            df1h = yf.download(pair, interval="1h", period="30d", progress=False,
                               auto_adjust=True, threads=False)
        if df1h is None or df1h.empty:
            return None
        if isinstance(df1h.columns, pd.MultiIndex):
            df1h.columns = df1h.columns.get_level_values(0)
        df1h = df1h.dropna()
        if len(df1h) < 50:
            return None

        close1h = ensure_series(df1h["Close"])
        e12_1h  = last(ema(close1h, 12))
        e26_1h  = last(ema(close1h, 26))
        e50_1h  = last(ema(close1h, 50))

        if None in (e12_1h, e26_1h, e50_1h):
            result = None
        elif e12_1h > e26_1h > e50_1h:
            result = "BULL"
        elif e12_1h < e26_1h < e50_1h:
            result = "BEAR"
        else:
            result = None  # Mixed — e.g. ranging or transition

        # Store in cache as minimal DataFrame
        trend_val = result if result is not None else "NONE"
        _cache_1h.set(cache_key, pd.DataFrame([{"trend": trend_val}]))
        return result

    except Exception as e:
        log.debug(f"{pair.replace('=X','')} 1h trend failed: {e}")
        return None

# ═════════════════════════════════════════════════════════════════════════════
# SESSION
# ═════════════════════════════════════════════════════════════════════════════

def get_market_session() -> str:
    h = datetime.now(timezone.utc).hour
    if   h < 8:  return "ASIAN"
    elif h < 13: return "EUROPEAN"
    elif h < 16: return "OVERLAP"
    elif h < 21: return "US"
    return "LATE_US"

def calculate_dynamic_session_bonus(pair: str, session: str, config: Dict) -> int:
    if not config.get("advanced", {}).get("enable_session_filtering", True): return 0
    bonuses = config.get("advanced", {}).get("session_bonuses", {}).get(session, {})
    if session == "ASIAN":
        if "JPY" in pair:                          return bonuses.get("JPY_pairs", 0)
        if any(c in pair for c in ["AUD","NZD"]): return bonuses.get("AUD_NZD_pairs", 0)
        return bonuses.get("other", 0)
    if session == "EUROPEAN":
        # v2.2.0: explicit GBP cross check before generic EUR/GBP catch-all
        if pair in ["GBPAUD", "GBPCAD"]: return bonuses.get("GBP_crosses", 0)
        if any(c in pair for c in ["EUR","GBP"]) and pair not in ["EURUSD","GBPUSD"]:
            return bonuses.get("EUR_GBP_crosses", 0)
        if any(c in pair for c in ["EUR","GBP"]): return bonuses.get("EUR_GBP_pairs", 0)
        return bonuses.get("other", 0)
    if session == "OVERLAP":  return bonuses.get("all_major_pairs", 0)
    if session == "US":
        if pair in ["GBPAUD","GBPCAD"]: return bonuses.get("GBP_crosses", 0)
        if "USD" in pair and pair in ["EURUSD","GBPUSD","USDCAD"]: return bonuses.get("USD_majors", 0)
        return bonuses.get("other", 0)
    if session == "LATE_US":  return bonuses.get("all_major_pairs", 0)
    return 0

# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def classify_market_state(adx: float, atr: float, entry: float) -> str:
    atr_pct = atr / entry if entry > 0 else 0
    if adx < 15:   return "CHOPPY"
    if adx > 25:   return "TRENDING_STRONG_HIGH_VOL" if atr_pct > 0.002 else "TRENDING_STRONG"
    if adx > 20:   return "TRENDING_MODERATE"
    return "CONSOLIDATING"

def get_signal_type(e12, e26, e200, r, adx=None) -> str:
    rl, rh = (30,70) if adx and adx>30 else ((35,65) if adx and adx>25 else (40,60))
    if e12 > e26 > e200: return "momentum" if r > rh else "trend-continuation"
    if e12 < e26 < e200: return "momentum" if r < rl else "trend-continuation"
    if (e12>e26 and r<rl) or (e12<e26 and r>rh): return "reversal"
    return "breakout"

def calculate_hold_time(rr: float, atr: float) -> str:
    if rr > 2.5 or atr > 0.002: return "SWING"
    if rr > 1.8 or atr > 0.0015: return "INTRADAY"
    return "SHORT"

def calculate_eligible_modes(score: int, adx: float, config: Dict) -> List[str]:
    modes = []
    for m in ["conservative", "aggressive"]:
        s = config["settings"][m]
        if score >= s["threshold"] and adx >= s["min_adx"]:
            modes.append(m)
    return modes

def classify_signal_tier(score: int) -> str:
    tiers = CONFIG.get("tiers", {})
    if score >= tiers.get("A_plus_min_score", 80): return "A+"
    if score >= tiers.get("A_min_score", 72):      return "A"
    if score >= tiers.get("B_min_score", 62):      return "B"
    return "C"

def calculate_signal_freshness(ts: datetime) -> Dict:
    age    = (datetime.now(timezone.utc) - ts).total_seconds() / 60
    status = ("FRESH" if age<15 else "RECENT" if age<30 else "AGING" if age<60 else "STALE")
    return {"status": status, "age_minutes": round(age,1), "confidence_decay": round(max(0, 100-age*2), 1)}

def calculate_market_volatility(sigs: List[Dict]) -> str:
    if not sigs: return "CALM"
    avg = sum(s.get("atr",0) for s in sigs) / len(sigs)
    return "HIGH" if avg>0.002 else ("NORMAL" if avg>0.0015 else "CALM")

def calculate_market_sentiment_bias(sigs: List[Dict]) -> str:
    if not sigs: return "MIXED"
    bull = sum(1 for s in sigs if s.get("direction")=="BUY")
    bear = sum(1 for s in sigs if s.get("direction")=="SELL")
    return "BULLISH" if bull>bear*1.5 else ("BEARISH" if bear>bull*1.5 else "MIXED")

# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def validate_signal_quality(signal: Dict, config: Dict) -> Tuple[bool, List[str]]:
    warnings  = []
    val_cfg   = config.get("advanced", {}).get("validation", {})
    ms        = (config["settings"]["aggressive"] if config.get("mode")=="all"
                 else config["settings"].get(signal.get("eligible_modes",["conservative"])[0], config["settings"]["conservative"]))

    if val_cfg.get("require_direction", True) and signal.get("direction") not in ("BUY","SELL"):
        return False, [f"Invalid direction: {signal.get('direction')}"]

    entry, sl, tp = signal.get("entry_price",0), signal.get("sl",0), signal.get("tp",0)
    if entry<=0 or abs(entry-sl)==0: return False, ["Invalid entry/SL"]

    sl_dist = abs(entry-sl)
    if (signal.get("spread",0.0002)/sl_dist if sl_dist else 1) > val_cfg.get("max_spread_ratio",0.25):
        return False, ["High spread ratio"]

    atr = signal.get("atr")
    if atr is None or not isinstance(atr,(int,float)): return False, ["Missing/invalid ATR"]

    atr_pct = atr/entry
    max_a, min_a = (0.006,0.0001) if "JPY" in signal["pair"] else (0.005,0.00001)
    if atr_pct<=min_a or atr_pct>max_a: return False, [f"Invalid ATR: {atr_pct:.4%}"]

    sl_pips = price_to_pips(signal["pair"], sl_dist)
    min_sl  = val_cfg.get("min_sl_pips",{}).get("JPY_pairs" if "JPY" in signal["pair"] else "other",12)
    if val_cfg.get("reject_missing_pips",True) and sl_pips<1: return False, [f"Missing pips: {sl_pips:.1f}"]
    if sl_pips < min_sl: return False, [f"SL too tight: {sl_pips:.1f}"]
    if signal["risk_reward"] < min(ms.get("min_risk_reward",2.5), 2.5): return False, [f"Poor R:R: {signal['risk_reward']:.2f}"]
    if abs(sl-entry)/entry > val_cfg.get("max_sl_distance_pct",0.02): return False, ["SL too far"]
    if abs(tp-entry)/entry > val_cfg.get("max_tp_distance_pct",0.05): return False, ["TP too far"]

    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(signal["timestamp"].replace("Z","+00:00"))).total_seconds()
        if age > val_cfg.get("max_signal_age_seconds",900): return False, [f"Stale: {age/60:.1f}min"]
    except Exception as e:
        return False, [f"Invalid timestamp: {e}"]

    return True, warnings

# ═════════════════════════════════════════════════════════════════════════════
# EQUITY PROTECTION
# ═════════════════════════════════════════════════════════════════════════════

def check_equity_protection(config: Dict) -> Tuple[bool, str]:
    pf = Path("signal_state/trading_paused.json")
    if SIGNAL_ONLY_MODE and pf.exists():
        try:
            with open(pf) as f: d = json.load(f)
            pu = datetime.fromisoformat(d.get("paused_until","").replace("Z","+00:00"))
            if datetime.now(timezone.utc) < pu:
                return False, f"Paused for {(pu-datetime.now(timezone.utc)).total_seconds()/60:.1f}min"
            pf.unlink()
        except Exception: pass
    return True, ""

# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL ID / DUPLICATES
# ═════════════════════════════════════════════════════════════════════════════

def generate_deterministic_signal_id(pair, direction, entry, session, date) -> str:
    raw = f"{pair}|{direction}|{session}|{date}"
    return f"{pair}_{direction}_{hashlib.sha1(raw.encode()).hexdigest()[:12]}"

def get_existing_signals_today() -> List[str]:
    today = datetime.now(timezone.utc).date()
    ids: set = set()
    df_file = Path("signal_state/dashboard_state.json")
    if df_file.exists():
        try:
            with open(df_file) as f: data = json.load(f)
            for s in (data.get("signals_by_mode",{}).get("aggressive",[]) +
                      data.get("signals_by_mode",{}).get("conservative",[])):
                try:
                    if datetime.fromisoformat(s.get("timestamp","").replace("Z","+00:00")).date()==today:
                        if sid := s.get("signal_id"): ids.add(sid)
                except Exception: pass
        except Exception: pass
    if PERFORMANCE_TRACKER:
        try:
            for t in PERFORMANCE_TRACKER.history.get("signals",[]):
                try:
                    if datetime.fromisoformat(t.get("timestamp","").replace("Z","+00:00")).date()==today:
                        if sid := t.get("signal_id"): ids.add(sid)
                except Exception: pass
        except Exception: pass
    log.info(f"Found {len(ids)} existing signals today")
    return list(ids)

def get_existing_pair_counts_today() -> Dict[str, int]:
    """
    Returns {pair: count} for all signals already fired today across ALL
    previous runs. Pre-seeds filter_pair_limits so the daily per-pair cap
    is enforced globally — not just within a single run's batch.

    Prevents same pair firing in opposite directions on same day:
    e.g. GBPJPY SELL at 10:46 then GBPJPY BUY at 12:45 — both full stops.
    e.g. GBPCAD SELL at 08:01 then GBPCAD BUY at 16:45 — both full stops.
    Bug root cause: signal_ids include direction so they weren't seen as
    duplicates, and pair limits only counted within the current run batch.
    """
    today = datetime.now(timezone.utc).date()
    counts: Dict[str, int] = {}
    sources: List[Dict] = []

    df_file = Path("signal_state/dashboard_state.json")
    if df_file.exists():
        try:
            with open(df_file) as f: data = json.load(f)
            sources += (data.get("signals_by_mode",{}).get("aggressive",[]) +
                        data.get("signals_by_mode",{}).get("conservative",[]))
        except Exception: pass

    if PERFORMANCE_TRACKER:
        try:
            sources += PERFORMANCE_TRACKER.history.get("signals", [])
        except Exception: pass

    seen_ids: set = set()
    for s in sources:
        try:
            if datetime.fromisoformat(s.get("timestamp","").replace("Z","+00:00")).date() != today:
                continue
            sid = s.get("signal_id","")
            if sid in seen_ids: continue
            if sid: seen_ids.add(sid)
            pair = s.get("pair","")
            if pair:
                counts[pair] = counts.get(pair, 0) + 1
        except Exception: pass

    if counts:
        log.info(f"Today's existing pair counts (cross-run cap): {counts}")
    return counts

def _safe_date(ts: str):
    """Parse a timestamp string and return its UTC date, or None on failure."""
    try:
        return datetime.fromisoformat(ts.replace("Z","+00:00")).date()
    except Exception:
        return None

def is_duplicate_signal(sid: str, existing: List[str]) -> bool:
    return sid in existing

# ═════════════════════════════════════════════════════════════════════════════
# THRESHOLD OPTIMISATION
# ═════════════════════════════════════════════════════════════════════════════

def optimize_thresholds_if_needed(config: Dict) -> Dict:
    tuning = config.get("performance_tuning",{})
    if not tuning.get("auto_adjust_thresholds",False) or not PERFORMANCE_TRACKER:
        return config
    try:
        stats = PERFORMANCE_TRACKER.history.get("stats",{})
        if stats.get("total_trades",0) < tuning.get("min_trades_for_optimization",50): return config
        lof = Path("signal_state/last_optimization.json")
        if lof.exists():
            with open(lof) as f: last = datetime.fromisoformat(json.load(f).get("timestamp"))
            if (datetime.now(timezone.utc)-last).days < tuning.get("optimization_interval_days",14): return config
    except Exception: return config

    oc   = copy.deepcopy(config)
    wr   = float(stats.get("win_rate",0))/100 if stats.get("win_rate",0)>1 else float(stats.get("win_rate",0))
    exp  = stats.get("expectancy_pips", stats.get("expectancy",0))
    mode = oc.get("mode","conservative")
    ct   = oc["settings"][mode]["threshold"]
    adj  = (5 if exp<tuning.get("optimization_inputs",{}).get("min_expectancy",5) else
            3 if wr<tuning.get("target_win_rate",0.5)-0.05 else
           -3 if wr>tuning.get("target_win_rate",0.5)+0.10 and exp>1.0 else 0)
    if adj:
        nt = max(55 if mode=="aggressive" else 60, min(70 if mode=="aggressive" else 75, ct+adj))
        oc["settings"][mode]["threshold"] = nt
        log.info(f"Threshold: {ct} -> {nt}")
        lof.parent.mkdir(exist_ok=True)
        with open(lof,"w") as f:
            json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "old":ct,"new":nt},f)
    return oc

# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL RESOLUTION
# ═════════════════════════════════════════════════════════════════════════════

def resolve_active_signals():
    df_file = Path("signal_state/dashboard_state.json")
    if not df_file.exists(): return 0
    try:
        with open(df_file) as f: data = json.load(f)
        raw_signals = (data.get("signals_by_mode",{}).get("aggressive",[]) +
                       data.get("signals_by_mode",{}).get("conservative",[]))
    except Exception: return 0

    seen_sids: set = set()
    signals: List[Dict] = []
    for s in raw_signals:
        sid = s.get("signal_id","")
        if sid and sid not in seen_sids:
            signals.append(s); seen_sids.add(sid)
        elif not sid:
            signals.append(s)

    already_recorded: set = set()
    if PERFORMANCE_TRACKER:
        already_recorded = {s.get("signal_id","") for s in PERFORMANCE_TRACKER.history.get("signals",[])}

    resolved, active = 0, []
    for sig in signals:
        if sig.get("status") != "OPEN":
            active.append(sig); continue

        pair      = sig.get("pair","")
        sid       = sig.get("signal_id","")
        direction = sig.get("direction","BUY")
        entry     = sig.get("entry_price",0)
        sl        = sig.get("sl",0)
        tp        = sig.get("tp",0)

        if sid and sid in already_recorded:
            log.debug(f"Skipping {sid} — already in history"); continue

        try:
            yf_pair = f"{pair.replace('=X','')}=X"
            df = yf.download(yf_pair, interval="1m", period="1d", progress=False, auto_adjust=True)
            if df is None or df.empty:
                active.append(sig); continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            now  = datetime.now(timezone.utc)
            close_val = df["Close"].iloc[-1]
            if hasattr(close_val, 'iloc'): close_val = close_val.iloc[0]
            curr = float(close_val)

            try:
                entry_time = datetime.fromisoformat(sig.get("timestamp","").replace("Z","+00:00"))
                exp_str = sig.get("metadata",{}).get("expires_at","")
                exp_time = (datetime.fromisoformat(exp_str.replace("Z","+00:00"))
                            if exp_str else entry_time + timedelta(minutes=15))
                idx = df.index
                if not hasattr(idx, 'tz') or idx.tz is None:
                    idx = idx.tz_localize("UTC")
                elif str(idx.tz) != "UTC":
                    idx = idx.tz_convert("UTC")
                cut = min(exp_time, now)
                mask = (idx >= entry_time) & (idx <= cut)
                window_df = df[mask.values] if mask.any() else df
            except Exception:
                window_df = df
                exp_time  = now

            if not window_df.empty:
                h = window_df["High"].max()
                l = window_df["Low"].min()
                if hasattr(h, 'iloc'): h = h.iloc[0]
                if hasattr(l, 'iloc'): l = l.iloc[0]
                period_high = float(h)
                period_low  = float(l)
            else:
                period_high = curr
                period_low  = curr

            outcome = None
            if direction == "BUY":
                if period_low  <= sl: outcome, ep = "LOSS", sl
                elif period_high >= tp: outcome, ep = "WIN",  tp
            else:
                if period_high >= sl: outcome, ep = "LOSS", sl
                elif period_low  <= tp: outcome, ep = "WIN",  tp

            if not outcome:
                if now > exp_time:
                    outcome, ep = "EXPIRED", curr

            if outcome:
                pips = price_to_pips(pair, abs(ep - entry))
                if outcome == "LOSS":    pips = -pips
                elif outcome == "EXPIRED":
                    pips = price_to_pips(pair, ep - entry if direction=="BUY" else entry - ep)

                if PERFORMANCE_TRACKER:
                    PERFORMANCE_TRACKER.record_trade(
                        signal_id=sid, pair=pair, direction=direction,
                        entry_price=entry, exit_price=ep, sl=sl, tp=tp,
                        outcome=outcome, pips=pips,
                        confidence=sig.get("confidence"), score=sig.get("score"),
                        session=sig.get("session"), entry_time=sig.get("timestamp"),
                        exit_time=now.isoformat(), tier=sig.get("tier"),
                        eligible_modes=sig.get("eligible_modes"),
                        sentiment_applied=sig.get("sentiment_applied",False),
                        sentiment_score=sig.get("sentiment_score",0.0),
                        sentiment_adjustment=sig.get("sentiment_adjustment",0.0),
                        estimated_win_rate=sig.get("estimated_win_rate"),
                        risk_reward=sig.get("risk_reward"),
                        adx=sig.get("adx"), rsi=sig.get("rsi"), atr=sig.get("atr"))
                    already_recorded.add(sid)
                    resolved += 1
                    icon = "✅" if outcome=="WIN" else ("❌" if outcome=="LOSS" else "⏱")
                    log.info(f"{icon} {pair} {direction} {outcome} ({pips:+.1f}p)")
            else:
                active.append(sig)

        except Exception as e:
            log.error(f"Error resolving {sid}: {e}"); active.append(sig)

    if resolved > 0:
        data["signals_by_mode"] = split_signals_by_mode(active)
        with open(df_file,"w") as f: json.dump(data,f,indent=2)
        log.info(f"Resolved {resolved}, {len(active)} still active")
    return resolved

# ═════════════════════════════════════════════════════════════════════════════
# PAIR LIMIT FILTER
# ═════════════════════════════════════════════════════════════════════════════

def filter_pair_limits(signals: List[Dict], config: Dict,
                       existing_counts: Optional[Dict[str, int]] = None) -> List[Dict]:
    limits   = config.get("advanced",{}).get("pair_limits",{"GBPUSD":5,"GBPJPY":2,"EURGBP":0,"default":3})
    dir_filt = config.get("advanced",{}).get("directional_filters",{})
    # Pre-seed counts with signals already fired today in previous runs
    counts: Dict[str,int] = dict(existing_counts) if existing_counts else {}
    out = []
    for sig in sorted(signals, key=lambda x: x["score"], reverse=True):
        pair, direction = sig["pair"], sig["direction"]
        if pair in dir_filt:
            pf      = dir_filt[pair]
            allowed = pf.get("allowed_directions",[])
            blocked = pf.get("blocked_directions",[])
            reason  = pf.get("reason","directional bias")
            if blocked and direction in blocked:
                log.info(f"Blocked {pair} {direction} ({reason})"); continue
            if allowed and direction not in allowed:
                log.info(f"Blocked {pair} {direction} (only {allowed} allowed)"); continue
        lim = limits.get(pair, limits.get("default",3))
        if lim == 0:
            log.info(f"Blocked {pair} (blocklist)"); continue
        if counts.get(pair,0) < lim:
            out.append(sig); counts[pair] = counts.get(pair,0)+1
        else:
            log.info(f"Skipped {pair} (daily cap {lim} reached — {counts.get(pair,0)} already fired today)")
    if len(out) < len(signals):
        log.info(f"Pair filter: {len(signals)} -> {len(out)}")
    return out

# ═════════════════════════════════════════════════════════════════════════════
# NEWS AGGREGATOR — HuggingFace FinBERT Sentiment
# ═════════════════════════════════════════════════════════════════════════════

class NewsAggregator:
    def __init__(self):
        self.newsapi_key    = os.getenv("NEWSAPI_KEY","")
        self.marketaux_key  = os.getenv("MARKETAUX_API_KEY","")
        self.hf_api_key     = HF_API_KEY
        sent_cfg            = (CONFIG or {}).get("sentiment", {})
        self.primary_model  = sent_cfg.get("hf_model_primary",  HF_PRIMARY_MODEL)
        self.fallback_model = sent_cfg.get("hf_model_fallback",  HF_FALLBACK_MODEL)
        self.cache_ttl      = sent_cfg.get("cache_ttl_minutes",  30) * 60
        self._min_interval  = sent_cfg.get("rate_limit_seconds", 2.0)
        self.min_score_thr  = sent_cfg.get("min_score_threshold", 60)
        self.max_adj_pts    = sent_cfg.get("max_adjustment_pts",  10)
        self._article_cache  : Dict[str, Tuple[float, List[str]]] = {}
        self._sentiment_cache: Dict[str, Tuple[float, float]]     = {}
        self._last_call      : Dict[str, float]                   = {}
        if not self.hf_api_key:
            log.warning("HF_API_KEY not set - FinBERT disabled. Add secret: HF_API_KEY=hf_xxxx")

    def _rate_limit(self, key: str):
        elapsed = time.time() - self._last_call.get(key, 0)
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call[key] = time.time()

    def _call_hf_inference(self, texts: List[str], model: str) -> Optional[List[Dict]]:
        if not self.hf_api_key or not texts: return None
        url     = f"{HF_API_BASE}/{model}"
        headers = {"Authorization": f"Bearer {self.hf_api_key}", "Content-Type": "application/json"}
        results    = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch   = texts[i: i + batch_size]
            payload = {"inputs": batch, "options": {"wait_for_model": True}}
            try:
                self._rate_limit("huggingface")
                resp = requests.post(url, headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    for item in resp.json():
                        best = (max(item, key=lambda x: x.get("score",0)) if isinstance(item, list) else item)
                        results.append(best)
                elif resp.status_code == 503:
                    log.info(f"HF model loading ({model}), waiting 20s...")
                    time.sleep(20)
                    resp2 = requests.post(url, headers=headers, json=payload, timeout=60)
                    if resp2.status_code == 200:
                        for item in resp2.json():
                            best = (max(item, key=lambda x: x.get("score",0)) if isinstance(item, list) else item)
                            results.append(best)
                    else: return None
                elif resp.status_code == 401:
                    log.error("HF_API_KEY invalid or expired"); return None
                else:
                    log.warning(f"HF API {resp.status_code}: {resp.text[:200]}"); return None
            except requests.Timeout:
                log.warning(f"HF API timeout ({model})"); return None
            except Exception as e:
                log.warning(f"HF API call failed: {e}"); return None
        return results or None

    def _analyse_with_finbert(self, texts: List[str]) -> Optional[float]:
        if not texts or not self.hf_api_key: return None
        truncated = [t[:2000] for t in texts if t and t.strip()]
        if not truncated: return None
        raw = self._call_hf_inference(truncated, self.primary_model)
        if raw is None:
            log.info(f"Primary model failed, trying fallback: {self.fallback_model}")
            raw = self._call_hf_inference(truncated, self.fallback_model)
        if not raw: return None
        w_sum, w_total = 0.0, 0.0
        for item in raw:
            label      = item.get("label", "neutral")
            confidence = item.get("score", 0.5)
            numeric    = HF_LABEL_MAP.get(label, 0.0)
            w_sum    += numeric * confidence
            w_total  += confidence
        if w_total == 0: return 0.0
        score = w_sum / w_total
        label_str = ("positive" if score > 0.1 else "negative" if score < -0.1 else "neutral")
        log.info(f"FinBERT: {len(raw)} texts -> avg={score:+.3f} ({label_str})")
        return max(-1.0, min(1.0, score))

    def _fetch_newsapi_articles(self, currency: str) -> List[str]:
        key = f"newsapi_{currency}"
        cached = self._article_cache.get(key)
        if cached and time.time() - cached[0] < self.cache_ttl: return cached[1]
        if not self.newsapi_key: return []
        try:
            self._rate_limit("newsapi")
            r = requests.get("https://newsapi.org/v2/everything",
                             params={"q": f"{currency} currency forex", "language": "en",
                                     "sortBy": "publishedAt", "pageSize": 15, "apiKey": self.newsapi_key},
                             timeout=10)
            if r.status_code != 200: return []
            texts = []
            for a in r.json().get("articles",[]):
                text = f"{a.get('title','')}. {a.get('description','')}".strip(". ")
                if text: texts.append(text)
            self._article_cache[key] = (time.time(), texts)
            log.info(f"NewsAPI: {len(texts)} articles for {currency}")
            return texts
        except Exception as e:
            log.warning(f"NewsAPI fetch failed for {currency}: {e}")
            return []

    def _fetch_marketaux_articles(self, currency: str) -> List[str]:
        key = f"marketaux_{currency}"
        cached = self._article_cache.get(key)
        if cached and time.time() - cached[0] < self.cache_ttl: return cached[1]
        if not self.marketaux_key: return []
        try:
            self._rate_limit("marketaux")
            r = requests.get("https://api.marketaux.com/v1/news/all",
                             params={"symbols": f"{currency}USD", "filter_entities": "true",
                                     "language": "en", "limit": 10, "api_token": self.marketaux_key},
                             timeout=10)
            if r.status_code != 200: return []
            texts = []
            for a in r.json().get("data",[]):
                text = f"{a.get('title','')}. {a.get('description','')}".strip(". ")
                if text: texts.append(text)
            self._article_cache[key] = (time.time(), texts)
            log.info(f"MarketAux: {len(texts)} articles for {currency}")
            return texts
        except Exception as e:
            log.warning(f"MarketAux fetch failed for {currency}: {e}")
            return []

    def get_currency_sentiment(self, currency: str) -> float:
        cached = self._sentiment_cache.get(currency)
        if cached and time.time() - cached[0] < self.cache_ttl:
            log.info(f"{currency} sentiment (cached): {cached[1]:+.3f}")
            return cached[1]
        newsapi_texts   = self._fetch_newsapi_articles(currency)
        marketaux_texts = self._fetch_marketaux_articles(currency)
        all_texts       = newsapi_texts + marketaux_texts
        if not all_texts:
            log.warning(f"No articles for {currency}, returning neutral 0.0")
            return 0.0
        log.info(f"Analysing {len(all_texts)} articles for {currency} ({len(newsapi_texts)} NewsAPI + {len(marketaux_texts)} MarketAux)")
        sentiment = self._analyse_with_finbert(all_texts)
        if sentiment is None:
            log.warning(f"FinBERT unavailable for {currency}, returning 0.0")
            sentiment = 0.0
        self._sentiment_cache[currency] = (time.time(), sentiment)
        log.info(f"{currency} final FinBERT sentiment: {sentiment:+.3f}")
        return sentiment

# ═════════════════════════════════════════════════════════════════════════════
# SENTIMENT ENHANCEMENT + USD GATE
# ═════════════════════════════════════════════════════════════════════════════

def enhance_with_sentiment(signals: List[Dict], news_agg: NewsAggregator) -> List[Dict]:
    if not signals: return signals
    try:
        currencies: set = set()
        for sig in signals:
            p = sig.get("pair","")
            if len(p) >= 6:
                currencies.add(p[:3]); currencies.add(p[3:6])
        cur_sent: Dict[str,float] = {}
        for cur in currencies:
            try: cur_sent[cur] = news_agg.get_currency_sentiment(cur)
            except Exception as e:
                log.warning(f"Sentiment failed for {cur}: {e}"); cur_sent[cur] = 0.0
        for sig in signals:
            p = sig.get("pair","")
            if len(p) < 6:
                sig.update({"sentiment_score":0.0,"sentiment_applied":True}); continue
            base, quote = p[:3], p[3:6]
            direction   = sig.get("direction","BUY")
            net = (cur_sent.get(base,0.0) - cur_sent.get(quote,0.0) if direction=="BUY"
                   else cur_sent.get(quote,0.0) - cur_sent.get(base,0.0))
            orig = sig.get("score",50)
            adj  = round(max(-float(news_agg.max_adj_pts), min(float(news_agg.max_adj_pts), net*5)), 1)
            new  = round(max(0, min(100, orig+adj)), 0)
            sig.update({"sentiment_score": round(net,3), "sentiment_adjustment": adj,
                        "score_before_sentiment": orig, "score": new, "sentiment_applied": True})
            log.info(f"{p} {direction}: net={net:+.3f} score {orig}->{new} ({adj:+.1f})")
        return signals
    except Exception as e:
        log.error(f"Sentiment enhancement failed: {e}")
        for sig in signals:
            sig.update({"sentiment_score":0.0,"sentiment_applied":True})
        return signals

def _gate_usd_with_sentiment(signals: List[Dict], news_agg: NewsAggregator) -> List[Dict]:
    if not signals: return signals
    gate_threshold = CONFIG.get("advanced", {}).get("usd_pair_rules", {}).get("sentiment_gate_threshold", 0.0)
    currencies: set = set()
    for sig in signals:
        p = sig.get("pair", "")
        if len(p) >= 6:
            currencies.add(p[:3]); currencies.add(p[3:6])
    cur_sent: Dict[str, float] = {}
    for cur in currencies:
        try: cur_sent[cur] = news_agg.get_currency_sentiment(cur)
        except Exception as e:
            log.warning(f"Sentiment unavailable for {cur}: {e} — defaulting to 0.0")
            cur_sent[cur] = 0.0
    passed = []
    for sig in signals:
        p         = sig.get("pair", "")
        direction = sig.get("direction", "BUY")
        if len(p) < 6:
            sig.update({"sentiment_applied": True, "sentiment_score": 0.0, "sentiment_adjustment": 0.0})
            passed.append(sig); continue
        base, quote = p[:3], p[3:6]
        net = (cur_sent.get(base, 0.0) - cur_sent.get(quote, 0.0) if direction == "BUY"
               else cur_sent.get(quote, 0.0) - cur_sent.get(base, 0.0))
        sig.update({"sentiment_applied": True, "sentiment_score": round(net, 3),
                    "sentiment_adjustment": 0.0, "score_before_sentiment": sig.get("score", 0)})
        if net < gate_threshold:
            log.info(f"  USD GATE BLOCKED: {p} {direction} net_sentiment={net:+.3f} < gate_threshold={gate_threshold}")
            continue
        label = ("agrees" if net > 0.1 else "neutral" if abs(net) <= 0.1 else "weak-disagree-allowed")
        log.info(f"  USD GATE PASSED: {p} {direction} net_sentiment={net:+.3f} ({label})")
        passed.append(sig)
    dropped = len(signals) - len(passed)
    if dropped:
        log.info(f"USD sentiment gate: {dropped} signal(s) blocked (macro disagrees with direction)")
    else:
        log.info(f"USD sentiment gate: all {len(passed)} signal(s) passed")
    return passed

# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_signal(pair: str) -> Tuple[Optional[Dict], bool]:
    # ── Permanent blocklist — data-confirmed no edge ──────────────────────────
    pair_clean_check = pair.replace("=X", "").upper()
    if pair_clean_check in _PERMANENTLY_BLOCKED_PAIRS:
        log.info(f"  {pair_clean_check} PERMANENTLY BLOCKED — confirmed no edge (data-driven)")
        return None, True

    df, ok = download(pair)
    if not ok or len(df) < MIN_ROWS: return None, ok
    try:
        close = ensure_series(df["Close"])
        high  = ensure_series(df["High"])
        low   = ensure_series(df["Low"])
        e12   = last(ema(close,12));  e26 = last(ema(close,26))
        e200  = last(ema(close,200)); r   = last(rsi(close))
        a     = last(adx_calc(high,low,close))
        atr   = last(atr_calc(high,low,close))
        curr  = last(close)
    except Exception as e:
        log.warning(f"{pair} indicators failed: {e}"); return None, ok

    min_t = (CONFIG["settings"]["aggressive"]["threshold"] if CONFIG.get("mode")=="all"
             else min(CONFIG["settings"]["aggressive"]["threshold"], CONFIG["settings"]["conservative"]["threshold"]))
    min_a = (CONFIG["settings"]["aggressive"]["min_adx"] if CONFIG.get("mode")=="all"
             else min(CONFIG["settings"]["aggressive"]["min_adx"], CONFIG["settings"]["conservative"]["min_adx"]))

    if None in (e12, e26, e200, r, a, curr, atr):
        log.info(f"{pair.replace('=X','')} skipped: missing indicators"); return None, ok
    if a < min_a:
        log.info(f"{pair.replace('=X','')} skipped: ADX {a:.1f} < {min_a}"); return None, ok

    bull = bear = 0
    full_bull = e12>e26>e200;  full_bear = e12<e26<e200
    part_bull = e12>e26 and not full_bull
    part_bear = e12<e26 and not full_bear

    if full_bull: bull+=25
    elif full_bear: bear+=25
    elif part_bull: bull+=12
    elif part_bear: bear+=12

    if full_bull and 35<r<45: bull+=15
    elif full_bear and 55<r<65: bear+=15

    if r<30:
        bull+=20
        if full_bear: bear-=10
    elif r>70:
        bear+=20
        if full_bull: bull-=10
    elif r<38 and not full_bear: bull+=10
    elif r>62 and not full_bull: bear+=10

    if a>25:     (bull:=bull+20) if e12>e26 else (bear:=bear+20)
    elif a>20:   (bull:=bull+14) if e12>e26 else (bear:=bear+14)
    elif a>min_a:(bull:=bull+8)  if e12>e26 else (bear:=bear+8)

    ed = abs(curr-e12)/e12 if e12>0 else 0
    if ed>0.0008:   (bull:=bull+10) if curr>e12 else (bear:=bear+10)
    elif ed>0.0002: (bull:=bull+5)  if curr>e12 else (bear:=bear+5)

    ec = abs(e12-e26)/e26 if e26>0 else 0
    if ec>0.0008:   (bull:=bull+8) if e12>e26 else (bear:=bear+8)
    elif ec>0.0002: (bull:=bull+4) if e12>e26 else (bear:=bear+4)

    session = get_market_session()
    bonus   = calculate_dynamic_session_bonus(pair.replace("=X",""), session, CONFIG)
    if e12>e26: bull+=bonus
    elif e12<e26: bear+=bonus

    bull = max(0,bull); bear = max(0,bear)
    diff = abs(bull-bear)
    hint = "BULL" if bull>bear else ("BEAR" if bear>bull else "TIE")
    log.info(f"{pair.replace('=X','')} | score={diff} ({hint}) bull={bull} bear={bear} | RSI={r:.0f} ADX={a:.0f} | {session}")

    if bull==bear: return None, ok

    sess_thresh = CONFIG.get("advanced",{}).get("session_thresholds",{}).get(session, min_t)
    if diff < sess_thresh:
        log.info(f"  BLOCKED: {diff} < session threshold {sess_thresh} ({session})"); return None, ok
    if diff < min_t:
        log.info(f"  BLOCKED: {diff} < min threshold {min_t}"); return None, ok

    pair_clean = pair.replace("=X", "").upper()
    usd_rules  = CONFIG.get("advanced", {}).get("usd_pair_rules", {})
    if usd_rules.get("enabled", False) and pair_clean in _USD_RESTRICTED_PAIRS:
        allowed_sessions = usd_rules.get("allowed_sessions", ["US"])
        if session not in allowed_sessions:
            log.info(f"  BLOCKED: {pair_clean} restricted to {allowed_sessions} — current session={session}")
            return None, ok
        usd_min_score = usd_rules.get("min_score", 65)
        if diff < usd_min_score:
            log.info(f"  BLOCKED: {pair_clean} USD pair score {diff} < USD min {usd_min_score}")
            return None, ok

    if ECONOMIC_CALENDAR:
        blocked, event_name = _is_calendar_blackout(pair, ECONOMIC_CALENDAR)
        if blocked:
            log.info(f"  BLOCKED: news blackout — {event_name}"); return None, ok

    direction = "BUY" if bull>bear else "SELL"
    conf      = "VERY_STRONG" if diff>=75 else ("STRONG" if diff>=65 else "MODERATE")
    tier      = classify_signal_tier(diff)

    # ── 1-Hour Trend Filter — v2.2.2-SAFE ──────────────────────────────
    # Elder Triple Screen principle: 15m signal must align with 1h EMA structure.
    # Sub-threshold (55-61): REQUIRE 1h agreement — this is what unlocks the lower threshold
    # Above threshold (62+): 1h DISAGREEMENT blocks — raises quality across the board
    # Above threshold (62+) with inconclusive 1h: passes on 15m strength alone
    # Result: more signals on trending days, zero extra noise on choppy days
    old_hard_threshold = 62
    trend_1h = get_1h_trend(pair)
    signal_has_1h_confirmation = False
    if trend_1h is not None:
        trend_agrees = (direction == "BUY"  and trend_1h == "BULL") or \
                       (direction == "SELL" and trend_1h == "BEAR")
        if not trend_agrees:
            log.info(f"  BLOCKED: 1h trend {trend_1h} conflicts with 15m {direction} — MTF disagreement")
            return None, ok
        log.info(f"  ✓ 1h trend {trend_1h} confirms {direction} — MTF aligned")
        signal_has_1h_confirmation = True
    else:
        # 1h inconclusive (ranging/transitioning) — only pass if score >= original hard threshold
        if diff < old_hard_threshold:
            log.info(f"  BLOCKED: score {diff} < {old_hard_threshold} and 1h trend inconclusive — MTF confirmation required for sub-{old_hard_threshold} signals")
            return None, ok
        log.info(f"  1h inconclusive — passing on 15m conviction (score={diff} >= {old_hard_threshold})")

    if tier == "C":
        log.info(f"  BLOCKED: Tier C (score={diff} < B_min={CONFIG.get('tiers',{}).get('B_min_score',55)}) — net negative historically")
        return None, ok

    if tier == "A":
        log.info(f"  BLOCKED: Tier A (score={diff}, 72-79) — 0% WR across all 6 trades, -107 pips. "
                 f"High scores = already-extended moves, entries too late. Keeping B and A+ only.")
        return None, ok

    spread   = get_spread(pair)
    atr_stop = CONFIG.get("advanced", {}).get("atr_stop_multiplier", CONFIG["settings"]["aggressive"].get("atr_stop_multiplier", 2.5))
    atr_tgt  = CONFIG.get("advanced", {}).get("atr_target_multiplier", CONFIG["settings"]["aggressive"].get("atr_target_multiplier", 7.0))

    if direction=="BUY":  sl=curr-atr_stop*atr; tp=curr+atr_tgt*atr
    else:                 sl=curr+atr_stop*atr; tp=curr-atr_tgt*atr

    rr = abs(tp-curr)/abs(curr-sl) if abs(curr-sl)>0 else 0
    min_rr = CONFIG.get("advanced",{}).get("min_risk_reward", CONFIG["settings"]["aggressive"].get("min_risk_reward",2.5))
    if rr<min_rr:
        log.info(f"  BLOCKED: RR {rr:.2f} < {min_rr}"); return None, ok
    if rr>10: return None, ok

    now     = datetime.now(timezone.utc)
    val_cfg      = CONFIG.get("advanced",{}).get("validation",{})
    session_exp  = val_cfg.get("session_expiry_minutes",{})
    default_mins = val_cfg.get("max_signal_age_seconds", 900) / 60
    expiry_mins  = session_exp.get(session, default_mins)
    expires = now + timedelta(minutes=expiry_mins)
    sid = generate_deterministic_signal_id(pair.replace("=X",""), direction, curr, session, now.strftime("%Y%m%d"))

    signal = {
        "signal_id": sid, "id": sid,
        "pair": pair.replace("=X",""), "direction": direction,
        "score": diff, "technical_score": diff,
        "sentiment_score": 0.0, "sentiment_adjustment": 0.0, "score_before_sentiment": diff,
        "confidence": conf, "tier": tier,
        "rsi": round(r,1), "adx": round(a,1), "atr": round(atr,5), "volume_ratio": 0,
        "session": session,
        "entry_price": round(curr,5), "sl": round(sl,5), "tp": round(tp,5),
        "risk_reward": round(rr,2), "spread": round(spread,5),
        "timestamp": now.isoformat(), "status": "OPEN",
        "hold_time": calculate_hold_time(rr,atr),
        "eligible_modes": calculate_eligible_modes(diff,a,CONFIG),
        "freshness": calculate_signal_freshness(now),
        "sentiment_applied": False,
        "htf_confirmed": signal_has_1h_confirmation,
        "htf_trend_1h": trend_1h if trend_1h else "INCONCLUSIVE",
        "metadata": {
            "signal_type": get_signal_type(e12,e26,e200,r,a),
            "market_state": classify_market_state(a,atr,curr),
            "timeframe": INTERVAL, "valid_for_minutes": int(expiry_mins),
            "generated_at": now.isoformat(), "expires_at": expires.isoformat(),
            "session_active": session in ("EUROPEAN","US","OVERLAP"),
            "signal_generator_version": "2.2.2-SAFE",
            "atr_stop_multiplier": atr_stop, "atr_target_multiplier": atr_tgt,
        },
    }

    is_valid, warnings = validate_signal_quality(signal, CONFIG)
    if not is_valid:
        log.info(f"{pair} rejected: {', '.join(warnings)}"); return None, ok
    return signal, ok

# ═════════════════════════════════════════════════════════════════════════════
# MULTI-MODE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def split_signals_by_mode(signals: List[Dict]) -> Dict[str, List[Dict]]:
    buckets: Dict[str, List[Dict]] = {"aggressive":[], "conservative":[]}
    for s in signals:
        for m in s.get("eligible_modes",[]):
            if m in buckets: buckets[m].append(s)
    return buckets

def select_high_potential(signals: List[Dict]) -> List[Dict]:
    return [s for s in signals if s.get("tier") in ("A+","A")]

def quick_micro_backtest(signal: Dict) -> float:
    base = 0.50
    tier = signal.get("tier","C")
    if tier=="A+": base+=0.15
    elif tier=="A": base+=0.10
    elif tier=="B": base+=0.05
    if signal.get("confidence")=="VERY_STRONG": base+=0.05
    elif signal.get("confidence")=="STRONG":    base+=0.03
    if signal.get("metadata",{}).get("session_active"):                       base+=0.02
    if signal.get("metadata",{}).get("market_state") in ("TRENDING_STRONG","TRENDING_STRONG_HIGH_VOL"): base+=0.03
    return min(base, 0.95)

# ═════════════════════════════════════════════════════════════════════════════
# CORRELATION FILTER + RISK LIMITS
# ═════════════════════════════════════════════════════════════════════════════

def filter_correlated_signals_enhanced(signals, max_corr=1, enabled=True):
    if not enabled or len(signals)<=1: return signals
    filtered, groups = [], {}
    for sig in sorted(signals, key=lambda x: x["score"], reverse=True):
        gk = None
        for cg in CORRELATED_PAIRS:
            if f"{sig['pair']}=X" in cg:
                gk = frozenset(cg); break
        if gk:
            cnt = groups.get(gk,0)
            if cnt < max_corr:
                filtered.append(sig); groups[gk]=cnt+1
            else:
                log.info(f"Skipping {sig['pair']} {sig['direction']} (correlation limit)")
        else:
            filtered.append(sig)
    if len(filtered)<len(signals):
        log.info(f"Correlation filter: {len(signals)} -> {len(filtered)}")
    return filtered

def check_risk_limits(signals, config, mode, existing_counts: Optional[Dict[str,int]] = None):
    risk = config.get("risk_management",{}); ms = config["settings"][mode]; warns = []
    mp = risk.get("max_open_positions",3)
    if len(signals)>mp:
        warns.append(f"Limiting {mode} to {mp} positions")
        signals = sorted(signals, key=lambda x: x["score"], reverse=True)[:mp]
    md = risk.get("max_daily_risk_pips",150); tr = 0.0; filtered = []
    for sig in signals:
        rp = price_to_pips(sig.get("pair",""), abs(sig.get("entry_price",0)-sig.get("sl",0)))
        if tr+rp<=md: filtered.append(sig); tr+=rp
        else: warns.append(f"Skipped {sig['pair']} - {mode} risk limit")
    filtered = filter_pair_limits(filtered, config, existing_counts=existing_counts)
    if config.get("advanced",{}).get("enable_correlation_filter",True):
        filtered = filter_correlated_signals_enhanced(
            filtered, ms.get("max_correlated_signals",1), enabled=True)
    return filtered, warns

# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def calculate_daily_pips(signals):
    today = datetime.now(timezone.utc).date()
    total = 0.0
    for s in signals:
        try:
            ts = s.get("timestamp", "")
            if not ts: continue
            if datetime.fromisoformat(ts.replace("Z", "+00:00")).date() != today: continue
            total += price_to_pips(s.get("pair", ""), abs(s.get("entry_price", 0) - s.get("sl", 0)))
        except Exception:
            continue
    return round(total, 1)

def get_performance_summary():
    if not PERFORMANCE_TRACKER: return {"stats":{},"analytics":{},"equity":{}}
    try: return PERFORMANCE_TRACKER.get_dashboard_summary()
    except Exception: return {"stats":{},"analytics":{},"equity":{}}

def filter_expired_signals(signals):
    now    = datetime.now(timezone.utc)
    active = []
    already_recorded: set = set()
    if PERFORMANCE_TRACKER:
        already_recorded = {s.get("signal_id","") for s in PERFORMANCE_TRACKER.history.get("signals",[])}

    for sig in signals:
        status = sig.get("status","OPEN")
        if status == "ACTIVE": sig["status"] = "OPEN"; status = "OPEN"
        if status != "OPEN": continue
        try:
            exp    = sig.get("metadata",{}).get("expires_at")
            is_exp = False
            if exp:
                is_exp = now >= datetime.fromisoformat(exp.replace("Z","+00:00"))
            else:
                st     = datetime.fromisoformat(sig["timestamp"].replace("Z","+00:00"))
                is_exp = (now-st).total_seconds() >= CONFIG.get("advanced",{}).get("validation",{}).get("max_signal_age_seconds",900)

            if not is_exp:
                active.append(sig)
            else:
                sid = sig.get("signal_id","")
                if PERFORMANCE_TRACKER and sid and sid not in already_recorded:
                    pair      = sig.get("pair","")
                    entry     = sig.get("entry_price", sig.get("entry",0.0))
                    direction = sig.get("direction","BUY")
                    sl        = sig.get("stop_loss", sig.get("sl",0.0))
                    tp        = sig.get("take_profit", sig.get("tp",0.0))
                    PERFORMANCE_TRACKER.record_trade(
                        signal_id=sid, pair=pair, direction=direction,
                        entry_price=entry, exit_price=entry, sl=sl, tp=tp,
                        outcome="EXPIRED", pips=0,
                        confidence=sig.get("confidence"), score=sig.get("score"),
                        session=sig.get("session"), entry_time=sig.get("timestamp"),
                        exit_time=now.isoformat(), tier=sig.get("tier"),
                        eligible_modes=sig.get("eligible_modes"),
                        sentiment_applied=sig.get("sentiment_applied", False),
                        sentiment_score=sig.get("sentiment_score", 0.0),
                        sentiment_adjustment=sig.get("sentiment_adjustment", 0.0),
                        estimated_win_rate=sig.get("estimated_win_rate"),
                        risk_reward=sig.get("risk_reward"),
                        adx=sig.get("adx"), rsi=sig.get("rsi"), atr=sig.get("atr"))
                    already_recorded.add(sid)
                    log.info(f"EXPIRED {sig.get('pair','')} {sig.get('direction','')} recorded to history")
                elif sid and sid in already_recorded:
                    log.debug(f"Skipping EXPIRED record for {sid} — already in history")
        except Exception as e:
            log.warning(f"filter_expired_signals error: {e}"); continue

    if len(active) < len(signals):
        log.info(f"Filtered {len(signals)-len(active)} expired signals")
    return active

def write_dashboard_state(signals, downloads, news_calls=0, mkt_calls=0,
                          config=None, mode=None, settings=None, pair_prices=None):
    cfg  = config or CONFIG; md = mode or MODE
    sigs = filter_expired_signals(signals)
    mb   = split_signals_by_mode(sigs)
    perf = get_performance_summary(); stats = perf.get("stats",{}) or {}
    can_trade, pause = check_equity_protection(cfg)

    hist = []
    if PERFORMANCE_TRACKER:
        try:
            for s in PERFORMANCE_TRACKER.history.get("signals",[]):
                if s.get("status") not in ["WIN","LOSS","EXPIRED","OPEN"]: continue
                hist.append(s)
            hist.sort(key=lambda x: x.get("timestamp",""), reverse=True)
            log.info(f"Loaded {len(hist)} historical signals (full history)")
        except Exception as e:
            log.warning(f"Could not load historical signals: {e}")

    if signals:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        seen_ids  = {s.get("signal_id","") for s in hist}
        for s in signals:
            sid = s.get("signal_id","")
            ts  = s.get("timestamp","")
            if sid and sid not in seen_ids and ts.startswith(today_str):
                hist.insert(0, {**s, "status": "OPEN"})
                seen_ids.add(sid)

    if not hist:
        df_file = Path("signal_state/dashboard_state.json")
        if df_file.exists():
            try:
                with open(df_file) as f: hist = json.load(f).get("historical_signals",[])
                log.info(f"Loaded {len(hist)} from existing dashboard")
            except Exception: pass

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_pool = [s for s in (list(sigs) + hist) if s.get("timestamp","").startswith(today_str)]
    seen_pod: set = set()
    today_pool_deduped = []
    for s in today_pool:
        sid = s.get("signal_id","")
        key = sid if sid else s.get("pair","") + s.get("direction","") + s.get("timestamp","")
        if key not in seen_pod:
            today_pool_deduped.append(s); seen_pod.add(key)
    new_pick = (max(today_pool_deduped, key=lambda x: x.get("score",0)) if today_pool_deduped else None)
    existing_pick = None
    df_file_pod = Path("signal_state/dashboard_state.json")
    if df_file_pod.exists():
        try:
            existing_data = json.loads(df_file_pod.read_text())
            ep = existing_data.get("pick_of_the_day")
            if ep and ep.get("timestamp","").startswith(today_str):
                existing_pick = ep
        except Exception: pass
    if new_pick and existing_pick:
        pick_of_the_day = (new_pick if (new_pick.get("score",0) >= existing_pick.get("score",0)) else existing_pick)
    else:
        pick_of_the_day = new_pick or existing_pick

    dashboard = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_signals": len(sigs),
        "active_signals_by_mode": {"aggressive": len(mb["aggressive"]), "conservative": len(mb["conservative"])},
        "session": get_market_session(), "mode": md,
        "sentiment_enabled": USE_SENTIMENT, "sentiment_engine": "finbert" if USE_SENTIMENT else "disabled",
        "multi_mode": True,
        "equity_protection": {"enabled": cfg.get("risk_management",{}).get("equity_protection",{}).get("enable",False), "can_trade": can_trade, "pause_reason": pause or None},
        "market_state": {"volatility": calculate_market_volatility(sigs), "sentiment_bias": calculate_market_sentiment_bias(sigs), "session": get_market_session()},
        "signals_by_mode": mb, "historical_signals": hist,
        "api_usage": {"yfinance": {"successful_downloads": downloads}, "sentiment": {"enabled": USE_SENTIMENT, "engine": "finbert", "newsapi": news_calls, "marketaux": mkt_calls}},
        "stats": {"total_trades": stats.get("total_trades",0), "win_rate": stats.get("win_rate",0), "total_pips": stats.get("total_pips",0), "wins": stats.get("wins",0), "losses": stats.get("losses",0), "expectancy": stats.get("expectancy_pips",0)},
        "performance_stats": {"total_trades": stats.get("total_trades",0), "wins": stats.get("wins",0), "losses": stats.get("losses",0), "expired": stats.get("expired",0), "win_rate": stats.get("win_rate",0), "total_pips": stats.get("total_pips",0), "avg_win": stats.get("avg_win_pips",0), "avg_loss": stats.get("avg_loss_pips",0), "expectancy": stats.get("expectancy_pips",0), "by_mode": perf.get("analytics",{}).get("by_mode",{}), "by_tier": perf.get("analytics",{}).get("by_tier",{}), "by_session": perf.get("analytics",{}).get("by_session",{}), "by_pair": perf.get("analytics",{}).get("by_pair",{})},
        "risk_management": {"theoretical_max_pips": calculate_daily_pips(sigs), "total_risk_pips": sum(price_to_pips(s.get("pair",""), abs(s.get("entry_price",0)-s.get("sl",0))) for s in sigs), "max_daily_risk": cfg.get("risk_management",{}).get("max_daily_risk_pips",150)},
        "analytics": perf.get("analytics",{}),
        "equity_curve": perf.get("equity",{}).get("curve",[]),
        "pair_prices": pair_prices or {},
        "upcoming_events": _get_upcoming_events(4),
        "pick_of_the_day": pick_of_the_day,
        "system": {"last_update": datetime.now(timezone.utc).isoformat(), "signal_only_mode": SIGNAL_ONLY_MODE, "version": "2.2.2-SAFE"},
    }

    out = Path("signal_state"); out.mkdir(exist_ok=True)
    with open(out/"dashboard_state.json","w") as f: json.dump(dashboard,f,indent=2)
    log.info(f"Dashboard written | Aggressive: {len(mb['aggressive'])} | Conservative: {len(mb['conservative'])}")
    if stats.get("total_trades",0)>0:
        log.info(f"{stats['total_trades']} trades | WR: {stats['win_rate']:.1f}% | Pips: {stats['total_pips']:.1f}")
    write_health_check(sigs, downloads, news_calls, mkt_calls, can_trade, pause, md)

def write_health_check(signals, downloads, news, mkt, can_trade, pause, mode):
    status = ("paused" if not can_trade else
              "warning" if downloads==0 or (len(signals)==0 and downloads>0 and can_trade) else "ok")
    issues = ([pause] if pause else ["No data"] if downloads==0 else ["No signals"] if len(signals)==0 and downloads>0 else [])
    health = {
        "status": status, "last_run": datetime.now(timezone.utc).isoformat(),
        "signal_count": len(signals), "issues": issues, "can_trade": can_trade,
        "api_status": {"yfinance": "ok" if downloads>0 else "degraded", "newsapi": "ok" if news>0 else "disabled", "marketaux": "ok" if mkt>0 else "disabled", "finbert_hf": "ok" if HF_API_KEY else "disabled"},
        "system_info": {"mode": mode, "pairs_monitored": len(PAIRS), "version": "2.2.2-SAFE", "sentiment_engine": "finbert" if HF_API_KEY else "disabled"},
    }
    with open(Path("signal_state/health.json"),"w") as f: json.dump(health,f,indent=2)
    log.info(f"Health: {status.upper()}")

# ═════════════════════════════════════════════════════════════════════════════
# TIME GUARD + CLEANUP
# ═════════════════════════════════════════════════════════════════════════════

def in_execution_window():
    lr = Path("signal_state/last_run.txt")
    ls = Path("signal_state/last_success.txt")
    now = datetime.now(timezone.utc)
    if lr.exists():
        try:
            last = datetime.fromisoformat(lr.read_text().strip().replace("Z","+00:00"))
            if ls.exists():
                lo = datetime.fromisoformat(ls.read_text().strip().replace("Z","+00:00"))
                if now-lo < timedelta(minutes=10): log.info(f"Already ran at {lo}"); return False
            elif now-last < timedelta(minutes=2): log.info("Waiting for retry window"); return False
        except Exception: pass
    lr.parent.mkdir(exist_ok=True); lr.write_text(now.isoformat())
    return True

def mark_success():
    p = Path("signal_state/last_success.txt"); p.parent.mkdir(exist_ok=True)
    p.write_text(datetime.now(timezone.utc).isoformat())

def cleanup_legacy_signals():
    df_file = Path("signal_state/dashboard_state.json")
    if not df_file.exists(): return
    try:
        data = json.loads(df_file.read_text())
        raw = (data["signals_by_mode"].get("aggressive",[]) + data["signals_by_mode"].get("conservative",[])
               if "signals_by_mode" in data else data.get("signals",[]))
        seen_ids: set = set()
        all_sigs = []
        for s in raw:
            sid = s.get("signal_id") or s.get("id")
            if sid and sid in seen_ids: continue
            if sid: seen_ids.add(sid)
            all_sigs.append(s)
        now  = datetime.now(timezone.utc)
        today = now.date()
        cleaned = []
        for sig in all_sigs:
            if sig.get("status")=="ACTIVE": sig["status"]="OPEN"
            if sig.get("status")!="OPEN": continue
            ts = sig.get("timestamp","")
            if ts:
                try:
                    sig_date = datetime.fromisoformat(ts.replace("Z","+00:00")).date()
                    if sig_date < today: continue
                except Exception: pass
            exp = sig.get("metadata",{}).get("expires_at")
            if exp:
                try:
                    exp_time = datetime.fromisoformat(exp.replace("Z","+00:00"))
                    if now < exp_time or (now - exp_time).total_seconds() < 7200:
                        cleaned.append(sig)
                except Exception: pass
            else:
                cleaned.append(sig)
        removed = len(all_sigs) - len(cleaned)
        if removed > 0:
            data["signals_by_mode"] = split_signals_by_mode(cleaned)
            df_file.write_text(json.dumps(data, indent=2))
            log.info(f"Cleaned {removed} stale signals ({len(cleaned)} remain)")
    except Exception: pass

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    if not in_execution_window(): return

    if PERFORMANCE_TRACKER:
        resolve_active_signals(); time.sleep(1)

    cleanup_legacy_signals()

    existing: List[Dict] = []
    df_file = Path("signal_state/dashboard_state.json")
    if df_file.exists():
        try:
            data = json.loads(df_file.read_text())
            if "signals_by_mode" in data:
                raw = (data["signals_by_mode"].get("aggressive",[]) + data["signals_by_mode"].get("conservative",[]))
            else:
                raw = data.get("signals",[])
            seen_ids: set = set()
            existing = []
            for s in raw:
                if s.get("status","OPEN") not in ("OPEN","ACTIVE"): continue
                sid = s.get("signal_id") or s.get("id")
                if sid and sid in seen_ids: continue
                if sid: seen_ids.add(sid)
                existing.append(s)
            log.info(f"{len(existing)} active signals from previous cycles")
        except Exception: pass

    can_trade, pause = check_equity_protection(CONFIG)
    if not can_trade:
        log.warning(f"Trading paused: {pause}")
        write_dashboard_state(filter_expired_signals(existing),0,0,0,CONFIG,MODE,SETTINGS)
        return

    opt_cfg  = (optimize_thresholds_if_needed(CONFIG)
                if CONFIG.get("performance_tuning",{}).get("auto_adjust_thresholds",False) else CONFIG)
    opt_mode = opt_cfg["mode"]

    log.info(f"Trade Beacon v2.2.2-SAFE | Sentiment={'FinBERT ON' if USE_SENTIMENT else 'OFF'} | HF_KEY={'set' if HF_API_KEY else 'missing'}")
    log.info(f"Monitoring {len(PAIRS)} pairs")
    log.info(f"Aggressive:   Score>={opt_cfg['settings']['aggressive']['threshold']} ADX>={opt_cfg['settings']['aggressive']['min_adx']}")
    log.info(f"Conservative: Score>={opt_cfg['settings']['conservative']['threshold']} ADX>={opt_cfg['settings']['conservative']['min_adx']}")

    new_signals: List[Dict] = []
    downloads = 0
    existing_ids    = get_existing_signals_today()
    existing_counts = get_existing_pair_counts_today()

    # ── Daily loss circuit breaker ────────────────────────────────────────────
    # If 3+ losses already today, stop generating new signals for the rest of
    # the day. Prevents cascading stops on whipsaw/choppy days (Mar 23: 6 losses,
    # -318 pips in one day).
    MAX_DAILY_LOSSES = 3
    if PERFORMANCE_TRACKER:
        today = datetime.now(timezone.utc).date()
        today_losses = sum(
            1 for s in PERFORMANCE_TRACKER.history.get("signals", [])
            if s.get("status") == "LOSS"
            and _safe_date(s.get("timestamp","")) == today
        )
        if today_losses >= MAX_DAILY_LOSSES:
            log.warning(f"🛑 DAILY LOSS CIRCUIT BREAKER: {today_losses} losses today "
                        f"(limit={MAX_DAILY_LOSSES}). No new signals for rest of day.")
            write_dashboard_state(filter_expired_signals(existing), 0, 0, 0,
                                  opt_cfg, opt_mode, None, {})
            return
    pair_prices: Dict[str, float] = {}
    _cache.clear()

    max_w = opt_cfg.get("advanced",{}).get("parallel_workers",3)
    with ThreadPoolExecutor(max_workers=min(max_w, len(PAIRS))) as executor:
        futures = {executor.submit(generate_signal, p): p for p in PAIRS}
        for future in as_completed(futures):
            pair = futures[future]
            try:
                sig, ok = future.result()
                if ok: downloads+=1
                clean = pair.replace("=X","")
                try:
                    cached_df = _cache.get(clean)
                    if cached_df is not None and not cached_df.empty:
                        close_val = cached_df["Close"].iloc[-1]
                        if hasattr(close_val, 'iloc'): close_val = close_val.iloc[0]
                        p = float(close_val)
                        if p > 0: pair_prices[clean] = round(p, 5)
                except Exception: pass
                if sig:
                    if is_duplicate_signal(sig["signal_id"], existing_ids):
                        log.info(f"{clean} - Duplicate skipped"); continue
                    new_signals.append(sig)
                    log.info(f"{clean} - Score: {sig['score']} [{sig['tier']}] ({'+'.join(sig['eligible_modes'])}) RR: {sig['risk_reward']:.2f}")
            except Exception as e:
                log.error(f"{pair.replace('=X','')} failed: {e}")

    news_calls = mkt_calls = 0

    if USE_SENTIMENT and new_signals:
        usd_rules = CONFIG.get("advanced", {}).get("usd_pair_rules", {})
        use_gate  = usd_rules.get("sentiment_gate", True)
        usd_sigs = [s for s in new_signals if s.get("pair","") in _USD_RESTRICTED_PAIRS]
        gbp_sigs = [s for s in new_signals if s.get("pair","") not in _USD_RESTRICTED_PAIRS]
        for s in gbp_sigs:
            s.update({"sentiment_applied": False, "sentiment_score": 0.0, "sentiment_adjustment": 0.0})
        if usd_sigs and use_gate:
            agg = NewsAggregator()
            if not agg.hf_api_key:
                log.warning("HF_API_KEY not set — USD sentiment gate disabled, signals pass through.")
                for s in usd_sigs:
                    s.update({"sentiment_applied": False, "sentiment_score": 0.0, "sentiment_adjustment": 0.0})
            else:
                log.info(f"FinBERT USD gate: {len(usd_sigs)} USD signal(s) (GBP excluded — pure technical)")
                try:
                    usd_sigs = _gate_usd_with_sentiment(usd_sigs, agg)
                    all_curs: set = set()
                    for s in new_signals:
                        if s.get("pair","") in _USD_RESTRICTED_PAIRS:
                            p = s.get("pair","")
                            if len(p) >= 6:
                                all_curs.add(p[:3]); all_curs.add(p[3:6])
                    news_calls = mkt_calls = len(all_curs)
                except Exception as e:
                    log.error(f"USD sentiment gate failed: {e} — signals pass through")
                    for s in usd_sigs:
                        s.update({"sentiment_applied": False, "sentiment_score": 0.0, "sentiment_adjustment": 0.0})
        else:
            for s in usd_sigs:
                s.update({"sentiment_applied": False, "sentiment_score": 0.0, "sentiment_adjustment": 0.0})
        new_signals = usd_sigs + gbp_sigs
    else:
        for s in new_signals:
            s.update({"sentiment_applied": False, "sentiment_score": 0.0, "sentiment_adjustment": 0.0})

    elite = select_high_potential(new_signals)
    for s in elite:
        s["estimated_win_rate"] = quick_micro_backtest(s)
        log.info(f"{s['pair']} [{s['tier']}] score={s.get('score',0)} est_wr={s['estimated_win_rate']:.2f}")

    log.info(f"Micro-backtest: {len(elite)} elite signals analysed")
    pass_ids = {s["signal_id"] for s in elite if s.get("estimated_win_rate", 0) >= 0.65}
    new_signals = [s for s in new_signals if s.get("tier") not in ("A+", "A") or s["signal_id"] in pass_ids]
    elite_ids = {s["signal_id"] for s in elite}
    for s in new_signals:
        if s["signal_id"] not in elite_ids: s["estimated_win_rate"] = None

    if BROWSERLESS_TOKEN and new_signals:
        try:
            new_signals = _validate_signal_prices(new_signals)
        except Exception as e:
            log.warning(f"Live price validation skipped: {e}")

    mb = split_signals_by_mode(new_signals)
    agg_f,  aw = check_risk_limits(mb["aggressive"],  opt_cfg, "aggressive",  existing_counts=existing_counts)
    cons_f, cw = check_risk_limits(mb["conservative"], opt_cfg, "conservative", existing_counts=existing_counts)
    for w in aw+cw: log.warning(w)

    all_new: List[Dict] = []
    seen: set = set()
    for sig in agg_f+cons_f:
        if sig["signal_id"] not in seen:
            all_new.append(sig); seen.add(sig["signal_id"])

    all_sigs = filter_expired_signals(all_new+existing)
    if len(all_sigs)>200:
        all_sigs = sorted(all_sigs, key=lambda s: s.get("timestamp",""), reverse=True)[:200]
        log.info("Signal cap: trimmed to 200")

    log.info(f"Complete | New: {len(all_new)} | Existing: {len(existing)} | Total: {len(all_sigs)}")
    log.info(f"Breakdown - Aggressive: {len(agg_f)} | Conservative: {len(cons_f)}")

    if len(all_new)==0 and len(existing)==0:
        sess = get_market_session()
        st   = CONFIG.get("advanced",{}).get("session_thresholds",{}).get(sess, min(CONFIG["settings"]["aggressive"]["threshold"], CONFIG["settings"]["conservative"]["threshold"]))
        log.info(f"Zero-signal diagnostic: session={sess} threshold={st}")

    write_dashboard_state(all_sigs, downloads, news_calls, mkt_calls, opt_cfg, opt_mode, None, pair_prices)

    if all_new:
        mb = split_signals_by_mode(all_new)
        print("\n" + "="*100)
        print("TRADE BEACON v2.2.2-SAFE — MULTI-MODE SIGNALS")
        print("="*100)
        for label, key, icon in [("AGGRESSIVE","aggressive","⚡"), ("CONSERVATIVE","conservative","🛡️")]:
            sigs = mb[key]
            if sigs:
                print(f"\n{icon} {label} SIGNALS ({len(sigs)})")
                print("-"*100)
                df = pd.DataFrame(sigs)
                print(df[["signal_id","pair","direction","score","tier","confidence","risk_reward","sentiment_score"]].to_string(index=False))
        print("="*100+"\n")
        pd.DataFrame(all_new).to_csv("signals.csv", index=False)
        log.info("signals.csv written")

    mark_success()
    log.info("Run completed — v2.2.2-SAFE")


if __name__ == "__main__":
    main()
