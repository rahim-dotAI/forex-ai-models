"""
Trade Beacon v2.1.2 - Forex Signal Generator (INSTITUTIONAL GRADE)
MULTI-MODE EDITION: Generates Aggressive + Conservative signals simultaneously
with tier-based selective enhancement (sentiment/backtest only on top-tier)

PERFORMANCE OPTIMIZATIONS v2.1.3:
- Lowered tier thresholds (A+: 75, A: 68, B: 60) to generate quality signals
- Balanced Conservative/Aggressive thresholds (55/50 instead of 60/60)
- Added session-specific thresholds (block ASIAN/LATE_US/OVERLAP poor performers)
- Added pair limits (max 2 GBPJPY, block EURGBP, allow 5 GBPUSD)
- Increased ATR stop multiplier (2.5x from 2.0x) to reduce stop-outs
- Increased minimum R:R (2.5 from 2.0) for better risk-adjusted trades

FIXED: Line 1162 - Changed PERFORMANCE_TRACKER.signals to PERFORMANCE_TRACKER.history.get("signals", [])
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
import requests
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

from performance_tracker import PerformanceTracker

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("trade-beacon")

# Mode validation
SIGNAL_ONLY_MODE = True
if not SIGNAL_ONLY_MODE:
    raise RuntimeError("❌ Execution logic disabled in signal-only mode")
log.info("🛡️ Signal-only mode validated")

# Configuration
PAIRS = ["USDJPY=X", "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDCHF=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X"]
SPREADS = {"USDJPY": 0.002, "EURUSD": 0.00015, "GBPUSD": 0.0002, "AUDUSD": 0.00018, "NZDUSD": 0.0002, "USDCAD": 0.0002, "USDCHF": 0.0002, "EURJPY": 0.002, "GBPJPY": 0.003, "EURGBP": 0.00015}
CORRELATED_PAIRS = [{"EURUSD=X", "GBPUSD=X"}, {"EURUSD=X", "EURGBP=X"}, {"GBPUSD=X", "EURGBP=X"}, {"USDJPY=X", "EURJPY=X"}, {"USDJPY=X", "GBPJPY=X"}, {"EURJPY=X", "GBPJPY=X"}]

INTERVAL = "15m"
LOOKBACK = "14d"
MIN_ROWS = 220
USE_VOLUME_FOR_FX = False

CONFIG = None
MODE = None
USE_SENTIMENT = False
SETTINGS = None
PERFORMANCE_TRACKER = None

# Unified pip calculation
def price_to_pips(pair: str, price_diff: float) -> float:
    return abs(price_diff) / (0.01 if "JPY" in pair else 0.0001)

def ensure_series(data):
    return data.iloc[:, 0].squeeze() if isinstance(data, pd.DataFrame) else data.squeeze()

# Retry decorator
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
                            log.warning(f"⚠️ Rate limited, waiting {wait}s...")
                            time.sleep(wait)
                        else:
                            raise
                    else:
                        raise
        return wrapper
    return decorator

# Config loader
def load_config():
    config_path = Path("config.json")
    if not config_path.exists():
        log.warning("⚠️ config.json not found, using defaults")
        return _default_config()
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    mode = config.get("mode", "conservative")
    if mode not in ["aggressive", "conservative"]:
        config["mode"] = "conservative"
    
    log.info(f"✅ Config loaded: mode={mode}, sentiment={config.get('use_sentiment', False)}")
    return config

def _default_config():
    return {
        "mode": "conservative",
        "use_sentiment": False,
        "settings": {
            "aggressive": {
                "threshold": 50,  # UPDATED: was 60
                "min_adx": 18,     # UPDATED: was 20
                "rsi_oversold": 30, 
                "rsi_overbought": 70, 
                "min_risk_reward": 2.5,  # UPDATED: was 2.0
                "atr_stop_multiplier": 2.5,  # UPDATED: was 1.8
                "atr_target_multiplier": 5.0,  # UPDATED: was 4.0
                "max_correlated_signals": 2
            },
            "conservative": {
                "threshold": 55,  # UPDATED: was 60
                "min_adx": 20, 
                "rsi_oversold": 30, 
                "rsi_overbought": 70, 
                "min_risk_reward": 2.5,  # UPDATED: was 2.2
                "atr_stop_multiplier": 2.5,  # UPDATED: was 2.0
                "atr_target_multiplier": 5.0,  # UPDATED: was 4.5
                "max_correlated_signals": 1
            }
        },
        "advanced": {
            "enable_session_filtering": True,
            "enable_correlation_filter": True,
            "cache_ttl_minutes": 5,
            "parallel_workers": 3,
            "session_bonuses": {
                "ASIAN": {"JPY_pairs": 3, "AUD_NZD_pairs": 3, "other": 0}, 
                "EUROPEAN": {"EUR_GBP_pairs": 3, "EUR_GBP_crosses": 2, "other": 0}, 
                "OVERLAP": {"all_major_pairs": 2}, 
                "US": {"USD_majors": 3, "other": 0}, 
                "LATE_US": {"all_major_pairs": 0}
            },
            "session_thresholds": {
                "ASIAN": 65,      # UPDATED: Block poor performance (33% WR)
                "EUROPEAN": 50,   # Keep normal (57% WR)
                "US": 50,         # Keep normal (77% WR)
                "LATE_US": 65,    # UPDATED: Block poor performance (42% WR)
                "OVERLAP": 999    # UPDATED: Block completely (0% WR)
            },
            "pair_limits": {
                "GBPUSD": 5,      # UPDATED: Allow more (100% WR)
                "GBPJPY": 2,      # UPDATED: Limit heavily (42% WR, was flooding with 33 signals)
                "EURGBP": 0,      # UPDATED: Block completely (0% WR)
                "default": 3      # Default limit for other pairs
            },
            "validation": {
                "max_signal_age_seconds": 900, 
                "min_sl_pips": {"JPY_pairs": 20, "other": 12}, 
                "max_spread_ratio": 0.25, 
                "max_sl_distance_pct": 0.02, 
                "max_tp_distance_pct": 0.05, 
                "require_direction": True, 
                "reject_missing_pips": True
            }
        },
        "risk_management": {
            "max_daily_risk_pips": 150, 
            "max_open_positions": 3, 
            "stop_trading_on_drawdown_pips": 100, 
            "equity_protection": {
                "enable": False, 
                "max_consecutive_losses": 3, 
                "pause_minutes_after_hit": 120
            }
        },
        "performance_tracking": {
            "enable": True, 
            "history_file": "signal_state/signal_history.json", 
            "idempotency": {"enabled": True}, 
            "analytics": {"track_by_pair": True}
        },
        "performance_tuning": {
            "auto_adjust_thresholds": False, 
            "min_trades_for_optimization": 50, 
            "target_win_rate": 0.5, 
            "optimization_inputs": {"min_expectancy": 0.5}
        }
    }

# API validation
def validate_api_keys():
    if not os.getenv("NEWSAPI_KEY") or not os.getenv("MARKETAUX_API_KEY"):
        return False
    try:
        r = requests.get("https://newsapi.org/v2/top-headlines", params={"country": "us", "pageSize": 1, "apiKey": os.getenv("NEWSAPI_KEY")}, timeout=5)
        if r.status_code in (401, 429):
            return False
        log.info("✅ NewsAPI validated")
    except:
        return False
    return True

CONFIG = load_config()
MODE = CONFIG["mode"]
USE_SENTIMENT = CONFIG.get("use_sentiment", False) and validate_api_keys()
SETTINGS = CONFIG["settings"][MODE]
CACHE_TTL = CONFIG.get("advanced", {}).get("cache_ttl_minutes", 5) * 60

if CONFIG.get("performance_tracking", {}).get("enable", True):
    try:
        PERFORMANCE_TRACKER = PerformanceTracker(history_file=CONFIG["performance_tracking"].get("history_file", "signal_state/signal_history.json"))
        log.info("✅ Performance tracker initialized")
    except Exception as e:
        log.error(f"⚠️ Tracker init failed: {e}")
        PERFORMANCE_TRACKER = None

# Cache
class MarketDataCache:
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self._cache = {}
        self._timestamps = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        with self._lock:
            if key not in self._cache or time.time() - self._timestamps.get(key, 0) > self.ttl:
                return None
            return self._cache[key]
    
    def set(self, key: str, value: pd.DataFrame):
        with self._lock:
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

_cache = MarketDataCache(ttl=CACHE_TTL)

# Technical indicators
def last(series: pd.Series):
    if series is None or series.empty:
        return None
    try:
        val = float(series.iloc[-1])
        return None if pd.isna(val) else val
    except:
        return None

@retry_with_backoff(max_retries=3, backoff_factor=10)
def download(pair: str) -> Tuple[pd.DataFrame, bool]:
    # Normalize cache key to avoid pollution
    cache_key = pair.replace("=X", "")
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached, True
    
    try:
        df = yf.download(pair, interval="15m", period=LOOKBACK, progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty or len(df) < MIN_ROWS:
            log.warning(f"⚠️ {pair} 15m insufficient, trying 1h...")
            df = yf.download(pair, interval="1h", period="60d", progress=False, auto_adjust=True, threads=False)
        
        if df is None or df.empty or len(df.dropna()) < MIN_ROWS:
            return pd.DataFrame(), False
        
        df = df.dropna()
        _cache.set(cache_key, df)
        return df, True
    except Exception as e:
        log.error(f"❌ {pair} download failed: {e}")
        return pd.DataFrame(), False

def ema(s, p): return EMAIndicator(s, window=p).ema_indicator()
def rsi(s, p=14): return RSIIndicator(s, window=p).rsi()
def adx_calc(h, l, c): return ADXIndicator(h, l, c, window=14).adx()
def atr_calc(h, l, c): return AverageTrueRange(h, l, c, window=14).average_true_range()
def get_spread(pair: str) -> float: return SPREADS.get(pair.replace("=X", ""), 0.0002)

# Market session
def get_market_session() -> str:
    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 8: return "ASIAN"
    elif 8 <= hour < 13: return "EUROPEAN"
    elif 13 <= hour < 16: return "OVERLAP"
    elif 16 <= hour < 21: return "US"
    else: return "LATE_US"

def calculate_dynamic_session_bonus(pair: str, session: str, config: Dict) -> int:
    """Calculate session-based score bonus with clear branch separation."""
    if not config.get("advanced", {}).get("enable_session_filtering", True):
        return 0
    bonuses = config.get("advanced", {}).get("session_bonuses", {}).get(session, {})
    
    if session == "ASIAN":
        if "JPY" in pair: return bonuses.get("JPY_pairs", 0)
        elif any(c in pair for c in ["AUD", "NZD"]): return bonuses.get("AUD_NZD_pairs", 0)
        return bonuses.get("other", 0)
    elif session == "EUROPEAN":
        if any(c in pair for c in ["EUR", "GBP"]) and pair not in ["EURUSD", "GBPUSD"]: 
            return bonuses.get("EUR_GBP_crosses", 0)
        elif any(c in pair for c in ["EUR", "GBP"]): 
            return bonuses.get("EUR_GBP_pairs", 0)
        return bonuses.get("other", 0)
    elif session == "OVERLAP":
        return bonuses.get("all_major_pairs", 0)
    elif session == "US":
        if "USD" in pair and pair in ["EURUSD", "GBPUSD", "USDCAD"]:
            return bonuses.get("USD_majors", 0)
        return bonuses.get("other", 0)
    elif session == "LATE_US":
        return bonuses.get("all_major_pairs", 0)
    
    return 0

# Signal helpers
def classify_market_state(adx: float, atr: float, entry: float) -> str:
    atr_pct = atr / entry if entry > 0 else 0
    if adx < 15: return "CHOPPY"
    elif adx > 25: return "TRENDING_STRONG_HIGH_VOL" if atr_pct > 0.002 else "TRENDING_STRONG"
    elif adx > 20: return "TRENDING_MODERATE"
    return "CONSOLIDATING"

def get_signal_type(e12: float, e26: float, e200: float, rsi: float, adx: float = None) -> str:
    rsi_low, rsi_high = (30, 70) if adx and adx > 30 else ((35, 65) if adx and adx > 25 else (40, 60))
    if e12 > e26 > e200: return "momentum" if rsi > rsi_high else "trend-continuation"
    elif e12 < e26 < e200: return "momentum" if rsi < rsi_low else "trend-continuation"
    elif (e12 > e26 and rsi < rsi_low) or (e12 < e26 and rsi > rsi_high): return "reversal"
    return "breakout"

def calculate_hold_time(rr: float, atr: float) -> str:
    if rr > 2.5 or atr > 0.002: return "SWING"
    elif rr > 1.8 or atr > 0.0015: return "INTRADAY"
    return "SHORT"

def calculate_eligible_modes(score: int, adx: float, config: Dict) -> List[str]:
    """
    Calculate eligible trading modes based on score and ADX only.
    In 'all' mode, returns all modes for every signal (no filtering).
    """
    # If mode is 'all', every signal qualifies for all modes
    if config.get("mode") == "all":
        return ["aggressive", "conservative"]
    
    # Legacy mode-specific filtering (backward compatibility)
    modes = []
    for mode_name in ["conservative", "aggressive"]:
        s = config["settings"][mode_name]
        if score >= s["threshold"] and adx >= s["min_adx"]:
            modes.append(mode_name)
    return modes

# NEW: Tier classification system - UPDATED THRESHOLDS
def classify_signal_tier(score: int) -> str:
    """
    Classify signal quality into tiers:
    A+ = Institutional grade (75+) - LOWERED from 80
    A  = Premium (68-74) - LOWERED from 72
    B  = Standard (60-67) - LOWERED from 65
    C  = Entry level (45-59) - LOWERED from 60
    
    PERFORMANCE OPTIMIZATION: Lowered all thresholds by 5-7 points
    to ensure we generate A/B tier signals instead of only C tier.
    """
    if score >= 75:
        return "A+"
    elif score >= 68:
        return "A"
    elif score >= 60:
        return "B"
    else:
        return "C"

def calculate_signal_freshness(ts: datetime) -> dict:
    age = (datetime.now(timezone.utc) - ts).total_seconds() / 60
    status = "FRESH" if age < 15 else ("RECENT" if age < 30 else ("AGING" if age < 60 else "STALE"))
    return {"status": status, "age_minutes": round(age, 1), "confidence_decay": round(max(0, 100 - age * 2), 1)}

def calculate_market_volatility(sigs: List[Dict]) -> str:
    if not sigs: return "CALM"
    avg_atr = sum(s.get("atr", 0) for s in sigs) / len(sigs)
    return "HIGH" if avg_atr > 0.002 else ("NORMAL" if avg_atr > 0.0015 else "CALM")

def calculate_market_sentiment(sigs: List[Dict]) -> str:
    if not sigs: return "MIXED"
    bull = sum(1 for s in sigs if s.get("direction") == "BUY")
    bear = sum(1 for s in sigs if s.get("direction") == "SELL")
    return "BULLISH" if bull > bear * 1.5 else ("BEARISH" if bear > bull * 1.5 else "MIXED")

# Signal validation
def validate_signal_quality(signal: Dict, config: Dict) -> Tuple[bool, List[str]]:
    warnings = []
    val_cfg = config.get("advanced", {}).get("validation", {})
    
    # In 'all' mode, use unified settings; otherwise use signal's first eligible mode
    if config.get("mode") == "all":
        mode_settings = config["settings"]["aggressive"]  # Use unified settings
    else:
        modes = signal.get("eligible_modes", ["conservative"])
        mode_settings = config["settings"].get(modes[0], config["settings"]["conservative"])
    
    # Direction
    if val_cfg.get("require_direction", True) and signal.get('direction') not in ("BUY", "SELL"):
        warnings.append(f"Invalid direction: {signal.get('direction')}")
        return False, warnings
    
    entry = signal.get('entry_price', 0)
    sl = signal.get('sl', 0)
    tp = signal.get('tp', 0)
    
    if entry <= 0 or abs(entry - sl) == 0:
        warnings.append("Invalid entry/SL")
        return False, warnings
    
    # Spread ratio
    spread = float(signal.get("spread", 0.0002))
    sl_dist = abs(entry - sl)
    spread_ratio = spread / sl_dist if sl_dist > 0 else 1
    if spread_ratio > val_cfg.get("max_spread_ratio", 0.25):
        warnings.append(f"High spread ratio: {spread_ratio:.1%}")
        return False, warnings
    
    # ATR validation
    atr = signal.get("atr")
    if atr is None or not isinstance(atr, (int, float)):
        warnings.append("Missing/invalid ATR")
        return False, warnings
    
    # Conservative static bounds (tightened for FX reality, JPY pairs need more headroom)
    atr_pct = atr / entry
    max_atr, min_atr = (0.006, 0.0001) if "JPY" in signal['pair'] else (0.005, 0.00001)
    if atr_pct <= min_atr or atr_pct > max_atr:
        warnings.append(f"Invalid ATR: {atr_pct:.4%}")
        return False, warnings
    
    # Pip validation
    sl_pips = price_to_pips(signal['pair'], sl_dist)
    min_sl = val_cfg.get("min_sl_pips", {}).get("JPY_pairs" if "JPY" in signal['pair'] else "other", 12)
    if val_cfg.get("reject_missing_pips", True) and sl_pips < 1:
        warnings.append(f"Missing pips: {sl_pips:.1f}")
        return False, warnings
    if sl_pips < min_sl:
        warnings.append(f"SL too tight: {sl_pips:.1f}")
        return False, warnings
    
    # R:R (use minimum from either mode) - UPDATED to 2.5
    min_rr = min(mode_settings.get("min_risk_reward", 2.5), 2.5)
    if signal['risk_reward'] < min_rr:
        warnings.append(f"Poor R:R: {signal['risk_reward']:.2f}")
        return False, warnings
    
    # SL/TP distance
    if abs(sl - entry) / entry > val_cfg.get("max_sl_distance_pct", 0.02):
        warnings.append(f"SL too far: {abs(sl - entry) / entry:.2%}")
        return False, warnings
    if abs(tp - entry) / entry > val_cfg.get("max_tp_distance_pct", 0.05):
        warnings.append(f"TP too far: {abs(tp - entry) / entry:.2%}")
        return False, warnings
    
    # Age
    try:
        sig_time = datetime.fromisoformat(signal['timestamp'].replace('Z', '+00:00'))
        age = (datetime.now(timezone.utc) - sig_time).total_seconds()
        if age > val_cfg.get("max_signal_age_seconds", 900):
            warnings.append(f"Stale: {age/60:.1f}min")
            return False, warnings
    except Exception as e:
        warnings.append(f"Invalid timestamp: {e}")
        return False, warnings
    
    return True, warnings

# Equity protection
def check_equity_protection(config: Dict) -> Tuple[bool, str]:
    if SIGNAL_ONLY_MODE:
        pause_file = Path("signal_state/trading_paused.json")
        if pause_file.exists():
            try:
                with open(pause_file, 'r') as f:
                    pause_data = json.load(f)
                    paused_until = datetime.fromisoformat(pause_data.get("paused_until", "").replace("Z", "+00:00"))
                    if datetime.now(timezone.utc) < paused_until:
                        remaining = (paused_until - datetime.now(timezone.utc)).total_seconds() / 60
                        return False, f"Paused for {remaining:.1f}min"
                    else:
                        pause_file.unlink()
            except:
                pass
    return True, ""

# Deterministic ID
def generate_deterministic_signal_id(pair: str, direction: str, entry: float, session: str, date: str) -> str:
    # Round entry to prevent float noise from creating duplicate IDs
    data = f"{pair}|{direction}|{round(entry, 5)}|{session}|{date}"
    return f"{pair}_{direction}_{hashlib.sha1(data.encode()).hexdigest()[:12]}"

# Duplicate detection
def get_existing_signals_today() -> List[str]:
    today = datetime.now(timezone.utc).date()
    existing_ids = set()
    
    # Dashboard
    dashboard_file = Path("signal_state/dashboard_state.json")
    if dashboard_file.exists():
        try:
            with open(dashboard_file, 'r') as f:
                data = json.load(f)
                for s in data.get("signals_by_mode", {}).get("aggressive", []) + data.get("signals_by_mode", {}).get("conservative", []):
                    try:
                        if datetime.fromisoformat(s.get("timestamp", "").replace('Z', '+00:00')).date() == today:
                            if sid := s.get("signal_id"):
                                existing_ids.add(sid)
                    except:
                        continue
        except:
            pass
    
    # Performance tracker
    if PERFORMANCE_TRACKER and hasattr(PERFORMANCE_TRACKER, 'history'):
        try:
            for trade in PERFORMANCE_TRACKER.history.get("signals", []):
                try:
                    if datetime.fromisoformat(trade.get("timestamp", "").replace('Z', '+00:00')).date() == today:
                        if sid := trade.get("signal_id"):
                            existing_ids.add(sid)
                except:
                    continue
        except:
            pass
    
    log.info(f"📋 Found {len(existing_ids)} existing signals today")
    return list(existing_ids)

def is_duplicate_signal(sid: str, existing: List[str]) -> bool:
    return sid in existing

# Optimization
def optimize_thresholds_if_needed(config: Dict) -> Dict:
    tuning = config.get("performance_tuning", {})
    if not tuning.get("auto_adjust_thresholds", False) or not PERFORMANCE_TRACKER:
        return config
    
    try:
        stats = PERFORMANCE_TRACKER.history.get("stats", {})
        if stats.get("total_trades", 0) < tuning.get("min_trades_for_optimization", 50):
            return config
        
        last_opt_file = Path("signal_state/last_optimization.json")
        if last_opt_file.exists():
            with open(last_opt_file, 'r') as f:
                last = datetime.fromisoformat(json.load(f).get("timestamp"))
                if (datetime.now(timezone.utc) - last).days < tuning.get("optimization_interval_days", 14):
                    return config
    except:
        return config
    
    opt_config = copy.deepcopy(config)
    wr = float(stats.get("win_rate", 0)) / 100 if stats.get("win_rate", 0) > 1 else float(stats.get("win_rate", 0))
    exp = stats.get("expectancy_pips", stats.get("expectancy", 0))
    
    mode = opt_config.get("mode", "conservative")
    curr_thresh = opt_config["settings"][mode]["threshold"]
    
    adj = 0
    if exp < tuning.get("optimization_inputs", {}).get("min_expectancy", 0.5):
        adj = 5
    elif wr < tuning.get("target_win_rate", 0.5) - 0.05:
        adj = 3
    elif wr > tuning.get("target_win_rate", 0.5) + 0.10 and exp > 1.0:
        adj = -3
    
    if adj != 0:
        new_thresh = max(55 if mode == "aggressive" else 60, min(70 if mode == "aggressive" else 75, curr_thresh + adj))
        opt_config["settings"][mode]["threshold"] = new_thresh
        log.info(f"✅ Threshold: {curr_thresh} → {new_thresh}")
        
        last_opt_file.parent.mkdir(exist_ok=True)
        with open(last_opt_file, 'w') as f:
            json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "old": curr_thresh, "new": new_thresh}, f)
    
    return opt_config

# Signal resolution
def resolve_active_signals():
    dashboard_file = Path("signal_state/dashboard_state.json")
    if not dashboard_file.exists():
        return 0
    
    try:
        with open(dashboard_file, 'r') as f:
            data = json.load(f)
            signals = data.get("signals_by_mode", {}).get("aggressive", []) + data.get("signals_by_mode", {}).get("conservative", [])
    except:
        return 0
    
    resolved = 0
    active = []
    
    for sig in signals:
        if sig.get("status") != "OPEN":
            active.append(sig)
            continue
        
        pair, sid, direction = sig.get("pair"), sig.get("signal_id"), sig.get("direction")
        entry, sl, tp = sig.get("entry_price"), sig.get("sl"), sig.get("tp")
        
        try:
            # Always clean and reconstruct to ensure consistency
            clean_pair = pair.replace("=X", "")
            ticker = f"{clean_pair}=X"
            df = yf.download(ticker, interval="1m", period="1d", progress=False, auto_adjust=True)
            if df is None or df.empty:
                active.append(sig)
                continue
            
            curr = float(df["Close"].iloc[-1])
            outcome = None
            
            if direction == "BUY":
                if curr <= sl: outcome, exit_p = "LOSS", sl
                elif curr >= tp: outcome, exit_p = "WIN", tp
            else:
                if curr >= sl: outcome, exit_p = "LOSS", sl
                elif curr <= tp: outcome, exit_p = "WIN", tp
            
            if not outcome:
                expires = sig.get('metadata', {}).get('expires_at')
                if expires and datetime.now(timezone.utc) > datetime.fromisoformat(expires.replace('Z', '+00:00')):
                    outcome, exit_p = "EXPIRED", curr
            
            if outcome:
                pips = price_to_pips(pair, abs(exit_p - entry))
                if outcome == "LOSS": pips = -pips
                elif outcome == "EXPIRED": pips = price_to_pips(pair, exit_p - entry) if direction == "BUY" else price_to_pips(pair, entry - exit_p)
                
                if PERFORMANCE_TRACKER:
                    PERFORMANCE_TRACKER.record_trade(
                        signal_id=sid, pair=pair, direction=direction,
                        entry_price=entry, exit_price=exit_p, sl=sl, tp=tp,
                        outcome=outcome, pips=pips,
                        confidence=sig.get("confidence"),
                        score=sig.get("score"),
                        session=sig.get("session"),
                        entry_time=sig.get("timestamp"),
                        exit_time=datetime.now(timezone.utc).isoformat(),
                        tier=sig.get("tier"),
                        eligible_modes=sig.get("eligible_modes"),
                        sentiment_applied=sig.get("sentiment_applied", False),
                        sentiment_score=sig.get("sentiment_score", 0.0),
                        sentiment_adjustment=sig.get("sentiment_adjustment", 0.0),
                        estimated_win_rate=sig.get("estimated_win_rate")
                    )
                    resolved += 1
                    log.info(f"{'✅' if outcome == 'WIN' else '❌'} {pair} {direction} - {outcome} ({pips:+.1f}p) [{sig.get('tier', 'C')}]")
            else:
                active.append(sig)
        except Exception as e:
            log.error(f"❌ Error resolving {sid}: {e}")
            active.append(sig)
    
    if resolved > 0:
        # Re-split active signals by mode
        mode_buckets = split_signals_by_mode(active)
        with open(dashboard_file, 'r') as f:
            dashboard_data = json.load(f)
        dashboard_data["signals_by_mode"] = mode_buckets
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        log.info(f"✅ Resolved {resolved}, {len(active)} active")
    
    return resolved

# NEW: Pair limit filter - PERFORMANCE OPTIMIZATION
def filter_pair_limits(signals: List[Dict], config: Dict) -> List[Dict]:
    """
    Limit signals per pair to prevent over-concentration.
    PERFORMANCE OPTIMIZATION: Prevents flooding (e.g., 33 GBPJPY signals)
    """
    pair_limits = config.get("advanced", {}).get("pair_limits", {
        "GBPUSD": 5,    # 100% WR - allow more
        "GBPJPY": 2,    # 42% WR - limit heavily
        "EURGBP": 0,    # 0% WR - block completely
        "default": 3    # Default limit for other pairs
    })
    
    pair_counts = {}
    filtered = []
    
    # Sort by score to keep best signals
    for sig in sorted(signals, key=lambda x: x['score'], reverse=True):
        pair = sig['pair']
        limit = pair_limits.get(pair, pair_limits.get("default", 3))
        
        # Block completely if limit is 0
        if limit == 0:
            log.info(f"⛔ Blocked {pair} (blocklist - 0% WR)")
            continue
        
        count = pair_counts.get(pair, 0)
        if count < limit:
            filtered.append(sig)
            pair_counts[pair] = count + 1
        else:
            log.info(f"⚠️ Skipped {pair} (max {limit} signals reached)")
    
    if len(filtered) < len(signals):
        log.info(f"📊 Pair limits: {len(signals)} → {len(filtered)} signals")
    
    return filtered

# Signal generation
def generate_signal(pair: str) -> Tuple[Optional[dict], bool]:
    df, ok = download(pair)
    if not ok or len(df) < MIN_ROWS:
        return None, ok
    
    try:
        close, high, low = ensure_series(df["Close"]), ensure_series(df["High"]), ensure_series(df["Low"])
        e12, e26, e200 = last(ema(close, 12)), last(ema(close, 26)), last(ema(close, 200))
        r, a, atr, curr = last(rsi(close)), last(adx_calc(high, low, close)), last(atr_calc(high, low, close)), last(close)
    except Exception as e:
        log.warning(f"⚠️ {pair} indicators failed: {e}")
        return None, ok
    
    # Minimum threshold check
    if CONFIG.get("mode") == "all":
        # In 'all' mode, use the single unified threshold from either setting
        min_threshold = CONFIG["settings"]["aggressive"]["threshold"]
        min_adx = CONFIG["settings"]["aggressive"]["min_adx"]
    else:
        # Legacy mode: use lowest threshold across modes
        min_threshold = min(CONFIG["settings"]["aggressive"]["threshold"], CONFIG["settings"]["conservative"]["threshold"])
        min_adx = min(CONFIG["settings"]["aggressive"]["min_adx"], CONFIG["settings"]["conservative"]["min_adx"])
    
    if None in (e12, e26, e200, r, a, curr, atr) or a < min_adx:
        return None, ok
    
    bull = bear = 0
    
    # EMA structure
    if e12 > e26 > e200: bull += 25
    elif e12 < e26 < e200: bear += 25
    
    # Pullback
    rsi_os, rsi_ob = 30, 70  # Use aggressive thresholds for signal generation
    if e12 > e26 > e200 and rsi_os + 5 < r < 45: bull += 15
    elif e12 < e26 < e200 and 55 < r < rsi_ob - 5: bear += 15
    
    # RSI
    if r < rsi_os: bull += 25
    elif r > rsi_ob: bear += 25
    
    # ADX
    if a > 25:
        if e12 > e26:
            bull += 20
        else:
            bear += 20
    elif a > min_adx:
        if e12 > e26:
            bull += 10
        else:
            bear += 10
    
    # Session
    session = get_market_session()
    bonus = calculate_dynamic_session_bonus(pair.replace("=X", ""), session, CONFIG)
    if e12 > e26:
        bull += bonus
    elif e12 < e26:
        bear += bonus
    
    diff = abs(bull - bear)
    
    # Reject ambiguous ties (bull == bear means no clear direction)
    if bull == bear:
        return None, ok
    
    # PERFORMANCE OPTIMIZATION: Session-specific threshold check
    session_thresholds = CONFIG.get("advanced", {}).get("session_thresholds", {})
    session_min_threshold = session_thresholds.get(session, min_threshold)
    
    if diff < session_min_threshold:
        # log.info(f"⚠️ {pair.replace('=X', '')} rejected: score {diff} < session threshold {session_min_threshold} ({session})")
        return None, ok
    
    if diff < min_threshold:
        return None, ok
    
    direction = "BUY" if bull > bear else "SELL"
    
    # Confidence
    if diff >= 75: conf = "VERY_STRONG"
    elif diff >= 65: conf = "STRONG"
    else: conf = "MODERATE"
    
    # NEW: Tier classification (with updated thresholds)
    tier = classify_signal_tier(diff)
    
    # SL/TP (use updated conservative multipliers as baseline)
    # PERFORMANCE OPTIMIZATION: Increased from 2.0/4.5 to 2.5/5.0
    spread = get_spread(pair)
    atr_stop, atr_tgt = 2.5, 5.0  # UPDATED: was 2.0, 4.5
    if direction == "BUY":
        sl, tp = curr - atr_stop * atr, curr + atr_tgt * atr
    else:
        sl, tp = curr + atr_stop * atr, curr - atr_tgt * atr
    
    rr = abs(tp - curr) / abs(curr - sl) if abs(curr - sl) > 0 else 0
    
    # PERFORMANCE OPTIMIZATION: Increased minimum R:R from 2.0 to 2.5
    if rr < 2.5:  # UPDATED: was 2.0
        return None, ok
    if rr > 10:  # Reject pathological spikes
        return None, ok
    
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=CONFIG.get("advanced", {}).get("validation", {}).get("max_signal_age_seconds", 900) / 60)
    
    sid = generate_deterministic_signal_id(pair.replace("=X", ""), direction, curr, session, now.strftime('%Y%m%d'))
    
    signal = {
        "signal_id": sid, "id": sid, "pair": pair.replace("=X", ""), "direction": direction,
        "score": diff, "technical_score": diff, "sentiment_score": 0, "confidence": conf,
        "tier": tier,  # NEW: Tier classification
        "rsi": round(r, 1), "adx": round(a, 1), "atr": round(atr, 5), 
        "volume_ratio": 0,
        "session": session, "entry_price": round(curr, 5), "sl": round(sl, 5), "tp": round(tp, 5),
        "risk_reward": round(rr, 2), "spread": round(spread, 5), "timestamp": now.isoformat(), "status": "OPEN",
        "hold_time": calculate_hold_time(rr, atr), "eligible_modes": calculate_eligible_modes(diff, a, CONFIG),
        "freshness": calculate_signal_freshness(now),
        "sentiment_applied": False,  # NEW: Track if sentiment was applied
        "metadata": {
            "signal_type": get_signal_type(e12, e26, e200, r, a), "market_state": classify_market_state(a, atr, curr),
            "timeframe": INTERVAL, "valid_for_minutes": 15, "generated_at": now.isoformat(), "expires_at": expires.isoformat(),
            "session_active": session in ("EUROPEAN", "US", "OVERLAP"), "signal_generator_version": "2.1.3-OPTIMIZED",
            "atr_stop_multiplier": atr_stop, "atr_target_multiplier": atr_tgt
        }
    }
    
    is_valid, warnings = validate_signal_quality(signal, CONFIG)
    if not is_valid:
        log.info(f"❌ {pair} rejected: {', '.join(warnings)}")
        return None, ok
    
    return signal, ok

# NEW: Multi-mode helpers
def split_signals_by_mode(signals: List[Dict]) -> Dict[str, List[Dict]]:
    """Split signals into mode-specific buckets"""
    buckets = {
        "aggressive": [],
        "conservative": []
    }
    
    for s in signals:
        for m in s.get("eligible_modes", []):
            if m in buckets:
                buckets[m].append(s)
    
    return buckets

def select_high_potential(signals: List[Dict]) -> List[Dict]:
    """Select top-tier signals for enhanced analysis"""
    return [s for s in signals if s.get("tier") in ("A+", "A")]

def quick_micro_backtest(signal: Dict) -> float:
    """
    Quick probabilistic win rate estimation based on signal characteristics
    Returns: Estimated win probability (0-1)
    """
    base = 0.5
    
    # Tier bonus
    tier = signal.get("tier", "C")
    if tier == "A+":
        base += 0.15
    elif tier == "A":
        base += 0.10
    elif tier == "B":
        base += 0.05
    
    # Confidence bonus
    if signal.get("confidence") == "VERY_STRONG":
        base += 0.05
    elif signal.get("confidence") == "STRONG":
        base += 0.03
    
    # Session bonus
    if signal.get("metadata", {}).get("session_active"):
        base += 0.02
    
    # Market state bonus
    if signal.get("metadata", {}).get("market_state") in ("TRENDING_STRONG", "TRENDING_STRONG_HIGH_VOL"):
        base += 0.03
    
    return min(base, 0.95)

# Correlation filter
def filter_correlated_signals_enhanced(signals: List[Dict], max_corr: int = 1, enabled: bool = True) -> List[Dict]:
    """
    Filter correlated signals regardless of direction.
    BUY EURUSD + SELL GBPUSD are still correlated and should be limited.
    
    If enabled=False, returns all signals without filtering.
    """
    if not enabled or len(signals) <= 1:
        return signals
    
    filtered = []
    groups = {}
    
    for sig in sorted(signals, key=lambda x: x['score'], reverse=True):
        pair = f"{sig['pair']}=X"
        
        group_key = None
        for corr_group in CORRELATED_PAIRS:
            if pair in corr_group:
                # Direction-agnostic correlation control
                group_key = frozenset(corr_group)
                break
        
        if group_key:
            count = groups.get(group_key, 0)
            if count < max_corr:
                filtered.append(sig)
                groups[group_key] = count + 1
            else:
                log.info(f"⚠️ Skipping {sig['pair']} {sig['direction']} (correlation limit)")
        else:
            filtered.append(sig)
    
    if len(filtered) < len(signals):
        log.info(f"🔗 Correlation filter: {len(signals)} → {len(filtered)}")
    return filtered

def check_risk_limits(signals: List[Dict], config: Dict, mode: str) -> Tuple[List[Dict], List[str]]:
    """Apply risk limits per mode with UPDATED pair filtering"""
    risk = config.get("risk_management", {})
    mode_settings = config["settings"][mode]
    warnings = []
    
    max_pos = risk.get("max_open_positions", 3)
    if len(signals) > max_pos:
        warnings.append(f"Limiting {mode} to {max_pos} positions")
        signals = sorted(signals, key=lambda x: x['score'], reverse=True)[:max_pos]
    
    max_daily = risk.get("max_daily_risk_pips", 150)
    total_risk = 0
    filtered = []
    
    for sig in signals:
        risk_pips = price_to_pips(sig.get('pair', ''), abs(sig.get('entry_price', 0) - sig.get('sl', 0)))
        if total_risk + risk_pips <= max_daily:
            filtered.append(sig)
            total_risk += risk_pips
        else:
            warnings.append(f"Skipped {sig['pair']} - {mode} risk limit")
    
    # PERFORMANCE OPTIMIZATION: Apply pair limits BEFORE correlation filter
    filtered = filter_pair_limits(filtered, config)
    
    if config.get("advanced", {}).get("enable_correlation_filter", True):
        max_corr = mode_settings.get("max_correlated_signals", 1)
        filtered = filter_correlated_signals_enhanced(filtered, max_corr, enabled=True)
    else:
        log.info(f"🔓 Correlation filter disabled - allowing all correlated pairs")
    
    return filtered, warnings

def enhance_with_sentiment(signals: List[Dict], news_agg) -> List[Dict]:
    """
    Apply sentiment analysis to high-confidence signals only.
    Uses both NewsAPI and MarketAux with smart rate limiting.
    Only analyzes signals with score >= 60 to conserve API quota.
    """
    if not signals:
        return signals
    
    try:
        # Get relevant currencies from signals
        currencies = set()
        for sig in signals:
            pair = sig.get("pair", "")
            # Extract base and quote currencies (e.g., GBPUSD -> GBP, USD)
            if len(pair) >= 6:
                currencies.add(pair[:3])  # Base currency
                currencies.add(pair[3:6])  # Quote currency
        
        # Fetch news for relevant currencies (limited to conserve API calls)
        currency_sentiment = {}
        for currency in currencies:
            try:
                sentiment_score = news_agg.get_currency_sentiment(currency)
                currency_sentiment[currency] = sentiment_score
                log.info(f"📰 {currency} sentiment: {sentiment_score:+.2f}")
            except Exception as e:
                log.warning(f"⚠️ Failed to get {currency} sentiment: {e}")
                currency_sentiment[currency] = 0.0
        
        # Apply sentiment to each signal
        for sig in signals:
            pair = sig.get("pair", "")
            if len(pair) < 6:
                sig["sentiment_score"] = 0.0
                sig["sentiment_applied"] = True
                continue
            
            base_curr = pair[:3]
            quote_curr = pair[3:6]
            
            # Calculate sentiment: positive for base currency = bullish signal
            base_sentiment = currency_sentiment.get(base_curr, 0.0)
            quote_sentiment = currency_sentiment.get(quote_curr, 0.0)
            
            # Net sentiment: base bullish + quote bearish = buy signal gets boost
            direction = sig.get("direction", "BUY")
            if direction == "BUY":
                # Buying base currency (bullish on base, bearish on quote is good)
                net_sentiment = base_sentiment - quote_sentiment
            else:  # SELL
                # Selling base currency (bearish on base, bullish on quote is good)
                net_sentiment = quote_sentiment - base_sentiment
            
            # Apply sentiment boost/penalty to score
            original_score = sig.get("score", 50)
            sentiment_adjustment = net_sentiment * 5  # Scale: -10 to +10 points
            
            # Cap adjustment to ±10 points
            sentiment_adjustment = max(-10, min(10, sentiment_adjustment))
            
            adjusted_score = original_score + sentiment_adjustment
            adjusted_score = max(0, min(100, adjusted_score))  # Keep in 0-100 range
            
            # Update signal
            sig["sentiment_score"] = round(net_sentiment, 2)
            sig["sentiment_adjustment"] = round(sentiment_adjustment, 1)
            sig["score_before_sentiment"] = original_score
            sig["score"] = round(adjusted_score, 0)
            sig["sentiment_applied"] = True
            
            log.info(f"💭 {pair} {direction}: sentiment={net_sentiment:+.2f}, "
                    f"score {original_score} → {adjusted_score} ({sentiment_adjustment:+.1f})")
        
        return signals
        
    except Exception as e:
        log.error(f"❌ Sentiment enhancement failed: {e}")
        # Mark as attempted but failed
        for sig in signals:
            sig["sentiment_score"] = 0
            sig["sentiment_applied"] = True
        return signals


class NewsAggregator:
    """Aggregates news sentiment from NewsAPI and MarketAux APIs"""
    
    def __init__(self):
        self.newsapi_key = os.getenv("NEWSAPI_KEY")
        self.marketaux_key = os.getenv("MARKETAUX_API_KEY")
        self.cache = {}
        self.cache_ttl = 1800  # 30 minutes
        self.last_call_time = {}
        self.min_call_interval = 2  # 2 seconds between calls
        
    def _rate_limit(self, api_name: str):
        """Enforce rate limiting between API calls"""
        last_call = self.last_call_time.get(api_name, 0)
        elapsed = time.time() - last_call
        if elapsed < self.min_call_interval:
            time.sleep(self.min_call_interval - elapsed)
        self.last_call_time[api_name] = time.time()
    
    def _is_cached(self, key: str) -> bool:
        """Check if data is in cache and not expired"""
        if key not in self.cache:
            return False
        cached_time, _ = self.cache[key]
        return (time.time() - cached_time) < self.cache_ttl
    
    def _get_cached(self, key: str):
        """Get cached data"""
        if self._is_cached(key):
            _, data = self.cache[key]
            return data
        return None
    
    def _set_cache(self, key: str, data):
        """Set cache data"""
        self.cache[key] = (time.time(), data)
    
    @retry_with_backoff(max_retries=2, backoff_factor=5)
    def get_currency_sentiment(self, currency: str) -> float:
        """
        Get sentiment score for a currency from news sources.
        Returns: float between -1.0 (very bearish) and +1.0 (very bullish)
        """
        cache_key = f"sentiment_{currency}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        sentiment_scores = []
        
        # Try NewsAPI first
        if self.newsapi_key:
            try:
                self._rate_limit("newsapi")
                newsapi_sentiment = self._fetch_newsapi_sentiment(currency)
                if newsapi_sentiment is not None:
                    sentiment_scores.append(newsapi_sentiment)
            except Exception as e:
                log.warning(f"⚠️ NewsAPI failed for {currency}: {e}")
        
        # Try MarketAux
        if self.marketaux_key:
            try:
                self._rate_limit("marketaux")
                marketaux_sentiment = self._fetch_marketaux_sentiment(currency)
                if marketaux_sentiment is not None:
                    sentiment_scores.append(marketaux_sentiment)
            except Exception as e:
                log.warning(f"⚠️ MarketAux failed for {currency}: {e}")
        
        # Average the scores
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            self._set_cache(cache_key, avg_sentiment)
            return avg_sentiment
        
        # No sentiment data available
        return 0.0
    
    def _fetch_newsapi_sentiment(self, currency: str) -> Optional[float]:
        """Fetch sentiment from NewsAPI"""
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"{currency} currency forex",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10,
            "apiKey": self.newsapi_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None
        
        data = response.json()
        articles = data.get("articles", [])
        
        if not articles:
            return 0.0
        
        # Simple sentiment analysis based on keywords
        positive_keywords = ["surge", "rally", "gain", "rise", "strength", "bullish", "up", "higher", "strong"]
        negative_keywords = ["fall", "drop", "decline", "weak", "bearish", "down", "lower", "plunge", "crash"]
        
        sentiment_sum = 0
        for article in articles:
            title = (article.get("title", "") + " " + article.get("description", "")).lower()
            
            positive_count = sum(1 for word in positive_keywords if word in title)
            negative_count = sum(1 for word in negative_keywords if word in title)
            
            if positive_count > negative_count:
                sentiment_sum += 1
            elif negative_count > positive_count:
                sentiment_sum -= 1
        
        # Normalize to -1.0 to +1.0 range
        return max(-1.0, min(1.0, sentiment_sum / max(len(articles), 1)))
    
    def _fetch_marketaux_sentiment(self, currency: str) -> Optional[float]:
        """Fetch sentiment from MarketAux"""
        url = "https://api.marketaux.com/v1/news/all"
        params = {
            "symbols": f"{currency}USD",
            "filter_entities": "true",
            "language": "en",
            "limit": 10,
            "api_token": self.marketaux_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None
        
        data = response.json()
        articles = data.get("data", [])
        
        if not articles:
            return 0.0
        
        # MarketAux provides sentiment in entities
        sentiment_sum = 0
        sentiment_count = 0
        
        for article in articles:
            entities = article.get("entities", [])
            for entity in entities:
                if currency in entity.get("symbol", ""):
                    sentiment_score = entity.get("sentiment_score", 0)
                    # MarketAux sentiment is typically -1 to +1
                    sentiment_sum += sentiment_score
                    sentiment_count += 1
        
        if sentiment_count > 0:
            return sentiment_sum / sentiment_count
        
        return 0.0

# Dashboard
def calculate_daily_pips(signals: List[Dict]) -> float:
    """Calculate total risk in pips for today's signals (uses SL distance, not TP)"""
    today = datetime.now(timezone.utc).date()
    return round(sum(
        price_to_pips(s.get('pair', ''), abs(s.get('entry_price', 0) - s.get('sl', 0)))
        for s in signals
        if datetime.fromisoformat(s.get('timestamp', '')).date() == today
    ), 1)

def get_performance_summary() -> Dict:
    if not PERFORMANCE_TRACKER:
        return {"stats": {}, "analytics": {}, "equity": {}}
    try:
        return PERFORMANCE_TRACKER.get_dashboard_summary()
    except:
        return {"stats": {}, "analytics": {}, "equity": {}}

def filter_expired_signals(signals: List[Dict]) -> List[Dict]:
    now = datetime.now(timezone.utc)
    active = []
    
    for sig in signals:
        status = sig.get("status", "OPEN")
        if status == "ACTIVE":
            sig["status"] = "OPEN"
            status = "OPEN"
        
        if status != "OPEN":
            continue
        
        try:
            expires = sig.get('metadata', {}).get('expires_at')
            if expires and now < datetime.fromisoformat(expires.replace('Z', '+00:00')):
                active.append(sig)
            elif not expires:
                sig_time = datetime.fromisoformat(sig['timestamp'].replace('Z', '+00:00'))
                if (now - sig_time).total_seconds() < CONFIG.get("advanced", {}).get("validation", {}).get("max_signal_age_seconds", 900):
                    active.append(sig)
        except:
            # Discard signals with corrupt timestamps
            continue
    
    if len(active) < len(signals):
        log.info(f"⏰ Filtered {len(signals) - len(active)} expired")
    return active

def write_dashboard_state(signals: list, downloads: int, news_calls: int = 0, mkt_calls: int = 0, config: Dict = None, mode: str = None, settings: Dict = None):
    cfg = config or CONFIG
    md = mode or MODE
    signals = filter_expired_signals(signals)
    
    # NEW: Split signals by mode instead of filtering
    mode_buckets = split_signals_by_mode(signals)
    
    perf = get_performance_summary()
    stats = perf.get("stats", {}) or {}
    can_trade, pause = check_equity_protection(cfg)
    
    # Calculate stats per mode
    agg_count = len(mode_buckets["aggressive"])
    cons_count = len(mode_buckets["conservative"])
    
    # Get historical signals from tracker - FIXED LINE 1162
    historical_signals = []
    if PERFORMANCE_TRACKER:
        try:
            from datetime import timedelta
            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
            
            # ✅ FIXED: Use history.get() instead of .signals attribute
            all_signals = PERFORMANCE_TRACKER.history.get("signals", [])
            
            for s in all_signals:
                if s.get("status") not in ["WIN", "LOSS", "EXPIRED"]:
                    continue
                try:
                    ts = datetime.fromisoformat(s.get("timestamp", "").replace('Z', '+00:00'))
                    if ts >= seven_days_ago:
                        historical_signals.append(s)
                except:
                    continue  # Skip signals with invalid timestamps
            
            log.info(f"📊 Loaded {len(historical_signals)} historical signals from tracker")
        except Exception as e:
            log.warning(f"Could not load historical signals: {e}")
    
    # If tracker is empty, try loading from existing dashboard
    if not historical_signals:
        dashboard_file = Path("signal_state/dashboard_state.json")
        if dashboard_file.exists():
            try:
                with open(dashboard_file) as f:
                    old_data = json.load(f)
                    historical_signals = old_data.get("historical_signals", [])
                    log.info(f"📊 Loaded {len(historical_signals)} from existing dashboard")
            except:
                pass
    
    dashboard = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_signals": len(signals),
        "active_signals_by_mode": {
            "aggressive": agg_count,
            "conservative": cons_count
        },
        "session": get_market_session(),
        "mode": md,  # Keep for backward compatibility
        "sentiment_enabled": USE_SENTIMENT,
        "multi_mode": True,  # NEW: Flag for multi-mode operation
        "equity_protection": {"enabled": cfg.get("risk_management", {}).get("equity_protection", {}).get("enable", False), "can_trade": can_trade, "pause_reason": pause or None},
        "market_state": {"volatility": calculate_market_volatility(signals), "sentiment_bias": calculate_market_sentiment(signals), "session": get_market_session()},
        "signals_by_mode": mode_buckets,  # NEW: Signals split by mode
        "historical_signals": historical_signals,  # NEW: Historical data for dashboard
        "api_usage": {"yfinance": {"successful_downloads": downloads}, "sentiment": {"enabled": USE_SENTIMENT, "newsapi": news_calls, "marketaux": mkt_calls}},
        "stats": {"total_trades": stats.get("total_trades", 0), "win_rate": stats.get("win_rate", 0), "total_pips": stats.get("total_pips", 0), "wins": stats.get("wins", 0), "losses": stats.get("losses", 0), "expectancy": stats.get("expectancy_pips", 0)},
        "performance_stats": {
            "total_trades": stats.get("total_trades", 0),
            "wins": stats.get("wins", 0),
            "losses": stats.get("losses", 0),
            "expired": stats.get("expired", 0),
            "win_rate": stats.get("win_rate", 0),
            "total_pips": stats.get("total_pips", 0),
            "avg_win": stats.get("avg_win_pips", 0),
            "avg_loss": stats.get("avg_loss_pips", 0),
            "expectancy": stats.get("expectancy_pips", 0),
            "by_mode": perf.get("analytics", {}).get("by_mode", {}),
            "by_tier": perf.get("analytics", {}).get("by_tier", {}),
            "by_session": perf.get("analytics", {}).get("by_session", {}),
            "by_pair": perf.get("analytics", {}).get("by_pair", {}),
        },
        "risk_management": {"theoretical_max_pips": calculate_daily_pips(signals), "total_risk_pips": sum(price_to_pips(s.get('pair', ''), abs(s.get('entry_price', 0) - s.get('sl', 0))) for s in signals), "max_daily_risk": cfg.get("risk_management", {}).get("max_daily_risk_pips", 150)},
        "analytics": perf.get("analytics", {}),
        "equity_curve": perf.get("equity", {}).get("curve", []),
        "system": {"last_update": datetime.now(timezone.utc).isoformat(), "signal_only_mode": SIGNAL_ONLY_MODE, "version": "2.1.3-OPTIMIZED"}
    }
    
    output_dir = Path("signal_state")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "dashboard_state.json", 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    log.info(f"📊 Dashboard written")
    log.info(f"📊 Aggressive: {agg_count} | Conservative: {cons_count} | Total: {len(signals)}")
    if stats.get("total_trades", 0) > 0:
        log.info(f"📈 {stats['total_trades']} trades | WR: {stats['win_rate']:.1f}% | Pips: {stats['total_pips']:.1f}")
    
    write_health_check(signals, downloads, news_calls, mkt_calls, can_trade, pause, md)

def write_health_check(signals, downloads, news, mkt, can_trade, pause, mode):
    status = "paused" if not can_trade else ("warning" if downloads == 0 or (len(signals) == 0 and downloads > 0 and can_trade) else "ok")
    issues = [pause] if pause else (["No data"] if downloads == 0 else (["No signals"] if len(signals) == 0 and downloads > 0 else []))
    
    health = {
        "status": status,
        "last_run": datetime.now(timezone.utc).isoformat(),
        "signal_count": len(signals),
        "issues": issues,
        "can_trade": can_trade,
        "api_status": {"yfinance": "ok" if downloads > 0 else "degraded", "newsapi": "ok" if news > 0 else "disabled", "marketaux": "ok" if mkt > 0 else "disabled"},
        "system_info": {"mode": mode, "pairs_monitored": len(PAIRS), "version": "2.1.3-OPTIMIZED"}
    }
    
    with open(Path("signal_state/health.json"), "w") as f:
        json.dump(health, f, indent=2)
    
    log.info(f"{'✅' if status == 'ok' else '⚠️'} Health: {status.upper()}")

# Time guard
def in_execution_window():
    last_run = Path("signal_state/last_run.txt")
    success = Path("signal_state/last_success.txt")
    now = datetime.now(timezone.utc)
    
    if last_run.exists():
        try:
            last = datetime.fromisoformat(last_run.read_text().strip().replace("Z", "+00:00"))
            if success.exists():
                last_ok = datetime.fromisoformat(success.read_text().strip().replace("Z", "+00:00"))
                if now - last_ok < timedelta(minutes=10):
                    log.info(f"⏱ Already ran at {last_ok}")
                    return False
            elif now - last < timedelta(minutes=2):
                log.info(f"⏱ Waiting for retry window")
                return False
        except:
            pass
    
    last_run.parent.mkdir(exist_ok=True)
    last_run.write_text(now.isoformat())
    return True

def mark_success():
    Path("signal_state/last_success.txt").parent.mkdir(exist_ok=True)
    Path("signal_state/last_success.txt").write_text(datetime.now(timezone.utc).isoformat())

# Cleanup
def cleanup_legacy_signals():
    dashboard = Path("signal_state/dashboard_state.json")
    if not dashboard.exists():
        return
    
    try:
        data = json.loads(dashboard.read_text())
        
        # Handle both old and new format
        if "signals_by_mode" in data:
            all_signals = data["signals_by_mode"].get("aggressive", []) + data["signals_by_mode"].get("conservative", [])
        else:
            all_signals = data.get("signals", [])
        
        now = datetime.now(timezone.utc)
        cleaned = []
        
        for sig in all_signals:
            if sig.get("status") == "ACTIVE":
                sig["status"] = "OPEN"
            if sig.get("status") != "OPEN":
                continue
            
            expires = sig.get('metadata', {}).get('expires_at')
            if expires:
                try:
                    if now < datetime.fromisoformat(expires.replace('Z', '+00:00')):
                        cleaned.append(sig)
                except:
                    pass
        
        if len(cleaned) < len(all_signals):
            # Re-split cleaned signals
            mode_buckets = split_signals_by_mode(cleaned)
            data["signals_by_mode"] = mode_buckets
            dashboard.write_text(json.dumps(data, indent=2))
            log.info(f"🧹 Cleaned {len(all_signals) - len(cleaned)} stale signals")
    except:
        pass

# Main
def main():
    cleanup_legacy_signals()
    
    if not in_execution_window():
        return
    
    if PERFORMANCE_TRACKER:
        resolve_active_signals()
        time.sleep(1)
    
    # Load existing active signals
    existing = []
    dashboard = Path("signal_state/dashboard_state.json")
    if dashboard.exists():
        try:
            data = json.loads(dashboard.read_text())
            if "signals_by_mode" in data:
                existing = data["signals_by_mode"].get("aggressive", []) + data["signals_by_mode"].get("conservative", [])
                existing = [s for s in existing if s.get("status", "OPEN") in ("OPEN", "ACTIVE")]
            else:
                existing = [s for s in data.get("signals", []) if s.get("status", "OPEN") in ("OPEN", "ACTIVE")]
            log.info(f"📋 {len(existing)} active signals from previous cycles")
        except:
            pass
    
    can_trade, pause = check_equity_protection(CONFIG)
    if not can_trade:
        log.warning(f"⏸️ {pause}")
        write_dashboard_state(filter_expired_signals(existing), 0, 0, 0, CONFIG, MODE, SETTINGS)
        return
    
    opt_cfg = optimize_thresholds_if_needed(CONFIG) if CONFIG.get("performance_tuning", {}).get("auto_adjust_thresholds", False) else CONFIG
    opt_mode = opt_cfg["mode"]
    
    log.info(f"🚀 Trade Beacon v2.1.3-OPTIMIZED - {'Generating ALL SIGNALS (Maximum Mode)' if opt_cfg.get('mode') == 'all' else 'Generating ALL signal types'} | Sentiment={'ON' if USE_SENTIMENT else 'OFF'}")
    log.info(f"📊 Monitoring {len(PAIRS)} pairs")
    if opt_cfg.get("mode") == "all":
        log.info(f"🎯 ALL MODE: Score≥{opt_cfg['settings']['aggressive']['threshold']} ADX≥{opt_cfg['settings']['aggressive']['min_adx']} (No mode filtering)")
    else:
        log.info(f"🎯 Aggressive: Score≥{opt_cfg['settings']['aggressive']['threshold']} ADX≥{opt_cfg['settings']['aggressive']['min_adx']}")
        log.info(f"🎯 Conservative: Score≥{opt_cfg['settings']['conservative']['threshold']} ADX≥{opt_cfg['settings']['conservative']['min_adx']}")
    
    new_signals = []
    downloads = 0
    existing_ids = get_existing_signals_today()
    
    _cache.clear()
    
    max_workers = opt_cfg.get("advanced", {}).get("parallel_workers", 3)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(PAIRS))) as executor:
        futures = {executor.submit(generate_signal, pair): pair for pair in PAIRS}
        
        for future in as_completed(futures):
            pair = futures[future]
            try:
                sig, ok = future.result()
                if ok:
                    downloads += 1
                
                if sig:
                    if is_duplicate_signal(sig['signal_id'], existing_ids):
                        log.info(f"⏭️ {pair.replace('=X', '')} - Duplicate skipped")
                        continue
                    
                    new_signals.append(sig)
                    modes_str = "+".join(sig['eligible_modes'])
                    log.info(f"✅ {pair.replace('=X', '')} - Score: {sig['score']} [{sig['tier']}] ({modes_str}) RR: {sig['risk_reward']:.2f}")
            except Exception as e:
                log.error(f"❌ {pair.replace('=X', '')} failed: {e}")
            
            time.sleep(0.5)
    
    # Apply sentiment ONLY to high-confidence signals (score >= 60) to conserve API quota
    if USE_SENTIMENT and new_signals:
        try:
            # Filter for high-confidence signals only (score >= 60)
            high_confidence = [s for s in new_signals if s.get("score", 0) >= 60]

            if high_confidence:
                news = NewsAggregator()
                log.info(f"💭 Applying sentiment to {len(high_confidence)}/{len(new_signals)} high-confidence signals (score ≥60)")

                enhanced = enhance_with_sentiment(high_confidence, news)

                # ✅ FIXED: Merge enhanced signals back by signal_id
                enhanced_map = {s["signal_id"]: s for s in enhanced}
                for i, sig in enumerate(new_signals):
                    if sig["signal_id"] in enhanced_map:
                        new_signals[i] = enhanced_map[sig["signal_id"]]
                    else:
                        new_signals[i]["sentiment_applied"] = False
                        new_signals[i]["sentiment_score"] = 0

                log.info(f"✅ Sentiment analysis complete: {len(enhanced)} signals enhanced")
            else:
                log.info(f"ℹ️  No signals meet sentiment threshold (score ≥60), skipping to conserve API quota")
                for s in new_signals:
                    s["sentiment_applied"] = False
                    s["sentiment_score"] = 0
        except Exception as e:
            log.error(f"❌ Sentiment failed: {e}")
            for s in new_signals:
                s["sentiment_applied"] = False
                s["sentiment_score"] = 0
    
    # Apply micro-backtest to high-potential signals (A+ and A tier)
    # Uses sentiment-adjusted scores if sentiment was applied
    elite = select_high_potential(new_signals)
    for s in elite:
        # Use sentiment-adjusted score if available
        effective_score = s.get("score", 0)
        s["estimated_win_rate"] = quick_micro_backtest(s)
        log.info(f"🎲 {s['pair']} [{s['tier']}] sentiment_applied={s.get('sentiment_applied', False)} score={effective_score} estimated_wr={s['estimated_win_rate']:.2f}")
    
    # Filter by estimated win rate (keep only >55% probability)
    log.info(f"🎲 Micro-backtest: {len(elite)} elite signals analyzed")
    high_probability = [s for s in elite if s.get("estimated_win_rate", 0) >= 0.55]
    if len(high_probability) < len(elite):
        log.info(f"📊 Win-rate filter: {len(elite)} → {len(high_probability)} signals (≥55% estimated WR)")
    
    # CRITICAL: Remove low-probability elite signals from new_signals
    elite_pass_ids = {s["signal_id"] for s in high_probability}
    new_signals = [
        s for s in new_signals
        if s.get("tier") not in ("A+", "A") or s.get("signal_id") in elite_pass_ids
    ]
    
    # Mark all signals with backtest data
    elite_ids = {s["signal_id"] for s in elite}
    for s in new_signals:
        if s["signal_id"] not in elite_ids:
            s["estimated_win_rate"] = None
    
    # Split signals by mode and apply mode-specific risk limits
    mode_buckets = split_signals_by_mode(new_signals)
    
    # Apply risk limits per mode (includes pair limits now)
    agg_filtered, agg_warnings = check_risk_limits(mode_buckets["aggressive"], opt_cfg, "aggressive")
    cons_filtered, cons_warnings = check_risk_limits(mode_buckets["conservative"], opt_cfg, "conservative")
    
    for w in agg_warnings + cons_warnings:
        log.warning(f"⚠️ {w}")
    
    # Combine all signals (they're already in mode buckets, but we need flat list for existing merge)
    all_new = []
    seen = set()
    for sig in agg_filtered + cons_filtered:
        if sig["signal_id"] not in seen:
            all_new.append(sig)
            seen.add(sig["signal_id"])
    
    all_signals = filter_expired_signals(all_new + existing)
    
    # HARDENING: Prevent JSON bloat - keep only most recent signals
    MAX_STORED_SIGNALS = 200
    if len(all_signals) > MAX_STORED_SIGNALS:
        # Sort by timestamp descending and keep most recent
        all_signals = sorted(all_signals, key=lambda s: s.get("timestamp", ""), reverse=True)[:MAX_STORED_SIGNALS]
        log.info(f"📦 Signal cap applied: trimmed to {MAX_STORED_SIGNALS} most recent")
    
    log.info(f"\n✅ Complete | New: {len(all_new)} | Existing: {len(existing)} | Total: {len(all_signals)}")
    log.info(f"📊 Breakdown - Aggressive: {len(agg_filtered)} | Conservative: {len(cons_filtered)}")
    
    write_dashboard_state(all_signals, downloads, 0, 0, opt_cfg, opt_mode, None)
    
    if all_new:
        # Print both mode sections
        mode_buckets = split_signals_by_mode(all_new)
        
        print("\n" + "="*100)
        print(f"🎯 MULTI-MODE SIGNALS (v2.1.3-OPTIMIZED - INSTITUTIONAL GRADE)")
        print("="*100)
        
        if mode_buckets["aggressive"]:
            print(f"\n⚡ AGGRESSIVE SIGNALS ({len(mode_buckets['aggressive'])})")
            print("-"*100)
            df_agg = pd.DataFrame(mode_buckets["aggressive"])
            print(df_agg[["signal_id", "pair", "direction", "score", "tier", "confidence", "risk_reward"]].to_string(index=False))
        
        if mode_buckets["conservative"]:
            print(f"\n🛡️ CONSERVATIVE SIGNALS ({len(mode_buckets['conservative'])})")
            print("-"*100)
            df_cons = pd.DataFrame(mode_buckets["conservative"])
            print(df_cons[["signal_id", "pair", "direction", "score", "tier", "confidence", "risk_reward"]].to_string(index=False))
        
        print("="*100 + "\n")
        
        # Also save CSV with mode info
        all_df = pd.DataFrame(all_new)
        all_df.to_csv("signals.csv", index=False)
        log.info("📄 signals.csv written")
    
    mark_success()
    log.info("✅ Run completed - v2.1.3-OPTIMIZED INSTITUTIONAL GRADE")

if __name__ == "__main__":
    main()
