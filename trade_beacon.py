"""
Trade Beacon v2.0.3 - Forex Signal Generator (COMPLETE ENHANCED VERSION)
==========================================

ENHANCEMENTS (v2.0.3):
- Configurable ATR multipliers and risk-reward minimums
- Enhanced signal validation (stale signals, unrealistic prices)
- Dynamic session bonuses from config
- Risk management limits (max positions, daily risk, correlation)
- Performance-based optimization support
- Mode-specific parameter loading

This is a SIGNAL GENERATOR ONLY - no trade execution logic.
"""

import logging
import sys
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import time
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import yfinance as yf
import requests
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# Import performance tracker
from performance_tracker import track_performance

# =========================
# SIGNAL GENERATOR MODE - CRITICAL SAFEGUARD
# =========================
SIGNAL_ONLY_MODE = True  # NEVER EXECUTE TRADES - THIS IS A SIGNAL GENERATOR

def validate_signal_mode():
    """
    Ensure this system never executes trades.
    
    This is a SIGNAL GENERATOR, not a trading system. It produces market intelligence
    and opportunity notifications, but NEVER places orders or manages positions.
    """
    if not SIGNAL_ONLY_MODE:
        raise RuntimeError(
            "❌ CRITICAL: Execution logic is disabled in signal generator mode. "
            "This system produces signals only. For trade execution, use a separate trading bot."
        )
    
    log.info("🛡️ Signal-only mode validated - No execution logic will run")

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("trade-beacon")

# Validate mode on startup
validate_signal_mode()

# =========================
# CONFIGURATION
# =========================
# Expanded pair list for more opportunities
PAIRS = [
    "USDJPY=X",   # Major - USD/JPY
    "EURUSD=X",   # Major - EUR/USD
    "GBPUSD=X",   # Major - GBP/USD
    "AUDUSD=X",   # Major - AUD/USD
    "NZDUSD=X",   # Major - NZD/USD
    "USDCAD=X",   # Major - USD/CAD
    "USDCHF=X",   # Major - USD/CHF
    "EURJPY=X",   # Cross - EUR/JPY
    "GBPJPY=X",   # Cross - GBP/JPY
    "EURGBP=X",   # Cross - EUR/GBP
]

# Typical forex spreads in pips (conservative estimates)
SPREADS = {
    "USDJPY": 0.002,    # ~2 pips
    "EURUSD": 0.00015,  # ~1.5 pips
    "GBPUSD": 0.0002,   # ~2 pips
    "AUDUSD": 0.00018,  # ~1.8 pips
    "NZDUSD": 0.0002,   # ~2 pips
    "USDCAD": 0.0002,   # ~2 pips
    "USDCHF": 0.0002,   # ~2 pips
    "EURJPY": 0.002,    # ~2 pips
    "GBPJPY": 0.003,    # ~3 pips
    "EURGBP": 0.00015,  # ~1.5 pips
}

# Correlation pairs (to avoid double exposure)
CORRELATED_PAIRS = [
    {"EURUSD=X", "GBPUSD=X"},  # Both have EUR/GBP exposure
    {"EURUSD=X", "EURGBP=X"},
    {"GBPUSD=X", "EURGBP=X"},
    {"USDJPY=X", "EURJPY=X"},  # JPY exposure
    {"USDJPY=X", "GBPJPY=X"},
    {"EURJPY=X", "GBPJPY=X"},
]

INTERVAL = "15m"
LOOKBACK = "14d"
MIN_ROWS = 220
CACHE_TTL_SECONDS = 300  # 5 minutes

# =========================
# DATA SHAPE HELPER (CRITICAL FIX)
# =========================
def ensure_series(data):
    """
    Robustly convert yfinance data to 1D Series.
    Fixes "Data must be 1-dimensional" errors.
    
    Args:
        data: pandas DataFrame or Series from yfinance
    
    Returns:
        1D pandas Series
    """
    if isinstance(data, pd.DataFrame):
        # Take first column if multi-column
        data = data.iloc[:, 0]
    return data.squeeze()


# =========================
# RETRY DECORATOR WITH EXPONENTIAL BACKOFF
# =========================
def retry_with_backoff(max_retries=3, backoff_factor=5):
    """Retry decorator with exponential backoff for rate limits"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "rate limit" in error_msg or "429" in error_msg or "too many requests" in error_msg:
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * backoff_factor
                            log.warning(f"⚠️ Rate limited, waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            log.error(f"❌ Rate limit exceeded after {max_retries} attempts")
                            raise
                    else:
                        # Non-rate-limit error, raise immediately
                        raise
            raise Exception(f"Failed after {max_retries} attempts")
        return wrapper
    return decorator

# =========================
# LOAD DYNAMIC CONFIG
# =========================
def load_config():
    config_path = Path("config.json")
    if not config_path.exists():
        log.warning("⚠️ config.json not found, using enhanced defaults")
        return _default_config()
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # VALIDATE CONFIG
    mode = config.get("mode", "aggressive")
    if mode not in ["aggressive", "conservative"]:
        log.error(f"❌ Invalid mode '{mode}', defaulting to aggressive")
        config["mode"] = "aggressive"
        mode = "aggressive"
    
    if mode not in config.get("settings", {}):
        log.error(f"❌ Settings missing for mode '{mode}'")
        raise ValueError(f"Config incomplete for mode: {mode}")
    
    # ✅ ENFORCE MINIMUM THRESHOLD
    if config["settings"]["aggressive"]["threshold"] < 45:
        log.warning(f"⚠️ Aggressive threshold too low, raising to 48")
        config["settings"]["aggressive"]["threshold"] = 48
    
    log.info(f"✅ Config loaded: mode={mode}, sentiment={config.get('use_sentiment', False)}")
    return config

def _default_config():
    """Enhanced default configuration"""
    return {
        "mode": "aggressive",
        "use_sentiment": False,
        "settings": {
            "aggressive": {
                "threshold": 48,
                "min_adx": 20,
                "rsi_oversold": 35,
                "rsi_overbought": 65,
                "min_volume_ratio": 1.1,
                "volume_penalty": 10,
                "min_risk_reward": 1.5,
                "atr_stop_multiplier": 2.5,
                "atr_target_multiplier": 5.0,
                "max_correlated_signals": 2,
                "max_signal_age_minutes": 60
            },
            "conservative": {
                "threshold": 58,
                "min_adx": 25,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "min_volume_ratio": 1.3,
                "volume_penalty": 15,
                "min_risk_reward": 2.0,
                "atr_stop_multiplier": 3.0,
                "atr_target_multiplier": 6.0,
                "max_correlated_signals": 1,
                "max_signal_age_minutes": 45
            }
        },
        "advanced": {
            "enable_session_filtering": True,
            "enable_correlation_filter": True,
            "cache_ttl_minutes": 5,
            "session_bonuses": {
                "ASIAN": {"JPY_pairs": 5, "AUD_NZD_pairs": 5, "other": 0},
                "EUROPEAN": {"EUR_GBP_pairs": 5, "EUR_GBP_crosses": 3, "other": 0},
                "OVERLAP": {"all_major_pairs": 3},
                "US": {"USD_majors": 5, "other": 0}
            },
            "validation": {
                "max_price_change_pct": 0.05,
                "max_signal_age_seconds": 300,
                "min_sl_pips": {"JPY_pairs": 15, "other": 10},
                "max_spread_ratio": 0.3
            }
        },
        "risk_management": {
            "max_daily_risk_pips": 200,
            "max_open_positions": 5,
            "max_correlated_exposure": 2
        }
    }

# =========================
# API KEY VALIDATION
# =========================
def validate_api_keys():
    """Validate API keys on startup"""
    required_keys = {
        'newsapi_key': os.environ.get('newsapi_key'),
        'MARKETAUX_API_KEY': os.environ.get('MARKETAUX_API_KEY')
    }
    
    missing_keys = [k for k, v in required_keys.items() if not v]
    
    if missing_keys:
        log.warning(f"⚠️ Missing API keys: {', '.join(missing_keys)}")
        log.warning("Sentiment analysis will be disabled")
        return False
    
    # Test NewsAPI
    try:
        r = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"country": "us", "pageSize": 1, "apiKey": required_keys['newsapi_key']},
            timeout=5
        )
        if r.status_code == 401:
            log.error("❌ Invalid NewsAPI key")
            return False
        elif r.status_code == 429:
            log.warning("⚠️ NewsAPI rate limit reached")
            return False
        log.info("✅ NewsAPI key validated")
    except Exception as e:
        log.warning(f"⚠️ NewsAPI validation failed: {e}")
        return False
    
    # Test Marketaux
    try:
        r = requests.get(
            "https://api.marketaux.com/v1/news/all",
            params={"api_token": required_keys['MARKETAUX_API_KEY'], "limit": 1},
            timeout=5
        )
        if r.status_code == 401:
            log.error("❌ Invalid Marketaux key")
            return False
        elif r.status_code == 429:
            log.warning("⚠️ Marketaux rate limit reached")
            return False
        log.info("✅ Marketaux key validated")
    except Exception as e:
        log.warning(f"⚠️ Marketaux validation failed: {e}")
        return False
    
    return True

CONFIG = load_config()
MODE = CONFIG["mode"]
USE_SENTIMENT = CONFIG.get("use_sentiment", False) and validate_api_keys()
SETTINGS = CONFIG["settings"][MODE]

# =========================
# ENHANCED SIGNAL VALIDATION (v2.0.3)
# =========================
def validate_signal_quality(signal: Dict, config: Dict) -> Tuple[bool, List[str]]:
    """
    Enhanced quality check before emitting signal.
    
    v2.0.3 improvements:
    - Configurable validation thresholds
    - Unrealistic price change detection
    - Stale signal filtering
    - Mode-specific validation
    
    Returns: (is_valid, list_of_warnings)
    """
    warnings = []
    validation_config = config.get("advanced", {}).get("validation", {})
    mode = config.get("mode", "aggressive")
    mode_settings = config["settings"][mode]
    
    # Check 1: Valid stop loss distance
    sl_distance = abs(signal['entry_price'] - signal['sl'])
    if sl_distance == 0:
        warnings.append("Zero stop loss distance")
        return False, warnings
    
    # Check 2: Spread vs Stop Loss ratio (capped)
    spread = signal['spread']
    effective_spread = min(spread, sl_distance * 0.25)
    spread_ratio = effective_spread / sl_distance if sl_distance > 0 else 1
    
    max_spread_ratio = validation_config.get("max_spread_ratio", 0.3)
    if spread_ratio > max_spread_ratio:
        warnings.append(f"High spread ratio: {spread_ratio:.1%} of SL (max: {max_spread_ratio:.1%})")
        return False, warnings
    
    # Check 3: Pair-specific ATR validation
    if "JPY" in signal['pair']:
        max_atr = 1.0
        min_atr = 0.001
    else:
        max_atr = 0.01
        min_atr = 0.00001
    
    if signal['atr'] <= min_atr or signal['atr'] > max_atr:
        warnings.append(f"Invalid ATR: {signal['atr']} (range: {min_atr}-{max_atr})")
        return False, warnings
    
    # Check 4: Minimum pip distance (configurable)
    min_sl_pips_config = validation_config.get("min_sl_pips", {})
    
    if "JPY" in signal['pair']:
        min_sl_pips = min_sl_pips_config.get("JPY_pairs", 15)
        pip_value = 0.01
    else:
        min_sl_pips = min_sl_pips_config.get("other", 10)
        pip_value = 0.0001
    
    sl_pips = sl_distance / pip_value
    if sl_pips < min_sl_pips:
        warnings.append(f"Stop too tight: {sl_pips:.1f} pips < {min_sl_pips}")
        return False, warnings
    
    # Check 5: Risk-reward sanity (mode-specific)
    min_rr = mode_settings.get("min_risk_reward", 1.5)
    if signal['risk_reward'] < min_rr:
        warnings.append(f"Poor R:R: {signal['risk_reward']:.2f} < {min_rr}")
        return False, warnings
    
    # Check 6: NEW - Unrealistic price change detection
    max_price_change = validation_config.get("max_price_change_pct", 0.05)
    price_change_pct = abs(signal['tp'] - signal['sl']) / signal['entry_price']
    
    if price_change_pct > max_price_change:
        warnings.append(f"Unrealistic TP/SL spread: {price_change_pct:.1%} > {max_price_change:.1%}")
        return False, warnings
    
    # Check 7: NEW - Stale signal detection
    max_age = validation_config.get("max_signal_age_seconds", 300)
    try:
        signal_time = datetime.fromisoformat(signal['timestamp'].replace('Z', '+00:00'))
        signal_age = (datetime.now(timezone.utc) - signal_time).total_seconds()
        
        if signal_age > max_age:
            warnings.append(f"Stale signal: {signal_age/60:.1f} minutes old (max: {max_age/60:.1f})")
            return False, warnings
    except Exception as e:
        warnings.append(f"Invalid timestamp: {e}")
        return False, warnings
    
    return True, warnings


# =========================
# BACKEND INTELLIGENCE FUNCTIONS
# =========================
def calculate_hold_time(risk_reward: float, atr: float) -> str:
    """
    Calculate suggested hold time based on R:R and volatility.
    Frontend will trust this instead of calculating.
    """
    if risk_reward > 2.5 or atr > 0.002:
        return "SWING"
    elif risk_reward > 1.8 or atr > 0.0015:
        return "INTRADAY"
    return "SHORT"


def calculate_eligible_modes(score: int, adx: float, volume_ratio: float, 
                            rsi: float, config: Dict) -> List[str]:
    """
    Determine which modes this signal qualifies for.
    Frontend will filter based on this instead of re-calculating.
    """
    modes = []
    
    # Conservative criteria
    conservative_settings = config["settings"]["conservative"]
    if (score >= conservative_settings["threshold"] and
        adx >= conservative_settings["min_adx"] and
        volume_ratio >= conservative_settings["min_volume_ratio"] and
        conservative_settings["rsi_oversold"] <= rsi <= conservative_settings["rsi_overbought"]):
        modes.append("conservative")
    
    # Aggressive criteria
    aggressive_settings = config["settings"]["aggressive"]
    if (score >= aggressive_settings["threshold"] and
        adx >= aggressive_settings["min_adx"] and
        volume_ratio >= aggressive_settings["min_volume_ratio"] and
        aggressive_settings["rsi_oversold"] <= rsi <= aggressive_settings["rsi_overbought"]):
        modes.append("aggressive")
    
    return modes


def calculate_signal_freshness(timestamp: datetime) -> dict:
    """
    Calculate signal freshness metadata.
    Frontend will trust this instead of calculating age.
    """
    age_minutes = (datetime.now(timezone.utc) - timestamp).total_seconds() / 60
    
    if age_minutes < 15:
        status = "FRESH"
    elif age_minutes < 30:
        status = "RECENT"
    elif age_minutes < 60:
        status = "AGING"
    else:
        status = "STALE"
    
    confidence_decay = max(0, 100 - (age_minutes * 2))
    
    return {
        "status": status,
        "age_minutes": round(age_minutes, 1),
        "confidence_decay": round(confidence_decay, 1)
    }


def calculate_market_volatility(signals: List[Dict]) -> str:
    """
    Calculate overall market volatility state.
    Frontend will display this instead of inferring.
    """
    if not signals:
        return "CALM"
    
    avg_atr = sum(s.get("atr", 0) for s in signals) / len(signals)
    avg_volume = sum(s.get("volume_ratio", 1) for s in signals) / len(signals)
    
    if avg_atr > 0.002 or avg_volume > 1.5:
        return "HIGH"
    elif avg_atr > 0.0015 or avg_volume > 1.2:
        return "NORMAL"
    return "CALM"


def calculate_market_sentiment(signals: List[Dict]) -> str:
    """
    Calculate overall market sentiment bias.
    Frontend will display this instead of inferring.
    """
    if not signals:
        return "MIXED"
    
    bullish = sum(1 for s in signals if s.get("direction") == "BUY")
    bearish = sum(1 for s in signals if s.get("direction") == "SELL")
    
    if bullish > bearish * 1.5:
        return "BULLISH"
    elif bearish > bullish * 1.5:
        return "BEARISH"
    return "MIXED"


# =========================
# MARKET SESSION DETECTION
# =========================
def get_market_session() -> str:
    """Return active forex session"""
    hour = datetime.now(timezone.utc).hour
    
    if 0 <= hour < 8:
        return "ASIAN"
    elif 8 <= hour < 13:
        return "EUROPEAN"
    elif 13 <= hour < 16:
        return "OVERLAP"
    elif 16 <= hour < 21:
        return "US"
    else:
        return "LATE_US"


def calculate_dynamic_session_bonus(pair: str, session: str, config: Dict) -> int:
    """
    Calculate session bonus based on config and pair characteristics.
    
    Replaces hardcoded get_session_bonus() with configurable version.
    """
    if not config.get("advanced", {}).get("enable_session_filtering", True):
        return 0
    
    session_config = config.get("advanced", {}).get("session_bonuses", {})
    
    if session not in session_config:
        return 0
    
    bonuses = session_config[session]
    
    # Determine pair category
    if session == "ASIAN":
        if "JPY" in pair:
            return bonuses.get("JPY_pairs", 0)
        elif any(curr in pair for curr in ["AUD", "NZD"]):
            return bonuses.get("AUD_NZD_pairs", 0)
        return bonuses.get("other", 0)
    
    elif session in ["EUROPEAN", "OVERLAP"]:
        if any(curr in pair for curr in ["EUR", "GBP"]) and pair not in ["EURUSD", "GBPUSD"]:
            return bonuses.get("EUR_GBP_crosses", 0)
        elif any(curr in pair for curr in ["EUR", "GBP"]):
            return bonuses.get("EUR_GBP_pairs", 0)
        elif session == "OVERLAP":
            return bonuses.get("all_major_pairs", 0)
        return bonuses.get("other", 0)
    
    elif session == "US":
        if "USD" in pair and pair in ["EURUSD", "GBPUSD", "USDCAD"]:
            return bonuses.get("USD_majors", 0)
        return bonuses.get("other", 0)
    
    return 0


# =========================
# SENTIMENT MODEL HEALTH PERSISTENCE
# =========================
class ModelHealthTracker:
    """Track failed models and implement cooldown periods"""
    
    def __init__(self, cache_file: str = "signal_state/model_health.json"):
        self.cache_file = Path(cache_file)
        self.cooldown_hours = 1
        self.failed_models = {}
        self._load()
    
    def _load(self):
        """Load failed model cache"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.failed_models = {
                        int(k): datetime.fromisoformat(v) 
                        for k, v in data.items()
                    }
                log.info(f"📋 Loaded model health cache: {len(self.failed_models)} failed models")
            except Exception as e:
                log.warning(f"⚠️ Could not load model health cache: {e}")
                self.failed_models = {}
    
    def _save(self):
        """Save failed model cache"""
        self.cache_file.parent.mkdir(exist_ok=True)
        try:
            data = {str(k): v.isoformat() for k, v in self.failed_models.items()}
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log.warning(f"⚠️ Could not save model health cache: {e}")
    
    def mark_failed(self, model_idx: int):
        """Mark a model as failed"""
        self.failed_models[model_idx] = datetime.now(timezone.utc)
        self._save()
    
    def is_available(self, model_idx: int) -> bool:
        """Check if model is available (not in cooldown)"""
        if model_idx not in self.failed_models:
            return True
        
        retry_time = self._get_retry_time(model_idx)
        if datetime.now(timezone.utc) >= retry_time:
            del self.failed_models[model_idx]
            self._save()
            return True
        
        return False
    
    def _get_retry_time(self, model_idx: int) -> datetime:
        """Get the time when model can be retried"""
        failed_time = self.failed_models.get(model_idx)
        if failed_time:
            return failed_time + timedelta(hours=self.cooldown_hours)
        return datetime.now(timezone.utc)


# =========================
# IMPROVED SENTIMENT ANALYSIS WITH THREAD SAFETY
# =========================
class SentimentAnalyzer:
    """
    Multi-model sentiment analyzer with lazy verification and thread safety.
    """
    
    def __init__(self, hf_api_key: str = None):
        self.hf_api_key = hf_api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "TradeBeacon/1.0"})
        if hf_api_key:
            self.session.headers.update({"Authorization": f"Bearer {hf_api_key}"})
        
        self.models = [
            {
                "name": "FinBERT-tone",
                "url": "https://api-inference.huggingface.co/models/yiyanghkust/finbert-tone",
                "label_map": {"positive": "positive", "negative": "negative", "neutral": "neutral"}
            },
            {
                "name": "DistilRoBERTa-financial",
                "url": "https://api-inference.huggingface.co/models/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
                "label_map": {"positive": "positive", "negative": "negative", "neutral": "neutral"}
            },
            {
                "name": "Twitter-RoBERTa",
                "url": "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest",
                "label_map": {"label_2": "positive", "label_0": "negative", "label_1": "neutral"}
            },
            {
                "name": "FinBERT-ESG",
                "url": "https://api-inference.huggingface.co/models/yiyanghkust/finbert-esg",
                "label_map": {"positive": "positive", "negative": "negative", "neutral": "neutral"}
            }
        ]
        
        self.current_model_idx = 0
        self._model_lock = threading.Lock()
        self.health_tracker = ModelHealthTracker()
        
        available_count = sum(1 for i in range(len(self.models)) if self.health_tracker.is_available(i))
        log.info(f"🤖 Sentiment analyzer initialized with {len(self.models)} models ({available_count} available)")
    
    def _get_next_available_model(self) -> Optional[int]:
        """Get next available model index (thread-safe)"""
        with self._model_lock:
            attempts = 0
            while attempts < len(self.models):
                if self.health_tracker.is_available(self.current_model_idx):
                    idx = self.current_model_idx
                    self.current_model_idx = (self.current_model_idx + 1) % len(self.models)
                    return idx
                
                self.current_model_idx = (self.current_model_idx + 1) % len(self.models)
                attempts += 1
            
            return None
    
    def _normalize_label(self, raw_label: str, model_idx: int) -> str:
        """Normalize model-specific labels to positive/negative/neutral"""
        label_map = self.models[model_idx].get("label_map", {})
        normalized = label_map.get(raw_label.lower(), "neutral")
        return normalized
    
    def analyze(self, text: str, max_retries: int = 2) -> Dict:
        """Analyze sentiment with automatic model fallback (thread-safe)."""
        if not text or len(text.strip()) < 10:
            return {"label": "neutral", "score": 0.0}
        
        for attempt in range(len(self.models)):
            model_idx = self._get_next_available_model()
            
            if model_idx is None:
                log.warning("⚠️ No available sentiment models")
                return {"label": "neutral", "score": 0.0}
            
            model = self.models[model_idx]
            
            try:
                result = self._call_model(model, model_idx, text, max_retries)
                if result:
                    return result
                else:
                    log.warning(f"⚠️ {model['name']} failed, trying next model...")
                    self.health_tracker.mark_failed(model_idx)
                    
            except Exception as e:
                log.warning(f"⚠️ {model['name']} error: {e}")
                self.health_tracker.mark_failed(model_idx)
        
        log.warning("⚠️ All sentiment models failed, defaulting to neutral")
        return {"label": "neutral", "score": 0.0}
    
    def _call_model(self, model: Dict, model_idx: int, text: str, max_retries: int) -> Optional[Dict]:
        """Call a specific HuggingFace model"""
        for retry in range(max_retries):
            try:
                response = self.session.post(
                    model["url"],
                    json={"inputs": text[:512]},
                    timeout=15
                )
                
                if response.status_code == 503:
                    if retry < max_retries - 1:
                        wait_time = 10 * (retry + 1)
                        log.info(f"⏳ {model['name']} loading, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return None
                
                if response.status_code == 410:
                    log.warning(f"⚠️ {model['name']} is deprecated (410 Gone)")
                    return None
                
                if response.status_code == 429:
                    if retry < max_retries - 1:
                        log.warning(f"⚠️ {model['name']} rate limited, waiting...")
                        time.sleep(5 * (retry + 1))
                        continue
                    else:
                        return None
                
                response.raise_for_status()
                result = response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        top = max(result[0], key=lambda x: x.get('score', 0))
                    else:
                        top = max(result, key=lambda x: x.get('score', 0))
                    
                    raw_label = top.get('label', 'neutral')
                    normalized_label = self._normalize_label(raw_label, model_idx)
                    score = round(top.get('score', 0.0), 3)
                    
                    if retry > 0:
                        log.info(f"✅ {model['name']} succeeded on retry {retry + 1}")
                    
                    return {"label": normalized_label, "score": score}
                
                return {"label": "neutral", "score": 0.0}
                
            except requests.exceptions.RequestException as e:
                if retry < max_retries - 1:
                    time.sleep(2)
                else:
                    return None
            except Exception as e:
                log.debug(f"Unexpected error: {e}")
                return None
        
        return None


class NewsAggregator:
    """Fetch forex news using NewsAPI and Marketaux"""
    
    def __init__(self):
        self.newsapi_key = os.environ.get('newsapi_key')
        if not self.newsapi_key:
            key_file = Path("newsapi_key.txt")
            if key_file.exists():
                self.newsapi_key = key_file.read_text().strip()
        
        self.marketaux_key = os.environ.get('MARKETAUX_API_KEY')
        if not self.marketaux_key:
            key_file = Path("marketaux_key.txt")
            if key_file.exists():
                self.marketaux_key = key_file.read_text().strip()
        
        self.newsapi_calls = 0
        self.marketaux_calls = 0
    
    def get_news(self, pairs: List[str]) -> List[Dict]:
        """Fetch recent forex news from both NewsAPI and Marketaux."""
        all_articles = []
        
        currencies = set()
        for pair in pairs:
            clean = pair.replace("=X", "")
            if len(clean) >= 6:
                currencies.add(clean[:3])
                currencies.add(clean[3:6])
            else:
                currencies.add(clean[:3])
        
        log.info(f"🔍 Searching news for currencies: {', '.join(sorted(currencies))}")
        
        if self.newsapi_key:
            newsapi_articles = self._fetch_newsapi(currencies, limit=5)
            all_articles.extend(newsapi_articles)
            self.newsapi_calls += 1
        else:
            log.warning("⚠️ No NewsAPI key - skipping NewsAPI")
        
        if self.marketaux_key:
            marketaux_articles = self._fetch_marketaux(currencies, limit=5)
            all_articles.extend(marketaux_articles)
            self.marketaux_calls += 1
        else:
            log.warning("⚠️ No Marketaux key - skipping Marketaux")
        
        if not all_articles:
            log.warning("⚠️ No news sources available - sentiment disabled")
        else:
            log.info(f"📰 Fetched {len(all_articles)} articles total")
        
        return all_articles
    
    def _fetch_newsapi(self, currencies: set, limit: int = 5) -> List[Dict]:
        """Fetch news from NewsAPI"""
        try:
            query = " OR ".join(sorted(currencies))
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": f"forex ({query})",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": limit,
                "apiKey": self.newsapi_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get("articles", [])[:limit]:
                articles.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "published": article.get("publishedAt", ""),
                    "source_api": "NewsAPI"
                })
            
            log.info(f"✅ NewsAPI: {len(articles)} articles")
            return articles
            
        except Exception as e:
            log.error(f"❌ NewsAPI fetch failed: {e}")
            return []
    
    def _fetch_marketaux(self, currencies: set, limit: int = 5) -> List[Dict]:
        """Fetch news from Marketaux API"""
        try:
            symbols = ",".join(sorted(list(currencies)[:5]))
            
            url = "https://api.marketaux.com/v1/news/all"
            params = {
                "symbols": symbols,
                "filter_entities": "true",
                "language": "en",
                "limit": limit,
                "api_token": self.marketaux_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get("data", [])[:limit]:
                entities = article.get("entities", [])
                sentiment_score = None
                if entities and len(entities) > 0:
                    sentiment_score = entities[0].get("sentiment_score")
                
                articles.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "source": article.get("source", "Marketaux"),
                    "published": article.get("published_at", ""),
                    "source_api": "Marketaux",
                    "marketaux_sentiment": sentiment_score
                })
            
            log.info(f"✅ Marketaux: {len(articles)} articles")
            return articles
            
        except Exception as e:
            log.error(f"❌ Marketaux fetch failed: {e}")
            return []


def filter_articles_for_pair(pair: str, all_articles: List[Dict]) -> List[Dict]:
    """Filter articles relevant to a specific currency pair."""
    clean_pair = pair.replace("=X", "")
    currencies = [clean_pair[:3], clean_pair[3:6]] if len(clean_pair) >= 6 else [clean_pair[:3]]
    
    relevant_articles = []
    
    for article in all_articles:
        text = f"{article.get('title', '')} {article.get('description', '')}".upper()
        
        if any(curr in text for curr in currencies):
            relevant_articles.append(article)
    
    return relevant_articles


def analyze_sentiment_from_articles(pair: str, articles: List[Dict], 
                                    analyzer: SentimentAnalyzer) -> Dict:
    """Analyze sentiment from pre-fetched articles."""
    
    if not articles:
        return {
            "adjustment": 0, 
            "sentiment": "neutral", 
            "news_count": 0, 
            "sources": {"newsapi": 0, "marketaux": 0}
        }
    
    sentiments = []
    newsapi_count = 0
    marketaux_count = 0
    
    for article in articles:
        text = f"{article['title']} {article['description']}"
        sentiment = analyzer.analyze(text)
        sentiments.append(sentiment)
        
        if article.get('source_api') == 'NewsAPI':
            newsapi_count += 1
        elif article.get('source_api') == 'Marketaux':
            marketaux_count += 1
    
    positive = sum(1 for s in sentiments if s['label'] == 'positive')
    negative = sum(1 for s in sentiments if s['label'] == 'negative')
    
    if positive > negative:
        adjustment = min(20, positive * 10)
        overall = "bullish"
    elif negative > positive:
        adjustment = max(-20, -negative * 10)
        overall = "bearish"
    else:
        adjustment = 0
        overall = "neutral"
    
    return {
        "adjustment": adjustment,
        "sentiment": overall,
        "news_count": len(articles),
        "positive": positive,
        "negative": negative,
        "sources": {
            "newsapi": newsapi_count,
            "marketaux": marketaux_count
        }
    }


# =========================
# TTL-BASED CACHE FOR MARKET DATA
# =========================
class MarketDataCache:
    """Time-based cache with automatic expiration"""
    
    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS):
        self.ttl = ttl_seconds
        self._cache = {}
        self._timestamps = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached data if still valid"""
        with self._lock:
            if key not in self._cache:
                return None
            
            age = time.time() - self._timestamps.get(key, 0)
            if age > self.ttl:
                del self._cache[key]
                del self._timestamps[key]
                return None
            
            return self._cache[key]
    
    def set(self, key: str, value: pd.DataFrame):
        """Cache data with timestamp"""
        with self._lock:
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def clear(self):
        """Clear all cached data"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

# Global cache instance
_market_cache = MarketDataCache()

# =========================
# TECHNICAL ANALYSIS UTILS WITH SAFE CACHING
# =========================
def last(series: pd.Series):
    return None if series is None or series.empty else float(series.iloc[-1])

@retry_with_backoff(max_retries=3, backoff_factor=5)
def download(pair: str) -> Tuple[pd.DataFrame, bool]:
    """
    Download data with TTL-based caching
    
    Returns:
        Tuple of (DataFrame, bool) where bool indicates successful download
    """
    cached = _market_cache.get(pair)
    if cached is not None:
        log.debug(f"📦 Using cached data for {pair}")
        return cached, True
    
    log.debug(f"📥 Downloading fresh data for {pair}")
    
    try:
        df = yf.download(
            pair,
            interval=INTERVAL,
            period=LOOKBACK,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        
        if df is None or df.empty:
            log.warning(f"⚠️ {pair} download returned empty data")
            return pd.DataFrame(), False
        
        df = df.dropna()
        
        if len(df) < MIN_ROWS:
            log.warning(f"⚠️ {pair} insufficient data: {len(df)} rows")
            return df, False
        
        _market_cache.set(pair, df)
        return df, True
        
    except Exception as e:
        log.error(f"❌ {pair} download failed: {e}")
        return pd.DataFrame(), False

def ema(series, period):
    return EMAIndicator(series, window=period).ema_indicator()

def rsi(series, period=14):
    return RSIIndicator(series, window=period).rsi()

def adx_calc(high, low, close):
    return ADXIndicator(high, low, close, window=14).adx()

def atr_calc(high, low, close):
    return AverageTrueRange(high, low, close, window=14).average_true_range()

def get_spread(pair: str) -> float:
    """Get typical spread for a pair"""
    clean_pair = pair.replace("=X", "")
    return SPREADS.get(clean_pair, 0.0002)

def classify_market_state(adx: float, atr: float, volume_ratio: float) -> str:
    """Classify current market conditions for signal context."""
    if adx < 15:
        return "CHOPPY"
    elif adx > 25 and volume_ratio > 1.2:
        return "TRENDING_STRONG"
    elif adx > 20:
        return "TRENDING_MODERATE"
    else:
        return "CONSOLIDATING"

def get_signal_type(e12: float, e26: float, e200: float, rsi: float) -> str:
    """Classify the type of trading opportunity."""
    if e12 > e26 > e200:
        if rsi > 60:
            return "momentum"
        else:
            return "trend-continuation"
    elif e12 < e26 < e200:
        if rsi < 40:
            return "momentum"
        else:
            return "trend-continuation"
    elif (e12 > e26 and rsi < 40) or (e12 < e26 and rsi > 60):
        return "reversal"
    else:
        return "breakout"

# =========================
# ENHANCED SIGNAL ENGINE WITH CONFIGURABLE PARAMETERS
# =========================
def generate_signal(pair: str) -> Tuple[Optional[dict], bool]:
    """
    Generate trading signal with configurable parameters from config.
    
    v2.0.3 enhancements:
    - ATR multipliers from config
    - Mode-specific minimum R:R
    - Dynamic session bonuses
    - Enhanced validation
    
    Returns:
        Tuple of (signal_dict or None, bool) where bool indicates download success
    """
    df, download_success = download(pair)
    
    if not download_success or len(df) < MIN_ROWS:
        log.warning(f"⚠️ {pair} not enough candles ({len(df)}), skipping")
        return None, download_success
    
    try:
        # ✅ CRITICAL FIX: Use ensure_series for robust data handling
        close = ensure_series(df["Close"])
        high = ensure_series(df["High"])
        low = ensure_series(df["Low"])
        volume = ensure_series(df["Volume"])

        e12 = last(ema(close, 12))
        e26 = last(ema(close, 26))
        e200 = last(ema(close, 200))
        r = last(rsi(close))
        a = last(adx_calc(high, low, close))
        atr = last(atr_calc(high, low, close))
        current_price = last(close)
        
        avg_volume = volume.rolling(window=20).mean()
        current_volume = last(volume)
        avg_vol = last(avg_volume)
        volume_ratio = current_volume / avg_vol if avg_vol and avg_vol > 0 else 1.0

    except Exception as e:
        log.warning(f"⚠️ {pair} indicator calc failed: {e}")
        return None, download_success

    if None in (e12, e26, e200, r, a, current_price, atr):
        log.warning(f"⚠️ {pair} indicators incomplete, skipping")
        return None, download_success

    # ADX filter (from config)
    min_adx = SETTINGS.get("min_adx", 20)
    if a < min_adx:
        log.info(f"❌ {pair} | ADX too low ({a:.1f} < {min_adx})")
        return None, download_success

    bull = bear = 0

    # EMA Trend Structure (40 points)
    if e12 > e26 > e200:
        bull += 40
    elif e12 < e26 < e200:
        bear += 40

    # RSI Context (20-30 points)
    rsi_oversold = SETTINGS.get("rsi_oversold", 35)
    rsi_overbought = SETTINGS.get("rsi_overbought", 65)
    
    if MODE == "conservative":
        if r < rsi_oversold:
            bull += 30
        elif r > rsi_overbought:
            bear += 30
    else:
        if r < rsi_oversold:
            bull += 20
        elif r > rsi_overbought:
            bear += 20

    # ADX Trend Strength (10-20 points)
    if a > 25:  # Strong trend
        if e12 > e26:
            bull += 20
        elif e12 < e26:
            bear += 20
    elif a > min_adx:
        if e12 > e26:
            bull += 10
        elif e12 < e26:
            bear += 10
    
    # ✅ Volume handling
    min_volume_ratio = SETTINGS.get("min_volume_ratio", 1.0)
    volume_penalty = SETTINGS.get("volume_penalty", 10)
    
    if volume_ratio >= min_volume_ratio:
        if volume_ratio > 1.5:
            bonus = 10
        elif volume_ratio > 1.2:
            bonus = 5
        else:
            bonus = 3
        
        if e12 > e26:
            bull += bonus
        else:
            bear += bonus
    else:
        # Low volume reduces confidence in current direction
        if e12 > e26:
            bull -= volume_penalty
        else:
            bear -= volume_penalty
    
    # ✅ Dynamic session bonus (from config)
    session = get_market_session()
    clean_pair = pair.replace("=X", "")
    session_bonus = calculate_dynamic_session_bonus(clean_pair, session, CONFIG)
    
    if e12 > e26:
        bull += session_bonus
    else:
        bear += session_bonus

    diff = abs(bull - bear)

    # Threshold check (from config)
    threshold = SETTINGS.get("threshold", 48)
    if diff < threshold:
        return None, download_success

    direction = "BUY" if bull > bear else "SELL"

    if diff >= 70:
        confidence = "EXCELLENT"
    elif diff >= 60:
        confidence = "STRONG"
    elif diff >= 50:
        confidence = "GOOD"
    else:
        confidence = "MODERATE"

    # Get spread for this pair
    spread = get_spread(pair)
    
    # ✅ Entry price uses mid-price (spread is informational only)
    entry_price = current_price
    
    # ✅ ATR-based dynamic stops (NOW CONFIGURABLE)
    atr_stop_mult = SETTINGS.get("atr_stop_multiplier", 2.5)
    atr_target_mult = SETTINGS.get("atr_target_multiplier", 5.0)
    
    if direction == "BUY":
        sl = entry_price - (atr_stop_mult * atr)
        tp = entry_price + (atr_target_mult * atr)
    else:
        sl = entry_price + (atr_stop_mult * atr)
        tp = entry_price - (atr_target_mult * atr)
    
    # Calculate actual risk-reward ratio
    risk = abs(entry_price - sl)
    reward = abs(tp - entry_price)
    risk_reward = reward / risk if risk > 0 else 0
    
    # ✅ Mode-specific minimum RR filter (from config)
    min_rr = SETTINGS.get("min_risk_reward", 1.5)
    if risk_reward < min_rr:
        log.info(f"❌ {pair} | Poor risk-reward ({risk_reward:.2f} < {min_rr})")
        return None, download_success
    
    # Calculate signal metadata
    signal_type = get_signal_type(e12, e26, e200, r)
    market_state = classify_market_state(a, atr, volume_ratio)
    
    # Generate timestamp and expiration
    now = datetime.now(timezone.utc)
    valid_for_minutes = SETTINGS.get("max_signal_age_minutes", 60)
    expires_at = now + timedelta(minutes=valid_for_minutes)
    
    # Calculate backend intelligence
    hold_time = calculate_hold_time(risk_reward, atr)
    eligible_modes = calculate_eligible_modes(diff, a, volume_ratio, r, CONFIG)
    freshness = calculate_signal_freshness(now)
    
    # Generate unique signal ID
    signal_id = f"{clean_pair}_{now.strftime('%Y%m%d_%H%M%S')}"

    signal = {
        # Unique identifier
        "signal_id": signal_id,
        
        "pair": clean_pair,
        "direction": direction,
        "score": diff,
        "technical_score": diff,
        "sentiment_score": 0,
        "confidence": confidence,
        "rsi": round(r, 1),
        "adx": round(a, 1),
        "atr": round(atr, 5),
        "volume_ratio": round(volume_ratio, 2),
        "session": session,
        "entry_price": round(entry_price, 5),
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "risk_reward": round(risk_reward, 2),
        "spread": round(spread, 5),
        "timestamp": now.isoformat(),
        
        # Backend intelligence
        "hold_time": hold_time,
        "eligible_modes": eligible_modes,
        "freshness": freshness,
        
        # Signal metadata
        "metadata": {
            "signal_type": signal_type,
            "market_state": market_state,
            "timeframe": INTERVAL,
            "valid_for_minutes": valid_for_minutes,
            "generated_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "session_active": session,
            "signal_generator_version": "2.0.3",
            "atr_stop_multiplier": atr_stop_mult,
            "atr_target_multiplier": atr_target_mult
        }
    }
    
    # ✅ v2.0.3: Enhanced validation with config
    is_valid, warnings = validate_signal_quality(signal, CONFIG)
    
    if not is_valid:
        log.info(f"❌ {pair} | Signal rejected: {', '.join(warnings)}")
        return None, download_success
    
    if warnings:
        log.debug(f"⚠️ {pair} | Signal warnings: {', '.join(warnings)}")
    
    return signal, download_success

# =========================
# ENHANCED CORRELATION FILTER
# =========================
def filter_correlated_signals_enhanced(signals: List[Dict], max_correlated: int = 2) -> List[Dict]:
    """
    Enhanced correlation filter with configurable limits.
    """
    if len(signals) <= 1:
        return signals
    
    filtered = []
    correlation_groups = {}
    
    sorted_signals = sorted(signals, key=lambda x: x['score'], reverse=True)
    
    for signal in sorted_signals:
        pair = f"{signal['pair']}=X"
        
        # Find which correlation group this pair belongs to
        assigned_group = None
        for corr_group in CORRELATED_PAIRS:
            if pair in corr_group:
                group_key = frozenset(corr_group)
                assigned_group = group_key
                break
        
        if assigned_group:
            # Check if we've hit the limit for this group
            count = correlation_groups.get(assigned_group, 0)
            if count < max_correlated:
                filtered.append(signal)
                correlation_groups[assigned_group] = count + 1
            else:
                log.info(f"⚠️ Skipping {signal['pair']} (correlation group limit: {max_correlated})")
        else:
            # Not in any correlation group, always include
            filtered.append(signal)
    
    if len(filtered) < len(signals):
        log.info(f"🔗 Enhanced correlation filter: {len(signals)} → {len(filtered)} signals")
    
    return filtered


def check_risk_limits(signals: List[Dict], config: Dict) -> Tuple[List[Dict], List[str]]:
    """
    Filter signals based on risk management rules.
    
    Returns: (filtered_signals, warnings)
    """
    risk_config = config.get("risk_management", {})
    warnings = []
    
    # Check 1: Max open positions
    max_positions = risk_config.get("max_open_positions", 5)
    if len(signals) > max_positions:
        warnings.append(f"Limiting to {max_positions} positions (had {len(signals)})")
        signals = sorted(signals, key=lambda x: x['score'], reverse=True)[:max_positions]
    
    # Check 2: Max daily risk
    max_daily_risk = risk_config.get("max_daily_risk_pips", 200)
    total_risk_pips = 0
    
    filtered = []
    for signal in signals:
        entry = signal.get('entry_price', 0)
        sl = signal.get('sl', 0)
        pair = signal.get('pair', '')
        
        if entry > 0 and sl > 0:
            pip_value = 0.01 if "JPY" in pair else 0.0001
            risk_pips = abs(entry - sl) / pip_value
            
            if total_risk_pips + risk_pips <= max_daily_risk:
                filtered.append(signal)
                total_risk_pips += risk_pips
            else:
                warnings.append(f"Skipped {pair} - would exceed daily risk limit ({total_risk_pips:.1f}/{max_daily_risk} pips)")
        else:
            filtered.append(signal)
    
    # Check 3: Max correlated exposure
    mode = config.get("mode", "aggressive")
    max_correlated = config["settings"][mode].get("max_correlated_signals", 2)
    
    if config.get("advanced", {}).get("enable_correlation_filter", True):
        filtered = filter_correlated_signals_enhanced(filtered, max_correlated)
    
    return filtered, warnings


# =========================
# SENTIMENT ENHANCEMENT
# =========================
def enhance_with_sentiment(signals: List[Dict], news_agg: NewsAggregator) -> List[Dict]:
    """Add sentiment analysis to signals with proper direction logic."""
    
    if not USE_SENTIMENT or not signals:
        return signals
    
    log.info("\n" + "="*70)
    log.info("📰 Analyzing news sentiment from NewsAPI + Marketaux...")
    log.info("="*70)
    
    hf_key = os.environ.get('HF_API_KEY') or os.environ.get('HUGGINGFACE_API_KEY')
    analyzer = SentimentAnalyzer(hf_api_key=hf_key)
    
    all_pairs = [f"{sig['pair']}=X" for sig in signals]
    log.info(f"🔍 Fetching news for {len(all_pairs)} pairs: {', '.join(all_pairs)}")
    
    all_articles = news_agg.get_news(all_pairs)
    
    enhanced = []
    
    for signal in signals:
        pair = signal['pair']
        pair_ticker = f"{pair}=X"
        
        pair_articles = filter_articles_for_pair(pair_ticker, all_articles)
        
        sentiment_data = analyze_sentiment_from_articles(
            pair_ticker, 
            pair_articles, 
            analyzer
        )
        
        original_score = signal['technical_score']
        adjustment = sentiment_data['adjustment']
        
        direction_multiplier = 1 if signal['direction'] == 'BUY' else -1
        final_adjustment = adjustment * direction_multiplier
        
        signal['score'] = original_score + final_adjustment
        signal['score'] = max(0, min(100, signal['score']))
        signal['sentiment_score'] = adjustment
        
        # Re-validate against threshold after sentiment
        threshold = SETTINGS.get("threshold", 48)
        if signal['score'] < threshold:
            log.info(f"❌ {pair} | Signal too weak after sentiment ({signal['score']} < {threshold})")
            continue
        
        # Recalculate eligible modes after sentiment adjustment
        signal['eligible_modes'] = calculate_eligible_modes(
            signal['score'],
            signal['adx'],
            signal['volume_ratio'],
            signal['rsi'],
            CONFIG
        )
        
        # Update confidence based on combined score
        if signal['score'] >= 70:
            signal['confidence'] = "EXCELLENT"
        elif signal['score'] >= 60:
            signal['confidence'] = "STRONG"
        elif signal['score'] >= 50:
            signal['confidence'] = "GOOD"
        else:
            signal['confidence'] = "MODERATE"
        
        signal['sentiment'] = {
            "overall": sentiment_data['sentiment'],
            "adjustment": adjustment,
            "original_score": original_score,
            "news_count": sentiment_data['news_count'],
            "sources": sentiment_data.get('sources', {})
        }
        
        log.info(f"💡 {pair} | Direction: {signal['direction']} | "
                f"Sentiment: {sentiment_data['sentiment']} ({adjustment:+d}) | "
                f"Score: {original_score} → {signal['score']} ({final_adjustment:+d}) | "
                f"Articles: {sentiment_data['news_count']}")
        
        enhanced.append(signal)
    
    log.info(f"📊 API Usage: NewsAPI calls={news_agg.newsapi_calls}, "
             f"Marketaux calls={news_agg.marketaux_calls}")
    
    return enhanced

# =========================
# ENHANCED DASHBOARD WRITER
# =========================
def calculate_daily_pips(signals: List[Dict]) -> float:
    """Calculate total pips from today's signals"""
    today = datetime.now(timezone.utc).date()
    daily_pips = 0
    
    for signal in signals:
        try:
            signal_time = datetime.fromisoformat(signal.get('timestamp', ''))
            if signal_time.date() == today:
                entry = signal.get('entry_price', 0)
                tp = signal.get('tp', 0)
                if entry and tp:
                    pips = abs(tp - entry) * 10000
                    daily_pips += pips
        except Exception:
            continue
    
    return round(daily_pips, 1)


def write_dashboard_state(signals: list, successful_downloads: int, newsapi_calls: int = 0, marketaux_calls: int = 0):
    """Write comprehensive dashboard state with backend intelligence"""
    
    session = get_market_session()
    performance = track_performance(signals)
    daily_pips = calculate_daily_pips(signals)
    
    # Calculate market state (frontend will trust this)
    market_volatility = calculate_market_volatility(signals)
    market_sentiment = calculate_market_sentiment(signals)

    dashboard_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_signals": len(signals),
        "session": session,
        "mode": MODE,
        "sentiment_enabled": USE_SENTIMENT,
        
        # Market state for frontend
        "market_state": {
            "volatility": market_volatility,
            "sentiment_bias": market_sentiment,
            "session": session,
            "trending_pairs": [s['pair'] for s in signals if s.get('metadata', {}).get('market_state') == 'TRENDING_STRONG']
        },
        
        "signals": signals,
        
        "api_usage": {
            "yfinance": {"successful_downloads": successful_downloads},
            "sentiment": {
                "enabled": USE_SENTIMENT,
                "newsapi": newsapi_calls,
                "marketaux": marketaux_calls
            }
        },
        
        "stats": {
            "total_trades": performance["stats"].get("total_trades", 0),
            "win_rate": performance["stats"].get("win_rate", 0),
            "total_pips": performance["stats"].get("total_pips", 0),
            "wins": performance["stats"].get("wins", 0),
            "losses": performance["stats"].get("losses", 0)
        },
        
        "risk_management": {
            "daily_pips": daily_pips,
            "total_risk_pips": performance["risk_management"].get("total_risk_pips", 0),
            "max_drawdown": performance["risk_management"].get("max_drawdown", 0),
            "average_risk_reward": performance["risk_management"].get("average_risk_reward", 0)
        },
        
        "system": {
            "last_update": datetime.now(timezone.utc).isoformat(),
            "data_sources_available": successful_downloads > 0,
            "sentiment_available": newsapi_calls > 0 or marketaux_calls > 0,
            "version": "2.0.3"
        }
    }

    output_dir = Path("signal_state")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "dashboard_state.json"
    
    with open(output_file, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    log.info(f"📊 Dashboard written to {output_file}")
    
    stats = performance["stats"]
    if stats["total_trades"] > 0:
        log.info(f"📈 Performance: {stats['total_trades']} trades | "
                f"Win Rate: {stats['win_rate']}% | "
                f"Total Pips: {stats['total_pips']} | "
                f"Daily Pips: {daily_pips}")
    
    write_health_check(signals, successful_downloads, newsapi_calls, marketaux_calls)


def write_health_check(signals: list, successful_downloads: int, newsapi_calls: int, marketaux_calls: int):
    """Write health check file for monitoring"""
    
    status = "ok"
    issues = []
    
    if successful_downloads == 0:
        status = "error"
        issues.append("No market data available")
    
    if len(signals) == 0 and successful_downloads > 0:
        status = "warning"
        issues.append("No signals generated")
    
    health = {
        "status": status,
        "last_run": datetime.now(timezone.utc).isoformat(),
        "signal_count": len(signals),
        "issues": issues,
        "api_status": {
            "yfinance": "ok" if successful_downloads > 0 else "error",
            "newsapi": "ok" if newsapi_calls > 0 else ("disabled" if not USE_SENTIMENT else "error"),
            "marketaux": "ok" if marketaux_calls > 0 else ("disabled" if not USE_SENTIMENT else "error")
        },
        "system_info": {
            "mode": MODE,
            "pairs_monitored": len(PAIRS),
            "last_success": datetime.now(timezone.utc).isoformat() if status == "ok" else None,
            "version": "2.0.3"
        }
    }
    
    output_dir = Path("signal_state")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "health.json", "w") as f:
        json.dump(health, f, indent=2)
    
    if status == "ok":
        log.info("✅ System health: OK")
    elif status == "warning":
        log.warning(f"⚠️ System health: WARNING - {', '.join(issues)}")
    else:
        log.error(f"❌ System health: ERROR - {', '.join(issues)}")


# =========================
# TIME-WINDOW GUARD
# =========================
def in_execution_window():
    last_run_file = Path("signal_state/last_run.txt")
    success_file = Path("signal_state/last_success.txt")
    now = datetime.now(timezone.utc)

    if last_run_file.exists():
        with open(last_run_file, 'r') as f:
            last_run_str = f.read().strip()
        try:
            last_run = datetime.fromisoformat(last_run_str)
            
            if success_file.exists():
                with open(success_file, 'r') as f:
                    last_success_str = f.read().strip()
                try:
                    last_success = datetime.fromisoformat(last_success_str)
                    if now - last_success < timedelta(minutes=10):
                        log.info(f"⏱ Already ran successfully at {last_success} - exiting")
                        return False
                except Exception:
                    pass
            else:
                if now - last_run < timedelta(minutes=2):
                    log.info(f"⏱ Last run failed at {last_run}, waiting for retry window (2 min)")
                    return False
                else:
                    log.info(f"⚠️ Last run failed, attempting retry...")
        except Exception:
            pass

    last_run_file.parent.mkdir(exist_ok=True)
    with open(last_run_file, 'w') as f:
        f.write(now.isoformat())
    return True

def mark_success():
    """Mark run as successful"""
    success_file = Path("signal_state/last_success.txt")
    success_file.parent.mkdir(exist_ok=True)
    with open(success_file, 'w') as f:
        f.write(datetime.now(timezone.utc).isoformat())

# =========================
# MAIN WITH PARALLEL PROCESSING
# =========================
def main():
    if not in_execution_window():
        return

    sentiment_status = "ON" if USE_SENTIMENT else "OFF"
    log.info(f"🚀 Starting Trade Beacon v2.0.3 - Mode={MODE} | Sentiment={sentiment_status}")
    log.info(f"📊 Monitoring {len(PAIRS)} pairs: {', '.join([p.replace('=X', '') for p in PAIRS])}")
    log.info(f"💰 Features: Configurable ATR | Enhanced validation | Risk limits")
    
    active = []
    successful_downloads = 0
    newsapi_calls = 0
    marketaux_calls = 0

    _market_cache.clear()
    log.info("🔄 Cache cleared for fresh data")

    log.info("🔍 Analyzing pairs in parallel...")
    
    with ThreadPoolExecutor(max_workers=min(5, len(PAIRS))) as executor:
        futures = {executor.submit(generate_signal, pair): pair for pair in PAIRS}
        
        for future in as_completed(futures):
            pair = futures[future]
            try:
                sig, download_ok = future.result()
                
                # ✅ Track download success
                if download_ok:
                    successful_downloads += 1
                
                if sig:
                    active.append(sig)
                    log.info(f"✅ {pair.replace('=X', '')} - Signal generated "
                            f"(Score: {sig['score']}, RR: {sig['risk_reward']:.2f}, "
                            f"Modes: {', '.join(sig['eligible_modes'])})")
                else:
                    if download_ok:
                        log.info(f"⏭️ {pair.replace('=X', '')} - No signal")
                    else:
                        log.warning(f"⚠️ {pair.replace('=X', '')} - Download failed")
            except Exception as e:
                log.error(f"❌ {pair.replace('=X', '')} failed: {e}")

    # ✅ Apply risk management filters
    if active:
        active, risk_warnings = check_risk_limits(active, CONFIG)
        for warning in risk_warnings:
            log.warning(f"⚠️ Risk Management: {warning}")

    # ✅ Sentiment enhancement
    if USE_SENTIMENT and active:
        try:
            news_agg = NewsAggregator()
            active = enhance_with_sentiment(active, news_agg)
            newsapi_calls = news_agg.newsapi_calls
            marketaux_calls = news_agg.marketaux_calls
            log.info("✅ Sentiment analysis complete")
        except Exception as e:
            log.error(f"❌ Sentiment analysis failed: {e}")
            log.info("⚠️ Continuing with technical signals only")

    log.info(f"\n✅ Cycle complete | Active signals: {len(active)}")
    write_dashboard_state(active, successful_downloads, newsapi_calls, marketaux_calls)

    if active:
        df = pd.DataFrame(active)
        df.to_csv("signals.csv", index=False)
        log.info("📄 signals.csv written")
        
        print("\n" + "="*80)
        print(f"🎯 {MODE.upper()} SIGNALS {'+ SENTIMENT' if USE_SENTIMENT else ''} (v2.0.3):")
        print("="*80)
        
        display_cols = ["signal_id", "pair", "direction", "score", "confidence", 
                       "hold_time", "risk_reward", "eligible_modes"]
        print(df[display_cols].to_string(index=False))
        print("="*80 + "\n")
        
    else:
        log.info("✅ No strong signals this cycle")
    
    mark_success()
    log.info("✅ Run completed successfully - Trade Beacon v2.0.3")


if __name__ == "__main__":
    main()
