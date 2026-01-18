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
    {"EURUSD", "GBPUSD"},  # Both have EUR/GBP exposure
    {"EURUSD", "EURGBP"},
    {"GBPUSD", "EURGBP"},
    {"USDJPY", "EURJPY"},  # JPY exposure
    {"USDJPY", "GBPJPY"},
    {"EURJPY", "GBPJPY"},
]

INTERVAL = "15m"
LOOKBACK = "14d"
MIN_ROWS = 220
CACHE_TTL_SECONDS = 300  # 5 minutes

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
        log.warning("⚠️ config.json not found, using aggressive defaults")
        return {
            "mode": "aggressive",
            "use_sentiment": True,
            "settings": {
                "aggressive": {
                    "threshold": 30,
                    "min_adx": 15,
                    "rsi_oversold": 45,
                    "rsi_overbought": 55,
                    "min_volume_ratio": 0.8,
                    "volume_penalty": 5
                },
                "conservative": {
                    "threshold": 50,
                    "min_adx": 20,
                    "rsi_oversold": 35,
                    "rsi_overbought": 65,
                    "min_volume_ratio": 1.0,
                    "volume_penalty": 10
                }
            }
        }
    
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
    
    log.info(f"✅ Config loaded: mode={mode}, sentiment={config.get('use_sentiment', True)}")
    return config

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
USE_SENTIMENT = CONFIG.get("use_sentiment", True) and validate_api_keys()
SETTINGS = CONFIG["settings"][MODE]

SIGNAL_THRESHOLD = SETTINGS["threshold"]
MIN_ADX = SETTINGS.get("min_adx", 15)
ADX_STRONG = 25
RSI_OVERSOLD = SETTINGS.get("rsi_oversold", 45)
RSI_OVERBOUGHT = SETTINGS.get("rsi_overbought", 55)
MIN_VOLUME_RATIO = SETTINGS.get("min_volume_ratio", 0.8)
VOLUME_PENALTY = SETTINGS.get("volume_penalty", 5)

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

def get_session_bonus(pair: str, session: str) -> int:
    """Apply session-based scoring bonus"""
    bonus = 0
    
    if session == "ASIAN":
        if "JPY" in pair:
            bonus = 5
        elif pair in ["AUDUSD=X", "NZDUSD=X"]:
            bonus = 5
    
    elif session in ["EUROPEAN", "OVERLAP"]:
        if pair in ["EURUSD=X", "GBPUSD=X", "EURGBP=X"]:
            bonus = 5
        elif "EUR" in pair or "GBP" in pair:
            bonus = 3
    
    elif session == "OVERLAP":
        bonus += 3
    
    elif session == "US":
        if pair in ["EURUSD=X", "GBPUSD=X", "USDCAD=X"]:
            bonus = 5
    
    return bonus

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
        
        # Lazy verification - don't verify on startup
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
        """
        Analyze sentiment with automatic model fallback (thread-safe).
        """
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
                # Expired
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
def download(pair: str) -> pd.DataFrame:
    """Download data with TTL-based caching"""
    # Check cache first
    cached = _market_cache.get(pair)
    if cached is not None:
        log.debug(f"📦 Using cached data for {pair}")
        return cached
    
    # Download fresh data
    log.debug(f"📥 Downloading fresh data for {pair}")
    df = yf.download(
        pair,
        interval=INTERVAL,
        period=LOOKBACK,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.dropna()
    
    # Validate data quality
    if len(df) < MIN_ROWS:
        log.warning(f"⚠️ {pair} insufficient data: {len(df)} rows")
        return df
    
    # Cache the result
    _market_cache.set(pair, df)
    
    return df

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
    return SPREADS.get(clean_pair, 0.0002)  # Default 2 pips

def classify_market_state(adx: float, atr: float, volume_ratio: float) -> str:
    """
    Classify current market conditions for signal context.
    
    Returns:
        - CHOPPY: Low trend strength, difficult trading conditions
        - CONSOLIDATING: Moderate conditions, range-bound
        - TRENDING_MODERATE: Clear trend forming
        - TRENDING_STRONG: Strong directional movement with volume
    """
    if adx < 15:
        return "CHOPPY"
    elif adx > 25 and volume_ratio > 1.2:
        return "TRENDING_STRONG"
    elif adx > 20:
        return "TRENDING_MODERATE"
    else:
        return "CONSOLIDATING"

def get_signal_type(e12: float, e26: float, e200: float, rsi: float) -> str:
    """
    Classify the type of trading opportunity.
    
    Returns:
        - trend-continuation: Following established trend
        - reversal: Counter-trend opportunity
        - breakout: Breaking consolidation
        - momentum: Strong directional move
    """
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

def calculate_signal_freshness(timestamp: str) -> dict:
    """
    Calculate how fresh/stale a signal is and apply confidence decay.
    
    Signals lose validity over time as market conditions change.
    
    Returns:
        Dict with freshness classification, age, and confidence decay factor
    """
    try:
        generated = datetime.fromisoformat(timestamp)
        age_minutes = (datetime.now(timezone.utc) - generated).total_seconds() / 60
        
        if age_minutes < 15:
            freshness = "FRESH"
        elif age_minutes < 30:
            freshness = "RECENT"
        elif age_minutes < 60:
            freshness = "AGING"
        else:
            freshness = "STALE"
        
        # Confidence decays 2% per minute (100% at 0min, 50% at 25min, 0% at 50min)
        confidence_decay = max(0, 100 - (age_minutes * 2))
        
        return {
            "freshness": freshness,
            "age_minutes": round(age_minutes, 1),
            "confidence_decay": round(confidence_decay, 1)
        }
    except Exception:
        return {
            "freshness": "UNKNOWN",
            "age_minutes": 0,
            "confidence_decay": 100
        }

# =========================
# SIGNAL ENGINE WITH SPREAD AWARENESS
# =========================
def generate_signal(pair: str) -> dict | None:
    """Generate trading signal with spread-adjusted SL/TP"""
    df = download(pair)
    if len(df) < MIN_ROWS:
        log.warning(f"⚠️ {pair} not enough candles ({len(df)}), skipping")
        return None
    
    try:
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

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
        return None

    if None in (e12, e26, e200, r, a, current_price, atr):
        log.warning(f"⚠️ {pair} indicators incomplete, skipping")
        return None

    # ADX filter
    if a < MIN_ADX:
        log.info(f"❌ {pair} | ADX too low ({a:.1f} < {MIN_ADX})")
        return None

    bull = bear = 0

    # EMA Trend Structure (40 points)
    if e12 > e26 > e200:
        bull += 40
    elif e12 < e26 < e200:
        bear += 40

    # RSI Context (20-30 points)
    if MODE == "conservative":
        if r < RSI_OVERSOLD:
            bull += 30
        elif r > RSI_OVERBOUGHT:
            bear += 30
    else:
        if r < RSI_OVERSOLD:
            bull += 20
        elif r > RSI_OVERBOUGHT:
            bear += 20

    # ADX Trend Strength (10-20 points)
    if a > ADX_STRONG:
        if e12 > e26:
            bull += 20
        elif e12 < e26:
            bear += 20
    elif a > MIN_ADX:
        if e12 > e26:
            bull += 10
        elif e12 < e26:
            bear += 10
    
    # Volume handling - penalty not rejection
    if volume_ratio >= MIN_VOLUME_RATIO:
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
        penalty = VOLUME_PENALTY
        if e12 > e26:
            bear += penalty
        else:
            bull += penalty
    
    # Session bonus
    session = get_market_session()
    session_bonus = get_session_bonus(pair, session)
    if e12 > e26:
        bull += session_bonus
    else:
        bear += session_bonus

    diff = abs(bull - bear)

    if diff < SIGNAL_THRESHOLD:
        return None

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
    
    # ATR-based dynamic stops with spread adjustment
    if direction == "BUY":
        entry_price = current_price + (spread / 2)  # Account for spread
        sl = entry_price - (1.5 * atr)
        tp = entry_price + (2.5 * atr)
    else:
        entry_price = current_price - (spread / 2)  # Account for spread
        sl = entry_price + (1.5 * atr)
        tp = entry_price - (2.5 * atr)
    
    # Calculate actual risk-reward ratio
    risk = abs(entry_price - sl)
    reward = abs(tp - entry_price)
    risk_reward = reward / risk if risk > 0 else 0
    
    # Minimum RR filter
    if risk_reward < 1.5:
        log.info(f"❌ {pair} | Poor risk-reward ({risk_reward:.2f} < 1.5)")
        return None
    
    # Calculate signal metadata
    signal_type = get_signal_type(e12, e26, e200, r)
    market_state = classify_market_state(a, atr, volume_ratio)
    
    # Generate timestamp and expiration
    now = datetime.now(timezone.utc)
    valid_for_minutes = 60  # Signals valid for 1 hour
    expires_at = now + timedelta(minutes=valid_for_minutes)

    return {
        "pair": pair.replace("=X", ""),
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
        # ✅ NEW: Signal metadata for subscribers
        "metadata": {
            "signal_type": signal_type,
            "market_state": market_state,
            "timeframe": INTERVAL,
            "valid_for_minutes": valid_for_minutes,
            "generated_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "session_active": session,
            "signal_generator_version": "2.0.0"
        }
    }

# =========================
# CORRELATION FILTER
# =========================
def filter_correlated_signals(signals: List[Dict]) -> List[Dict]:
    """Remove correlated signals to avoid double exposure"""
    if len(signals) <= 1:
        return signals
    
    filtered = []
    pairs_taken = set()
    
    # Sort by score (keep highest quality signals)
    sorted_signals = sorted(signals, key=lambda x: x['score'], reverse=True)
    
    for signal in sorted_signals:
        pair = f"{signal['pair']}=X"
        
        # Check if correlated with already selected pairs
        is_correlated = False
        for taken_pair in pairs_taken:
            for corr_group in CORRELATED_PAIRS:
                if pair in corr_group and taken_pair in corr_group:
                    is_correlated = True
                    log.info(f"⚠️ Skipping {signal['pair']} (correlated with {taken_pair.replace('=X', '')})")
                    break
            if is_correlated:
                break
        
        if not is_correlated:
            filtered.append(signal)
            pairs_taken.add(pair)
    
    if len(filtered) < len(signals):
        log.info(f"🔗 Correlation filter: {len(signals)} → {len(filtered)} signals")
    
    return filtered

# =========================
# SENTIMENT ENHANCEMENT - FIXED DIRECTION LOGIC
# =========================
def enhance_with_sentiment(signals: List[Dict], news_agg: NewsAggregator) -> List[Dict]:
    """
    Add sentiment analysis to signals with proper direction logic.
    """
    
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
        
        pair_articles = filter_articles_for_pair(pair, all_articles)
        
        sentiment_data = analyze_sentiment_from_articles(
            pair_ticker, 
            pair_articles, 
            analyzer
        )
        
        original_score = signal['technical_score']
        adjustment = sentiment_data['adjustment']
        
        # CRITICAL FIX: Sentiment should SUPPORT the direction
        # BUY + bullish news = boost
        # BUY + bearish news = penalty
        # SELL + bearish news = boost (double negative = positive)
        # SELL + bullish news = penalty
        direction_multiplier = 1 if signal['direction'] == 'BUY' else -1
        final_adjustment = adjustment * direction_multiplier
        
        signal['score'] = original_score + final_adjustment
        signal['score'] = max(0, min(100, signal['score']))
        signal['sentiment_score'] = adjustment
        
        # Re-validate against threshold after sentiment
        if signal['score'] < SIGNAL_THRESHOLD:
            log.info(f"❌ {pair} | Signal too weak after sentiment ({signal['score']} < {SIGNAL_THRESHOLD})")
            continue
        
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
# DASHBOARD & HEALTH CHECK
# =========================
def write_dashboard_state(signals: list, successful_downloads: int, newsapi_calls: int = 0, marketaux_calls: int = 0):
    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 8:
        session = "ASIAN"
    elif 8 <= hour < 16:
        session = "EUROPEAN"
    else:
        session = "US"

    performance = track_performance(signals)

    dashboard_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_signals": len(signals),
        "session": session,
        "mode": MODE,
        "sentiment_enabled": USE_SENTIMENT,
        "signals": signals,
        "api_usage": {
            "yfinance": {"successful_downloads": successful_downloads},
            "sentiment": {
                "enabled": USE_SENTIMENT,
                "newsapi": newsapi_calls,
                "marketaux": marketaux_calls
            }
        },
        "stats": performance["stats"],
        "risk_management": performance["risk_management"]
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
                f"Total Pips: {stats['total_pips']}")
    
    write_health_check(signals, successful_downloads, newsapi_calls, marketaux_calls)

def write_health_check(signals: list, successful_downloads: int, newsapi_calls: int, marketaux_calls: int):
    """Write health check file for monitoring"""
    health = {
        "status": "ok",
        "last_run": datetime.now(timezone.utc).isoformat(),
        "signal_count": len(signals),
        "api_status": {
            "yfinance": "ok" if successful_downloads > 0 else "error",
            "newsapi": "ok" if newsapi_calls > 0 else ("disabled" if not USE_SENTIMENT else "error"),
            "marketaux": "ok" if marketaux_calls > 0 else ("disabled" if not USE_SENTIMENT else "error")
        },
        "mode": MODE,
        "pairs_monitored": len(PAIRS)
    }
    
    output_dir = Path("signal_state")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "health.json", "w") as f:
        json.dump(health, f, indent=2)

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
    log.info(f"🚀 Starting Trade Beacon - Mode={MODE} | Sentiment={sentiment_status}")
    log.info(f"📊 Monitoring {len(PAIRS)} pairs: {', '.join([p.replace('=X', '') for p in PAIRS])}")
    log.info(f"💰 Spread-aware pricing enabled")
    
    active = []
    successful_downloads = 0
    newsapi_calls = 0
    marketaux_calls = 0

    # Clear cache at start of each run to ensure fresh data
    _market_cache.clear()
    log.info("🔄 Cache cleared for fresh data")

    # Generate technical signals with parallel processing
    log.info("🔍 Analyzing pairs in parallel...")
    
    with ThreadPoolExecutor(max_workers=min(5, len(PAIRS))) as executor:
        futures = {executor.submit(generate_signal, pair): pair for pair in PAIRS}
        
        for future in as_completed(futures):
            pair = futures[future]
            try:
                sig = future.result()
                if sig:
                    active.append(sig)
                    successful_downloads += 1
                    log.info(f"✅ {pair.replace('=X', '')} - Signal generated (RR: {sig['risk_reward']:.2f})")
                else:
                    successful_downloads += 1
                    log.info(f"⏭️ {pair.replace('=X', '')} - No signal")
            except Exception as e:
                log.error(f"❌ {pair.replace('=X', '')} failed: {e}")

    # Apply correlation filter
    if len(active) > 1:
        active = filter_correlated_signals(active)

    # Enhance with sentiment
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
        print(f"🎯 {MODE.upper()} SIGNALS {'+ SENTIMENT' if USE_SENTIMENT else ''} (SPREAD-ADJUSTED):")
        print("="*80)
        
        display_cols = ["pair", "direction", "score", "confidence", "rsi", "adx", 
                       "volume_ratio", "session", "risk_reward", "spread"]
        print(df[display_cols].to_string(index=False))
        print("="*80 + "\n")
        
        # Show signal metadata
        print("📊 SIGNAL METADATA:")
        print("="*80)
        for _, row in df.iterrows():
            meta = row.get('metadata', {})
            if isinstance(meta, dict):
                freshness_data = calculate_signal_freshness(row['timestamp'])
                print(f"{row['pair']}: {meta.get('signal_type', 'N/A').upper()} | "
                      f"Market: {meta.get('market_state', 'N/A')} | "
                      f"Valid: {meta.get('valid_for_minutes', 0)}min | "
                      f"Expires: {meta.get('expires_at', 'N/A')[:16]} | "
                      f"Freshness: {freshness_data['freshness']}")
        print("="*80 + "\n")
        
        # Show sentiment details if enabled
        if USE_SENTIMENT and "sentiment" in df.columns:
            print("📰 SENTIMENT DETAILS:")
            print("="*80)
            for _, row in df.iterrows():
                sent = row.get('sentiment', {})
                if isinstance(sent, dict):
                    tech_score = row.get('technical_score', 0)
                    final_score = row.get('score', 0)
                    sentiment_impact = final_score - tech_score
                    
                    print(f"{row['pair']} ({row['direction']}): "
                          f"{sent.get('overall', 'N/A').upper()} "
                          f"({sent.get('adjustment', 0):+d} → {sentiment_impact:+d} pts, "
                          f"{sent.get('news_count', 0)} articles) | "
                          f"Tech: {tech_score} → Final: {final_score}")
            print("="*80 + "\n")
        
        # Show risk summary
        print("💰 RISK SUMMARY:")
        print("="*80)
        total_risk = 0
        for _, row in df.iterrows():
            risk_pips = abs(row['entry_price'] - row['sl']) * 10000  # Convert to pips
            reward_pips = abs(row['tp'] - row['entry_price']) * 10000
            print(f"{row['pair']}: Risk {risk_pips:.1f} pips | Reward {reward_pips:.1f} pips | "
                  f"RR {row['risk_reward']:.2f} | Spread {row['spread']*10000:.1f} pips")
            total_risk += risk_pips
        print(f"\nTotal Risk Exposure: {total_risk:.1f} pips")
        print("="*80 + "\n")
        
        # Export signals with metadata for API/webhook consumers
        signals_with_metadata = []
        for _, row in df.iterrows():
            signal_dict = row.to_dict()
            # Add freshness calculation
            signal_dict['freshness'] = calculate_signal_freshness(row['timestamp'])
            signals_with_metadata.append(signal_dict)
        
        # Write enriched signals for API consumers
        output_dir = Path("signal_state")
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / "signals_enriched.json", 'w') as f:
            json.dump(signals_with_metadata, f, indent=2)
        log.info("📡 Enriched signals written to signal_state/signals_enriched.json")
        
    else:
        log.info("✅ No strong signals this cycle")
    
    mark_success()
    log.info("✅ Run completed successfully")


if __name__ == "__main__":
    main()
