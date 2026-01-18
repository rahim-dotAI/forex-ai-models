import logging
import sys
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import time

import pandas as pd
import yfinance as yf
import requests
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# Import performance tracker
from performance_tracker import track_performance

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("trade-beacon")

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
                    "min_adx": None,
                    "rsi_oversold": 45,
                    "rsi_overbought": 55
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
# CONFIG
# =========================
PAIRS = ["USDJPY=X", "AUDUSD=X", "NZDUSD=X"]
INTERVAL = "15m"
LOOKBACK = "14d"
MIN_ROWS = 220

CONFIG = load_config()
MODE = CONFIG["mode"]
USE_SENTIMENT = CONFIG.get("use_sentiment", True)
SETTINGS = CONFIG["settings"][MODE]

SIGNAL_THRESHOLD = SETTINGS["threshold"]
MIN_ADX = SETTINGS.get("min_adx")
RSI_OVERSOLD = SETTINGS.get("rsi_oversold", 45)
RSI_OVERBOUGHT = SETTINGS.get("rsi_overbought", 55)

# =========================
# IMPROVED SENTIMENT ANALYSIS WITH FALLBACK MODELS
# =========================
class SentimentAnalyzer:
    """
    Multi-model sentiment analyzer with automatic fallback.
    Handles deprecated models gracefully.
    """
    
    def __init__(self, hf_api_key: str = None):
        self.hf_api_key = hf_api_key
        self.session = requests.Session()
        if hf_api_key:
            self.session.headers.update({"Authorization": f"Bearer {hf_api_key}"})
        
        # Updated model list with working alternatives (in priority order)
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
        self.failed_models = set()
        log.info(f"🤖 Sentiment analyzer initialized with {len(self.models)} fallback models")
    
    def _normalize_label(self, raw_label: str, model_idx: int) -> str:
        """Normalize model-specific labels to positive/negative/neutral"""
        label_map = self.models[model_idx].get("label_map", {})
        normalized = label_map.get(raw_label.lower(), "neutral")
        return normalized
    
    def analyze(self, text: str, max_retries: int = 2) -> Dict:
        """
        Analyze sentiment with automatic model fallback.
        
        Returns:
            Dict with 'label' (positive/negative/neutral) and 'score' (0.0-1.0)
        """
        if not text or len(text.strip()) < 10:
            log.debug("Text too short for sentiment analysis")
            return {"label": "neutral", "score": 0.0}
        
        # Try each model until one works
        for attempt in range(len(self.models)):
            if self.current_model_idx in self.failed_models:
                self.current_model_idx = (self.current_model_idx + 1) % len(self.models)
                continue
            
            model = self.models[self.current_model_idx]
            
            try:
                result = self._call_model(model, text, max_retries)
                if result:
                    return result
                else:
                    # Model failed, try next
                    log.warning(f"⚠️ {model['name']} failed, trying next model...")
                    self.failed_models.add(self.current_model_idx)
                    self.current_model_idx = (self.current_model_idx + 1) % len(self.models)
                    
            except Exception as e:
                log.warning(f"⚠️ {model['name']} error: {e}")
                self.failed_models.add(self.current_model_idx)
                self.current_model_idx = (self.current_model_idx + 1) % len(self.models)
        
        # All models failed
        log.warning("⚠️ All sentiment models failed, defaulting to neutral")
        return {"label": "neutral", "score": 0.0}
    
    def _call_model(self, model: Dict, text: str, max_retries: int) -> Optional[Dict]:
        """Call a specific HuggingFace model"""
        for retry in range(max_retries):
            try:
                response = self.session.post(
                    model["url"],
                    json={"inputs": text[:512]},  # Truncate to avoid token limits
                    timeout=15
                )
                
                # Handle model loading
                if response.status_code == 503:
                    if retry < max_retries - 1:
                        wait_time = 10 * (retry + 1)
                        log.info(f"⏳ {model['name']} loading, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return None
                
                # Handle gone/deprecated models
                if response.status_code == 410:
                    log.warning(f"⚠️ {model['name']} is deprecated (410 Gone)")
                    return None
                
                # Handle rate limits
                if response.status_code == 429:
                    if retry < max_retries - 1:
                        log.warning(f"⚠️ {model['name']} rate limited, waiting...")
                        time.sleep(5 * (retry + 1))
                        continue
                    else:
                        return None
                
                response.raise_for_status()
                result = response.json()
                
                # Parse response
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        # Format: [[{"label": "...", "score": ...}]]
                        top = max(result[0], key=lambda x: x.get('score', 0))
                    else:
                        # Format: [{"label": "...", "score": ...}]
                        top = max(result, key=lambda x: x.get('score', 0))
                    
                    raw_label = top.get('label', 'neutral')
                    normalized_label = self._normalize_label(raw_label, self.current_model_idx)
                    score = round(top.get('score', 0.0), 3)
                    
                    if retry > 0:
                        log.info(f"✅ {model['name']} succeeded on retry {retry + 1}")
                    
                    return {"label": normalized_label, "score": score}
                
                return {"label": "neutral", "score": 0.0}
                
            except requests.exceptions.RequestException as e:
                if retry < max_retries - 1:
                    log.debug(f"Request error, retrying... ({e})")
                    time.sleep(2)
                else:
                    log.debug(f"Request failed after retries: {e}")
                    return None
            except Exception as e:
                log.debug(f"Unexpected error: {e}")
                return None
        
        return None


class NewsAggregator:
    """Fetch forex news using NewsAPI and Marketaux"""
    
    def __init__(self):
        # Try environment variables first (GitHub Actions), then local files
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
        
        # Track API usage to stay within free tiers
        self.newsapi_calls = 0
        self.marketaux_calls = 0
    
    def get_news(self, pairs: List[str]) -> List[Dict]:
        """
        Fetch recent forex news from both NewsAPI and Marketaux.
        
        ✅ OPTIMIZED: Makes only 1 call per API regardless of number of pairs
        
        API Limits:
        - NewsAPI: 100 requests/day
        - Marketaux: 100 requests/day
        
        Usage: 96 runs/day × 1 call/run = 96 calls/day per API ✅
        """
        all_articles = []
        
        # Extract ALL currencies from ALL pairs (deduplicated)
        currencies = set()
        for pair in pairs:
            clean = pair.replace("=X", "")
            if len(clean) >= 6:
                currencies.add(clean[:3])  # First currency
                currencies.add(clean[3:6])  # Second currency
            else:
                currencies.add(clean[:3])
        
        log.info(f"🔍 Searching news for currencies: {', '.join(sorted(currencies))}")
        
        # ✅ Fetch from NewsAPI - SINGLE CALL for all currencies
        if self.newsapi_key:
            newsapi_articles = self._fetch_newsapi(currencies, limit=5)
            all_articles.extend(newsapi_articles)
            self.newsapi_calls += 1
        else:
            log.warning("⚠️ No NewsAPI key - skipping NewsAPI")
        
        # ✅ Fetch from Marketaux - SINGLE CALL for all currencies
        if self.marketaux_key:
            marketaux_articles = self._fetch_marketaux(currencies, limit=5)
            all_articles.extend(marketaux_articles)
            self.marketaux_calls += 1
        else:
            log.warning("⚠️ No Marketaux key - skipping Marketaux")
        
        if not all_articles:
            log.warning("⚠️ No news sources available - sentiment disabled")
        else:
            log.info(f"📰 Fetched {len(all_articles)} articles total "
                    f"(NewsAPI: {len([a for a in all_articles if a.get('source_api') == 'NewsAPI'])}, "
                    f"Marketaux: {len([a for a in all_articles if a.get('source_api') == 'Marketaux'])})")
        
        return all_articles
    
    def _fetch_newsapi(self, currencies: set, limit: int = 5) -> List[Dict]:
        """Fetch news from NewsAPI - SINGLE CALL"""
        try:
            # Build query with OR operator for all currencies
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
        """
        Fetch news from Marketaux API - SINGLE CALL
        Optimized for forex news with financial focus.
        """
        try:
            # Marketaux uses symbols like USD, EUR, JPY
            # Build query with all currencies (comma-separated)
            symbols = ",".join(sorted(list(currencies)[:5]))  # Limit to 5 to keep query manageable
            
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
                articles.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "source": article.get("source", "Marketaux"),
                    "published": article.get("published_at", ""),
                    "source_api": "Marketaux",
                    # Store Marketaux sentiment for comparison (optional)
                    "marketaux_sentiment": article.get("entities", [{}])[0].get("sentiment_score") 
                        if article.get("entities") else None
                })
            
            log.info(f"✅ Marketaux: {len(articles)} articles")
            return articles
            
        except Exception as e:
            log.error(f"❌ Marketaux fetch failed: {e}")
            return []


def filter_articles_for_pair(pair: str, all_articles: List[Dict]) -> List[Dict]:
    """
    Filter articles relevant to a specific currency pair.
    
    Example: For USDJPY, look for articles mentioning USD or JPY
    """
    # Extract currencies from pair (e.g., "USDJPY" -> ["USD", "JPY"])
    currencies = [pair[:3], pair[3:6]] if len(pair) >= 6 else [pair[:3]]
    
    relevant_articles = []
    
    for article in all_articles:
        text = f"{article.get('title', '')} {article.get('description', '')}".upper()
        
        # Check if any currency is mentioned
        if any(curr in text for curr in currencies):
            relevant_articles.append(article)
    
    return relevant_articles


def analyze_sentiment_from_articles(pair: str, articles: List[Dict], 
                                    analyzer: SentimentAnalyzer) -> Dict:
    """
    Analyze sentiment from pre-fetched articles (no API calls here).
    """
    
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
        
        # Track which API the article came from
        if article.get('source_api') == 'NewsAPI':
            newsapi_count += 1
        elif article.get('source_api') == 'Marketaux':
            marketaux_count += 1
        
        log.info(f"📰 [{article.get('source_api', 'Unknown')}] "
                f"{article['title'][:50]}... | "
                f"{sentiment['label']} ({sentiment['score']:.2f})")
    
    positive = sum(1 for s in sentiments if s['label'] == 'positive')
    negative = sum(1 for s in sentiments if s['label'] == 'negative')
    
    # Calculate adjustment with cap at ±20
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
# TECHNICAL ANALYSIS UTILS
# =========================
def last(series: pd.Series):
    return None if series is None or series.empty else float(series.iloc[-1])

def download(pair: str) -> pd.DataFrame:
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
    return df.dropna()

def ema(series, period):
    return EMAIndicator(series, window=period).ema_indicator()

def rsi(series, period=14):
    return RSIIndicator(series, window=period).rsi()

def adx_calc(high, low, close):
    return ADXIndicator(high, low, close, window=14).adx()

def atr_calc(high, low, close):
    return AverageTrueRange(high, low, close, window=14).average_true_range()

# =========================
# SIGNAL ENGINE
# =========================
def generate_signal(pair: str) -> dict | None:
    df = download(pair)
    if len(df) < MIN_ROWS:
        log.warning(f"⚠️ {pair} not enough candles ({len(df)}), skipping")
        return None
    
    try:
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()

        e12 = last(ema(close, 12))
        e26 = last(ema(close, 26))
        e200 = last(ema(close, 200))
        r = last(rsi(close))
        a = last(adx_calc(high, low, close))
        atr = last(atr_calc(high, low, close))
        current_price = last(close)

    except Exception as e:
        log.warning(f"⚠️ {pair} indicator calc failed: {e}")
        return None

    if None in (e12, e26, e200, r, a, current_price, atr):
        log.warning(f"⚠️ {pair} indicators incomplete, skipping")
        return None

    if MIN_ADX is not None and a < MIN_ADX:
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
    if a > 25:
        if e12 > e26:
            bull += 20
        elif e12 < e26:
            bear += 20
    elif MIN_ADX and a > MIN_ADX:
        if e12 > e26:
            bull += 10
        elif e12 < e26:
            bear += 10
    elif not MIN_ADX and a > 15:
        if e12 > e26:
            bull += 10
        elif e12 < e26:
            bear += 10

    diff = abs(bull - bear)
    
    quality = (
        "⭐⭐⭐" if diff >= 70 else
        "⭐⭐"  if diff >= 60 else
        "⭐"   if diff >= 50 else
        ""
    )

    log.info(
        f"{pair} | Bull={bull} Bear={bear} Diff={diff} {quality} | RSI={r:.1f} ADX={a:.1f}"
    )

    if diff < SIGNAL_THRESHOLD:
        log.info(f"❌ {pair} | Signal too weak (diff={diff} < {SIGNAL_THRESHOLD})")
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

    # ATR-based dynamic stops
    if direction == "BUY":
        sl = current_price - (1.5 * atr)
        tp = current_price + (2.5 * atr)
    else:
        sl = current_price + (1.5 * atr)
        tp = current_price - (2.5 * atr)

    return {
        "pair": pair.replace("=X", ""),
        "direction": direction,
        "score": diff,
        "technical_score": diff,
        "confidence": confidence,
        "rsi": round(r, 1),
        "adx": round(a, 1),
        "atr": round(atr, 5),
        "entry_price": round(current_price, 5),
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "risk_reward": 1.67,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# =========================
# SENTIMENT ENHANCEMENT (FIXED)
# =========================
def enhance_with_sentiment(signals: List[Dict], news_agg: NewsAggregator) -> List[Dict]:
    """
    Add sentiment analysis to signals.
    ✅ OPTIMIZED: Fetch news ONCE for all pairs, then filter per signal.
    """
    
    if not USE_SENTIMENT or not signals:
        return signals
    
    log.info("\n" + "="*70)
    log.info("📰 Analyzing news sentiment from NewsAPI + Marketaux...")
    log.info("="*70)
    
    # Get HF API key if available
    hf_key = os.environ.get('HF_API_KEY') or os.environ.get('HUGGINGFACE_API_KEY')
    analyzer = SentimentAnalyzer(hf_api_key=hf_key)
    
    # ✅ FIX: Fetch news ONCE for all pairs
    all_pairs = [f"{sig['pair']}=X" for sig in signals]
    log.info(f"🔍 Fetching news for {len(all_pairs)} pairs: {', '.join(all_pairs)}")
    
    # This makes only 1 NewsAPI call + 1 Marketaux call total
    all_articles = news_agg.get_news(all_pairs)
    
    enhanced = []
    
    for signal in signals:
        pair = signal['pair']
        pair_ticker = f"{pair}=X"
        
        # ✅ FIX: Filter articles relevant to this pair
        pair_articles = filter_articles_for_pair(pair, all_articles)
        
        # Analyze sentiment for this pair's articles
        sentiment_data = analyze_sentiment_from_articles(
            pair_ticker, 
            pair_articles, 
            analyzer
        )
        
        original_score = signal['score']
        adjustment = sentiment_data['adjustment']
        
        # Apply adjustment based on direction
        if signal['direction'] == 'BUY':
            signal['score'] += adjustment
        else:
            signal['score'] -= adjustment
        
        signal['score'] = max(0, min(100, signal['score']))
        
        # Update confidence based on new score
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
        
        log.info(f"💡 {pair} | Sentiment: {sentiment_data['sentiment']} | "
                f"Score: {original_score} → {signal['score']} ({adjustment:+d}) | "
                f"Articles: {sentiment_data['news_count']}")
        
        enhanced.append(signal)
    
    # Log API usage stats
    log.info(f"📊 API Usage: NewsAPI calls={news_agg.newsapi_calls}, "
             f"Marketaux calls={news_agg.marketaux_calls}")
    
    return enhanced

# =========================
# DASHBOARD
# =========================
def write_dashboard_state(signals: list, api_calls: int, newsapi_calls: int = 0, marketaux_calls: int = 0):
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
            "yfinance": {"calls": api_calls},
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

# =========================
# TIME-WINDOW GUARD (FIXED)
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
            
            # Check if last run succeeded
            if success_file.exists():
                with open(success_file, 'r') as f:
                    last_success_str = f.read().strip()
                try:
                    last_success = datetime.fromisoformat(last_success_str)
                    # If successful run was recent, skip
                    if now - last_success < timedelta(minutes=10):
                        log.info(f"⏱ Already ran successfully at {last_success} - exiting")
                        return False
                except Exception:
                    pass
            else:
                # Last run failed, allow retry after 2 minutes
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
# MAIN
# =========================
def main():
    if not in_execution_window():
        return

    sentiment_status = "ON" if USE_SENTIMENT else "OFF"
    log.info(f"🚀 Starting Trade Beacon - Mode={MODE} | Sentiment={sentiment_status}")
    
    active = []
    api_calls = 0
    newsapi_calls = 0
    marketaux_calls = 0

    # Generate technical signals
    for pair in PAIRS:
        log.info(f"🔍 Analyzing {pair.replace('=X','')}...")
        sig = generate_signal(pair)
        api_calls += 1
        if sig:
            active.append(sig)

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
    write_dashboard_state(active, api_calls, newsapi_calls, marketaux_calls)

    if active:
        df = pd.DataFrame(active)
        df.to_csv("signals.csv", index=False)
        log.info("📄 signals.csv written")
        
        print("\n" + "="*70)
        print(f"🎯 {MODE.upper()} SIGNALS {'+ SENTIMENT' if USE_SENTIMENT else ''}:")
        print("="*70)
        
        display_cols = ["pair", "direction", "score", "confidence", "rsi", "adx", "entry_price", "sl", "tp"]
        print(df[display_cols].to_string(index=False))
        print("="*70 + "\n")
        
        # Show sentiment details if enabled
        if USE_SENTIMENT and "sentiment" in df.columns:
            print("📰 SENTIMENT DETAILS:")
            print("="*70)
            for _, row in df.iterrows():
                sent = row.get('sentiment', {})
                if isinstance(sent, dict):
                    print(f"{row['pair']}: {sent.get('overall', 'N/A').upper()} "
                          f"({sent.get('adjustment', 0):+d} points, "
                          f"{sent.get('news_count', 0)} articles)")
            print("="*70 + "\n")
    else:
        log.info("✅ No strong signals this cycle")
    
    # Mark run as successful
    mark_success()
    log.info("✅ Run completed successfully")


if __name__ == "__main__":
    main()
