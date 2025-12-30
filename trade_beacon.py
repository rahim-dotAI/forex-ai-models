#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI FOREX BRAIN - OPTIMIZED SCHEDULED SYSTEM (10-MINUTE INTERVALS)
=================================================================
✅ Runs every 10 minutes instead of continuously
✅ Week 1 improvements: Signal duplication fix, trailing stops, increased cache
✅ Week 2 improvements: Market session awareness, multi-timeframe confirmation
✅ Week 3 improvements: Portfolio risk manager, performance analytics, volatility sizing
✅ Efficient API usage with smart caching
"""

import os
import sys
from pathlib import Path
import json
import time
import requests
import yfinance as yf
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
import logging
import traceback

# ======================================================
# CONFIGURATION - OPTIMIZED FOR 10-MINUTE RUNS
# ======================================================
ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY', '1W58NPZXOG5SLHZ6')
BROWSERLESS_TOKEN = os.environ.get('BROWSERLESS_TOKEN', '2TMVUBAjFwrr7Tb283f0da6602a4cb698b81778bda61967f7')
MARKETAUX_API_KEY = os.environ.get('MARKETAUX_API_KEY', '')

os.environ['ALPHA_VANTAGE_KEY'] = ALPHA_VANTAGE_KEY
os.environ['BROWSERLESS_TOKEN'] = BROWSERLESS_TOKEN

# ======================================================
# Environment Detection & Setup
# ======================================================
print("=" * 70)
print("🚀 AI FOREX BRAIN - OPTIMIZED SCHEDULED MODE")
print("=" * 70)

try:
    import google.colab
    IN_COLAB = True
    ENV_NAME = "Google Colab"
except ImportError:
    IN_COLAB = False
    ENV_NAME = "Local/GitHub Actions"

IN_GHA = "GITHUB_ACTIONS" in os.environ
if IN_GHA:
    ENV_NAME = "GitHub Actions"

if IN_COLAB:
    BASE_FOLDER = Path("/content")
    SAVE_FOLDER = BASE_FOLDER / "forex-ai-models"
elif IN_GHA:
    BASE_FOLDER = Path.cwd()
    SAVE_FOLDER = BASE_FOLDER
else:
    BASE_FOLDER = Path.cwd()
    SAVE_FOLDER = BASE_FOLDER

DIRECTORIES = {
    "data_raw": SAVE_FOLDER / "data" / "raw" / "yfinance",
    "data_processed": SAVE_FOLDER / "data" / "processed",
    "database": SAVE_FOLDER / "database",
    "logs": SAVE_FOLDER / "logs",
    "outputs": SAVE_FOLDER / "outputs",
    "memory": SAVE_FOLDER / "memory",
    "signal_state": SAVE_FOLDER / "signal_state",
}

for dir_name, dir_path in DIRECTORIES.items():
    dir_path.mkdir(parents=True, exist_ok=True)

print(f"🌍 Environment: {ENV_NAME}")
print(f"📂 Base Folder: {BASE_FOLDER}")
print(f"💾 Save Folder: {SAVE_FOLDER}")
print("=" * 70)

CSV_FOLDER = DIRECTORIES["data_raw"]
PICKLE_FOLDER = DIRECTORIES["data_processed"]
DB_PATH = DIRECTORIES["database"] / "memory_v90.db"
LOG_PATH = DIRECTORIES["logs"] / "pipeline.log"
OUTPUT_PATH = DIRECTORIES["outputs"] / "signals.json"
MEMORY_DIR = DIRECTORIES["memory"]
STATE_DIR = DIRECTORIES["signal_state"]

SYSTEM_STATE_FILE = STATE_DIR / "system_state.json"
ACTIVE_SIGNALS_FILE = STATE_DIR / "active_signals.json"
DASHBOARD_STATE_FILE = STATE_DIR / "dashboard_state.json"
LEARNING_MEMORY_FILE = MEMORY_DIR / "learning_memory.json"
TRADE_HISTORY_FILE = MEMORY_DIR / "trade_history.json"
PRICE_SOURCE_STATE_FILE = STATE_DIR / "price_source_rotation.json"
API_RATE_LIMIT_FILE = STATE_DIR / "api_rate_limits.json"
PERFORMANCE_ANALYTICS_FILE = MEMORY_DIR / "performance_analytics.json"

# Trading pairs by session
ALL_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "NZD/USD"]

# ✅ OPTIMIZED SETTINGS FOR 10-MINUTE RUNS
MAX_ACTIVE_SIGNALS = 5
ATR_SL_MULT = 2.0
ATR_TP_MULT = 3.0
MIN_CONFIDENCE = 0.72
RUN_INTERVAL_MINUTES = 10  # Run every 10 minutes
PRICE_CACHE_DURATION = 600  # 10 minutes (increased from 60s)
NEWS_CACHE_DURATION = 1800  # 30 minutes (increased from 15min)
HISTORICAL_DATA_CACHE = 3600  # 1 hour for historical data

# API Limits - optimized for scheduled runs
API_LIMITS = {
    'yfinance': {'daily_limit': 100, 'description': 'YFinance', 'enabled': True},
    'alpha_vantage': {'daily_limit': 25, 'description': 'Alpha Vantage', 'enabled': True},
    'browserless': {'daily_limit': 5, 'description': 'Browserless', 'enabled': False, 'enable_date': '2025-01-19T00:00:00+00:00'},
    'marketaux': {'daily_limit': 90, 'description': 'Marketaux', 'enabled': True}
}

# Correlation groups for portfolio risk management
CORRELATION_GROUPS = {
    'EUR': ['EUR/USD', 'EUR/GBP', 'EUR/JPY'],
    'USD_MAJORS': ['EUR/USD', 'GBP/USD', 'AUD/USD', 'NZD/USD'],
    'JPY_CROSSES': ['USD/JPY', 'EUR/JPY', 'GBP/JPY'],
    'COMMODITY': ['AUD/USD', 'NZD/USD', 'USD/CAD']
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

FAILED_FALLBACK_SYMBOLS: Set[str] = set()

# ============================================================================
# WEEK 2: MARKET SESSION AWARENESS
# ============================================================================
def get_market_session() -> Tuple[str, List[str]]:
    """Identify current trading session and return active pairs"""
    utc_hour = datetime.now(timezone.utc).hour
    
    if 0 <= utc_hour < 8:
        return "ASIAN", ["USD/JPY", "AUD/USD", "NZD/USD"]
    elif 8 <= utc_hour < 16:
        return "EUROPEAN", ["EUR/USD", "GBP/USD", "EUR/GBP"]
    elif 16 <= utc_hour < 24:
        return "US", ["EUR/USD", "GBP/USD", "USD/CAD"]
    
    return "OVERLAP", ALL_PAIRS

# ============================================================================
# API RATE LIMITER
# ============================================================================
class APIRateLimiter:
    def __init__(self):
        self.limits = API_LIMITS.copy()
        self.calls = {}
        self.last_reset_date = None
        self.load_state()
        self.check_browserless_enable()
        
    def load_state(self):
        if API_RATE_LIMIT_FILE.exists():
            try:
                with open(API_RATE_LIMIT_FILE, 'r') as f:
                    data = json.load(f)
                    last_date = data.get('date')
                    today = datetime.now(timezone.utc).date().isoformat()
                    if last_date == today:
                        self.calls = data.get('calls', {})
                        self.last_reset_date = last_date
                    else:
                        self.reset_counters()
            except Exception as e:
                logger.debug(f"Failed to load API limiter state: {e}")
                self.reset_counters()
        else:
            self.reset_counters()
    
    def save_state(self):
        try:
            with open(API_RATE_LIMIT_FILE, 'w') as f:
                json.dump({
                    'date': datetime.now(timezone.utc).date().isoformat(),
                    'calls': self.calls,
                    'last_updated': datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save API limiter state: {e}")
    
    def reset_counters(self):
        self.calls = {api: 0 for api in self.limits.keys()}
        self.last_reset_date = datetime.now(timezone.utc).date().isoformat()
        global FAILED_FALLBACK_SYMBOLS
        FAILED_FALLBACK_SYMBOLS.clear()
        self.save_state()
        logger.info("🔄 API rate limit counters reset for new day")
    
    def check_browserless_enable(self):
        now = datetime.now(timezone.utc)
        enable_date_str = self.limits['browserless']['enable_date']
        enable_date = datetime.fromisoformat(enable_date_str.replace('Z', '+00:00'))
        if now >= enable_date and not self.limits['browserless']['enabled']:
            self.limits['browserless']['enabled'] = True
            logger.info("✅ BROWSERLESS API ENABLED")
            self.save_state()
    
    def can_make_call(self, api_name: str) -> Tuple[bool, str]:
        today = datetime.now(timezone.utc).date().isoformat()
        if self.last_reset_date != today:
            self.reset_counters()
        
        if api_name not in self.limits:
            return False, f"Unknown API: {api_name}"
        
        if not self.limits[api_name]['enabled']:
            return False, f"{api_name} is disabled"
        
        current_calls = self.calls.get(api_name, 0)
        limit = self.limits[api_name]['daily_limit']
        
        if current_calls >= limit:
            return False, f"{api_name} daily limit reached ({current_calls}/{limit})"
        
        return True, f"OK ({current_calls}/{limit})"
    
    def record_call(self, api_name: str, success: bool = True):
        if api_name not in self.calls:
            self.calls[api_name] = 0
        self.calls[api_name] += 1
        self.save_state()
    
    def get_stats(self) -> Dict:
        stats = {}
        for api_name, config in self.limits.items():
            current = self.calls.get(api_name, 0)
            limit = config['daily_limit']
            stats[api_name] = {
                'enabled': config['enabled'],
                'calls': current,
                'limit': limit,
                'remaining': limit - current,
                'percentage': (current / limit * 100) if limit > 0 else 0
            }
        return stats
    
    def get_summary(self) -> str:
        """Generate API usage summary string"""
        lines = ["📊 API Usage Summary:"]
        for api_name, stats in self.get_stats().items():
            status = "✅" if stats['enabled'] else "❌"
            lines.append(
                f"{status} {api_name}: {stats['calls']}/{stats['limit']} "
                f"({stats['percentage']:.0f}%) - {self.limits[api_name]['description']}"
            )
        return "\n".join(lines)

api_limiter = APIRateLimiter()

# ============================================================================
# PRICE SOURCE ROTATION
# ============================================================================
class PriceSourceRotation:
    def __init__(self):
        self.sources = {
            'yfinance': {'weight': 10, 'calls': 0, 'failures': 0},
            'alpha_vantage': {'weight': 20, 'calls': 0, 'failures': 0},
            'browserless': {'weight': 5, 'calls': 0, 'failures': 0}
        }
        self.cache = {}
        self.load_state()

    def load_state(self):
        if PRICE_SOURCE_STATE_FILE.exists():
            try:
                with open(PRICE_SOURCE_STATE_FILE, 'r') as f:
                    data = json.load(f)
                    for source, stats in data.get('sources', {}).items():
                        if source in self.sources:
                            self.sources[source].update(stats)
            except:
                pass

    def save_state(self):
        try:
            with open(PRICE_SOURCE_STATE_FILE, 'w') as f:
                json.dump({'sources': self.sources}, f, indent=2)
        except:
            pass

    def get_cached_price(self, pair: str) -> Optional[float]:
        if pair in self.cache:
            age = time.time() - self.cache[pair]['timestamp']
            if age < PRICE_CACHE_DURATION:
                return self.cache[pair]['price']
        return None

    def cache_price(self, pair: str, price: float, source: str):
        self.cache[pair] = {'price': price, 'timestamp': time.time(), 'source': source}

price_rotation = PriceSourceRotation()

# ============================================================================
# PRICE FETCHERS
# ============================================================================
def fetch_live_price(pair: str) -> Optional[float]:
    """Fetch live price with caching"""
    cached_price = price_rotation.get_cached_price(pair)
    if cached_price:
        return cached_price
    
    try:
        symbol = pair.replace('/', '') + '=X'
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d', interval='1m')
        if not data.empty:
            price = float(data['Close'].iloc[-1])
            price_rotation.cache_price(pair, price, 'yfinance')
            api_limiter.record_call('yfinance', True)
            return price
    except Exception as e:
        logger.debug(f"Price fetch error for {pair}: {e}")
    
    return None

def fetch_historical_data(pair: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch or load historical forex data with caching"""
    pair_name = pair.replace('/', '_')
    cache_file = PICKLE_FOLDER / f"{pair_name}_{interval}_{period}.pkl"
    
    if cache_file.exists():
        try:
            df = pd.read_pickle(cache_file)
            file_age = time.time() - cache_file.stat().st_mtime
            if len(df) > 0 and file_age < HISTORICAL_DATA_CACHE:
                logger.debug(f"💾 Using cached data for {pair}")
                return df
        except:
            pass
    
    try:
        symbol = pair.replace('/', '') + '=X'
        logger.info(f"📥 Fetching data for {pair}...")
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.columns = [col.lower() for col in df.columns]
        df.to_pickle(cache_file)
        logger.info(f"✅ Cached {len(df)} candles for {pair}")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to fetch {pair}: {e}")
        return pd.DataFrame()

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================
class TechnicalIndicators:
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def macd(data: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14):
        high = df['high']
        low = df['low']
        close = df['close']
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        return adx, plus_di, minus_di

    @staticmethod
    def bollinger_bands(data: pd.Series, period=20, std_dev=2):
        middle = data.rolling(period).mean()
        std = data.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

# ============================================================================
# WEEK 3: PORTFOLIO RISK MANAGER
# ============================================================================
class PortfolioRiskManager:
    def __init__(self):
        self.max_correlated_pairs = 2
        self.max_daily_loss_pips = 500
        self.daily_pips = 0
        self.load_state()
    
    def load_state(self):
        state_file = STATE_DIR / "portfolio_risk.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    today = datetime.now(timezone.utc).date().isoformat()
                    if data.get('date') == today:
                        self.daily_pips = data.get('daily_pips', 0)
                    else:
                        self.daily_pips = 0
            except:
                pass
    
    def save_state(self):
        state_file = STATE_DIR / "portfolio_risk.json"
        try:
            with open(state_file, 'w') as f:
                json.dump({
                    'date': datetime.now(timezone.utc).date().isoformat(),
                    'daily_pips': self.daily_pips
                }, f, indent=2)
        except:
            pass
    
    def can_take_signal(self, signal: Dict, active_signals: List) -> Tuple[bool, str]:
        """Check if signal passes risk checks"""
        # Check daily loss limit
        if self.daily_pips < -self.max_daily_loss_pips:
            return False, f"Daily loss limit reached ({self.daily_pips:.1f} pips)"
        
        # Check correlation
        pair = signal['pair']
        correlated_count = self.count_correlated_pairs(pair, active_signals)
        if correlated_count >= self.max_correlated_pairs:
            return False, f"Too many correlated pairs active ({correlated_count})"
        
        return True, "OK"
    
    def count_correlated_pairs(self, pair: str, active_signals: List) -> int:
        """Count active correlated positions"""
        count = 0
        for group_name, pairs in CORRELATION_GROUPS.items():
            if pair in pairs:
                for signal in active_signals:
                    if signal.pair in pairs and signal.pair != pair:
                        count += 1
        return count
    
    def update_daily_pips(self, pips: float):
        """Update daily P&L"""
        self.daily_pips += pips
        self.save_state()

# ============================================================================
# WEEK 3: PERFORMANCE ANALYTICS
# ============================================================================
class PerformanceAnalytics:
    def __init__(self):
        self.metrics = self.load_metrics()
    
    def load_metrics(self) -> Dict:
        if PERFORMANCE_ANALYTICS_FILE.exists():
            try:
                with open(PERFORMANCE_ANALYTICS_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'daily_stats': {},
            'hourly_performance': {},
            'pair_performance': {},
            'best_hours': [],
            'worst_hours': [],
            'max_drawdown': 0,
            'peak_balance': 0
        }
    
    def save_metrics(self):
        try:
            with open(PERFORMANCE_ANALYTICS_FILE, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except:
            pass
    
    def update_performance(self, signal: Dict, outcome: str, pips: float):
        """Update performance metrics"""
        now = datetime.now(timezone.utc)
        hour_key = f"{now.hour:02d}:00"
        date_key = now.date().isoformat()
        
        # Hourly stats
        if hour_key not in self.metrics['hourly_performance']:
            self.metrics['hourly_performance'][hour_key] = {
                'trades': 0, 'wins': 0, 'total_pips': 0
            }
        
        self.metrics['hourly_performance'][hour_key]['trades'] += 1
        if outcome == 'TP_HIT':
            self.metrics['hourly_performance'][hour_key]['wins'] += 1
        self.metrics['hourly_performance'][hour_key]['total_pips'] += pips
        
        # Daily stats
        if date_key not in self.metrics['daily_stats']:
            self.metrics['daily_stats'][date_key] = {
                'trades': 0, 'wins': 0, 'total_pips': 0
            }
        
        self.metrics['daily_stats'][date_key]['trades'] += 1
        if outcome == 'TP_HIT':
            self.metrics['daily_stats'][date_key]['wins'] += 1
        self.metrics['daily_stats'][date_key]['total_pips'] += pips
        
        self.save_metrics()
    
    def get_best_trading_hours(self, top_n: int = 3) -> List[str]:
        """Find most profitable hours"""
        hourly = self.metrics['hourly_performance']
        sorted_hours = sorted(
            hourly.items(),
            key=lambda x: x[1]['total_pips'],
            reverse=True
        )
        return [hour for hour, _ in sorted_hours[:top_n]]

# ============================================================================
# DATACLASSES
# ============================================================================
@dataclass
class Signal:
    id: str
    pair: str
    direction: str
    entry_price: float
    sl: float
    tp: float
    created_at: str
    expires_at: str
    status: str
    confidence: float
    strategy: str
    indicators: Dict = field(default_factory=dict)
    patterns: List[str] = field(default_factory=list)
    confidence_factors: List[str] = field(default_factory=list)
    scores: Dict = field(default_factory=dict)
    outcome: Optional[str] = None
    outcome_time: Optional[str] = None
    outcome_pips: float = 0.0
    trailing_stop_activated: bool = False
    original_sl: float = 0.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

# ============================================================================
# NEWS ANALYZER
# ============================================================================
class NewsAnalyzer:
    def __init__(self):
        self.last_news_check = 0
        self.news_cache = []

    def fetch_news(self) -> List[Dict]:
        """Fetch news with caching"""
        if time.time() - self.last_news_check < NEWS_CACHE_DURATION:
            return self.news_cache
        
        if not MARKETAUX_API_KEY:
            return []
        
        can_call, _ = api_limiter.can_make_call('marketaux')
        if not can_call:
            return self.news_cache
        
        try:
            url = "https://api.marketaux.com/v1/news/all"
            params = {
                'api_token': MARKETAUX_API_KEY,
                'symbols': 'EUR,USD,GBP,JPY,AUD,CAD,NZD',
                'language': 'en',
                'limit': 3
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.news_cache = data.get('data', [])
                api_limiter.record_call('marketaux', True)
                self.last_news_check = time.time()
                logger.info(f"📰 Fetched {len(self.news_cache)} news articles")
        except:
            pass
        
        return self.news_cache

    def analyze_sentiment(self, pair: str) -> Dict:
        """Analyze sentiment for pair"""
        news = self.fetch_news()
        base_curr, quote_curr = pair.split('/')
        
        positive = 0
        negative = 0
        
        for article in news:
            content = f"{article.get('title', '')} {article.get('description', '')}".lower()
            if base_curr.lower() in content or quote_curr.lower() in content:
                sentiment = article.get('sentiment', 'neutral')
                if sentiment == 'positive':
                    positive += 1
                elif sentiment == 'negative':
                    negative += 1
        
        total = positive + negative
        score = (positive - negative) / total if total > 0 else 0.0
        
        return {
            'score': score,
            'positive_count': positive,
            'negative_count': negative
        }

# ============================================================================
# LEARNING MEMORY
# ============================================================================
class LearningMemory:
    def __init__(self):
        self.memory = self.load_memory()
        self.trade_history = self.load_trade_history()

    def load_memory(self) -> Dict:
        if LEARNING_MEMORY_FILE.exists():
            try:
                with open(LEARNING_MEMORY_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'pair_performance': {pair: {'wins': 0, 'losses': 0} for pair in ALL_PAIRS},
            'strategy_performance': {},
            'learned_adjustments': {
                'atr_sl_mult': ATR_SL_MULT,
                'atr_tp_mult': ATR_TP_MULT,
                'min_confidence': MIN_CONFIDENCE
            }
        }

    def load_trade_history(self) -> List[Dict]:
        if TRADE_HISTORY_FILE.exists():
            try:
                with open(TRADE_HISTORY_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return []

    def save_memory(self):
        try:
            with open(LEARNING_MEMORY_FILE, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except:
            pass

    def save_trade_history(self):
        try:
            with open(TRADE_HISTORY_FILE, 'w') as f:
                json.dump(self.trade_history[-1000:], f, indent=2)
        except:
            pass

    def record_trade(self, signal: Dict, outcome: str, pips: float):
        """Record trade and update learning"""
        trade = {
            'pair': signal['pair'],
            'direction': signal['direction'],
            'strategy': signal['strategy'],
            'confidence': signal['confidence'],
            'outcome': outcome,
            'pips': pips,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        self.trade_history.append(trade)
        
        pair = signal['pair']
        if outcome == 'TP_HIT':
            self.memory['pair_performance'][pair]['wins'] += 1
        else:
            self.memory['pair_performance'][pair]['losses'] += 1
        
        strategy = signal['strategy']
        if strategy not in self.memory['strategy_performance']:
            self.memory['strategy_performance'][strategy] = {'wins': 0, 'losses': 0}
        
        if outcome == 'TP_HIT':
            self.memory['strategy_performance'][strategy]['wins'] += 1
        else:
            self.memory['strategy_performance'][strategy]['losses'] += 1
        
        self.adjust_parameters()
        self.save_memory()
        self.save_trade_history()

    def adjust_parameters(self):
        """Adjust trading parameters based on performance"""
        if len(self.trade_history) < 20:
            return
        
        recent = self.trade_history[-50:]
        wins = sum(1 for t in recent if t['outcome'] == 'TP_HIT')
        win_rate = wins / len(recent)
        
        if win_rate < 0.60:
            self.memory['learned_adjustments']['min_confidence'] = min(0.80, MIN_CONFIDENCE + 0.02)
        elif win_rate > 0.75:
            self.memory['learned_adjustments']['min_confidence'] = max(0.65, MIN_CONFIDENCE - 0.01)

    def get_pair_confidence_modifier(self, pair: str) -> float:
        """Get confidence modifier based on pair performance"""
        perf = self.memory['pair_performance'].get(pair, {'wins': 0, 'losses': 0})
        total = perf['wins'] + perf['losses']
        
        if total < 5:
            return 1.0
        
        win_rate = perf['wins'] / total
        if win_rate > 0.75:
            return 1.05
        elif win_rate < 0.50:
            return 0.95
        return 1.0

# ============================================================================
# WEEK 2: MULTI-TIMEFRAME ANALYSIS
# ============================================================================
def check_higher_timeframe_trend(pair: str) -> Tuple[str, float]:
    """Check daily trend for multi-timeframe confirmation"""
    try:
        df_daily = fetch_historical_data(pair, period="1y", interval="1d")
        if len(df_daily) < 50:
            return "NEUTRAL", 0.5
        
        close = df_daily['close']
        ema50 = TechnicalIndicators.ema(close, 50).iloc[-1]
        current = close.iloc[-1]
        
        if current > ema50 * 1.005:  # 0.5% above
            return "BULLISH", 0.7
        elif current < ema50 * 0.995:  # 0.5% below
            return "BEARISH", 0.7
        else:
            return "NEUTRAL", 0.5
    except:
        return "NEUTRAL", 0.5

# ============================================================================
# WEEK 3: VOLATILITY-BASED SIZING
# ============================================================================
def calculate_volatility_adjusted_multipliers(df: pd.DataFrame) -> Tuple[float, float]:
    """Adjust SL/TP based on volatility"""
    try:
        recent_volatility = df['close'].pct_change().tail(20).std() * 100
        
        if recent_volatility > 1.5:  # High volatility
            return 2.5, 4.0
        elif recent_volatility < 0.5:  # Low volatility
            return 1.5, 2.5
        else:
            return ATR_SL_MULT, ATR_TP_MULT
    except:
        return ATR_SL_MULT, ATR_TP_MULT

# ============================================================================
# SIGNAL GENERATION WITH ALL IMPROVEMENTS
# ============================================================================
def generate_signal(pair: str, news_analyzer: NewsAnalyzer, memory: LearningMemory, 
                   active_signals: List[Signal], risk_manager: PortfolioRiskManager) -> Optional[Dict]:
    """Generate signal with all Week 1-3 improvements"""
    
    # ✅ WEEK 1: Check for duplicate signals
    if any(s.pair == pair for s in active_signals):
        logger.debug(f"Skipping {pair} - already has active signal")
        return None
    
    # Get sentiment
    sentiment = news_analyzer.analyze_sentiment(pair)
    
    # Fetch data
    df = fetch_historical_data(pair, period="5y", interval="1d")
    if len(df) < 200:
        logger.warning(f"Insufficient data for {pair}")
        return None
    
    try:
        # Calculate indicators
        close = df['close']
        ema12 = TechnicalIndicators.ema(close, 12).iloc[-1]
        ema26 = TechnicalIndicators.ema(close, 26).iloc[-1]
        ema50 = TechnicalIndicators.ema(close, 50).iloc[-1]
        ema200 = TechnicalIndicators.ema(close, 200).iloc[-1]
        
        rsi = TechnicalIndicators.rsi(close).iloc[-1]
        adx_value, plus_di, minus_di = TechnicalIndicators.adx(df)
        macd_line, signal_line, histogram = TechnicalIndicators.macd(close)
        
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close)
        current_price_position = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        
        # Score the setup
        bullish_score = 0
        bearish_score = 0
        factors = []
        
        # Trend analysis
        if ema12 > ema26 > ema50:
            bullish_score += 25
            factors.append("Strong uptrend (EMA alignment)")
        elif ema12 < ema26 < ema50:
            bearish_score += 25
            factors.append("Strong downtrend (EMA alignment)")
        
        if close.iloc[-1] > ema200:
            bullish_score += 15
            factors.append("Above 200 EMA")
        else:
            bearish_score += 15
            factors.append("Below 200 EMA")
        
        # RSI
        if 30 < rsi < 50:
            bullish_score += 15
            factors.append(f"RSI bullish ({rsi:.1f})")
        elif 50 < rsi < 70:
            bearish_score += 15
            factors.append(f"RSI bearish ({rsi:.1f})")
        
        # Bollinger bands
        if current_price_position < 0.3:
            bullish_score += 20
            factors.append("Near lower Bollinger")
        elif current_price_position > 0.7:
            bearish_score += 20
            factors.append("Near upper Bollinger")
        
        # ADX trend strength
        if adx_value.iloc[-1] > 25:
            if plus_di.iloc[-1] > minus_di.iloc[-1]:
                bullish_score += 10
            else:
                bearish_score += 10
            factors.append(f"Strong trend (ADX {adx_value.iloc[-1]:.1f})")
        
        # ✅ WEEK 2: Multi-timeframe confirmation
        htf_trend, htf_confidence = check_higher_timeframe_trend(pair)
        if htf_trend == "BULLISH":
            bullish_score += 15
            factors.append("Daily uptrend confirmation")
        elif htf_trend == "BEARISH":
            bearish_score += 15
            factors.append("Daily downtrend confirmation")
        
        # Determine direction
        if bullish_score > bearish_score + 20:
            direction = 'BUY'
            base_confidence = MIN_CONFIDENCE + (bullish_score - bearish_score) / 150
        elif bearish_score > bullish_score + 20:
            direction = 'SELL'
            base_confidence = MIN_CONFIDENCE + (bearish_score - bullish_score) / 150
        else:
            return None
        
        # ✅ WEEK 2: Multi-timeframe filter
        if direction == 'BUY' and htf_trend == "BEARISH":
            logger.info(f"❌ Rejecting BUY {pair} - against daily trend")
            return None
        elif direction == 'SELL' and htf_trend == "BULLISH":
            logger.info(f"❌ Rejecting SELL {pair} - against daily trend")
            return None
        
        # Sentiment filter
        if direction == 'BUY' and sentiment['score'] < -0.5:
            logger.info(f"❌ Rejecting BUY {pair} - negative sentiment")
            return None
        elif direction == 'SELL' and sentiment['score'] > 0.5:
            logger.info(f"❌ Rejecting SELL {pair} - positive sentiment")
            return None
        
        # Apply learning modifiers
        pair_modifier = memory.get_pair_confidence_modifier(pair)
        
        # Determine strategy
        if current_price_position < 0.3 or current_price_position > 0.7:
            strategy = 'MEAN_REVERSION'
        elif adx_value.iloc[-1] > 25:
            strategy = 'TREND_FOLLOWING'
        else:
            strategy = 'BREAKOUT'
        
        adjusted_confidence = base_confidence * pair_modifier
        learned_min_conf = memory.memory['learned_adjustments']['min_confidence']
        
        if adjusted_confidence < learned_min_conf:
            return None
        
        # Get current price
        current_price = fetch_live_price(pair)
        if current_price is None:
            logger.warning(f"No price data for {pair}")
            return None
        
        # ✅ WEEK 3: Volatility-based position sizing
        atr = TechnicalIndicators.atr(df).iloc[-1]
        sl_mult, tp_mult = calculate_volatility_adjusted_multipliers(df)
        
        # Calculate SL/TP
        if direction == 'BUY':
            sl = current_price - (atr * sl_mult)
            tp = current_price + (atr * tp_mult)
        else:
            sl = current_price + (atr * sl_mult)
            tp = current_price - (atr * tp_mult)
        
        signal_data = {
            'pair': pair,
            'direction': direction,
            'entry_price': current_price,
            'sl': sl,
            'tp': tp,
            'confidence': min(0.95, adjusted_confidence),
            'strategy': strategy,
            'indicators': {
                'rsi': float(rsi),
                'adx': float(adx_value.iloc[-1]),
                'ema12': float(ema12),
                'ema200': float(ema200)
            },
            'patterns': [],
            'confidence_factors': factors,
            'scores': {
                'bullish': bullish_score,
                'bearish': bearish_score
            },
            'volatility_adjusted': True,
            'sl_multiplier': sl_mult,
            'tp_multiplier': tp_mult
        }
        
        # ✅ WEEK 3: Portfolio risk check
        can_take, reason = risk_manager.can_take_signal(signal_data, active_signals)
        if not can_take:
            logger.info(f"❌ Risk check failed for {pair}: {reason}")
            return None
        
        return signal_data
        
    except Exception as e:
        logger.error(f"Error generating signal for {pair}: {e}")
        return None

# ============================================================================
# SIGNAL MANAGER WITH TRAILING STOPS
# ============================================================================
class SignalManager:
    def __init__(self):
        self.active_signals: List[Signal] = []
        self.archived_signals: List[Signal] = []
        self.load_state()

    def load_state(self):
        if ACTIVE_SIGNALS_FILE.exists():
            try:
                with open(ACTIVE_SIGNALS_FILE, 'r') as f:
                    data = json.load(f)
                    self.active_signals = [Signal.from_dict(s) for s in data]
            except:
                self.active_signals = []

    def save_state(self):
        try:
            with open(ACTIVE_SIGNALS_FILE, 'w') as f:
                json.dump([s.to_dict() for s in self.active_signals], f, indent=2)
        except:
            pass

    def broadcast_signal(self, signal_data: Dict) -> Signal:
        """Create and broadcast new signal"""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=4)
        
        signal = Signal(
            id=f"{signal_data['pair']}_{int(now.timestamp())}",
            pair=signal_data['pair'],
            direction=signal_data['direction'],
            entry_price=signal_data['entry_price'],
            sl=signal_data['sl'],
            tp=signal_data['tp'],
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
            status='ACTIVE',
            confidence=signal_data['confidence'],
            strategy=signal_data['strategy'],
            indicators=signal_data.get('indicators', {}),
            patterns=signal_data.get('patterns', []),
            confidence_factors=signal_data.get('confidence_factors', []),
            scores=signal_data.get('scores', {}),
            trailing_stop_activated=False,
            original_sl=signal_data['sl']
        )
        
        self.active_signals.append(signal)
        self.save_state()
        
        pip_multiplier = 100 if 'JPY' in signal.pair else 10000
        risk_pips = abs(signal.entry_price - signal.sl) * pip_multiplier
        reward_pips = abs(signal.tp - signal.entry_price) * pip_multiplier
        rr = reward_pips / risk_pips if risk_pips > 0 else 0
        
        logger.info(f"🎯 NEW SIGNAL: {signal.direction} {signal.pair} @ {signal.entry_price:.5f}")
        logger.info(f"   Confidence: {signal.confidence*100:.0f}% | R:R 1:{rr:.2f}")
        logger.info(f"   SL: {signal.sl:.5f} | TP: {signal.tp:.5f}")
        return signal

    # ✅ WEEK 1: Trailing Stop Implementation
    def update_trailing_stop(self, signal: Signal, current_price: float):
        """Move SL to breakeven/profit after certain pip movement"""
        pip_mult = 100 if 'JPY' in signal.pair else 10000
        
        if signal.direction == 'BUY':
            profit_pips = (current_price - signal.entry_price) * pip_mult
            if profit_pips >= 50 and not signal.trailing_stop_activated:
                new_sl = signal.entry_price + (30 / pip_mult)
                if new_sl > signal.sl:
                    signal.sl = new_sl
                    signal.trailing_stop_activated = True
                    logger.info(f"✅ Trailing stop activated: {signal.pair} SL moved to {new_sl:.5f}")
        else:  # SELL
            profit_pips = (signal.entry_price - current_price) * pip_mult
            if profit_pips >= 50 and not signal.trailing_stop_activated:
                new_sl = signal.entry_price - (30 / pip_mult)
                if new_sl < signal.sl:
                    signal.sl = new_sl
                    signal.trailing_stop_activated = True
                    logger.info(f"✅ Trailing stop activated: {signal.pair} SL moved to {new_sl:.5f}")

    def update_signal_outcomes(self, memory: LearningMemory, risk_manager: PortfolioRiskManager, 
                               analytics: PerformanceAnalytics):
        """Update signal outcomes with trailing stops"""
        now = datetime.now(timezone.utc)
        archived = []
        
        for signal in self.active_signals:
            # Check expiry
            expires_at = datetime.fromisoformat(signal.expires_at.replace('Z', '+00:00'))
            if now >= expires_at:
                signal.status = 'EXPIRED'
                signal.outcome = 'EXPIRED'
                signal.outcome_time = now.isoformat()
                archived.append(signal)
                continue
            
            # Get current price
            current_price = fetch_live_price(signal.pair)
            if current_price is None:
                continue
            
            # ✅ Update trailing stop
            self.update_trailing_stop(signal, current_price)
            
            try:
                pip_multiplier = 100 if 'JPY' in signal.pair else 10000
                
                if signal.direction == 'BUY':
                    if current_price >= signal.tp:
                        signal.status = 'TP_HIT'
                        signal.outcome = 'TP_HIT'
                        signal.outcome_pips = (current_price - signal.entry_price) * pip_multiplier
                        signal.outcome_time = now.isoformat()
                        archived.append(signal)
                        memory.record_trade(signal.to_dict(), 'TP_HIT', signal.outcome_pips)
                        risk_manager.update_daily_pips(signal.outcome_pips)
                        analytics.update_performance(signal.to_dict(), 'TP_HIT', signal.outcome_pips)
                        logger.info(f"✅ TP HIT: {signal.pair} (+{signal.outcome_pips:.1f} pips)")
                    elif current_price <= signal.sl:
                        signal.status = 'SL_HIT'
                        signal.outcome = 'SL_HIT'
                        signal.outcome_pips = (current_price - signal.entry_price) * pip_multiplier
                        signal.outcome_time = now.isoformat()
                        archived.append(signal)
                        memory.record_trade(signal.to_dict(), 'SL_HIT', signal.outcome_pips)
                        risk_manager.update_daily_pips(signal.outcome_pips)
                        analytics.update_performance(signal.to_dict(), 'SL_HIT', signal.outcome_pips)
                        logger.info(f"❌ SL HIT: {signal.pair} ({signal.outcome_pips:.1f} pips)")
                else:  # SELL
                    if current_price <= signal.tp:
                        signal.status = 'TP_HIT'
                        signal.outcome = 'TP_HIT'
                        signal.outcome_pips = (signal.entry_price - current_price) * pip_multiplier
                        signal.outcome_time = now.isoformat()
                        archived.append(signal)
                        memory.record_trade(signal.to_dict(), 'TP_HIT', signal.outcome_pips)
                        risk_manager.update_daily_pips(signal.outcome_pips)
                        analytics.update_performance(signal.to_dict(), 'TP_HIT', signal.outcome_pips)
                        logger.info(f"✅ TP HIT: {signal.pair} (+{signal.outcome_pips:.1f} pips)")
                    elif current_price >= signal.sl:
                        signal.status = 'SL_HIT'
                        signal.outcome = 'SL_HIT'
                        signal.outcome_pips = (signal.entry_price - current_price) * pip_multiplier
                        signal.outcome_time = now.isoformat()
                        archived.append(signal)
                        memory.record_trade(signal.to_dict(), 'SL_HIT', signal.outcome_pips)
                        risk_manager.update_daily_pips(signal.outcome_pips)
                        analytics.update_performance(signal.to_dict(), 'SL_HIT', signal.outcome_pips)
                        logger.info(f"❌ SL HIT: {signal.pair} ({signal.outcome_pips:.1f} pips)")
            except Exception as e:
                logger.debug(f"Error checking signal outcome: {e}")
        
        # Archive completed signals
        for signal in archived:
            self.active_signals.remove(signal)
            self.archived_signals.append(signal)
        
        if archived:
            self.save_state()

    def can_broadcast_new_signal(self) -> bool:
        """Check if we can add more signals"""
        return len(self.active_signals) < MAX_ACTIVE_SIGNALS

    def get_stats(self) -> Dict:
        """Get trading statistics"""
        completed = [s for s in self.archived_signals if s.outcome in ['TP_HIT', 'SL_HIT']]
        total = len(completed)
        wins = sum(1 for s in completed if s.outcome == 'TP_HIT')
        total_pips = sum(s.outcome_pips for s in completed)
        
        return {
            'total_signals': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': (wins / total * 100) if total > 0 else 0,
            'total_pips': total_pips,
            'active_signals': len(self.active_signals)
        }

# ============================================================================
# DASHBOARD STATE
# ============================================================================
def save_dashboard_state(signal_manager: SignalManager, news_analyzer: NewsAnalyzer, 
                        memory: LearningMemory, risk_manager: PortfolioRiskManager,
                        analytics: PerformanceAnalytics, session: str):
    """Save complete system state"""
    state = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'next_run': (datetime.now(timezone.utc) + timedelta(minutes=RUN_INTERVAL_MINUTES)).isoformat(),
        'run_interval_minutes': RUN_INTERVAL_MINUTES,
        'current_session': session,
        'active_signals': len(signal_manager.active_signals),
        'signals': [s.to_dict() for s in signal_manager.active_signals],
        'stats': signal_manager.get_stats(),
        'api_usage': api_limiter.get_stats(),
        'learning_stats': {
            'total_trades': len(memory.trade_history),
            'pair_performance': memory.memory['pair_performance'],
            'learned_params': memory.memory['learned_adjustments']
        },
        'risk_management': {
            'daily_pips': risk_manager.daily_pips,
            'max_daily_loss': risk_manager.max_daily_loss_pips,
            'max_correlated_pairs': risk_manager.max_correlated_pairs
        },
        'best_trading_hours': analytics.get_best_trading_hours(),
        'recent_news': news_analyzer.news_cache[:5]
    }
    
    try:
        with open(DASHBOARD_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save dashboard state: {e}")

# ============================================================================
# MAIN FUNCTION - SCHEDULED 10-MINUTE RUN
# ============================================================================
def main():
    """
    Main function - runs once every 10 minutes
    Optimized for scheduled execution (cron, GitHub Actions, etc.)
    """
    logger.info("=" * 80)
    logger.info("🚀 AI FOREX BRAIN - 10-MINUTE CYCLE START")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Check if markets are open
    now = datetime.now(timezone.utc)
    if now.weekday() in [5, 6]:
        logger.info("🏖️ WEEKEND - Markets closed")
        return
    
    # Get current market session
    session, active_pairs = get_market_session()
    logger.info(f"📊 Session: {session}")
    logger.info(f"🎯 Active Pairs: {', '.join(active_pairs)}")
    
    # Initialize components
    news_analyzer = NewsAnalyzer()
    memory = LearningMemory()
    signal_manager = SignalManager()
    risk_manager = PortfolioRiskManager()
    analytics = PerformanceAnalytics()
    
    # Update existing signals
    logger.info("🔄 Checking active signals...")
    signal_manager.update_signal_outcomes(memory, risk_manager, analytics)
    
    # Generate new signals if space available
    if signal_manager.can_broadcast_new_signal():
        logger.info(f"🔍 Scanning for new signals...")
        
        # Prioritize pairs based on current session
        for pair in active_pairs:
            if not signal_manager.can_broadcast_new_signal():
                break
            
            logger.info(f"   Analyzing {pair}...")
            signal_data = generate_signal(
                pair, 
                news_analyzer, 
                memory, 
                signal_manager.active_signals,
                risk_manager
            )
            
            if signal_data:
                signal_manager.broadcast_signal(signal_data)
            else:
                logger.debug(f"   No signal for {pair}")
    else:
        logger.info(f"⏸️ Max signals reached ({MAX_ACTIVE_SIGNALS})")
    
    # Save dashboard state
    save_dashboard_state(signal_manager, news_analyzer, memory, risk_manager, analytics, session)
    
    # Display summary
    stats = signal_manager.get_stats()
    elapsed = time.time() - start_time
    
    logger.info("=" * 80)
    logger.info("📊 CYCLE SUMMARY")
    logger.info(f"⏱️  Execution time: {elapsed:.2f}s")
    logger.info(f"🎯 Active signals: {len(signal_manager.active_signals)}/{MAX_ACTIVE_SIGNALS}")
    logger.info(f"📈 Win rate: {stats['win_rate']:.1f}%")
    logger.info(f"💰 Total pips: {stats['total_pips']:.1f}")
    logger.info(f"🏆 Wins: {stats['wins']} | Losses: {stats['losses']}")
    logger.info(f"📱 Daily P&L: {risk_manager.daily_pips:.1f} pips")
    logger.info("=" * 80)
    logger.info(f"⏰ Next run: {(datetime.now(timezone.utc) + timedelta(minutes=RUN_INTERVAL_MINUTES)).strftime('%H:%M UTC')}")
    logger.info("=" * 80)

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("✅ AI FOREX BRAIN - OPTIMIZED SCHEDULED MODE")
    print("=" * 70)
    print(f"⏰ Run Interval: Every {RUN_INTERVAL_MINUTES} minutes")
    print(f"📊 Monitoring {len(ALL_PAIRS)} currency pairs")
    print(f"🎯 Maximum {MAX_ACTIVE_SIGNALS} concurrent signals")
    print("=" * 70)
    print("🔧 IMPROVEMENTS IMPLEMENTED:")
    print("   ✅ Week 1: Signal duplication fix, trailing stops, increased cache")
    print("   ✅ Week 2: Market session awareness, multi-timeframe confirmation")
    print("   ✅ Week 3: Portfolio risk manager, performance analytics, volatility sizing")
    print("=" * 70)
    print(api_limiter.get_summary())
    print("=" * 70)
    
    main()
